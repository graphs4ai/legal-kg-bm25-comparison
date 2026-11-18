"""OAB evaluator: prompts a QA model per question and records results and metrics.

Configured and launched from ``run_eval.py``.
"""

import asyncio
import inspect
import json
import random
import re
import time
from datetime import date, datetime
from pathlib import Path

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from src.utils.currency import PROVIDER_CURRENCY
from src.utils.llm_client import call_deepinfra, call_maritaca, call_ollama
from src.utils.logging_utils import log_experiment_config, log_summary
from src.data.oab_splits import cutoff_for


def question_cutoff(metadata: dict) -> date | None:
    """Legal cutoff of a question (its edition's notice date), or None if unknown."""

    try:
        return cutoff_for(metadata)
    except (KeyError, ValueError, TypeError):
        return None


# Outcome of each question; no_answer is a response with no parseable A-D choice.
STATUS_ANSWERED = "answered"
STATUS_NO_ANSWER = "no_answer"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

# Words that introduce a choice in the answer tag, e.g. "Letra B", "Alternativa: C".
_CHOICE_KEYWORD = r"(?:letra|alternativa|op[cç][aã]o|item|resposta)"


def extract_answer(text: str | None) -> str | None:
    """The A-D choice inside ``<answer>…</answer>``, or None when there is no single choice.

    Patterns are anchored, so refusals ("Nenhuma das alternativas…") and
    enumerations ("as opções A, B, C ou D") never match.
    """

    if not text:
        return None

    tag = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if not tag:
        return None

    content = tag.group(1).strip()

    # Bare letter: "B", "(C)", "D.", "A)".
    clean = re.fullmatch(r"\(?\s*([A-Da-d])\s*[\)\.\:]?", content)
    if clean:
        return clean.group(1).upper()

    # Echoed option: "B: Conselho Federal da OAB."; uppercase only, so "a: ..." cannot match.
    echoed = re.match(r"\(?([A-D])\s*[\)\.\:\-–]\s+\S", content)
    if echoed:
        return echoed.group(1)

    # Keyword then letter: "Letra B", "Alternativa: C", "Resposta D".
    prefixed = re.match(
        rf"{_CHOICE_KEYWORD}\b\W*([A-Da-d])(?![A-Za-zÀ-ÿ])",
        content,
        re.IGNORECASE,
    )
    if prefixed:
        return prefixed.group(1).upper()

    # One word then a lone letter, for keyword typos like "Letá C".
    two_token = re.fullmatch(
        r"[A-Za-zÀ-ÿ]+\W+([A-Da-d])(?![A-Za-zÀ-ÿ])\s*",
        content,
        re.IGNORECASE,
    )
    if two_token:
        return two_token.group(1).upper()

    return None


# Config fields copied into each metrics entry.
_METRICS_CONFIG_FIELDS = [
    "provider",
    "cost_currency",
    "run_id",
    "model",
    "temperature",
    "think",
    "chat_template_kwargs",
    "num_predict",
    "concurrency",
    "sample_size",
    "sample_random",
    "sample_seed",
    "sample_per_group",
    "sample_group_key",
    "condition",
    "retrieval_method",
    "retrieval_top_k",
    "retrieval_query_mode",
    "retrieval_max_context_tokens",
    "laws_version",
    "retrieval_unit",
    "retrieval_scope",
    "graph_version",
]


def classify_status(result: dict) -> str:
    """Outcome status of a result line, inferred from ``predicted_answer`` when absent."""

    status = result.get("status")
    if status:
        return status
    return STATUS_ANSWERED if result.get("predicted_answer") else STATUS_NO_ANSWER


def aggregate_result_metrics(results: list[dict]) -> dict:
    """Run-level counts, accuracy, tokens, timing and cost from per-question results.

    Accuracy is over every attempted question, so timeouts, errors and
    unparseable answers count as wrong. Usage averages are over responded questions.
    """

    total = len(results)
    correct = answered = no_answer = timeout = errors = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_request_seconds = 0.0
    estimated_cost_sum = 0.0
    responded = 0

    for result in results:
        status = classify_status(result)

        if status == STATUS_ANSWERED:
            answered += 1
            if result.get("correct"):
                correct += 1
        elif status == STATUS_TIMEOUT:
            timeout += 1
        elif status == STATUS_ERROR:
            errors += 1
        else:
            no_answer += 1

        # Timeouts and errors carry no usage data.
        if result.get("input_tokens") is not None:
            responded += 1
            total_input_tokens += result.get("input_tokens") or 0
            total_output_tokens += result.get("output_tokens") or 0
            total_request_seconds += result.get("elapsed_seconds") or 0
            estimated_cost_sum += result.get("estimated_cost") or 0

    # Every non-answered outcome.
    skipped = no_answer + timeout + errors

    return {
        "total": total,
        "correct": correct,
        "answered": answered,
        "no_answer": no_answer,
        "timeout": timeout,
        "errors": errors,
        "skipped": skipped,
        "accuracy": correct / total if total else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_request_seconds": total_request_seconds,
        "estimated_cost": estimated_cost_sum,
        "avg_input_tokens": total_input_tokens / responded if responded else 0,
        "avg_output_tokens": total_output_tokens / responded if responded else 0,
        "avg_request_seconds": total_request_seconds / responded if responded else 0,
    }


def assemble_metrics(
    config: dict,
    aggregates: dict,
    eval_total_seconds: float,
) -> dict:
    """Merge run config and aggregated metrics into one ordered metrics entry."""

    metrics = {field: config.get(field) for field in _METRICS_CONFIG_FIELDS}
    metrics.update(
        {
            "total": aggregates["total"],
            "correct": aggregates["correct"],
            "answered": aggregates["answered"],
            "no_answer": aggregates["no_answer"],
            "timeout": aggregates["timeout"],
            "errors": aggregates["errors"],
            "skipped": aggregates["skipped"],
            "accuracy": aggregates["accuracy"],
            "total_input_tokens": aggregates["total_input_tokens"],
            "total_output_tokens": aggregates["total_output_tokens"],
            "total_tokens": aggregates["total_tokens"],
            "total_request_seconds": aggregates["total_request_seconds"],
            "eval_total_seconds": eval_total_seconds,
            "estimated_cost": aggregates["estimated_cost"],
            "avg_input_tokens": aggregates["avg_input_tokens"],
            "avg_output_tokens": aggregates["avg_output_tokens"],
            "avg_request_seconds": aggregates["avg_request_seconds"],
        }
    )
    return metrics


class OABEvaluator:
    """Evaluates one QA model on an OAB split and writes a new run directory.

    Args:
        think: Reasoning setting; a bool for Ollama, a reasoning_effort string for DeepInfra.
        chat_template_kwargs: Template variables for models whose reasoning switch
            lives in the chat template.
        sample_size / sample_random / sample_seed: First N or a seeded random N.
        sample_per_group: N questions per ``sample_group_key`` group; exclusive
            with sample_size.
        provider: "deepinfra", "ollama" or "maritaca"; None infers it from
            concurrency (1 means Ollama).
        retriever: Retrieval backend; None runs the no-retrieval condition.
        laws_version / retrieval_unit / retrieval_scope / graph_version:
            Recorded in the run metrics.
        condition: Output folder ``results/runs/<condition>/``; inferred as
            "rag" or "vanilla" from the retriever when None.
    """

    def __init__(
        self,
        dataset_path: Path,
        prompt_path: Path,
        model: str,
        temperature: float = 0.1,
        think: str | bool = False,
        chat_template_kwargs: dict | None = None,
        num_predict: int = None,
        sample_size: int | None = None,
        sample_random: bool = False,
        sample_seed: int | None = None,
        sample_per_group: int | None = None,
        sample_group_key: str = "exam_id",
        concurrency: int = 1,
        provider: str | None = None,
        request_timeout_seconds: int = 180,
        retriever=None,
        retrieval_top_k: int = 5,
        laws_version: str | None = None,
        retrieval_unit: str | None = None,
        retrieval_scope: str | None = None,
        graph_version: str | None = None,
        condition: str | None = None,
    ):
        self.dataset_path = dataset_path
        self.prompt_path = prompt_path
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.model = model
        self.temperature = temperature
        self.think = think
        self.chat_template_kwargs = chat_template_kwargs
        self.num_predict = num_predict
        if sample_size and sample_per_group:
            raise ValueError(
                "sample_size and sample_per_group are mutually exclusive")
        self.sample_size = sample_size
        self.sample_random = sample_random
        self.sample_seed = sample_seed
        self.sample_per_group = sample_per_group
        self.sample_group_key = sample_group_key
        self.concurrency = concurrency
        self.provider = provider or (
            "ollama" if concurrency == 1 else "deepinfra")
        if self.provider not in ("deepinfra", "ollama", "maritaca"):
            raise ValueError(f"Unknown provider: {self.provider!r}")
        self.request_timeout_seconds = request_timeout_seconds
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.laws_version = laws_version
        self.retrieval_unit = retrieval_unit
        self.retrieval_scope = retrieval_scope
        self.graph_version = graph_version

        self.condition = condition or (
            "rag" if retriever is not None else "vanilla")

        # results/runs/<condition>/, one metrics.json per condition.
        self.project_root = Path(__file__).resolve().parents[2]
        self.experiments_dir = self.project_root / "results" / "runs"
        self.outputs_dir = self.experiments_dir / self.condition
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = self._create_run_dir()
        self.results_path = self.run_dir / "results.jsonl"
        self.metrics_path = self.outputs_dir / "metrics.json"

        logger.remove()
        logger.add(
            self.run_dir / f"{self.run_dir.name}.log",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
        )
        logger.info("Initialized evaluator")
        log_experiment_config(
            run_dir=self.run_dir,
            model=self.model,
            dataset_path=self.dataset_path,
            prompt_path=self.prompt_path,
            sample_size=self.sample_size,
            sample_random=self.sample_random,
            sample_seed=self.sample_seed,
            temperature=self.temperature,
            think=self.think,
            chat_template_kwargs=self.chat_template_kwargs,
            num_predict=self.num_predict,
            concurrency=self.concurrency,
            condition=self.condition,
            retrieval_method=(
                None if self.retriever is None else self.retriever.method
            ),
            retrieval_top_k=(
                None if self.retriever is None else self.retrieval_top_k
            ),
            results_path=self.results_path,
            metrics_path=self.metrics_path,
        )

    def _create_run_dir(self) -> Path:
        """Create the next run directory; run ids are global across conditions."""

        numbers = []
        for run_path in self.experiments_dir.glob("*/run_*"):
            if not run_path.is_dir():
                continue
            try:
                numbers.append(int(run_path.name.split("_")[1]))
            except (IndexError, ValueError):
                continue

        next_id = max(numbers, default=0) + 1
        run_dir = self.outputs_dir / f"run_{next_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def load_dataset(self) -> list[dict]:
        """Load the split, sampled when sample_size or sample_per_group is set."""

        logger.info("Loading dataset: {}", self.dataset_path)

        with self.dataset_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        if self.sample_per_group:
            rng = random.Random(self.sample_seed)
            groups: dict = {}
            for row in rows:
                key = row.get("metadata", {}).get(self.sample_group_key)
                groups.setdefault(key, []).append(row)
            rows = [
                item
                for key in sorted(groups, key=str)
                for item in rng.sample(
                    groups[key],
                    min(self.sample_per_group, len(groups[key])))
            ]
            logger.info("Stratified sample: {} per {} over {} groups",
                        self.sample_per_group, self.sample_group_key,
                        len(groups))
        elif self.sample_size:
            if self.sample_random:
                rng = random.Random(self.sample_seed)
                rows = rng.sample(rows, min(self.sample_size, len(rows)))
            else:
                rows = rows[: self.sample_size]

        logger.info("Loaded {} questions", len(rows))

        return rows

    def build_prompt(self, item: dict) -> tuple[str, dict | None]:
        """Fill the prompt template for a question.

        With a retriever, also fills ``{context}`` with records in force at the
        question's cutoff and returns their provenance; otherwise returns None.
        """

        alternatives = "\n".join(
            f"{k}: {v}"
            for k, v in item["choices"].items()
        )

        # No retrieval: no context slot.
        if self.retriever is None:
            prompt = self.prompt_template.format(
                question=item["question"],
                alternatives=alternatives,
            )
            return prompt, None

        query = self.retriever.build_query(item)
        metadata = item.get("metadata", {})
        exam_year = metadata.get("exam_year")
        kwargs = {}
        # Retrievers that take a cutoff get the notice date; the rest filter by year.
        if "cutoff" in inspect.signature(self.retriever.retrieve).parameters:
            kwargs["cutoff"] = question_cutoff(metadata)
        chunks = self.retriever.retrieve(
            query=query,
            exam_year=exam_year,
            top_k=self.retrieval_top_k,
            **kwargs,
        )
        context = self.retriever.format_context(chunks)

        prompt = self.prompt_template.format(
            context=context,
            question=item["question"],
            alternatives=alternatives,
        )

        retrieval_meta = {
            # doc_id is globally unique; chunk_id repeats across laws.
            "retrieved_chunk_ids": [
                c.get("doc_id") or c["chunk_id"] for c in chunks],
            "retrieval_method": self.retriever.method,
            "retrieval_top_k": self.retrieval_top_k,
            "n_retrieved": len(chunks),
            "context_chars": len(context),
        }

        return prompt, retrieval_meta

    def parse_response(self, text: str) -> dict:
        """Extract the answer and the justification from a response."""

        answer = extract_answer(text)
        justification_match = re.search(
            r"<justification>\s*(.*?)\s*</justification>",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        justification = (
            justification_match.group(1).strip()
            if justification_match
            else None
        )

        return {
            "answer": answer,
            "justification": justification,
        }

    def _call_api(self, prompt: str) -> dict:
        """Call the configured provider."""

        if self.provider == "maritaca":
            return call_maritaca(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.num_predict,
            )
        if self.provider == "deepinfra":
            return call_deepinfra(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                reasoning_effort=self.think,
                max_tokens=self.num_predict,
                chat_template_kwargs=self.chat_template_kwargs,
            )
        return call_ollama(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            think=self.think,
            num_predict=self.num_predict,
        )

    async def _process_item(
        self,
        item: dict,
        idx: int,
        total: int,
        semaphore: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        fout,
    ) -> dict | None:
        """Answer one question and write its result line, whatever the outcome."""

        async with semaphore:
            logger.info("[{}/{}] START {}", idx, total, item["id"])

            prompt, retrieval_meta = self.build_prompt(item)

            # Base record shared by every outcome.
            result = {
                "id": item["id"],
                "question": item["question"],
                "choices": item["choices"],
                "gold_answer": item["answer"],
                "predicted_answer": None,
                "correct": False,
                "justification": None,
                "prompt": prompt,
                "raw_output": None,
                "input_tokens": None,
                "output_tokens": None,
                "elapsed_seconds": None,
                "estimated_cost": None,
                "timestamp": datetime.utcnow().isoformat(),
                "finish_reason": None,
                "status": STATUS_ERROR,
                "metadata": item["metadata"],
            }

            if retrieval_meta is not None:
                result.update(retrieval_meta)

            try:
                # Blocking provider call runs in a thread pool.
                loop = asyncio.get_running_loop()

                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self._call_api(prompt),
                        ),
                        timeout=self.request_timeout_seconds,
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        "[{}/{}] TIMEOUT {} | Exceeded {}s",
                        idx,
                        total,
                        item["id"],
                        self.request_timeout_seconds,
                    )
                    result["status"] = STATUS_TIMEOUT
                    await self._write_result(result, write_lock, fout)
                    return result

                parsed = self.parse_response(response["text"])

                predicted = parsed["answer"]
                gold = item["answer"]
                is_correct = predicted == gold if predicted else False

                logger.info(
                    "[{}/{}] DONE {} | Correct={} | Elapsed={:.2f}s",
                    idx,
                    total,
                    item["id"],
                    is_correct,
                    response["elapsed_seconds"],
                )

                result.update(
                    {
                        "predicted_answer": predicted,
                        "correct": is_correct,
                        "justification": parsed["justification"],
                        "raw_output": response["text"],
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                        "elapsed_seconds": response["elapsed_seconds"],
                        "estimated_cost": response["estimated_cost"],
                        "cost_currency": response.get("cost_currency", "USD"),
                        "finish_reason": response.get("finish_reason"),
                        # An unparseable answer counts as wrong.
                        "status": (
                            STATUS_ANSWERED if predicted else STATUS_NO_ANSWER
                        ),
                    }
                )

                await self._write_result(result, write_lock, fout)
                return result

            except Exception:
                logger.exception(
                    "Failed processing question {}",
                    item["id"],
                )
                result["status"] = STATUS_ERROR
                await self._write_result(result, write_lock, fout)
                return result

    async def _write_result(self, result: dict, write_lock, fout) -> None:
        """Append one result line under the write lock."""

        async with write_lock:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    async def _evaluate_async(self, dataset: list[dict]) -> list[dict | None]:
        """Answer every question under the concurrency limit; results keep dataset order."""

        semaphore = asyncio.Semaphore(self.concurrency)

        write_lock = asyncio.Lock()

        total = len(dataset)

        with self.results_path.open("w", encoding="utf-8") as fout:
            tasks = [
                self._process_item(
                    item, idx, total, semaphore, write_lock, fout)
                for idx, item in enumerate(dataset, start=1)
            ]

            return await tqdm_asyncio.gather(
                *tasks,
                desc="Evaluating",
                unit="question",
            )

    def evaluate(self):
        """Run the evaluation and append its metrics to the condition's metrics.json."""

        dataset = self.load_dataset()

        # results.jsonl is written incrementally.
        start_time = time.time()
        results = asyncio.run(self._evaluate_async(dataset))
        eval_total_seconds = time.time() - start_time

        aggregates = aggregate_result_metrics(results)

        config = {
            "provider": self.provider,
            "cost_currency": PROVIDER_CURRENCY.get(self.provider, "USD"),
            "run_id": self.run_dir.name,
            "model": self.model,
            "temperature": self.temperature,
            "think": self.think,
            "chat_template_kwargs": self.chat_template_kwargs,
            "num_predict": self.num_predict,
            "concurrency": self.concurrency,
            "sample_size": self.sample_size,
            "sample_random": self.sample_random,
            "sample_seed": self.sample_seed,
            "sample_per_group": self.sample_per_group,
            "sample_group_key": self.sample_group_key,
            "condition": self.condition,
            "retrieval_method": (
                None if self.retriever is None else self.retriever.method
            ),
            "retrieval_top_k": (
                None if self.retriever is None else self.retrieval_top_k
            ),
            "retrieval_query_mode": getattr(self.retriever, "query_mode", None),
            "retrieval_max_context_tokens": getattr(
                self.retriever, "max_context_tokens",
                getattr(getattr(self.retriever, "config", None),
                        "max_context_tokens", None)),
            "laws_version": (
                None if self.retriever is None else self.laws_version
            ),
            "retrieval_unit": self.retrieval_unit,
            "retrieval_scope": self.retrieval_scope,
            "graph_version": self.graph_version,
        }

        metrics = assemble_metrics(config, aggregates, eval_total_seconds)

        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(metrics)
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)

        logger.info("Finished evaluation")
        metrics_with_paths = {
            **metrics,
            "results_path": str(self.results_path),
            "metrics_path": str(self.metrics_path),
        }
        log_summary(metrics_with_paths)
