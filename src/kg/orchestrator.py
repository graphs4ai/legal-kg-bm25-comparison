"""State machine (no LLM) driving the OE -> KE -> KF builder/critic agents.

Progress is checkpointed to ``state.json``; every routing decision goes to
``decisions.jsonl``.

    OE: per-shard builder/critic loops, then a merge critic; types kept by shard support.
    KE: per batch, builder extracts triples, critic flags violations, one retry.
    KF: blocking proposes merge candidates, builder groups, critic accepts or rejects.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from src.kg.agents import Agent
from src.kg.config import KGBuildConfig
from src.kg.graph_store import GraphStore, build_backbone, slugify
from src.kg.fusion import UnionFind, block_by_similarity, route_citations
from src.kg.packing import build_batches
from src.kg.schemas import (
    MODALITIES,
    STRUCTURAL_RELATIONS,
    ClusterMergeDecision,
    ExtractionProposal,
    ExtractionReview,
    MergeReview,
    Ontology,
    OntologyMergeReview,
    OntologyProposal,
    OntologyReview,
)
from src.kg.validation import (
    INDEX_DROPPED,
    drop_invalid_triples,
    rebase_violations,
    surviving_indices,
    validate_proposal,
)
from src.rag.text_utils import normalize_text

# Spelling-variant match for attributing type names ("legal_subject" vs "LegalSubjects").
_NAME_MATCH_THRESHOLD = 0.85


def _extraction_view(ontology: Ontology) -> Ontology:
    """Return the ontology without the structural layer the backbone owns."""

    return Ontology(
        version=ontology.version,
        notes=(
            "Camada semântica: os tipos abaixo classificam entidades e "
            "relações extraídas do texto dos artigos. Eles não correspondem "
            "a nós pré-existentes do grafo."
        ),
        entity_types=[
            e for e in ontology.entity_types
            if e.name not in ("NormativeAct", "Article")
        ],
        relation_types=[
            r for r in ontology.relation_types
            if r.name not in STRUCTURAL_RELATIONS
        ],
    )


def _spread_across_acts(batches: list, limit: int) -> list:
    """Take ``limit`` batches round-robin over acts, evenly spaced within each act."""

    by_act: dict[str, list] = {}
    for batch in batches:
        by_act.setdefault(batch.act_id, []).append(batch)

    order = {id(batch): index for index, batch in enumerate(batches)}
    queues = list(by_act.values())
    shares = [0] * len(queues)
    allotted = 0
    round_index = 0
    while allotted < limit and any(
        len(queue) > round_index for queue in queues
    ):
        for position, queue in enumerate(queues):
            if round_index < len(queue):
                shares[position] += 1
                allotted += 1
                if allotted == limit:
                    break
        round_index += 1

    picked: list = []
    for queue, share in zip(queues, shares):
        for i in range(share):
            picked.append(queue[(2 * i + 1) * len(queue) // (2 * share)])

    return sorted(picked, key=lambda batch: order[id(batch)])


def _name_words(name: str) -> list[str]:
    """Split a type name into lowercase words (snake_case or PascalCase)."""

    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^A-Za-z0-9]+", parts.lower()) if w]


def _generalized_to(name: str, candidates: list[str]) -> str | None:
    """Return the longest candidate whose words prefix ``name`` (``applies_to``
    for ``applies_to_subject``), else None.
    """

    words = _name_words(name)
    best: str | None = None
    best_len = 0
    for candidate in candidates:
        c_words = _name_words(candidate)
        if (0 < len(c_words) < len(words)
                and words[:len(c_words)] == c_words
                and len(c_words) > best_len):
            best, best_len = candidate, len(c_words)
    return best


def _number_triples(triples: list[dict]) -> str:
    """Render triples as ``[i] {json}`` lines; ``i`` is the critic's ``triple_index``."""

    return "\n".join(
        f"[{index}] {json.dumps(triple, ensure_ascii=False)}"
        for index, triple in enumerate(triples)
    ) or "(nenhuma)"


def _ke_feedback(violations: list, fallback: str) -> str:
    """Render KE violations as retry feedback; dropped triples are listed apart."""

    indexed = [v for v in violations if v.triple_index != INDEX_DROPPED]
    dropped = [v for v in violations if v.triple_index == INDEX_DROPPED]
    lines = [
        f"- Tripla [{v.triple_index}]: [{v.code}] {v.description}"
        for v in sorted(indexed, key=lambda v: v.triple_index)
    ]
    if dropped:
        lines.append(
            "Triplas já descartadas pelas verificações (não as repita):")
        lines.extend(f"- [{v.code}] {v.description}" for v in dropped)
    return "\n".join(lines) or fallback or "Extraia mais triplas válidas."


def _closest_name(name: str, candidates: list[str]) -> str | None:
    """Return the candidate that is a spelling variant of ``name``, else None."""

    def _key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    target = _key(name)
    best, best_score = None, 0.0
    for candidate in candidates:
        score = SequenceMatcher(None, target, _key(candidate)).ratio()
        if score > best_score:
            best, best_score = candidate, score
    return best if best_score >= _NAME_MATCH_THRESHOLD else None


class KGOrchestrator:
    """State machine driving the OE -> KE -> KF construction pipeline."""

    STAGES = ("oe", "ke", "kf", "done")

    def __init__(
        self,
        config: KGBuildConfig,
        agents: dict[str, Agent],
        units: list[dict],
        build_log=None,
    ):
        """
        Args:
            units (list[dict]): Article rows to process.
            build_log (BuildLog | None): Records stage timings and refreshes
                ``metrics.json`` after every stage.
        """

        self.config = config
        self.agents = agents
        self.units = units
        self.build_log = build_log

        self.output_dir = config.output_dir
        self.ontology_dir = self.output_dir / "ontology"
        self.extractions_path = self.output_dir / "extractions" / "extractions.jsonl"
        self.graph_dir = self.output_dir / "graph"
        self.state_path = self.output_dir / "state.json"
        self.decisions_path = self.output_dir / "decisions.jsonl"

        for folder in (self.output_dir, self.ontology_dir,
                       self.extractions_path.parent):
            folder.mkdir(parents=True, exist_ok=True)

        self.state = self._load_or_init_state()
        self._save_config()

        # Guards state counters mutated from worker threads.
        self._state_lock = threading.Lock()

    def _load_or_init_state(self) -> dict:
        """Load the saved state when resuming, else initialize a fresh one."""

        if self.config.resume and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            logger.info("Resuming from state: stage={}", state.get("stage"))
            return state
        return {
            "stage": "oe",
            "oe": {"shards": 0, "approved": False, "support": {}},
            "ke": {
                "done_units": 0, "retries": 0, "retries_improved": 0,
                "failed_units": [],
            },
            "kf": {
                "done": False, "auto_merges": 0, "citation_merges": 0,
                "similarity_merges": 0, "llm_merges_accepted": 0,
                "llm_merges_rejected": 0, "schema_extensions": [],
            },
            "usage": {
                "input_tokens": 0, "output_tokens": 0, "estimated_cost": 0.0,
            },
        }

    def _save_state(self) -> None:
        """Persist the current state to ``state.json``."""
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_config(self) -> None:
        """Persist the build configuration to ``config.json``."""
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config.to_json_dict(), ensure_ascii=False,
                       indent=2, default=str),
            encoding="utf-8",
        )

    def _log(self, stage: str, event: str, **detail) -> None:
        """Append one decision record to ``decisions.jsonl``."""

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "event": event,
            **detail,
        }
        with self.decisions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("[{}] {} {}", stage, event, detail if detail else "")

    def _track_usage(self, result) -> None:
        """Accumulate token/cost usage from an agent result into the state."""
        with self._state_lock:
            self.state["usage"]["input_tokens"] += result.usage["input_tokens"]
            self.state["usage"]["output_tokens"] += result.usage["output_tokens"]
            self.state["usage"]["estimated_cost"] += (
                result.usage["estimated_cost"])

    def run(
        self, stages: tuple[str, ...] = ("oe", "ke", "kf")
    ) -> GraphStore | None:
        """Run the allowed ``stages`` from the saved stage; returns the graph or None."""

        store: GraphStore | None = None
        if self.config.kg_update:
            self._reopen_for_update(stages)
        # A reused ontology is copied, so OE need not be in ``stages``.
        if self.config.oe_reuse_from and self.state["stage"] == "oe":
            self.run_oe()
            self.state["stage"] = "ke"
            self._save_state()
        if "oe" in stages and self.state["stage"] == "oe":
            self._run_stage("oe", self.run_oe)
            self.state["stage"] = "ke"
            self._save_state()
        if "ke" in stages and self.state["stage"] == "ke":
            self._run_stage("ke", self.run_ke)
            self.state["stage"] = "kf"
            self._save_state()
        if "kf" in stages and self.state["stage"] == "kf":
            store = self._run_stage("kf", self.run_kf)
            self.state["stage"] = "done"
            self._save_state()
        if store is None and self.graph_dir.joinpath("nodes.jsonl").exists():
            store = GraphStore.load(self.graph_dir)
        return store

    async def _bounded(self, awaitable):
        """Await a unit, under ``stage_call_timeout_seconds`` when set."""

        timeout = self.config.stage_call_timeout_seconds
        if timeout is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout)

    def _reopen_for_update(self, stages: tuple) -> None:
        """Move a finished build's stage pointer back to the earliest stage re-run."""

        current = self.state["stage"]
        target = next((s for s in ("ke", "kf") if s in stages), None)
        if target is None or current == "oe":
            return
        order = ("oe", "ke", "kf", "done")
        if order.index(target) >= order.index(current):
            return

        self.state["stage"] = target
        self.state["kf"]["done"] = False
        self._log("kf" if target == "kf" else target, "update_reopened",
                  from_stage=current, to_stage=target,
                  redo_batches=len(self.config.ke_redo_batch_ids or ()))
        logger.info("Update: reopening build at stage={} (was {})",
                    target, current)
        self._save_state()

    def _run_stage(self, stage: str, runner):
        """Run one stage, bracketed by its manifest timings."""

        if self.build_log is None:
            return runner()

        self.build_log.stage_started(stage)
        try:
            return runner()
        finally:
            self.build_log.stage_finished(stage)
            self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        """Recompute ``metrics.json`` from the artifacts written so far."""

        from src.kg.kg_metrics import write_metrics

        try:
            write_metrics(self.output_dir)
        except Exception as exc:
            logger.warning("Could not refresh metrics.json: {}", exc)

    # Ontology Engineering

    def load_core_ontology(self) -> Ontology:
        """Load the fixed structural ontology."""

        return Ontology.model_validate(json.loads(
            self.config.ontology_core_path.read_text(encoding="utf-8")))

    def load_seed_ontology(self) -> Ontology:
        """Load the semantic-overlay seed."""

        return Ontology.model_validate(json.loads(
            self.config.ontology_seed_path.read_text(encoding="utf-8")))

    def load_accepted_ontology(self) -> Ontology:
        """Load the ontology accepted by the OE stage (core plus overlay)."""

        path = self.ontology_dir / "ontology_accepted.json"
        return Ontology.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def build_shards(self) -> list[dict]:
        """Partition the corpus into OE shards (``name``, ``act_ids``, ``articles``).

        The largest acts get a shard each; the rest go to size-balanced tail shards.
        """

        substantive = [u for u in self.units if not u.get("is_boilerplate")]
        by_act: dict[str, list[dict]] = {}
        for unit in substantive:
            by_act.setdefault(unit["act_id"], []).append(unit)

        ranked = sorted(by_act, key=lambda a: -len(by_act[a]))
        dedicated = ranked[: self.config.oe_dedicated_acts]
        tail = ranked[self.config.oe_dedicated_acts:]

        shards = [
            {"name": act_id, "act_ids": [act_id], "articles": by_act[act_id]}
            for act_id in dedicated
        ]

        # Greedy: each act joins the lightest tail shard.
        n_tail = max(0, min(self.config.oe_tail_shards, len(tail)))
        if n_tail:
            buckets: list[dict] = [
                {"name": f"tail_{i + 1}", "act_ids": [], "articles": []}
                for i in range(n_tail)
            ]
            for act_id in tail:
                target = min(buckets, key=lambda b: len(b["articles"]))
                target["act_ids"].append(act_id)
                target["articles"].extend(by_act[act_id])
            shards.extend(b for b in buckets if b["articles"])

        if self.config.oe_shard_filter:
            wanted = set(self.config.oe_shard_filter)
            shards = [s for s in shards if s["name"] in wanted]

        return shards

    def _shard_sample_text(self, shard: dict) -> str:
        """Sample a shard's articles round-robin over structural paths, as labeled text."""

        by_path: dict[str, list[dict]] = {}
        for article in shard["articles"]:
            path = article.get("path")
            key = "" if path is None or pd.isna(path) else str(path)
            by_path.setdefault(key, []).append(article)

        paths = sorted(by_path)
        sample: list[dict] = []
        budget = min(self.config.oe_articles_per_shard, len(shard["articles"]))
        index = 0
        while len(sample) < budget and any(by_path.values()):
            bucket = by_path[paths[index % len(paths)]]
            if bucket:
                sample.append(bucket.pop(0))
            index += 1

        def _text(article: dict) -> str:
            """Caput, or the full text when the caput is empty."""
            caput = article.get("caput")
            if caput is None or pd.isna(caput) or not str(caput).strip():
                caput = article.get("article_text") or ""
            return str(caput).strip()

        def _path(article: dict) -> str:
            path = article.get("path")
            return "" if path is None or pd.isna(path) else f" — {path}"

        return "\n\n".join(
            f"[{a['lei_label']} — {a['tipo']}{_path(a)}]\n"
            f"Art. {a['article_number']}. {_text(a)}"
            for a in sample
        )

    def _run_oe_shard(self, shard: dict, core: Ontology) -> OntologyProposal:
        """Run the builder/critic loop for one shard over the core and its articles.

        The overlay seed is withheld; it enters only at the merge.
        """

        sample_text = self._shard_sample_text(shard)
        core_json = core.model_dump_json(indent=2)
        feedback = "Nenhum (primeira proposta)."
        proposal: OntologyProposal | None = None
        approved = False

        for iteration in range(1, self.config.oe_max_iterations + 1):
            context = {"shard": shard["name"], "iteration": iteration}
            builder_result = self.agents["oe_builder"].run(
                _context=context,
                laws_sample=sample_text,
                core_ontology=core_json,
                feedback=feedback,
            )
            self._track_usage(builder_result)
            proposal = builder_result.output
            self._log("oe", "shard_proposal", shard=shard["name"],
                      iteration=iteration,
                      entity_types=len(proposal.entity_types),
                      relation_types=len(proposal.relation_types),
                      call_id=builder_result.call_id)

            critic_result = self.agents["oe_critic"].run(
                _context=context,
                laws_sample=sample_text,
                core_ontology=core_json,
                proposal=proposal.model_dump_json(indent=2),
            )
            self._track_usage(critic_result)
            review: OntologyReview = critic_result.output
            blocking = review.blocking_issues()
            # Branch on blocking issues; the critic's own approval is only logged.
            self._log("oe", "shard_review", shard=shard["name"],
                      iteration=iteration, approval=not blocking,
                      critic_approval=review.approval,
                      issues=len(review.issues), blocking=len(blocking),
                      call_id=critic_result.call_id)

            if not blocking:
                approved = True
                break
            feedback = "\n".join(
                f"- [{issue.severity}/{issue.category}] {issue.description}"
                + (f" (sugestão: {issue.suggestion})" if issue.suggestion else "")
                for issue in review.issues
            ) or review.summary

        if not approved:
            self._log("oe", "shard_forced_accept", shard=shard["name"])
        return proposal

    async def _run_oe_shards_async(
        self, shards: list[dict], core: Ontology,
        proposals: dict[str, OntologyProposal],
    ) -> None:
        """Run every pending shard concurrently, filling ``proposals`` in place."""

        semaphore = asyncio.Semaphore(self.config.resolved_concurrency())
        loop = asyncio.get_running_loop()

        async def _run(shard: dict) -> None:
            async with semaphore:
                try:
                    proposal = await self._bounded(loop.run_in_executor(
                        executor, lambda: self._run_oe_shard(shard, core)))
                except Exception as exc:
                    self._log("oe", "shard_failed", shard=shard["name"],
                              error=f"{type(exc).__name__}: {exc}"[:200])
                    return
            proposals[shard["name"]] = proposal
            shard_path = self.ontology_dir / f"overlay_{shard['name']}.json"
            shard_path.write_text(
                proposal.to_ontology(shard["name"]).model_dump_json(indent=2),
                encoding="utf-8",
            )

        with ThreadPoolExecutor(
                max_workers=self.config.resolved_concurrency()) as executor:
            await tqdm_asyncio.gather(
                *[_run(shard) for shard in shards],
                desc="OE shards", unit="shard")

    def run_oe(self) -> Ontology:
        """Run sharded OE and return the accepted ontology (core plus merged overlay)."""

        if self.config.oe_reuse_from:
            return self._reuse_ontology(Path(self.config.oe_reuse_from))

        core = self.load_core_ontology()
        seed = self.load_seed_ontology()
        shards = self.build_shards()
        self._log("oe", "shards_built", shards=len(shards),
                  sizes={s["name"]: len(s["articles"]) for s in shards})

        proposals: dict[str, OntologyProposal] = {}
        pending: list[dict] = []
        for shard in shards:
            shard_path = self.ontology_dir / f"overlay_{shard['name']}.json"

            # A finished shard is reloaded instead of re-run.
            if self.config.resume and shard_path.exists():
                saved = Ontology.model_validate(
                    json.loads(shard_path.read_text(encoding="utf-8")))
                proposals[shard["name"]] = OntologyProposal(
                    entity_types=saved.entity_types,
                    relation_types=saved.relation_types,
                    rationale=saved.notes,
                )
                self._log("oe", "shard_reused", shard=shard["name"],
                          entity_types=len(saved.entity_types),
                          relation_types=len(saved.relation_types))
                continue

            pending.append(shard)

        if pending:
            asyncio.run(self._run_oe_shards_async(pending, core, proposals))

        overlay = self._merge_overlays(proposals, core, seed)
        accepted = core.merged_with(
            overlay, version=f"{self.config.kg_version}_oe")

        (self.ontology_dir / "overlay_accepted.json").write_text(
            overlay.model_dump_json(indent=2), encoding="utf-8")
        (self.ontology_dir / "ontology_accepted.json").write_text(
            accepted.model_dump_json(indent=2), encoding="utf-8")

        self.state["oe"]["shards"] = len(shards)
        self.state["oe"]["approved"] = True
        self._save_state()
        return accepted

    def _reuse_ontology(self, source_dir: Path) -> Ontology:
        """Copy a previous build's accepted ontology from its ``ontology`` folder."""

        accepted_path = source_dir / "ontology_accepted.json"
        accepted = Ontology.model_validate(
            json.loads(accepted_path.read_text(encoding="utf-8")))
        for name in ("ontology_accepted.json", "overlay_accepted.json"):
            source = source_dir / name
            if source.exists():
                (self.ontology_dir / name).write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8")
        self._log("oe", "ontology_reused", source=str(source_dir),
                  entity_types=len(accepted.entity_types),
                  relation_types=len(accepted.relation_types))
        self.state["oe"]["approved"] = True
        self._save_state()
        return accepted

    @staticmethod
    def _proposals_digest(proposals: dict[str, OntologyProposal]) -> str:
        """Aggregate shard proposals into a JSON digest, one entry per type name."""

        def _collect(kind: str) -> list[dict]:
            table: dict[str, dict] = {}
            for shard_name, proposal in proposals.items():
                for item in getattr(proposal, kind):
                    entry = table.setdefault(item.name, {
                        "name": item.name,
                        "proposed_by": [],
                        "descriptions": [],
                        "domain": set(),
                        "range": set(),
                    })
                    entry["proposed_by"].append(shard_name)
                    description = (item.description or "").strip()
                    if description and description not in entry["descriptions"]:
                        entry["descriptions"].append(description)
                    entry["domain"].update(getattr(item, "domain", []) or [])
                    entry["range"].update(getattr(item, "range", []) or [])

            rows = []
            for entry in sorted(
                table.values(), key=lambda e: (-len(e["proposed_by"]), e["name"])
            ):
                row = {
                    "name": entry["name"],
                    "shard_support": len(entry["proposed_by"]),
                    "proposed_by": entry["proposed_by"],
                    "descriptions": entry["descriptions"][:3],
                }
                if entry["domain"]:
                    row["domain"] = sorted(entry["domain"])
                if entry["range"]:
                    row["range"] = sorted(entry["range"])
                rows.append(row)
            return rows

        return json.dumps(
            {
                "shards": sorted(proposals),
                "entity_types": _collect("entity_types"),
                "relation_types": _collect("relation_types"),
            },
            ensure_ascii=False, indent=2,
        )

    def _merge_overlays(
        self,
        proposals: dict[str, OntologyProposal],
        core: Ontology,
        seed: Ontology,
    ) -> Ontology:
        """Consolidate per-shard overlays into the single overlay used by KE.

        Support transfers to survivors through ``merged_from``; seed types are
        exempt from the support threshold; a ceiling caps each list.
        """

        # Shard support per proposed type name.
        entity_support: dict[str, set[str]] = {}
        relation_support: dict[str, set[str]] = {}
        for shard_name, proposal in proposals.items():
            for entity in proposal.entity_types:
                entity_support.setdefault(entity.name, set()).add(shard_name)
            for relation in proposal.relation_types:
                relation_support.setdefault(
                    relation.name, set()).add(shard_name)

        merge_result = self.agents["oe_merge_critic"].run(
            _context={"shards": sorted(proposals)},
            core_ontology=core.model_dump_json(indent=2),
            seed_ontology=seed.model_dump_json(indent=2),
            proposals=self._proposals_digest(proposals),
        )
        self._track_usage(merge_result)
        merged: OntologyMergeReview = merge_result.output
        self._log("oe", "merge", entity_types=len(merged.entity_types),
                  relation_types=len(merged.relation_types),
                  dropped=len(merged.dropped),
                  call_id=merge_result.call_id)

        support_tables = {
            "entity": entity_support, "relation": relation_support}
        self._sanitize_merge(merged, support_tables)
        self._reconcile_merge(merged, support_tables)

        def _support(name: str, table: dict[str, set[str]]) -> int:
            shards = set(table.get(name, set()))
            for absorbed in merged.merged_from.get(name, []):
                shards |= table.get(absorbed, set())
            return len(shards)

        seed_entities = {e.name: e for e in seed.entity_types}
        seed_relations = {r.name: r for r in seed.relation_types}
        discarded = set(merged.dropped)

        # A seed type absorbed via ``merged_from`` passes its exemption to the survivor.
        absorbed_seed: dict[str, str] = {}
        for survivor, names in merged.merged_from.items():
            for name in names:
                if name in seed_entities or name in seed_relations:
                    absorbed_seed[name] = survivor
        seed_heirs = set(absorbed_seed.values())

        # Never require more shards than ran.
        threshold = min(self.config.oe_min_shard_support, len(proposals))
        stats = {"entity": {}, "relation": {}}

        def _filter(items, support_table, seed_names, kind):
            kept = []
            for item in items:
                support = _support(item.name, support_table)
                is_seed = item.name in seed_names or item.name in seed_heirs
                stats[kind][item.name] = support
                if is_seed or support >= threshold:
                    kept.append((is_seed, support, item))
                else:
                    self._log("oe", "type_dropped", kind=kind,
                              name=item.name, support=support,
                              threshold=threshold)
            return kept

        kept_entities = _filter(
            merged.entity_types, entity_support, seed_entities, "entity")
        kept_relations = _filter(
            merged.relation_types, relation_support, seed_relations, "relation")

        # Only seed relations are restored.
        self._restore_seed(kept_relations, seed_relations, discarded,
                           absorbed_seed, relation_support, stats["relation"],
                           "relation")

        # Seed types sort first so the ceiling never cuts them.
        kept_entities.sort(key=lambda item: (-int(item[0]), -item[1]))
        kept_relations.sort(key=lambda item: (-int(item[0]), -item[1]))
        entity_ceiling = self.config.oe_ceiling_entity_types
        relation_ceiling = self.config.oe_ceiling_relation_types
        if len(kept_entities) > entity_ceiling:
            self._log("oe", "ceiling_applied", kind="entity",
                      kept=entity_ceiling, proposed=len(kept_entities))
        if len(kept_relations) > relation_ceiling:
            self._log("oe", "ceiling_applied", kind="relation",
                      kept=relation_ceiling, proposed=len(kept_relations))

        self.state["oe"]["support"] = stats
        self._log("oe", "overlay_accepted",
                  entity_types=min(len(kept_entities), entity_ceiling),
                  relation_types=min(len(kept_relations), relation_ceiling))

        entities = [e for _, _, e in kept_entities[:entity_ceiling]]
        relations = [r for _, _, r in kept_relations[:relation_ceiling]]
        self._anchor_relation_domains(relations)
        self._prune_dangling_types(entities, relations, core)

        return Ontology(
            version="overlay_merged",
            entity_types=entities,
            relation_types=relations,
            notes=merged.rationale,
        )

    def _anchor_relation_domains(self, relations: list) -> None:
        """Set every overlay relation's domain to ``["Article"]``, in place."""

        rewritten = {
            r.name: list(r.domain or [])
            for r in relations if list(r.domain or []) != ["Article"]
        }
        for relation in relations:
            relation.domain = ["Article"]
        if rewritten:
            self._log("oe", "domains_anchored", relations=rewritten)

    def _prune_dangling_types(
        self,
        entities: list,
        relations: list,
        core: Ontology,
    ) -> None:
        """Broaden a domain/range naming a dropped type to all surviving overlay types."""

        valid = {e.name for e in entities} | core.entity_type_names()
        overlay_names = sorted(e.name for e in entities)

        for relation in relations:
            for field in ("domain", "range"):
                declared = getattr(relation, field) or []
                dangling = [name for name in declared if name not in valid]
                if not dangling:
                    continue
                kept = overlay_names or [
                    name for name in declared if name in valid
                ]
                setattr(relation, field, kept)
                self._log("oe", "dangling_types_pruned",
                          relation=relation.name, field=field,
                          dropped=dangling, kept=kept)

    def _sanitize_merge(
        self,
        merged: OntologyMergeReview,
        support_tables: dict[str, dict[str, set[str]]],
    ) -> None:
        """Repair a self-contradicting merge answer in place; survivor lists win.

        Survivors and ``merged_from`` keys leave ``dropped``; a non-survivor key
        is re-keyed to a matching survivor or removed; self, survivor and
        cross-kind absorptions are discarded.
        """

        survivors = {
            "entity": [e.name for e in merged.entity_types],
            "relation": [r.name for r in merged.relation_types],
        }
        all_survivors = set(survivors["entity"]) | set(survivors["relation"])
        kind_of_survivor = {name: "entity" for name in survivors["entity"]}
        kind_of_survivor.update(
            {name: "relation" for name in survivors["relation"]})
        repairs: dict[str, list[str]] = {
            "dropped_survivor": [], "dropped_absorber": [], "rekeyed": [],
            "unkeyed": [], "self_absorbed": [], "absorbed_survivor": [],
            "cross_kind": [],
        }

        contradicted = [name for name in merged.dropped if name in all_survivors]
        if contradicted:
            merged.dropped = [
                name for name in merged.dropped if name not in all_survivors
            ]
            repairs["dropped_survivor"] = sorted(set(contradicted))

        absorbers = [
            name for name in merged.dropped if name in merged.merged_from
        ]
        if absorbers:
            merged.dropped = [
                name for name in merged.dropped
                if name not in merged.merged_from
            ]
            repairs["dropped_absorber"] = sorted(set(absorbers))

        rekeyed: dict[str, list[str]] = {}
        for key, absorbed in merged.merged_from.items():
            target = key
            if key not in all_survivors:
                # A non-survivor key has no kind: match against both lists.
                target = (
                    _closest_name(key, survivors["entity"] + survivors["relation"])
                    or _generalized_to(
                        key, survivors["entity"] + survivors["relation"])
                )
                if target is None:
                    repairs["unkeyed"].append(key)
                    continue
                repairs["rekeyed"].append(f"{key}->{target}")

            kind = kind_of_survivor[target]
            other = "relation" if kind == "entity" else "entity"
            for name in absorbed:
                if name == target:
                    repairs["self_absorbed"].append(name)
                elif name in all_survivors:
                    repairs["absorbed_survivor"].append(name)
                elif (name not in support_tables[kind]
                        and name in support_tables[other]):
                    repairs["cross_kind"].append(f"{name}->{target}")
                else:
                    rekeyed.setdefault(target, []).append(name)

        merged.merged_from = {
            key: sorted(set(names)) for key, names in rekeyed.items() if names
        }
        if any(repairs.values()):
            self._log("oe", "merge_sanitized",
                      **{key: sorted(set(value))
                         for key, value in repairs.items() if value})

    def _reconcile_merge(
        self,
        merged: OntologyMergeReview,
        support_tables: dict[str, dict[str, set[str]]],
    ) -> None:
        """Attribute proposed names the merge left unaccounted to a matching survivor.

        Extends ``merged.merged_from`` in place.
        """

        survivors = {
            "entity": [e.name for e in merged.entity_types],
            "relation": [r.name for r in merged.relation_types],
        }
        accounted = set(survivors["entity"]) | set(survivors["relation"])
        accounted |= set(merged.dropped)
        for absorbed in merged.merged_from.values():
            accounted |= set(absorbed)

        attributed: dict[str, str] = {}
        unresolved: list[str] = []
        for kind, table in support_tables.items():
            for name in table:
                if name in accounted:
                    continue
                target = (
                    _closest_name(name, survivors[kind])
                    or _generalized_to(name, survivors[kind])
                )
                if target:
                    merged.merged_from.setdefault(target, []).append(name)
                    attributed[name] = target
                else:
                    unresolved.append(name)

        if attributed or unresolved:
            self._log("oe", "merge_unaccounted",
                      attributed=attributed, unresolved=sorted(unresolved))

    def _restore_seed(
        self,
        kept: list[tuple],
        seed_types: dict,
        discarded: set[str],
        absorbed: dict[str, str],
        support_table: dict[str, set[str]],
        stats: dict[str, int],
        kind: str,
    ) -> None:
        """Re-add seed types the merge omitted without dropping or absorbing them.

        Extends ``kept`` (``(is_seed, support, item)`` rows) and ``stats`` in place.
        """

        present = {item.name for _, _, item in kept}
        for name, seed_type in seed_types.items():
            if name in present:
                continue
            if name in absorbed:
                self._log("oe", "seed_absorbed", kind=kind, name=name,
                          survivor=absorbed[name])
                continue
            if name in discarded:
                self._log("oe", "seed_dropped", kind=kind, name=name)
                continue
            support = len(support_table.get(name, set()))
            stats[name] = support
            kept.append((True, support, seed_type))
            self._log("oe", "seed_restored", kind=kind, name=name)

    # Knowledge Extraction

    def _done_batch_ids(self) -> set[str]:
        """Return batch ids already recorded in ``extractions.jsonl``."""

        if not self.extractions_path.exists():
            return set()
        done = set()
        with self.extractions_path.open("r", encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["batch_id"])
        return done

    def _ke_needs_retry(self, kept: list[dict], violations: list,
                        truncated: bool = False) -> bool:
        """Whether one extraction pass needs a retry.

        A truncated pass always retries; blocking and advisory violations retry
        above an allowance proportional to the surviving triples.
        """

        if truncated:
            return True

        blocking = [v for v in violations if v.severity == "blocking"]
        advisory = [v for v in violations if v.severity != "blocking"]
        blocking_allowed = max(
            self.config.ke_violation_threshold,
            self.config.ke_violation_rate * len(kept),
        )
        advisory_allowed = max(
            self.config.ke_violation_threshold,
            self.config.ke_advisory_rate * len(kept),
        )
        if len(blocking) > blocking_allowed:
            return True
        if len(advisory) > advisory_allowed:
            return True
        if len(kept) < self.config.ke_min_triples:
            return True
        return False

    def run_ke(self) -> None:
        """Extract triples for every pending batch concurrently, appending records as they finish."""

        ontology = self.load_accepted_ontology()
        batches = build_batches(
            self.units,
            token_budget=self.config.ke_batch_tokens,
            max_articles=self.config.ke_batch_articles,
            solo_tokens=self.config.ke_solo_tokens,
        )
        if self.config.ke_max_batches:
            batches = _spread_across_acts(batches, self.config.ke_max_batches)
            self._log("ke", "batches_capped",
                      kept=len(batches), cap=self.config.ke_max_batches,
                      acts=sorted({b.act_id for b in batches}))
        done = self._done_batch_ids()
        redo = set(self.config.ke_redo_batch_ids or ())
        pending = [
            b for b in batches
            if b.batch_id not in done or b.batch_id in redo
        ]
        if redo:
            known = {b.batch_id for b in batches}
            self._log("ke", "batches_redone",
                      requested=len(redo), matched=len(redo & known),
                      unknown=sorted(redo - known)[:20])
            if redo - known:
                logger.warning(
                    "KE: {} requested redo batch id(s) are not in this "
                    "corpus and were ignored", len(redo - known))
        logger.info(
            "KE: {} batches pending ({} done, {} forced redo) over {} "
            "articles, concurrency={}",
            len(pending), len(done), len(redo & {b.batch_id for b in batches}),
            sum(len(b.articles) for b in pending),
            self.config.resolved_concurrency(),
        )
        if not pending:
            self._sync_ke_counters()
            self._save_state()
            return

        asyncio.run(self._run_ke_async(pending, ontology))
        self._sync_ke_counters()
        self._save_state()

    def _sync_ke_counters(self) -> None:
        """Reconcile the KE unit counters with the batch ids on disk."""

        done_ids = self._done_batch_ids()
        self.state["ke"]["done_units"] = len(done_ids)
        self.state["ke"]["failed_units"] = [
            batch_id for batch_id in self.state["ke"]["failed_units"]
            if batch_id not in done_ids
        ]

    async def _run_ke_async(self, batches: list, ontology: Ontology) -> None:
        """Process batches concurrently, appending records as they complete."""

        semaphore = asyncio.Semaphore(self.config.resolved_concurrency())
        write_lock = asyncio.Lock()
        ontology_json = _extraction_view(ontology).model_dump_json(indent=2)

        with self.extractions_path.open("a", encoding="utf-8") as fout, \
                ThreadPoolExecutor(
                    max_workers=self.config.resolved_concurrency()
                ) as executor:
            await tqdm_asyncio.gather(*[
                self._process_batch(
                    batch, ontology, ontology_json, semaphore, write_lock,
                    fout, executor)
                for batch in batches
            ], desc="Extracting", unit="batch")

    async def _process_batch(self, batch, ontology: Ontology,
                             ontology_json: str, semaphore, write_lock,
                             fout, executor) -> None:
        """Run one batch through builder, checks and critic, then append its record."""

        async with semaphore:
            loop = asyncio.get_running_loop()
            try:
                record = await self._bounded(loop.run_in_executor(
                    executor,
                    lambda: self._extract_batch(
                        batch, ontology, ontology_json),
                ))
            except Exception as exc:
                if batch.batch_id not in self.state["ke"]["failed_units"]:
                    self.state["ke"]["failed_units"].append(batch.batch_id)
                self._log("ke", "batch_failed", batch_id=batch.batch_id,
                          error=f"{type(exc).__name__}: {exc}"[:200])
                return

            async with write_lock:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                self.state["ke"]["done_units"] += 1
                if batch.batch_id in self.state["ke"]["failed_units"]:
                    self.state["ke"]["failed_units"].remove(batch.batch_id)
                if (self.state["ke"]["done_units"]
                        % self.config.state_save_every == 0):
                    self._save_state()

    def _extract_batch(self, batch, ontology: Ontology,
                       ontology_json: str) -> dict:
        """Run builder -> checks -> critic (-> retry) for one batch; returns its record."""

        builder = self.agents["ke_builder"]
        critic = self.agents["ke_critic"]
        article_text = batch.to_prompt_text()
        article_numbers = batch.article_numbers
        call_ids: list[str] = []

        def _pass(previous: str, feedback: str, attempt: int):
            context = {
                "batch_id": batch.batch_id,
                "act_id": batch.act_id,
                "article_ids": batch.article_ids,
                "pass": attempt,
            }
            builder_result = builder.run(
                _context=context,
                article_text=article_text,
                article_numbers=", ".join(article_numbers),
                ontology=ontology_json,
                previous_triples=previous,
                feedback=feedback,
            )
            self._track_usage(builder_result)
            proposal: ExtractionProposal = builder_result.output

            raw_checks = validate_proposal(
                proposal, ontology, article_text, article_numbers)
            survivors = surviving_indices(proposal, raw_checks)
            kept = drop_invalid_triples(proposal, raw_checks)
            # Put check violations in the critic's index space (``kept``).
            checks = rebase_violations(raw_checks, survivors)

            if not kept:
                call_ids.extend(
                    cid for cid in (builder_result.call_id,) if cid)
                return (proposal, kept, checks, ExtractionReview(),
                        builder_result.truncated)

            critic_result = critic.run(
                _context=context,
                article_text=article_text,
                ontology=ontology_json,
                triples=_number_triples(kept),
            )
            self._track_usage(critic_result)
            review: ExtractionReview = critic_result.output
            normalized, remapped = review.normalized_violations()
            if remapped:
                self._log("ke", "codes_normalized", batch_id=batch.batch_id,
                          pass_=attempt, codes=sorted(set(remapped)))
            review = review.model_copy(update={"violations": normalized})

            call_ids.extend(
                cid for cid in (builder_result.call_id, critic_result.call_id)
                if cid
            )
            return proposal, kept, checks, review, builder_result.truncated

        proposal, kept, checks, review, truncated = _pass(
            "Nenhuma (primeira extração).", "Nenhum (primeira extração).", 1)
        violations = checks + list(review.violations)

        def _opt_int(value):
            return None if value is None or pd.isna(value) else int(value)

        # Validity window of each wording in the batch.
        article_windows = {
            article["article_id"]: {
                "valid_from": _opt_int(article.get("valid_from")),
                "valid_until": _opt_int(article.get("valid_until")),
            }
            for article in batch.articles
        }

        record = {
            "batch_id": batch.batch_id,
            "act_id": batch.act_id,
            "article_ids": batch.article_ids,
            "article_windows": article_windows,
            "triples": kept,
            "violations": [v.model_dump() for v in violations],
            "retried": False,
            "retry_improved": None,
            "truncated": truncated,
            "call_ids": call_ids,
        }

        retries = 0
        while (self._ke_needs_retry(kept, violations, truncated)
               and retries < self.config.ke_max_retries_per_unit):
            retries += 1
            with self._state_lock:
                self.state["ke"]["retries"] += 1
            self._log("ke", "retry", batch_id=batch.batch_id,
                      violations=len(violations), triples=len(kept),
                      truncated=truncated)

            (retry_proposal, retry_kept, retry_checks, retry_review,
             retry_truncated) = _pass(
                _number_triples(kept),
                _ke_feedback(violations, review.feedback),
                retries + 1)
            retry_violations = retry_checks + list(retry_review.violations)

            # Rank passes by triples net of blocking flags, then fewer advisory flags.
            def _score(triples, violations):
                scored = [
                    v for v in violations if v.triple_index != INDEX_DROPPED]
                blocking = sum(
                    1 for v in scored if v.severity == "blocking")
                advisory = len(scored) - blocking
                return (len(triples) - blocking, -advisory, len(triples))

            improved = (
                _score(retry_kept, retry_violations) > _score(kept, violations)
            )
            if improved:
                with self._state_lock:
                    self.state["ke"]["retries_improved"] += 1
            else:
                retry_proposal, retry_kept = proposal, kept
                retry_violations = violations
                retry_truncated = truncated
            record.update({
                "retried": True,
                "retry_improved": improved,
                "first_pass_triples": record["triples"],
                "first_pass_violations": record["violations"],
                "triples": retry_kept,
                "violations": [v.model_dump() for v in retry_violations],
                "truncated": retry_truncated,
            })
            proposal, kept, violations, review, truncated = (
                retry_proposal, retry_kept, retry_violations, retry_review,
                retry_truncated)

        self._log("ke", "batch_done", batch_id=batch.batch_id,
                  articles=len(batch.articles), triples=len(kept),
                  violations=len(violations), retried=record["retried"])
        return record

    # Knowledge Fusion

    def _load_extractions(self) -> list[dict]:
        """Load the latest extraction record per batch, sorted by batch id."""

        latest: dict[str, dict] = {}
        superseded = 0
        with self.extractions_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                batch_id = record.get("batch_id", "")
                if batch_id in latest:
                    superseded += 1
                latest[batch_id] = record
        if superseded:
            logger.info(
                "KF: {} superseded extraction record(s) ignored in favour of "
                "the latest per batch", superseded)
        return sorted(latest.values(), key=lambda r: r.get("batch_id", ""))

    @staticmethod
    def _triple_provenance(record: dict, triple: dict) -> dict:
        """Provenance of one triple: its article and that wording's validity window."""

        act_id = record["act_id"]
        number = triple.get("article_number")
        article_id = f"{act_id}:art_{number}" if number else None
        if article_id not in record.get("article_ids", []):
            # Unresolved attribution falls back to the batch's first article.
            article_id = (record.get("article_ids") or [None])[0]
        window = (record.get("article_windows") or {}).get(article_id, {})
        return {
            "article_id": article_id, "act_id": act_id,
            "valid_from": window.get("valid_from"),
            "valid_until": window.get("valid_until"),
        }

    def _collect_mentions(
        self, records: list[dict]
    ) -> dict[tuple[str, str], dict]:
        """Group object mentions by (type, normalized text)."""

        groups: dict[tuple[str, str], dict] = {}
        for record in records:
            for triple in record["triples"]:
                prov = self._triple_provenance(record, triple)
                # Only the object is an extracted entity; the subject is the article.
                mention = triple["object"]
                key = (mention["type"], normalize_text(mention["text"]))
                group = groups.setdefault(key, {
                    "name": mention["text"], "type": mention["type"],
                    "aliases": [], "provenance": [], "count": 0,
                    "act_id": prov.get("act_id"),
                })
                group["count"] += 1
                if (mention["text"] != group["name"]
                        and mention["text"] not in group["aliases"]):
                    group["aliases"].append(mention["text"])
                if prov not in group["provenance"]:
                    group["provenance"].append(prov)
        return groups

    def _act_registry(self):
        """Act registry over this build's corpus, for resolving citation mentions."""

        from src.data.laws_citations import ActRegistry, normalize_tipo

        acts: dict[str, dict] = {}
        for unit in self.units:
            act_id = unit["act_id"]
            if act_id in acts:
                continue
            num = unit.get("num_lei")
            acts[act_id] = {
                "tipo": unit.get("tipo"),
                "tipo_key": normalize_tipo(unit.get("tipo") or ""),
                "num": (
                    None if num is None or (isinstance(num, float) and num != num)
                    else str(int(num)) if isinstance(num, (int, float))
                    else str(num)
                ),
                "ano": (
                    None if unit.get("ano") is None
                    or (isinstance(unit.get("ano"), float)
                        and unit["ano"] != unit["ano"])
                    else int(unit["ano"])
                ),
                "label": unit.get("lei_label"),
            }
        return ActRegistry(acts)

    def _describe_mention(self, groups: dict, key) -> str:
        """One-line description of a mention (name, type, sources) for a prompt."""

        group = groups[key]
        sources = ", ".join(
            p["article_id"] for p in group["provenance"][:4]
            if p.get("article_id")
        )
        return (f"'{group['name']}' (tipo: {group['type']}; "
                f"ocorrências: {group['count']}; fontes: {sources})")

    @staticmethod
    def _resolve_label(
        name: str, ids: dict, labels: dict, by_name: dict
    ) -> str | None:
        """Map one builder entry back to its mention key, or None.

        Accepts the mention id ("m3"), the full label, or the quoted name.
        """

        stripped = name.strip()
        if stripped in ids:
            return ids[stripped]
        id_match = re.match(r"^(m\d+)\b[:.]?", stripped)
        if id_match and id_match.group(1) in ids:
            return ids[id_match.group(1)]
        if name in labels:
            return labels[name]
        match = re.search(r"'([^']*)'", name)
        return by_name.get(match.group(1)) if match else None

    def _review_cluster(
        self, groups: dict, candidate, cluster_id: str = "",
    ) -> tuple[list[list], dict, str | None, str | None]:
        """Ask the KF agents which mentions of a cluster co-refer.

        Runs on a worker thread, so it only reads ``groups``.

        Returns:
            tuple: Accepted key groups, survivor-key -> canonical-name renames,
                and the builder/critic call ids (None when nothing is accepted).
        """

        ids = {
            f"m{index}": key
            for index, key in enumerate(candidate.keys, start=1)
        }
        labels = {
            self._describe_mention(groups, key): key
            for key in candidate.keys
        }
        by_name = {groups[key]["name"]: key for key in candidate.keys}
        listing = "\n".join(
            f"- {mention_id}: {self._describe_mention(groups, key)}"
            for mention_id, key in ids.items()
        )
        context = {"cluster_id": cluster_id, "size": len(candidate.keys)}

        builder_result = self.agents["kf_builder"].run(
            _context=context, mentions=listing)
        self._track_usage(builder_result)
        decision: ClusterMergeDecision = builder_result.output

        # No critic call when the builder merged nothing.
        if not any(len(names) > 1 for names in decision.groups):
            return [], {}, None, None

        critic_result = self.agents["kf_critic"].run(
            _context=context,
            mentions=listing,
            decision=decision.model_dump_json(indent=2),
        )
        self._track_usage(critic_result)
        review: MergeReview = critic_result.output

        if review.verdict != "accept":
            with self._state_lock:
                self.state["kf"]["llm_merges_rejected"] += 1
            self._log("kf", "cluster_rejected", cluster_id=cluster_id,
                      size=len(candidate.keys),
                      call_id=critic_result.call_id)
            return [], {}, None, None

        accepted: list[list] = []
        renames: dict = {}
        for index, names in enumerate(decision.groups):
            keys = [
                self._resolve_label(n, ids, labels, by_name) for n in names
            ]
            keys = [k for k in keys if k is not None]
            if len(keys) > 1:
                accepted.append(keys)
                canonical = (
                    decision.canonical_names[index]
                    if index < len(decision.canonical_names) else ""
                )
                if canonical:
                    renames[keys[0]] = canonical
        if not accepted and any(len(names) > 1 for names in decision.groups):
            self._log("kf", "cluster_unresolved", cluster_id=cluster_id,
                      groups=decision.groups,
                      call_id=builder_result.call_id)
        return accepted, renames, builder_result.call_id, critic_result.call_id

    def _apply_cluster_result(
        self, groups: dict, cluster_id: str, accepted: list[list],
        renames: dict, builder_call_id: str, critic_call_id: str,
    ) -> None:
        """Fold one cluster's accepted result into shared state, on the event loop."""

        if not accepted:
            return
        for key, canonical in renames.items():
            groups[key]["name"] = canonical
        with self._state_lock:
            self.state["kf"]["llm_merges_accepted"] += len(accepted)
        self._log("kf", "cluster_accepted", cluster_id=cluster_id,
                  groups=len(accepted),
                  merged=sum(len(k) for k in accepted),
                  call_id=builder_call_id, critic_call_id=critic_call_id)

    async def _run_kf_round_async(
        self, groups: dict, review: list, round_index: int,
        judged: set[frozenset], renames: dict | None = None,
    ) -> list[list]:
        """Judge every review cluster of one round concurrently.

        Args:
            judged (set[frozenset]): Key sets already judged; extended in place.
            renames (dict | None): Survivor key -> canonical name; extended in place.

        Returns:
            list[list]: Accepted merge groups from every cluster.
        """

        semaphore = asyncio.Semaphore(self.config.resolved_concurrency())
        loop = asyncio.get_running_loop()

        async def _judge(index: int, candidate) -> list[list]:
            cluster_id = f"r{round_index}-c{index:04d}"
            key_set = frozenset(candidate.keys)
            if key_set in judged:
                self._log("kf", "cluster_skipped", cluster_id=cluster_id,
                          size=len(candidate.keys), reason="already_judged")
                return []
            async with semaphore:
                try:
                    (accepted, cluster_renames, builder_call_id,
                     critic_call_id) = await self._bounded(
                        loop.run_in_executor(
                            executor,
                            lambda: self._review_cluster(
                                groups, candidate, cluster_id),
                        ))
                    judged.add(key_set)
                    self._apply_cluster_result(
                        groups, cluster_id, accepted, cluster_renames,
                        builder_call_id, critic_call_id)
                    if accepted and renames is not None:
                        renames.update(cluster_renames)
                    return accepted
                except Exception as exc:
                    self._log("kf", "cluster_failed", cluster_id=cluster_id,
                              size=len(candidate.keys),
                              error=f"{type(exc).__name__}: {exc}"[:160])
                    return []

        with ThreadPoolExecutor(
                max_workers=self.config.resolved_concurrency()) as executor:
            results = await tqdm_asyncio.gather(
                *[_judge(i, c) for i, c in enumerate(review, start=1)],
                desc=f"KF round {round_index}", unit="cluster")
        return [group for accepted in results for group in accepted]

    def _apply_merges(self, groups: dict, union, key_groups: list) -> None:
        """Union mention keys and fold their aliases and provenance together."""

        for keys in key_groups:
            # The best-attested surface form survives.
            keys = sorted(
                keys,
                key=lambda k: (-groups[k]["count"], len(groups[k]["name"])),
            )
            survivor = union.find(keys[0])
            for key in keys[1:]:
                if union.union(survivor, key):
                    target, source = groups[survivor], groups[key]
                    for alias in [source["name"], *source["aliases"]]:
                        if (alias != target["name"]
                                and alias not in target["aliases"]):
                            target["aliases"].append(alias)
                    for record in source["provenance"]:
                        if record not in target["provenance"]:
                            target["provenance"].append(record)
                    target["count"] += source["count"]

    @property
    def fusion_state_path(self) -> Path:
        """Path of the persisted fusion decisions."""

        return self.graph_dir / "fusion_state.json"

    def _kf_fingerprint(self) -> dict:
        """Model, prompt version and prompt hash of each KF agent."""

        fingerprint = {}
        for name in ("kf_builder", "kf_critic"):
            agent = self.agents.get(name)
            if agent is None:
                continue
            fingerprint[name] = {
                "model": agent.config.model,
                "prompt_version": agent.config.prompt_version,
                "prompt_sha256": agent.prompt_sha256(),
            }
        return fingerprint

    def _save_fusion_state(
        self, merges: list[list], renames: dict, judged: set[frozenset],
    ) -> None:
        """Persist agent verdicts (merges, renames, judged clusters) for reuse."""

        payload = {
            "kg_version": self.config.kg_version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self._kf_fingerprint(),
            "merges": [[list(key) for key in group] for group in merges],
            "renames": [
                [list(key), name] for key, name in sorted(renames.items())
            ],
            "judged": [
                sorted(list(key) for key in keys) for keys in judged
            ],
        }
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.fusion_state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_fusion_state(self) -> dict | None:
        """Read saved fusion decisions; None if absent or a KF agent changed."""

        if not self.fusion_state_path.exists():
            return None

        payload = json.loads(
            self.fusion_state_path.read_text(encoding="utf-8"))
        stored = payload.get("fingerprint", {})
        current = self._kf_fingerprint()
        if stored != current:
            changed = sorted(
                name for name in set(stored) | set(current)
                if stored.get(name) != current.get(name)
            )
            logger.warning(
                "KF: ignoring fusion state — {} changed since it was written; "
                "re-fusing from scratch", ", ".join(changed),
            )
            self._log("kf", "fusion_state_invalidated", agents=changed)
            return None

        return {
            "merges": [
                [tuple(key) for key in group]
                for group in payload.get("merges", [])
            ],
            "renames": {
                tuple(key): name for key, name in payload.get("renames", [])
            },
            "judged": {
                frozenset(tuple(key) for key in keys)
                for keys in payload.get("judged", [])
            },
        }

    def _restore_fusion_state(
        self, prior: dict, groups: dict, union,
    ) -> tuple[int, set[frozenset]]:
        """Re-apply saved merges to the current mention pool, skipping missing keys.

        Returns:
            tuple[int, set[frozenset]]: Mentions absorbed, and judged key sets.
        """

        restored = [
            [key for key in group if key in groups]
            for group in prior["merges"]
        ]
        restored = [group for group in restored if len(group) > 1]
        self._apply_merges(groups, union, restored)

        for key, name in prior["renames"].items():
            if key in groups and union.find(key) == key:
                groups[key]["name"] = name

        merged_mentions = sum(len(group) - 1 for group in restored)
        dropped = len(prior["merges"]) - len(restored)
        self._log("kf", "fusion_state_restored",
                  merges=len(restored), merged_mentions=merged_mentions,
                  judged=len(prior["judged"]), stale_merges=dropped)
        return merged_mentions, set(prior["judged"])

    def run_kf(self) -> GraphStore:
        """Fuse extracted mentions into the final graph and save it.

        Citations merge by parsed id; the rest are blocked by similarity, with
        ambiguous clusters sent to the agents, for up to ``kf_max_rounds``.
        """

        ontology = self.load_accepted_ontology()
        records = self._load_extractions()
        groups = self._collect_mentions(records)
        registry = self._act_registry()

        exact_merged = sum(g["count"] - 1 for g in groups.values())
        self.state["kf"]["auto_merges"] = exact_merged
        self._log("kf", "auto_merge", groups=len(groups),
                  merged_mentions=exact_merged)

        union = UnionFind()
        for key in groups:
            union.find(key)

        source_acts = {k: g.get("act_id") for k, g in groups.items()}
        resolved, citation_candidates = route_citations(
            groups, registry, source_acts)
        self._apply_merges(
            groups, union, [c.keys for c in citation_candidates])
        self.state["kf"]["citation_merges"] = sum(
            len(c.keys) - 1 for c in citation_candidates)
        self._log("kf", "citations_routed", resolved=len(resolved),
                  merged_groups=len(citation_candidates))

        prior = self._load_fusion_state()
        judged: set[frozenset] = set()
        agent_merges: list[list] = []
        renames: dict = {}
        if prior:
            restored, judged = self._restore_fusion_state(prior, groups, union)
            agent_merges = [
                [key for key in group if key in groups]
                for group in prior["merges"]
            ]
            agent_merges = [g for g in agent_merges if len(g) > 1]
            renames = {
                key: name for key, name in prior["renames"].items()
                if key in groups
            }
            self.state["kf"]["restored_merges"] = restored

        for round_index in range(1, self.config.kf_max_rounds + 1):
            live = {
                key: group for key, group in groups.items()
                if union.find(key) == key
            }
            auto, review, method = block_by_similarity(
                live,
                skip=set(resolved),
                auto_threshold=self.config.kf_auto_threshold,
                review_threshold=self.config.kf_review_threshold,
                max_cluster_size=self.config.kf_max_cluster_size,
                max_review_clusters=self.config.kf_max_review_clusters,
                use_embeddings=self.config.kf_use_embeddings,
            )
            if self.config.kf_max_candidates:
                review = review[: self.config.kf_max_candidates]

            self._log("kf", "blocking", round=round_index, method=method,
                      auto=len(auto), review=len(review))
            if not auto and not review:
                break

            self._apply_merges(groups, union, [c.keys for c in auto])
            self.state["kf"]["similarity_merges"] += sum(
                len(c.keys) - 1 for c in auto)

            accepted = asyncio.run(self._run_kf_round_async(
                groups, review, round_index, judged, renames))
            self._apply_merges(groups, union, accepted)
            agent_merges.extend(accepted)

            # Nothing merged: the next round would see the same clusters.
            if not auto and not accepted:
                self._log("kf", "round_idle", round=round_index,
                          review=len(review))
                break

        merge_map = {
            key: union.find(key) for key in groups if union.find(key) != key
        }
        store = self._assemble_graph(ontology, records, groups, merge_map)
        self._save_fusion_state(agent_merges, renames, judged)
        store.save(self.graph_dir)
        store.export_cypher(self.graph_dir / "graph.cypher")
        self.state["kf"]["done"] = True
        self._log("kf", "graph_saved", **store.stats())
        self._save_state()
        return store

    @staticmethod
    def _merge_root(merge_map: dict, key):
        """Follow merge redirects to a group's surviving root key."""

        while key in merge_map and merge_map[key] != key:
            key = merge_map[key]
        return key

    def _assemble_graph(
        self,
        ontology: Ontology,
        records: list[dict],
        groups: dict[tuple[str, str], dict],
        merge_map: dict,
    ) -> GraphStore:
        """Assemble the final graph: backbone + entities + triple edges."""

        store = build_backbone(pd.DataFrame(self.units), meta={
            "kg_version": self.config.kg_version,
            "ontology_version": ontology.version,
        })

        known_relations = ontology.relation_type_names() | {"contains", "mentions"}
        unknown_relations = set(self.state["kf"]["schema_extensions"])

        def _provenance_window(prov: dict) -> dict:
            """Validity of the wording that asserted a triple, else the article's."""

            if prov.get("valid_from") is not None or prov.get("valid_until") is not None:
                return {"valid_from": prov.get("valid_from"),
                        "valid_until": prov.get("valid_until")}
            node = store.nodes.get(f"article:{prov.get('article_id')}")
            if node is None:
                return {"valid_from": None, "valid_until": None}
            return {"valid_from": node.attrs.get("valid_from"),
                    "valid_until": node.attrs.get("valid_until")}

        def _window(provenance: list[dict]) -> dict:
            """Widest validity interval covering several assertions."""

            starts, open_ended, ends = [], False, []
            for prov in provenance:
                window = _provenance_window(prov)
                if window["valid_from"] is not None:
                    starts.append(window["valid_from"])
                if window["valid_until"] is None:
                    open_ended = True
                else:
                    ends.append(window["valid_until"])
            return {
                "valid_from": min(starts) if starts else None,
                "valid_until": None if open_ended or not ends else max(ends),
            }

        def _node_id_for(key: tuple[str, str]) -> str:
            root = self._merge_root(merge_map, key)
            group = groups[root]
            return f"{slugify(group['type'])}:{slugify(group['name'])}"

        for root, group in groups.items():
            if root in merge_map:
                continue
            store.add_node(
                node_id=f"{slugify(group['type'])}:{slugify(group['name'])}",
                name=group["name"],
                type=group["type"],
                aliases=group["aliases"],
                provenance=group["provenance"],
                attrs=_window(group["provenance"]),
            )
            for prov in group["provenance"]:
                article_id = f"article:{prov['article_id']}"
                if article_id in store.nodes:
                    store.add_edge(
                        article_id, "mentions",
                        f"{slugify(group['type'])}:{slugify(group['name'])}",
                        provenance=[prov],
                        attrs=_window([prov]),
                    )

        for record in records:
            for triple in record["triples"]:
                prov = self._triple_provenance(record, triple)
                # Edges run from the asserting article to the entity.
                source_id = f"article:{prov['article_id']}"
                target_key = (
                    triple["object"]["type"],
                    normalize_text(triple["object"]["text"]),
                )
                if source_id not in store.nodes or target_key not in groups:
                    continue
                target_id = _node_id_for(target_key)
                if source_id == target_id:
                    continue
                relation = triple["relation"]
                if relation not in known_relations:
                    if relation not in unknown_relations:
                        unknown_relations.add(relation)
                        self.state["kf"]["schema_extensions"].append(relation)
                        self._log("kf", "schema_extension", relation=relation)
                attrs = _window([prov])
                if triple.get("confidence") is not None:
                    attrs["confidence"] = triple["confidence"]
                if triple.get("modality") in MODALITIES:
                    attrs["modality"] = triple["modality"]
                store.add_edge(
                    source_id, relation, target_id,
                    provenance=[prov], attrs=attrs,
                )

        return store
