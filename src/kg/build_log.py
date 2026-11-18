"""Per-build logging under ``results/kg/<kg_version>/``.

``CallLogger`` writes one JSONL record per agent call to ``calls/<stage>.jsonl``;
``BuildLog`` appends one ``manifest.json`` entry per invocation and owns ``build.log``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.kg.config import PROJECT_ROOT

# Free providers: a missing estimated_cost from these is a real zero.
LOCAL_PROVIDERS = {"ollama", "mock"}

# dedup: template ref + variables, repeated values in calls/prompts/<sha>.txt;
# inline: full prompt; hash_only: sha256 only.
PROMPT_MODES = ("dedup", "inline", "hash_only")

# Shorter prompt variables always stay inline; longer ones move to the side
# table once a second call renders the same value.
VAR_INLINE_CHARS = 400

STATUS_OK = "ok"
STATUS_PARSE_FAILED = "parse_failed"
STATUS_ERROR = "error"


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    """Return the sha256 hex digest of a string."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str | None) -> str | None:
    """Return the sha256 hex digest of a file, or None when it is missing."""

    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: Path | str | None, role: str) -> dict:
    """Describe one input file for the manifest (role, path, size, sha256)."""

    resolved = Path(path) if path is not None else None
    exists = bool(resolved and resolved.is_file())
    return {
        "role": role,
        "path": str(resolved) if resolved else None,
        "exists": exists,
        "bytes": resolved.stat().st_size if exists else None,
        "sha256": sha256_file(resolved),
    }


def git_state() -> dict:
    """Return the repository commit, branch and dirty flag (None without git)."""

    def _run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                args, cwd=PROJECT_ROOT, capture_output=True,
                text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


class CallLogger:
    """Append-only JSONL log of every agent call, one file per stage.

    ``read_prompt`` reconstructs the rendered prompt from a record.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str = "run_001",
        prompt_mode: str = "dedup",
        store_raw_output: bool = True,
        store_parsed_output: bool = False,
        enabled: bool = True,
    ):
        """
        Args:
            output_dir (Path): Build folder (``results/kg/<kg_version>``).
            prompt_mode (str): One of ``PROMPT_MODES``.
            store_parsed_output (bool): Also persist the validated schema dump.
        """

        if prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}"
            )

        self.calls_dir = Path(output_dir) / "calls"
        self.prompts_dir = self.calls_dir / "prompts"
        self.run_id = run_id
        self.prompt_mode = prompt_mode
        self.store_raw_output = store_raw_output
        self.store_parsed_output = store_parsed_output
        self.enabled = enabled

        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._value_seen: dict[str, int] = {}
        self._template_sha: dict[tuple[str, str], str | None] = {}

        if self.enabled:
            self.calls_dir.mkdir(parents=True, exist_ok=True)
            if self.prompt_mode == "dedup":
                self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, stage: str) -> Path:
        """Return the JSONL file a stage's records are appended to."""

        return self.calls_dir / f"{stage}.jsonl"

    def _next_call_id(self, stage: str) -> str:
        """Return the next call id (``oe-00007``), continuing an existing file."""

        with self._lock:
            if stage not in self._counters:
                path = self.path_for(stage)
                existing = 0
                if path.exists():
                    with path.open("r", encoding="utf-8") as handle:
                        existing = sum(1 for line in handle if line.strip())
                self._counters[stage] = existing
            self._counters[stage] += 1
            return f"{stage}-{self._counters[stage]:05d}"

    def _template_fingerprint(self, agent: str, version: str) -> dict:
        """Describe the prompt template a call was rendered from."""

        key = (agent, version)
        if key not in self._template_sha:
            from src.kg.prompt_registry import PromptRegistry

            self._template_sha[key] = sha256_file(
                PromptRegistry().path(agent, version))
        return {
            "agent": agent,
            "version": version,
            "sha256": self._template_sha[key],
        }

    def _store_value(self, value: str):
        """Inline a prompt variable, or reference the side table once it repeats."""

        if len(value) < VAR_INLINE_CHARS:
            return value

        digest = sha256_text(value)
        with self._lock:
            seen = self._value_seen.get(digest, 0)
            self._value_seen[digest] = seen + 1

        target = self.prompts_dir / f"{digest}.txt"
        if seen == 0 and not target.exists():
            return value

        if not target.exists():
            # Atomic write so concurrent readers never see a partial file.
            temp = target.with_suffix(f".{threading.get_ident()}.tmp")
            temp.write_text(value, encoding="utf-8")
            temp.replace(target)
        return {
            "ref": f"prompts/{digest}.txt",
            "sha256": digest,
            "chars": len(value),
        }

    def _store_prompt(
        self,
        prompt: str,
        agent_config,
        variables: dict | None,
    ) -> dict:
        """Return the prompt fields of a record for the configured mode."""

        fields = {
            "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt),
            "prompt": None,
            "prompt_template": None,
            "prompt_vars": None,
        }

        if self.prompt_mode == "inline":
            fields["prompt"] = prompt
        elif self.prompt_mode == "dedup":
            fields["prompt_template"] = self._template_fingerprint(
                agent_config.name, agent_config.prompt_version)
            fields["prompt_vars"] = {
                name: self._store_value(str(value))
                for name, value in (variables or {}).items()
            }

        return fields

    def read_value(self, value):
        """Resolve one ``prompt_vars`` entry; empty if its side-table file is gone."""

        if isinstance(value, dict):
            path = self.calls_dir / value.get("ref", "")
            return path.read_text(encoding="utf-8") if path.exists() else ""
        return value

    def read_prompt(self, record: dict) -> str | None:
        """Recover a record's rendered prompt, re-rendering it if needed; None if lost."""

        if record.get("prompt") is not None:
            return record["prompt"]

        template = record.get("prompt_template")
        variables = record.get("prompt_vars")
        if not template or variables is None:
            return None

        from src.kg.prompt_registry import PromptRegistry

        try:
            return PromptRegistry().render(
                template["agent"], template["version"],
                **{k: self.read_value(v) for k, v in variables.items()},
            )
        except (FileNotFoundError, KeyError):
            return None

    def record(
        self,
        stage: str,
        agent_config,
        prompt: str,
        attempts: list[dict],
        context: dict | None = None,
        variables: dict | None = None,
        output=None,
        status: str = STATUS_OK,
        error: str | None = None,
    ) -> dict | None:
        """Append one call record and return it (None when logging is disabled).

        Args:
            attempts (list[dict]): One entry per model call (text, tokens, cost,
                finish_reason, parse_error).
            context (dict | None): Stage-specific traceability keys.
            status (str): ``ok``, ``parse_failed`` or ``error``.
        """

        if not self.enabled:
            return None

        provider = agent_config.provider
        accepted = attempts[-1] if attempts else {}

        input_tokens = sum(a.get("input_tokens") or 0 for a in attempts)
        output_tokens = sum(a.get("output_tokens") or 0 for a in attempts)
        elapsed = sum(a.get("elapsed_seconds") or 0.0 for a in attempts)

        cost = 0.0
        cost_reported = False
        for attempt in attempts:
            reported = attempt.get("estimated_cost")
            if reported is not None:
                cost += float(reported)
                cost_reported = True

        finish_reason = accepted.get("finish_reason")
        record = {
            "call_id": self._next_call_id(stage),
            "run_id": self.run_id,
            "timestamp": utc_now(),
            "stage": stage,
            "agent": agent_config.name,
            "context": context or {},
            "model": agent_config.model,
            "provider": provider,
            "prompt_version": agent_config.prompt_version,
            "params": {
                "temperature": agent_config.temperature,
                "max_tokens": agent_config.max_tokens,
                "num_ctx": agent_config.num_ctx,
                "think": agent_config.think,
            },
            **self._store_prompt(prompt, agent_config, variables),
            "raw_output": (
                accepted.get("text") if self.store_raw_output else None
            ),
            "thinking": accepted.get("thinking"),
            "parsed_ok": status == STATUS_OK,
            "parse_attempts": len(attempts),
            "parse_error": error,
            "output": (
                output.model_dump(mode="json")
                if (self.store_parsed_output and output is not None)
                else None
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "elapsed_seconds": elapsed,
            "estimated_cost": cost,
            "cost_reported": cost_reported,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "status": status,
        }

        # Earlier attempts are the responses that failed to parse.
        if len(attempts) > 1:
            record["discarded_attempts"] = [
                {
                    "text": a.get("text"),
                    "parse_error": a.get("parse_error"),
                    "finish_reason": a.get("finish_reason"),
                    "output_tokens": a.get("output_tokens"),
                }
                for a in attempts[:-1]
            ]

        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path_for(stage).open("a", encoding="utf-8") as handle:
                handle.write(line)
        return record


class BuildLog:
    """Run-level log of one build: ``manifest.json``, ``build.log`` and the ``CallLogger``."""

    def __init__(
        self,
        config,
        stages: tuple[str, ...] = ("oe", "ke", "kf"),
        inputs: dict[str, Path] | None = None,
        attach_sink: bool = True,
    ):
        """
        Args:
            inputs (dict[str, Path] | None): Extra input files to fingerprint,
                keyed by role (e.g. ``{"articles_csv": path}``).
        """

        self.config = config
        self.stages = tuple(stages)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_dir / "manifest.json"
        self.metrics_path = self.output_dir / "metrics.json"
        self.log_path = self.output_dir / "build.log"

        self.manifest = self._load_manifest()
        self.run_id = f"run_{len(self.manifest['runs']) + 1:03d}"
        self.extra_inputs = dict(inputs or {})

        self.calls = CallLogger(
            output_dir=self.output_dir,
            run_id=self.run_id,
            prompt_mode=config.log_prompts,
            store_raw_output=config.log_raw_output,
            store_parsed_output=config.log_parsed_output,
            enabled=config.log_calls,
        )

        self._entry: dict = {}
        self._started = None
        self._sink_id = None
        if attach_sink:
            self._sink_id = logger.add(
                self.log_path, level="INFO", rotation="20 MB",
                retention="30 days", enqueue=True,
            )

    def _load_manifest(self) -> dict:
        """Load the build manifest, initializing it on first use."""

        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "kg_version": self.config.kg_version,
            "output_dir": str(self.output_dir),
            "created_at": utc_now(),
            "runs": [],
        }

    def _save_manifest(self) -> None:
        """Persist the manifest to disk."""

        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2,
                       default=str),
            encoding="utf-8",
        )

    def _agent_fingerprints(self) -> dict:
        """Describe every agent's model, parameters and prompt template."""

        from src.kg.prompt_registry import PromptRegistry

        registry = PromptRegistry()
        table = {}
        for name, agent in self.config.agents.items():
            prompt_path = registry.path(name, agent.prompt_version)
            table[name] = {
                "model": agent.model,
                "provider": agent.provider,
                "prompt_version": agent.prompt_version,
                "temperature": agent.temperature,
                "max_tokens": agent.max_tokens,
                "num_ctx": agent.num_ctx,
                "think": agent.think,
                "prompt_path": str(prompt_path),
                "prompt_sha256": sha256_file(prompt_path),
            }
        return table

    def _input_fingerprints(self) -> list[dict]:
        """Fingerprint every input file this build reads."""

        roles = {
            "ontology_core": self.config.ontology_core_path,
            "ontology_seed": self.config.ontology_seed_path,
            **self.extra_inputs,
        }
        return [file_fingerprint(path, role) for role, path in roles.items()]

    def start(self) -> dict:
        """Open and return this invocation's manifest entry."""

        self._started = datetime.now(timezone.utc)
        config_json = json.dumps(
            self.config.to_json_dict(), ensure_ascii=False,
            sort_keys=True, default=str,
        )
        self._entry = {
            "run_id": self.run_id,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "duration_seconds": None,
            "stages_requested": list(self.stages),
            "stages": {},
            "kg_version": self.config.kg_version,
            "provider": self.config.provider,
            "concurrency": self.config.resolved_concurrency(),
            "agents": self._agent_fingerprints(),
            "inputs": self._input_fingerprints(),
            "config_sha256": sha256_text(config_json),
            "git": git_state(),
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "host": socket.gethostname(),
            },
            "logging": {
                "log_calls": self.config.log_calls,
                "log_prompts": self.config.log_prompts,
                "log_raw_output": self.config.log_raw_output,
                "log_parsed_output": self.config.log_parsed_output,
            },
        }
        self.manifest["runs"].append(self._entry)
        self._save_manifest()
        logger.info("Build run {} started | stages={} provider={}",
                    self.run_id, self.stages, self.config.provider)
        return self._entry

    def stage_started(self, stage: str) -> None:
        """Mark a stage as started in the manifest entry."""

        self._entry.setdefault("stages", {})[stage] = {
            "started_at": utc_now(), "finished_at": None, "seconds": None,
        }
        self._save_manifest()

    def stage_finished(self, stage: str) -> None:
        """Mark a stage as finished and record its wall-clock duration."""

        entry = self._entry.setdefault("stages", {}).setdefault(
            stage, {"started_at": utc_now()})
        started = datetime.fromisoformat(entry["started_at"])
        entry["finished_at"] = utc_now()
        entry["seconds"] = (
            datetime.now(timezone.utc) - started).total_seconds()
        self._save_manifest()

    def finish(self, status: str = "completed") -> None:
        """Close this invocation's manifest entry with ``status``."""

        self._entry["status"] = status
        self._entry["finished_at"] = utc_now()
        if self._started is not None:
            self._entry["duration_seconds"] = (
                datetime.now(timezone.utc) - self._started).total_seconds()
        self._save_manifest()
