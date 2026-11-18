"""Aggregate a KG build folder's artifacts (calls, manifest, state, decisions,
extractions, graph) into ``metrics.json``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.kg.build_log import LOCAL_PROVIDERS

STAGES = ("oe", "ke", "kf")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank and malformed lines; [] if missing."""

    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_json(path: Path, default=None):
    """Read a JSON file, or return ``default`` when missing or invalid."""

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_calls(output_dir: Path, stage: str | None = None) -> list[dict]:
    """Load call records from a build folder, for one stage or all."""

    calls_dir = Path(output_dir) / "calls"
    stages = (stage,) if stage else STAGES
    records: list[dict] = []
    for name in stages:
        records.extend(read_jsonl(calls_dir / f"{name}.jsonl"))
    return records


def _empty_usage() -> dict:
    """Return a zeroed usage accumulator."""

    return {
        "calls": 0, "ok": 0, "parse_failed": 0, "errors": 0,
        "reasks": 0, "truncated": 0,
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "elapsed_seconds": 0.0, "estimated_cost": 0.0,
        "cost_reported_calls": 0,
    }


def _accumulate(bucket: dict, call: dict) -> None:
    """Fold one call record into a usage accumulator, in place."""

    bucket["calls"] += 1
    status = call.get("status", "ok")
    if status == "ok":
        bucket["ok"] += 1
    elif status == "parse_failed":
        bucket["parse_failed"] += 1
    else:
        bucket["errors"] += 1

    bucket["reasks"] += max(0, (call.get("parse_attempts") or 1) - 1)
    bucket["truncated"] += int(bool(call.get("truncated")))
    bucket["input_tokens"] += call.get("input_tokens") or 0
    bucket["output_tokens"] += call.get("output_tokens") or 0
    bucket["total_tokens"] += call.get("total_tokens") or 0
    bucket["elapsed_seconds"] += call.get("elapsed_seconds") or 0.0
    bucket["estimated_cost"] += call.get("estimated_cost") or 0.0
    bucket["cost_reported_calls"] += int(bool(call.get("cost_reported")))


def _finalize(bucket: dict) -> dict:
    """Add per-call averages to a usage accumulator."""

    calls = bucket["calls"] or 1
    bucket["avg_input_tokens"] = bucket["input_tokens"] / calls
    bucket["avg_output_tokens"] = bucket["output_tokens"] / calls
    bucket["avg_seconds"] = bucket["elapsed_seconds"] / calls
    return bucket


def aggregate_calls(calls: list[dict]) -> dict:
    """Aggregate call records by stage, agent, model and run.

    ``cost_complete`` says whether every paid call reported a cost.
    """

    totals = _empty_usage()
    by_stage: dict[str, dict] = defaultdict(_empty_usage)
    by_agent: dict[str, dict] = defaultdict(_empty_usage)
    by_model: dict[str, dict] = defaultdict(_empty_usage)
    by_run: dict[str, dict] = defaultdict(_empty_usage)
    providers: Counter = Counter()
    paid_without_cost = 0

    for call in calls:
        _accumulate(totals, call)
        _accumulate(by_stage[call.get("stage", "unknown")], call)
        _accumulate(by_agent[call.get("agent", "unknown")], call)
        _accumulate(by_model[call.get("model", "unknown")], call)
        _accumulate(by_run[call.get("run_id", "unknown")], call)

        provider = call.get("provider", "unknown")
        providers[provider] += 1
        if provider not in LOCAL_PROVIDERS and not call.get("cost_reported"):
            paid_without_cost += 1

    return {
        **_finalize(totals),
        "providers": dict(providers),
        "paid_calls_without_cost": paid_without_cost,
        "cost_complete": paid_without_cost == 0,
        "by_stage": {k: _finalize(v) for k, v in sorted(by_stage.items())},
        "by_agent": {k: _finalize(v) for k, v in sorted(by_agent.items())},
        "by_model": {k: _finalize(v) for k, v in sorted(by_model.items())},
        "by_run": {k: _finalize(v) for k, v in sorted(by_run.items())},
    }


def summarize_oe(output_dir: Path, decisions: list[dict],
                 state: dict) -> dict:
    """Summarize the Ontology Engineering stage."""

    reviews = [d for d in decisions if d.get("event") == "shard_review"]
    proposals = [d for d in decisions if d.get("event") == "shard_proposal"]
    approved_shards = {
        d["shard"] for d in reviews if d.get("approval")
    }
    reviewed_shards = {d["shard"] for d in reviews}
    forced = [
        d["shard"] for d in decisions
        if d.get("event") == "shard_forced_accept"
    ]

    issue_severity: Counter = Counter()
    for call in load_calls(output_dir, "oe"):
        if call.get("agent") != "oe_critic" or not call.get("raw_output"):
            continue
        for issue in issues_from_raw(call["raw_output"]):
            issue_severity[issue.get("severity", "advisory")] += 1

    accepted = read_json(output_dir / "ontology" / "ontology_accepted.json")
    overlay = read_json(output_dir / "ontology" / "overlay_accepted.json")

    return {
        "shards": state.get("oe", {}).get("shards", len(reviewed_shards)),
        "shards_reviewed": len(reviewed_shards),
        "shards_approved": len(approved_shards),
        "shard_approval_rate": (
            len(approved_shards) / len(reviewed_shards)
            if reviewed_shards else None
        ),
        "shards_forced_accept": len(forced),
        "builder_iterations": len(proposals),
        "critic_reviews": len(reviews),
        "issues_total": sum(d.get("issues", 0) for d in reviews),
        "issues_blocking": sum(d.get("blocking", 0) for d in reviews),
        "issues_by_severity": dict(issue_severity),
        "types_dropped": len(
            [d for d in decisions if d.get("event") == "type_dropped"]),
        "seed_restored": len(
            [d for d in decisions if d.get("event") == "seed_restored"]),
        "overlay_entity_types": len((overlay or {}).get("entity_types", [])),
        "overlay_relation_types": len(
            (overlay or {}).get("relation_types", [])),
        "accepted_entity_types": len((accepted or {}).get("entity_types", [])),
        "accepted_relation_types": len(
            (accepted or {}).get("relation_types", [])),
        "support": state.get("oe", {}).get("support", {}),
    }


def issues_from_raw(raw_output: str) -> list[dict]:
    """Pull the ``issues`` list out of a raw critic response ([] if unparseable)."""

    from src.kg.agents import AgentOutputError, extract_json_block

    try:
        data = json.loads(extract_json_block(raw_output))
    except (AgentOutputError, json.JSONDecodeError):
        return []
    issues = data.get("issues")
    return [i for i in issues if isinstance(i, dict)] if isinstance(
        issues, list) else []


def summarize_ke(output_dir: Path, decisions: list[dict],
                 state: dict) -> dict:
    """Summarize the Knowledge Extraction stage."""

    # Append-only file: the latest record per batch id counts.
    by_batch: dict = {}
    for record in read_jsonl(
            output_dir / "extractions" / "extractions.jsonl"):
        by_batch[record.get("batch_id", "")] = record
    records = [by_batch[key] for key in sorted(by_batch)]
    superseded = sum(
        1 for _ in read_jsonl(
            output_dir / "extractions" / "extractions.jsonl")) - len(records)

    violation_codes: Counter = Counter()
    modalities: Counter = Counter()
    triples = 0
    retried = 0
    improved = 0
    empty_batches = 0
    articles = set()

    truncated_batches = []
    for record in records:
        count = len(record.get("triples", []))
        triples += count
        if count == 0:
            empty_batches += 1
        if record.get("truncated"):
            truncated_batches.append(record.get("batch_id"))
        retried += int(bool(record.get("retried")))
        improved += int(bool(record.get("retry_improved")))
        articles.update(record.get("article_ids", []))
        for triple in record.get("triples", []):
            modalities[triple.get("modality") or "none"] += 1
        for violation in record.get("violations", []):
            violation_codes[violation.get("code", "unknown")] += 1

    failed = [
        d for d in decisions if d.get("event") == "batch_failed"
    ]
    return {
        "batches": len(records),
        "batches_superseded": superseded,
        "batches_failed": len(failed),
        "batches_empty": empty_batches,
        # Accepted extractions that ended at the token limit.
        "batches_truncated": len(truncated_batches),
        "truncated_batch_ids": truncated_batches,
        "batches_redone": len(
            [d for d in decisions if d.get("event") == "batches_redone"]),
        "articles_covered": len(articles),
        "triples": triples,
        "avg_triples_per_batch": triples / len(records) if records else 0.0,
        "retried_batches": retried,
        "retries_improved": improved,
        "retry_success_rate": improved / retried if retried else None,
        "triples_by_modality": dict(modalities.most_common()),
        "violations": sum(violation_codes.values()),
        "violations_by_code": dict(violation_codes.most_common()),
        "failed_batch_ids": state.get("ke", {}).get("failed_units", []),
    }


def summarize_kf(decisions: list[dict], state: dict) -> dict:
    """Summarize the Knowledge Fusion stage."""

    kf = state.get("kf", {})
    rounds = [d for d in decisions if d.get("event") == "blocking"]
    saved = next(
        (d for d in reversed(decisions) if d.get("event") == "graph_saved"),
        {},
    )
    return {
        "done": kf.get("done", False),
        "semantic_nodes": saved.get("semantic_nodes"),
        "semantic_edges": saved.get("semantic_edges"),
        "enriched_articles": saved.get("enriched_articles"),
        "auto_merges": kf.get("auto_merges", 0),
        "citation_merges": kf.get("citation_merges", 0),
        "similarity_merges": kf.get("similarity_merges", 0),
        "llm_merges_accepted": kf.get("llm_merges_accepted", 0),
        "llm_merges_rejected": kf.get("llm_merges_rejected", 0),
        "clusters_failed": len(
            [d for d in decisions if d.get("event") == "cluster_failed"]),
        "clusters_skipped_already_judged": len([
            d for d in decisions
            if d.get("event") == "cluster_skipped"
            and d.get("reason") == "already_judged"
        ]),
        "restored_merges": kf.get("restored_merges", 0),
        "fusion_state_reused": any(
            d.get("event") == "fusion_state_restored" for d in decisions),
        "rounds": len(rounds),
        "blocking_method": rounds[-1].get("method") if rounds else None,
        "schema_extensions": kf.get("schema_extensions", []),
    }


def _config_provider(config: dict) -> str | None:
    """Read the KE builder's provider out of a saved ``config.json``."""

    agents = config.get("agents") or {}
    builder = agents.get("ke_builder") or {}
    return builder.get("provider")


def build_metrics(output_dir: Path) -> dict:
    """Assemble the full metrics document for one build folder."""

    output_dir = Path(output_dir)
    config = read_json(output_dir / "config.json", {}) or {}
    manifest = read_json(output_dir / "manifest.json", {}) or {}
    state = read_json(output_dir / "state.json", {}) or {}
    decisions = read_jsonl(output_dir / "decisions.jsonl")
    graph_meta = read_json(output_dir / "graph" / "meta.json", {}) or {}
    calls = load_calls(output_dir)

    runs = manifest.get("runs", [])
    stage_seconds: dict[str, float] = defaultdict(float)
    for run in runs:
        for stage, timing in (run.get("stages") or {}).items():
            stage_seconds[stage] += timing.get("seconds") or 0.0

    usage = aggregate_calls(calls)

    return {
        "kg_version": config.get("kg_version") or manifest.get("kg_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": state.get("stage"),
        "provider": _config_provider(config) or (
            runs[-1].get("provider") if runs else None),
        "runs": len(runs),
        "run_ids": [r.get("run_id") for r in runs],
        "last_run": runs[-1].get("run_id") if runs else None,
        "stages_completed": sorted(stage_seconds),
        "stage_seconds": dict(stage_seconds),
        "wall_clock_seconds": sum(
            r.get("duration_seconds") or 0.0 for r in runs),
        "usage": usage,
        # Running totals from state.json, which also count unlogged calls.
        "state_usage": state.get("usage", {}),
        "oe": summarize_oe(output_dir, decisions, state),
        "ke": summarize_ke(output_dir, decisions, state),
        "kf": summarize_kf(decisions, state),
        "graph": {
            "nodes": graph_meta.get("nodes", 0),
            "edges": graph_meta.get("edges", 0),
            "nodes_by_type": graph_meta.get("nodes_by_type", {}),
            "edges_by_relation": graph_meta.get("edges_by_relation", {}),
            "ontology_version": graph_meta.get("ontology_version"),
        },
        "decisions": dict(
            Counter(d.get("event") for d in decisions).most_common()),
    }


def log_build_summary(metrics: dict, output_dir: Path) -> None:
    """Log the end-of-build summary block."""

    from loguru import logger

    usage = metrics.get("usage", {})
    cost_note = "" if usage.get("cost_complete", True) else (
        f"  ({usage.get('paid_calls_without_cost')} paid calls "
        f"reported no cost)"
    )

    lines = [
        "=" * 56,
        "KG BUILD SUMMARY",
        "=" * 56,
        "",
        f"KG Version          : {metrics.get('kg_version')}",
        f"Provider            : {metrics.get('provider')}",
        f"Pipeline Stage      : {metrics.get('stage')}",
        f"Runs In Folder      : {metrics.get('runs')} "
        f"(last: {metrics.get('last_run')})",
        "",
        f"Agent Calls         : {usage.get('calls', 0)}",
        f"  ok / parse_failed : {usage.get('ok', 0)} / "
        f"{usage.get('parse_failed', 0)}",
        f"  errors / re-asks  : {usage.get('errors', 0)} / "
        f"{usage.get('reasks', 0)}",
        f"  truncated         : {usage.get('truncated', 0)}",
        "",
        f"Input Tokens        : {usage.get('input_tokens', 0)}",
        f"Output Tokens       : {usage.get('output_tokens', 0)}",
        f"Total Tokens        : {usage.get('total_tokens', 0)}",
        f"Estimated Cost      : ${usage.get('estimated_cost', 0.0):.4f}"
        f"{cost_note}",
        f"Request Time        : {usage.get('elapsed_seconds', 0.0):.1f}s",
        f"Wall Clock          : {metrics.get('wall_clock_seconds', 0.0):.1f}s",
        "",
    ]

    for stage in STAGES:
        bucket = usage.get("by_stage", {}).get(stage)
        if not bucket:
            continue
        lines.append(
            f"{stage.upper():4s} calls={bucket['calls']:5d} "
            f"tokens={bucket['total_tokens']:9d} "
            f"cost=${bucket['estimated_cost']:.4f} "
            f"time={bucket['elapsed_seconds']:.0f}s"
        )

    oe, ke, kf = metrics.get("oe", {}), metrics.get("ke", {}), metrics.get(
        "kf", {})
    rate = oe.get("shard_approval_rate")
    lines.extend([
        "",
        f"OE  shards approved : {oe.get('shards_approved')}/"
        f"{oe.get('shards_reviewed')}"
        + (f" ({rate:.0%})" if rate is not None else ""),
        f"OE  accepted types  : {oe.get('accepted_entity_types')} entities, "
        f"{oe.get('accepted_relation_types')} relations",
        f"KE  batches         : {ke.get('batches')} "
        f"({ke.get('batches_failed')} failed, {ke.get('retried_batches')} "
        f"retried)",
        f"KE  triples         : {ke.get('triples')} "
        f"({ke.get('violations')} violations)",
        f"KF  merges          : auto={kf.get('auto_merges')} "
        f"citation={kf.get('citation_merges')} "
        f"similarity={kf.get('similarity_merges')} "
        f"llm={kf.get('llm_merges_accepted')}",
        f"Graph               : {metrics.get('graph', {}).get('nodes')} nodes, "
        f"{metrics.get('graph', {}).get('edges')} edges",
        "",
        f"Build Folder        : {output_dir}",
        "",
        "=" * 56,
    ])

    logger.info("\n{}", "\n".join(lines))


def write_metrics(output_dir: Path) -> Path:
    """Recompute and write ``metrics.json`` for a build folder."""

    output_dir = Path(output_dir)
    path = output_dir / "metrics.json"
    path.write_text(
        json.dumps(build_metrics(output_dir), ensure_ascii=False, indent=2,
                   default=str),
        encoding="utf-8",
    )
    return path
