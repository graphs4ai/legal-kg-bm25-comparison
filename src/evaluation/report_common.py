"""Shared helpers of the reports: run discovery, statistics, grounding, style."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.run_eval import SPLIT_PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
MODEL_NAMES = {
    "ibm-granite/granite-4.2-8b": "Granite-4.2-8B",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral-Small-3.2",
    "tencent/Hy3": "Hy3",
    "sabiazinho-4": "Sabiazinho-4",
}

CONDITIONS = ("vanilla", "rag", "graphrag")
CONDITION_LABELS = {"vanilla": "No retrieval", "rag": "RAG", "graphrag": "GraphRAG"}

# Aqua is low-contrast on the surface, so every chart carries labels or a table.
CONDITION_COLORS = {"vanilla": "#2a78d6", "rag": "#eb6834", "graphrag": "#1baf7a"}
POS_COLOR = "#2a78d6"
NEG_COLOR = "#e34948"
MID_COLOR = "#f0efec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e8e7e2"

_ART_NUMBER = r"\d+(?:\.\d+)*(?:\s*[-–]\s*[A-Za-z]\b)?"
_JUSTIFICATION_ART_RE = re.compile(
    rf"art(?:igo)?s?\.?\s*({_ART_NUMBER}(?:\s*(?:,|e)\s*{_ART_NUMBER})*)",
    re.IGNORECASE,
)
_ART_NUMBER_RE = re.compile(_ART_NUMBER)


# --- Runs ---
@lru_cache(maxsize=None)
def split_ids() -> dict[str, frozenset[str]]:
    """Question ids of every split, keyed by split name."""

    ids = {}
    for split, path in SPLIT_PATHS.items():
        with path.open(encoding="utf-8") as f:
            ids[split] = frozenset(json.loads(line)["id"] for line in f)
    return ids


def _split_of(results_path: Path) -> str | None:
    """Split of a run, from its first question id (the splits are disjoint)."""

    with results_path.open(encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        return None
    qid = json.loads(first)["id"]
    for split, split_set in split_ids().items():
        if qid in split_set:
            return split
    return None


def load_index(conditions=CONDITIONS) -> list[dict]:
    """Every finished run (one with a ``metrics.json`` entry), oldest first.

    Each entry gains ``condition``, ``run_dir``, ``run_number`` and ``split``.
    """

    runs = []
    for condition in conditions:
        path = RUNS_DIR / condition / "metrics.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            run_dir = RUNS_DIR / condition / entry["run_id"]
            results = run_dir / "results.jsonl"
            if not results.exists():
                continue
            runs.append({
                **entry,
                "condition": condition,
                "run_dir": run_dir,
                "run_number": int(entry["run_id"].split("_")[1]),
                "split": _split_of(results),
            })
    return sorted(runs, key=lambda r: r["run_number"])


_KEEP_FIELDS = (
    "id", "correct", "status", "predicted_answer", "gold_answer",
    "input_tokens", "output_tokens", "finish_reason", "metadata",
    "retrieved_chunk_ids", "justification",
)


def load_results(run: dict, prompts: bool = False) -> dict[str, dict]:
    """A run's rows keyed by question id, trimmed to the report fields (plus prompts if asked)."""

    keep = _KEEP_FIELDS + (("prompt",) if prompts else ())
    rows = {}
    with (run["run_dir"] / "results.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = {k: row.get(k) for k in keep}
    return rows


def graphrag_tag(cell: dict | None) -> str | None:
    """Method tag a GraphRAG run of ``cell`` records, or None when ``cell`` is None."""

    if cell is None:
        return None
    from src.evaluation import run_eval

    return run_eval.graphrag_method(cell)


def current_retrieval(run: dict, split: str) -> bool:
    """Whether a retrieval run used the configured context cap and corpus."""

    from src.evaluation import run_eval

    return (run.get("retrieval_max_context_tokens") == run_eval.MAX_CONTEXT_TOKENS
            and run.get("laws_version") == run_eval.corpus_for(split)[2])


def load_runs(runs: dict[str, dict[str, dict]]) -> dict:
    """Per-question rows of every selected run (model -> condition -> run)."""

    return {key: {cond: load_results(run, prompts=cond != "vanilla")
                  for cond, run in by_cond.items()}
            for key, by_cond in runs.items()}


def shared_articles(rows_a: dict, rows_b: dict) -> dict[str, int]:
    """Articles two runs served in common, per question."""

    return {q: len(set(r.get("retrieved_chunk_ids") or [])
                   & set(rows_b[q].get("retrieved_chunk_ids") or []))
            for q, r in rows_a.items() if q in rows_b}


def run_label(key: str, analysis: dict) -> str:
    """Display name of a model key, read from its no-retrieval run."""

    return model_label(analysis["runs"][key]["vanilla"]["model"])


def is_correct(row: dict) -> bool:
    return row.get("correct") in (True, "True", "true", 1)


def model_label(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def subject(row: dict) -> str | None:
    value = (row.get("metadata") or {}).get("question_type")
    return value if isinstance(value, str) else None


# --- Statistics ---
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""

    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(b: int, c: int) -> float:
    """Continuity-corrected McNemar p-value on the discordant counts."""

    if b + c == 0:
        return float("nan")
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def paired(base: dict[str, bool], other: dict[str, bool]) -> dict:
    """Paired comparison of two runs on their shared questions.

    Args:
        base: Correctness per question id of the reference run.
        other: Correctness per question id of the compared run.

    Returns:
        dict: ``n``, ``diff`` (other − base, proportion), ``lo``/``hi``
            (95% paired interval), ``p`` (McNemar), ``lost`` (base right,
            other wrong) and ``won`` (the reverse).
    """

    ids = base.keys() & other.keys()
    n = len(ids)
    lost = sum(base[i] and not other[i] for i in ids)
    won = sum(other[i] and not base[i] for i in ids)
    diff = (won - lost) / n if n else float("nan")
    se = np.sqrt(max(won + lost - (won - lost) ** 2 / n, 0)) / n if n else float("nan")
    return {"n": n, "diff": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se,
            "p": mcnemar(lost, won), "lost": lost, "won": won}


# --- Grounding ---
def _article_numbers(text: str) -> set[str]:
    """Article numbers named in a piece of text, normalized for comparison.

    Enumerations are expanded, so "arts. 12 e 13" yields both numbers.
    """

    numbers = set()
    for run in _JUSTIFICATION_ART_RE.findall(str(text or "")):
        for raw in _ART_NUMBER_RE.findall(run):
            canonical = raw.replace(".", "").replace(" ", "").upper()
            canonical = canonical.replace("–", "-")
            if canonical:
                numbers.add(canonical)
    return numbers


def _context_block(prompt: str) -> str:
    """The injected legal context of a retrieval prompt, empty when absent."""

    match = re.search(
        r"<contexto_legal>(.*?)</contexto_legal>", str(prompt or ""), re.S)
    return match.group(1) if match else ""


def _grounding(record: dict) -> bool | None:
    """Whether an answer cites an article its context did not contain.

    Returns None when the answer cites no article at all.
    """

    cited = _article_numbers(record.get("justification"))
    if not cited:
        return None
    return bool(cited - _article_numbers(_context_block(record.get("prompt"))))


# --- Rendering ---
def pct(value: float, digits: int = 1) -> str:
    return "—" if value is None or np.isnan(value) else f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    return "—" if value is None or np.isnan(value) else f"{value * 100:+.{digits}f}"


def p_value(p: float) -> str:
    if p is None or np.isnan(p):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def interval(lo: float, hi: float) -> str:
    return f"[{pp(lo)}, {pp(hi)}]"


def table(header: list[str], rows: list[list], align: str = "") -> str:
    """Markdown table; ``align`` holds ``l``/``r`` per column."""

    align = align or "l" * len(header)
    seps = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(seps) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#d8d7d2",
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 9.5,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save_figure(fig, figures_dir: Path, name: str) -> str:
    """Write a figure; return its path relative to ``REPORTS_DIR``."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(REPORTS_DIR).as_posix()
