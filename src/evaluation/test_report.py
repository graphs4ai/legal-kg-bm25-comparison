"""TEST report: every QA model under no retrieval, RAG and GraphRAG.

    python -m src.evaluation.test_report
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from src.evaluation import run_eval
from src.evaluation.report_common import (
    CONDITION_COLORS, CONDITION_LABELS, CONDITIONS, INK, INK_MUTED,
    MID_COLOR, NEG_COLOR, POS_COLOR, REPORTS_DIR,
    _grounding, current_retrieval, graphrag_tag, interval, mcnemar, is_correct, load_index,
    load_runs, run_label,
    paired, pct, pp, p_value, save_figure, setup_style, split_ids, subject,
    table, wilson,
)
from src.evaluation.run_eval import SPLIT_PATHS
from src.utils.currency import format_cost, record_cost_usd

REPORT_NAME = "test"
FIGURES_DIR = REPORTS_DIR / "figures" / REPORT_NAME

# The three paired comparisons, as (reference, compared).
COMPARISONS = (("vanilla", "rag"), ("vanilla", "graphrag"), ("rag", "graphrag"))
DIVERGING = LinearSegmentedColormap.from_list("div", [NEG_COLOR, MID_COLOR, POS_COLOR])

# Question-level bootstrap for the mean difference over models.
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0


# --- Run selection ---
def select_runs(index: list[dict]) -> dict[str, dict[str, dict]]:
    """Latest TEST run per model and condition (model -> condition -> run)."""

    n_test = len(split_ids()["test"])

    def matches(run, condition):
        if run["condition"] != condition or run["total"] != n_test:
            return False
        if condition == "vanilla":
            return True
        current = current_retrieval(run, "test")
        if condition == "rag":
            return (current and run["retrieval_method"] == run_eval.RAG_TEST_METHOD
                    and run["retrieval_top_k"] == run_eval.RAG_TEST_TOP_K
                    and (run.get("retrieval_query_mode") or "question_choices")
                    == "question_choices")
        return current and run.get("retrieval_method") == graph_tag

    graph_tag = graphrag_tag(run_eval.GRAPHRAG_TEST_CELL)
    chosen: dict[str, dict[str, dict]] = defaultdict(dict)
    test_runs = [r for r in index if r["split"] == "test"]
    for key, settings in run_eval.QA_MODELS.items():
        for condition in CONDITIONS:
            for run in test_runs:
                if run["model"] == settings["model"] and matches(run, condition):
                    chosen[key][condition] = run
    return dict(chosen)


# --- Analysis ---
def analyse(runs: dict, rows: dict) -> dict:
    models = [k for k in run_eval.QA_MODELS if len(runs.get(k, {})) == len(CONDITIONS)]
    correct = {k: {c: {q: is_correct(r) for q, r in rows[k][c].items()}
                   for c in CONDITIONS} for k in models}
    models.sort(key=lambda k: sum(correct[k]["vanilla"].values()) / len(correct[k]["vanilla"]))

    acc = {}
    for k in models:
        for c in CONDITIONS:
            n, hits = len(correct[k][c]), sum(correct[k][c].values())
            acc[k, c] = {"n": n, "acc": hits / n, "ci": wilson(hits, n)}
    comp = {(k, a, b): paired(correct[k][a], correct[k][b])
            for k in models for a, b in COMPARISONS}

    # Mean difference over models; the bootstrap resamples questions across all models.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shared = sorted(set.intersection(*(set(correct[k]["vanilla"]) for k in models)))
    pooled = {}
    for a_, b_ in COMPARISONS:
        per_q = np.array([np.mean([correct[k][b_][q] - correct[k][a_][q] for k in models])
                          for q in shared])
        draws = rng.integers(0, len(per_q), size=(BOOTSTRAP_DRAWS, len(per_q)))
        means = per_q[draws].mean(axis=1)
        pooled[a_, b_] = {"diff": float(per_q.mean()),
                          "lo": float(np.percentile(means, 2.5)),
                          "hi": float(np.percentile(means, 97.5)),
                          "n": len(shared)}

    # Rescue: vanilla wrong -> right; harm: vanilla right -> wrong.
    churn = {}
    for k in models:
        van = correct[k]["vanilla"]
        wrong = [q for q in van if not van[q]]
        right = [q for q in van if van[q]]
        for c in ("rag", "graphrag"):
            churn[k, c] = {
                "rescue": sum(correct[k][c][q] for q in wrong) / len(wrong),
                "harm": sum(not correct[k][c][q] for q in right) / len(right),
                "n_wrong": len(wrong), "n_right": len(right),
            }

    # Peer difficulty: how many of the other models answer without retrieval.
    bands = defaultdict(lambda: defaultdict(list))
    band_sizes = defaultdict(list)
    for k in models:
        peers = [m for m in models if m != k]
        per_band = defaultdict(list)
        for q in correct[k]["vanilla"]:
            per_band[sum(correct[m]["vanilla"][q] for m in peers)].append(q)
        for band, qs in per_band.items():
            band_sizes[band].append(len(qs))
            for c in ("rag", "graphrag"):
                gain = np.mean([correct[k][c][q] for q in qs]) - np.mean(
                    [correct[k]["vanilla"][q] for q in qs])
                bands[band][c].append(gain)

    subjects = defaultdict(list)
    for q, row in rows[models[0]]["vanilla"].items():
        subjects[subject(row) or "—"].append(q)
    subj = {}
    subj_comp = {}
    for s, qs in subjects.items():
        for k in models:
            for c in CONDITIONS:
                subj[s, k, c] = np.mean([correct[k][c][q] for q in qs])
            for a_, b_ in COMPARISONS:
                subj_comp[s, k, a_, b_] = paired(
                    {q: correct[k][a_][q] for q in qs}, {q: correct[k][b_][q] for q in qs})
    # Holm adjustment within each comparison, over every (subject, model) test.
    for a_, b_ in COMPARISONS:
        keys = [key for key in subj_comp if key[2:] == (a_, b_)]
        ranked = sorted(keys, key=lambda key: np.nan_to_num(subj_comp[key]["p"], nan=1.0))
        running = 0.0
        for i, key in enumerate(ranked):
            p = np.nan_to_num(subj_comp[key]["p"], nan=1.0)
            running = max(running, min(1.0, (len(ranked) - i) * p))
            subj_comp[key]["p_holm"] = running

    # Evidence overlap between the two retrievers, per question.
    overlap = {}
    for k in models:
        same, jac, counts = [], [], []
        for q, r_row in rows[k]["rag"].items():
            a = set(r_row.get("retrieved_chunk_ids") or [])
            b = set(rows[k]["graphrag"][q].get("retrieved_chunk_ids") or [])
            union = a | b
            same.append(a == b)
            jac.append(len(a & b) / len(union) if union else 1.0)
            counts.append(len(a & b))
        # Share of each RAG rank (BM25 order) that GraphRAG also serves.
        kept_by_rank = defaultdict(list)
        for q, r_row in rows[k]["rag"].items():
            b = set(rows[k]["graphrag"][q].get("retrieved_chunk_ids") or [])
            for rank, art in enumerate(r_row.get("retrieved_chunk_ids") or [], start=1):
                kept_by_rank[rank].append(art in b)
        overlap[k] = {"same": same, "jaccard": float(np.mean(jac)),
                      "shared": float(np.mean(counts)),
                      "shared_per_q": dict(zip(rows[k]["rag"], counts)),
                      "kept_by_rank": {r: float(np.mean(v)) for r, v in kept_by_rank.items()}}

    # Grounding: share of citing answers that cite an article not in context.
    grounding = {}
    for k in models:
        for c in ("rag", "graphrag"):
            flags = [_grounding(r) for r in rows[k][c].values()]
            cited = [f for f in flags if f is not None]
            grounding[k, c] = {"cited": len(cited) / len(flags),
                               "outside": sum(cited) / len(cited) if cited else float("nan")}

    reliability = {}
    for k in models:
        for c in CONDITIONS:
            run, rs = runs[k][c], rows[k][c].values()
            cost = record_cost_usd(run)
            reliability[k, c] = {
                "run": run["run_id"],
                "no_answer": sum(r.get("status") != "answered" for r in rs),
                "truncated": sum(r.get("finish_reason") == "length" for r in rs),
                "tokens_in": run["avg_input_tokens"],
                "tokens_out": run["avg_output_tokens"],
                "cost": cost,
                "cost_1k": cost / run["total"] * 1000 if cost is not None else None,
            }

    return {"models": models, "correct": correct, "acc": acc, "comp": comp, "pooled": pooled,
            "churn": churn, "bands": bands, "band_sizes": band_sizes,
            "subjects": subjects, "subj": subj, "subj_comp": subj_comp, "overlap": overlap,
            "grounding": grounding, "reliability": reliability, "runs": runs}


def future_text_effect(rag_rows: dict) -> dict:
    """How flat RAG's top k changes when BM25 indexes no text dated after the cutoff.

    Rebuilds the index per cutoff and compares with the articles a RAG TEST run served.
    """

    from src.data.laws_versioning import wording_at
    from src.data.oab_splits import cutoff_for
    from src.rag.bm25 import FastBM25
    from src.rag.text_utils import tokenize

    retriever = run_eval._law_retriever("test", run_eval.RAG_TEST_METHOD)
    with SPLIT_PATHS["test"].open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    by_cutoff = defaultdict(list)
    for item in items:
        if item["id"] in rag_rows:
            by_cutoff[cutoff_for(item.get("metadata", {}))].append(item)

    def day(value):
        return None if value is None else date.fromisoformat(str(value)[:10])

    k = run_eval.RAG_TEST_TOP_K
    shared, dropped = [], np.zeros(k)
    for cutoff, group in by_cutoff.items():
        keys, texts = [], []
        for key, versions in retriever._articles.items():
            for version in versions:
                start = day(version.get("valid_from_date"))
                if start is not None and start > cutoff:
                    continue
                provisions = version.get("provisions") or []
                kept = [p for p in provisions
                        if day(p.get("valid_from")) is None or day(p.get("valid_from")) <= cutoff]
                text = (version["display_text"] if len(kept) == len(provisions)
                        else " ".join([version.get("header") or "", version.get("caput_line") or ""]
                                      + [p.get("text") or "" for p in kept]))
                keys.append(key)
                texts.append(text)
        bm25 = FastBM25([tokenize(t) for t in texts])
        for item in group:
            scores = np.asarray(bm25.get_scores(tokenize(retriever.build_query(item))))
            top, seen = [], set()
            for i in np.argsort(-scores, kind="stable"):
                if keys[i] in seen:
                    continue
                seen.add(keys[i])
                if wording_at(retriever._articles[keys[i]], cutoff.year, cutoff) is not None:
                    top.append(keys[i])
                if len(top) == k:
                    break
            served = rag_rows[item["id"]].get("retrieved_chunk_ids") or []
            shared.append(len(set(top) & set(served)))
            for rank, article in enumerate(served[:k]):
                if article not in top:
                    dropped[rank] += 1
    n = len(shared)
    return {"shared": float(np.mean(shared)), "k": k, "n": n,
            "top3_kept": float(1 - dropped[:3].sum() / (3 * n)),
            "dropped_by_rank": (dropped / n).tolist()}


def current_wording_gap(rag_rows: dict) -> dict:
    """How often a served article's wording at the cutoff differs from its latest one."""

    from src.data.laws_versioning import wording_at
    from src.data.oab_splits import cutoff_for

    retriever = run_eval._law_retriever("test", run_eval.RAG_TEST_METHOD)
    with SPLIT_PATHS["test"].open(encoding="utf-8") as f:
        cutoffs = {item["id"]: cutoff_for(item.get("metadata", {}))
                   for item in map(json.loads, f)}

    def day(value):
        return None if value is None else date.fromisoformat(str(value)[:10])

    starts = [day(v.get("valid_from_date")) for vs in retriever._articles.values() for v in vs]
    starts += [day(p.get("valid_from")) for vs in retriever._articles.values()
               for v in vs for p in v.get("provisions") or []]
    as_of = max(d for d in starts if d is not None)

    articles, questions, top1 = [], [], []
    for qid, row in rag_rows.items():
        cutoff = cutoffs[qid]
        differs = []
        for key in row.get("retrieved_chunk_ids") or []:
            versions = retriever._articles.get(key)
            then = wording_at(versions, cutoff.year, cutoff)
            now = wording_at(versions, as_of.year, as_of)
            differs.append(then is not None and now is not None
                           and then["display_text"] != now["display_text"])
        articles += differs
        questions.append(any(differs))
        top1.append(bool(differs) and differs[0])
    return {"articles": float(np.mean(articles)), "questions": float(np.mean(questions)),
            "top1": float(np.mean(top1)), "n": len(questions), "as_of": as_of.isoformat()}


# --- Figures ---
def fig_accuracy(a: dict) -> str:
    models = a["models"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.26
    xs = np.arange(len(models))
    for i, c in enumerate(CONDITIONS):
        for x, k in zip(xs, models):
            cell = a["acc"][k, c]
            pos = x + (i - 1) * (width + 0.02)
            ax.bar(pos, cell["acc"] * 100, width, color=CONDITION_COLORS[c],
                   label=CONDITION_LABELS[c] if x == 0 else None)
            ax.plot([pos, pos], [cell["ci"][0] * 100, cell["ci"][1] * 100],
                    color=INK, linewidth=1)
            ax.text(pos, cell["ci"][1] * 100 + 0.8, f"{cell['acc'] * 100:.0f}",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [run_label(k, a) for k in models])
    ax.set_ylabel("TEST accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Accuracy by model and condition (95% CI)", loc="left")
    return save_figure(fig, FIGURES_DIR, "accuracy")


def fig_forest(a: dict) -> str:
    models = a["models"]
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(10, 0.55 * len(models) + 1.4),
                             sharey=True)
    titles = {("vanilla", "rag"): "RAG − No retrieval",
              ("vanilla", "graphrag"): "GraphRAG − No retrieval",
              ("rag", "graphrag"): "GraphRAG − RAG"}
    ys = np.arange(len(models))[::-1]
    for ax, (ref, cmp_) in zip(axes, COMPARISONS):
        for y, k in zip(ys, models):
            d = a["comp"][k, ref, cmp_]
            ax.plot([d["lo"] * 100, d["hi"] * 100], [y, y], color=INK_MUTED, linewidth=1.5)
            ax.plot(d["diff"] * 100, y, "o", markersize=8, color=CONDITION_COLORS[cmp_],
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.text(d["hi"] * 100, y + 0.18, f"{d['diff'] * 100:+.1f}",
                    ha="left", va="bottom", fontsize=8, color=INK)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_ylim(-0.6, len(models) - 0.3)
        ax.margins(x=0.12)
        ax.set_title(titles[ref, cmp_], loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("points (95% paired CI)")
    axes[0].set_yticks(ys, [run_label(k, a) for k in models])
    fig.tight_layout()
    return save_figure(fig, FIGURES_DIR, "paired_gains")


def fig_gain_vs_baseline(a: dict) -> str:
    models = a["models"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for c in ("rag", "graphrag"):
        xs = [a["acc"][k, "vanilla"]["acc"] * 100 for k in models]
        ys = [a["comp"][k, "vanilla", c]["diff"] * 100 for k in models]
        ax.plot(xs, ys, marker="o", markersize=8, color=CONDITION_COLORS[c],
                markeredgecolor="white", markeredgewidth=1.5, label=CONDITION_LABELS[c])
    for k in models:
        x = a["acc"][k, "vanilla"]["acc"] * 100
        y = max(a["comp"][k, "vanilla", c]["diff"] for c in ("rag", "graphrag")) * 100
        ax.annotate(run_label(k, a), (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, linewidth=1)
    ax.margins(x=0.1, y=0.15)
    ax.set_xlabel("No-retrieval accuracy (%)")
    ax.set_ylabel("Gain over no retrieval (points)")
    ax.legend(loc="upper right")
    ax.set_title("Gain against no-retrieval accuracy", loc="left")
    return save_figure(fig, FIGURES_DIR, "gain_vs_baseline")


def fig_bands(a: dict) -> str:
    bands = sorted(a["bands"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for c in ("rag", "graphrag"):
        ys = [np.mean(a["bands"][b][c]) * 100 for b in bands]
        ax.plot(bands, ys, marker="o", markersize=8, color=CONDITION_COLORS[c],
                markeredgecolor="white", markeredgewidth=1.5, label=CONDITION_LABELS[c])
    ax.axhline(0, color=INK, linewidth=1)
    n_peers = len(a["models"]) - 1
    ax.set_xticks(bands, [f"{b} of {n_peers}" for b in bands])
    ax.set_xlabel("Peer models answering correctly without retrieval")
    ax.set_ylabel("Mean gain over no retrieval (points)")
    ax.legend(loc="upper right")
    ax.set_title("Gain by question difficulty (mean over models)", loc="left")
    return save_figure(fig, FIGURES_DIR, "difficulty")


def fig_subjects(a: dict, order: list[str]) -> str:
    models = a["models"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.34 * len(order) + 1.6), sharey=True)
    for ax, c in zip(axes, ("rag", "graphrag")):
        grid = np.array([[(a["subj"][s, k, c] - a["subj"][s, k, "vanilla"]) * 100
                          for k in models] for s in order])
        bound = max(abs(grid).max(), 1)
        ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(0, -bound, bound), aspect="auto")
        for i in range(len(order)):
            for j in range(len(models)):
                ax.text(j, i, f"{grid[i, j]:+.0f}", ha="center", va="center",
                        fontsize=7.5, color=INK)
        ax.set_xticks(range(len(models)), [run_label(k, a) for k in models],
                      rotation=30, ha="right")
        ax.set_title(f"{CONDITION_LABELS[c]} − No retrieval (points)", loc="left", fontsize=10)
        ax.grid(False)
    axes[0].set_yticks(range(len(order)),
                       [f"{s} ({len(a['subjects'][s])})" for s in order])
    fig.tight_layout()
    return save_figure(fig, FIGURES_DIR, "subjects")


def fig_cost(a: dict) -> str:
    models = a["models"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for k in models:
        pts = [(a["reliability"][k, c]["cost_1k"], a["acc"][k, c]["acc"] * 100)
               for c in CONDITIONS]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#c9c8c3", linewidth=1.2,
                zorder=1)
        for c, (x, y) in zip(CONDITIONS, pts):
            ax.scatter(x, y, s=55, color=CONDITION_COLORS[c], edgecolor="white",
                       linewidth=1.5, zorder=2,
                       label=CONDITION_LABELS[c] if k == models[0] else None)
        ax.annotate(run_label(k, a), pts[-1], textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.margins(x=0.08)
    ax.set_xlabel("Cost per 1,000 questions (USD, log scale)")
    ax.set_ylabel("TEST accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_title("Accuracy against cost", loc="left")
    return save_figure(fig, FIGURES_DIR, "cost")


# --- Report ---
def _headline(a: dict) -> list[str]:
    models = a["models"]
    lines = []
    for c in ("rag", "graphrag"):
        gains = [a["comp"][k, "vanilla", c] for k in models]
        sig = sum(g["p"] < 0.05 and g["diff"] > 0 for g in gains)
        lines.append(
            f"- **{CONDITION_LABELS[c]} vs no retrieval:** {pp(min(g['diff'] for g in gains))} "
            f"to {pp(max(g['diff'] for g in gains))} points; significant gain for "
            f"{sig} of {len(models)} models.")
    gr = [a["comp"][k, "rag", "graphrag"] for k in models]
    sig = sum(g["p"] < 0.05 for g in gr)
    lines.append(
        f"- **GraphRAG vs RAG:** {pp(min(g['diff'] for g in gr))} to "
        f"{pp(max(g['diff'] for g in gr))} points; significant for {sig} of {len(models)} "
        f"models; {sum(g['diff'] > 0 for g in gr)} above RAG, "
        f"{sum(g['diff'] < 0 for g in gr)} below.")
    for c in ("rag", "graphrag"):
        gains = [a["comp"][k, "vanilla", c]["diff"] for k in models]
        monotone = all(x >= y for x, y in zip(gains, gains[1:]))
        lines.append(
            f"- **{CONDITION_LABELS[c]} gain vs baseline:** "
            + (" > ".join(f"{pp(g)}" for g in gains))
            + f" from weakest to strongest model"
            + (" — shrinks at every step." if monotone else " — not monotone."))
    bands = sorted(a["bands"])
    lo, hi = bands[0], bands[-1]
    lines.append(
        f"- **Difficulty:** mean GraphRAG gain {pp(np.mean(a['bands'][lo]['graphrag']))} "
        f"on questions no peer answers vs {pp(np.mean(a['bands'][hi]['graphrag']))} on those "
        f"every peer answers (RAG {pp(np.mean(a['bands'][lo]['rag']))} vs "
        f"{pp(np.mean(a['bands'][hi]['rag']))}).")
    shared = np.mean([a["overlap"][k]["shared"] for k in models])
    lines.append(
        f"- **Evidence:** GraphRAG and RAG share {shared:.1f} of "
        f"{run_eval.RAG_TEST_TOP_K} articles per question on average "
        f"(mean Jaccard {np.mean([a['overlap'][k]['jaccard'] for k in models]):.2f}).")
    return lines


def build() -> str:
    index = load_index()
    runs = select_runs(index)
    rows = load_runs(runs)
    a = analyse(runs, rows)
    models = a["models"]
    setup_style()
    n = a["acc"][models[0], "vanilla"]["n"]
    graph_tag = a["runs"][models[0]]["graphrag"]["retrieval_method"]

    lines = [
        "# TEST — No retrieval vs RAG vs GraphRAG",
        "",
        f"- Split: TEST, {n} questions; reported once, never tuned on.",
        f"- Models ({len(models)}, weakest first): "
        + ", ".join(run_label(k, a) for k in models) + ".",
        f"- RAG: {run_eval.RAG_TEST_METHOD} k={run_eval.RAG_TEST_TOP_K}. "
        f"GraphRAG: `{graph_tag}`. Both capped at {run_eval.MAX_CONTEXT_TOKENS:,} context tokens; "
        "configs chosen on DEV (see `dev_sweeps.md`); layer ablation of the GraphRAG "
        "config in `layers.md`.",
        "- Accuracy counts every question; no-answers count as wrong. "
        "Paired tests: McNemar with continuity correction; 95% paired intervals.",
        "",
        "## Headline",
        "",
        *_headline(a),
        "",
        "## 1. Accuracy",
        "",
        f"![Accuracy]({fig_accuracy(a)})",
        "",
        table(["Model"] + [CONDITION_LABELS[c] for c in CONDITIONS]
              + ["95% CI (no retr.)", "95% CI (RAG)", "95% CI (GraphRAG)"],
              [[run_label(k, a)] + [pct(a["acc"][k, c]["acc"]) for c in CONDITIONS]
               + [f"{pct(a['acc'][k, c]['ci'][0])}–{pct(a['acc'][k, c]['ci'][1])}"
                  for c in CONDITIONS] for k in models], "lrrrrrr"),
        "",
        "## 2. Paired gains",
        "",
        f"![Paired gains]({fig_forest(a)})",
        "",
    ]
    comp_rows = []
    for k in models:
        for ref, cmp_ in COMPARISONS:
            d = a["comp"][k, ref, cmp_]
            comp_rows.append([
                run_label(k, a), f"{CONDITION_LABELS[cmp_]} − {CONDITION_LABELS[ref]}",
                pp(d["diff"]), interval(d["lo"], d["hi"]), p_value(d["p"]),
                f"{d['won']} / {d['lost']}"])
    lines += [
        table(["Model", "Comparison", "Δ (points)", "95% CI", "p", "Won / lost"],
              comp_rows, "llrrrr"),
        "",
        "### Mean over models",
        "",
        f"- Mean difference over the {len(models)} models, with a 95% bootstrap interval "
        f"({BOOTSTRAP_DRAWS:,} resamples of the {next(iter(a['pooled'].values()))['n']} "
        "questions, each keeping every model's answers).",
        "",
        table(["Comparison", "Mean Δ (points)", "95% CI"],
              [[f"{CONDITION_LABELS[b_]} − {CONDITION_LABELS[a_]}", pp(v["diff"]),
                interval(v["lo"], v["hi"])] for (a_, b_), v in a["pooled"].items()],
              "lrr"),
        "",
        "## 3. Gain against baseline strength",
        "",
        f"![Gain vs baseline]({fig_gain_vs_baseline(a)})",
        "",
        "## 4. Rescue and harm",
        "",
        "- Rescue: share of questions wrong without retrieval that retrieval gets right. "
        "Harm: share right without retrieval that retrieval gets wrong.",
        "",
        table(["Model", "Wrong / right w/o retrieval", "RAG rescue", "RAG harm",
               "GraphRAG rescue", "GraphRAG harm"],
              [[run_label(k, a),
                f"{a['churn'][k, 'rag']['n_wrong']} / {a['churn'][k, 'rag']['n_right']}",
                pct(a["churn"][k, "rag"]["rescue"]), pct(a["churn"][k, "rag"]["harm"]),
                pct(a["churn"][k, "graphrag"]["rescue"]), pct(a["churn"][k, "graphrag"]["harm"])]
               for k in models], "lrrrrr"),
        "",
        "## 5. Question difficulty",
        "",
        "- Each question is banded, per model, by how many of the *other* models answer it "
        "without retrieval; no model is graded against its own answer.",
        "",
        f"![Difficulty]({fig_bands(a)})",
        "",
        table(["Peers correct", "Band size (range over models)", "Mean RAG gain",
               "Mean GraphRAG gain"],
              [[f"{b} of {len(models) - 1}",
                f"{min(a['band_sizes'][b])}–{max(a['band_sizes'][b])}",
                pp(np.mean(a["bands"][b]["rag"])), pp(np.mean(a["bands"][b]["graphrag"]))]
               for b in sorted(a["bands"])], "lrrr"),
        "",
    ]

    order = sorted(a["subjects"], key=lambda s: -len(a["subjects"][s]))
    subj_rows = []
    for s in order:
        van = np.mean([a["subj"][s, k, "vanilla"] for k in models])
        rag = np.mean([a["subj"][s, k, "rag"] - a["subj"][s, k, "vanilla"] for k in models])
        gr = np.mean([a["subj"][s, k, "graphrag"] - a["subj"][s, k, "vanilla"] for k in models])
        subj_rows.append([s, len(a["subjects"][s]), pct(van), pp(rag), pp(gr), pp(gr - rag)])
    lines += [
        "## 6. Subjects",
        "",
        f"![Subjects]({fig_subjects(a, order)})",
        "",
        "- Means over the models; subjects with few questions swing widely.",
        "",
        table(["Subject", "Questions", "No retrieval", "RAG gain", "GraphRAG gain",
               "GraphRAG − RAG"], subj_rows, "lrrrrr"),
        "",
    ]

    def _sig_counts(s, a_, b_, key):
        tests = [a["subj_comp"][s, k, a_, b_] for k in models]
        up = sum(t[key] < 0.05 and t["diff"] > 0 for t in tests)
        down = sum(t[key] < 0.05 and t["diff"] < 0 for t in tests)
        return f"{up} / {down}"

    n_tests = len(order) * len(models)
    test_rows = [[s, len(a["subjects"][s]),
                  _sig_counts(s, "vanilla", "rag", "p"),
                  _sig_counts(s, "vanilla", "rag", "p_holm"),
                  _sig_counts(s, "vanilla", "graphrag", "p"),
                  _sig_counts(s, "vanilla", "graphrag", "p_holm"),
                  _sig_counts(s, "rag", "graphrag", "p"),
                  _sig_counts(s, "rag", "graphrag", "p_holm")] for s in order]
    lines += [
        "### Paired tests by subject",
        "",
        f"- McNemar per subject and model; cells count models significantly above / below "
        f"(p<0.05). Holm: adjusted within each comparison over its {n_tests} "
        "subject × model tests.",
        "",
        table(["Subject", "Questions", "RAG vs none", "RAG vs none (Holm)",
               "GraphRAG vs none", "GraphRAG vs none (Holm)",
               "GraphRAG vs RAG", "GraphRAG vs RAG (Holm)"], test_rows, "lrrrrrrr"),
        "",
    ]

    ev_rows = []
    for k in models:
        ov = a["overlap"][k]
        ev_rows.append((f"{ov['shared']:.2f}", f"{ov['jaccard']:.2f}"))
    if len(set(ev_rows)) == 1:
        ev_table = table(["Models", "Shared articles / q", "Mean Jaccard"],
                         [["all (retrieval does not depend on the QA model)", *ev_rows[0]]],
                         "lrr")
    else:
        ev_table = table(["Model", "Shared articles / q", "Mean Jaccard"],
                         [[run_label(k, a), *row] for k, row in zip(models, ev_rows)], "lrr")

    # GraphRAG − RAG by articles served in common, pooled over models.
    k_top = run_eval.RAG_TEST_TOP_K
    edges = [(0, k_top // 3), (k_top // 3 + 1, 2 * k_top // 3), (2 * k_top // 3 + 1, k_top)]
    bin_rows = []
    for lo, hi in edges:
        won = lost = n_q = 0
        for k in models:
            shared = a["overlap"][k]["shared_per_q"]
            ids = [q for q, c in shared.items() if lo <= c <= hi]
            d = paired({q: a["correct"][k]["rag"][q] for q in ids},
                       {q: a["correct"][k]["graphrag"][q] for q in ids})
            won, lost, n_q = won + d["won"], lost + d["lost"], len(ids)
        total = n_q * len(models)
        bin_rows.append([f"{lo}–{hi} of {k_top}", n_q,
                         pp((won - lost) / total) if total else "—",
                         p_value(mcnemar(lost, won)), f"{won} / {lost}"])
    lines += [
        "## 7. Evidence overlap (RAG vs GraphRAG)",
        "",
        ev_table,
        "",
        "- GraphRAG − RAG by articles served in common (pooled over models):",
        "",
        table(["Shared articles", "Questions", "GraphRAG − RAG", "p", "Won / lost"],
              bin_rows, "lrrrr"),
        "",
        "- Share of the article at each RAG rank (BM25 order) that GraphRAG also serves "
        f"({run_label(models[0], a)}; retrieval does not depend on the QA model):",
        "",
        table(["RAG rank", "Also served by GraphRAG"],
              [[r, pct(v)] for r, v in sorted(a["overlap"][models[0]]["kept_by_rank"].items())],
              "rr"),
        "",
    ]
    fte = future_text_effect(rows[models[0]]["rag"])
    lines += [
        "### Text dated after the cutoff",
        "",
        "- BM25 indexes every wording of an article, with all its provisions; the text "
        "served is the wording in force at the cutoff. Rebuilding the index per cutoff "
        "without later wordings and provisions:",
        f"- Articles kept: {fte['shared']:.2f} of {fte['k']} "
        f"({pct(1 - fte['shared'] / fte['k'])} of the served articles change); "
        f"top 3 kept {pct(fte['top3_kept'])} ({fte['n']} questions).",
        "",
        table(["RAG rank", "Leaves the top k"],
              [[i + 1, pct(v)] for i, v in enumerate(fte["dropped_by_rank"])], "rr"),
        "",
    ]
    cwg = current_wording_gap(rows[models[0]]["rag"])
    lines += [
        "### Served wording vs current wording",
        "",
        f"- Wording served at the cutoff compared with the one in force on {cwg['as_of']}, "
        "the latest date the corpus records:",
        f"- Served articles that differ: {pct(cwg['articles'])}; first-ranked article: "
        f"{pct(cwg['top1'])}; questions with at least one: {pct(cwg['questions'])} "
        f"({cwg['n']} questions).",
        "",
        "## 8. Citation grounding",
        "",
        "- Among answers that cite an article number: share citing one the context did not "
        "contain (a comparator between conditions, not recall).",
        "",
        table(["Model", "RAG: cites any", "RAG: cites outside context",
               "GraphRAG: cites any", "GraphRAG: cites outside context"],
              [[run_label(k, a), pct(a["grounding"][k, "rag"]["cited"]),
                pct(a["grounding"][k, "rag"]["outside"]),
                pct(a["grounding"][k, "graphrag"]["cited"]),
                pct(a["grounding"][k, "graphrag"]["outside"])] for k in models], "lrrrr"),
        "",
        "## 9. Reliability and cost",
        "",
        f"![Cost]({fig_cost(a)})",
        "",
    ]
    rel_rows = []
    for k in models:
        for c in CONDITIONS:
            r = a["reliability"][k, c]
            rel_rows.append([run_label(k, a), CONDITION_LABELS[c], f"`{r['run']}`",
                             r["no_answer"], r["truncated"], f"{r['tokens_in']:,.0f}",
                             f"{r['tokens_out']:,.0f}", format_cost(r["cost"], 3),
                             format_cost(r["cost_1k"], 3)])
    total = sum(r["cost"] or 0 for r in a["reliability"].values())
    lines += [
        table(["Model", "Condition", "Run", "No answer", "Hit token cap", "Input tok/q",
               "Output tok/q", "Cost", "Cost / 1k q"], rel_rows, "lllrrrrrr"),
        "",
        f"- Total TEST cost: {format_cost(total, 2)}.",
        "- Hit token cap: output reached `num_predict`; the answer is parsed from "
        "whatever was written, so this only matters when it also shows as no answer.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{REPORT_NAME}.md"
    path.write_text(build(), encoding="utf-8")
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
