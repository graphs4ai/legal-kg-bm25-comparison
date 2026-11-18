"""Layer ablation report: the TEST GraphRAG config with only its structural or semantic layer.

    python -m src.evaluation.layers_report
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import run_eval, test_report
from src.evaluation.report_common import (
    CONDITION_COLORS, INK, INK_MUTED, REPORTS_DIR,
    _grounding, current_retrieval, graphrag_tag, interval, is_correct, load_index,
    load_runs, mcnemar, paired, run_label, shared_articles, pct, pp, p_value, save_figure, setup_style,
    split_ids, table, wilson,
)
from src.utils.currency import format_cost, record_cost_usd

REPORT_NAME = "layers"
FIGURES_DIR = REPORTS_DIR / "figures" / REPORT_NAME

# Ablation cells by name, as run_eval defines them.
LAYERS = {"structural": run_eval.GRAPHRAG_STRUCTURAL,
          "semantic": run_eval.GRAPHRAG_SEMANTIC}
ORDER = ("vanilla", "rag", "structural", "semantic", "graphrag")
RETRIEVAL = ORDER[1:]
LABELS = {"vanilla": "No retrieval", "rag": "RAG", "structural": "Structural",
          "semantic": "Semantic", "graphrag": "GraphRAG (full)"}
# No retrieval in gray, so the four retrieval hues stay distinct.
COLORS = {"vanilla": INK_MUTED, "rag": CONDITION_COLORS["rag"],
          "graphrag": CONDITION_COLORS["graphrag"],
          "structural": "#2a78d6", "semantic": "#4a3aa7"}
# Conditions each layer is compared against.
REFERENCES = ("vanilla", "rag", "graphrag")
# A paired 95% interval inside ±this many points counts as equivalent.
EQUIV_MARGIN_PP = 2.0


# --- Run selection ---
def select_runs(index: list[dict]) -> dict[str, dict[str, dict]]:
    """Latest TEST run per model and condition or layer (model -> name -> run)."""

    chosen = test_report.select_runs(index)
    n_test = len(split_ids()["test"])
    tags = {graphrag_tag(cell): name for name, cell in LAYERS.items()}
    keys = {settings["model"]: key for key, settings in run_eval.QA_MODELS.items()}
    for run in index:
        if (run["split"] != "test" or run["condition"] != "graphrag"
                or run["total"] != n_test or run["model"] not in keys
                or not current_retrieval(run, "test")):
            continue
        name = tags.get(run.get("retrieval_method"))
        if name:
            chosen.setdefault(keys[run["model"]], {})[name] = run
    return chosen


# --- Analysis ---
def gain_retained(layer: float, vanilla: float, full: float) -> float:
    """Share of the full config's gain over no retrieval a layer keeps."""

    gain = full - vanilla
    return (layer - vanilla) / gain if gain else float("nan")


def shared_bins(top_k: int) -> list[tuple[int, int]]:
    """Three bins of articles served in common, out of ``top_k``."""

    return [(0, top_k // 3), (top_k // 3 + 1, 2 * top_k // 3), (2 * top_k // 3 + 1, top_k)]


def analyse(runs: dict, rows: dict) -> dict:
    models = [k for k in run_eval.QA_MODELS if all(c in runs.get(k, {}) for c in ORDER)]
    correct = {k: {c: {q: is_correct(r) for q, r in rows[k][c].items()} for c in ORDER}
               for k in models}
    models.sort(key=lambda k: np.mean(list(correct[k]["vanilla"].values())))

    acc, tokens = {}, {}
    for k in models:
        for c in ORDER:
            n, hits = len(correct[k][c]), sum(correct[k][c].values())
            acc[k, c] = {"n": n, "acc": hits / n, "ci": wilson(hits, n)}
            tokens[k, c] = runs[k][c]["avg_input_tokens"]
    # Context tokens: input tokens above the no-retrieval prompt.
    context = {(k, c): tokens[k, c] - tokens[k, "vanilla"] for k in models for c in ORDER}

    margin = EQUIV_MARGIN_PP / 100
    comp = {}
    for k in models:
        for layer in LAYERS:
            for ref in REFERENCES:
                d = paired(correct[k][ref], correct[k][layer])
                d["equivalent"] = -margin <= d["lo"] and d["hi"] <= margin
                comp[k, ref, layer] = d

    retained = {(k, layer): gain_retained(acc[k, layer]["acc"], acc[k, "vanilla"]["acc"],
                                          acc[k, "graphrag"]["acc"])
                for k in models for layer in LAYERS}

    # Overlap with the full config and RAG; layer − full by shared articles, pooled.
    top_k = run_eval.GRAPHRAG_TOP_K
    overlap, bins = {}, {}
    for layer in LAYERS:
        for ref in ("rag", "graphrag"):
            overlap[layer, ref] = float(np.mean([
                np.mean(list(shared_articles(rows[k][layer], rows[k][ref]).values()))
                for k in models]))
        for lo, hi in shared_bins(top_k):
            won = lost = n_q = 0
            for k in models:
                ids = [q for q, c in shared_articles(rows[k][layer], rows[k]["graphrag"]).items()
                       if lo <= c <= hi]
                d = paired({q: correct[k]["graphrag"][q] for q in ids},
                           {q: correct[k][layer][q] for q in ids})
                won, lost, n_q = won + d["won"], lost + d["lost"], len(ids)
            total = n_q * len(models)
            bins[layer, lo, hi] = {"n": n_q, "won": won, "lost": lost,
                                   "diff": (won - lost) / total if total else float("nan"),
                                   "p": mcnemar(lost, won)}

    churn = {}
    for k in models:
        van = correct[k]["vanilla"]
        wrong = [q for q in van if not van[q]]
        right = [q for q in van if van[q]]
        for c in RETRIEVAL:
            churn[k, c] = {"rescue": np.mean([correct[k][c][q] for q in wrong]),
                           "harm": np.mean([not correct[k][c][q] for q in right])}

    grounding = {}
    for k in models:
        for c in RETRIEVAL:
            flags = [_grounding(r) for r in rows[k][c].values()]
            cited = [f for f in flags if f is not None]
            grounding[k, c] = {"cited": len(cited) / len(flags),
                               "outside": sum(cited) / len(cited) if cited else float("nan")}

    reliability = {}
    for k in models:
        for c in ORDER:
            run, rs = runs[k][c], rows[k][c].values()
            reliability[k, c] = {
                "run": run["run_id"],
                "no_answer": sum(r.get("status") != "answered" for r in rs),
                "truncated": sum(r.get("finish_reason") == "length" for r in rs),
                "tokens_out": run["avg_output_tokens"],
                "cost": record_cost_usd(run),
            }

    return {"models": models, "acc": acc, "tokens": tokens, "context": context,
            "comp": comp, "retained": retained, "overlap": overlap, "bins": bins,
            "churn": churn, "grounding": grounding, "reliability": reliability,
            "runs": runs}


# --- Figures ---
def fig_accuracy_tokens(a: dict) -> str:
    models = a["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 3.2))
    for ax, k in zip(np.atleast_1d(axes), models):
        van = a["acc"][k, "vanilla"]["acc"] * 100
        ax.axhline(van, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(0, van, " No retr.", va="bottom", ha="left", fontsize=7.5, color=INK_MUTED)
        for c in RETRIEVAL:
            ax.scatter(a["context"][k, c], a["acc"][k, c]["acc"] * 100, s=55,
                       color=COLORS[c], edgecolor="white", linewidth=1.5, zorder=3,
                       label=LABELS[c] if k == models[0] else None)
        ax.set_xlim(left=0)
        ax.margins(x=0.25, y=0.25)
        ax.set_title(run_label(k, a), loc="left", fontsize=10)
        ax.set_xlabel("context tokens / q")
    np.atleast_1d(axes)[0].set_ylabel("TEST accuracy (%)")
    fig.suptitle("Accuracy against context size", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.legend(loc="lower center", ncol=len(RETRIEVAL), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(fig, FIGURES_DIR, "accuracy_vs_context")


def fig_forest(a: dict) -> str:
    models = a["models"]
    fig, axes = plt.subplots(1, len(REFERENCES), figsize=(10, 0.6 * len(models) + 1.5),
                             sharey=True)
    ys = np.arange(len(models))[::-1]
    offsets = dict(zip(LAYERS, (0.14, -0.14)))
    for ax, ref in zip(axes, REFERENCES):
        ax.axvspan(-EQUIV_MARGIN_PP, EQUIV_MARGIN_PP, color="#efeee9", zorder=0)
        for layer in LAYERS:
            for y, k in zip(ys, models):
                d = a["comp"][k, ref, layer]
                yy = y + offsets[layer]
                ax.plot([d["lo"] * 100, d["hi"] * 100], [yy, yy], color=INK_MUTED,
                        linewidth=1.3)
                ax.plot(d["diff"] * 100, yy, "o", markersize=7, color=COLORS[layer],
                        markeredgecolor="white", markeredgewidth=1.3,
                        label=LABELS[layer] if (ref, k) == (REFERENCES[0], models[0]) else None)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.margins(x=0.1)
        ax.set_title(f"Layer − {LABELS[ref]}", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("points (95% paired CI)")
    axes[0].set_yticks(ys, [run_label(k, a) for k in models])
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save_figure(fig, FIGURES_DIR, "paired")


# --- Report ---
def _range(values) -> str:
    return f"{pp(min(values))} to {pp(max(values))}"


def _headline(a: dict) -> list[str]:
    models = a["models"]
    n = len(models)
    lines = []
    for layer in LAYERS:
        for ref in REFERENCES:
            ds = [a["comp"][k, ref, layer] for k in models]
            sig_up = sum(d["p"] < 0.05 and d["diff"] > 0 for d in ds)
            sig_down = sum(d["p"] < 0.05 and d["diff"] < 0 for d in ds)
            lines.append(
                f"- **{LABELS[layer]} − {LABELS[ref]}:** {_range([d['diff'] for d in ds])} "
                f"points; significantly above for {sig_up} of {n}, below for {sig_down}; "
                f"equivalent within ±{EQUIV_MARGIN_PP:g} for "
                f"{sum(d['equivalent'] for d in ds)}.")
        kept = [a["retained"][k, layer] for k in models]
        share = [a["context"][k, layer] / a["context"][k, "graphrag"] for k in models]
        lines.append(
            f"- **{LABELS[layer]} keeps** {pct(min(kept), 0)} to {pct(max(kept), 0)} of the "
            f"full config's gain (weakest to strongest: "
            + ", ".join(pct(v, 0) for v in kept)
            + f") with {pct(np.mean(share), 0)} of its context tokens.")
    return lines


def build() -> str:
    runs = select_runs(load_index())
    rows = load_runs(runs)
    a = analyse(runs, rows)
    models = a["models"]
    if not models:
        raise SystemExit("No model has every condition and layer on TEST.")
    setup_style()
    n = a["acc"][models[0], "vanilla"]["n"]
    tags = {c: a["runs"][models[0]][c]["retrieval_method"] for c in ("graphrag", *LAYERS)}

    lines = [
        "# TEST — GraphRAG layer ablation",
        "",
        f"- Split: TEST, {n} questions. The ablations are analyses of the fixed TEST "
        "config; nothing is selected from them.",
        f"- Models ({len(models)}, weakest first): "
        + ", ".join(run_label(k, a) for k in models) + ".",
        f"- GraphRAG (full): `{tags['graphrag']}`.",
        f"- Structural: `{tags['structural']}` — walk over citations, art. N±1 and "
        "sections (parser-built, no LLM), started from BM25 and the articles the question "
        "cites; article text.",
        f"- Semantic: `{tags['semantic']}` — walk over the extracted entities, started "
        "from BM25 and the linked entities; each article as its assertions in force, no text.",
        f"- RAG: {run_eval.RAG_TEST_METHOD} k={run_eval.RAG_TEST_TOP_K}. Every retrieval "
        f"condition serves {run_eval.GRAPHRAG_TOP_K} articles under a "
        f"{run_eval.MAX_CONTEXT_TOKENS:,}-token cap.",
        "- Paired tests: McNemar with continuity correction; 95% paired intervals. "
        f"Equivalent: the interval lies inside ±{EQUIV_MARGIN_PP:g} points.",
        "",
        "## Headline",
        "",
        *_headline(a),
        "",
        "## 1. Accuracy and context size",
        "",
        f"![Accuracy vs context]({fig_accuracy_tokens(a)})",
        "",
        table(["Model"] + [LABELS[c] for c in ORDER],
              [[run_label(k, a)] + [pct(a["acc"][k, c]["acc"]) for c in ORDER] for k in models],
              "l" + "r" * len(ORDER)),
        "",
        "- Context tokens / q: input tokens above the no-retrieval prompt of the same model.",
        "",
        table(["Model"] + [LABELS[c] for c in RETRIEVAL],
              [[run_label(k, a)] + [f"{a['context'][k, c]:,.0f}" for c in RETRIEVAL]
               for k in models], "l" + "r" * len(RETRIEVAL)),
        "",
        "## 2. Paired comparisons",
        "",
        f"![Paired]({fig_forest(a)})",
        "",
    ]
    comp_rows = []
    for layer in LAYERS:
        for ref in REFERENCES:
            for k in models:
                d = a["comp"][k, ref, layer]
                comp_rows.append([
                    f"{LABELS[layer]} − {LABELS[ref]}", run_label(k, a), pp(d["diff"]),
                    interval(d["lo"], d["hi"]), p_value(d["p"]), f"{d['won']} / {d['lost']}",
                    "yes" if d["equivalent"] else "no"])
    lines += [
        table(["Comparison", "Model", "Δ (points)", "95% CI", "p", "Won / lost",
               f"Within ±{EQUIV_MARGIN_PP:g}"], comp_rows, "llrrrrl"),
        "",
        "## 3. Gain retained",
        "",
        "- (layer − no retrieval) / (full − no retrieval): share of the full config's gain "
        "the layer keeps alone. Negative: below no retrieval. Unstable when the full "
        "gain is small.",
        "",
        table(["Model", "Full gain"]
              + [f"{LABELS[layer]} {col}" for layer in LAYERS
                 for col in ("retained", "context share")],
              [[run_label(k, a),
                pp(a["acc"][k, "graphrag"]["acc"] - a["acc"][k, "vanilla"]["acc"])]
               + [cell for layer in LAYERS for cell in (
                   pct(a["retained"][k, layer], 0),
                   pct(a["context"][k, layer] / a["context"][k, "graphrag"], 0))]
               for k in models], "lr" + "rr" * len(LAYERS)),
        "",
        "## 4. Evidence overlap",
        "",
        table(["Layer", f"Shared with full (of {run_eval.GRAPHRAG_TOP_K})", "Shared with RAG"],
              [[LABELS[layer], f"{a['overlap'][layer, 'graphrag']:.1f}",
                f"{a['overlap'][layer, 'rag']:.1f}"] for layer in LAYERS], "lrr"),
        "",
        "- Layer − full by articles served in common (pooled over models). Where most "
        "articles are shared, the gap is mostly what each serves about them, not which.",
        "",
        table(["Layer", "Shared with full", "Questions", "Layer − full", "p", "Won / lost"],
              [[LABELS[layer], f"{lo}–{hi}", b["n"], pp(b["diff"]), p_value(b["p"]),
                f"{b['won']} / {b['lost']}"]
               for layer in LAYERS
               for lo, hi in shared_bins(run_eval.GRAPHRAG_TOP_K)
               for b in [a["bins"][layer, lo, hi]]], "llrrrr"),
        "",
        "## 5. Rescue and harm",
        "",
        "- Rescue: share of questions wrong without retrieval that the condition gets right. "
        "Harm: share right without retrieval that it gets wrong.",
        "",
        table(["Model"] + [f"{LABELS[c]} {m}" for c in RETRIEVAL for m in ("rescue", "harm")],
              [[run_label(k, a)] + [pct(a["churn"][k, c][m]) for c in RETRIEVAL
                                 for m in ("rescue", "harm")] for k in models],
              "l" + "rr" * len(RETRIEVAL)),
        "",
        "## 6. Citation grounding",
        "",
        "- Among answers that cite an article number: share citing one the context did not "
        "contain. Semantic contexts name each article in its header, so the check applies.",
        "",
        table(["Model"] + [LABELS[c] for c in RETRIEVAL],
              [[run_label(k, a)] + [
                  f"{pct(a['grounding'][k, c]['outside'])} "
                  f"(cites {pct(a['grounding'][k, c]['cited'], 0)})" for c in RETRIEVAL]
               for k in models], "l" + "r" * len(RETRIEVAL)),
        "",
        "## 7. Runs and cost",
        "",
    ]
    rel_rows = []
    for k in models:
        for c in ORDER:
            r = a["reliability"][k, c]
            rel_rows.append([run_label(k, a), LABELS[c], f"`{r['run']}`", r["no_answer"],
                             r["truncated"], f"{a['tokens'][k, c]:,.0f}",
                             f"{r['tokens_out']:,.0f}", format_cost(r["cost"], 3)])
    layer_cost = sum(a["reliability"][k, c]["cost"] or 0 for k in models for c in LAYERS)
    lines += [
        table(["Model", "Condition", "Run", "No answer", "Hit token cap", "Input tok/q",
               "Output tok/q", "Cost"], rel_rows, "lllrrrrr"),
        "",
        f"- Ablation runs cost: {format_cost(layer_cost, 2)}.",
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
