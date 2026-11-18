"""DEV sweeps report: no-retrieval baselines, the RAG and GraphRAG sweeps, and cost.

    python -m src.evaluation.dev_sweeps_report
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import run_eval
from src.evaluation.report_common import (
    CONDITION_COLORS, INK, INK_MUTED, REPORTS_DIR, current_retrieval, graphrag_tag,
    is_correct, load_index, load_results, model_label, paired, pct, pp,
    p_value, interval, save_figure, setup_style, table, wilson,
)
from src.utils.currency import format_cost, record_cost_usd

REPORT_NAME = "dev_sweeps"
FIGURES_DIR = REPORTS_DIR / "figures" / REPORT_NAME

# Pick rule shared by both sweeps: the most accurate cell, unless a cell
# within this many points uses fewer input tokens.
PICK_MARGIN_PP = 1.0

METHOD_COLORS = {"bm25": CONDITION_COLORS["rag"], "dense": "#4a3aa7"}

# Walk-tag fields: display name, and the value a tag without the field means.
WALK_FIELDS = {
    "d": ("damping", None),
    "seed": ("seed weight", None),
    "emb": ("concept embeddings", "off"),
    "bm": ("BM25 start top-N", None),
    "men": ("mentions", None),
    "sem": ("entity edges", None),
    "cit": ("citations", None),
    "adj": ("adjacency ±1", None),
    "adj2": ("adjacency ±2", None),
    "sec": ("section", None),
    "dn": ("degree norm", "0"),
    "keep": ("keep BM25 top", "0"),
    "seeds": ("walk seeds", "all"),
    "ann": ("assertions served", "off"),
    "content": ("article content", "text"),
    "k": ("k", None),
}

# Suffixes of the "assertions" content token.
_CONTENT_FLAGS = {"head": "caput", "nopath": "no path", "prov": "by paragraph"}


def parse_walk_tag(tag: str) -> dict[str, str]:
    """Field values of a ``ppr_…`` method tag, one per ``WALK_FIELDS`` key."""

    tokens = tag.split("_")[1:]
    values: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("adj2", "seeds"):
            values[token] = tokens[i + 1]
            i += 2
            continue
        if token == "assertions":
            flags = []
            while i + 1 < len(tokens) and tokens[i + 1] in _CONTENT_FLAGS:
                flags.append(_CONTENT_FLAGS[tokens[i + 1]])
                i += 1
            values["content"] = "assertions" + (f" ({', '.join(flags)})" if flags else "")
            i += 1
            continue
        if token in ("emb", "noemb"):
            values["emb"] = "on" if token == "emb" else "off"
        else:
            match = re.match(r"([a-z]+?)([\d.x]+)$", token)
            if match:
                values[match.group(1)] = match.group(2)
        i += 1
    return {key: values.get(key, default) for key, (_, default) in WALK_FIELDS.items()}


def pick_by_rule(cells: list[dict]) -> dict | None:
    """The fewest-token cell within ``PICK_MARGIN_PP`` of the most accurate one."""

    if not cells:
        return None
    best = max(c["accuracy"] for c in cells)
    close = [c for c in cells if best - c["accuracy"] <= PICK_MARGIN_PP / 100 + 1e-9]
    return min(close, key=lambda c: c["tokens"])


# --- Run selection ---
def _latest(runs: list[dict], key) -> dict[tuple, dict]:
    """Latest run per key (runs are oldest first)."""

    chosen = {}
    for run in runs:
        chosen[key(run)] = run
    return chosen


def collect() -> dict:
    """Every run the report shows, with loaded correctness."""

    index = load_index()
    dev_full = [r for r in index if r["split"] == "dev"
                and r["total"] == len(_dev_ids())]
    sweep_model = run_eval.QA_MODELS[run_eval.SWEEP_MODEL]["model"]

    vanilla = _latest(
        [r for r in dev_full if r["condition"] == "vanilla"
         and r["model"] in _qa_model_ids()],
        key=lambda r: r["model"])

    def current(run):
        return run["model"] == sweep_model and current_retrieval(run, "dev")

    rag = _latest(
        [r for r in dev_full if r["condition"] == "rag" and current(r)],
        key=lambda r: (r["retrieval_method"], r["retrieval_top_k"],
                       r.get("retrieval_query_mode") or "question_choices"))
    graph = _latest(
        [r for r in dev_full if r["condition"] == "graphrag" and current(r)
         and str(r.get("retrieval_method", "")).startswith("ppr")],
        key=lambda r: r["retrieval_method"])

    test_tag = graphrag_tag(run_eval.GRAPHRAG_TEST_CELL)

    def correctness(run):
        return {qid: is_correct(row) for qid, row in load_results(run).items()}

    return {
        "sweep_model": sweep_model,
        "vanilla": {m: {"run": r, "correct": correctness(r)} for m, r in vanilla.items()},
        "rag": [{"run": r, "correct": correctness(r)} for r in rag.values()],
        "graph": [{"run": r, "correct": correctness(r)} for r in graph.values()],
        "graph_test_tag": test_tag,
    }


def _dev_ids():
    from src.evaluation.report_common import split_ids
    return split_ids()["dev"]


def _qa_model_ids() -> dict[str, str]:
    return {s["model"]: key for key, s in run_eval.QA_MODELS.items()}


def _cell(entry: dict, reference: dict[str, bool] | None) -> dict:
    """Accuracy, tokens, cost and the paired comparison of one sweep cell."""

    run, correct = entry["run"], entry["correct"]
    n = len(correct)
    hits = sum(correct.values())
    cell = {
        "run": run,
        "correct": correct,
        "n": n,
        "accuracy": hits / n,
        "ci": wilson(hits, n),
        "tokens": run["avg_input_tokens"],
        "cost": record_cost_usd(run),
    }
    if reference is not None:
        cell["vs_ref"] = paired(reference, correct)
    return cell


# --- Figures ---
def _fig_vanilla(models: list[dict], sweep_model: str) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ys = np.arange(len(models))
    for y, m in zip(ys, models):
        is_sweep = m["model"] == sweep_model
        ax.barh(y, m["accuracy"] * 100, height=0.55,
                color=CONDITION_COLORS["vanilla"], alpha=1 if is_sweep else 0.45)
        lo, hi = m["ci"]
        ax.plot([lo * 100, hi * 100], [y, y], color=INK, linewidth=1.2)
        ax.text(hi * 100 + 0.8, y, f"{m['accuracy'] * 100:.1f}%"
                + ("  ← sweep model" if is_sweep else ""),
                va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [model_label(m["model"]) for m in models])
    ax.set_xlabel("No-retrieval accuracy on DEV (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_title("QA models on DEV, no retrieval (95% CI)", loc="left")
    return save_figure(fig, FIGURES_DIR, "vanilla_dev")


def _fig_rag_k(cells: list[dict], vanilla_acc: float, qonly: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for method in sorted({c["run"]["retrieval_method"] for c in cells}):
        rows = sorted((c for c in cells if c["run"]["retrieval_method"] == method),
                      key=lambda c: c["run"]["retrieval_top_k"])
        ks = [c["run"]["retrieval_top_k"] for c in rows]
        accs = [c["accuracy"] * 100 for c in rows]
        color = METHOD_COLORS.get(method, INK_MUTED)
        ax.plot(ks, accs, marker="o", markersize=7, color=color,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(method, (ks[-1], accs[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", color=INK, fontsize=9)
    for c in qonly:
        ax.plot(c["run"]["retrieval_top_k"], c["accuracy"] * 100, marker="X",
                markersize=9, color=METHOD_COLORS.get(c["run"]["retrieval_method"], INK),
                markeredgecolor="white")
        ax.annotate("question only", (c["run"]["retrieval_top_k"], c["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, -2), fontsize=9, color=INK)
    ax.axhline(vanilla_acc * 100, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("no retrieval", (ax.get_xlim()[0], vanilla_acc * 100),
                textcoords="offset points", xytext=(4, 4), fontsize=9, color=INK_MUTED)
    ax.set_xlabel("k (articles retrieved)")
    ax.set_ylabel("DEV accuracy (%)")
    ax.set_xticks(sorted({c["run"]["retrieval_top_k"] for c in cells}))
    ax.set_title("RAG sweep: accuracy by method and k", loc="left")
    return save_figure(fig, FIGURES_DIR, "rag_by_k")


def _fig_tokens(rag: list[dict], graph: list[dict], vanilla_acc: float,
                rag_pick: dict | None, graph_test: dict | None) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method in sorted({c["run"]["retrieval_method"] for c in rag}):
        rows = sorted((c for c in rag if c["run"]["retrieval_method"] == method),
                      key=lambda c: c["tokens"])
        ax.plot([c["tokens"] for c in rows], [c["accuracy"] * 100 for c in rows],
                marker="o", markersize=7, color=METHOD_COLORS.get(method, INK_MUTED),
                markeredgecolor="white", markeredgewidth=1.5, label=f"RAG {method}")
    ax.scatter([c["tokens"] for c in graph], [c["accuracy"] * 100 for c in graph],
               s=60, color=CONDITION_COLORS["graphrag"], edgecolor="white",
               linewidth=1.5, zorder=3, label="GraphRAG cells")
    for cell, text in ((rag_pick, "RAG pick"), (graph_test, "GraphRAG → TEST")):
        if cell:
            ax.annotate(text, (cell["tokens"], cell["accuracy"] * 100),
                        textcoords="offset points", xytext=(6, 8), fontsize=9,
                        color=INK, arrowprops={"arrowstyle": "-", "color": INK_MUTED})
    ax.axhline(vanilla_acc * 100, color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.annotate("no retrieval", (ax.get_xlim()[0], vanilla_acc * 100),
                textcoords="offset points", xytext=(4, 4), fontsize=9, color=INK_MUTED)
    ax.set_xlabel("Average input tokens per question")
    ax.set_ylabel("DEV accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_title("Accuracy against context size, every DEV cell", loc="left")
    return save_figure(fig, FIGURES_DIR, "accuracy_vs_tokens")


def _fig_graph_forest(graph: list[dict], labels: list[str],
                      test_tag: str | None) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 0.45 * len(graph) + 1.2))
    ys = np.arange(len(graph))[::-1]
    for y, cell, label in zip(ys, graph, labels):
        cmp = cell["vs_ref"]
        chosen = cell["run"]["retrieval_method"] == test_tag
        ax.plot([cmp["lo"] * 100, cmp["hi"] * 100], [y, y], color=INK_MUTED, linewidth=1.5)
        ax.plot(cmp["diff"] * 100, y, "o", markersize=8,
                color=CONDITION_COLORS["graphrag"],
                markeredgecolor=INK if chosen else "white", markeredgewidth=1.5)
        ax.text(max(cmp["hi"] * 100, 0) + 0.3, y,
                f"{cmp['diff'] * 100:+.1f}" + ("  (→ TEST)" if chosen else ""),
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(ys, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("GraphRAG − RAG pick, DEV accuracy (points, 95% paired CI)")
    ax.set_title("GraphRAG cells against the RAG pick", loc="left")
    return save_figure(fig, FIGURES_DIR, "graph_vs_rag")


# --- Report ---
def build() -> str:
    data = collect()
    sweep_model = data["sweep_model"]
    setup_style()
    lines = [
        "# DEV sweeps — retrieval tuning",
        "",
        f"- Split: DEV, {len(_dev_ids())} questions, disjoint from TEST.",
        f"- Sweep model: **{model_label(sweep_model)}** "
        f"(`{run_eval.SWEEP_MODEL}` in `run_eval.QA_MODELS`).",
        f"- Pick rule (both sweeps): most accurate cell, unless a cell within "
        f"{PICK_MARGIN_PP:g} point uses fewer input tokens.",
        f"- Context cap: {run_eval.MAX_CONTEXT_TOKENS:,} tokens for both retrievers.",
        "",
    ]

    # 1 --- No retrieval
    models = []
    for model, entry in data["vanilla"].items():
        cell = _cell(entry, None)
        cell["model"] = model
        rows = load_results(entry["run"])
        cell["no_answer"] = sum(r.get("status") != "answered" for r in rows.values())
        models.append(cell)
    models.sort(key=lambda c: c["accuracy"])
    lines += ["## 1. No retrieval — sweep-model choice", ""]
    if models:
        rank = [c["model"] for c in models]
        lines += [
            f"- Ranking (weakest first): "
            + " < ".join(f"{model_label(m)} {pct(c['accuracy'])}"
                         for m, c in zip(rank, models)) + ".",
            f"- Sweep model rank: {rank.index(sweep_model) + 1} of {len(rank)}."
            if sweep_model in rank else "- Sweep model has no DEV vanilla run.",
            "",
            f"![No-retrieval DEV accuracy]({_fig_vanilla(models, sweep_model)})",
            "",
            table(["Model", "Run", "Accuracy", "95% CI", "No answer", "Tokens/q", "Cost"],
                  [[model_label(c["model"]), f"`{c['run']['run_id']}`",
                    pct(c["accuracy"]), f"{pct(c['ci'][0])}–{pct(c['ci'][1])}",
                    c["no_answer"], f"{c['tokens']:,.0f}", format_cost(c["cost"], 3)]
                   for c in models], "llrrrrr"),
            "",
        ]
    vanilla_ref = data["vanilla"].get(sweep_model, {}).get("correct")
    vanilla_acc = (sum(vanilla_ref.values()) / len(vanilla_ref)) if vanilla_ref else float("nan")

    # 2 --- RAG
    rag_all = [_cell(e, vanilla_ref) for e in data["rag"]]
    grid = [c for c in rag_all
            if (c["run"].get("retrieval_query_mode") or "question_choices") == "question_choices"]
    qonly = [c for c in rag_all if c not in grid]
    grid.sort(key=lambda c: (c["run"]["retrieval_method"], c["run"]["retrieval_top_k"]))
    rag_pick = next((c for c in grid
                     if c["run"]["retrieval_method"] == run_eval.RAG_TEST_METHOD
                     and c["run"]["retrieval_top_k"] == run_eval.RAG_TEST_TOP_K), None)
    rule_rag = pick_by_rule(grid)
    lines += ["## 2. RAG sweep", ""]
    if grid:
        best = max(grid, key=lambda c: c["accuracy"])
        by_method = {}
        for c in grid:
            by_method.setdefault(c["run"]["retrieval_method"], []).append(c)
        lines += [
            f"- Grid: {len(by_method)} methods × "
            f"{len({c['run']['retrieval_top_k'] for c in grid})} k values = {len(grid)} cells.",
            f"- Most accurate: {best['run']['retrieval_method']} k={best['run']['retrieval_top_k']} "
            f"({pct(best['accuracy'])}).",
            f"- Rule pick: {rule_rag['run']['retrieval_method']} k={rule_rag['run']['retrieval_top_k']} "
            f"({pct(rule_rag['accuracy'])}, {rule_rag['tokens']:,.0f} tokens/q).",
            f"- TEST config (`run_eval`): {run_eval.RAG_TEST_METHOD} k={run_eval.RAG_TEST_TOP_K}"
            + (" — matches the rule." if rag_pick is rule_rag else " — **differs from the rule pick**."),
        ]
        for method, cells in sorted(by_method.items()):
            lines.append(
                f"- {method}: {pct(min(c['accuracy'] for c in cells))}–"
                f"{pct(max(c['accuracy'] for c in cells))} across k; largest p vs no retrieval "
                f"{p_value(max(c['vs_ref']['p'] for c in cells))}."
                if vanilla_ref else f"- {method}: see table.")
        for c in qonly:
            base = next((g for g in grid if g["run"]["retrieval_method"] == c["run"]["retrieval_method"]
                         and g["run"]["retrieval_top_k"] == c["run"]["retrieval_top_k"]), None)
            if base:
                cmp = paired(base["correct"], c["correct"])
                lines.append(
                    f"- Question-only query ({c['run']['retrieval_method']} k={c['run']['retrieval_top_k']}): "
                    f"{pct(c['accuracy'])}, {pp(cmp['diff'])} vs question + choices "
                    f"(p {p_value(cmp['p'])}).")
        lines += [
            "",
            f"![RAG by k]({_fig_rag_k(grid, vanilla_acc, qonly)})",
            "",
            table(["Method", "k", "Query", "Run", "Accuracy", "vs no retrieval", "95% CI", "p",
                   "Tokens/q", "Cost", ""],
                  [[c["run"]["retrieval_method"], c["run"]["retrieval_top_k"],
                    "question only" if c in qonly else "question + choices",
                    f"`{c['run']['run_id']}`", pct(c["accuracy"]),
                    pp(c["vs_ref"]["diff"]) if vanilla_ref else "—",
                    interval(c["vs_ref"]["lo"], c["vs_ref"]["hi"]) if vanilla_ref else "—",
                    p_value(c["vs_ref"]["p"]) if vanilla_ref else "—",
                    f"{c['tokens']:,.0f}", format_cost(c["cost"], 3),
                    "**TEST**" if c is rag_pick else ""]
                   for c in grid + qonly], "lrllrrrrrrl"),
            "",
        ]

    # 3 --- GraphRAG
    rag_ref = rag_pick["correct"] if rag_pick else None
    graph = [_cell(e, rag_ref) for e in data["graph"]]
    graph.sort(key=lambda c: c["run"]["run_number"])
    lines += ["## 3. GraphRAG sweep (Personalized PageRank walk)", ""]
    if graph and rag_ref is not None:
        parsed = [parse_walk_tag(c["run"]["retrieval_method"]) for c in graph]
        varying = [k for k in WALK_FIELDS if len({p[k] for p in parsed}) > 1]
        fixed = [k for k in WALK_FIELDS if k not in varying]
        base = parsed[0]
        labels = []
        for p in parsed:
            changes = [f"{WALK_FIELDS[k][0]} {p[k]}" for k in varying if p[k] != base[k]]
            labels.append(", ".join(changes) if changes else "base")
        test_cell = next((c for c in graph
                          if c["run"]["retrieval_method"] == data["graph_test_tag"]), None)
        rule_graph = pick_by_rule(graph)
        better = sum(c["vs_ref"]["p"] < 0.05 and c["vs_ref"]["diff"] > 0 for c in graph)
        worse = sum(c["vs_ref"]["p"] < 0.05 and c["vs_ref"]["diff"] < 0 for c in graph)
        diffs = [c["vs_ref"]["diff"] for c in graph]
        lines += [
            f"- {len(graph)} cells, each vs the RAG pick on the same {graph[0]['vs_ref']['n']} questions.",
            f"- Fixed across cells: " + ", ".join(
                f"{WALK_FIELDS[k][0]} {base[k]}" for k in fixed) + ".",
            f"- Base cell (`{graph[0]['run']['run_id']}`): "
            + ", ".join(f"{WALK_FIELDS[k][0]} {base[k]}" for k in varying) + ".",
            f"- GraphRAG − RAG: {pp(min(diffs))} to {pp(max(diffs))} points; "
            f"significantly better in {better}, significantly worse in {worse}, "
            f"tied in {len(graph) - better - worse} of {len(graph)} cells (p<0.05).",
            f"- Rule pick: `{rule_graph['run']['run_id']}` "
            f"({labels[graph.index(rule_graph)]}, {pct(rule_graph['accuracy'])}, "
            f"{rule_graph['tokens']:,.0f} tokens/q).",
        ]
        if test_cell:
            vs_rule = paired(rule_graph["correct"], test_cell["correct"])
            lines.append(
                f"- Carried to TEST: `{test_cell['run']['run_id']}` "
                f"({labels[graph.index(test_cell)]}, {pct(test_cell['accuracy'])})"
                + (" — the rule pick." if test_cell is rule_graph
                   else f" — not the rule pick; {pp(vs_rule['diff'])} vs it "
                        f"(p {p_value(vs_rule['p'])})."))
        lines += [
            "",
            f"![GraphRAG vs RAG]({_fig_graph_forest(graph, labels, data['graph_test_tag'])})",
            "",
            table(["Run", "Change vs base", "Accuracy", "vs RAG pick", "95% CI", "p",
                   "GraphRAG-only / RAG-only right", "Tokens/q", "Cost", ""],
                  [[f"`{c['run']['run_id']}`", label, pct(c["accuracy"]),
                    pp(c["vs_ref"]["diff"]), interval(c["vs_ref"]["lo"], c["vs_ref"]["hi"]),
                    p_value(c["vs_ref"]["p"]),
                    f"{c['vs_ref']['won']} / {c['vs_ref']['lost']}",
                    f"{c['tokens']:,.0f}", format_cost(c["cost"], 3),
                    "**TEST**" if c is test_cell else ""]
                   for c, label in zip(graph, labels)], "llrrrrrrrl"),
            "",
        ]
        # Cells with the same walk that differ only in what they serve.
        serve_keys = ("ann", "content")
        pairs = []
        for i, (ci_, pi) in enumerate(zip(graph, parsed)):
            for cj, pj in zip(graph[i + 1:], parsed[i + 1:]):
                same_walk = all(pi[k] == pj[k] for k in WALK_FIELDS if k not in serve_keys)
                if same_walk and any(pi[k] != pj[k] for k in serve_keys):
                    pairs.append((ci_, pi, cj, pj))
        if pairs:
            rows = []
            for ci_, pi, cj, pj in pairs:
                a_rows, b_rows = load_results(ci_["run"]), load_results(cj["run"])
                shared = [len(set(a_rows[q].get("retrieved_chunk_ids") or [])
                              & set(b_rows[q].get("retrieved_chunk_ids") or []))
                          for q in a_rows.keys() & b_rows.keys()]
                cmp = paired(ci_["correct"], cj["correct"])

                def serves(p):
                    return f"{p['content']}" + (f" + assertions {p['ann']}" if p["ann"] != "off" else "")
                rows.append([f"`{ci_['run']['run_id']}` → `{cj['run']['run_id']}`",
                             f"{serves(pi)} → {serves(pj)}",
                             f"{np.mean(shared):.1f}",
                             f"{pct(ci_['accuracy'])} → {pct(cj['accuracy'])}",
                             pp(cmp["diff"]), interval(cmp["lo"], cmp["hi"]), p_value(cmp["p"]),
                             f"{cmp['won']} / {cmp['lost']}"])
            lines += [
                "### Same walk, different context",
                "",
                "- Pairs of cells with the same walk settings that differ only in what "
                "they serve. Where every article is shared, the difference isolates the "
                "representation; fewer shared articles mean the runs also chose differently.",
                "",
                table(["Runs", "Serves", "Shared articles / q", "Accuracy", "Δ", "95% CI",
                       "p", "Won / lost"], rows, "llrrrrrr"),
                "",
            ]
        lines += [
            "## 4. Context size",
            "",
            "- Each DEV cell's accuracy against its average input tokens; "
            "the dashed line is no retrieval.",
            "",
            f"![Accuracy vs tokens]({_fig_tokens(grid, graph, vanilla_acc, rag_pick, test_cell)})",
            "",
        ]
    else:
        lines += ["- No GraphRAG walk cells or no RAG pick on DEV yet.", ""]

    # 5 --- Effort
    effort = [
        ["No retrieval", len(models), format_cost(sum(c["cost"] or 0 for c in models), 2)],
        ["RAG", len(grid) + len(qonly), format_cost(sum(c["cost"] or 0 for c in grid + qonly), 2)],
        ["GraphRAG", len(graph), format_cost(sum(c["cost"] or 0 for c in graph), 2)],
    ]
    lines += ["## 5. Tuning effort", "",
              table(["Condition", "DEV runs", "Cost"], effort, "lrr"), ""]
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{REPORT_NAME}.md"
    path.write_text(build(), encoding="utf-8")
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
