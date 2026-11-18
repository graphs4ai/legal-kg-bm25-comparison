"""Evaluation runner: edit the config below, pick the runs in ``main()``, then run

    python -m src.evaluation.run_eval

Each run lands in ``results/runs/<condition>/run_###/``.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path

from src.evaluation.oab_eval import OABEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- Data and prompts ---

DATA_OAB_TEST = PROJECT_ROOT / "data" / \
    "oab" / "splits" / "test.jsonl"  # 1,256
DATA_OAB_DEV = PROJECT_ROOT / "data" / "oab" / "splits" / "dev.jsonl"    # 954
SPLIT_PATHS = {"test": DATA_OAB_TEST, "dev": DATA_OAB_DEV}

PROMPT_VANILLA = PROJECT_ROOT / "prompts" / "oab_qa.txt"
PROMPT_RAG = PROJECT_ROOT / "prompts" / "oab_qa.rag.txt"
# RAG and GraphRAG share one prompt; they differ only in the retrieved context.
PROMPT_GRAPHRAG = PROMPT_RAG


# --- QA models ---

TEMPERATURE = 0.0
NUM_PREDICT = 512

# Provider settings: Ollama needs concurrency=1 and a boolean think; DeepInfra
# takes think="none" and parallel requests; Maritaca needs provider="maritaca".
QA_MODELS = {
    "granite": {
        "model": "ibm-granite/granite-4.2-8b",
        "think": "none",
        # Needs both think="none" and the template switch.
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        "concurrency": 50,
    },
    "mistral": {
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "think": "none",
        "concurrency": 50,
    },
    "hy3": {
        "model": "tencent/Hy3",
        "think": "none",
        # Reasoning level is a chat-template variable; "no_think" answers directly.
        "chat_template_kwargs": {"reasoning_effort": "no_think"},
        "concurrency": 50,
    },
    "sabiazinho": {
        "model": "sabiazinho-4",
        "provider": "maritaca",
        # Keeps retrieval prompts under Maritaca tier 1's 2M input tokens/min.
        "concurrency": 6,
    },
}

# DEV sweep model: the second-weakest by no-retrieval DEV accuracy.
SWEEP_MODEL = "mistral"

TEST_MODELS = tuple(QA_MODELS)


# --- Corpus ---

# Corpus version; selects both the corpus CSV and the dense embedding cache.
RETRIEVAL_VERSION = "v5"

LAWS_DIR = PROJECT_ROOT / "data" / "laws"
# Article corpus by version; one file serves every split.
ARTICLES_CSV = {"v5": LAWS_DIR / "articles_v5.csv"}
# Corpus key per split.
CORPUS_BY_SPLIT = {"v5": {"test": "v5", "dev": "v5"}}
# Wordings are resolved per question at its cutoff, so no year filter.
TEMPORAL_FILTER = False


def corpus_for(split: str) -> tuple[Path, Path, str]:
    """Corpus CSV, embedding cache folder and version key for a split."""

    key = CORPUS_BY_SPLIT[RETRIEVAL_VERSION][split]
    return (
        ARTICLES_CSV[key],
        PROJECT_ROOT / "results" / "retrieval" / "flat" / key,
        key,
    )


def articles_for() -> Path | None:
    """Article corpus of ``RETRIEVAL_VERSION``, or None when it has none."""

    return ARTICLES_CSV.get(RETRIEVAL_VERSION)


# --- Retrieval configurations ---

# RAG
# bm25 runs on CPU; dense encodes locally with multilingual-e5-large, caching
# the corpus embeddings on first use.
SWEEP_METHODS = ("bm25", "dense")
RAG_SWEEP_TOP_K = (3, 5, 10, 15)

# Question-only query (no alternatives), at the sweep's best method and k.
RAG_QUESTION_ONLY_METHOD = "bm25"
RAG_QUESTION_ONLY_TOP_K = 15

# TEST config: the RAG sweep's pick.
RAG_TEST_METHOD = "bm25"
RAG_TEST_TOP_K = 10

# Articles flat retrieval may return: "graph" keeps the articles GRAPH_VERSION
# holds, so RAG and GraphRAG differ only in structure; "corpus" keeps them all.
RAG_SCOPE = "graph"
# Retrieval unit of flat RAG: "article" returns whole articles, the unit
# GraphRAG recovers; "unit" splits long articles into provisions.
RAG_UNIT = "article"

# Token budget for the retrieved text, the same for RAG and GraphRAG.
MAX_CONTEXT_TOKENS = 12000

# GraphRAG
GRAPH_VERSION = "kg_v1_1"

# Personalized PageRank walk seeded from the BM25 ranking and linked entities;
# unset parameters take the GraphRAGConfig defaults.
GRAPHRAG_BASE = {"ranking": "ppr"}

# Same number of articles as RAG.
GRAPHRAG_TOP_K = RAG_TEST_TOP_K

# Round 1: damping sweep.
GRAPHRAG_DAMPING_GRID = (0.3, 0.5, 0.7)

# Built by ``python -m src.graphrag.build_entity_embeddings``.
ENTITY_EMBEDDINGS_DIR = (
    PROJECT_ROOT / "results" / "retrieval" / "graph" / GRAPH_VERSION
    / "entity_embeddings")

# Round 2: damping 0.7 with concept-embedding entry, or with full-weight structure.
_CONCEPT_ENTRY = {"linker_embedding": True, "ppr_seed_weight": 0.5}
_FULL_STRUCTURE = {"ppr_weight_adjacent": 1.0, "ppr_weight_adjacent2": 0.5,
                   "ppr_weight_section": 1.0, "ppr_weight_citation": 1.0}
GRAPHRAG_ROUND2 = [
    {"ppr_damping": 0.7, **_CONCEPT_ENTRY},
    {"ppr_damping": 0.7, **_FULL_STRUCTURE},
]

# Round 3: damping 0.7, art. N±1 and section at full weight, N±2 off.
GRAPHRAG_ROUND3 = [
    {"ppr_damping": 0.7, "ppr_weight_adjacent": 1.0, "ppr_weight_adjacent2": 0.0,
     "ppr_weight_section": 1.0, "ppr_weight_citation": 1.0},
]

# Round 4: damping 0.7, each article followed by its assertions about the
# question's top 5 linked entities; then the same walk serving the articles'
# assertions in place of their text (DEV only).
GRAPHRAG_ROUND4 = [
    {"ppr_damping": 0.7, "annotate_entities": 5, "annotate_max_lines": 3},
    {"ppr_damping": 0.7, "article_content": "assertions"},
]

# TEST config: the DEV pick.
GRAPHRAG_TEST_CELL = GRAPHRAG_ROUND4[0]

# Layer ablations of the TEST config: one graph layer walks and fills the context.
# Structural: citations, art. N±1 and sections; serves article text.
GRAPHRAG_STRUCTURAL = {**GRAPHRAG_TEST_CELL, "annotate_entities": 0,
                       "ppr_weight_mentions": 0.0, "ppr_weight_semantic": 0.0,
                       "ppr_seed_kinds": "references"}
# Semantic: extracted entities; serves each article as its assertions in force.
GRAPHRAG_SEMANTIC = {**GRAPHRAG_TEST_CELL, "annotate_entities": 0,
                     "ppr_weight_citation": 0.0, "ppr_weight_adjacent": 0.0,
                     "ppr_weight_adjacent2": 0.0, "ppr_weight_section": 0.0,
                     "ppr_seed_kinds": "entities", "article_content": "assertions",
                     "assertions_path": False, "assertions_by_provision": True}
GRAPHRAG_LAYERS = [GRAPHRAG_STRUCTURAL, GRAPHRAG_SEMANTIC]

def _graphrag_method_tag(config, top_k: int) -> str:
    """Method tag of a walk config, e.g. "ppr_d0.7_seed0.3_noemb_..._k10"."""

    return "_".join([
        f"ppr_d{config.ppr_damping:g}",
        f"seed{config.ppr_seed_weight:g}",
        "emb" if config.linker_embedding else "noemb",
        f"bm{config.ppr_bm25_top_n}",
        f"men{config.ppr_weight_mentions:g}",
        f"sem{config.ppr_weight_semantic:g}",
        f"cit{config.ppr_weight_citation:g}",
        f"adj{config.ppr_weight_adjacent:g}",
        f"adj2_{config.ppr_weight_adjacent2:g}",
        f"sec{config.ppr_weight_section:g}",
        f"dn{config.ppr_degree_norm:g}",
        f"keep{config.ppr_keep_lexical}",
        *([f"seeds_{config.ppr_seed_kinds}"]
          if config.ppr_seed_kinds != "all" else []),
        *([f"ann{config.annotate_entities}x{config.annotate_max_lines}"]
          if config.annotate_entities else []),
        *(["assertions"
           + ("" if config.assertions_path else "_nopath")
           + ("_prov" if config.assertions_by_provision else "")]
          if config.article_content == "assertions" else []),
        f"k{top_k}",
    ])


# --- Building blocks ---

def _evaluate(
    split: str,
    condition: str,
    prompt_path: Path,
    model: str,
    retriever=None,
    top_k: int | None = None,
    **scope,
) -> None:
    """Run one evaluation of one QA model (key in ``QA_MODELS``) on one split.

    ``scope`` holds sampling options such as ``sample_per_group``.
    """

    retrieval = {} if retriever is None else {
        "retriever": retriever,
        "retrieval_top_k": top_k,
        "laws_version": corpus_for(split)[2],
        **({"retrieval_unit": RAG_UNIT, "retrieval_scope": RAG_SCOPE,
            "graph_version": GRAPH_VERSION if RAG_SCOPE == "graph" else None}
           if condition == "rag" else {"graph_version": GRAPH_VERSION}),
    }
    OABEvaluator(
        dataset_path=SPLIT_PATHS[split],
        prompt_path=prompt_path,
        condition=condition,

        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT,
        **QA_MODELS[model],

        **scope,
        **retrieval,
    ).evaluate()


def _law_retriever(split: str, method: str, query_mode: str = "question_choices"):
    """Flat retriever (bm25 or dense) over the split's corpus."""

    # Lazy import: vanilla runs need no RAG dependencies.
    from src.rag import LawRetriever

    laws_csv, embeddings_dir, _ = corpus_for(split)
    return LawRetriever(
        csv_path=laws_csv, method=method, cache_dir=embeddings_dir,
        query_mode=query_mode, temporal_filter=TEMPORAL_FILTER, unit=RAG_UNIT,
        article_ids=_graph_article_ids() if RAG_SCOPE == "graph" else None,
        max_context_tokens=MAX_CONTEXT_TOKENS)


def _graph_article_ids() -> set[str]:
    """Article ids ``GRAPH_VERSION`` holds, read from its nodes file."""

    import json

    from src.kg.config import KG_RESULTS_DIR

    ids = set()
    with open(KG_RESULTS_DIR / GRAPH_VERSION / "graph" / "nodes.jsonl",
              encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)
            if node["type"] == "Article":
                ids.add(node["node_id"][len("article:"):])
    return ids


def graphrag_config(split: str, **overrides):
    """GraphRAG config over ``GRAPH_VERSION`` with optional overrides."""

    from src.kg.config import GraphRAGConfig, KG_RESULTS_DIR

    return GraphRAGConfig(
        graph_dir=KG_RESULTS_DIR / GRAPH_VERSION / "graph",
        articles_csv=articles_for(),
        linker="lexical",
        **{"max_context_tokens": MAX_CONTEXT_TOKENS, **overrides},
    )


def graphrag_method(cell: dict) -> str:
    """Method tag a run of ``cell`` (overrides on ``GRAPHRAG_BASE``) records."""

    return _graphrag_method_tag(
        replace(graphrag_config("test", **GRAPHRAG_BASE), **cell), GRAPHRAG_TOP_K)


def _graph_retriever(split: str, **overrides):
    """Graph retriever over ``GRAPH_VERSION``; its BM25 ranking is RAG's retriever."""

    from src.graphrag import GraphRetriever

    return GraphRetriever(config=graphrag_config(split, **overrides),
                          lexical=_law_retriever(split, "bm25"))


def _graphrag_cells(retriever, cells, split: str, models) -> None:
    """Evaluate each override set, each applied to the base config, with one loaded retriever."""

    base_config = retriever.config
    for overrides in cells:
        retriever.config = replace(base_config, **overrides)
        retriever.method = _graphrag_method_tag(
            retriever.config, GRAPHRAG_TOP_K)
        for model in models:
            _evaluate(split, "graphrag", PROMPT_GRAPHRAG, model,
                      retriever=retriever, top_k=GRAPHRAG_TOP_K)
    retriever.config = base_config


# --- Runs ---

def run_vanilla(split: str = "test", models=tuple(QA_MODELS)) -> None:
    """Vanilla condition (no retrieval) for each model on a split."""

    for model in models:
        _evaluate(split, "vanilla", PROMPT_VANILLA, model)


def run_rag_sweep() -> None:
    """RAG sweep (method x k) on DEV with the sweep model."""

    for method in SWEEP_METHODS:
        retriever = _law_retriever("dev", method)
        for top_k in RAG_SWEEP_TOP_K:
            _evaluate("dev", "rag", PROMPT_RAG, SWEEP_MODEL,
                      retriever=retriever, top_k=top_k)


def run_rag_question_only() -> None:
    """The sweep's best RAG cell with the question stem alone as the query."""

    if RAG_QUESTION_ONLY_METHOD is None or RAG_QUESTION_ONLY_TOP_K is None:
        raise SystemExit(
            "Set RAG_QUESTION_ONLY_METHOD / RAG_QUESTION_ONLY_TOP_K from the sweep.")
    retriever = _law_retriever(
        "dev", RAG_QUESTION_ONLY_METHOD, query_mode="question")
    _evaluate("dev", "rag", PROMPT_RAG, SWEEP_MODEL,
              retriever=retriever, top_k=RAG_QUESTION_ONLY_TOP_K)


def run_rag_test(models=TEST_MODELS) -> None:
    """RAG on the TEST split with the sweep-chosen config."""

    retriever = _law_retriever("test", RAG_TEST_METHOD)
    for model in models:
        _evaluate("test", "rag", PROMPT_RAG, model,
                  retriever=retriever, top_k=RAG_TEST_TOP_K)


def run_graphrag_damping() -> None:
    """GraphRAG round 1: damping sweep on DEV with the sweep model."""

    _graphrag_cells(_graph_retriever("dev", **GRAPHRAG_BASE),
                    [{"ppr_damping": d} for d in GRAPHRAG_DAMPING_GRID],
                    "dev", (SWEEP_MODEL,))


def run_graphrag_round2() -> None:
    """GraphRAG round 2 on DEV with the sweep model."""

    _graphrag_cells(
        _graph_retriever("dev", **GRAPHRAG_BASE,
                         entity_embeddings_dir=ENTITY_EMBEDDINGS_DIR),
        GRAPHRAG_ROUND2, "dev", (SWEEP_MODEL,))


def run_graphrag_round3() -> None:
    """GraphRAG round 3 on DEV with the sweep model."""

    _graphrag_cells(_graph_retriever("dev", **GRAPHRAG_BASE),
                    GRAPHRAG_ROUND3, "dev", (SWEEP_MODEL,))


def run_graphrag_round4() -> None:
    """GraphRAG round 4 on DEV with the sweep model."""

    _graphrag_cells(_graph_retriever("dev", **GRAPHRAG_BASE),
                    GRAPHRAG_ROUND4, "dev", (SWEEP_MODEL,))


def run_graphrag_test(models=TEST_MODELS) -> None:
    """GraphRAG on the TEST split with the DEV-chosen config."""

    if GRAPHRAG_TEST_CELL is None:
        raise SystemExit("Set GRAPHRAG_TEST_CELL to the DEV pick.")
    embeddings = (
        {"entity_embeddings_dir": ENTITY_EMBEDDINGS_DIR}
        if GRAPHRAG_TEST_CELL.get("linker_embedding") else {})
    _graphrag_cells(_graph_retriever("test", **GRAPHRAG_BASE, **embeddings),
                    [GRAPHRAG_TEST_CELL], "test", models)


def run_graphrag_layers(split: str = "test", models=TEST_MODELS) -> None:
    """GraphRAG layer ablations: ``GRAPHRAG_LAYERS`` on a split."""

    _graphrag_cells(_graph_retriever(split, **GRAPHRAG_BASE),
                    GRAPHRAG_LAYERS, split, models)


def main() -> None:
    # Uncomment the runs to launch; they execute in order.
    runs = [
        # partial(run_vanilla, "test"),
        # partial(run_vanilla, "dev"),

        # run_rag_sweep,
        # run_rag_question_only,
        # run_rag_test,

        # run_graphrag_damping,
        # run_graphrag_round2,
        # run_graphrag_round3,
        # run_graphrag_round4,
        # run_graphrag_test,
        # run_graphrag_layers
    ]
    if not runs:
        raise SystemExit("No run selected — uncomment one in main().")
    for run in runs:
        run()


if __name__ == "__main__":
    main()
