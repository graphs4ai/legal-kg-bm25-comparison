"""Configuration dataclasses for KG construction and graph retrieval.

Values are set in the runners; each build saves its config as JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Article corpus: one row per wording, with validity years.
DEFAULT_ARTICLES_CSV = PROJECT_ROOT / "data" / "laws" / "articles_v5.csv"
ONTOLOGY_DIR = PROJECT_ROOT / "data" / "kg" / "ontology"
# Core: fixed structural layer shown to OE; overlay: semantic layer OE designs.
DEFAULT_ONTOLOGY_CORE = ONTOLOGY_DIR / "core_v1.json"
DEFAULT_ONTOLOGY_SEED = ONTOLOGY_DIR / "overlay_seed_v1.json"
KG_RESULTS_DIR = PROJECT_ROOT / "results" / "kg"
KG_PROMPTS_DIR = PROJECT_ROOT / "prompts" / "kg"

# DeepInfra model ids (HF repo form). Builder and critic of a pair always come
# from different model families.
BUILDER_MODEL = "Qwen/Qwen3.5-9B"
CRITIC_MODEL = "google/gemma-4-26B-A4B-it"

# OE is few calls but constrains all extraction, so it uses larger models.
OE_BUILDER_MODEL = "google/gemma-4-26B-A4B-it"
OE_CRITIC_MODEL = "Qwen/Qwen3.6-35B-A3B"

# Ollama tag-form ids, keyed by agent name (several agents share a model).
OLLAMA_MODELS = {
    "oe_builder": "gemma4:26b",
    "oe_critic": "qwen3.6:35b",
    "oe_merge_critic": "qwen3.6:35b",
    "ke_builder": "qwen3.5:9b",
    "ke_critic": "gemma4:26b",
    "kf_builder": "qwen3.5:9b",
    "kf_critic": "gemma4:26b",
}

# Smaller stand-ins for low-memory machines; same family split.
OLLAMA_MODELS_LIGHT = {
    "oe_builder": "qwen3.5:9b",
    "oe_critic": "gemma4:26b",
    "oe_merge_critic": "gemma4:26b",
    "ke_builder": "qwen3.5:9b",
    "ke_critic": "gemma4:e4b",
    "kf_builder": "qwen3.5:9b",
    "kf_critic": "gemma4:e4b",
}

# Ollama context window; the server default silently drops the prompt start.
OLLAMA_NUM_CTX = 12288

# Paraphrase encoder for KF mention blocking; KF thresholds are calibrated to it.
KF_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Concurrent in-flight agent calls per provider.
PROVIDER_CONCURRENCY = {"mock": 8, "ollama": 1, "deepinfra": 50}

# oe_merge_critic consolidates the per-shard ontology proposals into one overlay.
AGENT_NAMES = (
    "oe_builder", "oe_critic", "oe_merge_critic",
    "ke_builder", "ke_critic",
    "kf_builder", "kf_critic",
)

# DeepInfra (HF repo form) model per agent.
AGENT_MODELS = {
    "oe_builder": OE_BUILDER_MODEL,
    "oe_critic": OE_CRITIC_MODEL,
    "oe_merge_critic": OE_CRITIC_MODEL,
    "ke_builder": BUILDER_MODEL,
    "ke_critic": CRITIC_MODEL,
    "kf_builder": BUILDER_MODEL,
    "kf_critic": CRITIC_MODEL,
}

# Builder/critic pairs that must not share a model family.
CROSS_FAMILY_PAIRS = (
    ("oe_builder", "oe_critic"),
    ("oe_builder", "oe_merge_critic"),
    ("ke_builder", "ke_critic"),
    ("kf_builder", "kf_critic"),
)


def model_family(model: str) -> str:
    """Family of a model id in either naming form ("Qwen/Qwen3.5-9B" -> "qwen")."""

    name = model.lower().rsplit("/", 1)[-1]
    return re.split(r"[:\-._0-9]", name)[0]


def cross_family_violations(models: dict[str, str]) -> list[str]:
    """Messages for pairs in ``CROSS_FAMILY_PAIRS`` that share a model family."""

    problems = []
    for builder, critic in CROSS_FAMILY_PAIRS:
        if builder not in models or critic not in models:
            continue
        if model_family(models[builder]) == model_family(models[critic]):
            problems.append(
                f"{builder} ({models[builder]}) and {critic} "
                f"({models[critic]}) share the "
                f"{model_family(models[builder])} family"
            )
    return problems

# Prompt version per agent under prompts/kg/<agent>/; absent agents use v1.
AGENT_PROMPT_VERSIONS = {
    "oe_builder": "v2",
    "oe_critic": "v2",
    "ke_builder": "v6",
    "ke_critic": "v4",
    "kf_builder": "v2",
    "kf_critic": "v2",
}

# Output token ceiling per agent, where it differs from the AgentConfig default.
AGENT_MAX_TOKENS = {
    "ke_builder": 8192,
}

# Per-request deadline (seconds) per agent, where it differs from the default.
AGENT_REQUEST_TIMEOUTS = {
    "ke_builder": 900.0,
}


@dataclass
class AgentConfig:
    """Model and prompt settings for one agent.

    Attributes:
        name (str): Agent id; also its prompt folder under ``prompts/kg/``.
        provider (str): "deepinfra", "ollama", or "mock".
        think (str | bool): Reasoning effort (DeepInfra) or think flag (Ollama).
        num_ctx (int | None): Ollama context window; must cover prompt plus
            ``max_tokens``.
        request_timeout (float): DeepInfra per-request deadline in seconds.
        request_max_retries (int): DeepInfra retries after a timeout.
    """

    name: str
    model: str = BUILDER_MODEL
    provider: str = "deepinfra"
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_tokens: int = 4096
    think: str | bool = "none"
    num_ctx: int | None = None
    request_timeout: float = 300.0
    request_max_retries: int = 2


def default_agents(
    provider: str = "deepinfra",
    light: bool = False,
    num_ctx: int | None = None,
) -> dict[str, AgentConfig]:
    """Return the agent-model assignment for every construction agent.

    Args:
        provider (str): "deepinfra", "ollama", or "mock", for every agent.
        light (bool): Use the smaller Ollama stand-ins.
        num_ctx (int | None): Ollama context window (default ``OLLAMA_NUM_CTX``).

    Raises:
        ValueError: If a builder/critic pair resolves to one model family.
    """

    think = False if provider == "ollama" else "none"
    table = OLLAMA_MODELS_LIGHT if light else OLLAMA_MODELS

    models = {
        name: (table[name] if provider == "ollama" else AGENT_MODELS[name])
        for name in AGENT_NAMES
    }
    problems = cross_family_violations(models)
    if problems:
        raise ValueError(
            "Builder and critic must come from different model families: "
            + "; ".join(problems)
        )

    return {
        name: AgentConfig(
            name=name,
            model=models[name],
            provider=provider,
            prompt_version=AGENT_PROMPT_VERSIONS.get(name, "v1"),
            max_tokens=AGENT_MAX_TOKENS.get(name, AgentConfig.max_tokens),
            request_timeout=AGENT_REQUEST_TIMEOUTS.get(
                name, AgentConfig.request_timeout),
            think=think,
            num_ctx=(
                (num_ctx or OLLAMA_NUM_CTX) if provider == "ollama" else None
            ),
        )
        for name in AGENT_NAMES
    }


@dataclass
class KGBuildConfig:
    """Full configuration of one KG construction run.

    Attributes:
        kg_version (str): Outputs land in ``results/kg/<kg_version>/``.
        ontology_seed_path (Path): Overlay seed; enters only at the OE merge.
        oe_reuse_from (Path | None): A previous build's ``ontology`` folder to
            copy instead of running OE.
        oe_max_iterations (int): Builder/critic loops per shard before the
            last proposal is force-accepted.
        oe_dedicated_acts (int): Largest acts that get a shard to themselves.
        oe_tail_shards (int): Size-balanced shards for the remaining acts.
        oe_shard_filter (list[str] | None): Run OE only on these shards.
        oe_min_shard_support (int): Shards that must propose a type for it
            to enter the merged overlay.
        oe_ceiling_entity_types (int): Backstop on overlay entity types.
        oe_ceiling_relation_types (int): Backstop on overlay relation types.
        kg_update (bool): Reopen a finished graph at KE; OE is not re-run.
        ke_redo_batch_ids (list[str] | None): Batches to extract again; the
            new record supersedes the old one.
        ke_max_batches (int | None): Cap on extraction batches (None = all).
        ke_batch_tokens (int): Soft cap on article tokens per extraction call.
        ke_batch_articles (int): Cap on articles per extraction call.
        ke_solo_tokens (int): Articles above this size are sent alone.
        ke_max_retries_per_unit (int): Re-extraction passes allowed per batch.
        ke_violation_threshold (int): Retry when critic violations exceed it.
        ke_min_triples (int): Fewer triples per batch triggers a retry
            (0 disables).
        ke_violation_rate (float): Blocking violations tolerated per triple.
        ke_advisory_rate (float): Advisory violations tolerated per triple.
        kf_auto_threshold (float): Similarity at or above which mentions are
            merged without an agent.
        kf_review_threshold (float): Similarity above which mentions form a
            cluster for the KF builder.
        kf_max_cluster_size (int): Largest review cluster sent to an agent.
        kf_max_review_clusters (int): Review clusters kept per round,
            highest-scoring first (0 = none).
        kf_use_embeddings (bool): Block mentions with the encoder, else by
            token overlap.
        kf_max_candidates (int): Cap on review clusters (0 = no cap).
        kf_max_rounds (int): Fusion passes.
        parse_max_retries (int): Re-asks on JSON/schema failure.
        max_units (int | None): Cap on article units processed (None = all).
        unit_filter_docs (list[str] | None): Restrict units to these law ids.
        state_save_every (int): Persist state every N processed units.
        stage_call_timeout_seconds (float | None): Wall-clock ceiling on one
            orchestration unit; the abandoned call still completes and bills.
        log_prompts (str): One of ``build_log.PROMPT_MODES``.
        concurrency (int | None): In-flight agent calls; None uses
            ``PROVIDER_CONCURRENCY``.
    """

    kg_version: str = "kg_v1"
    ontology_core_path: Path = DEFAULT_ONTOLOGY_CORE
    ontology_seed_path: Path = DEFAULT_ONTOLOGY_SEED
    oe_reuse_from: Path | None = None
    results_root: Path = KG_RESULTS_DIR

    oe_max_iterations: int = 2
    oe_articles_per_shard: int = 30
    oe_dedicated_acts: int = 6
    oe_tail_shards: int = 3
    oe_shard_filter: list[str] | None = None
    oe_min_shard_support: int = 2
    oe_ceiling_entity_types: int = 20
    oe_ceiling_relation_types: int = 25

    kg_update: bool = False
    ke_redo_batch_ids: list[str] | None = None
    ke_max_batches: int | None = None
    ke_batch_tokens: int = 800
    ke_batch_articles: int = 5
    ke_solo_tokens: int = 500
    ke_max_retries_per_unit: int = 1
    ke_violation_threshold: int = 2
    ke_min_triples: int = 0
    ke_violation_rate: float = 0.2
    ke_advisory_rate: float = 0.6

    kf_auto_threshold: float = 0.95
    kf_review_threshold: float = 0.82
    kf_use_embeddings: bool = True
    kf_max_cluster_size: int = 15
    kf_max_review_clusters: int = 400
    kf_max_candidates: int = 0
    kf_max_rounds: int = 2

    parse_max_retries: int = 1
    max_units: int | None = None
    unit_filter_docs: list[str] | None = None
    resume: bool = True
    state_save_every: int = 10
    stage_call_timeout_seconds: float | None = None
    concurrency: int | None = None

    log_calls: bool = True
    log_prompts: str = "dedup"
    log_raw_output: bool = True
    log_parsed_output: bool = False

    agents: dict[str, AgentConfig] = field(default_factory=default_agents)

    @property
    def output_dir(self) -> Path:
        """Output folder of this build: ``<results_root>/<kg_version>/``."""
        return self.results_root / self.kg_version

    @property
    def provider(self) -> str:
        """Provider of the agents in this build (from the KE builder)."""
        return self.agents["ke_builder"].provider

    def resolved_concurrency(self) -> int:
        """Return the effective concurrency: explicit, else the provider default."""

        if self.concurrency is not None:
            return max(1, self.concurrency)
        return PROVIDER_CONCURRENCY.get(self.provider, 1)

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable dict of the full configuration."""
        data = asdict(self)
        data["ontology_core_path"] = str(self.ontology_core_path)
        data["ontology_seed_path"] = str(self.ontology_seed_path)
        data["oe_reuse_from"] = (
            str(self.oe_reuse_from) if self.oe_reuse_from else None)
        data["results_root"] = str(self.results_root)
        return data


@dataclass
class GraphRAGConfig:
    """Retrieval-side parameters for the GraphRAG condition.

    Attributes:
        graph_dir (Path): Folder with ``nodes.jsonl`` / ``edges.jsonl``.
        laws_csv (Path | None): Chunk corpus, only for ``text_recovery="chunk"``.
        linker (str): Question-to-graph linking: "lexical" or "llm".
        hops (int): Subgraph expansion depth.
        max_seeds (int): Max seed nodes linked from the question.
        fanout_cap (int): Max neighbors expanded per node per hop.
        max_nodes (int): Cap on subgraph nodes.
        max_edges (int): Cap on serialized subgraph edges.
        hop_decay (float): Score multiplier applied per hop.
        articles_csv (Path | None): Article corpus, for ``text_recovery="article"``.
        expand_citations (bool): Follow citation edges out of linked articles.
        citation_hops (int): Depth of the citation expansion.
        text_recovery (str): "article" (wording at the exam year), "chunk", or
            "none" (graph-only context).
        include_graph_edges (bool): Include the serialized edge list.
        max_edges_per_node (int): Cap on serialized edges per node.
        serialize_relation_denylist (tuple[str, ...]): Relations left out of
            the serialized edges; traversal still follows them.
        hub_damp (bool): Divide seed and edge scores by node degree.
        linker_token_overlap (bool): Also score entities by IDF-weighted token
            overlap with the question.
        linker_max_df_frac (float): Drop query tokens found in more than this
            fraction of entity nodes.
        linker_min_token_len (int): Shortest query token used for overlap.
        max_context_tokens (int | None): Budget for recovered text (4 chars per
            token); the first record is always kept.
        edges_follow_text (bool): Serialize only edges whose articles appear
            in the recovered text.
        linker_embedding (bool): Also score entities by embedding similarity.
        entity_embeddings_dir (Path | None): Cache from
            ``build_entity_embeddings.py``.
        linker_embedding_top_n (int): Nearest entities pulled by similarity.
        linker_method_weights (tuple[float, float, float]): Weights for
            (phrase, token-overlap, embedding) scores, each normalized to its
            best hit per query.
        linker_article_text (bool): Seed articles by BM25 over their wording.
        linker_article_text_top_n (int): Articles pulled by BM25.
        linker_article_text_weight (float): Weight of the BM25 channel.
        linker_article_refs (bool): Seed articles the question cites
            explicitly ("art. 5º da CF"); a bare number without an act is ignored.
        linker_reference_score (float): Seed score of a cited article, above
            1.0 and exempt from hub damping.
        expand_siblings (bool): Recover art. N±1 of top articles when they
            share entity neighbors with the subgraph.
        sibling_min_shared (int): Shared entity neighbors a sibling needs.
        sibling_max_parents (int): Top articles whose siblings are considered.
        ranking (str): "expand" (hop-decay scoring) or "ppr" (Personalized
            PageRank started from BM25 articles and entity seeds).
        ppr_damping (float): Probability of following an edge at each step.
        ppr_bm25_top_n (int): BM25 articles that receive start mass.
        ppr_seed_weight (float): Start mass share of entity seeds and cited
            articles; the rest goes to the BM25 articles.
        ppr_weight_mentions (float): Edge weight article-entity, times
            ``1 / ln(e + degree)`` of the entity.
        ppr_weight_semantic (float): Edge weight entity-entity.
        ppr_weight_citation (float): Edge weight of a citation, both directions.
        ppr_weight_adjacent (float): Edge weight art. N - N±1 of one act.
        ppr_weight_adjacent2 (float): Edge weight art. N - N±2; 0 disables.
        ppr_weight_section (float): Edge weight article - its deepest section.
        ppr_iterations (int): Maximum power iterations.
        ppr_degree_norm (float): Rank by walk score / weighted degree^β;
            0 disables.
        ppr_keep_lexical (int): Top BM25 articles always served; 0 disables.
        annotate_entities (int): List under each article its assertions on the
            question's top-N linked entities; 0 disables.
        annotate_max_lines (int): Most assertions listed under one article.
        article_content (str): "text" (wording in force) or "assertions"
            (header, number, section path, then assertions in force by label).
        assertions_path (bool): With "assertions", show the section path.
        assertions_by_provision (bool): With "assertions", drop an assertion
            whose entity comes from a caput or paragraph not in force at the
            cutoff.
        ppr_seed_kinds (str): Seeds of the walk: "all", "entities" or
            "references".
    """

    graph_dir: Path = KG_RESULTS_DIR / "kg_v1_1" / "graph"
    laws_csv: Path | None = None
    articles_csv: Path | None = DEFAULT_ARTICLES_CSV
    linker: str = "lexical"
    hops: int = 1
    expand_citations: bool = True
    citation_hops: int = 1
    max_seeds: int = 20
    fanout_cap: int = 25
    max_nodes: int = 40
    max_edges: int = 40
    hop_decay: float = 0.5
    text_recovery: str = "article"
    include_graph_edges: bool = False
    max_edges_per_node: int = 3
    serialize_relation_denylist: tuple[str, ...] = ()
    linker_token_overlap: bool = True
    linker_max_df_frac: float = 0.02
    linker_min_token_len: int = 4
    max_context_tokens: int | None = 4000
    edges_follow_text: bool = True
    linker_embedding: bool = False
    entity_embeddings_dir: Path | None = None
    linker_embedding_top_n: int = 30
    linker_method_weights: tuple[float, float, float] = (1.0, 0.8, 0.7)
    linker_article_text: bool = True
    linker_article_text_top_n: int = 15
    linker_article_text_weight: float = 1.0
    linker_article_refs: bool = True
    linker_reference_score: float = 1.5
    expand_siblings: bool = True
    sibling_min_shared: int = 1
    sibling_max_parents: int = 8
    hub_damp: bool = True
    ranking: str = "ppr"
    ppr_damping: float = 0.7
    ppr_bm25_top_n: int = 50
    ppr_seed_weight: float = 0.3
    ppr_weight_mentions: float = 1.0
    ppr_weight_semantic: float = 1.0
    ppr_weight_citation: float = 1.0
    ppr_weight_adjacent: float = 0.5
    ppr_weight_adjacent2: float = 0.0
    ppr_weight_section: float = 0.5
    ppr_iterations: int = 50
    ppr_degree_norm: float = 0.0
    ppr_keep_lexical: int = 0
    annotate_entities: int = 0
    annotate_max_lines: int = 3
    article_content: str = "text"
    assertions_path: bool = True
    assertions_by_provision: bool = False
    ppr_seed_kinds: str = "all"
