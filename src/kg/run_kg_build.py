"""KG construction runner: edit the config below, then run

    python -m src.kg.run_kg_build

Outputs go to ``results/kg/<KG_VERSION>/``; interrupted builds resume from ``state.json``.
"""

import pandas as pd
from loguru import logger

from src.kg.agents import Agent
from src.kg.build_log import BuildLog
from src.kg.config import PROJECT_ROOT, KGBuildConfig, default_agents
from src.kg.kg_metrics import log_build_summary, read_json, write_metrics
from src.kg.mocks import MockLLM
from src.kg.orchestrator import KGOrchestrator
from src.kg.prompt_registry import PromptRegistry
from src.kg.schemas import (
    ExtractionProposal,
    ExtractionReview,
    ClusterMergeDecision,
    MergeReview,
    OntologyMergeReview,
    OntologyProposal,
    OntologyReview,
)

# "mock" (no API cost) | "ollama" (local) | "deepinfra"
PROVIDER = "mock"
KG_VERSION = "kg_v1"                   # output folder under results/kg/

# Exam years served; wordings never in force in this window are dropped.
CORPUS_WINDOW = (2010, 2018)

# Subset of ("oe", "ke", "kf"); a later run resumes from state.json.
STAGES = ("oe", "ke", "kf")

MAX_ARTICLES = None          # None = full corpus
ACT_FILTER = None            # e.g. ["cf_1988", "lei_8906_1994"]
SKIP_BOILERPLATE = True      # drop vetoed / vigência-only provisions
CONCURRENCY = None           # None = provider default (ollama 1, deepinfra 50)

# Ollama only: smaller stand-in models.
OLLAMA_LIGHT = True

# A previous build's ontology folder to copy instead of running OE.
OE_REUSE_FROM = None

# OE: shards = dedicated + tail; calls <= shards x 2 x iterations + 1.
OE_DEDICATED_ACTS = 6
OE_TAIL_SHARDS = 3
OE_ARTICLES_PER_SHARD = 30
OE_MAX_ITERATIONS = 2
OE_SHARD_FILTER = None       # None = every shard, e.g. ["cf_1988"]

# Update a finished graph: KE runs missing units plus KE_REDO_BATCHES.
KG_UPDATE = False
# Batch ids to extract again; the new record supersedes the old one.
KE_REDO_BATCHES = []

KE_MAX_BATCHES = None        # None = every batch

KF_REVIEW_THRESHOLD = 0.86
KF_MAX_REVIEW_CLUSTERS = 5000  # review clusters per round

# Call logs under results/kg/<KG_VERSION>/calls/.
LOG_CALLS = True
LOG_PROMPTS = "dedup"        # "dedup" | "inline" | "hash_only"
LOG_RAW_OUTPUT = True
LOG_PARSED_OUTPUT = False

ARTICLES_CSV = PROJECT_ROOT / "data" / "laws" / "articles_v5.csv"

AGENT_SCHEMAS = {
    "oe_builder": OntologyProposal,
    "oe_critic": OntologyReview,
    "oe_merge_critic": OntologyMergeReview,
    "ke_builder": ExtractionProposal,
    "ke_critic": ExtractionReview,
    "kf_builder": ClusterMergeDecision,
    "kf_critic": MergeReview,
}


def build_agents(
    config: KGBuildConfig, llm_fn=None, call_logger=None
) -> dict[str, Agent]:
    """Instantiate the construction agents from a build config."""

    registry = PromptRegistry()
    return {
        name: Agent(
            config=agent_config,
            schema=AGENT_SCHEMAS[name],
            registry=registry,
            llm_fn=llm_fn,
            parse_max_retries=config.parse_max_retries,
            call_logger=call_logger,
        )
        for name, agent_config in config.agents.items()
    }


def load_articles() -> list[dict]:
    """Read the article corpus and apply the configured build scope."""

    df = pd.read_csv(ARTICLES_CSV, low_memory=False)
    if CORPUS_WINDOW:
        first, last = CORPUS_WINDOW
        valid_from = pd.to_numeric(df["valid_from"], errors="coerce")
        valid_until = pd.to_numeric(df["valid_until"], errors="coerce")
        df = df[
            valid_from.fillna(-(10 ** 9)).le(last)
            & valid_until.fillna(10 ** 9).ge(first)
        ]
    if ACT_FILTER:
        df = df[df["act_id"].isin(ACT_FILTER)]
    if SKIP_BOILERPLATE:
        df = df[~df["is_boilerplate"].astype(bool)]
    if MAX_ARTICLES:
        df = df.head(MAX_ARTICLES)
    return df.to_dict("records")


def main() -> None:
    config = KGBuildConfig(
        kg_version=KG_VERSION,
        max_units=MAX_ARTICLES,
        unit_filter_docs=ACT_FILTER,
        concurrency=CONCURRENCY,
        oe_reuse_from=OE_REUSE_FROM,
        oe_dedicated_acts=OE_DEDICATED_ACTS,
        oe_tail_shards=OE_TAIL_SHARDS,
        oe_shard_filter=OE_SHARD_FILTER,
        oe_articles_per_shard=OE_ARTICLES_PER_SHARD,
        oe_max_iterations=OE_MAX_ITERATIONS,
        kg_update=KG_UPDATE,
        ke_redo_batch_ids=KE_REDO_BATCHES,
        ke_max_batches=KE_MAX_BATCHES,
        kf_review_threshold=KF_REVIEW_THRESHOLD,
        kf_max_review_clusters=KF_MAX_REVIEW_CLUSTERS,
        log_calls=LOG_CALLS,
        log_prompts=LOG_PROMPTS,
        log_raw_output=LOG_RAW_OUTPUT,
        log_parsed_output=LOG_PARSED_OUTPUT,
        agents=default_agents(provider=PROVIDER, light=OLLAMA_LIGHT),
    )

    build_log = BuildLog(
        config=config,
        stages=STAGES,
        inputs={"articles_csv": ARTICLES_CSV},
    )
    build_log.start()

    units = load_articles()
    logger.info(
        "Loaded {} articles from {} | provider={} concurrency={} stages={}",
        len(units), ARTICLES_CSV.name, config.provider,
        config.resolved_concurrency(), STAGES,
    )

    llm_fn = MockLLM() if PROVIDER == "mock" else None
    agents = build_agents(config, llm_fn=llm_fn, call_logger=build_log.calls)

    orchestrator = KGOrchestrator(
        config=config, agents=agents, units=units, build_log=build_log)

    if "oe" in STAGES and orchestrator.state["stage"] == "oe":
        shards = orchestrator.build_shards()
        logger.info(
            "OE will run {} shards, {} articles sampled per shard "
            "(<= {} agent calls)",
            len(shards), config.oe_articles_per_shard,
            len(shards) * 2 * config.oe_max_iterations + 1,
        )
        for shard in shards:
            logger.info("  {:22s} acts={:3d} articles={:5d}",
                        shard["name"], len(shard["act_ids"]),
                        len(shard["articles"]))

    try:
        store = orchestrator.run(stages=STAGES)
    except BaseException:
        build_log.finish(status="failed")
        raise
    build_log.finish(status="completed")

    if store is not None:
        logger.info("Graph saved to {}: {}",
                    orchestrator.graph_dir, store.stats())

    metrics_path = write_metrics(config.output_dir)
    log_build_summary(read_json(metrics_path, {}) or {}, config.output_dir)


if __name__ == "__main__":
    main()
