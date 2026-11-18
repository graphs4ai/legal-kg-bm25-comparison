"""Encode every non-article graph node's name with E5 for semantic entity linking.

Writes ``embeddings.npy`` and ``node_ids.json`` under
``results/retrieval/graph/<version>/entity_embeddings/``; skips if present.

    python -m src.graphrag.build_entity_embeddings
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.evaluation.run_eval import GRAPH_VERSION
from src.kg.config import KG_RESULTS_DIR
from src.kg.graph_store import GraphStore
from src.rag.text_utils import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODER_NAME = "intfloat/multilingual-e5-large"
_E5_PASSAGE_PREFIX = "passage: "

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "retrieval" / "graph" / GRAPH_VERSION
    / "entity_embeddings"
)
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
NODE_IDS_PATH = OUTPUT_DIR / "node_ids.json"

# Article and act nodes are matched as text; only concept nodes get embeddings.
_EXCLUDED_TYPES = frozenset({"Article", "Act", "NormativeAct"})


def build_entity_embeddings() -> None:
    """Encode every entity node's name and cache the resulting matrix."""

    if EMBEDDINGS_PATH.exists() and NODE_IDS_PATH.exists():
        logger.info("Entity embeddings already exist at {}, skipping",
                    OUTPUT_DIR)
        return

    import numpy as np
    from sentence_transformers import SentenceTransformer

    store = GraphStore.load(KG_RESULTS_DIR / GRAPH_VERSION / "graph")
    node_ids, passages = [], []
    for node in store.nodes.values():
        if node.type in _EXCLUDED_TYPES:
            continue
        node_ids.append(node.node_id)
        passages.append(_E5_PASSAGE_PREFIX + normalize_text(node.name))

    logger.info("Encoding {} entity names with {}", len(node_ids), ENCODER_NAME)
    encoder = SentenceTransformer(ENCODER_NAME)
    embeddings = encoder.encode(
        passages,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings.astype("float32"))
    NODE_IDS_PATH.write_text(
        json.dumps(node_ids, ensure_ascii=False), encoding="utf-8")
    logger.info("Cached {} entity embeddings to {}", len(node_ids), OUTPUT_DIR)


if __name__ == "__main__":
    build_entity_embeddings()
