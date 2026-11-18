"""Flat BM25 or dense (E5) retrieval over the law corpus.

On an article corpus every wording is indexed and the one served is the
wording in force at the question's cutoff, each article at most once.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.laws_versioning import (
    load_article_versions,
    load_unit_versions,
    wording_at,
)
from src.rag.text_utils import apply_context_budget, normalize_text, tokenize

# E5 encoders expect asymmetric query/passage prefixes.
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class LawRetriever:
    """Retrieve the top-K law articles (or chunks) for a query."""

    def __init__(
        self,
        csv_path: str | Path,
        method: str = "bm25",
        encoder_name: str = "intfloat/multilingual-e5-large",
        text_column: str = "lei_text_llm",
        cache_dir: str | Path = "results/retrieval/flat/v5",
        query_mode: str = "question_choices",
        temporal_filter: bool = True,
        unit: str = "unit",
        article_ids=None,
        max_context_tokens: int | None = None,
    ):
        """Load the corpus and build the BM25 or dense index.

        Args:
            csv_path (str | Path): Article corpus (or chunk corpus).
            cache_dir (str | Path): Folder for the dense embedding cache;
                must match the corpus version.
            query_mode (str): "question" (stem) or "question_choices".
            temporal_filter (bool): Year filter for a chunk corpus; an article
                corpus always serves the wording in force at the cutoff.
            unit (str): "unit" splits an article into caput and provisions;
                "article" serves it whole.
            article_ids (Iterable[str] | None): Keep only these articles.
            max_context_tokens (int | None): Context budget; None keeps all.
        """

        if unit not in ("unit", "article"):
            raise ValueError(f"Unknown unit: {unit!r} (use 'unit' or 'article')")
        if method not in ("bm25", "dense"):
            raise ValueError(f"Unknown method: {method!r} (use 'bm25' or 'dense')")
        if query_mode not in ("question", "question_choices"):
            raise ValueError(f"Unknown query_mode: {query_mode!r}")

        self.method = method
        self.text_column = text_column
        self.encoder_name = encoder_name
        self.cache_dir = Path(cache_dir)
        self.query_mode = query_mode
        self.temporal_filter = temporal_filter
        self.unit = unit
        self.article_ids = None if article_ids is None else set(article_ids)
        self.max_context_tokens = max_context_tokens

        # Article corpus only: article_id -> version chain.
        self._articles: dict[str, list[dict]] | None = None
        self.records = self._load_records(Path(csv_path))
        logger.info(
            "Loaded {} chunks from {} for {} retrieval",
            len(self.records),
            csv_path,
            method,
        )

        self._bm25 = None
        self._embeddings = None
        self._encoder = None

        if method == "bm25":
            self._build_bm25()
        else:
            self._build_dense()

    def _load_article_records(self, csv_path: Path) -> list[dict]:
        """Read an article corpus into one indexed record per wording."""

        self._articles = (
            load_unit_versions(csv_path) if self.unit == "unit"
            else load_article_versions(csv_path)
        )
        if self.article_ids is not None:
            self._articles = {
                key: versions for key, versions in self._articles.items()
                if key.split("#", 1)[0] in self.article_ids
            }
        records = []
        for key, versions in self._articles.items():
            for version in versions:
                records.append({
                    **version,
                    "unit_id": version.get("unit_id", key),
                    "doc_id": key,
                    "vintage_year": version["valid_from"],
                    "text": version["display_text"],
                })
        return records

    def _load_records(self, csv_path: Path) -> list[dict]:
        """Read the corpus CSV into a flat list of records."""

        if "article_id" in pd.read_csv(csv_path, nrows=0).columns:
            return self._load_article_records(csv_path)

        df = pd.read_csv(csv_path)

        # Vintage is the last amendment year, falling back to the enactment year.
        df["ano_ultima_alteracao"] = pd.to_numeric(
            df["ano_ultima_alteracao"], errors="coerce"
        )
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df["vintage_year"] = df["ano_ultima_alteracao"].fillna(df["ano"])

        records = []
        for _, row in df.iterrows():
            records.append(
                {
                    # doc_id is unique; chunk_id repeats across documents.
                    "doc_id": row.get("doc_id"),
                    "chunk_id": row.get("chunk_id"),
                    "lei_label": row.get("lei_label"),
                    "num_lei": row.get("num_lei"),
                    "ano": None if pd.isna(row.get("ano")) else int(row["ano"]),
                    "ano_ultima_alteracao": (
                        None
                        if pd.isna(row.get("ano_ultima_alteracao"))
                        else int(row["ano_ultima_alteracao"])
                    ),
                    "vintage_year": (
                        None
                        if pd.isna(row.get("vintage_year"))
                        else int(row["vintage_year"])
                    ),
                    "area": row.get("area"),
                    "path": row.get("path"),
                    "text": str(row.get(self.text_column) or ""),
                }
            )

        return records

    def _build_bm25(self) -> None:
        """Tokenize every record and build the BM25 index."""

        from src.rag.bm25 import FastBM25

        tokenized_corpus = [tokenize(rec["text"]) for rec in self.records]
        self._bm25 = FastBM25(tokenized_corpus)

    def _cache_path(self) -> Path:
        """Dense cache path, keyed by a hash of the encoder, column and every text."""

        hasher = hashlib.sha1(
            f"{self.encoder_name}|{self.text_column}".encode("utf-8"))
        for rec in self.records:
            hasher.update(b"\x00" + rec["text"].encode("utf-8"))
        digest = hasher.hexdigest()[:16]
        safe_encoder = self.encoder_name.replace("/", "_")
        return self.cache_dir / f"{safe_encoder}_{self.text_column}_{digest}.npy"

    def _build_dense(self) -> None:
        """Encode every record, L2-normalized, through the disk cache."""

        import numpy as np
        from sentence_transformers import SentenceTransformer

        cache_path = self._cache_path()

        self._encoder = SentenceTransformer(self.encoder_name)

        if cache_path.exists():
            logger.info("Loading cached embeddings: {}", cache_path)
            self._embeddings = np.load(cache_path)
            return

        logger.info(
            "Encoding {} chunks with {} (first run; will cache)",
            len(self.records),
            self.encoder_name,
        )
        passages = [
            _E5_PASSAGE_PREFIX + normalize_text(rec["text"])
            for rec in self.records
        ]
        embeddings = self._encoder.encode(
            passages,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Cached embeddings to {}", cache_path)
        self._embeddings = embeddings

    def build_query(self, item: dict, query_mode: str | None = None) -> str:
        """Query text from an OAB item: the stem, plus the choices in "question_choices" mode."""

        mode = query_mode or self.query_mode
        question = item.get("question", "")

        if mode == "question":
            return question

        choices = item.get("choices", {}) or {}
        choices_text = " ".join(str(v) for v in choices.values())
        return f"{question} {choices_text}".strip()

    def retrieve(
        self,
        query: str,
        exam_year: int | None = None,
        top_k: int = 5,
        cutoff: date | None = None,
    ) -> list[dict]:
        """Top-K records in force at the question's cutoff, best first, with a "score".

        ``cutoff`` picks the wording in force that day (falls back to
        ``exam_year``); on a chunk corpus ``exam_year`` drives the year filter.
        """

        scores = self._score(query)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict] = []
        if self._articles is not None:
            for idx, rec, in_force in self._in_force_articles(
                    order, exam_year, cutoff):
                result = {**rec, **in_force, "text": in_force["display_text"]}
                result["score"] = float(scores[idx])
                results.append(result)
                if len(results) >= top_k:
                    break
            order = []

        for idx in order:
            rec = self.records[idx]

            if self.temporal_filter and exam_year is not None:
                vintage = rec["vintage_year"]
                # Unknown vintage cannot be shown in force, so it is dropped.
                if vintage is None or vintage > exam_year:
                    continue

            result = dict(rec)
            result["score"] = float(scores[idx])
            results.append(result)

            if len(results) >= top_k:
                break

        return apply_context_budget(results, self.max_context_tokens)

    def _in_force_articles(self, order, exam_year, cutoff):
        """Yield (index, record, wording in force) once per article along a
        ranking; articles with no wording in force are skipped."""

        seen: set[str] = set()
        for idx in order:
            rec = self.records[idx]
            key = rec["doc_id"]
            if key in seen:
                continue
            seen.add(key)
            in_force = wording_at(self._articles[key], exam_year, cutoff)
            if in_force is not None:
                yield idx, rec, in_force

    def rank_articles(
        self, query: str, exam_year: int | None = None, top_n: int = 50,
        cutoff: date | None = None,
    ) -> list[tuple[str, float]]:
        """(article_id, score) pairs ``retrieve`` serves from, best first; article corpus only."""

        if self._articles is None:
            raise ValueError("rank_articles needs an article corpus")
        scores = self._score(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = []
        for idx, rec, _ in self._in_force_articles(order, exam_year, cutoff):
            ranked.append((rec["doc_id"], float(scores[idx])))
            if len(ranked) >= top_n:
                break
        return ranked

    def _score(self, query: str):
        """Score of every record against the query (BM25, or cosine for dense)."""

        if self.method == "bm25":
            return self._bm25.get_scores(tokenize(query))

        import numpy as np

        query_vec = self._encoder.encode(
            [_E5_QUERY_PREFIX + normalize_text(query)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return np.asarray(self._embeddings) @ query_vec

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """Number the retrieved texts into the ``{context}`` prompt block."""

        blocks = [
            f"[{i}] {chunk['text'].strip()}"
            for i, chunk in enumerate(chunks, start=1)
        ]
        return "\n\n".join(blocks)
