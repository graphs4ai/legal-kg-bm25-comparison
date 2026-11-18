"""Candidate blocking and clustering for Knowledge Fusion.

Mentions the citation parser resolves merge by canonical id; the rest are
blocked by similarity, and only ambiguous groups reach an agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from src.rag.text_utils import normalize_text

# Rows of the similarity matrix scored at once, to bound memory.
_SIMILARITY_BLOCK = 512


class UnionFind:
    """Disjoint-set over mention keys, with path compression."""

    def __init__(self):
        self.parent: dict = {}

    def find(self, item):
        """Return the representative of an item's set, compressing the path."""

        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left, right) -> bool:
        """Merge two sets; True when they were distinct."""

        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True

    def groups(self) -> dict:
        """Return representative -> members."""

        out: dict = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return out


@dataclass
class MergeCandidate:
    """A group of mention keys that may denote the same entity.

    Attributes:
        score (float): Similarity of the group; 1.0 for citations.
        reason (str): ``citation`` or ``similarity``.
    """

    keys: list = field(default_factory=list)
    score: float = 0.0
    reason: str = "similarity"


# Text naming an article or an act; skips parser calls on unrelated mentions.
_CITATION_HINT = re.compile(
    r"\bart(?:igo)?s?\b|\blei\b|\bdecreto\b|\bconstitui|\bc[oó]digo\b|"
    r"\bestatuto\b",
    re.IGNORECASE,
)


def looks_like_citation(text: str) -> bool:
    """Whether a mention names an article or an act."""

    return bool(_CITATION_HINT.search(str(text or "")))


def route_citations(
    mentions: dict,
    registry,
    source_acts: dict | None = None,
) -> tuple[dict, list[MergeCandidate]]:
    """Group mentions that resolve to the same citation.

    Args:
        mentions (dict): Mention key -> group dict with a ``name``.
        registry: ``ActRegistry`` used to resolve citations.
        source_acts (dict | None): Mention key -> act id the mention was
            found in, used to resolve bare article references.

    Returns:
        tuple[dict, list[MergeCandidate]]: Mention key -> canonical id, and one
            candidate per canonical id with more than one mention.
    """

    from src.data.laws_citations import canonical_mention_id

    source_acts = source_acts or {}
    resolved: dict = {}
    by_canonical: dict = {}

    for key, group in mentions.items():
        name = group.get("name", "")
        if not looks_like_citation(name):
            continue
        canonical = canonical_mention_id(
            name, registry, source_acts.get(key))
        if canonical:
            resolved[key] = canonical
            by_canonical.setdefault(canonical, []).append(key)

    candidates = [
        MergeCandidate(keys=keys, score=1.0, reason="citation")
        for keys in by_canonical.values() if len(keys) > 1
    ]
    return resolved, candidates


_ENCODER = None


def _encoder(device: str | None = None):
    """Return the cached blocking encoder, or None without sentence-transformers.

    A named ``device`` reloads unless the cached encoder is already on it.
    """

    global _ENCODER
    if _ENCODER is not None:
        if device is None or str(_ENCODER.device).startswith(device):
            return _ENCODER
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    from src.kg.config import KF_ENCODER_MODEL
    _ENCODER = SentenceTransformer(KF_ENCODER_MODEL, device=device)
    return _ENCODER


def _embedding_pairs(names: list[str], threshold: float):
    """Return (i, j, cosine) pairs above a threshold, or None without an encoder."""

    try:
        import numpy as np
    except ImportError:
        return None

    encoder = _encoder()
    if encoder is None:
        return None

    texts = [f"query: {normalize_text(n)}" for n in names]

    def _encode(model):
        return np.asarray(model.encode(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ))

    try:
        vectors = _encode(encoder)
    except Exception as exc:
        # A local LLM server may still hold the GPU; fall back to CPU.
        if "out of memory" not in str(exc).lower():
            raise
        logger.warning(
            "KF encoder out of GPU memory, falling back to CPU: {}",
            str(exc).split(".")[0])
        cpu_encoder = _encoder(device="cpu")
        vectors = _encode(cpu_encoder)

    pairs = []
    for start in range(0, len(names), _SIMILARITY_BLOCK):
        block = vectors[start:start + _SIMILARITY_BLOCK] @ vectors.T
        rows, cols = np.nonzero(block >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            i = start + row
            if i < col:
                pairs.append((i, col, float(block[row, col])))
    return pairs


def _lexical_pairs(names: list[str], threshold: float):
    """Pairs above a character-similarity threshold, as an embedding fallback.

    Unsafe on citations ("art. 4o" vs "art. 7o"), which are routed before this.
    """

    from difflib import SequenceMatcher

    normalized = [normalize_text(n) for n in names]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = SequenceMatcher(
                None, normalized[i], normalized[j]).ratio()
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs


# Portuguese number words -> numerals, so "cinco dias" and "5 dias" compare equal.
_NUMBER_WORDS = {
    word: str(value) for words, value in [
        (("zero",), 0), (("um", "uma"), 1), (("dois", "duas"), 2),
        (("tres",), 3), (("quatro",), 4), (("cinco",), 5), (("seis",), 6),
        (("sete",), 7), (("oito",), 8), (("nove",), 9), (("dez",), 10),
        (("onze",), 11), (("doze",), 12), (("treze",), 13),
        (("catorze", "quatorze"), 14), (("quinze",), 15),
        (("dezesseis",), 16), (("dezessete",), 17), (("dezoito",), 18),
        (("dezenove",), 19), (("vinte",), 20), (("trinta",), 30),
        (("quarenta",), 40), (("cinquenta",), 50), (("sessenta",), 60),
        (("setenta",), 70), (("oitenta",), 80), (("noventa",), 90),
        (("cem", "cento"), 100), (("duzentos", "duzentas"), 200),
        (("trezentos", "trezentas"), 300), (("quatrocentos", "quatrocentas"), 400),
        (("quinhentos", "quinhentas"), 500), (("seiscentos", "seiscentas"), 600),
        (("setecentos", "setecentas"), 700), (("oitocentos", "oitocentas"), 800),
        (("novecentos", "novecentas"), 900), (("mil",), 1000),
    ] for word in words
}


def _digit_sequences(text: str) -> set[str]:
    """Return the distinct number strings in ``text``, numeral or spelled out."""

    found = set(re.findall(r"\d+", text))
    for token in re.findall(r"[a-zà-ÿ]+", normalize_text(text)):
        if token in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[token])
    return found


def block_by_similarity(
    mentions: dict,
    skip: set | None = None,
    auto_threshold: float = 0.95,
    review_threshold: float = 0.82,
    use_embeddings: bool = True,
    max_cluster_size: int = 15,
    max_review_clusters: int = 0,
) -> tuple[list[MergeCandidate], list[MergeCandidate], str]:
    """Group similar mentions of the same entity type into merge candidates.

    Args:
        mentions (dict): Mention key -> group dict with ``name`` and ``type``.
        skip (set | None): Mention keys already resolved as citations.
        auto_threshold (float): At or above this, merge without asking.
        review_threshold (float): Between this and ``auto_threshold``, ask an
            agent.
        max_cluster_size (int): Largest review cluster (0 = no limit).
        max_review_clusters (int): Review clusters kept, highest-scoring
            first (0 = no ceiling).

    Returns:
        tuple: (auto-merge candidates, review candidates, method used).
    """

    skip = skip or set()
    by_type: dict = {}
    for key, group in mentions.items():
        if key in skip:
            continue
        by_type.setdefault(group.get("type", ""), []).append(key)

    auto: list[MergeCandidate] = []
    review: list[MergeCandidate] = []
    method = "lexical"

    for keys in by_type.values():
        if len(keys) < 2:
            continue
        names = [mentions[k]["name"] for k in keys]
        digits = [_digit_sequences(name) for name in names]

        pairs = None
        if use_embeddings:
            pairs = _embedding_pairs(names, review_threshold)
            if pairs is not None:
                method = "embedding"
        if pairs is None:
            pairs = _lexical_pairs(names, review_threshold)

        # Confident pairs merge transitively; the rest become review groups.
        # A pair whose names carry different numbers is never auto-merged.
        union = UnionFind()
        ambiguous: dict = {}
        for i, j, score in pairs:
            same_numbers = not (digits[i] ^ digits[j])
            if score >= auto_threshold and same_numbers:
                union.union(keys[i], keys[j])
            else:
                ambiguous.setdefault(keys[i], []).append((keys[j], score))

        for members in union.groups().values():
            if len(members) > 1:
                auto.append(MergeCandidate(
                    keys=members, score=auto_threshold, reason="similarity"))

        # One candidate per connected component of the review band.
        review_union = UnionFind()
        scores: dict[tuple, float] = {}
        for anchor, others in ambiguous.items():
            for other, score in others:
                review_union.union(anchor, other)
                scores[(anchor, other)] = score

        for members in review_union.groups().values():
            if len(members) < 2:
                continue
            member_set = set(members)
            best = max(
                (s for (a, b), s in scores.items()
                 if a in member_set and b in member_set),
                default=review_threshold,
            )
            for chunk in _chunked(members, max_cluster_size):
                if len(chunk) > 1:
                    review.append(MergeCandidate(
                        keys=chunk, score=best, reason="similarity"))

    if max_review_clusters and len(review) > max_review_clusters:
        review.sort(key=lambda candidate: -candidate.score)
        logger.warning(
            "KF blocking: {} review clusters exceeds the ceiling of {}; "
            "keeping the highest-scoring ones. The review threshold is "
            "probably too low for this corpus.",
            len(review), max_review_clusters,
        )
        review = review[:max_review_clusters]

    return auto, review, method


def _chunked(items: list, size: int) -> list[list]:
    """Split a list into even consecutive chunks of at most ``size`` (0 = no split)."""

    if size <= 0 or len(items) <= size:
        return [list(items)]

    # Even chunks avoid a single-mention remainder, which would be dropped.
    count = -(-len(items) // size)
    base, extra = divmod(len(items), count)
    chunks, start = [], 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        chunks.append(list(items[start:end]))
        start = end
    return chunks
