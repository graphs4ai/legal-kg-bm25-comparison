"""BM25 over an inverted index: the same scores as ``BM25Okapi.get_scores``,
touching only the documents that hold a query term.
"""

from __future__ import annotations


class FastBM25:
    """Drop-in for ``BM25Okapi.get_scores`` backed by term postings."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5,
                 b: float = 0.75, epsilon: float = 0.25):
        """Precompute per-term postings from a tokenized corpus; k1, b and
        epsilon as in ``BM25Okapi``."""

        import numpy as np
        from rank_bm25 import BM25Okapi

        base = BM25Okapi(corpus, k1=k1, b=b, epsilon=epsilon)
        self.corpus_size = base.corpus_size
        doc_len = np.array(base.doc_len)

        docs: dict[str, list[int]] = {}
        freqs: dict[str, list[int]] = {}
        for index, doc in enumerate(base.doc_freqs):
            for term, tf in doc.items():
                docs.setdefault(term, []).append(index)
                freqs.setdefault(term, []).append(tf)

        # Same operand order as BM25Okapi.get_scores, so scores are bit-identical.
        self._postings = {}
        for term, indices in docs.items():
            idx = np.array(indices)
            tf = np.array(freqs[term])
            weight = (base.idf.get(term) or 0) * (
                tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[idx] / base.avgdl)))
            self._postings[term] = (idx, weight)

    def get_scores(self, query: list[str]):
        """BM25 score of every document, in corpus order; repeated tokens count again."""

        import numpy as np

        score = np.zeros(self.corpus_size)
        for term in query:
            hit = self._postings.get(term)
            if hit is not None:
                score[hit[0]] += hit[1]
        return score
