"""Personalized PageRank over the knowledge graph, for graph-ranked retrieval.

The walk starts from the BM25 article ranking and the entity seeds and spreads
over article-entity links, entity relations, citations and statute structure
(neighbouring articles, shared section). Only nodes and edges in force at the
question's cutoff take part; the adjacency is cached per (weights, cutoff).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse

from src.kg.graph_store import GraphStore

# Edge types of the walk graph.
MENTIONS, SEMANTIC, CITATION, ADJACENT, ADJACENT2, SECTION = range(6)
_CITATION_RELATIONS = {"refers_to", "amends", "revokes"}
_SKIPPED_RELATIONS = {"contains"}
_SKIPPED_TYPES = {"NormativeAct"}
_OPEN_LO, _OPEN_HI = -(10 ** 9), 10 ** 9


def _interval(attrs: dict) -> tuple[int, int]:
    """Validity of a node or edge as day ordinals; a year-only bound spans
    its whole year, an absent bound is open."""

    start, end = attrs.get("valid_from_date"), attrs.get("valid_until_date")
    year_from, year_until = attrs.get("valid_from"), attrs.get("valid_until")
    lo = (date.fromisoformat(start).toordinal() if start
          else date(int(year_from), 1, 1).toordinal() if year_from is not None
          else _OPEN_LO)
    hi = (date.fromisoformat(end).toordinal() if end
          else date(int(year_until), 12, 31).toordinal() if year_until is not None
          else _OPEN_HI)
    return lo, hi


def _window(exam_year: int | None, cutoff: date | None) -> tuple[int, int] | None:
    """The span an element must overlap to be in force: the cutoff day, or
    the exam year without one; None disables the filter."""

    if cutoff is not None:
        return cutoff.toordinal(), cutoff.toordinal()
    if exam_year is not None:
        return date(exam_year, 1, 1).toordinal(), date(exam_year, 12, 31).toordinal()
    return None


def _section_id(row) -> str:
    """Walk node of an article's section, within its act and container."""

    return f"section:{row.act_id}:{row.container}:{row.path}"


class GraphWalk:
    """The walk graph of one knowledge graph and article corpus."""

    def __init__(self, store: GraphStore, articles_csv):
        """Build the walk graph; ``articles_csv`` gives article order and sections."""

        ids = [n for n, node in store.nodes.items() if node.type not in _SKIPPED_TYPES]
        articles = [n for n in ids if n.startswith("article:")]
        wanted = {"article_id", "act_id", "container", "seq", "path"}
        corpus = pd.read_csv(articles_csv, low_memory=False,
                             usecols=lambda column: column in wanted)
        # A corpus without order or headings gives no structural edges.
        corpus = corpus.assign(**{
            "container": corpus.get("container", "main"),
            "seq": corpus["seq"] if "seq" in corpus else range(len(corpus)),
            "path": corpus.get("path", None),
        })
        corpus = corpus[corpus["article_id"].isin({a[len("article:"):] for a in articles})]
        # Sections are nodes, so a shared section is one edge per article, not a clique.
        latest = corpus.sort_values("seq").groupby("article_id").last()
        # An article's place in its act is its first wording's position.
        latest["seq"] = corpus.groupby("article_id")["seq"].min()
        sections = sorted({
            _section_id(row)
            for row in latest.itertuples() if isinstance(row.path, str) and row.path})
        self.nodes = ids + sections
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.is_article = np.array([n.startswith("article:") for n in self.nodes])

        lo, hi = np.full(len(self.nodes), _OPEN_LO), np.full(len(self.nodes), _OPEN_HI)
        for n in ids:
            lo[self.index[n]], hi[self.index[n]] = _interval(store.nodes[n].attrs)
        self._node_lo, self._node_hi = lo, hi

        src, dst, kind, base, e_lo, e_hi = [], [], [], [], [], []

        def add(a, b, edge_kind, weight=1.0, interval=(_OPEN_LO, _OPEN_HI)):
            src.append(self.index[a]); dst.append(self.index[b])
            kind.append(edge_kind); base.append(weight)
            e_lo.append(interval[0]); e_hi.append(interval[1])

        # One article-entity edge per pair, spanning the union of its relations' intervals.
        pairs: dict[tuple[str, str], tuple[int, int]] = {}
        for edge in store.edges.values():
            if (edge.relation in _SKIPPED_RELATIONS or edge.source not in self.index
                    or edge.target not in self.index):
                continue
            interval = _interval(edge.attrs)
            if edge.relation in _CITATION_RELATIONS:
                if edge.target.startswith("article:"):
                    add(edge.source, edge.target, CITATION, 1.0, interval)
            elif edge.source.startswith("article:") != edge.target.startswith("article:"):
                article, entity = ((edge.source, edge.target)
                                   if edge.source.startswith("article:")
                                   else (edge.target, edge.source))
                lo, hi = pairs.get((article, entity), (_OPEN_HI, _OPEN_LO))
                pairs[(article, entity)] = (min(lo, interval[0]), max(hi, interval[1]))
            elif not edge.source.startswith("article:"):
                add(edge.source, edge.target, SEMANTIC, 1.0, interval)
        articles_per_entity: dict[str, int] = {}
        for _, entity in pairs:
            articles_per_entity[entity] = articles_per_entity.get(entity, 0) + 1
        for (article, entity), interval in pairs.items():
            # Rare entities carry more of the walk than hub entities.
            add(article, entity, MENTIONS,
                1.0 / math.log(math.e + articles_per_entity[entity]), interval)

        for (_, _), group in latest.reset_index().groupby(["act_id", "container"]):
            ordered = [f"article:{a}" for a in group.sort_values("seq")["article_id"]]
            for i, article in enumerate(ordered):
                if i + 1 < len(ordered):
                    add(article, ordered[i + 1], ADJACENT)
                if i + 2 < len(ordered):
                    add(article, ordered[i + 2], ADJACENT2)
        for row in latest.itertuples():
            if isinstance(row.path, str) and row.path:
                add(f"article:{row.Index}", _section_id(row), SECTION)

        self._src, self._dst = np.array(src), np.array(dst)
        self._kind, self._base = np.array(kind), np.array(base)
        self._edge_lo, self._edge_hi = np.array(e_lo), np.array(e_hi)
        self._matrices: dict = {}

    def _transition(self, weights: tuple[float, ...], window) -> tuple:
        """Transposed row-stochastic matrix of the in-force graph, plus the
        node mask and the weighted degree of every node."""

        key = (weights, window)
        if key in self._matrices:
            return self._matrices[key]
        weight = np.asarray(weights)[self._kind] * self._base
        if window is None:
            node_ok = np.ones(len(self.nodes), dtype=bool)
            edge_ok = weight > 0
        else:
            node_ok = (self._node_lo <= window[1]) & (self._node_hi >= window[0])
            edge_ok = ((weight > 0) & (self._edge_lo <= window[1])
                       & (self._edge_hi >= window[0])
                       & node_ok[self._src] & node_ok[self._dst])
        s, d, w = self._src[edge_ok], self._dst[edge_ok], weight[edge_ok]
        n = len(self.nodes)
        # Undirected walk: every edge carries mass both ways.
        matrix = sparse.csr_matrix(
            (np.concatenate([w, w]), (np.concatenate([s, d]), np.concatenate([d, s]))),
            shape=(n, n))
        out = np.asarray(matrix.sum(axis=1)).ravel()
        inverse = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
        transposed = (sparse.diags(inverse) @ matrix).T.tocsr()
        entry = (transposed, node_ok, out)
        self._matrices[key] = entry
        return entry

    def rank(self, start: dict[str, float], exam_year: int | None,
             cutoff: date | None, weights: tuple[float, ...], damping: float,
             iterations: int = 50, tolerance: float = 1e-10,
             degree_norm: float = 0.0) -> dict[str, float]:
        """Walk scores of the nodes in force, from a start distribution.

        Args:
            start (dict[str, float]): node_id -> start mass (need not sum to 1).
            exam_year (int | None): Exam year, without a cutoff.
            cutoff (date | None): Legal cutoff of the question.
            weights (tuple[float, ...]): Weight per edge type, in the order
                MENTIONS, SEMANTIC, CITATION, ADJACENT, ADJACENT2, SECTION.
            damping (float): Probability of following an edge at each step.
            iterations (int): Maximum power iterations.
            tolerance (float): L1 change at which the walk has converged.
            degree_norm (float): Exponent β; scores are divided by the
                node's weighted degree^β (nodes with no in-force edge keep
                their score). 0 returns the walk scores as they are.

        Returns:
            dict[str, float]: node_id -> score for every node with a positive
                score; empty when no start node is in force.
        """

        transposed, node_ok, degree = self._transition(
            tuple(weights), _window(exam_year, cutoff))
        dangling = degree == 0
        restart = np.zeros(len(self.nodes))
        for node_id, mass in start.items():
            i = self.index.get(node_id)
            if i is not None and node_ok[i] and mass > 0:
                restart[i] += mass
        total = restart.sum()
        if total <= 0:
            return {}
        restart /= total
        scores = restart.copy()
        for _ in range(iterations):
            # Mass at a node with no in-force edge restarts.
            step = damping * (transposed @ scores + scores[dangling].sum() * restart)
            updated = (1 - damping) * restart + step
            converged = np.abs(updated - scores).sum() < tolerance
            scores = updated
            if converged:
                break
        if degree_norm:
            scores = np.where(dangling, scores,
                              scores / np.maximum(degree, 1e-12) ** degree_norm)
        positive = np.flatnonzero(scores > 0)
        return {self.nodes[i]: float(scores[i]) for i in positive}
