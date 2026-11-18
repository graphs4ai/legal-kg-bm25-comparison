"""File-backed property graph: JSONL nodes/edges with an in-memory adjacency index.

Every node and edge carries provenance to its source articles. ``export_cypher``
writes a Neo4j load script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.rag.text_utils import normalize_text

# Attribute keys that hold a validity interval.
_WINDOW_KEYS = ("valid_from", "valid_until", "valid_from_date", "valid_until_date")


@dataclass
class Node:
    """A graph node.

    Attributes:
        node_id (str): Unique id (e.g. ``act:cf_1988``, ``article:cf_1988:art_8``).
        type (str): Ontology entity-type name.
        provenance (list[dict]): Source records.
    """

    node_id: str
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


@dataclass
class Edge:
    """A directed, typed graph edge."""

    edge_id: str
    source: str
    relation: str
    target: str
    provenance: list[dict] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def slugify(text: str) -> str:
    """Normalize a name into a lowercase, accent-free, underscore slug."""

    normalized = normalize_text(text)
    slug = "".join(c if c.isalnum() else "_" for c in normalized)
    return "_".join(part for part in slug.split("_") if part)


class GraphStore:
    """In-memory property graph with JSONL persistence and Cypher export."""

    def __init__(self, meta: dict | None = None):

        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.meta: dict = meta or {}
        self._adjacency: dict[str, set[str]] = {}
        self._name_index: dict[str, set[str]] = {}
        self._edge_counter = 0

    def add_node(
        self,
        node_id: str,
        name: str,
        type: str,
        aliases: list[str] | None = None,
        provenance: list[dict] | None = None,
        attrs: dict | None = None,
    ) -> Node:
        """Add a node, or append new aliases and provenance to an existing one."""

        if node_id in self.nodes:
            node = self.nodes[node_id]
            for alias in (aliases or []):
                if alias not in node.aliases:
                    node.aliases.append(alias)
            for record in (provenance or []):
                if record not in node.provenance:
                    node.provenance.append(record)
            return node

        node = Node(
            node_id=node_id,
            name=name,
            type=type,
            aliases=list(aliases or []),
            provenance=list(provenance or []),
            attrs=dict(attrs or {}),
        )
        self.nodes[node_id] = node
        self._adjacency.setdefault(node_id, set())
        self._index_name(node)
        return node

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        provenance: list[dict] | None = None,
        attrs: dict | None = None,
    ) -> Edge:
        """Add an edge, or extend an identical existing one.

        A duplicate's provenance is appended and its validity window widened
        to cover both.

        Raises:
            KeyError: If source or target node does not exist.
        """

        if source not in self.nodes:
            raise KeyError(f"Unknown source node: {source}")
        if target not in self.nodes:
            raise KeyError(f"Unknown target node: {target}")

        for edge_id in self._adjacency[source]:
            edge = self.edges[edge_id]
            if (edge.source == source and edge.target == target
                    and edge.relation == relation):
                for record in (provenance or []):
                    if record not in edge.provenance:
                        edge.provenance.append(record)
                incoming = attrs or {}
                if any(key in incoming for key in _WINDOW_KEYS):
                    edge.attrs.update(union_window([edge.attrs, incoming]))
                return edge

        self._edge_counter += 1
        edge = Edge(
            edge_id=f"e{self._edge_counter:06d}",
            source=source,
            relation=relation,
            target=target,
            provenance=list(provenance or []),
            attrs=dict(attrs or {}),
        )
        self.edges[edge.edge_id] = edge
        self._adjacency[source].add(edge.edge_id)
        self._adjacency[target].add(edge.edge_id)
        return edge

    def _index_name(self, node: Node) -> None:
        """Index a node's name and aliases for lexical lookup."""
        for surface in [node.name, *node.aliases]:
            self._name_index.setdefault(
                normalize_text(surface), set()).add(node.node_id)

    def _unindex_name(self, node: Node) -> None:
        """Remove a node from the name index."""
        for surface in [node.name, *node.aliases]:
            key = normalize_text(surface)
            if key in self._name_index:
                self._name_index[key].discard(node.node_id)
                if not self._name_index[key]:
                    del self._name_index[key]

    def find_by_name(self, name: str) -> list[Node]:
        """Find nodes whose name or alias matches (normalized equality)."""

        ids = self._name_index.get(normalize_text(name), set())
        return [self.nodes[node_id] for node_id in sorted(ids)]

    def degree(self, node_id: str) -> int:
        """Return how many edges touch a node (0 if unknown)."""

        return len(self._adjacency.get(node_id, ()))

    def neighbors(self, node_id: str) -> list[Edge]:
        """Return all edges touching a node, in and out."""

        return [self.edges[eid] for eid in sorted(self._adjacency.get(node_id, set()))]

    def stats(self) -> dict:
        """Return node/edge counts, by type and by relation."""

        backbone_types = {"NormativeAct", "Article"}
        backbone_relations = {"contains", "refers_to", "amends", "revokes"}
        by_type: dict[str, int] = {}
        for node in self.nodes.values():
            by_type[node.type] = by_type.get(node.type, 0) + 1
        by_relation: dict[str, int] = {}
        for edge in self.edges.values():
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        # Articles with at least one outgoing semantic edge.
        enriched = {
            edge.source for edge in self.edges.values()
            if edge.relation not in backbone_relations
            and edge.source.startswith("article:")
        }
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "semantic_nodes": sum(
                count for node_type, count in by_type.items()
                if node_type not in backbone_types
            ),
            "semantic_edges": sum(
                count for relation, count in by_relation.items()
                if relation not in backbone_relations
            ),
            "enriched_articles": len(enriched),
            "nodes_by_type": by_type,
            "edges_by_relation": by_relation,
        }

    def save(self, graph_dir: str | Path) -> None:
        """Write the graph to ``nodes.jsonl``, ``edges.jsonl``, ``meta.json``."""

        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)

        with (graph_dir / "nodes.jsonl").open("w", encoding="utf-8") as f:
            for node in self.nodes.values():
                f.write(json.dumps(asdict(node), ensure_ascii=False) + "\n")
        with (graph_dir / "edges.jsonl").open("w", encoding="utf-8") as f:
            for edge in self.edges.values():
                f.write(json.dumps(asdict(edge), ensure_ascii=False) + "\n")
        with (graph_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {**self.meta, **self.stats()}, f, ensure_ascii=False, indent=2
            )

    @classmethod
    def load(cls, graph_dir: str | Path) -> "GraphStore":
        """Load a graph saved by ``save``."""

        graph_dir = Path(graph_dir)
        meta_path = graph_dir / "meta.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists() else {}
        )
        store = cls(meta=meta)

        with (graph_dir / "nodes.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                node = Node(**json.loads(line))
                store.nodes[node.node_id] = node
                store._adjacency.setdefault(node.node_id, set())
                store._index_name(node)

        with (graph_dir / "edges.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                edge = Edge(**json.loads(line))
                store.edges[edge.edge_id] = edge
                store._adjacency[edge.source].add(edge.edge_id)
                store._adjacency[edge.target].add(edge.edge_id)
                number = int(edge.edge_id.lstrip("e"))
                store._edge_counter = max(store._edge_counter, number)

        return store

    def export_cypher(self, path: str | Path) -> None:
        """Write a Cypher script that loads the graph into Neo4j."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _props(data: dict) -> str:
            return json.dumps(data, ensure_ascii=False)

        lines = [
            "// Generated by src.kg.graph_store.GraphStore.export_cypher",
            "CREATE CONSTRAINT node_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.node_id IS UNIQUE;",
        ]
        for node in self.nodes.values():
            props = {
                "node_id": node.node_id,
                "name": node.name,
                "aliases": node.aliases,
                "provenance": json.dumps(node.provenance, ensure_ascii=False),
                **node.attrs,
            }
            label = "".join(
                part.capitalize() for part in slugify(node.type).split("_")
            ) or "Entity"
            lines.append(f"MERGE (n:Entity:{label} {{node_id: "
                         f"{json.dumps(node.node_id)}}}) SET n += {_props(props)};")
        for edge in self.edges.values():
            rel = slugify(edge.relation).upper() or "RELATED_TO"
            props = {
                "provenance": json.dumps(edge.provenance, ensure_ascii=False),
                **edge.attrs,
            }
            lines.append(
                f"MATCH (a:Entity {{node_id: {json.dumps(edge.source)}}}), "
                f"(b:Entity {{node_id: {json.dumps(edge.target)}}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r += {_props(props)};"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_backbone(articles, meta: dict | None = None) -> GraphStore:
    """Build the structural backbone from the article corpus, with no model call.

    One ``NormativeAct`` node per act (rank in ``hierarchy_level``, 0 =
    Constituição), one ``Article`` node per article, ``contains`` edges, and the
    ``refers_to``/``amends``/``revokes`` edges parsed from citations.

    Args:
        articles (pd.DataFrame): Article corpus, one row per wording.
    """

    import pandas as pd

    def _opt_int(value):
        return None if value is None or pd.isna(value) else int(value)

    def _opt_str(value):
        return None if value is None or pd.isna(value) else str(value)

    dated = "valid_from_date" in articles.columns

    def _window_of(row) -> dict:
        window = {
            "valid_from": _opt_int(row.get("valid_from")),
            "valid_until": _opt_int(row.get("valid_until")),
        }
        if dated:
            window["valid_from_date"] = _opt_str(row.get("valid_from_date"))
            window["valid_until_date"] = _opt_str(row.get("valid_until_date"))
        return window

    store = GraphStore(meta=meta)
    seen_acts: set[str] = set()
    act_windows: dict[str, list[dict]] = {}

    # One node per article, spanning all its wordings.
    for article_id, rows in articles.groupby("article_id", sort=False):
        versions = [
            {
                **_window_of(r),
                "version_year": _opt_int(r.get("version_year")),
                "version_seq": _opt_int(r.get("version_seq")),
            }
            for _, r in rows.iterrows()
        ]
        versions.sort(key=lambda v: (v["valid_from"] is None, v["valid_from"]))
        window = union_window(versions)

        row = rows.iloc[-1]
        act_id = row["act_id"]
        provenance = [{"article_id": article_id, "act_id": act_id}]
        act_windows.setdefault(act_id, []).append(window)

        act_node_id = f"act:{act_id}"
        if act_node_id not in seen_acts:
            seen_acts.add(act_node_id)
            store.add_node(
                node_id=act_node_id,
                name=row["lei_label"],
                type="NormativeAct",
                provenance=provenance,
                attrs={
                    "act_id": act_id,
                    "tipo": _opt_str(row.get("tipo")),
                    "hierarchy_level": _opt_int(row.get("hierarchy_level")),
                    "ano": _opt_int(row.get("ano")),
                },
            )

        store.add_node(
            node_id=f"article:{article_id}",
            name=f"Art. {row['article_number']} — {row['lei_label']}",
            type="Article",
            provenance=provenance,
            attrs={
                "article_number": _opt_str(row.get("article_number")),
                "container": _opt_str(row.get("container")),
                "path": _opt_str(row.get("path")),
                "version_year": _opt_int(row.get("version_year")),
                "tokens": _opt_int(row.get("tokens")),
                "is_boilerplate": bool(row.get("is_boilerplate", False)),
                **window,
                "versions": versions,
            },
        )
        store.add_edge(
            act_node_id, "contains", f"article:{article_id}",
            provenance=provenance,
            attrs=dict(window),
        )

    # Citation edges; a target absent from the corpus is dropped.
    for _, row in articles.iterrows():
        raw = row.get("citations")
        if not raw or (isinstance(raw, float) and pd.isna(raw)):
            continue

        source_id = f"article:{row['article_id']}"
        provenance = [
            {"article_id": row["article_id"], "act_id": row["act_id"]}
        ]
        # A citation is only asserted by the wording that carries it.
        window = _window_of(row)

        for citation in json.loads(raw):
            target_act = citation.get("target_act_id")
            if not target_act:
                continue
            relation = citation.get("relation", "refers_to")
            numbers = citation.get("article_numbers") or []
            container = citation.get("container")
            prefix = f"{target_act}:{container}" if container else target_act
            targets = (
                [f"article:{prefix}:art_{n}" for n in numbers]
                if numbers else [f"act:{target_act}"]
            )
            for target_id in targets:
                if target_id in store.nodes and target_id != source_id:
                    store.add_edge(source_id, relation, target_id,
                                   provenance=provenance, attrs=dict(window))

    for act_id, windows in act_windows.items():
        node = store.nodes.get(f"act:{act_id}")
        if node is None:
            continue
        node.attrs.update(union_window(windows))

    return store


def union_window(windows: list[dict]) -> dict:
    """Smallest interval covering every given interval, on years and ISO dates.

    An open end (None) on any input stays open; a missing start date on any
    input leaves the start date None so the year bound is used instead.
    """

    starts = [w.get("valid_from") for w in windows if w.get("valid_from") is not None]
    ends = [w.get("valid_until") for w in windows]
    out: dict = {
        "valid_from": min(starts) if starts else None,
        "valid_until": (
            None if not ends or any(e is None for e in ends) else max(ends)),
    }
    if any("valid_from_date" in w or "valid_until_date" in w for w in windows):
        start_dates = [w.get("valid_from_date") for w in windows]
        end_dates = [w.get("valid_until_date") for w in windows]
        out["valid_from_date"] = (
            None if not start_dates or any(s is None for s in start_dates)
            else min(start_dates))
        out["valid_until_date"] = (
            None if not end_dates or any(e is None for e in end_dates)
            else max(end_dates))
    return out
