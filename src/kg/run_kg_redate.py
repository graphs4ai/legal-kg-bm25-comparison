"""Re-date a finished graph against an article corpus, with no model call:
``SOURCE_BUILD`` + ``ARTICLES_CSV`` -> ``TARGET_BUILD``.

    python -m src.kg.run_kg_redate

The backbone is rebuilt from the corpus; semantic edges get day bounds from the
overlapping wordings, clipped to their years; entities get the union of their mentions.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from src.kg.config import KG_RESULTS_DIR, PROJECT_ROOT
from src.kg.graph_store import GraphStore, build_backbone, union_window

SOURCE_BUILD = "kg_v1"
TARGET_BUILD = "kg_v1_1"
ARTICLES_CSV = PROJECT_ROOT / "data" / "laws" / "articles_v5.csv"

BACKBONE = {"contains", "refers_to", "amends", "revokes"}
CITATIONS = {"refers_to", "amends", "revokes"}
STRUCTURAL_TYPES = {"NormativeAct", "Article"}


def opt(value):
    return None if value is None or pd.isna(value) else value


def wording_windows(corpus: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in corpus.itertuples(index=False):
        out.setdefault(r.article_id, []).append({
            "valid_from": None if opt(r.valid_from) is None else int(r.valid_from),
            "valid_until": None if opt(r.valid_until) is None else int(r.valid_until),
            "valid_from_date": opt(r.valid_from_date),
            "valid_until_date": opt(r.valid_until_date),
        })
    return out


def date_window(wordings, valid_from, valid_until):
    """Dates of the wordings overlapping a year interval, clipped to it."""

    overlapping = [
        w for w in wordings
        if (valid_until is None or w["valid_from"] is None
            or w["valid_from"] <= valid_until)
        and (valid_from is None or w["valid_until"] is None
             or w["valid_until"] >= valid_from)
    ]
    if not overlapping:
        return None
    union = union_window(overlapping)
    start, end = union["valid_from_date"], union["valid_until_date"]
    if valid_from is not None:
        floor = f"{valid_from:04d}-01-01"
        start = floor if start is None else max(start, floor)
    if valid_until is not None:
        ceiling = f"{valid_until:04d}-12-31"
        end = ceiling if end is None else min(end, ceiling)
    return {"valid_from_date": start, "valid_until_date": end}


def edge_dates(edge, wordings):
    """Union of the provenance windows matching a corpus wording, or None."""

    windows = []
    for prov in edge.provenance or [{}]:
        article_id = prov.get("article_id") or edge.source[len("article:"):]
        years = prov if ("valid_from" in prov or "valid_until" in prov) else edge.attrs
        window = date_window(wordings.get(article_id, []),
                             years.get("valid_from"), years.get("valid_until"))
        if window is not None:
            windows.append(window)
    return union_window(windows) if windows else None


def years_from_dates(attrs: dict) -> None:
    """Set the year bounds from the day bounds, where the day is known."""

    start, end = attrs.get("valid_from_date"), attrs.get("valid_until_date")
    if start:
        attrs["valid_from"] = int(start[:4])
    if end:
        attrs["valid_until"] = int(end[:4])
    elif "valid_until_date" in attrs:
        attrs["valid_until"] = None


def narrowed(attrs) -> bool:
    vf, vu = attrs.get("valid_from"), attrs.get("valid_until")
    df, du = attrs.get("valid_from_date"), attrs.get("valid_until_date")
    return bool((df and vf is not None and df > f"{vf:04d}-01-01")
                or (du and (vu is None or du < f"{vu:04d}-12-31")))


def citing(store):
    edges = [e for e in store.edges.values() if e.relation in CITATIONS]
    return len(edges), len({e.source for e in edges})


def main() -> None:
    source = GraphStore.load(KG_RESULTS_DIR / SOURCE_BUILD / "graph")
    articles = pd.read_csv(ARTICLES_CSV, low_memory=False)

    article_ids = {n.node_id[len("article:"):] for n in source.nodes.values()
                   if n.type == "Article"}
    corpus = articles[articles["article_id"].isin(article_ids)]
    meta = {**{k: v for k, v in source.meta.items()
               if k not in ("nodes", "edges", "nodes_by_type", "edges_by_relation")},
            "kg_version": TARGET_BUILD,
            "patched_from": SOURCE_BUILD,
            "articles_csv": ARTICLES_CSV.name,
            "patched_at": datetime.now(timezone.utc).isoformat()}
    store = build_backbone(corpus, meta=meta)

    dropped = sorted(n.node_id for n in source.nodes.values()
                     if n.type == "Article" and n.node_id not in store.nodes)
    dropped_set = set(dropped)

    # Edges first, so an entity whose mentions were all dropped is dropped too.
    wordings = wording_windows(corpus)
    kept_edges, stale = [], []
    for edge in source.edges.values():
        if edge.relation in BACKBONE or edge.source in dropped_set or edge.target in dropped_set:
            continue
        dates = edge_dates(edge, wordings) if edge.source.startswith("article:") else None
        if edge.source.startswith("article:") and dates is None:
            stale.append(f"{edge.source} -{edge.relation}-> {edge.target}")
            continue
        kept_edges.append((edge, dates))
    mentioned = {e.target for e, _ in kept_edges if e.relation == "mentions"}
    orphaned = []
    for node in source.nodes.values():
        if node.node_id in store.nodes or node.node_id in dropped_set:
            continue
        if node.type not in STRUCTURAL_TYPES and node.node_id not in mentioned:
            orphaned.append(node.node_id)
            continue
        store.add_node(node_id=node.node_id, name=node.name, type=node.type,
                       aliases=list(node.aliases),
                       provenance=list(node.provenance), attrs=dict(node.attrs))
    removed = dropped_set | set(orphaned)

    carried = dated = narrow = 0
    for edge, dates in kept_edges:
        if edge.source in removed or edge.target in removed:
            continue
        carried += 1
        attrs = dict(edge.attrs)
        if dates is not None:
            attrs.update(dates)
            dated += bool(dates["valid_from_date"] or dates["valid_until_date"])
            narrow += narrowed(attrs)
            years_from_dates(attrs)
        store.add_edge(edge.source, edge.relation, edge.target,
                       provenance=list(edge.provenance), attrs=attrs)

    mention_windows: dict[str, list[dict]] = {}
    for edge in store.edges.values():
        if edge.relation == "mentions":
            mention_windows.setdefault(edge.target, []).append(edge.attrs)
    entities = entities_dated = 0
    for node in store.nodes.values():
        if node.type in STRUCTURAL_TYPES:
            continue
        entities += 1
        union = union_window(mention_windows.get(node.node_id, []))
        if "valid_from_date" in union:
            node.attrs["valid_from_date"] = union["valid_from_date"]
            node.attrs["valid_until_date"] = union["valid_until_date"]
            node.attrs["valid_from"] = union.get("valid_from")
            node.attrs["valid_until"] = union.get("valid_until")
            entities_dated += bool(union["valid_from_date"] or union["valid_until_date"])

    out_dir = KG_RESULTS_DIR / TARGET_BUILD
    store.save(out_dir / "graph")

    cit_before, citing_before = citing(source)
    cit_after, citing_after = citing(store)
    report = {
        "source_build": SOURCE_BUILD,
        "articles_csv": str(ARTICLES_CSV.relative_to(PROJECT_ROOT)),
        "patched_at": meta["patched_at"],
        "articles": len(article_ids),
        "articles_redated": int(corpus["article_id"].nunique()),
        "articles_dropped_absent_from_corpus": dropped,
        "entities_dropped_without_mentions": sorted(orphaned),
        "nodes_before": len(source.nodes), "nodes_after": len(store.nodes),
        "edges_before": len(source.edges), "edges_after": len(store.edges),
        "citation_edges_before": cit_before, "citation_edges_after": cit_after,
        "citing_articles_before": citing_before, "citing_articles_after": citing_after,
        "carried_edges_dropped_no_wording": len(stale),
        "carried_edges": carried, "carried_edges_dated": dated,
        "carried_edges_narrowed": narrow,
        "entities": entities, "entities_dated": entities_dated,
        "edges_by_relation_before": source.stats()["edges_by_relation"],
        "edges_by_relation_after": store.stats()["edges_by_relation"],
    }
    (out_dir / "patch.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if not k.startswith("edges_by")}, indent=2))
    print("edges_by_relation after:", report["edges_by_relation_after"])


if __name__ == "__main__":
    main()
