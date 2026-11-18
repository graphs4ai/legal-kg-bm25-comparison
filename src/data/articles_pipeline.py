"""Article corpus: one row per wording of each article, with its validity interval.

    python -m src.data.articles_pipeline

Output: ``data/laws/articles_v5.csv``
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.laws_citations import (
    ActRegistry,
    extract_citations,
    hierarchy_level,
    positional_citations,
)
from src.data.laws_pipeline import (
    ENTRY_INTO_FORCE,
    RESOLVED_VERSION,
    OABCorpusPipeline,
    split_annexes,
    _normalize_whitespace,
    _strip_annotations,
    count_tokens,
    display_tipo,
    prepare_treaty_raw,
)
from src.data.laws_versioning import (
    ActDateIndex,
    EntryIntoForce,
    annotation_act_id,
    annotation_date,
    caput_annotation,
    caput_revocation,
    UNIT_SPLIT_TOKENS,
    assign_units,
    parse_annex_items,
    parse_occurrences,
    parse_provisions,
    resolve_all_versions,
)

DATA_PATH = Path("data/laws")

# An act's closing signature, which must never survive inside an article.
_SIGNOFF_IN_TEXT = r"(?im)^\s*Bras[ií]lia\s*,\s*\d{1,2}"
EXPORT_TEMPLATE = "articles_{version}.csv"

# The corpus has no "§": numbered paragraphs are a bare ordinal at line start.
_PARAGRAPH = re.compile(
    r"(?m)^[ \t]*(?:Par[áa]grafo\s+[úu]nico|\d+\s*[ºo°])\s*[.\-]?\s",
)
_INCISO = re.compile(r"(?m)^[ \t]*[IVXLC]+\s*[-–—.]\s")
_ALINEA = re.compile(r"(?m)^[ \t]*[a-z]\s*\)\s")

# Vetoed text, or an article that is only an effectiveness/revocation clause.
_VETADO = re.compile(r"^\(?\s*VETADO\s*\)?\.?$", re.IGNORECASE)
_VIGENCIA_ONLY = re.compile(
    r"^(?:est[ae]\s+\w+|o\s+disposto|as\s+disposi\w+)[^.]{0,80}"
    r"(?:entra(?:m)?\s+em\s+vigor|produz(?:ir[áa])?\s+efeitos)",
    re.IGNORECASE,
)
_REVOGA_ONLY = re.compile(
    r"^(?:revoga(?:m)?-se|fica(?:m)?\s+revogad)", re.IGNORECASE
)

# Leading article marker: "Art. 29-A." is a suffix; "Art. 10 - O dia" is a separator.
_ARTICLE_HEAD = re.compile(
    r"^\s*Art(?:igo)?\.?\s*[\d.]+[ºo°]?"
    r"(?:-\s?[A-Za-z](?![a-zà-ÿ])|(?-i:[A-Z])(?=[\s.\-–]))?"
    r"\s*[.\-–]?\s*",
    re.IGNORECASE,
)


def strip_article_head(text: str) -> str:
    """Remove the leading article marker from an article's text."""

    return _ARTICLE_HEAD.sub("", text, count=1).strip()


def split_caput(text: str) -> tuple[str, str]:
    """Split an article into (caput, remainder) at the first paragraph, inciso or alínea marker."""

    cuts = [
        match.start()
        for match in (
            _PARAGRAPH.search(text), _INCISO.search(text), _ALINEA.search(text)
        )
        if match
    ]
    if not cuts:
        return text.strip(), ""
    cut = min(cuts)
    return text[:cut].strip(), text[cut:].strip()


def classify_boilerplate(body: str) -> bool:
    """Return whether an article has no normative content (vetoed, effectiveness or revocation only)."""

    stripped = body.strip()
    if len(stripped.split()) <= 2:
        return True
    if _VETADO.match(stripped):
        return True
    if len(stripped.split()) <= 25 and (
        _VIGENCIA_ONLY.match(stripped) or _REVOGA_ONLY.match(stripped)
    ):
        return True
    return False


def _validity_bound_date(value, first: bool = False) -> date | None:
    """Resolve a bound (``2023`` or ``"2016-08-31"``) to a day; ``first`` takes a bare year's first day."""

    if value is None or pd.isna(value):
        return None
    text = str(value)
    if "-" in text:
        return date.fromisoformat(text)
    year = int(text[:4])
    return date(year, 1, 1) if first else date(year, 12, 31)


class ArticleCorpusPipeline:
    """Select documents, resolve versions, emit article rows, export."""

    def __init__(
        self,
        include_supplementary: bool = True,
        data_path: Path = DATA_PATH,
    ):

        self.include_supplementary = include_supplementary
        self.data_path = Path(data_path)
        self.corpus = OABCorpusPipeline(
            data_path=data_path, include_supplementary=include_supplementary
        )
        self.selected: pd.DataFrame | None = None
        self.articles: pd.DataFrame | None = None
        self.registry: ActRegistry | None = None
        self.in_force = EntryIntoForce()
        self.stats: dict = {}

    @property
    def export_path(self) -> Path:
        """Output CSV path."""
        return self.data_path / EXPORT_TEMPLATE.format(
            version=RESOLVED_VERSION)

    def select(self) -> pd.DataFrame:
        """Load and select the corpus documents."""

        self.corpus.load()
        self.selected = self.corpus.select()
        if self.include_supplementary:
            self.selected = self.corpus.select_supplementary(
                apply_validity=False)
        return self.selected

    def _build_registry(self) -> ActRegistry:
        """Build the act registry (keyed by document slug) from the selected documents."""

        from src.data.laws_citations import act_number, normalize_tipo

        acts: dict[str, dict] = {}
        for _, doc in self.selected.iterrows():
            num = doc.get("num")
            acts[doc["slug"]] = {
                "tipo": doc["tipo"],
                "tipo_key": normalize_tipo(doc["tipo"]),
                "num": act_number(num),
                "ano": None if pd.isna(doc.get("ano")) else int(doc["ano"]),
                "label": doc["label"],
            }
        return ActRegistry(acts)

    def build(self) -> pd.DataFrame:
        """Resolve versions and emit one row per article wording."""

        self.registry = self._build_registry()
        raw = self.corpus.raw
        self.in_force = EntryIntoForce.load(self.data_path / ENTRY_INTO_FORCE)
        self.dates = (
            ActDateIndex(raw["tipo"], raw["num_lei"], raw["data_lei"],
                         raw["lei_text"] if "lei_text" in raw else None,
                         in_force=self.in_force)
            if raw is not None else None
        )
        rows: list[dict] = []
        totals = {
            "occurrences": 0, "kept": 0, "superseded": 0,
            "not_yet_enacted": 0, "repealed": 0, "quoted": 0,
        }
        quoted_edges: list[dict] = []

        for _, doc in self.selected.iterrows():
            act_id = doc["slug"]
            act_year = None if pd.isna(doc.get("ano")) else int(doc["ano"])

            # A bound is a year or an ISO date; year intervals use the year, date intervals the day.
            act_valid_until = doc.get("act_valid_until")
            if act_valid_until is not None and pd.isna(act_valid_until):
                act_valid_until = None
            until_year = (
                None if act_valid_until is None else int(str(act_valid_until)[:4])
            )

            # An act with a later entry into force starts its intervals on that date.
            act_valid_from = doc.get("act_valid_from")
            if act_valid_from is not None and pd.isna(act_valid_from):
                act_valid_from = None
            base_year = (
                act_year if act_valid_from is None else int(str(act_valid_from)[:4])
            )

            raw_text = str(doc["raw_text"])
            # A promulgation decree's content is the treaty annexed after its sign-off.
            annex_texts: list[str] = []
            if doc.get("kind") == "tratado":
                raw_text = prepare_treaty_raw(raw_text)
            else:
                # Text after the sign-off is the act's annexes, not its last article.
                raw_text, annex_texts = split_annexes(raw_text)
            occurrences = parse_occurrences(
                raw_text, act_id, registry=self.registry
            )
            if not occurrences:
                raise ValueError(
                    f"{act_id}: no article markers parsed from its raw text"
                )
            # Wordings are ordered by enactment year; the entry-into-force day only clips starts.
            kept, stats = resolve_all_versions(
                occurrences, act_year or base_year, act_valid_until=until_year
            )
            for key in totals:
                totals[key] += getattr(stats, key)

            for target, count in stats.quoted_targets.items():
                quoted_edges.append(
                    {"act_id": act_id, "target_act_id": target,
                     "quoted_articles": count}
                )

            header = (
                doc["label"] if doc.get("kind") == "constituicao"
                else f"{doc['label']} ({act_year})" if doc.get("num") is None
                else f"{doc['label']} — "
                     f"{display_tipo(doc['tipo'], doc['num'], act_year)}"
            )

            act_date = doc.get("data_lei")
            if act_date is None or pd.isna(act_date):
                act_date = date(base_year, 1, 1) if base_year else None
                act_precision = "year" if base_year else "none"
            else:
                act_precision = "day"
            published, published_precision = act_date, act_precision
            # An act published before it took effect starts on its entry-into-force day.
            force_date = _validity_bound_date(act_valid_from, first=True)
            if force_date is not None and (
                act_date is None or force_date > act_date
            ):
                act_date, act_precision = force_date, (
                    "day" if "-" in str(act_valid_from) else "year")

            # Document order per container, so "o artigo anterior" resolves ("5-A" follows "5").
            neighbours: dict[tuple[str, str], tuple[str | None, str | None]] = {}
            for container in {o.occurrence.container for o in kept}:
                ordered, seen = [], set()
                for versioned in kept:
                    number = versioned.occurrence.article_number
                    if versioned.occurrence.container == container and number not in seen:
                        seen.add(number)
                        ordered.append(number)
                for index, number in enumerate(ordered):
                    neighbours[(container, number)] = (
                        ordered[index - 1] if index else None,
                        ordered[index + 1] if index + 1 < len(ordered) else None,
                    )

            for seq, versioned in enumerate(kept, start=1):
                occurrence = versioned.occurrence
                cleaned = _normalize_whitespace(
                    _strip_annotations(occurrence.text)
                )
                body = strip_article_head(cleaned)
                caput, remainder = split_caput(body)
                version_year = occurrence.effective_year(act_year or base_year)

                article_id = (
                    f"{act_id}:art_{occurrence.article_number}"
                    if occurrence.container == "main"
                    else f"{act_id}:{occurrence.container}:"
                         f"art_{occurrence.article_number}"
                )
                # Articles the act's vigência clause dates separately start on their own day.
                row_valid_from = act_valid_from
                if act_id in self.in_force:
                    own = self.in_force.deferred_start(act_id, published, article_id)
                    row_valid_from = own.isoformat() if own is not None else None

                caput_ann = caput_annotation(occurrence.text)
                if occurrence.kind == "original":
                    from_date, precision = act_date, act_precision
                    if row_valid_from != act_valid_from:
                        own = _validity_bound_date(row_valid_from, first=True)
                        from_date, precision = (
                            (own, "day") if own is not None
                            else (published, published_precision))
                else:
                    from_date, precision = annotation_date(
                        caput_ann, self.dates, article_id)
                    if from_date is None and version_year:
                        from_date, precision = date(version_year, 1, 1), "year"
                revoked_on, _ = annotation_date(
                    caput_revocation(occurrence.text), self.dates, article_id
                ) if occurrence.revoked_year else (None, None)
                provisions = parse_provisions(
                    occurrence.text, self.dates, article_id)
                assign_units(provisions)
                units = {p.unit for p in provisions if p.text}
                # Long articles are split per provision; short ones keep the article id.
                split = (
                    count_tokens(cleaned) >= UNIT_SPLIT_TOKENS and len(units) > 1
                )

                previous_number, next_number = neighbours.get(
                    (occurrence.container, occurrence.article_number), (None, None))
                citations = extract_citations(
                    cleaned, act_id, self.registry,
                    source_year=version_year,
                ) + positional_citations(
                    cleaned, act_id, previous_number, next_number,
                )
                citation_records = [
                    {
                        "relation": c.relation,
                        "target_act_id": c.target_act_id,
                        "article_numbers": c.article_numbers,
                        "scope": c.scope,
                        "resolution": c.resolution,
                        **({"container": c.container} if c.container else {}),
                    }
                    for c in citations
                ]

                rows.append({
                    "article_id": article_id,
                    "act_id": act_id,
                    "container": occurrence.container,
                    "article_number": occurrence.article_number,
                    "seq": seq,
                    "tipo": doc["tipo"],
                    "hierarchy_level": hierarchy_level(doc["tipo"]),
                    "lei_label": doc["label"],
                    "act_header": header,
                    "num_lei": doc.get("num"),
                    "ano": act_year,
                    "version_year": version_year,
                    "version_kind": occurrence.kind,
                    "source_act": annotation_act_id(caput_ann, self.dates),
                    "version_seq": versioned.version_seq,
                    # The year follows the resolved day; _align_year_bounds closes the previous wording.
                    "valid_from": (
                        from_date.year if from_date is not None and precision == "day"
                        else versioned.valid_from),
                    "valid_until": versioned.valid_until,
                    "valid_from_date": from_date,
                    "valid_until_date": (
                        revoked_on - timedelta(days=1) if revoked_on else None
                    ),
                    "date_precision": precision,
                    "provisions": json.dumps(
                        [p.to_record() for p in provisions], ensure_ascii=False),
                    "n_provisions": len(provisions),
                    "n_units": (len(units) + 1) if split else 1,
                    "split": split,
                    "path": occurrence.path or None,
                    "livro": occurrence.levels.get("livro") or None,
                    "titulo": occurrence.levels.get("titulo") or None,
                    "capitulo": occurrence.levels.get("capitulo") or None,
                    "secao": occurrence.levels.get("secao") or None,
                    "caput": caput,
                    "remainder": remainder,
                    "article_text": cleaned,
                    "tokens": count_tokens(cleaned),
                    "n_paragraphs": len(_PARAGRAPH.findall(body)),
                    "n_incisos": len(_INCISO.findall(body)),
                    "is_boilerplate": classify_boilerplate(body),
                    "is_annex": False,
                    "citations": json.dumps(
                        citation_records, ensure_ascii=False),
                    "n_citations": len(citation_records),
                    "n_citations_internal": sum(
                        1 for c in citations if c.scope == "internal"),
                    "n_citations_external": sum(
                        1 for c in citations if c.scope == "external"),
                    "area": doc["area"],
                    "source": doc.get("source", "legislaai"),
                    "act_valid_from": row_valid_from,
                    "act_valid_until": act_valid_until,
                })

            rows.extend(self._annex_rows(
                annex_texts, doc, act_id, header, act_date, act_precision,
                act_valid_from, act_valid_until, start_seq=len(kept) + 1,
            ))

        self.articles = pd.DataFrame(rows)
        self._close_date_intervals()
        self._align_year_bounds()
        self.stats = {
            **totals,
            "acts": int(self.selected.shape[0]),
            "articles": len(self.articles),
            "quoted_edges": quoted_edges,
        }

        key = ["article_id", "valid_from"]
        duplicates = self.articles.loc[
            self.articles.duplicated(subset=key), "article_id"
        ].unique()
        if len(duplicates):
            raise ValueError(
                f"Duplicate (article_id, valid_from): {list(duplicates)[:10]}"
            )

        overlaps = self._overlapping_intervals()
        if overlaps:
            raise ValueError(
                f"Overlapping validity intervals: {overlaps[:10]}"
            )

        leaked = self._articles_holding_a_signoff()
        if leaked:
            raise ValueError(
                f"Sign-off (and whatever follows it) left inside an article: "
                f"{leaked[:10]}"
            )

        inverted = self._inverted_date_intervals()
        if inverted:
            raise ValueError(
                f"Validity interval ends before it starts: {inverted[:10]}"
            )

        logger.info(
            "Built {:,} article versions ({:,} distinct articles) from {} acts",
            len(self.articles), self.articles["article_id"].nunique(),
            self.stats["acts"],
        )
        return self.articles

    def _annex_rows(
        self, annex_texts, doc, act_id, header, act_date, act_precision,
        act_valid_from, act_valid_until, start_seq,
    ) -> list[dict]:
        """One row per annex printed after an act's sign-off; its items become provisions."""

        rows = []
        for index, text in enumerate(annex_texts, start=1):
            container = f"anexo_{index}"
            cleaned = _normalize_whitespace(_strip_annotations(text))
            items = parse_annex_items(cleaned)
            assign_units(items)
            annex_units = {p.unit for p in items if p.text}
            annex_split = (
                count_tokens(cleaned) >= UNIT_SPLIT_TOKENS and len(annex_units) > 1
            )
            rows.append({
                "article_id": f"{act_id}:{container}:art_1",
                "act_id": act_id,
                "container": container,
                "article_number": "1",
                "seq": start_seq + index - 1,
                "tipo": doc["tipo"],
                "hierarchy_level": hierarchy_level(doc["tipo"]),
                "lei_label": doc["label"],
                "act_header": header,
                "num_lei": doc.get("num"),
                "ano": None if pd.isna(doc.get("ano")) else int(doc["ano"]),
                "version_year": None,
                "version_kind": "original",
                "source_act": None,
                "version_seq": 1,
                "valid_from": act_date.year if act_date else None,
                "valid_until": None,
                "valid_from_date": act_date,
                "valid_until_date": None,
                "date_precision": act_precision,
                "provisions": json.dumps(
                    [p.to_record() for p in items], ensure_ascii=False),
                "n_provisions": len(items),
                "n_units": (len(annex_units) + 1) if annex_split else 1,
                "split": annex_split,
                "path": container.upper(),
                "livro": None, "titulo": None, "capitulo": None, "secao": None,
                "caput": cleaned.split("\n", 1)[0],
                "remainder": "",
                "article_text": cleaned,
                "tokens": count_tokens(cleaned),
                "n_paragraphs": 0,
                "n_incisos": 0,
                "is_boilerplate": False,
                "is_annex": True,
                "citations": json.dumps([], ensure_ascii=False),
                "n_citations": 0,
                "n_citations_internal": 0,
                "n_citations_external": 0,
                "area": doc["area"],
                "source": doc.get("source", "legislaai"),
                "act_valid_from": act_valid_from,
                "act_valid_until": act_valid_until,
            })
        return rows

    def _close_date_intervals(self) -> None:
        """Fill ``valid_until_date`` from the next wording or the act cap; serialize dates as ISO.

        Wordings replaced before the act entered into force are dropped.
        """

        df = self.articles
        if "act_valid_from" in df.columns:
            force = df["act_valid_from"].apply(
                lambda v: _validity_bound_date(v, first=True))
            for idx in df.index:
                start = df.at[idx, "valid_from_date"]
                if (force[idx] is not None and isinstance(start, date)
                        and start < force[idx]):
                    df.at[idx, "valid_from_date"] = force[idx]
                    df.at[idx, "valid_from"] = force[idx].year
        order = df.sort_values(["article_id", "version_seq"]).index
        next_start = (
            df.loc[order, "valid_from_date"].groupby(df.loc[order, "article_id"]).shift(-1)
        )
        until = df["valid_until_date"].copy()
        last_in_chain = (
            df.loc[order, "article_id"].shift(-1) != df.loc[order, "article_id"])
        for idx, start in next_start.items():
            if isinstance(start, date) and pd.isna(until.get(idx)):
                until[idx] = start - timedelta(days=1)
        # Replaced, or revoked, on or before the day its act entered into force.
        def _clipped(idx):
            return ("act_valid_from" in df.columns
                    and _validity_bound_date(df.at[idx, "act_valid_from"], first=True)
                    == df.at[idx, "valid_from_date"])

        never_in_force = [
            idx for idx, start in next_start.items()
            if isinstance(start, date) and start <= df.at[idx, "valid_from_date"]
            and _clipped(idx)
        ] + [
            idx for idx in df.index
            if isinstance(until.get(idx), date) and isinstance(df.at[idx, "valid_from_date"], date)
            and until[idx] < df.at[idx, "valid_from_date"] and _clipped(idx)
        ]
        never_in_force = list(dict.fromkeys(never_in_force))
        if never_in_force:
            logger.info(f"Dropped {len(never_in_force)} wordings replaced "
                        f"before their act entered into force")
            df = df.drop(index=never_in_force)
            until = until.drop(index=never_in_force)
            self.articles = df
        cap = df["act_valid_until"].apply(_validity_bound_date)
        starts = df["valid_from_date"]
        for idx in df.index:
            if cap[idx] is not None and (pd.isna(until[idx]) or until[idx] > cap[idx]):
                until[idx] = cap[idx]
            # A wording whose successor was dropped ends with its last year.
            year_end = df.at[idx, "valid_until"] if "valid_until" in df.columns else None
            if (pd.isna(until[idx]) and last_in_chain.get(idx, False)
                    and year_end is not None and not pd.isna(year_end)):
                until[idx] = date(int(year_end), 12, 31)
            # A wording cannot end before it began.
            start = starts[idx]
            if (isinstance(start, date) and isinstance(until[idx], date)
                    and until[idx] < start):
                until[idx] = start
        df["valid_until_date"] = until
        for col in ("valid_from_date", "valid_until_date"):
            df[col] = df[col].apply(
                lambda d: d.isoformat() if isinstance(d, date) else None)

    def _align_year_bounds(self) -> None:
        """Close each wording's year interval the year before the next one starts."""

        df = self.articles
        order = df.sort_values(["article_id", "version_seq"]).index
        ids = df.loc[order, "article_id"].tolist()
        for prev, cur, prev_id, cur_id in zip(order, order[1:], ids, ids[1:]):
            if prev_id != cur_id:
                continue
            start, end = df.at[cur, "valid_from"], df.at[prev, "valid_until"]
            if pd.isna(start) or pd.isna(end) or end == start - 1:
                continue
            prev_start = df.at[prev, "valid_from"]
            if pd.isna(prev_start) or start - 1 >= prev_start:
                df.at[prev, "valid_until"] = start - 1
            else:
                df.at[cur, "valid_from"] = end + 1

    def _overlapping_intervals(self) -> list[str]:
        """Article ids whose wordings cover a year more than once."""

        offenders: list[str] = []
        columns = ["article_id", "valid_from", "valid_until"]
        for article_id, group in self.articles[columns].groupby("article_id"):
            if len(group) < 2:
                continue
            spans = sorted(
                (
                    -10**9 if pd.isna(row.valid_from) else int(row.valid_from),
                    10**9 if pd.isna(row.valid_until) else int(row.valid_until),
                )
                for row in group.itertuples()
            )
            if any(spans[i][1] >= spans[i + 1][0]
                   for i in range(len(spans) - 1)):
                offenders.append(str(article_id))
        return offenders

    def _articles_holding_a_signoff(self) -> list[str]:
        """Article ids that still contain the act's closing signature (a swallowed annex)."""

        df = self.articles
        held = df.loc[
            (~df["is_annex"].fillna(False))
            & df["article_text"].fillna("").str.contains(
                _SIGNOFF_IN_TEXT, regex=True),
            "article_id",
        ]
        return held.astype(str).tolist()

    def _inverted_date_intervals(self) -> list[str]:
        """Article ids with a wording whose day interval ends before it starts."""

        df = self.articles
        if "valid_from_date" not in df.columns:
            return []
        start = pd.to_datetime(df["valid_from_date"], errors="coerce")
        end = pd.to_datetime(df["valid_until_date"], errors="coerce")
        return df.loc[end.notna() & start.notna() & (end < start),
                      "article_id"].astype(str).tolist()

    def summarize(self) -> None:
        """Log the resolution counts and corpus shape."""

        stats = self.stats
        multi = self.articles.groupby("article_id").size()
        logger.info(
            "\n".join([
                "=" * 56,
                "ARTICLE CORPUS (all versions)",
                "=" * 56,
                f"Acts                : {stats['acts']}",
                f"Occurrences parsed  : {stats['occurrences']:,}",
                f"Article versions    : {stats['kept']:,}",
                f"Distinct articles   : {multi.size:,}",
                f"With >1 version     : {int((multi > 1).sum()):,}",
                f"Superseded dropped  : {stats['superseded']:,}",
                f"Repealed dropped    : {stats['repealed']:,}",
                f"Quoted (other acts) : {stats['quoted']:,}",
                f"Boilerplate flagged : "
                f"{int(self.articles['is_boilerplate'].sum()):,}",
                f"Total tokens        : {int(self.articles['tokens'].sum()):,}",
                f"Citations           : {int(self.articles['n_citations'].sum()):,}",
                "=" * 56,
            ])
        )

    def export(self) -> Path:
        """Write the article corpus to CSV and return its path."""

        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self.articles.to_csv(self.export_path, index=False)
        logger.info("Exported {} -> {}", len(self.articles), self.export_path)
        return self.export_path

    def run(self) -> pd.DataFrame:
        """Run select, build, summarize and export."""

        self.select()
        self.build()
        self.summarize()
        self.export()
        return self.articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-supplementary", action="store_true", default=True,
        help="Include manually sourced documents (default: on).",
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    pipeline = ArticleCorpusPipeline(
        include_supplementary=args.with_supplementary,
        data_path=args.data_path,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
