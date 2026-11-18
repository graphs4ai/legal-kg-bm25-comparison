"""Article-version resolution over raw law text.

The source lists each article's wordings inline, tagged with annotations such as
``(Redação dada pela Lei nº 9.714, de 1998)``. This module dates each wording and
provision from those annotations and selects the one in force at a cutoff.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Version annotations: "Redação dada" replaces, "Incluído" adds, "Revogado" repeals.
_ANN_REDACAO = re.compile(
    r"\(\s*(?:Reda[cç][ãa]o\s+(?:dada|da\s+pel|pel|restabelecid)|Restabelecid[oa]|"
    r"Revigorad[oa])[^)]{0,140}(?:\)|$)", re.IGNORECASE | re.MULTILINE
)
_ANN_INCLUIDO = re.compile(
    r"\(\s*(?:(?:Artigo|Par[áa]grafo)\s+)?(?:Inclu[íi]d[oa]|Acrescentad[oa])[^)]{0,140}(?:\)|$)",
    re.IGNORECASE | re.MULTILINE
)
_ANN_REVOGADO = re.compile(
    r"\(\s*Revogad[oa][^)]{0,140}(?:\)|$)", re.IGNORECASE | re.MULTILINE
)
_ANN_RENUMERADO = re.compile(
    r"\(\s*Renumerado[^)]{0,140}(?:\)|$)", re.IGNORECASE | re.MULTILINE
)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Article marker at line start. Group 2 is a tight letter suffix ("Art. 29-A.", "Art. 10A.");
# a spaced hyphen ("Art. 53 - Nas") is a separator.
_ARTICLE_MARKER = re.compile(
    r"(?m)^[ \t]*Art(?:igo)?\.?\s*(\d+(?:\.\d+)*)[ºo°]?"
    r"(-\s?[A-Za-z](?![a-zà-ÿ])|[A-Z](?=[\s.\-\u2013]))?",
)

# Opening of a quoted amendment block; the quoted articles belong to the target act.
_AMEND_TRIGGER = re.compile(
    r"passa(?:m)?\s+a\s+vigorar|passa(?:m)?\s+a\s+ter\s+a\s+seguinte|"
    r"fica(?:m)?\s+acrescid|acrescido\s+d[oe]s?\s+seguinte|"
    r"fica(?:m)?\s+assim\s+redigid",
    re.IGNORECASE,
)

# ADCT: a container that restarts numbering.
_ADCT = re.compile(
    r"(?m)^[ \t]*(ATO\s+DAS\s+DISPOSI[ÇC][ÕO]ES\s+CONSTITUCIONAIS\s+TRANSIT[ÓO]RIAS)",
    re.IGNORECASE,
)
# Annex openings, which likewise restart numbering.
_ANNEX = re.compile(
    r"(?m)^[ \t]*(ANEXO(?:\s+[IVXL\d]+)?|AP[ÊE]NDICE(?:\s+[IVXL\d]+)?)\s*$",
    re.IGNORECASE,
)

CONTAINER_MAIN = "main"
CONTAINER_ADCT = "adct"

KIND_ORIGINAL = "original"
KIND_REDACAO = "redacao"
KIND_INCLUIDO = "incluido"
KIND_REVOGADO = "revogado"

# The unit holding an article's opening rule.
UNIT_CAPUT = "caput"


def _norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_article_number(number: str, suffix: str | None = None) -> str:
    """Canonical article number with its letter suffix, e.g. ``"1234"`` or ``"3-A"``."""

    base = str(number).replace(".", "").strip()
    if suffix:
        letter = re.sub(r"[^A-Za-z]", "", suffix).upper()
        if letter:
            return f"{base}-{letter}"
    return base


@dataclass
class ArticleOccurrence:
    """One appearance (wording) of an article in an act's raw text.

    ``container`` is ``"main"``, ``"adct"``, ``"anexo_N"`` or ``"quoted:<target_act_id>"``;
    ``kind`` is ``original``, ``redacao``, ``incluido`` or ``revogado``.
    """

    article_number: str
    container: str
    start: int
    end: int
    text: str
    kind: str = KIND_ORIGINAL
    year: int | None = None
    quoted_target: str | None = None
    path: str = ""
    levels: dict[str, str] = field(default_factory=dict)
    revoked_year: int | None = None

    @property
    def is_quoted(self) -> bool:
        """True when this occurrence is another act's article, quoted here."""
        return self.container.startswith("quoted:")

    def effective_year(self, act_year: int | None) -> int | None:
        """Year this wording came into force; ``act_year`` for un-annotated original wording."""

        return self.year if self.year is not None else act_year

    def key(self) -> tuple[str, str]:
        """Identity of the article this occurrence is a version of."""
        return (self.container, self.article_number)


def _annotation(text: str) -> tuple[str, int | None]:
    """(kind, year) of an occurrence's annotation, read from the caput line only."""

    head = _caput_line(text)

    # A caput with text beside a revocation annotation is a real wording that ends there.
    if _ANN_REVOGADO.search(head) and _has_wording(head):
        for pattern, kind in (
            (_ANN_REDACAO, KIND_REDACAO),
            (_ANN_INCLUIDO, KIND_INCLUIDO),
            (_ANN_RENUMERADO, KIND_REDACAO),
        ):
            match = pattern.search(head)
            if match:
                return kind, _annotation_year(match.group(0))
        return KIND_ORIGINAL, None

    for pattern, kind in (
        (_ANN_REVOGADO, KIND_REVOGADO),
        (_ANN_REDACAO, KIND_REDACAO),
        (_ANN_INCLUIDO, KIND_INCLUIDO),
        (_ANN_RENUMERADO, KIND_REDACAO),
    ):
        match = pattern.search(head)
        if match:
            return kind, _annotation_year(match.group(0))

    return KIND_ORIGINAL, None


def _caput_line(text: str) -> str:
    """First line of an occurrence, where the article's own annotation sits."""
    return text.lstrip("\n").split("\n", 1)[0]


# Annotation year from its date clause ("de 1999", "de 11.10.1995"); act numbers can look like years.
_ANN_DATE_YEAR = re.compile(
    r"\bde\s*\.?\s*(?:\d{1,2}\s*o?\s*[./-]\s*\d{1,2}\s*[./-]\s*)?"
    r"(19\d{2}|20\d{2}|[12]\.\d{3})\b",
    re.IGNORECASE,
)
# The same clause with a two-digit year ("de 10.12.97").
_ANN_DATE_SHORT_YEAR = re.compile(
    r"\bde\s+\d{1,2}\s*o?\s*[./]\s*\d{1,2}\s*[./]\s*(\d{2})\b", re.IGNORECASE)


def _annotation_year(annotation: str) -> int | None:
    """Year an annotation names in its date clause, or None."""

    # The first date clause is the amending act's; later ones come from unclosed "(Vide …)" notes.
    dated = _ANN_DATE_YEAR.findall(annotation)
    if dated:
        return int(dated[0].replace(".", ""))
    short = _ANN_DATE_SHORT_YEAR.search(annotation)
    if short:
        return _full_year(short.group(1))
    years = [int(y) for y in _YEAR.findall(annotation) if 1900 <= int(y) <= 2100]
    return years[0] if years else None


def _has_wording(head: str) -> bool:
    """True when the caput line carries text beyond its marker and annotations."""
    body = re.sub(r"\([^)]*\)", " ", head)
    body = _ARTICLE_MARKER.sub("", body, count=1)
    return len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", body)) > 3


def revoked_year(text: str) -> int | None:
    """Revocation year on a caput line that still holds its wording, or None."""

    head = _caput_line(text)
    match = _ANN_REVOGADO.search(head)
    if match and _has_wording(head):
        return _annotation_year(match.group(0))
    return None


def _container_spans(raw: str) -> list[tuple[int, str]]:
    """(offset, container) boundaries that restart numbering, starting with (0, "main")."""

    spans: list[tuple[int, str]] = [(0, CONTAINER_MAIN)]

    # The heading also appears in the index at the top; the last occurrence is the section.
    adct = list(_ADCT.finditer(raw))
    if adct:
        spans.append((adct[-1].start(), CONTAINER_ADCT))

    for index, match in enumerate(_ANNEX.finditer(raw), start=1):
        spans.append((match.start(), f"anexo_{index}"))

    spans.sort(key=lambda item: item[0])
    return spans


def _container_at(spans: list[tuple[int, str]], offset: int) -> str:
    """Return the container covering a raw-text offset."""

    current = CONTAINER_MAIN
    for start, name in spans:
        if offset >= start:
            current = name
        else:
            break
    return current


def parse_occurrences(
    raw: str,
    act_id: str,
    registry=None,
) -> list[ArticleOccurrence]:
    """Split an act's raw text into article occurrences, in document order.

    A quoted amendment block runs from a host article naming another act until
    the host's numbering resumes; ``registry`` resolves its target act.
    """

    if not raw:
        return []

    from src.data.laws_pipeline import (
        _HEADING,
        _LEVEL_KEYS,
        _LEVEL_LABEL,
        _norm_kind,
    )

    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    spans = _container_spans(text)
    markers = list(_ARTICLE_MARKER.finditer(text))
    if not markers:
        return []

    # Headings preceding each article, for the structural breadcrumb.
    headings = [
        (m.start(), _norm_kind(m.group(1)),
         f"{m.group(2).strip()} – {m.group(3).strip()}".strip(" –"))
        for m in _HEADING.finditer(text)
    ]
    heading_index = 0
    levels: dict[str, str] = {key: "" for key in _LEVEL_KEYS}

    occurrences: list[ArticleOccurrence] = []

    # State for quoted-amendment tracking.
    quoted_target: str | None = None
    quoted_active = False
    host_number: int | None = None

    for index, match in enumerate(markers):
        start = match.start()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        body = text[start:end]

        number = normalize_article_number(match.group(1), match.group(2))
        base = int(match.group(1).replace(".", "")) if match.group(1) else None
        container = _container_at(spans, start)

        # Advance the running heading state up to this article's offset.
        while heading_index < len(headings) and headings[heading_index][0] < start:
            _, level, label = headings[heading_index]
            levels[_LEVEL_KEYS[level]] = label
            for deeper in _LEVEL_KEYS[level + 1:]:
                levels[deeper] = ""
            heading_index += 1
        path = " › ".join(
            f"{_LEVEL_LABEL[i]} {levels[key]}"
            for i, key in enumerate(_LEVEL_KEYS) if levels[key]
        )

        # A quoted block ends when numbering resumes the host's sequence.
        if quoted_active and base is not None and host_number is not None:
            resumes = (
                match.group(2) is None
                and host_number < base <= host_number + 3
            )
            if resumes:
                quoted_active = False
                quoted_target = None

        if quoted_active:
            occurrences.append(
                ArticleOccurrence(
                    article_number=number,
                    container=f"quoted:{quoted_target or 'unknown'}",
                    start=start, end=end, text=body,
                    kind=KIND_REDACAO, year=None,
                    quoted_target=quoted_target,
                    path=path, levels=dict(levels),
                )
            )
            continue

        kind, year = _annotation(body)
        occurrences.append(
            ArticleOccurrence(
                article_number=number, container=container,
                start=start, end=end, text=body, kind=kind, year=year,
                path=path, levels=dict(levels),
                revoked_year=revoked_year(body),
            )
        )

        # Does this host article open a quoted amendment block?
        if _AMEND_TRIGGER.search(body):
            target = None
            if registry is not None:
                from src.data.laws_citations import extract_citations
                for citation in extract_citations(body, act_id, registry):
                    if (citation.target_act_id
                            and citation.target_act_id != act_id):
                        target = citation.target_act_id
                        break
            quoted_target = target
            quoted_active = True
            host_number = base

    return occurrences


@dataclass
class ResolutionStats:
    """Counts describing one act's version resolution."""

    occurrences: int = 0
    kept: int = 0
    superseded: int = 0
    not_yet_enacted: int = 0
    repealed: int = 0
    quoted: int = 0
    quoted_targets: dict[str, int] = field(default_factory=dict)


def resolve_at_year(
    occurrences: list[ArticleOccurrence],
    target_year: int | None,
    act_year: int | None,
) -> tuple[list[ArticleOccurrence], ResolutionStats]:
    """Keep each article's wording in force at ``target_year`` (None: newest); drop quoted blocks."""

    stats = ResolutionStats(occurrences=len(occurrences))
    groups: dict[tuple[str, str], list[ArticleOccurrence]] = {}

    for occurrence in occurrences:
        if occurrence.is_quoted:
            stats.quoted += 1
            target = occurrence.quoted_target or "unknown"
            stats.quoted_targets[target] = (
                stats.quoted_targets.get(target, 0) + 1
            )
            continue
        groups.setdefault(occurrence.key(), []).append(occurrence)

    kept: list[ArticleOccurrence] = []
    for versions in groups.values():
        eligible = [
            v for v in versions
            if target_year is None
            or (v.effective_year(act_year) or 0) <= target_year
        ]
        if not eligible:
            stats.not_yet_enacted += len(versions)
            continue

        # Latest wording in force; ties go to document order (current version last).
        current = max(
            eligible,
            key=lambda v: (v.effective_year(act_year) or 0, v.start),
        )
        if current.kind == KIND_REVOGADO or (
            target_year is not None and current.revoked_year is not None
            and target_year >= current.revoked_year
        ):
            stats.repealed += 1
            stats.superseded += len(versions) - 1
            continue

        kept.append(current)
        stats.superseded += len(eligible) - 1

    kept.sort(key=lambda v: v.start)
    stats.kept = len(kept)
    return kept, stats


@dataclass
class VersionedOccurrence:
    """One wording of an article and the years it was in force (``valid_until`` None while current)."""

    occurrence: ArticleOccurrence
    valid_from: int | None
    valid_until: int | None
    version_seq: int


def resolve_all_versions(
    occurrences: list[ArticleOccurrence],
    act_year: int | None,
    act_valid_until: int | None = None,
) -> tuple[list[VersionedOccurrence], ResolutionStats]:
    """Every wording of every article with its year interval, capped at ``act_valid_until``.

    A ``revogado`` entry closes the preceding interval; wordings replaced within their year are dropped.
    """

    stats = ResolutionStats(occurrences=len(occurrences))
    groups: dict[tuple[str, str], list[ArticleOccurrence]] = {}

    for occurrence in occurrences:
        if occurrence.is_quoted:
            stats.quoted += 1
            target = occurrence.quoted_target or "unknown"
            stats.quoted_targets[target] = (
                stats.quoted_targets.get(target, 0) + 1
            )
            continue
        groups.setdefault(occurrence.key(), []).append(occurrence)

    kept: list[VersionedOccurrence] = []
    for versions in groups.values():
        chain = sorted(
            versions, key=lambda v: (v.effective_year(act_year) or 0, v.start))

        emitted = 0
        for index, current in enumerate(chain):
            valid_from = current.effective_year(act_year)
            following = chain[index + 1] if index + 1 < len(chain) else None
            valid_until = (
                (following.effective_year(act_year) or 0) - 1
                if following is not None else act_valid_until
            )
            if act_valid_until is not None and (
                valid_until is None or valid_until > act_valid_until
            ):
                valid_until = act_valid_until
            if current.revoked_year is not None and (
                valid_until is None or valid_until >= current.revoked_year
            ):
                valid_until = current.revoked_year - 1

            if current.kind == KIND_REVOGADO:
                stats.repealed += 1
                continue
            if (valid_until is not None and valid_from is not None
                    and valid_until < valid_from):
                stats.superseded += 1
                continue

            emitted += 1
            kept.append(VersionedOccurrence(
                occurrence=current,
                valid_from=valid_from,
                valid_until=valid_until,
                version_seq=emitted,
            ))

    kept.sort(key=lambda v: v.occurrence.start)
    stats.kept = len(kept)
    return kept, stats


def filter_raw_text(
    raw: str,
    act_id: str,
    act_year: int | None,
    target_year: int | None,
    registry=None,
) -> tuple[str, ResolutionStats]:
    """Raw act text keeping only the wordings in force at ``target_year``, plus resolution counts."""

    occurrences = parse_occurrences(raw, act_id, registry=registry)
    if not occurrences:
        return raw, ResolutionStats()

    kept, stats = resolve_at_year(occurrences, target_year, act_year)

    # Keep the text before the first article, then only the kept occurrences.
    head = raw[: occurrences[0].start]
    pieces = [head.rstrip("\n")] if head.strip() else []

    previous_end = occurrences[0].start
    for occurrence in kept:
        # Keep headings and container markers between kept occurrences.
        if occurrence.start > previous_end:
            between = raw[previous_end:occurrence.start]
            # Drop article text but keep heading lines.
            headings = [
                line for line in between.split("\n")
                if line.strip() and not _ARTICLE_MARKER.match(line)
                and (line.isupper() or _ADCT.match(line) or _ANNEX.match(line))
            ]
            if headings:
                pieces.append("\n".join(headings))
        pieces.append(occurrence.text.strip("\n"))
        previous_end = occurrence.end

    return "\n\n".join(p for p in pieces if p.strip()), stats


# Dates behind annotations

# Scrape ``tipo`` -> instrument key used by annotations.
_DATE_TIPO_KEYS = {
    "lei ordinaria": "lei",
    "lei complementar": "lei_complementar",
    "lei delegada": "lei_delegada",
    "decreto-lei": "decreto_lei",
    "decreto": "decreto",
    "emenda constitucional": "emenda_constitucional",
    "medida provisoria": "medida_provisoria",
}
# Instrument as written inside an annotation, longest first.
_DATE_INSTRUMENTS = [
    (r"lei\s+complementar|lcp", "lei_complementar"),
    (r"lei\s+delegada", "lei_delegada"),
    (r"emenda\s+(?:constituicao\s+)?constitu(?:i)?cional", "emenda_constitucional"),
    (r"medida\s+provisoria", "medida_provisoria"),
    (r"decreto[-\s]lei", "decreto_lei"),
    (r"decreto", "decreto"),
    (r"lei", "lei"),
]
# "pela Lei no 13.964, de 2019", "de 19.6.2001", "de 24 de dezembro de 2019"; also tolerates
# scrape variants ("Incluido Lei no", "13. 281", "1o.10.1973", "de.2009", "de 2.010").
_CITED_ACT = re.compile(
    r"(?:\bpel[ao]s?\s*,?\s*|\b)(?P<instr>" + "|".join(p for p, _ in _DATE_INSTRUMENTS) + r")"
    r"\s*n?\s?[o.]*\s*(?P<num>\d{1,3}(?:[.,]\s?\d{3}(?!\d))+(?:-(?:\d+|[a-z]\b))?"
    r"|\d+(?:-(?:\d+|[a-z]\b))?)(?!\d)"
    r"\s*(?:,\s*(?:de\s*\.?\s*)?|\.?\s*de\s*\.?\s*)"
    r"(?P<when>\d{1,2}\s*o?\s*[./]\s*\d{1,2}[./](?:\d{4}|\d{2})\b|"
    r"\d{1,2}\s+de\s+[a-z]+\s+de\s+\d{4}|\d{1,2}\s+de\s+\d{4}|[12]\.\d{3}\b|\d{4})",
    re.IGNORECASE,
)
_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5,
    "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
    "novembro": 11, "dezembro": 12,
}


def _full_year(text: str) -> int:
    """A year as written: four digits, or two ("97" -> 1997, "01" -> 2001)."""

    year = int(text)
    if len(text) == 2:
        return 1900 + year if year >= 30 else 2000 + year
    return year


def parse_pt_date(raw) -> Optional[date]:
    """Parse ``data_lei`` ("abril 9, 2000, 12:00 AM") or an annotation date ("19.6.2001"); else None."""

    if raw is None or raw != raw:
        return None
    text = _norm(str(raw)).strip()
    m = re.match(r"([a-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if m and m.group(1) in _PT_MONTHS:
        return date(int(m.group(3)), _PT_MONTHS[m.group(1)], int(m.group(2)))
    m = re.match(r"(\d{1,2})\s*o?\s*[./]\s*(\d{1,2})[./](\d{4}|\d{2})\b", text)
    if m:
        try:
            return date(_full_year(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = re.match(r"(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})", text)
    if m and m.group(2) in _PT_MONTHS:
        return date(int(m.group(3)), _PT_MONTHS[m.group(2)], int(m.group(1)))
    return None


def _act_number(raw) -> Optional[int]:
    if raw is None or raw != raw:
        return None
    digits = re.sub(r"[^\d]", "", str(raw).split("-")[0])
    return int(digits) if digits else None


# "este texto nao substitui o publicado no DOU 3.4.2013" (also "1o .8.2014").
_DOU_DATE = re.compile(
    r"publicad[oa]\s+no\s+d\.?o\.?u\.?\s*(?:de\s*)?(\d{1,2})\s*o?\s*[./]\s*(\d{1,2})\s*[./]\s*(\d{4})")
# "LEI No 10.406, DE 10 DE JANEIRO DE 2002".
_HEADER_DATE = re.compile(
    r"\bn[o°º]?\.?\s*([\d.]+)\s*,?\s*de\s+(\d{1,2})o?\s+de\s+([a-z]+)\s+de\s+(\d{4})")
# "Brasilia, em 2 de abril de 2013".
_SIGNATURE_DATE = re.compile(
    r"brasilia,?\s*(?:em\s*)?(\d{1,2})o?\s+de\s+([a-z]+)\s+de\s+(\d{4})")


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def act_publication_date(text, number: int | None, data_lei=None) -> Optional[date]:
    """Publication date of an act, read from its own text.

    The DOU note when 0–31 days after the header or signature date, else that date;
    then the note, then ``data_lei`` when its day is above 12 (it can swap day and month).
    """

    body = _norm(str(text)) if text is not None and text == text else ""
    m = _DOU_DATE.search(body)
    note = _safe_date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None
    signed = None
    if number is not None:
        for m in _HEADER_DATE.finditer(body[:4000]):
            if _act_number(m.group(1)) == number and m.group(3) in _PT_MONTHS:
                signed = _safe_date(int(m.group(4)), _PT_MONTHS[m.group(3)], int(m.group(2)))
                break
    if signed is None:
        m = _SIGNATURE_DATE.search(body)
        if m and m.group(2) in _PT_MONTHS:
            signed = _safe_date(int(m.group(3)), _PT_MONTHS[m.group(2)], int(m.group(1)))
    if signed is not None:
        if note is not None and 0 <= (note - signed).days <= 31:
            return note
        return signed
    if note is not None:
        return note
    written = parse_pt_date(data_lei)
    return written if written is not None and written.day > 12 else None


def _target_matches(target: dict, article_id: str | None, label: str | None) -> bool:
    """Whether an entry-into-force target covers one article or provision.

    ``target["article"]`` is an act id (every article of that act) or a full
    article id; ``target["label"]``, when set, limits it to that provision
    and the ones nested under it, and never matches the article's own row.
    """

    if article_id is None:
        return False
    scope = target["article"]
    if article_id != scope and not article_id.startswith(scope + ":"):
        return False
    wanted = target.get("label")
    if wanted is None:
        return True
    return label is not None and (label == wanted or label.startswith(wanted + "/"))


class EntryIntoForce:
    """Deferred entry into force per act, from ``entry_into_force.json``.

    Each act has an ``in_force`` day and optional ``targets``; the most specific target applies.
    """

    def __init__(self, acts: dict[str, dict] | None = None):

        self._acts = acts or {}

    @classmethod
    def load(cls, path) -> "EntryIntoForce":
        """Read the table; a missing file yields an empty table."""

        path = Path(path)
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8"))["acts"])

    def __contains__(self, act_id: str) -> bool:
        return act_id in self._acts

    def day(self, act_id: str | None, article_id: str | None = None,
            label: str | None = None) -> Optional[date]:
        """First day in force of one change an act made; None when it took effect on publication."""

        entry = self._acts.get(act_id) if act_id else None
        if entry is None:
            return None
        matches = [t for t in entry.get("targets", ())
                   if _target_matches(t, article_id, label)]
        if matches:
            # Most specific target wins: provision, then article, then act.
            best = max(matches, key=lambda t: (
                len(t.get("label") or "") + 1 if t.get("label") else 0,
                ":" in t["article"]))
            day = best.get("in_force")
        else:
            day = entry.get("in_force")
        return date.fromisoformat(day) if day else None


    def deferred_start(self, act_id: str | None, published: date | None,
                       article_id: str | None = None) -> Optional[date]:
        """First day in force of an act's own wording when later than ``published``, else None."""

        day = self.day(act_id, article_id)
        if day is None or (published is not None and day <= published):
            return None
        return day


class ActDateIndex:
    """Publication dates of the scraped acts by (instrument, number, year), plus entry into force."""

    def __init__(self, tipos, numbers, dates, texts=None,
                 in_force: EntryIntoForce | None = None):
        """Aligned scrape columns ``tipo``, ``num_lei``, ``data_lei`` and optional ``lei_text``."""

        self.in_force = in_force or EntryIntoForce()
        self._by_key: dict[tuple[str, int, int], date] = {}
        self._by_num: dict[tuple[str, int], set[date]] = {}
        tipos, numbers, dates = list(tipos), list(numbers), list(dates)
        texts = list(texts) if texts is not None else [None] * len(tipos)
        for tipo, num, when, text in zip(tipos, numbers, dates, texts):
            key = _DATE_TIPO_KEYS.get(_norm(str(tipo)))
            number = _act_number(num)
            published = act_publication_date(text, number, when)
            if key is None or number is None or published is None:
                continue
            self._by_key.setdefault((key, number, published.year), published)
            self._by_num.setdefault((key, number), set()).add(published)

    def lookup(self, instrument: str, number: int, year: int | None) -> Optional[date]:
        """Publication date of an act, or None when the scrape lacks it."""

        if year is not None:
            hit = self._by_key.get((instrument, number, year))
            if hit is not None:
                return hit
            # Acts around New Year may be cited by either the signing or the publication year.
            hit = self._by_key.get((instrument, number, year - 1))
            if hit is not None and hit.month == 12:
                return hit
            hit = self._by_key.get((instrument, number, year + 1))
            return hit if hit is not None and hit.month == 1 else None
        candidates = self._by_num.get((instrument, number), set())
        return next(iter(candidates)) if len(candidates) == 1 else None


@lru_cache(maxsize=None)
def cited_act(annotation: str) -> Optional[tuple[str, int | None, int | None, Optional[date]]]:
    """(instrument, number, year, date or None) of the act an annotation cites, or None."""

    m = _CITED_ACT.search(_norm(annotation))
    if not m:
        return None
    instrument = next(k for p, k in _DATE_INSTRUMENTS if re.fullmatch(p, m.group("instr")))
    number = _act_number(m.group("num"))
    when = m.group("when")
    full = None
    if re.fullmatch(r"\d\.?\d{3}", when):
        year = int(when.replace(".", ""))
    elif re.fullmatch(r"\d{1,2}\s+de\s+\d{4}", when):
        year = int(when[-4:])
    else:
        full = parse_pt_date(when)
        year = full.year if full else None
    # A mistyped number ("Lei no 1.4002") can read as an impossible year.
    if year is not None and not 1800 <= year <= 2100:
        return None
    return instrument, number, year, full


# Instrument key -> act id prefix, matching ``_slug_tipo`` in laws_pipeline.
_ACT_ID_PREFIX = {
    "lei": "lei",
    "lei_complementar": "lc",
    "lei_delegada": "ld",
    "decreto_lei": "dl",
    "decreto": "decreto",
    "emenda_constitucional": "ec",
    "medida_provisoria": "mp",
}


def annotation_act_id(
    annotation: str | None, index: ActDateIndex | None
) -> Optional[str]:
    """Corpus act id ("lei_13467_2017") of the act an annotation names, or None.

    The year comes from the publication date when known, else from the annotation.
    """

    if not annotation:
        return None
    cited = cited_act(annotation)
    if cited is None:
        return None
    instrument, number, year, when = cited
    prefix = _ACT_ID_PREFIX.get(instrument)
    if prefix is None or number is None:
        return None
    published = when or (index.lookup(instrument, number, year) if index else None)
    # Published in January after the year it names: filed under the named year.
    stamp = (
        year if published and year and published.year == year + 1
        else published.year if published else year
    )
    return f"{prefix}_{number}_{stamp}" if stamp else None


def annotation_date(
    annotation: str | None, index: ActDateIndex | None,
    article_id: str | None = None, label: str | None = None,
) -> tuple[Optional[date], str]:
    """(date, precision) an annotation's wording came into force.

    Precision is ``"day"``, ``"year"`` (date set to 1 January) or ``"none"``.
    """

    if not annotation:
        return None, "none"
    cited = cited_act(annotation)
    if cited is None:
        years = [int(y) for y in _YEAR.findall(annotation) if 1900 <= int(y) <= 2100]
        return (date(max(years), 1, 1), "year") if years else (None, "none")
    instrument, number, year, full = cited
    when, precision = full, "day" if full is not None else "none"
    if when is None and index is not None and number is not None:
        when = index.lookup(instrument, number, year)
        precision = "day" if when is not None else "none"
    if index is not None:
        deferred = index.in_force.day(
            annotation_act_id(annotation, index), article_id, label)
        if deferred is not None and (when is None or deferred > when):
            return deferred, "day"
    if when is not None:
        return when, precision
    return (date(year, 1, 1), "year") if year else (None, "none")


def caput_annotation(text: str) -> str | None:
    """The article's own version annotation, read from its caput line."""

    head = _caput_line(text)
    for pattern in (_ANN_REDACAO, _ANN_INCLUIDO, _ANN_RENUMERADO):
        m = pattern.search(head)
        if m:
            return m.group(0)
    return None


def caput_revocation(text: str) -> str | None:
    """The revocation annotation on the caput line, when present."""

    m = _ANN_REVOGADO.search(_caput_line(text))
    return m.group(0) if m else None


# Provisions below the caput

# "§ 1º" (Planalto), "1o" / "2o-B." or a bare "2 Se ..." (the scrape drops §), or "Paragrafo unico".
_PROV_PARAGRAPH = re.compile(
    r"^\s*(?:"
    r"\u00a7\s*(?P<n1>\d+)\s*[\u00bao\u00b0]?"
    r"(?:-(?P<s1>[A-Z])(?![a-z])|\s+-(?P<s1c>[A-Z])(?=[\s.])|\s+-\s*(?P<s1b>[A-Z])(?=\.))?\.?"
    r"|(?P<n2>\d+)\s*(?:[\u00bao\u00b0](?:\s?-(?P<s2>[A-Z])(?![a-z]))?\.?|(?=\s+[A-Za-z(]))"
    r"|(?P<n3>\d{2})\.(?=\s+(?-i:[A-Z(]))"
    r"|(?:\u00a7\s*)?Paragrafo\s+unico\b"
    r")",
    re.IGNORECASE,
)
# A short title line naming the next block ("Homicidio qualificado").
_PROV_HEADING = re.compile(r"^\s*[A-Z][^.;:]{0,80}(?:\([^)]*\))?\s*$")
# "I - ...", "I – ...", "II. ...", "VII-A - ...", or dash-less "VII contra:".
_PROV_INCISO = re.compile(
    r"^\s*((?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})(?:-[A-Z])?)"
    r"\s*(?:[-\u2013\u2014.]\s|\s+(?=[a-z(]))")
_PROV_ALINEA = re.compile(r"^\s*([a-z])\s*\)\s")
_PROV_ANN = re.compile(
    r"\(\s*(?:(?:Paragrafo|Inciso|Alinea|Caput|Item)\s+)?"
    r"(Redacao\s+(?:dada|da\s+pel|pel|restabelecid)|Restabelecid[oa]|Revigorad[oa]|"
    r"Incluid[oa]|Acrescentad[oa]|Revogad[oa]|Renumera(?:d[oa]|ndo))[^)]{0,160}\)",
    re.IGNORECASE,
)
_PROV_STRIP = re.compile(
    r"\(\s*(?:(?:paragrafo|inciso|alinea|caput|item)\s+)?"
    r"(?:vide|redacao (?:dada|da pel|pel|restabelecid)|restabelecid[oa]|"
    r"incluid[oa]|acrescentad[oa]|renumera(?:d[oa]|ndo)|revogad[oa]|vigencia|"
    r"regulament[oa]|regulamentacao|producao de efeito|revigorad[oa]|"
    r"com redacao|nr)[^()]*\)",
    re.IGNORECASE,
)


@dataclass
class Provision:
    """One paragraph, inciso or alínea of an article occurrence.

    ``label`` is like ``"1o/II/a"``; ``prior_unknown`` omits the provision before
    ``valid_from``; ``unit`` is its top-level provision ("1o/II/a" -> "1o").
    """

    label: str
    text: str
    kind: str = KIND_ORIGINAL
    annotation: str | None = None
    valid_from: date | None = None
    valid_until: date | None = None
    precision: str = "none"
    prior_unknown: bool = False
    source_act: str | None = None
    unit: str | None = None

    def to_record(self) -> dict:
        """JSON-ready dict with ISO dates."""
        return {
            "label": self.label, "text": self.text, "kind": self.kind,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "precision": self.precision, "prior_unknown": self.prior_unknown,
            "annotation": self.annotation, "source_act": self.source_act,
            "unit": self.unit,
        }


def _clean_provision(text: str) -> str:
    text = _PROV_STRIP.sub("", _norm_keep_case(text))
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _norm_keep_case(text: str) -> str:
    """Strip accents but keep case, matching the scrape's own orthography."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


# A penalty line ("Pena - reclusao ...", "Penalidade - multa") is its own provision,
# nested under the one it follows.
_PROV_SUBLINE = re.compile(
    r"^\s*(Penas?|Infracao|Penalidade|Medida\s+administrativa)\b\s*"
    r"(?:[-:\u2013\u2014]|(?=(?:reclusao|detencao|multa|prisao)\b))",
    re.IGNORECASE,
)
_SUBLINE_KEY = {"pena": "pena", "penas": "pena", "infracao": "infracao",
                "penalidade": "penalidade"}


def _split_provision_lines(text: str) -> list[tuple[str, str, str]]:
    """Group the lines below the caput into (label, block, title) triples.

    ``title`` is a preceding heading line; its annotation dates the block only when the
    block has none. Lines before the first paragraph or inciso form the ``caput`` block.
    """

    lines = text.lstrip("\n").split("\n")[1:]
    blocks: list[tuple[str, list[str], list[str]]] = []
    paragraph = inciso = None
    # The provision a penalty line belongs to: the last one opened.
    owner = UNIT_CAPUT
    heading: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        norm = _norm_keep_case(line)
        m = _PROV_PARAGRAPH.match(norm)
        if m:
            number = m.group("n1") or m.group("n2") or m.group("n3")
            suffix = m.group("s1") or m.group("s1c") or m.group("s1b") or m.group("s2")
            paragraph = (
                f"{number}o{'-' + suffix if suffix else ''}"
                if number else "paragrafo_unico"
            )
            inciso = None
            owner = paragraph
            blocks.append((paragraph, [line], heading))
            heading = []
            continue
        m = _PROV_INCISO.match(norm)
        if m:
            inciso = m.group(1)
            owner = f"{paragraph}/{inciso}" if paragraph else inciso
            blocks.append((owner, [line], heading))
            heading = []
            continue
        m = _PROV_ALINEA.match(norm)
        if m:
            parent = "/".join(x for x in (paragraph, inciso) if x)
            owner = f"{parent}/{m.group(1)}" if parent else m.group(1)
            blocks.append((owner, [line], heading))
            heading = []
            continue
        m = _PROV_SUBLINE.match(norm)
        if m:
            key = _SUBLINE_KEY.get(m.group(1).lower(), "medida")
            # An inciso ending in ":" owns the penalty; otherwise it belongs above the list.
            above = paragraph or UNIT_CAPUT
            if blocks and owner not in (paragraph, UNIT_CAPUT) and not (
                    _PROV_ANN.sub("", _norm_keep_case(blocks[-1][1][0])).rstrip().endswith(":")
                    or blocks[-1][0].rsplit("/", 1)[-1] in _SUBLINE_KEY.values()
                    or blocks[-1][0].endswith("/medida")
                    or any(b[0] == f"{above}/{key}" for b in blocks)):
                owner = above
            blocks.append((f"{owner}/{key}", [line], heading))
            heading = []
            continue
        if _PROV_HEADING.match(norm) and len(norm.split("(")[0].split()) <= 8:
            heading.append(line)
            continue
        if blocks:
            blocks[-1][1].append(line)
        else:
            blocks.append((UNIT_CAPUT, [line], heading))
            heading = []
    if heading and blocks:
        # Titles after the last provision belong to the next article.
        blocks[-1][1].extend(_PROV_ANN.sub("", _norm_keep_case(h)) for h in heading)
    # With no provision marker at all, the article is served whole.
    if all(label == UNIT_CAPUT for label, _, _ in blocks):
        return []
    return [(label, "\n".join(ls), "\n".join(title)) for label, ls, title in blocks]


def _closed_at(provision: "Provision", when: date) -> date:
    """Day before ``when``, but never before the provision began."""

    closed = when - timedelta(days=1)
    start = provision.valid_from
    return max(closed, start) if start is not None else closed


def parse_provisions(
    text: str, index: ActDateIndex | None = None, article_id: str | None = None,
) -> list[Provision]:
    """Split an occurrence below its caput into dated provisions, in document order.

    Consecutive blocks with one label form a version chain; a revocation stub closes the
    previous wording, or marks the provision revoked with unknown prior text.
    """

    out: list[Provision] = []
    for label, block, title in _split_provision_lines(text):
        norm_block = _norm_keep_case(block)
        ann = _PROV_ANN.search(norm_block) or _PROV_ANN.search(_norm_keep_case(title))
        if title:
            block = f"{title}\n{block}"
        kind = KIND_ORIGINAL
        annotation = None
        if ann:
            word = ann.group(1).lower()
            kind = (KIND_REVOGADO if word.startswith("revog")
                    else KIND_INCLUIDO if word.startswith(("inclu", "acresc"))
                    else KIND_REDACAO)
            annotation = ann.group(0)
        when, precision = annotation_date(annotation, index, article_id, label)
        source_act = annotation_act_id(annotation, index)
        cleaned = _clean_provision(block)
        is_stub = kind == KIND_REVOGADO and len(re.findall(r"[A-Za-z]{3,}", cleaned)) <= 3

        previous = next((p for p in reversed(out) if p.label == label), None)
        if kind == KIND_REVOGADO:
            # A revocation closes only a wording that started before it.
            if previous is not None and previous.text and when is not None and (
                    previous.valid_from is None or previous.valid_from < when):
                previous.valid_until = _closed_at(previous, when)
            if is_stub:
                if previous is None or not previous.text:
                    out.append(Provision(label, "", kind, annotation, None,
                                         (when - timedelta(days=1)) if when else None,
                                         precision, prior_unknown=True,
                                         source_act=source_act))
                continue
            # A revoked block with text and no earlier wording held until the day before.
            if previous is None or not previous.text or (
                    when is not None and previous.valid_from is not None
                    and previous.valid_from >= when):
                out.append(Provision(
                    label, cleaned, KIND_ORIGINAL, annotation, None,
                    (when - timedelta(days=1)) if when else None, "none",
                    source_act=source_act))
            continue
        prior_unknown = kind == KIND_REDACAO and previous is None
        if prior_unknown and _renumbered_only(annotation):
            # Same text under a new number was in force before, under its old label.
            prior_unknown = False
            moved = _same_text_earlier(out, cleaned, label)
            if moved is not None and when is not None:
                if moved.valid_until is None:
                    moved.valid_until = _closed_at(moved, when)
            else:
                when, precision = None, "none"
        if previous is not None and previous.text and when is not None:
            previous.valid_until = _closed_at(previous, when)
        out.append(Provision(
            label, cleaned, kind, annotation, when, None, precision,
            prior_unknown=prior_unknown, source_act=source_act,
        ))
    return out


_RENUMBERED = re.compile(r"renumerad[oa]|renumerando")
_REWORDED = re.compile(r"redacao|alterad|modificad")
# A provision's marker on normalised text, so one wording under two labels compares equal.
_LABEL_HEAD = re.compile(
    r"^\s*(?:paragrafo\s+unico|\d+\s*o?(?:-[a-z](?![a-z]))?|[a-z]\)|[ivxlc]+(?:-[a-z])?)"
    r"\s*[-.:\u2013\u2014]?\s*")


def _renumbered_only(annotation: str | None) -> bool:
    """A renumbering annotation that brings no new wording."""
    text = _norm(annotation or "")
    return bool(_RENUMBERED.search(text)) and not _REWORDED.search(text)


def _same_text_earlier(out: list["Provision"], text: str, label: str):
    """The earlier provision under another label with the same wording."""
    body = _LABEL_HEAD.sub("", _norm(text), count=1)
    if not body:
        return None
    for provision in reversed(out):
        if provision.label != label and provision.text and (
                _LABEL_HEAD.sub("", _norm(provision.text), count=1) == body):
            return provision
    return None


# Annex items: a code ("1.01 - Analise de sistemas") or a term in caps ("VIADUTO - obra ...").
_ANNEX_ITEM_CODE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s*[-–]?\s+(?=\S)")
_ANNEX_TERM = re.compile(r"^\s*([A-Z][A-Z\s/()-]{2,40}?)\s+[-–]\s+(?=\S)")


# Articles above this size are split into one unit per top-level provision.
UNIT_SPLIT_TOKENS = 500


def assign_units(provisions: list[Provision]) -> None:
    """Set each provision's ``unit`` to its top-level provision ("1o/II/a" -> "1o"), in place."""

    for provision in provisions:
        provision.unit = provision.label.split("/")[0] or UNIT_CAPUT


def parse_annex_items(text: str) -> list[Provision]:
    """Split an annex into provisions labelled by item code or term; empty for a plain table."""

    items: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        norm = _norm_keep_case(line)
        match = _ANNEX_ITEM_CODE.match(norm) or _ANNEX_TERM.match(norm)
        if match:
            items.append((match.group(1).strip(), [line.strip()]))
        elif items:
            items[-1][1].append(line.strip())
    return [
        Provision(label=label, text=_normalize_ws("\n".join(lines)))
        for label, lines in items
    ]


def _normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def article_text_at(
    caput: str, provisions: list[dict] | list[Provision], cutoff: date | None
) -> str:
    """Caput plus the provisions in force at ``cutoff`` (None: current ones), one per line."""

    parts = [caput.strip()]
    for prov in provisions:
        rec = prov.to_record() if isinstance(prov, Provision) else prov
        text = rec.get("text") or ""
        if not text:
            continue
        start = rec.get("valid_from")
        end = rec.get("valid_until")
        start = date.fromisoformat(start) if isinstance(start, str) else start
        end = date.fromisoformat(end) if isinstance(end, str) else end
        if cutoff is not None:
            if start is not None and start > cutoff:
                continue
            if end is not None and end < cutoff:
                continue
        elif end is not None:
            continue
        parts.append(text)
    return "\n".join(p for p in parts if p)


# Article corpus access

def load_article_versions(csv_path) -> dict[str, list[dict]]:
    """Load an article corpus as article_id -> wordings, oldest first."""

    import json

    import pandas as pd

    def _opt_int(value):
        return None if value is None or pd.isna(value) else int(value)

    def _opt_date(value):
        if value is None or pd.isna(value):
            return None
        return date.fromisoformat(str(value))

    df = pd.read_csv(csv_path, low_memory=False)
    dated = "valid_from_date" in df.columns
    articles: dict[str, list[dict]] = {}
    for row in df.itertuples(index=False):
        row = row._asdict()
        text = str(row["article_text"])
        act_header = row.get("act_header")
        header = f"[{act_header}]\n" if act_header and not pd.isna(act_header) else ""
        provisions = row.get("provisions")
        articles.setdefault(row["article_id"], []).append({
            "article_id": row["article_id"],
            "act_id": row.get("act_id"),
            "lei_label": row["lei_label"],
            "num_lei": row.get("num_lei"),
            "ano": _opt_int(row.get("ano")),
            "area": row.get("area"),
            "path": row.get("path"),
            "tokens": _opt_int(row.get("tokens")),
            "version_year": _opt_int(row.get("version_year")),
            "valid_from": _opt_int(row.get("valid_from")),
            "valid_until": _opt_int(row.get("valid_until")),
            "valid_from_date": _opt_date(row.get("valid_from_date")) if dated else None,
            "valid_until_date": _opt_date(row.get("valid_until_date")) if dated else None,
            # article_text already opens with "Art. N"; add only the act breadcrumb.
            "display_text": header + text,
            "header": header,
            "caput_line": text.split("\n", 1)[0],
            "provisions": (
                json.loads(provisions)
                if dated and isinstance(provisions, str) else None
            ),
            "split": bool(row.get("split")) if "split" in row else False,
            "n_units": _opt_int(row.get("n_units")) or 1,
        })
    for versions in articles.values():
        # Order by day where known, as ``wording_at`` scans it.
        versions.sort(key=lambda v: (
            v["valid_from_date"] is None, v["valid_from_date"])
            if dated else (v["valid_from"] is None, v["valid_from"]))
    return articles


def load_unit_versions(csv_path) -> dict[str, list[dict]]:
    """Load an article corpus as unit_id -> wordings, oldest first.

    A short article is one unit; a split one is its caput plus ``<article_id>#<unit>`` per provision.
    """

    units: dict[str, list[dict]] = {}
    for article_id, versions in load_article_versions(csv_path).items():
        for version in versions:
            if not version.get("split"):
                units.setdefault(article_id, []).append(version)
                continue
            for unit_id, record in _split_into_units(version).items():
                units.setdefault(unit_id, []).append(record)
    for chain in units.values():
        chain.sort(key=lambda v: (v["valid_from_date"] is None, v["valid_from_date"]))
    return units


def _split_into_units(version: dict) -> dict[str, dict]:
    """Break one wording into unit_id -> record for its caput and each provision unit."""

    article_id = version["article_id"]
    grouped: dict[str, list[dict]] = {}
    for provision in version["provisions"] or []:
        grouped.setdefault(provision.get("unit") or UNIT_CAPUT, []).append(provision)

    caput_provisions = grouped.get(UNIT_CAPUT, [])
    out: dict[str, dict] = {
        f"{article_id}#{UNIT_CAPUT}": {
            **version,
            "unit_id": f"{article_id}#{UNIT_CAPUT}",
            "unit": UNIT_CAPUT,
            "provisions": caput_provisions,
            "display_text": version["header"] + article_text_at(
                version["caput_line"], caput_provisions, None),
        }
    }
    for unit, provisions in grouped.items():
        if unit == UNIT_CAPUT:
            continue
        starts = [p["valid_from"] for p in provisions if p.get("valid_from")]
        ends = [p["valid_until"] for p in provisions if p.get("valid_until")]
        open_start = any(not p.get("valid_from") for p in provisions)
        open_end = any(not p.get("valid_until") for p in provisions)
        unit_id = f"{article_id}#{unit}"
        out[unit_id] = {
            **version,
            "unit_id": unit_id,
            "unit": unit,
            "provisions": provisions,
            # The unit's own head line, with the article marker for context.
            "display_text": version["header"] + article_text_at(
                _unit_head(version), provisions, None),
            "valid_from_date": (
                None if open_start or not starts
                else date.fromisoformat(min(starts))),
            "valid_until_date": (
                None if open_end or not ends
                else date.fromisoformat(max(ends))),
        }
    return out


def _unit_head(version: dict) -> str:
    """Opening line shown above a unit: its article's marker, for context."""

    match = _ARTICLE_MARKER.match(version["caput_line"])
    return f"{match.group(0).strip()} —" if match else ""


def _earlier_provisions(versions: list[dict], version: dict) -> list:
    """Provisions of ``version`` stored on a later wording, after a caput rewrite.

    A later wording holding a provision closed before its own start stores them; its
    provisions overlapping ``version``'s interval belong to ``version``.
    """

    def _day(value):
        return date.fromisoformat(value) if isinstance(value, str) else value

    own_start, own_end = version["valid_from_date"], version["valid_until_date"]
    for later in versions[versions.index(version) + 1:]:
        start = later["valid_from_date"]
        if start is None:
            continue
        records = [
            (prov, prov.to_record() if isinstance(prov, Provision) else prov)
            for prov in later["provisions"]
        ]
        if not any(_day(rec.get("valid_until")) is not None
                   and _day(rec.get("valid_until")) < start
                   for _, rec in records):
            continue
        overlapping = []
        for prov, rec in records:
            p_start, p_end = _day(rec.get("valid_from")), _day(rec.get("valid_until"))
            if own_start is not None and p_end is not None and p_end < own_start:
                continue
            if own_end is not None and p_start is not None and p_start > own_end:
                continue
            overlapping.append(prov)
        return overlapping
    return []


def wording_at(
    versions: list[dict], exam_year: int | None, cutoff: date | None = None
) -> dict | None:
    """Wording of one article in force at ``cutoff`` (by day) or else ``exam_year``; None if none.

    By day, provisions are assembled for that day; by year, the wording is shown in full.
    """

    if not versions:
        return None
    if cutoff is not None and versions[0]["valid_from_date"] is not None:
        for version in versions:
            start, end = version["valid_from_date"], version["valid_until_date"]
            if start is not None and start > cutoff:
                continue
            if end is not None and end < cutoff:
                continue
            # No parsed provisions: keep the body, unless a later wording stores them.
            if not version["provisions"]:
                earlier = _earlier_provisions(versions, version)
                if not earlier:
                    return version
                return {
                    **version,
                    "display_text": version["header"] + article_text_at(
                        version["caput_line"], earlier, cutoff),
                }
            return {
                **version,
                "display_text": version["header"] + article_text_at(
                    version["caput_line"], version["provisions"], cutoff),
            }
        return None
    if exam_year is None:
        return versions[-1]
    for version in versions:
        valid_from = version["valid_from"]
        valid_until = version["valid_until"]
        if valid_from is not None and valid_from > exam_year:
            continue
        if valid_until is not None and valid_until < exam_year:
            continue
        return version
    return None
