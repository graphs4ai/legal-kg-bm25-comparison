"""Rule-based legal-citation parser: resolves cross-references in article text to corpus act/article ids.

Act numbers are unique only within an instrument type (Lei vs Decreto-Lei); a
number-only match is a flagged fallback when unambiguous.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

# Relation emitted for a citation, by the language introducing it.
RELATION_REFERS_TO = "refers_to"
RELATION_AMENDS = "amends"
RELATION_REVOKES = "revokes"

# Instrument-type keys used for resolution, normalized from the corpus `tipo`.
TIPO_CONSTITUICAO = "constituicao"
TIPO_EMENDA = "emenda_constitucional"
TIPO_LEI = "lei_ordinaria"
TIPO_LEI_COMPLEMENTAR = "lei_complementar"
TIPO_DECRETO_LEI = "decreto_lei"
TIPO_DECRETO = "decreto"
TIPO_ATO_CFOAB = "ato_cfoab"

# Normative hierarchy: lower ranks outrank higher ones.
HIERARCHY_LEVEL = {
    TIPO_CONSTITUICAO: 0,
    TIPO_EMENDA: 0,
    TIPO_LEI_COMPLEMENTAR: 1,
    TIPO_LEI: 2,
    TIPO_DECRETO_LEI: 2,
    TIPO_DECRETO: 3,
    TIPO_ATO_CFOAB: 4,
}

# Short names of the major codes -> corpus act ids (checked present at load time).
ACT_ALIASES = {
    "constituicao federal": "cf_1988",
    "constituicao da republica": "cf_1988",
    "constituicao": "cf_1988",
    "cf": "cf_1988",
    "carta magna": "cf_1988",
    "codigo civil": "lei_10406_2002",
    "cc": "lei_10406_2002",
    "codigo de processo civil": "lei_13105_2015",
    "cpc": "lei_13105_2015",
    "codigo penal": "dl_2848_1940",
    "cp": "dl_2848_1940",
    "codigo de processo penal": "dl_3689_1941",
    "cpp": "dl_3689_1941",
    "consolidacao das leis do trabalho": "dl_5452_1943",
    "clt": "dl_5452_1943",
    "codigo de defesa do consumidor": "lei_8078_1990",
    "cdc": "lei_8078_1990",
    "estatuto da crianca e do adolescente": "lei_8069_1990",
    "eca": "lei_8069_1990",
    "estatuto da advocacia": "lei_8906_1994",
    "estatuto da oab": "lei_8906_1994",
    "eaoab": "lei_8906_1994",
    "codigo tributario nacional": "lei_5172_1966",
    "ctn": "lei_5172_1966",
    "codigo eleitoral": "lei_4737_1965",
    "codigo de transito brasileiro": "lei_9503_1997",
    "ctb": "lei_9503_1997",
    "lindb": "dl_4657_1942",
}

# Corpus `tipo` -> normalized instrument key.
_TIPO_MAP = {
    "constituicao": TIPO_CONSTITUICAO,
    "emenda constitucional": TIPO_EMENDA,
    "lei ordinaria": TIPO_LEI,
    "lei complementar": TIPO_LEI_COMPLEMENTAR,
    "decreto-lei": TIPO_DECRETO_LEI,
    "decreto lei": TIPO_DECRETO_LEI,
    "decreto": TIPO_DECRETO,
    "ato normativo cfoab": TIPO_ATO_CFOAB,
}

# Instrument named in a citation -> key. Order matters: "decreto-lei" before "decreto".
_INSTRUMENT_PATTERNS = [
    (r"emenda\s+constitucional", TIPO_EMENDA),
    (r"lei\s+complementar", TIPO_LEI_COMPLEMENTAR),
    (r"decreto[-\s]lei", TIPO_DECRETO_LEI),
    (r"constitui[cç][aã]o", TIPO_CONSTITUICAO),
    (r"decreto", TIPO_DECRETO),
    (r"lei", TIPO_LEI),
]

_ART_TOKEN = r"\d+(?:[.\-]\d+)*(?:\s*[ºo°](?![a-z]))?(?:\s*-\s*[A-Z](?![A-Za-z]))?"

# "art. 5º", "arts. 12 e 13", "artigo 1.234-A"; group 1 singular/plural, group 2 the numbers.
# The number never sits on the next line ("deste artigo.\n3o" is a lost §).
_ARTICLE_RE = re.compile(
    rf"\b(arts?\.?|artigos?(?!\.))[^\S\n]*({_ART_TOKEN}(?:\s*(?:,|e|ou|a)\s*{_ART_TOKEN})*)",
    re.IGNORECASE,
)

# The ADCT named after an article reference, in ``_norm`` form.
_ADCT_RE = re.compile(
    r"\badct\b|\bato das disposicoes constitucionais transitorias\b")
# After an alias, a year or a state that makes it another instrument.
_ALIAS_YEAR_RE = re.compile(r"\s*(?:federal\s+)?(?:de\s+|/\s*)?(1[89]\d\d|20\d\d)\b")
_STATE_INSTRUMENT_RE = re.compile(r"\s*(?:estadual|do estado|dos estados|de estado)\b")

# "deste Ato" inside the Constitution's own text means the ADCT.
_THIS_ACT_RE = re.compile(r"\b(?:deste|neste|desse|nesse) ato\b")
# "Lei nº 8.906", "Decreto-Lei n. 2.848", "Emenda Constitucional nº 103"
_ACT_NUM_RE = re.compile(
    r"\b(emenda\s+constitucional|lei\s+complementar|decreto[-\s]lei|lei|decreto)\s*"
    r"n?[ºo°.]{0,2}\s*(\d[\d.]*)(?:\s*[/,]?\s*(?:de\s+)?(\d{4}))?",
    re.IGNORECASE,
)
# Explicitly self-referential: "desta Lei", "deste Código"
_SELF_REF_RE = re.compile(
    r"\b(?:desta|deste|nesta|neste)\s+(?:lei|c[oó]digo|constitui[cç][aã]o|decreto)",
    re.IGNORECASE,
)

# The article's own opening marker ("Art. 5o ..."), not a citation.
_SELF_MARKER_RE = re.compile(
    rf"^\s*arts?\.?\s*(?:igos?)?\.?\s*{_ART_TOKEN}", re.IGNORECASE
)

_REVOKE_RE = re.compile(
    r"revoga(?:m|-se|do|da|dos|das)?\b|fica[m]?\s+revogad", re.IGNORECASE
)
_AMEND_RE = re.compile(
    r"passa(?:m)?\s+a\s+vigorar|nova\s+reda[cç][aã]o|altera(?:d[oa]|r|\b)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def act_number(num) -> str | None:
    """An act number in registry form ("7.347", 7347.0, "7347.0" -> "7347")."""

    if num is None or (isinstance(num, float) and math.isnan(num)):
        return None
    if isinstance(num, float) and num.is_integer():
        num = int(num)
    # A float that went through str() ("7347.0"); a thousands dot has 3 digits.
    text = re.sub(r"^(\d+)\.0$", r"\1", str(num).strip())
    return text.replace(".", "") or None


def normalize_tipo(tipo: str) -> str | None:
    """Map a corpus ``tipo`` (e.g. "Lei Ordinária") to an instrument key, or None."""

    return _TIPO_MAP.get(_norm(tipo))


def hierarchy_level(tipo: str) -> int | None:
    """Normative hierarchy level of a corpus ``tipo`` (Constituição 0 … Ato CFOAB 4), or None."""

    key = normalize_tipo(tipo)
    return HIERARCHY_LEVEL.get(key) if key else None


def normalize_article_number(raw: str) -> str:
    """Canonical article number: ``"5º"`` -> ``"5"``, ``"1.234"`` -> ``"1234"``, ``"5 - A"`` -> ``"5-A"``."""

    text = _norm(raw).replace(".", "")
    text = re.sub(r"\s*-\s*", "-", text).strip()
    # The ordinal sits before a suffix too ("3º-B").
    text = re.sub(r"[ºo°]\s*(?=-|$)", "", text).strip()
    return text.upper()


def _split_article_numbers(blob: str) -> list[str]:
    """Split "12 e 13" into ["12", "13"], deduplicated; ranges ("12 a 15") keep endpoints only."""

    # A letter after a hyphen is the article's suffix ("29-A"), never "a"/"e".
    blob = re.sub(r"\s*-\s*", "-", blob)
    parts = re.split(r"\s*(?:,|(?<![-\w])(?:[ae]|ou)(?!\w))\s*", blob, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        number = normalize_article_number(part)
        if number and number not in out:
            out.append(number)
    return out


@dataclass
class Citation:
    """One normative cross-reference found in article text.

    ``resolution`` is "alias", "tipo_number", "number_only", "self" or "unresolved".
    """

    raw: str
    relation: str = RELATION_REFERS_TO
    target_act_id: str | None = None
    article_numbers: list[str] = field(default_factory=list)
    scope: str = "unknown"
    tipo_hint: str | None = None
    num_hint: str | None = None
    ano_hint: int | None = None
    resolution: str = "unresolved"
    container: str | None = None

    def target_article_ids(self) -> list[str]:
        """Resolved ``<act_id>:art_<n>`` ids; empty when unresolved or no article cited."""

        if not self.target_act_id:
            return []
        prefix = (f"{self.target_act_id}:{self.container}" if self.container
                  else self.target_act_id)
        return [f"{prefix}:art_{n}" for n in self.article_numbers]


class ActRegistry:
    """Resolves citation hints to corpus act ids; ``acts`` maps act_id -> metadata."""

    def __init__(self, acts: dict[str, dict]):

        self.acts = acts
        self._by_tipo_num: dict[tuple[str, str], list[str]] = {}
        self._by_num: dict[str, list[str]] = {}

        for act_id, meta in acts.items():
            num = act_number(meta.get("num"))
            if not num:
                continue
            self._by_num.setdefault(num, []).append(act_id)
            tipo_key = meta.get("tipo_key")
            if tipo_key:
                self._by_tipo_num.setdefault(
                    (tipo_key, num), []).append(act_id)

        # Only keep aliases whose target is actually in this corpus.
        self.aliases = {
            alias: act_id
            for alias, act_id in ACT_ALIASES.items()
            if act_id in acts
        }

    def resolve(
        self,
        tipo_key: str | None,
        num: str | None,
        ano: int | None = None,
    ) -> tuple[str | None, str]:
        """Resolve an instrument hint to (act_id or None, resolution kind).

        Exact (tipo, number) first; number-only when unambiguous in the corpus
        (the Código Penal is often cited as "Lei 2.848"). ``ano`` breaks ties.
        """

        if not num:
            return None, "unresolved"
        num = num.replace(".", "").strip()

        if tipo_key:
            matches = self._by_tipo_num.get((tipo_key, num), [])
            if ano:
                exact = [a for a in matches if self.acts[a].get("ano") == ano]
                if exact:
                    return exact[0], "tipo_number"
            if len(matches) == 1:
                return matches[0], "tipo_number"
            if matches:
                return matches[0], "tipo_number"

        candidates = self._by_num.get(num, [])
        if ano:
            exact = [a for a in candidates if self.acts[a].get("ano") == ano]
            if len(exact) == 1:
                return exact[0], "number_only"
        if len(candidates) == 1:
            return candidates[0], "number_only"

        return None, "unresolved"

    def find_alias(self, text: str) -> tuple[int, str] | None:
        """(position in ``_norm(text)``, act id) of the earliest alias, longest on a tie.

        The act id is None when the following words name another instrument
        ("Constituição de 1967").
        """

        normalized = _norm(text)
        best = None
        for alias in self.aliases:
            act_id = self.aliases[alias]
            match = re.search(rf"\b{re.escape(alias)}\b", normalized)
            if match and (best is None or (match.start(), -len(alias)) < best[:2]):
                other = self._other_instrument(normalized[match.end():], act_id)
                best = (match.start(), -len(alias), None if other else act_id)
        return None if best is None else (best[0], best[2])

    def _other_instrument(self, after: str, act_id: str) -> bool:
        """Whether the words after an alias name another year's or a state's instrument."""

        if _STATE_INSTRUMENT_RE.match(after):
            return True
        year = _ALIAS_YEAR_RE.match(after)
        return bool(year) and int(year.group(1)) != self.acts.get(act_id, {}).get("ano")

    def resolve_alias(self, text: str) -> str | None:
        """Resolve a short-name citation ("CPC", "Código Civil") to an act id, or None."""

        normalized = _norm(text)
        # Longest alias first so "codigo de processo civil" beats "codigo".
        for alias in sorted(self.aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return self.aliases[alias]
        return None


def _relation_for(context: str) -> str:
    """Classify the relation implied by the text around a citation."""

    if _REVOKE_RE.search(context):
        return RELATION_REVOKES
    if _AMEND_RE.search(context):
        return RELATION_AMENDS
    return RELATION_REFERS_TO


# References by position ("artigo anterior", "artigo seguinte"); the caller supplies neighbours.
_RELATIVE_PREV = re.compile(
    r"\bartigos?\s+(?:imediatamente\s+)?(?:anterior(?:es)?|antecedentes?)\b"
    r"|\bn[úu]meros?\s+anterior(?:es)?\b",
    re.IGNORECASE,
)
_RELATIVE_NEXT = re.compile(
    r"\bartigos?\s+(?:seguintes?|subsequentes?|subseq[üu]entes?)\b",
    re.IGNORECASE,
)


def positional_citations(
    text: str,
    source_act_id: str,
    previous_number: str | None = None,
    next_number: str | None = None,
) -> list[Citation]:
    """Resolve "artigo anterior/seguinte" to the neighbouring article in document order.

    A plural form resolves to the nearest neighbour only.
    """

    if not text:
        return []
    citations: list[Citation] = []
    for pattern, number in ((_RELATIVE_PREV, previous_number),
                            (_RELATIVE_NEXT, next_number)):
        match = pattern.search(text)
        if not match or not number:
            continue
        citations.append(Citation(
            raw=match.group(0).strip(),
            relation=_relation_for(text[max(0, match.start() - 90): match.end() + 90]),
            target_act_id=source_act_id,
            article_numbers=[number],
            scope="internal",
            resolution="positional",
        ))
    return citations


def extract_citations(
    text: str,
    source_act_id: str,
    registry: ActRegistry,
    window: int = 90,
    source_year: int | None = None,
    skip_self_marker: bool = True,
) -> list[Citation]:
    """Extract normative cross-references from one article's text, deduplicated, in order.

    Args:
        window: characters after an article mention scanned for its act; none found means internal.
        source_year: year of the article's wording; target acts enacted later stay unresolved
            (a 1997 "Código de Processo Civil" is CPC/1973). None disables the guard.
        skip_self_marker: ignore a leading "Art. N"; False for a bare mention like "art. 4o".
    """

    if not text:
        return []

    def _guard(act_id: str | None, resolution: str) -> tuple[str | None, str]:
        """Reject a target enacted after the citing text was written."""
        if not act_id or source_year is None or act_id == source_act_id:
            return act_id, resolution
        target_year = registry.acts.get(act_id, {}).get("ano")
        if target_year and target_year > source_year:
            return None, "anachronistic"
        return act_id, resolution

    citations: list[Citation] = []
    consumed: list[tuple[int, int]] = []

    # Skip the leading "Art. N" so its window does not take the first citation's act.
    self_marker = _SELF_MARKER_RE.match(text) if skip_self_marker else None
    skip_until = self_marker.end() if self_marker else 0

    for match in _ARTICLE_RE.finditer(text):
        if match.start() < skip_until:
            continue
        blob = match.group(2)
        # After a single "art." a comma leads to its paragraph or inciso ("art. 121, 2o").
        if not match.group(1).rstrip(".").lower().endswith("s"):
            blob = blob.split(",", 1)[0]
        numbers = _split_article_numbers(blob)
        if not numbers:
            continue

        tail = text[match.end(): match.end() + window]
        context = text[max(0, match.start() - window): match.end() + window]

        act_id: str | None = None
        resolution = "self"
        tipo_hint = num_hint = None
        ano_hint = None
        container = None

        # The act a reference belongs to is the mention nearest after it.
        alias_hit = registry.find_alias(tail)
        act_match = _ACT_NUM_RE.search(tail)
        self_ref = _SELF_REF_RE.search(tail)
        tail_norm = _norm(tail)
        adct = _ADCT_RE.search(tail_norm) or (
            _THIS_ACT_RE.search(tail_norm) if source_act_id == "cf_1988" else None)
        mentions = [
            (hit.start(), kind) for hit, kind in (
                (act_match, "number"), (self_ref, "self"), (adct, "adct"))
            if hit is not None
        ]
        if alias_hit is not None:
            mentions.append((alias_hit[0], "alias"))
        nearest = min(mentions)[1] if mentions else "self"
        alias_id = alias_hit[1] if alias_hit else None

        if nearest == "adct" and "cf_1988" in registry.acts:
            act_id, resolution, container = "cf_1988", "alias", "adct"
        elif nearest == "number":
            raw_instrument = _norm(act_match.group(1))
            tipo_hint = next(
                (key for pat, key in _INSTRUMENT_PATTERNS
                 if re.fullmatch(pat, raw_instrument)),
                None,
            )
            num_hint = act_match.group(2)
            ano_hint = (
                int(act_match.group(3)) if act_match.group(3) else None
            )
            act_id, resolution = _guard(
                *registry.resolve(tipo_hint, num_hint, ano_hint))
            consumed.append((act_match.start() + match.end(),
                             act_match.end() + match.end()))
        elif nearest == "alias" and alias_id is None:
            act_id, resolution = None, "unresolved"
        elif nearest == "alias":
            act_id, resolution = _guard(alias_id, "alias")
        else:
            act_id, resolution = source_act_id, "self"

        citations.append(
            Citation(
                raw=match.group(0).strip(),
                relation=_relation_for(context),
                target_act_id=act_id,
                article_numbers=numbers,
                scope=(
                    "internal" if act_id == source_act_id
                    else "external" if act_id else "unknown"
                ),
                tipo_hint=tipo_hint,
                num_hint=num_hint,
                ano_hint=ano_hint,
                resolution=resolution,
                container=container,
            )
        )

    # Act-level citations with no article number, skipping spans already attributed.
    for match in _ACT_NUM_RE.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        raw_instrument = _norm(match.group(1))
        tipo_hint = next(
            (key for pat, key in _INSTRUMENT_PATTERNS
             if re.fullmatch(pat, raw_instrument)),
            None,
        )
        ano_hint = int(match.group(3)) if match.group(3) else None
        act_id, resolution = _guard(
            *registry.resolve(tipo_hint, match.group(2), ano_hint))
        if not act_id or act_id == source_act_id:
            continue
        context = text[max(0, match.start() - window): match.end() + window]
        citations.append(
            Citation(
                raw=match.group(0).strip(),
                relation=_relation_for(context),
                target_act_id=act_id,
                article_numbers=[],
                scope="external",
                tipo_hint=tipo_hint,
                num_hint=match.group(2),
                ano_hint=ano_hint,
                resolution=resolution,
            )
        )

    seen: set[tuple] = set()
    unique: list[Citation] = []
    for citation in citations:
        key = (
            citation.relation,
            citation.target_act_id,
            citation.container,
            tuple(citation.article_numbers),
        )
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


def canonical_mention_id(
    text: str, registry: ActRegistry, source_act_id: str | None = None
) -> str | None:
    """Canonical node id of a Law/Article mention: ``article:<act>:art_<n>``, ``act:<act_id>`` or None.

    ``source_act_id`` is assumed for bare article mentions.
    """

    citations = extract_citations(
        text, source_act_id or "", registry,
        window=len(text) + 1, skip_self_marker=False,
    )
    for citation in citations:
        ids = citation.target_article_ids()
        if ids:
            return f"article:{ids[0]}"
        if citation.target_act_id:
            return f"act:{citation.target_act_id}"

    alias_id = registry.resolve_alias(text)
    return f"act:{alias_id}" if alias_id else None
