"""GraphRAG retriever: rank graph nodes in force at the question's cutoff, then
serve the top articles' wording, optionally with graph relations.

Ranking is a Personalized PageRank walk (``ranking="ppr"``) or seed linking
plus k-hop expansion (``ranking="expand"``).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.laws_citations import ACT_ALIASES
from src.data.laws_versioning import load_article_versions, wording_at
from src.kg.config import GraphRAGConfig
from src.kg.graph_store import GraphStore
from src.rag.text_utils import apply_context_budget, normalize_text

# Edges that carry a normative cross-reference, followed during expansion.
_CITATION_RELATIONS = frozenset({"refers_to", "amends", "revokes"})

# Semantic relations an article asserts about an entity, with their labels.
_ASSERTION_LABELS = {
    "regulates": "Regula", "requires": "Exige", "prohibits": "Proíbe",
    "grants": "Concede", "attributes_competence_to": "Atribui competência",
    "establishes_sanction": "Sanção", "defines": "Define",
}
_MODALITY_LABELS = {
    "permission": "Permite", "prohibition": "Proíbe", "exception": "Exceção",
}
# Words left out of the stem overlap between an assertion and a wording.
_STOPWORDS = frozenset(
    "de da do das dos e a o os as em no na nos nas por para com ou que ao aos se "
    "um uma sua seu suas seus pelo pela pelos pelas sobre entre sem".split())
# Number words compared like numerals (deadlines, amounts, penalty ranges).
_NUMBER_WORDS = frozenset(
    "dois duas tres quatro cinco seis sete oito nove dez onze doze treze quatorze "
    "catorze quinze dezesseis dezessete dezoito dezenove vinte trinta quarenta "
    "cinquenta sessenta setenta oitenta noventa cem cento duzentos trezentos "
    "quinhentos mil milhao milhoes bilhao bilhoes dobro triplo metade terco".split())
_DIGIT_SEPARATOR_RE = re.compile(r"(?<=\d)[.,](?=\d)")


def _stems(text: str) -> set[str]:
    """Five-letter stems of the content words of a text."""

    return {w[:5] for w in _match_form(text).split() if len(w) >= 3 and w not in _STOPWORDS}


def _numbers(text: str) -> set[str]:
    """Numbers a text states, as numerals or number words (0 and 1 left out)."""

    words = _match_form(_DIGIT_SEPARATOR_RE.sub("", text)).split()
    return {w for w in words if (w.isdigit() and w not in ("0", "1")) or w in _NUMBER_WORDS}


class _WordingMatch:
    """Tell whether an entity name comes from an out-of-force wording: verbatim
    only there, mostly covered by its stems (>= 3/4 vs < 3/4 in force), or
    stating a number only it states.
    """

    def __init__(self, wording: str, out_of_force: list[str]):
        self._form = _match_form(wording)
        self._stems = _stems(wording)
        self._numbers = _numbers(wording)
        self._out = [(_match_form(t), _stems(t), _numbers(t)) for t in out_of_force]

    def out_of_force(self, name: str) -> bool:
        form = _match_form(name)
        if form in self._form:
            return False
        if any(form in out_form for out_form, _, _ in self._out):
            return True
        stems = _stems(name)
        if stems:
            covered_in = len(stems & self._stems) / len(stems)
            covered_out = max((len(stems & out) / len(stems) for _, out, _ in self._out),
                              default=0.0)
            if covered_out >= 0.75 and covered_in < 0.75 and covered_out - covered_in >= 0.25:
                return True
        missing = _numbers(name) - self._numbers
        return any(missing & out for _, _, out in self._out)


# Order of the label lines when an article is shown as its assertions.
_LABEL_ORDER = ("Concede", "Exige", "Proíbe", "Permite", "Exceção", "Regula",
                "Define", "Atribui competência", "Sanção")

_ARTICLE_UNIT_RE = re.compile(r"^(?P<law_id>.+):art_(?P<num>[^_].*?)(?:__\d+)?$")

# Trailing chunk index on a corpus doc_id ("cf_1988_c001" -> "cf_1988").
_DOC_CHUNK_SUFFIX_RE = re.compile(r"_c\d+$")

_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")

# Article reference in ``_reference_form`` text ("art 5o", "arts 3o virgula 4o e 5o", "arts 29 a 31").
_ART_REF_RE = re.compile(
    r"\b(art(?:igo)?s?)\s+"
    r"(\d+[a-z]?(?:\s(?:e|ou|a|virgula)\s\d+[a-z]?)*)")

# Thousands separator in a raw number ("1.654" -> "1654").
_THOUSANDS_RE = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{3})(?!\d)")

# Letter suffix of an article number ("29-A", "3º-B"), joined to the number.
_ART_SUFFIX_RE = re.compile(
    r"(\d)[º°]?(?:[-–]([A-Za-z])|\s[-–]\s([A-Z]))(?![A-Za-z])(?!\s+[A-ZÀ-Ý])")

_PARAGRAPH_SIGN_RE = re.compile(r"§+")

# Law-number reference ("lei no 8 906 94", "lei complementar 35").
_LAW_REF_RE = re.compile(
    r"\b(lei complementar|decreto lei|lei|lc|dl)\s+(?:(?:federal|ordinaria)\s+)?"
    r"(?:no?\s+)?(\d[\d\s]*\d|\d+)")

_LAW_REF_KIND = {"lc": "lei complementar", "dl": "decreto lei"}

# Act id forms that carry a law number ("lei_8906_1994", "lc_35_1979").
_ACT_ID_NUM_RE = re.compile(r"^(lei|lc|dl|del|decreto_lei)_(\d+)_(\d{4})$")

_ACT_ID_KIND = {
    "lei": "lei", "lc": "lei complementar",
    "dl": "decreto lei", "del": "decreto lei",
    "decreto_lei": "decreto lei",
}

# Trailing year on an act name ("Constituição Federal de 1988", "CPC/1973").
_ACT_NAME_YEAR_RE = re.compile(r"(?:\s+(?:de\s+)?|\s*/\s*)\d{4}$")

_ACT_STOPWORDS = {"de", "da", "do", "das", "dos", "e"}

# Parenthetical of an act name ("Código de Processo Civil (CPC/2015)").
_ACT_PAREN_RE = re.compile(r"\(([^)]*)\)")

# An acronym inside it ("CPC/2015", "CADE", "S.A."); anything else is prose.
_ACT_ACRONYM_RE = re.compile(r"[A-ZÀ-Ý][A-ZÀ-Ý.]{1,7}(?:/\d{4})?")

# Aliases beyond ``ACT_ALIASES``; for a tuple the temporal filter keeps the act in force.
_EXTRA_ACT_ALIASES = {
    "cpc": ("lei_13105_2015", "lei_5869_1973"),
    "codigo de processo civil": ("lei_13105_2015", "lei_5869_1973"),
    "crfb": "cf_1988",
    "constituicao da republica federativa do brasil": "cf_1988",
    "estatuto da advocacia e da oab": "lei_8906_1994",
    "adct": "cf_1988:adct",
    "ato das disposicoes constitucionais transitorias": "cf_1988:adct",
    "atos das disposicoes constitucionais transitorias": "cf_1988:adct",
}

# Words after an act surface that make it another instrument ("Constituição de 1967", a state's).
_OTHER_YEAR_RE = re.compile(r"(?:de |do |da )?(1[89]\d\d|20\d\d) ")
_STATE_RE = re.compile(r"(?:estadual|do estado|dos estados|de estado) ")


def _match_form(text: str) -> str:
    """Normalize text for whole-phrase matching: punctuation to spaces, padded with spaces."""

    return f" {_NON_ALPHANUM_RE.sub(' ', normalize_text(text)).strip()} "


def _reference_form(text: str) -> str:
    """``_match_form`` for reference parsing: thousands and letter suffixes
    joined, "§" and commas kept as words."""

    text = _THOUSANDS_RE.sub(r"\1\2", text)
    text = _ART_SUFFIX_RE.sub(lambda m: m.group(1) + (m.group(2) or m.group(3)), text)
    text = _PARAGRAPH_SIGN_RE.sub(" paragrafo ", text)
    return _match_form(text.replace(",", " virgula "))


def _merge_normalized(
    *score_maps: tuple[dict[str, float], float]
) -> dict[str, float]:
    """Merge (scores, weight) maps, each normalized to its own max, keeping
    each node's best weighted score."""

    merged: dict[str, float] = {}
    for scores, weight in score_maps:
        if not scores:
            continue
        top = max(scores.values())
        for node_id, score in scores.items():
            normalized = weight * score / top
            merged[node_id] = max(merged.get(node_id, 0.0), normalized)
    return merged


class GraphRetriever:
    """Retrieve graph evidence and source law text for an OAB question."""

    def __init__(
        self,
        config: GraphRAGConfig | None = None,
        store: GraphStore | None = None,
        query_mode: str = "question_choices",
        llm_fn=None,
        lexical=None,
    ):
        """
        Args:
            store (GraphStore | None): Preloaded graph; None loads ``config.graph_dir``.
            llm_fn (callable | None): LLM backend for the "llm" linker.
            lexical: Flat article retriever whose ``rank_articles`` starts the
                PPR walk.
        """

        self.config = config or GraphRAGConfig()
        self.store = store or GraphStore.load(self.config.graph_dir)
        self.query_mode = query_mode
        self.llm_fn = llm_fn
        self.lexical = lexical
        self._walk = None
        self._assertions = None
        self.method = (
            f"graph_{self.config.linker}_k{self.config.hops}"
            f"_{self.config.text_recovery}"
            + ("_cit" if self.config.expand_citations else "")
            + ("_sib" if self.config.expand_siblings else "")
        )
        self._linker_agent = None

        self._chunks = (
            self._load_chunk_map(self.config.laws_csv)
            if self.config.laws_csv else {}
        )
        self._article_chunks = self._build_article_chunk_index(self._chunks)
        self._articles = (
            load_article_versions(self.config.articles_csv)
            if self.config.articles_csv else {}
        )
        if self.config.text_recovery == "article" and not self._articles:
            raise ValueError(
                'text_recovery="article" requires articles_csv to be set')
        if self.config.text_recovery == "chunk" and not self._chunks:
            raise ValueError(
                'text_recovery="chunk" requires laws_csv to be set')
        self._match_keys = self._build_match_keys()
        self._token_postings, self._token_idf = self._build_token_index()
        self._article_ref_index = self._build_article_ref_index()
        self._act_surfaces, self._act_numbers = self._build_act_surfaces()
        self._article_bm25 = None
        self._article_bm25_ids = None
        if self.config.linker_article_text and self._articles:
            self._build_article_text_index()
        self._entity_embeddings = None
        self._entity_embedding_ids = None
        self._embedding_encoder = None
        if self.config.entity_embeddings_dir:
            self._load_entity_embeddings(self.config.entity_embeddings_dir)
        logger.info(
            "GraphRetriever ready: {} nodes, {} edges, {} chunks, {} articles",
            len(self.store.nodes), len(self.store.edges), len(self._chunks),
            len(self._articles),
        )

    def _article_at(
        self, article_id: str, exam_year: int | None, cutoff: date | None = None
    ) -> dict | None:
        """Wording of an article (id without ``article:``) in force at the cutoff, or None."""

        return wording_at(self._articles.get(article_id) or [], exam_year, cutoff)

    def _load_entity_embeddings(self, embeddings_dir: str | Path) -> None:
        """Load the entity-name embeddings from ``build_entity_embeddings``; a
        missing cache only logs a warning."""

        embeddings_dir = Path(embeddings_dir)
        embeddings_path = embeddings_dir / "embeddings.npy"
        node_ids_path = embeddings_dir / "node_ids.json"
        if not embeddings_path.exists() or not node_ids_path.exists():
            logger.warning(
                "Entity embeddings not found at {}; run "
                "`python -m src.graphrag.build_entity_embeddings` first",
                embeddings_dir)
            return

        import numpy as np

        self._entity_embeddings = np.load(embeddings_path)
        self._entity_embedding_ids = json.loads(
            node_ids_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_chunk_map(csv_path: str | Path) -> dict[str, dict]:
        """Load chunk records keyed by doc_id for text recovery."""

        df = pd.read_csv(csv_path)
        df["ano_ultima_alteracao"] = pd.to_numeric(
            df["ano_ultima_alteracao"], errors="coerce")
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df["vintage_year"] = df["ano_ultima_alteracao"].fillna(df["ano"])

        chunks = {}
        for _, row in df.iterrows():
            raw_numbers = row.get("article_numbers")
            if raw_numbers is None or (
                    isinstance(raw_numbers, float) and pd.isna(raw_numbers)):
                numbers = []
            elif isinstance(raw_numbers, str):
                numbers = json.loads(raw_numbers or "[]")
            else:
                numbers = list(raw_numbers)
            chunks[row["doc_id"]] = {
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "lei_label": row["lei_label"],
                "area": row["area"],
                "text": str(row["lei_text_llm"] or ""),
                "article_numbers": [str(n) for n in numbers],
                "vintage_year": (
                    None if pd.isna(row["vintage_year"])
                    else int(row["vintage_year"])
                ),
            }
        return chunks

    @staticmethod
    def _build_article_chunk_index(
        chunks: dict[str, dict]
    ) -> dict[tuple[str, str], str]:
        """Index chunks by the articles they contain: (act_id, number) -> doc_id."""

        index: dict[tuple[str, str], str] = {}
        for doc_id, chunk in chunks.items():
            act_id = _DOC_CHUNK_SUFFIX_RE.sub("", str(doc_id))
            for number in chunk["article_numbers"]:
                index.setdefault((act_id, number), doc_id)
        return index

    def _doc_for_article(
        self, node_id: str, config: GraphRAGConfig
    ) -> str | None:
        """doc_id of an article node's recovered text under ``config``'s recovery
        unit; None for non-article nodes."""

        if not node_id.startswith("article:"):
            return None
        article_id = node_id[len("article:"):]
        if config.text_recovery == "article":
            return article_id
        return self._doc_for_provenance({"article_id": article_id})

    def _doc_for_provenance(self, prov: dict) -> str | None:
        """doc_id of the chunk holding a provenance entry's article, or None."""

        match = _ARTICLE_UNIT_RE.match(str(prov.get("article_id", "")))
        if not match:
            return None
        act_id = prov.get("act_id") or match.group("law_id")
        return self._article_chunks.get((act_id, match.group("num")))

    def _build_match_keys(self) -> list[tuple[str, str]]:
        """(surface, node_id) pairs from node names and aliases; short single
        words are dropped."""

        keys = []
        for node in self.store.nodes.values():
            for surface in [node.name, *node.aliases]:
                normalized = _match_form(surface).strip()
                if len(normalized) >= 6 or len(normalized.split()) >= 2:
                    keys.append((normalized, node.node_id))
        return keys

    def _build_token_index(
        self,
    ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
        """IDF-weighted token postings of entity names, and token -> IDF.

        A posting's weight is the token's IDF over sqrt(node token count).
        """

        postings: dict[str, list[tuple[str, float]]] = {}
        node_tokens: dict[str, list[str]] = {}
        for node in self.store.nodes.values():
            if node.type in ("Article", "Act", "NormativeAct"):
                continue
            tokens = {
                t for t in _match_form(node.name).split()
                if len(t) >= self.config.linker_min_token_len
            }
            if tokens:
                node_tokens[node.node_id] = sorted(tokens)

        document_frequency: dict[str, int] = {}
        for tokens in node_tokens.values():
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        total = max(len(node_tokens), 1)
        idf = {
            token: math.log(1.0 + total / count)
            for token, count in document_frequency.items()
        }
        for node_id, tokens in node_tokens.items():
            norm = math.sqrt(len(tokens))
            for token in tokens:
                postings.setdefault(token, []).append(
                    (node_id, idf[token] / norm))

        cutoff = self.config.linker_max_df_frac * total
        postings = {
            token: entries for token, entries in postings.items()
            if document_frequency[token] <= cutoff
        }
        return postings, idf

    def _build_article_text_index(self) -> None:
        """BM25 index over each graph article's newest wording."""

        from src.rag.bm25 import FastBM25

        ids, corpus = [], []
        for article_id, versions in self._articles.items():
            if f"article:{article_id}" not in self.store.nodes:
                continue
            ids.append(article_id)
            corpus.append(normalize_text(versions[-1]["display_text"]).split())
        if not ids:
            return
        try:
            self._article_bm25 = FastBM25(corpus)
        except ImportError:
            logger.warning(
                "rank_bm25 not installed; linker_article_text disabled")
            return
        self._article_bm25_ids = ids

    def _article_text_seeds(
        self, query: str, config: GraphRAGConfig
    ) -> dict[str, float]:
        """BM25 score of the ``linker_article_text_top_n`` best article nodes;
        not hub-damped."""

        if self._article_bm25 is None:
            return {}
        scores = self._article_bm25.get_scores(normalize_text(query).split())
        top_n = min(config.linker_article_text_top_n, len(scores))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
        return {
            f"article:{self._article_bm25_ids[i]}": float(scores[i])
            for i in order if scores[i] > 0
        }

    def _build_article_ref_index(self) -> dict[tuple[str, str], tuple[str, ...]]:
        """Index article nodes by (act_id, article number)."""

        index: dict[tuple[str, str], list[str]] = {}
        for node in self.store.nodes.values():
            if node.type != "Article":
                continue
            match = _ARTICLE_UNIT_RE.match(node.node_id[len("article:"):])
            if not match:
                continue
            number = str(node.attrs.get("article_number") or match.group("num"))
            index.setdefault(
                (match.group("law_id"), number), []).append(node.node_id)
        return {key: tuple(ids) for key, ids in index.items()}

    def _build_act_surfaces(
        self,
    ) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], tuple[str, ...]]]:
        """Act lookup tables for the reference linker: surface -> act_ids, and
        (kind, law number) -> act_ids.

        Surfaces are act names, their parts and acronyms, ``ACT_ALIASES``, and
        initials ("lep") when 3+ letters long and unique to one act.
        """

        surfaces: dict[str, set[str]] = {}
        derived: dict[str, set[str]] = {}
        numbers: dict[tuple[str, str], set[str]] = {}

        def add(table: dict[str, set[str]], surface: str, act_id: str) -> None:
            normalized = _match_form(surface).strip()
            if len(normalized) >= 2:
                table.setdefault(normalized, set()).add(act_id)

        act_ids = set()
        for node in self.store.nodes.values():
            if node.type not in ("NormativeAct", "Act", "Law"):
                continue
            act_id = node.node_id.split(":", 1)[-1]
            act_ids.add(act_id)
            for surface in [node.name, *node.aliases]:
                surface = surface.strip()
                for inner in _ACT_PAREN_RE.findall(surface):
                    inner = inner.strip()
                    if _ACT_ACRONYM_RE.fullmatch(inner):
                        add(surfaces, inner, act_id)
                        add(surfaces, _ACT_NAME_YEAR_RE.sub("", inner), act_id)
                plain = _ACT_PAREN_RE.sub(" ", surface).strip()
                head, *rest = plain.split(" — ")
                # After the dash only a full name, not a topic ("— ICMS").
                named = [
                    part for part in rest
                    if len([w for w in _match_form(part).split()
                            if w not in _ACT_STOPWORDS]) >= 3
                ]
                for part in {surface, plain, head, *named}:
                    part = part.strip()
                    if not part:
                        continue
                    add(surfaces, part, act_id)
                    bare = _ACT_NAME_YEAR_RE.sub("", part)
                    add(surfaces, bare, act_id)
                    initials = [
                        word[0] for word in _match_form(bare).split()
                        if word not in _ACT_STOPWORDS
                    ]
                    if len(initials) >= 3:
                        add(derived, "".join(initials), act_id)
            id_match = _ACT_ID_NUM_RE.match(act_id)
            if id_match:
                kind = _ACT_ID_KIND[id_match.group(1)]
                numbers.setdefault(
                    (kind, id_match.group(2)), set()).add(act_id)

        for surface, ids in derived.items():
            if len(ids) == 1 and surface not in surfaces:
                surfaces[surface] = set(ids)
        # A target is an act, or a container that holds articles ("cf_1988:adct").
        known = act_ids | {law_id for law_id, _ in self._article_ref_index}
        aliases: dict[str, set[str]] = {}
        for alias, act_id in {**ACT_ALIASES, **_EXTRA_ACT_ALIASES}.items():
            for target in (act_id if isinstance(act_id, tuple) else (act_id,)):
                if target in known:
                    aliases.setdefault(alias, set()).add(target)
        surfaces.update(aliases)

        return (
            {s: tuple(sorted(ids)) for s, ids in surfaces.items()},
            {k: tuple(sorted(ids)) for k, ids in numbers.items()},
        )

    def build_query(self, item: dict, query_mode: str | None = None) -> str:
        """Query text from an OAB item: the stem, plus the choices in "question_choices" mode."""

        mode = query_mode or self.query_mode
        question = item.get("question", "")
        if mode == "question":
            return question
        choices = item.get("choices", {}) or {}
        return f"{question} {' '.join(str(v) for v in choices.values())}".strip()

    def link_entities(
        self, query: str, config: GraphRAGConfig | None = None
    ) -> dict[str, float]:
        """The ``max_seeds`` best seed nodes for the question: node_id -> score."""

        config = config or self.config
        query_norm = _match_form(query)
        phrase_scores = self._phrase_scores(query, query_norm, config)
        token_scores = (
            self._token_overlap(query_norm, config)
            if config.linker_token_overlap else {}
        )
        embedding_scores = (
            self._embedding_seeds(query, config)
            if config.linker_embedding else {}
        )
        article_scores = (
            self._article_text_seeds(query, config)
            if config.linker_article_text else {}
        )

        w_phrase, w_token, w_emb = config.linker_method_weights
        scores = _merge_normalized(
            (phrase_scores, w_phrase),
            (token_scores, w_token),
            (embedding_scores, w_emb),
            (article_scores, config.linker_article_text_weight),
        )

        # Explicitly cited articles ("art. 5º da CF") get a fixed, undamped score.
        if config.linker_article_refs:
            for node_id in self._reference_seeds(query):
                scores[node_id] = max(
                    scores.get(node_id, 0.0), config.linker_reference_score)

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return dict(top[: config.max_seeds])

    def _phrase_scores(
        self, query: str, query_norm: str, config: GraphRAGConfig
    ) -> dict[str, float]:
        """Damped whole-phrase match score of entity nodes against the question."""

        surfaces = self._llm_mentions(query) if config.linker == "llm" else None
        phrase_scores: dict[str, float] = {}

        if surfaces is not None:
            for surface in surfaces:
                for node in self.store.find_by_name(surface):
                    score = self._damp(
                        1.0 + len(normalize_text(surface)) / 20.0,
                        config, node.node_id)
                    phrase_scores[node.node_id] = max(
                        phrase_scores.get(node.node_id, 0.0), score)

        for key, node_id in self._match_keys:
            if f" {key} " not in query_norm:
                continue
            score = self._damp(
                len(key.split()) + len(key) / 20.0, config, node_id)
            phrase_scores[node_id] = max(phrase_scores.get(node_id, 0.0), score)

        return phrase_scores

    def _reference_seeds(self, query: str) -> set[str]:
        """Article nodes the query cites; a reference needs an article number and
        a resolvable act mention."""

        query_norm = _reference_form(query)

        act_hits: list[tuple[int, tuple[str, ...]]] = []
        for surface, act_ids in self._act_surfaces.items():
            start = 0
            while (pos := query_norm.find(f" {surface} ", start)) >= 0:
                start = pos + 1
                if not self._other_instrument(
                        query_norm[pos + len(surface) + 2:], surface, act_ids):
                    act_hits.append((pos, act_ids))
        for match in _LAW_REF_RE.finditer(query_norm):
            digits = match.group(2).split()
            # "8 906 94": try the full number, then without the trailing year.
            candidates = ["".join(digits)]
            if len(digits) > 1:
                candidates.append("".join(digits[:-1]))
            for number in candidates:
                kind = _LAW_REF_KIND.get(match.group(1), match.group(1))
                act_ids = self._act_numbers.get((kind, number))
                if act_ids:
                    act_hits.append((match.start(), act_ids))
                    break
        if not act_hits:
            return set()

        seeds: set[str] = set()
        for match in _ART_REF_RE.finditer(query_norm):
            acts = self._acts_near(match.start(), act_hits)
            group = match.group(2)
            # A comma after a single "art." leads to its paragraph or inciso.
            if not match.group(1).endswith("s"):
                group = group.split(" virgula ", 1)[0]
            for number in self._ref_numbers(group):
                for act_id in acts:
                    seeds.update(
                        self._article_ref_index.get((act_id, number), ()))
        return seeds

    @staticmethod
    def _other_instrument(after: str, surface: str, act_ids: tuple[str, ...]) -> bool:
        """Whether the words after an act surface name another instrument
        ("Constituição de 1967", "Constituição do Estado")."""

        if _STATE_RE.match(after):
            return True
        year = _OTHER_YEAR_RE.match(after)
        if year:
            years = {a.split(":")[0].rsplit("_", 1)[-1] for a in act_ids}
            return year.group(1) not in years
        return surface == "constituicao" and after.startswith("de ")

    @staticmethod
    def _acts_near(
        pos: int,
        act_hits: list[tuple[int, tuple[str, ...]]],
        window: int = 120,
    ) -> tuple[str, ...]:
        """Act ids an article reference at ``pos`` belongs to: the nearest
        following mention, else the nearest preceding, else the only act
        within ``3 * window``."""

        following = [h for h in act_hits if 0 <= h[0] - pos <= window]
        if following:
            return min(following)[1]
        preceding = [h for h in act_hits if 0 < pos - h[0] <= window]
        if preceding:
            return max(preceding)[1]
        distinct = {h[1] for h in act_hits}
        if len(distinct) == 1 and min(abs(h[0] - pos) for h in act_hits) <= 3 * window:
            return next(iter(distinct))
        return ()

    @staticmethod
    def _ref_numbers(group: str) -> list[str]:
        """Article numbers of one reference: ordinals, suffixes ("29a" -> "29-A"),
        lists and ranges of up to 10."""

        def parse(token: str) -> tuple[str, str] | None:
            match = re.fullmatch(r"(\d+)([a-z]?)", token)
            if match is None:
                return None
            base, letter = match.groups()
            # A trailing "o" is the ordinal of "5º", not a suffix.
            return base, "" if letter in ("", "o") else letter.upper()

        tokens = group.split()
        numbers: list[str] = []
        i = 0
        while i < len(tokens):
            parsed = parse(tokens[i])
            if parsed is None:
                i += 1
                continue
            base, suffix = parsed
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            end = parse(tokens[i + 2]) if nxt == "a" and i + 2 < len(tokens) else None
            if end is not None and not suffix and not end[1]:
                span = int(end[0]) - int(base)
                if 0 < span <= 10:
                    numbers.extend(str(n) for n in range(int(base), int(end[0]) + 1))
                else:
                    numbers.extend([base, end[0]])
                i += 3
                continue
            numbers.append(f"{base}-{suffix}" if suffix else base)
            i += 1
        return numbers

    def _embedding_seeds(
        self, query: str, config: GraphRAGConfig
    ) -> dict[str, float]:
        """Damped embedding similarity of the ``linker_embedding_top_n`` nearest
        entities; empty without an embedding cache."""

        if self._entity_embeddings is None:
            return {}

        import numpy as np

        if self._embedding_encoder is None:
            from sentence_transformers import SentenceTransformer
            from src.graphrag.build_entity_embeddings import ENCODER_NAME
            self._embedding_encoder = SentenceTransformer(ENCODER_NAME)

        query_vec = self._embedding_encoder.encode(
            "query: " + normalize_text(query),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        similarities = self._entity_embeddings @ query_vec
        top_n = min(config.linker_embedding_top_n, len(similarities))
        top_idx = np.argpartition(-similarities, top_n - 1)[:top_n]
        # Cosine similarities cluster in a narrow band, so min-max them over the candidates.
        lo = float(similarities[top_idx].min())
        hi = float(similarities[top_idx].max())
        span = (hi - lo) or 1.0
        return {
            self._entity_embedding_ids[i]: self._damp(
                (float(similarities[i]) - lo) / span, config,
                self._entity_embedding_ids[i])
            for i in top_idx
        }

    def _token_overlap(
        self, query_norm: str, config: GraphRAGConfig
    ) -> dict[str, float]:
        """Damped IDF-weighted token overlap of entity nodes with the query."""

        query_tokens = {
            t for t in query_norm.split()
            if len(t) >= config.linker_min_token_len
        }
        raw: dict[str, float] = {}
        # Sorted, so float sums and ties do not depend on the hash seed.
        for token in sorted(query_tokens):
            for node_id, weight in self._token_postings.get(token, ()):
                raw[node_id] = raw.get(node_id, 0.0) + weight
        return {
            node_id: self._damp(score, config, node_id)
            for node_id, score in raw.items()
        }

    def _damp(self, score: float, config: GraphRAGConfig,
              *node_ids: str) -> float:
        """Scale a score down by the highest degree among the nodes (if ``hub_damp``)."""

        if not config.hub_damp:
            return score
        degree = max((self.store.degree(n) for n in node_ids), default=0)
        return score / (1.0 + math.log1p(degree))

    def _llm_mentions(self, query: str) -> list[str]:
        """Entity mentions in the question from the LLM linker; empty on failure."""

        from src.kg.agents import Agent, AgentOutputError
        from src.kg.config import AgentConfig, BUILDER_MODEL
        from src.kg.schemas import QuestionEntities

        if self._linker_agent is None:
            self._linker_agent = Agent(
                config=AgentConfig(
                    name="question_linker", model=BUILDER_MODEL),
                schema=QuestionEntities,
                llm_fn=self.llm_fn,
            )
        try:
            result = self._linker_agent.run(question=query)
            return [m.text for m in result.output.mentions]
        except AgentOutputError as exc:
            logger.warning("Question linker failed ({}); falling back to "
                           "lexical only", exc)
            return []

    @staticmethod
    def _passes_temporal(
        element, exam_year: int | None, cutoff: date | None = None
    ) -> bool:
        """Whether a node or edge was in force at the question's cutoff.

        Bounds compare by day when both sides have a date, else by year; the
        year is the cutoff's, since a notice date can precede the exam year.
        """

        year = cutoff.year if cutoff is not None else exam_year
        if year is None:
            return True

        attrs = getattr(element, "attrs", None) or {}
        day = cutoff.isoformat() if cutoff is not None else None

        from_date = attrs.get("valid_from_date")
        valid_from = attrs.get("valid_from")
        if day is not None and from_date is not None:
            if from_date > day:
                return False
        elif valid_from is not None and valid_from > year:
            return False

        until_date = attrs.get("valid_until_date")
        valid_until = attrs.get("valid_until")
        if day is not None and until_date is not None:
            if until_date < day:
                return False
        elif valid_until is not None and valid_until < year:
            return False
        return True

    def retrieve(
        self,
        query: str,
        exam_year: int | None = None,
        top_k: int = 5,
        overrides: dict | None = None,
        cutoff: date | None = None,
    ) -> list[dict]:
        """Retrieve graph evidence and source text for a query.

        Args:
            exam_year (int | None): Temporal filter without a cutoff.
            cutoff (date | None): Legal cutoff of the question.
            top_k (int): Source-text units recovered.
            overrides (dict | None): ``GraphRAGConfig`` fields for this call only.

        Returns:
            list[dict]: An optional ``kind="graph"`` record with the serialized
                subgraph, then ``kind="text"`` records.
        """

        config = replace(self.config, **overrides) if overrides else self.config
        if config.text_recovery == "article" and not self._articles:
            raise ValueError(
                'text_recovery="article" requires articles_csv to be set')
        if config.text_recovery == "chunk" and not self._chunks:
            raise ValueError(
                'text_recovery="chunk" requires laws_csv to be set')

        if config.ranking == "ppr":
            node_scores = self._walk_scores(query, exam_year, cutoff, config)
            if config.ppr_keep_lexical and top_k > 0:
                node_scores = self._keep_lexical(
                    node_scores, query, exam_year, cutoff, config, top_k)
        else:
            seeds = self.link_entities(query, config)
            hops, scores = self._expand_seeds(seeds, config, exam_year, cutoff)

            if config.expand_citations:
                hops, scores = self._expand_citations(
                    hops, scores, config, exam_year, cutoff)

            # Filter before any budget, so out-of-force nodes take no slot.
            node_scores = {
                node_id: score for node_id, score in scores.items()
                if self._passes_temporal(
                    self.store.nodes[node_id], exam_year, cutoff)
            }

            if config.expand_siblings:
                node_scores = self._expand_siblings(
                    node_scores, exam_year, config, cutoff)

        text_records: list[dict] = []
        if config.text_recovery != "none" and top_k > 0:
            text_records = (
                self._recover_articles(node_scores, exam_year, top_k, cutoff)
                if config.text_recovery == "article"
                else self._recover_text(node_scores, exam_year, top_k)
            )
            if config.article_content == "assertions":
                text_records = self._assertion_records(
                    text_records, exam_year, cutoff, config)
            text_records = self._apply_context_budget(text_records, config)
            if config.annotate_entities:
                text_records = self._annotate(
                    text_records, query, exam_year, cutoff, config)

        # max_nodes bounds only the serialized subgraph, not text recovery.
        serial_scores = node_scores
        if len(serial_scores) > config.max_nodes:
            kept = sorted(
                serial_scores, key=lambda n: -serial_scores[n]
            )[:config.max_nodes]
            serial_scores = {n: serial_scores[n] for n in kept}

        edges = [
            edge for edge in self.store.edges.values()
            if edge.source in serial_scores and edge.target in serial_scores
        ]

        shipped = (
            {r["doc_id"] for r in text_records}
            if config.edges_follow_text and config.text_recovery != "none"
            and top_k > 0 else None
        )
        kept_edges = self._select_edges(
            edges, serial_scores, exam_year, config, shipped, cutoff)

        records: list[dict] = []
        if config.include_graph_edges and kept_edges:
            records.append({
                "kind": "graph",
                "doc_id": "graph",
                "chunk_id": "graph",
                "lei_label": "grafo",
                "text": self._serialize_edges(kept_edges),
                "score": max(serial_scores.values(), default=0.0),
                "n_edges": len(kept_edges),
                "n_nodes": len(serial_scores),
            })
        records.extend(text_records)
        return records

    def _lexical_ranking(
        self, query: str, exam_year: int | None, cutoff: date | None,
        config: GraphRAGConfig,
    ) -> dict[str, float]:
        """The flat BM25 ranking over graph articles in force, best first."""

        if self.lexical is None:
            raise ValueError('ranking="ppr" requires a lexical retriever')
        return {
            f"article:{article_id}": score
            for article_id, score in self.lexical.rank_articles(
                query, exam_year, config.ppr_bm25_top_n, cutoff)
            if score > 0 and f"article:{article_id}" in self.store.nodes
        }

    def _keep_lexical(
        self, node_scores: dict[str, float], query: str,
        exam_year: int | None, cutoff: date | None, config: GraphRAGConfig,
        top_k: int,
    ) -> dict[str, float]:
        """Walk scores with the BM25 top ``ppr_keep_lexical`` articles, plus the
        walk's best others up to ``top_k``, lifted above every other article."""

        kept = list(self._lexical_ranking(query, exam_year, cutoff, config))[
            :config.ppr_keep_lexical]
        walk_articles = sorted(
            (n for n in node_scores if n.startswith("article:") and n not in kept),
            key=lambda n: (-node_scores[n], n))
        served = kept + walk_articles[:max(top_k - len(kept), 0)]
        lift = max(node_scores.values(), default=0.0)
        scores = dict(node_scores)
        for node_id in served:
            scores[node_id] = scores.get(node_id, 0.0) + lift
        return scores

    def _entity_seeds(
        self, query: str, exam_year: int | None, cutoff: date | None,
        config: GraphRAGConfig,
    ) -> dict[str, float]:
        """The question's best ``max_seeds`` linked nodes in force, excluding
        normative acts; the article-text channel is off."""

        linked = self.link_entities(
            query, replace(config, linker_article_text=False,
                           max_seeds=len(self.store.nodes)))
        usable = sorted(
            ((node_id, score) for node_id, score in linked.items()
             if (node := self.store.nodes.get(node_id)) is not None
             and node.type != "NormativeAct"
             and self._passes_temporal(node, exam_year, cutoff)),
            key=lambda kv: (-kv[1], kv[0]))
        return dict(usable[:config.max_seeds])

    def _annotate(
        self, records: list[dict], query: str, exam_year: int | None,
        cutoff: date | None, config: GraphRAGConfig,
    ) -> list[dict]:
        """Attach to each article record its in-force assertions about the
        question's top ``annotate_entities`` linked entities."""

        seeds = self._entity_seeds(query, exam_year, cutoff, config)
        entities = [(node_id, score) for node_id, score in seeds.items()
                    if not node_id.startswith("article:")]
        matched = dict(entities[:config.annotate_entities])
        annotated = []
        for record in records:
            if record.get("kind") != "text":
                annotated.append(record)
                continue
            best: dict[str, object] = {}
            for edge in self._article_assertions(record["doc_id"]):
                if (edge.target in matched and edge.target not in best
                        and self._passes_temporal(edge, exam_year, cutoff)):
                    best[edge.target] = edge
            ranked = sorted(best.values(),
                            key=lambda e: (-matched[e.target], e.target))
            annotated.append({**record, "annotations": [
                self._assertion_line(edge)
                for edge in ranked[:config.annotate_max_lines]]})
        return annotated

    def _assertion_line(self, edge) -> str:
        """One assertion as "Label: entity"."""

        return f"{self._assertion_label(edge)}: {self.store.nodes[edge.target].name}"

    @staticmethod
    def _assertion_label(edge) -> str:
        """The relation's label, or the modality's for a permission,
        prohibition or exception; a granted right keeps "Concede"."""

        modality = (edge.attrs or {}).get("modality")
        if modality in _MODALITY_LABELS and edge.relation != "grants":
            return _MODALITY_LABELS[modality]
        return _ASSERTION_LABELS[edge.relation]

    def _article_assertions(self, article_id: str) -> list:
        """The semantic edges an article asserts, in storage order."""

        if self._assertions is None:
            self._assertions = {}
            for edge in self.store.edges.values():
                if (edge.relation in _ASSERTION_LABELS
                        and edge.source.startswith("article:")):
                    self._assertions.setdefault(edge.source, []).append(edge)
        return self._assertions.get(f"article:{article_id}", [])

    def _assertion_records(
        self, records: list[dict], exam_year: int | None, cutoff: date | None,
        config: GraphRAGConfig,
    ) -> list[dict]:
        """Replace each article record's wording with its in-force assertions,
        one line per label.

        With ``assertions_by_provision``, assertions from a wording not in
        force at the cutoff are left out.
        """

        replaced = []
        for record in records:
            if record.get("kind") != "text":
                replaced.append(record)
                continue
            article_id = record["doc_id"]
            version = self._article_at(article_id, exam_year, cutoff) or {}
            node = self.store.nodes.get(f"article:{article_id}")
            number = node.name.split(" — ")[0] if node is not None else article_id
            path = version.get("path") if config.assertions_path else None
            lines = [f"{number} · {path}" if isinstance(path, str) and path else number]
            wording = version.get("display_text", "")[len(version.get("header", "")):]
            in_force = _WordingMatch(wording, self._out_of_force_pieces(article_id, wording))
            grouped: dict[str, list[str]] = {}
            for edge in self._article_assertions(article_id):
                if not self._passes_temporal(edge, exam_year, cutoff):
                    continue
                name = self.store.nodes[edge.target].name
                if config.assertions_by_provision and in_force.out_of_force(name):
                    continue
                names = grouped.setdefault(self._assertion_label(edge), [])
                if name not in names:
                    names.append(name)
            for label in _LABEL_ORDER:
                if grouped.get(label):
                    lines.append(f"- {label}: " + "; ".join(grouped[label]))
            replaced.append({
                **record, "text": version.get("header", "") + "\n".join(lines)})
        return replaced

    def _out_of_force_pieces(self, article_id: str, wording: str) -> list[str]:
        """Every caput and paragraph wording of an article absent from the
        wording in force (``wording``)."""

        in_force = _match_form(wording)
        pieces = []
        for version in self._articles.get(article_id, []):
            texts = [version.get("caput_line", "")] + [
                p.get("text") or "" for p in version.get("provisions") or []]
            if not version.get("provisions"):
                texts.append(version.get("display_text", "")[len(version.get("header", "")):])
            for text in texts:
                form = _match_form(text)
                if form.strip() and form not in in_force and text not in pieces:
                    pieces.append(text)
        return pieces

    def _walk_start(
        self, query: str, exam_year: int | None, cutoff: date | None,
        config: GraphRAGConfig,
    ) -> dict[str, float]:
        """Walk start distribution: the BM25 ranking and the linked seeds, mixed
        by ``ppr_seed_weight`` and summing to 1."""

        lexical = self._lexical_ranking(query, exam_year, cutoff, config)
        seeds = self._entity_seeds(query, exam_year, cutoff, config)
        # Linked article nodes are explicit references; the rest are entities.
        if config.ppr_seed_kinds != "all":
            references = config.ppr_seed_kinds == "references"
            seeds = {node_id: score for node_id, score in seeds.items()
                     if node_id.startswith("article:") == references}
        parts = [(lexical, 1.0 - config.ppr_seed_weight),
                 (seeds, config.ppr_seed_weight)]
        parts = [(scores, share) for scores, share in parts if scores and share > 0]
        start: dict[str, float] = {}
        for scores, share in parts:
            total = sum(scores.values())
            for node_id, score in scores.items():
                start[node_id] = start.get(node_id, 0.0) + share * score / total
        total = sum(start.values())
        return {n: s / total for n, s in start.items()} if total else {}

    def _walk_scores(
        self, query: str, exam_year: int | None, cutoff: date | None,
        config: GraphRAGConfig,
    ) -> dict[str, float]:
        """Personalized PageRank scores of the graph nodes in force."""

        if self._walk is None:
            from src.graphrag.graph_walk import GraphWalk

            self._walk = GraphWalk(self.store, self.config.articles_csv)
        weights = (config.ppr_weight_mentions, config.ppr_weight_semantic,
                   config.ppr_weight_citation, config.ppr_weight_adjacent,
                   config.ppr_weight_adjacent2, config.ppr_weight_section)
        scores = self._walk.rank(
            self._walk_start(query, exam_year, cutoff, config), exam_year,
            cutoff, weights, config.ppr_damping, config.ppr_iterations,
            degree_norm=config.ppr_degree_norm)
        # Section nodes exist only in the walk graph.
        return {n: s for n, s in scores.items() if n in self.store.nodes}

    def _expand_seeds(
        self, seeds: dict[str, float], config: GraphRAGConfig,
        exam_year: int | None = None, cutoff: date | None = None,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Expand each seed k hops over edges in force and sum the decayed,
        degree-damped contributions.

        Returns:
            tuple[dict[str, int], dict[str, float]]: node -> shortest hop, and
                node -> accumulated score.
        """

        hops: dict[str, int] = {}
        scores: dict[str, float] = {}

        for seed_id, seed_score in seeds.items():
            if seed_id not in self.store.nodes:
                continue
            reached = {seed_id: 0}
            frontier = [seed_id]
            for depth in range(1, config.hops + 1):
                next_frontier: list[str] = []
                for node_id in frontier:
                    expanded = 0
                    for edge in self._ranked_neighbors(node_id, exam_year, cutoff):
                        if expanded >= config.fanout_cap:
                            break
                        other = (edge.target if edge.source == node_id
                                 else edge.source)
                        if other in reached:
                            continue
                        reached[other] = depth
                        next_frontier.append(other)
                        expanded += 1
                frontier = next_frontier
                if not frontier:
                    break

            for node_id, depth in reached.items():
                contribution = seed_score * (config.hop_decay ** depth)
                if depth:
                    contribution = self._damp(contribution, config, node_id)
                scores[node_id] = scores.get(node_id, 0.0) + contribution
                if depth < hops.get(node_id, depth + 1):
                    hops[node_id] = depth

        return hops, scores

    def _ranked_neighbors(
        self, node_id: str, exam_year: int | None = None,
        cutoff: date | None = None,
    ) -> list:
        """Incident edges in force, article endpoints first, then by ascending degree."""

        def rank(edge):
            other = edge.target if edge.source == node_id else edge.source
            return (not other.startswith("article:"), self.store.degree(other))

        return sorted(self._neighbors_at(node_id, exam_year, cutoff), key=rank)

    def _neighbors_at(
        self, node_id: str, exam_year: int | None = None,
        cutoff: date | None = None,
    ) -> list:
        """Incident edges in force at the cutoff."""

        return [e for e in self.store.neighbors(node_id)
                if self._passes_temporal(e, exam_year, cutoff)]

    @staticmethod
    def _apply_context_budget(
        records: list[dict], config: GraphRAGConfig
    ) -> list[dict]:
        """Trim recovered text to ``max_context_tokens``."""

        return apply_context_budget(records, config.max_context_tokens)

    def _select_edges(
        self,
        edges: list,
        node_scores: dict[str, float],
        exam_year: int | None,
        config: GraphRAGConfig,
        shipped: set[str] | None = None,
        cutoff: date | None = None,
    ) -> list:
        """Rank and prune subgraph edges for serialization, capped by
        ``max_edges`` and ``max_edges_per_node``.

        With ``shipped`` (doc_ids served), an edge needs an endpoint among them.
        """

        denied = set(config.serialize_relation_denylist)

        def _has_shipped_endpoint(edge) -> bool:
            if shipped is None:
                return True
            return (self._doc_for_article(edge.source, config) in shipped
                    or self._doc_for_article(edge.target, config) in shipped)

        scored = [
            (self._damp(node_scores[e.source] + node_scores[e.target],
                        config, e.source, e.target), e)
            for e in edges
            if e.source in node_scores and e.target in node_scores
            and e.relation not in denied
            and self._passes_temporal(e, exam_year, cutoff)
            and _has_shipped_endpoint(e)
        ]
        scored.sort(key=lambda pair: -pair[0])

        kept: list = []
        touches: dict[str, int] = {}
        for _, edge in scored:
            if len(kept) >= config.max_edges:
                break
            if any(touches.get(n, 0) >= config.max_edges_per_node
                   for n in (edge.source, edge.target)):
                continue
            kept.append(edge)
            touches[edge.source] = touches.get(edge.source, 0) + 1
            touches[edge.target] = touches.get(edge.target, 0) + 1
        return kept

    def _expand_citations(
        self,
        hops: dict[str, int],
        scores: dict[str, float],
        config: GraphRAGConfig | None = None,
        exam_year: int | None = None,
        cutoff: date | None = None,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Add the articles the visited articles cite, scored as the citing
        article decayed by one hop."""

        config = config or self.config
        extended_hops = dict(hops)
        extended_scores = dict(scores)
        frontier = [n for n in hops if n.startswith("article:")]

        for _ in range(1, config.citation_hops + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                # citing_score is already damped; it is decayed, not damped again.
                citing_score = extended_scores.get(node_id, 0.0)
                for edge in self._neighbors_at(node_id, exam_year, cutoff):
                    if edge.relation not in _CITATION_RELATIONS:
                        continue
                    if edge.source != node_id:
                        continue
                    target = edge.target
                    target_score = citing_score * config.hop_decay
                    if target not in extended_hops:
                        extended_hops[target] = extended_hops[node_id] + 1
                        extended_scores[target] = target_score
                        next_frontier.append(target)
                    else:
                        extended_scores[target] = max(
                            extended_scores[target], target_score)
            frontier = next_frontier
            if not frontier:
                break
        return extended_hops, extended_scores

    def _expand_siblings(
        self,
        node_scores: dict[str, float],
        exam_year: int | None,
        config: GraphRAGConfig,
        cutoff: date | None = None,
    ) -> dict[str, float]:
        """Add art. N±1 of top articles sharing at least ``sibling_min_shared``
        entities with the subgraph, scored as the parent decayed by one hop."""

        parents = sorted(
            (n for n in node_scores if n.startswith("article:")),
            key=lambda n: -node_scores[n],
        )[:config.sibling_max_parents]

        added: dict[str, float] = {}
        for node_id in parents:
            match = _ARTICLE_UNIT_RE.match(node_id[len("article:"):])
            if not match or not match.group("num").isdigit():
                continue
            act_id = match.group("law_id")
            number = int(match.group("num"))
            sibling_score = node_scores[node_id] * config.hop_decay
            for adjacent in (number - 1, number + 1):
                for sibling in self._article_ref_index.get(
                        (act_id, str(adjacent)), ()):
                    if sibling in node_scores:
                        continue
                    if not self._passes_temporal(
                            self.store.nodes[sibling], exam_year, cutoff):
                        continue
                    shared = set()
                    for edge in self._neighbors_at(sibling, exam_year, cutoff):
                        other = (edge.target if edge.source == sibling
                                 else edge.source)
                        if other in node_scores and not other.startswith(
                                ("article:", "act:", "law:")):
                            shared.add(other)
                    if len(shared) >= config.sibling_min_shared:
                        added[sibling] = max(
                            added.get(sibling, 0.0), sibling_score)
        if not added:
            return node_scores
        return {**node_scores, **added}

    def _recover_articles(
        self,
        node_scores: dict[str, float],
        exam_year: int | None,
        top_k: int,
        cutoff: date | None = None,
    ) -> list[dict]:
        """Wording in force of the ``top_k`` best article nodes, best first."""

        scored = []
        for node_id, score in node_scores.items():
            if not node_id.startswith("article:"):
                continue
            article = self._article_at(node_id[len("article:"):], exam_year, cutoff)
            if not article:
                continue
            scored.append((score, article))

        scored.sort(key=lambda item: -item[0])
        return [
            {
                "kind": "text",
                "doc_id": article["article_id"],
                "chunk_id": article["article_id"],
                "lei_label": article["lei_label"],
                "area": article.get("area"),
                "text": article["display_text"],
                "score": score,
            }
            for score, article in scored[:top_k]
        ]

    def _recover_text(
        self,
        node_scores: dict[str, float],
        exam_year: int | None,
        top_k: int,
    ) -> list[dict]:
        """The ``top_k`` chunks by summed score of the nodes whose provenance
        points at them, best first."""

        doc_scores: dict[str, float] = {}
        for node_id, score in node_scores.items():
            for prov in self.store.nodes[node_id].provenance:
                doc_id = self._doc_for_provenance(prov)
                if doc_id is not None:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])
        records = []
        for doc_id, score in ranked:
            chunk = self._chunks[doc_id]
            if exam_year is not None:
                vintage = chunk["vintage_year"]
                if vintage is None or vintage > exam_year:
                    continue
            records.append({
                "kind": "text",
                **{k: chunk[k] for k in
                   ("doc_id", "chunk_id", "lei_label", "area", "text")},
                "score": score,
            })
            if len(records) >= top_k:
                break
        return records

    def _edge_source_label(self, edge) -> str:
        """Source citation of an edge, like "Art. 5 — Código Civil"; empty when unknown."""

        for prov in edge.provenance:
            article = self.store.nodes.get(
                f"article:{prov.get('article_id', '')}")
            if article is not None:
                return article.name
            act = self.store.nodes.get(f"act:{prov.get('act_id', '')}")
            if act is not None:
                return act.name
        return ""

    def _serialize_edges(self, edges: list) -> str:
        """One line per edge: ``[A] — relation → [B] (fonte: Art. Y — Lei X)``."""

        lines = []
        for edge in edges:
            source = self.store.nodes[edge.source].name
            target = self.store.nodes[edge.target].name
            line = f"[{source}] — {edge.relation} → [{target}]"
            citation = self._edge_source_label(edge)
            if citation:
                line += f" (fonte: {citation})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def format_context(records: list[dict]) -> str:
        """Graph-relations block followed by numbered law-text blocks."""

        blocks = []
        text_index = 0
        for record in records:
            if record.get("kind") == "graph":
                blocks.append(
                    "Relações extraídas da legislação:\n" + record["text"])
            else:
                text_index += 1
                block = f"[{text_index}] {record['text'].strip()}"
                if record.get("annotations"):
                    block += "\nPontos relacionados à questão:\n" + "\n".join(
                        f"- {line}" for line in record["annotations"])
                blocks.append(block)
        return "\n\n".join(blocks)
