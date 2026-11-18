"""Deterministic checks of extraction output against the ontology and source text."""

from __future__ import annotations

import re
import unicodedata

from src.kg.schemas import (
    MODALITIES,
    STRUCTURAL_RELATIONS,
    ExtractionProposal,
    Ontology,
    TripleViolation,
)

CODE_UNKNOWN_ENTITY_TYPE = "unknown_entity_type"
CODE_UNKNOWN_RELATION = "unknown_relation"
CODE_UNKNOWN_ARTICLE = "unknown_article"
CODE_NON_VERBATIM_SPAN = "non_verbatim_span"
CODE_EMPTY_SPAN = "empty_span"
CODE_STRUCTURAL_RELATION = "structural_relation"
CODE_NON_ARTICLE_SUBJECT = "non_article_subject"
CODE_UNKNOWN_MODALITY = "unknown_modality"
CODE_NON_ATOMIC_SPAN = "non_atomic_span"
CODE_REPEATED_TRIPLE = "repeated_triple"

# Advisory: an object longer than this is likely a sentence, not one term.
NON_ATOMIC_WORDS = 12


def _normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""

    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def validate_proposal(
    proposal: ExtractionProposal,
    ontology: Ontology,
    source_text: str,
    article_numbers: list[str] | None = None,
) -> list[TripleViolation]:
    """Check a proposal against the ontology, the batch, and the source text.

    Args:
        article_numbers (list[str] | None): Article numbers in the batch; a
            triple attributed elsewhere is flagged.

    Returns:
        list[TripleViolation]: Violations indexed into ``proposal.all_triples()``.
    """

    entity_types = ontology.entity_type_names()
    relation_types = ontology.relation_type_names()
    known_articles = {str(n) for n in (article_numbers or [])}
    haystack = _normalize(source_text)

    violations: list[TripleViolation] = []
    seen: set[tuple] = set()
    for index, triple in enumerate(proposal.all_triples()):
        mention = triple.object
        span = _normalize(mention.text)

        # Same article, relation and object repeated: drop the copy.
        signature = (
            str(triple.article_number or ""), triple.relation,
            mention.type, span,
        )
        if signature in seen:
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_REPEATED_TRIPLE,
                description=f"repetição de '{mention.text}'"))
        seen.add(signature)
        if not span:
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_EMPTY_SPAN,
                description="objeto sem texto"))
        elif span not in haystack:
            # Advisory: the triple is kept and must not force a retry.
            violations.append(TripleViolation(
                triple_index=index, code=CODE_NON_VERBATIM_SPAN,
                description=f"objeto '{mention.text}' não ocorre no texto"))

        if len(span.split()) > NON_ATOMIC_WORDS:
            violations.append(TripleViolation(
                triple_index=index, code=CODE_NON_ATOMIC_SPAN,
                description=(
                    f"objeto com {len(span.split())} palavras; provável "
                    "frase ou enumeração em vez de um termo atômico")))

        if triple.modality is not None and triple.modality not in MODALITIES:
            violations.append(TripleViolation(
                triple_index=index, code=CODE_UNKNOWN_MODALITY,
                description=f"modalidade '{triple.modality}'"))

        if entity_types and mention.type not in entity_types:
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_UNKNOWN_ENTITY_TYPE,
                description=f"objeto com tipo '{mention.type}'"))

        # The subject must be the article itself.
        if triple.subject is not None and triple.subject.type != "Article":
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_NON_ARTICLE_SUBJECT,
                description=(
                    f"sujeito '{triple.subject.text}' "
                    f"(tipo '{triple.subject.type}') não é o artigo"),
            ))

        if triple.relation in STRUCTURAL_RELATIONS:
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_STRUCTURAL_RELATION,
                description=(
                    f"relação '{triple.relation}' é estrutural e já é "
                    "construída deterministicamente"),
            ))
        elif relation_types and triple.relation not in relation_types:
            violations.append(TripleViolation(
                severity="blocking",
                triple_index=index, code=CODE_UNKNOWN_RELATION,
                description=f"relação '{triple.relation}'"))

        if known_articles:
            number = str(triple.article_number or "")
            if number not in known_articles:
                violations.append(TripleViolation(
                    severity="blocking",
                    triple_index=index, code=CODE_UNKNOWN_ARTICLE,
                    description=(
                        f"artigo '{number}' não pertence ao lote "
                        f"({', '.join(sorted(known_articles))})"),
                ))

    return violations


# Codes that flag a triple without dropping it.
KEEP_CODES = frozenset({
    CODE_NON_VERBATIM_SPAN,
    CODE_NON_ATOMIC_SPAN,
    CODE_UNKNOWN_MODALITY,
})

# ``triple_index`` of a violation whose triple was dropped before review.
INDEX_DROPPED = -1


def surviving_indices(
    proposal: ExtractionProposal,
    violations: list[TripleViolation],
    keep_codes: set[str] | None = None,
) -> list[int]:
    """Return the ascending proposal indices of triples that survive the checks."""

    if keep_codes is None:
        keep_codes = set(KEEP_CODES)
    blocked = {
        v.triple_index for v in violations if v.code not in keep_codes
    }
    return [
        index for index in range(len(proposal.all_triples()))
        if index not in blocked
    ]


def drop_invalid_triples(
    proposal: ExtractionProposal,
    violations: list[TripleViolation],
    keep_codes: set[str] | None = None,
) -> list[dict]:
    """Return the serialized triples that survive the deterministic checks."""

    survivors = set(surviving_indices(proposal, violations, keep_codes))
    return [
        triple.model_dump(exclude_none=True)
        for index, triple in enumerate(proposal.all_triples())
        if index in survivors
    ]


def rebase_violations(
    violations: list[TripleViolation],
    survivors: list[int],
) -> list[TripleViolation]:
    """Reindex check violations onto the surviving triples' index space.

    A violation whose triple was dropped gets ``INDEX_DROPPED``.
    """

    rebased = {old: new for new, old in enumerate(survivors)}
    return [
        v.model_copy(update={
            "triple_index": rebased.get(v.triple_index, INDEX_DROPPED)})
        for v in violations
    ]
