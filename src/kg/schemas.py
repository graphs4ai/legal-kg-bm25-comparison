"""Pydantic schemas for the structured outputs of every KG-construction agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Ontology

class EntityType(BaseModel):
    """One entity type of the ontology (e.g. Law, Article, Concept)."""

    name: str
    description: str = ""
    attributes: list[str] = Field(default_factory=list)


class RelationType(BaseModel):
    """One relation type of the ontology (e.g. regulates, amends).

    ``domain``/``range`` list the entity-type names allowed as subject/object;
    empty lists mean unrestricted.
    """

    name: str
    description: str = ""
    domain: list[str] = Field(default_factory=list)
    range: list[str] = Field(default_factory=list)


class Ontology(BaseModel):
    """A versioned ontology schema: the ground truth for extraction."""

    version: str = "v0"
    entity_types: list[EntityType] = Field(default_factory=list)
    relation_types: list[RelationType] = Field(default_factory=list)
    notes: str = ""

    def entity_type_names(self) -> set[str]:
        """Return the set of entity-type names."""
        return {t.name for t in self.entity_types}

    def relation_type_names(self) -> set[str]:
        """Return the set of relation-type names."""
        return {r.name for r in self.relation_types}

    def merged_with(self, other: "Ontology", version: str) -> "Ontology":
        """Union with another ontology; on a name collision ``self`` wins."""

        entities = list(self.entity_types) + [
            t for t in other.entity_types
            if t.name not in self.entity_type_names()
        ]
        relations = list(self.relation_types) + [
            r for r in other.relation_types
            if r.name not in self.relation_type_names()
        ]
        return Ontology(
            version=version,
            entity_types=entities,
            relation_types=relations,
            notes=self.notes,
        )


# Ontology Engineering (OE)

class OntologyProposal(BaseModel):
    """OE Builder output: a proposed semantic overlay plus a short rationale."""

    entity_types: list[EntityType]
    relation_types: list[RelationType]
    rationale: str = ""

    def to_ontology(self, version: str) -> Ontology:
        """Convert the proposal into a versioned ``Ontology``."""
        return Ontology(
            version=version,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            notes=self.rationale,
        )


class OntologyMergeReview(BaseModel):
    """OE Merge Critic output: one consolidated overlay across all shards.

    ``merged_from`` maps each surviving type to the proposed names it absorbs.
    """

    entity_types: list[EntityType]
    relation_types: list[RelationType]
    merged_from: dict[str, list[str]] = Field(default_factory=dict)
    dropped: list[str] = Field(default_factory=list)
    rationale: str = ""

    def to_ontology(self, version: str) -> Ontology:
        """Convert the consolidated overlay into a versioned ``Ontology``."""
        return Ontology(
            version=version,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            notes=self.rationale,
        )


class CriticIssue(BaseModel):
    """A single problem flagged by a critic agent (unlabelled = advisory)."""

    category: str
    description: str
    suggestion: str = ""
    severity: Literal["blocking", "advisory"] = "advisory"


class OntologyReview(BaseModel):
    """OE Critic output: approval plus issues; the orchestrator branches on severities."""

    approval: bool
    issues: list[CriticIssue] = Field(default_factory=list)
    summary: str = ""

    def blocking_issues(self) -> list[CriticIssue]:
        """Return the issues that make the proposal unusable as it stands."""
        return [i for i in self.issues if i.severity == "blocking"]


# Knowledge Extraction (KE)

class EntityMention(BaseModel):
    """An entity mention: a verbatim span of the article and an ontology type."""

    text: str
    type: str


# Relations the backbone builds from the corpus; extraction must not emit them.
STRUCTURAL_RELATIONS = frozenset({
    "contains", "refers_to", "amends", "revokes", "mentions",
})

# Deontic force of an assertion, plus "exception" for a restricting condition.
MODALITIES = frozenset({
    "obligation", "permission", "prohibition", "exception",
})

DEONTIC_MODALITIES = frozenset({"obligation", "permission", "prohibition"})


class Triple(BaseModel):
    """One assertion an article makes, as (article, relation, object).

    The subject is the article identified by ``article_number``; ``modality``
    is one of ``MODALITIES`` or None.
    """

    relation: str
    object: EntityMention
    subject: EntityMention | None = None
    article_number: str | None = None
    modality: str | None = None
    confidence: float | None = None


class ArticleExtraction(BaseModel):
    """The triples the builder attributes to one article of a batch."""

    article_number: str
    triples: list[Triple] = Field(default_factory=list)


class ExtractionProposal(BaseModel):
    """KE Builder output: per-article ``articles`` or a flat ``triples`` list."""

    articles: list[ArticleExtraction] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    notes: str = ""

    def all_triples(self) -> list[Triple]:
        """Return every triple from both forms, each tagged with its article number."""

        collected = list(self.triples)
        for group in self.articles:
            for triple in group.triples:
                if triple.article_number is None:
                    triple = triple.model_copy(
                        update={"article_number": group.article_number})
                collected.append(triple)
        return collected


class TripleViolation(BaseModel):
    """A problem in one extracted triple, from the checks or the KE Critic.

    ``triple_index`` indexes the reviewed triple list; only ``blocking``
    violations trigger a re-extraction.
    """

    triple_index: int
    code: str
    description: str = ""
    severity: Literal["blocking", "advisory"] = "advisory"


# Codes the KE Critic prompt defines; any other code becomes "low_quality".
KE_CRITIC_CODES = frozenset({
    "unsupported", "inverted", "wrong_modality", "low_quality", "duplicate",
})


class ExtractionReview(BaseModel):
    """KE Critic output: violations found in an extraction proposal."""

    violations: list[TripleViolation] = Field(default_factory=list)
    feedback: str = ""

    def normalized_violations(
        self,
    ) -> tuple[list[TripleViolation], list[str]]:
        """Return violations with unknown codes remapped, and the remapped codes."""

        normalized, remapped = [], []
        for violation in self.violations:
            if violation.code in KE_CRITIC_CODES:
                normalized.append(violation)
            else:
                remapped.append(violation.code)
                normalized.append(
                    violation.model_copy(update={"code": "low_quality"}))
        return normalized, remapped

    @property
    def violation_count(self) -> int:
        """Number of flagged violations."""
        return len(self.violations)


# Knowledge Fusion (KF)

class MergeDecision(BaseModel):
    """KF Builder output: whether two entity mentions co-refer."""

    same_entity: bool
    canonical_name: str = ""
    rationale: str = ""


class ClusterMergeDecision(BaseModel):
    """KF Builder output: groups of co-referring mention labels, one name per group."""

    groups: list[list[str]] = Field(default_factory=list)
    canonical_names: list[str] = Field(default_factory=list)
    rationale: str = ""


class MergeReview(BaseModel):
    """KF Critic output: verdict on a builder merge decision."""

    verdict: Literal["accept", "reject"]
    rationale: str = ""


# GraphRAG question linking

class QuestionEntities(BaseModel):
    """LLM question-linker output: entity mentions found in an OAB question."""

    mentions: list[EntityMention] = Field(default_factory=list)
