"""Mock LLM for offline dry runs: injected as ``Agent.llm_fn``, no API call."""

from __future__ import annotations

import json
import re
import threading

from src.kg.config import AgentConfig


class MockLLM:
    """Returns scripted responses per agent name, then ``default_fn`` output."""

    def __init__(
        self,
        responses: dict[str, list[str]] | None = None,
        default_fn=None,
    ):
        """
        Args:
            responses (dict[str, list[str]] | None): Agent name -> FIFO of texts.
            default_fn (callable | None): ``(prompt, config) -> str`` fallback.
        """

        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.default_fn = default_fn or generate_default_response
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str, config: AgentConfig) -> dict:
        """Return the next scripted (or generated) response, provider-shaped."""

        with self._lock:
            self.calls.append({"agent": config.name, "prompt": prompt})
            queue = self.responses.get(config.name)
            text = queue.pop(0) if queue else None
        if text is None:
            text = self.default_fn(prompt, config)
        return {
            "provider": "mock",
            "text": text,
            "input_tokens": len(prompt.split()),
            "output_tokens": len(text.split()),
            "elapsed_seconds": 0.0,
            "estimated_cost": 0.0,
        }


def generate_default_response(prompt: str, config: AgentConfig) -> str:
    """Generate a minimal schema-valid response; critics approve everything."""

    name = config.name

    if name == "oe_builder":
        return json.dumps({
            "entity_types": [
                {"name": "Concept", "description": "Conceito jurídico"},
                {"name": "LegalSubject", "description": "Sujeito de direito"},
            ],
            "relation_types": [
                {"name": "regulates", "description": "Artigo disciplina"},
                {"name": "defines", "description": "Artigo define conceito"},
            ],
            "rationale": "Proposta mínima (mock).",
        }, ensure_ascii=False)

    if name == "oe_critic":
        return json.dumps({
            "approval": True, "issues": [], "summary": "Aprovado (mock).",
        }, ensure_ascii=False)

    if name == "oe_merge_critic":
        return json.dumps({
            "entity_types": [
                {"name": "Concept", "description": "Conceito jurídico"},
            ],
            "relation_types": [
                {"name": "regulates", "description": "Disciplina"},
            ],
            "merged_from": {},
            "dropped": [],
            "rationale": "Consolidação mínima (mock).",
        }, ensure_ascii=False)

    if name == "ke_builder":
        return json.dumps({
            "articles": [
                {
                    "article_number": number,
                    "triples": [
                        {
                            "relation": "regulates",
                            "object": {"text": fragment, "type": "Concept"},
                            "confidence": 0.9,
                        },
                    ],
                }
                for number, fragment in _batch_fragments(prompt)
            ],
            "notes": "Extração mínima (mock).",
        }, ensure_ascii=False)

    if name == "ke_critic":
        return json.dumps({
            "violations": [], "feedback": "Sem violações (mock).",
        }, ensure_ascii=False)

    if name == "kf_builder":
        # No grouping: every mention stays its own entity.
        return json.dumps({
            "groups": [], "canonical_names": [],
            "rationale": "Menções distintas (mock).",
        }, ensure_ascii=False)

    if name == "kf_critic":
        return json.dumps({
            "verdict": "accept",
            "rationale": "Decisão coerente (mock).",
        }, ensure_ascii=False)

    if name == "question_linker":
        return json.dumps({"mentions": []}, ensure_ascii=False)

    return "{}"


def _batch_fragments(prompt: str) -> list[tuple[str, str]]:
    """Pull (article number, verbatim fragment) pairs from a KE prompt."""

    blocks = re.findall(
        r"\[Art\.\s*([^\]]+)\]\n(.+?)(?=\n\n\[Art\.|\n</artigos>|$)",
        prompt, re.DOTALL,
    )
    pairs = []
    for number, body in blocks:
        words = body.split()
        fragment = " ".join(words[:4]) if words else number.strip()
        pairs.append((number.strip(), fragment))
    return pairs or [("1", "Art.")]
