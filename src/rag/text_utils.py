"""Text normalization, tokenization and the context budget shared by the retrievers.

Queries and documents are both lowercased and accent-stripped before scoring.
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace."""

    if not text:
        return ""

    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )

    return re.sub(r"\s+", " ", without_accents.lower()).strip()


_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z]+")


def tokenize(text: str) -> list[str]:
    """BM25 tokens: normalized, split on non-alphanumeric runs, length >= 2."""

    normalized = normalize_text(text)
    return [tok for tok in _TOKEN_SPLIT_RE.split(normalized) if len(tok) >= 2]


def apply_context_budget(records: list[dict], max_tokens: int | None) -> list[dict]:
    """Trim best-first records to a token budget (4 characters a token).

    An oversized first record is truncated; a later record that does not fit
    is skipped. None or 0 keeps every record.
    """

    if not max_tokens or not records:
        return records
    budget = max_tokens * 4
    kept, used = [], 0
    for record in records:
        size = len(record["text"])
        if not kept and size > budget:
            record = {**record, "text": record["text"][:budget] + " […]"}
            size = len(record["text"])
        elif kept and used + size > budget:
            continue
        kept.append(record)
        used += size
    return kept
