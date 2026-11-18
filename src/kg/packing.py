"""Batching of articles into knowledge-extraction calls.

Articles sharing a structural parent are packed up to a token and count budget;
long articles go alone; boilerplate provisions are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractionBatch:
    """A group of articles sent to the KE agents as one call.

    Attributes:
        batch_id (str): Stable id, derived from the first article.
        path (str): Structural parent shared by the articles.
    """

    batch_id: str
    act_id: str
    path: str
    articles: list[dict] = field(default_factory=list)

    @property
    def article_ids(self) -> list[str]:
        """Ids of the articles in this batch."""
        return [a["article_id"] for a in self.articles]

    @property
    def article_numbers(self) -> list[str]:
        """Article numbers the extractor may attribute a triple to."""
        return [str(a["article_number"]) for a in self.articles]

    @property
    def tokens(self) -> int:
        """Total token count of the batch's article texts."""
        return sum(int(a.get("tokens") or 0) for a in self.articles)

    def to_prompt_text(self) -> str:
        """Render the batch as the ``${article_text}`` block, labelled by article number."""

        return "\n\n".join(
            f"[Art. {a['article_number']}]\n{a['article_text']}"
            for a in self.articles
        )


def _path_of(article: dict) -> str:
    """Structural path of an article, with missing values normalized."""

    path = article.get("path")
    # A missing path arrives as NaN from pandas, which is truthy.
    if path is None or path != path:
        return ""
    return str(path)


def batch_id_for(article: dict) -> str:
    """Stable batch id: ``<article_id>@<valid_from>``, unique across wordings."""

    article_id = article["article_id"]
    valid_from = article.get("valid_from")
    if valid_from is None or valid_from != valid_from:
        return str(article_id)
    return f"{article_id}@{int(valid_from)}"


def build_batches(
    articles: list[dict],
    token_budget: int = 800,
    max_articles: int = 5,
    solo_tokens: int = 500,
    skip_boilerplate: bool = True,
) -> list[ExtractionBatch]:
    """Group articles into extraction batches, in corpus order per act and path.

    Args:
        token_budget (int): Soft cap on article tokens per batch.
        max_articles (int): Cap on articles per batch.
        solo_tokens (int): Articles above this size are sent alone.
        skip_boilerplate (bool): Drop contentless provisions.
    """

    batches: list[ExtractionBatch] = []
    current: ExtractionBatch | None = None

    def _flush() -> None:
        nonlocal current
        if current and current.articles:
            batches.append(current)
        current = None

    for article in articles:
        if skip_boilerplate and article.get("is_boilerplate"):
            continue

        tokens = int(article.get("tokens") or 0)
        act_id = article["act_id"]
        path = _path_of(article)

        if tokens > solo_tokens:
            _flush()
            batches.append(ExtractionBatch(
                batch_id=batch_id_for(article),
                act_id=act_id, path=path, articles=[article],
            ))
            continue

        # Two wordings of one article never share a batch (same article number).
        if current is not None and (
            current.act_id != act_id
            or current.path != path
            or len(current.articles) >= max_articles
            or current.tokens + tokens > token_budget
            or article["article_id"] in {a["article_id"]
                                         for a in current.articles}
        ):
            _flush()

        if current is None:
            current = ExtractionBatch(
                batch_id=batch_id_for(article), act_id=act_id, path=path)
        current.articles.append(article)

    _flush()
    return batches
