"""Cost currencies: runs store cost in the provider's billing currency;
reports convert it to USD and display ``$X (R$Y)``.
"""

BRL_PER_USD = 5.17  # 2026-09-23 23:21 UTC

# Provider billing currency, for entries without ``cost_currency``.
PROVIDER_CURRENCY = {"maritaca": "BRL"}


def cost_currency(record: dict) -> str:
    """Currency ("USD" or "BRL") of a result row or metrics entry's ``estimated_cost``."""

    return (record.get("cost_currency")
            or PROVIDER_CURRENCY.get(record.get("provider"), "USD"))


def to_usd(amount: float | None, currency: str = "USD") -> float | None:
    """Convert a cost in "USD" or "BRL" to USD; None stays None."""

    if amount is None:
        return None
    if currency == "BRL":
        return amount / BRL_PER_USD
    if currency == "USD":
        return amount
    raise ValueError(f"Unknown currency: {currency!r}")


def record_cost_usd(record: dict) -> float | None:
    """A record's ``estimated_cost`` in USD."""

    return to_usd(record.get("estimated_cost"), cost_currency(record))


def format_cost(usd: float | None, digits: int = 4) -> str:
    """Display a USD cost with its BRL equivalent, e.g. "$0.0595 (R$0.3076)"; None is "-"."""

    if usd is None:
        return "-"
    return f"${usd:.{digits}f} (R${usd * BRL_PER_USD:.{digits}f})"
