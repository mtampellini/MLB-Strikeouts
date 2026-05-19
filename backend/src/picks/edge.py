"""Phase 5: edge_pct and EV% computation."""
from __future__ import annotations

from .devig import american_to_decimal


def edge_pct(model_p: float, market_p: float) -> float:
    """``(model_p - market_p) / market_p``.

    Positive: model thinks the bet is more likely than the book does.
    Negative: book thinks the bet is more likely.

    Undefined when ``market_p <= 0`` — return 0 in that degenerate case
    rather than divide-by-zero.
    """
    if market_p <= 0:
        return 0.0
    return (model_p - market_p) / market_p


def ev_pct(model_p: float, american_odds: int) -> float:
    """Expected value as a percentage of stake.

    EV per $1 stake = model_p * (decimal - 1) - (1 - model_p)
    EV% = EV * 100
    """
    decimal = american_to_decimal(american_odds)
    ev = model_p * (decimal - 1.0) - (1.0 - model_p)
    return ev * 100.0
