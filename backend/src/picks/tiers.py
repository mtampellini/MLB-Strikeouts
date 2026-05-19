"""Phase 5: three-tier pick classification + ranking.

Tier definitions:

- **Primary**: ``edge_pct >= 0.20`` AND ``price >= PRIMARY_PRICE_CAP``
  (i.e., American odds less negative than -180). Ranked by edge_pct,
  top 10 only.
- **Secondary**: ``edge_pct >= 0.20`` but either above the price cap
  (heavy chalk) or rank > 10 within the primary candidates.
- **Shadow**: ``0.10 <= edge_pct < 0.20``.
- **None**: ``edge_pct < 0.10`` — not a pick.

Primary ranking is by ``edge_pct``, not EV%. EV% rewards long shots
(high decimal odds amplify EV at fixed model_p) — the HR-Picks-V7 model
learned that EV-ranking biases the slate toward speculative long-tail
plays. Edge_pct is the better discriminator for confidence in the model's
disagreement with the book.
"""
from __future__ import annotations

PRIMARY_EDGE_THRESHOLD = 0.20
SHADOW_EDGE_THRESHOLD = 0.10
# Tighter than HR model's -220 because K markets carry higher vig and
# small-mistake risk on heavy chalk is amplified at the higher implied
# probabilities.
PRIMARY_PRICE_CAP = -180
PRIMARY_RANK_LIMIT = 10


def classify_pick(edge: float, american_odds: int, rank_in_primary: int) -> str:
    """Return one of: ``"primary"`` | ``"secondary"`` | ``"shadow"`` | ``"none"``.

    Args:
        edge: ``edge_pct`` (e.g., 0.25 for 25% edge).
        american_odds: posted American odds at the book (e.g., -110, +150).
        rank_in_primary: 1-indexed rank within the slate's primary
            candidates ordered by edge_pct descending.
    """
    if edge < SHADOW_EDGE_THRESHOLD:
        return "none"
    if edge < PRIMARY_EDGE_THRESHOLD:
        return "shadow"
    # edge >= PRIMARY_EDGE_THRESHOLD
    if american_odds < PRIMARY_PRICE_CAP:
        # Heavier chalk than the cap allows -> secondary
        return "secondary"
    if rank_in_primary > PRIMARY_RANK_LIMIT:
        return "secondary"
    return "primary"


def rank_primary_picks(candidate_picks: list[dict]) -> list[dict]:
    """Rank candidate picks (those with ``edge_pct >= PRIMARY_EDGE_THRESHOLD``)
    by edge_pct descending. Returns a NEW list with ``rank_in_tier`` set
    on each entry (1-indexed).
    """
    sorted_picks = sorted(
        candidate_picks, key=lambda p: p["edge_pct"], reverse=True,
    )
    out: list[dict] = []
    for i, pick in enumerate(sorted_picks, start=1):
        out.append({**pick, "rank_in_tier": i})
    return out
