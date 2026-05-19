"""Phase 5: tests for tiers.py."""
from __future__ import annotations

import pytest

from src.picks.tiers import (
    PRIMARY_EDGE_THRESHOLD,
    PRIMARY_PRICE_CAP,
    PRIMARY_RANK_LIMIT,
    SHADOW_EDGE_THRESHOLD,
    classify_pick,
    rank_primary_picks,
)


# ---- classify_pick --------------------------------------------------------


def test_classify_primary_clean():
    """edge=0.25, price=-110, rank=3 -> primary."""
    assert classify_pick(0.25, -110, 3) == "primary"


def test_classify_primary_at_threshold():
    """edge=0.20 exactly -> primary (assuming price + rank OK)."""
    assert classify_pick(0.20, -110, 1) == "primary"


def test_classify_secondary_when_price_too_chalky():
    """edge=0.25 but price=-200 (below -180 cap) -> secondary."""
    assert classify_pick(0.25, -200, 1) == "secondary"


def test_classify_secondary_when_rank_above_limit():
    """edge=0.25, price=-110, rank=12 -> secondary (rank > 10)."""
    assert classify_pick(0.25, -110, 12) == "secondary"


def test_classify_shadow():
    """0.10 <= edge < 0.20 -> shadow."""
    assert classify_pick(0.15, -110, 1) == "shadow"
    assert classify_pick(0.10, -110, 1) == "shadow"
    # Shadow tier doesn't care about price cap or rank
    assert classify_pick(0.15, -300, 12) == "shadow"


def test_classify_none_when_below_shadow():
    assert classify_pick(0.08, -110, 1) == "none"
    assert classify_pick(0.0, -110, 1) == "none"
    assert classify_pick(-0.10, -110, 1) == "none"


def test_classify_price_cap_boundary():
    """price=-180 is the cap. Less negative (e.g. -179) -> primary; equal
    or more negative -> secondary."""
    assert classify_pick(0.25, -179, 1) == "primary"
    assert classify_pick(0.25, PRIMARY_PRICE_CAP, 1) == "primary"  # equal is fine
    assert classify_pick(0.25, -181, 1) == "secondary"


def test_classify_constants_match_spec():
    assert PRIMARY_EDGE_THRESHOLD == 0.20
    assert SHADOW_EDGE_THRESHOLD == 0.10
    assert PRIMARY_PRICE_CAP == -180
    assert PRIMARY_RANK_LIMIT == 10


# ---- rank_primary_picks ---------------------------------------------------


def test_rank_primary_picks_descending_by_edge():
    picks = [
        {"pick_id": "a", "edge_pct": 0.22},
        {"pick_id": "b", "edge_pct": 0.30},
        {"pick_id": "c", "edge_pct": 0.25},
    ]
    ranked = rank_primary_picks(picks)
    assert ranked[0]["pick_id"] == "b"
    assert ranked[0]["rank_in_tier"] == 1
    assert ranked[1]["pick_id"] == "c"
    assert ranked[1]["rank_in_tier"] == 2
    assert ranked[2]["pick_id"] == "a"
    assert ranked[2]["rank_in_tier"] == 3


def test_rank_primary_picks_edge_not_ev():
    """The HR-Picks lesson: rank by edge_pct, NOT EV%.

    Construct two picks where EV% ordering differs from edge ordering;
    confirm we use edge."""
    picks = [
        # high-EV long shot: model 0.20, market 0.167, +500 odds → EV ≈ 20%
        {"pick_id": "long_shot", "edge_pct": 0.197, "ev_pct": 20.0},
        # high-edge chalk: model 0.85, market 0.60, -150 odds → edge = 0.42
        {"pick_id": "high_edge_chalk", "edge_pct": 0.417, "ev_pct": 41.0},
    ]
    ranked = rank_primary_picks(picks)
    assert ranked[0]["pick_id"] == "high_edge_chalk"  # higher edge wins


def test_rank_primary_picks_does_not_mutate_input():
    picks = [{"pick_id": "a", "edge_pct": 0.30}]
    ranked = rank_primary_picks(picks)
    assert "rank_in_tier" not in picks[0]  # original untouched
    assert ranked[0]["rank_in_tier"] == 1


def test_rank_primary_picks_empty():
    assert rank_primary_picks([]) == []
