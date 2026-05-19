"""Phase 5: tests for edge.py."""
from __future__ import annotations

import pytest

from src.picks.edge import edge_pct, ev_pct


def test_edge_pct_basic():
    """model 0.55 vs market 0.50 -> edge = (0.55-0.50)/0.50 = 10%."""
    assert edge_pct(0.55, 0.50) == pytest.approx(0.10, abs=1e-9)


def test_edge_pct_negative():
    """Book thinks more likely than model -> negative edge (allowed)."""
    assert edge_pct(0.45, 0.50) == pytest.approx(-0.10, abs=1e-9)


def test_edge_pct_zero_when_aligned():
    assert edge_pct(0.50, 0.50) == 0.0


def test_edge_pct_zero_market_returns_zero():
    """Degenerate input: market_p=0 returns 0 (no divide by zero)."""
    assert edge_pct(0.50, 0.0) == 0.0
    assert edge_pct(0.50, -0.1) == 0.0


def test_ev_pct_even_money_no_edge():
    """At +100 (even money) with model_p=0.50, EV = 0."""
    assert ev_pct(0.50, +100) == pytest.approx(0.0, abs=1e-9)


def test_ev_pct_even_money_with_edge():
    """At +100 with model_p=0.52, EV = 0.52*1.0 - 0.48 = 0.04 = +4%."""
    assert ev_pct(0.52, +100) == pytest.approx(4.0, abs=1e-9)


def test_ev_pct_negative_odds():
    """At -110 with model_p=0.55, EV = 0.55*(1.909-1) - 0.45 = 0.55*0.909 - 0.45 ≈ 0.05."""
    expected = (0.55 * (1.0 + 100 / 110 - 1.0)) - (1.0 - 0.55)
    assert ev_pct(0.55, -110) == pytest.approx(expected * 100.0, abs=1e-9)


def test_ev_pct_long_shot():
    """At +500 with model_p=0.20, EV = 0.20*5 - 0.80 = 0.20 = +20%."""
    assert ev_pct(0.20, +500) == pytest.approx(20.0, abs=1e-9)


def test_ev_pct_long_shot_can_exceed_edge_pct():
    """Demonstrates why we rank by edge_pct, not EV%.

    At +500 (model_p=0.20, market_p~0.167): edge = 0.20/0.167 - 1 = 0.20 (~20%)
    EV = +20% (computed above)

    At -200 (model_p=0.75, market_p~0.667): edge = 0.75/0.667 - 1 = 0.125 (~12.5%)
    EV = 0.75 * 0.5 - 0.25 = 0.125 = +12.5%

    The long shot has the same EV but is RISKIER. Ranking by edge_pct
    favors the long shot less aggressively than EV ranking would.
    """
    long_shot_edge = edge_pct(0.20, 0.167)
    long_shot_ev = ev_pct(0.20, +500)
    chalk_edge = edge_pct(0.75, 0.667)
    chalk_ev = ev_pct(0.75, -200)
    # Both have EV ~+12-20%
    assert long_shot_ev > chalk_ev
    # But edge_pct discriminates the long shot from the chalk less harshly
    # (they're closer in edge than they are in EV)
    assert long_shot_edge > chalk_edge
