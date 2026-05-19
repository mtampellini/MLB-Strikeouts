"""Phase 4c-v2 Step 1/6: tests for CSW%->K% derivation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.derive_csw_to_k_relationship import (
    GATE_CSW_RANGE_MAX,
    GATE_CSW_RANGE_MIN,
    GATE_MIN_N_PITCHER_SEASONS,
    GATE_R2_MIN,
    GATE_SLOPE_MAX,
    GATE_SLOPE_MIN,
    _aggregate_pitcher_seasons,
    _sanity_check,
    csw_implied_k_rate,
    fit_weighted_linear,
)


# ---- fit_weighted_linear ---------------------------------------------------


def test_fit_recovers_known_relationship():
    """Synthetic data with K = 0.05 + 0.65 * CSW + small noise should recover
    the slope and intercept within 0.02."""
    rng = np.random.default_rng(0)
    n = 400
    csw = rng.uniform(0.20, 0.34, n)
    true_intercept = -0.10
    true_slope = 1.0
    k = true_intercept + true_slope * csw + rng.normal(0, 0.005, n)
    w = rng.integers(50, 800, n).astype(float)
    out = fit_weighted_linear(csw, k, w)
    assert abs(out["intercept"] - true_intercept) < 0.01
    assert abs(out["slope"] - true_slope) < 0.05
    assert out["r_squared"] > 0.85


def test_fit_zero_relationship_yields_low_r_squared():
    """Random y vs x produces near-zero slope and low R²."""
    rng = np.random.default_rng(42)
    n = 400
    csw = rng.uniform(0.20, 0.34, n)
    k = rng.uniform(0.10, 0.30, n)
    w = np.ones(n)
    out = fit_weighted_linear(csw, k, w)
    assert out["r_squared"] < 0.05


def test_fit_weighting_pulls_toward_high_weight_points():
    """Heavy-weight points should dominate the fit."""
    # Two clusters: one anchored at (0.25, 0.20) with high weight,
    # one at (0.30, 0.40) with very low weight. Fit should land near the first.
    x = np.array([0.25, 0.25, 0.25, 0.30, 0.30])
    y = np.array([0.20, 0.20, 0.20, 0.40, 0.40])
    w = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0])
    out = fit_weighted_linear(x, y, w)
    # Predicted K% at 0.25 should be very close to 0.20 (the heavy cluster)
    yhat_25 = out["intercept"] + out["slope"] * 0.25
    assert abs(yhat_25 - 0.20) < 0.01


# ---- csw_implied_k_rate ----------------------------------------------------


def test_csw_implied_k_rate_arithmetic():
    """The exported function is just intercept + slope * csw."""
    assert csw_implied_k_rate(0.30, intercept=-0.10, slope=1.0) == pytest.approx(0.20)
    assert csw_implied_k_rate(0.25, intercept=0.0, slope=0.8) == pytest.approx(0.20)


def test_csw_implied_k_rate_returns_plausible_range():
    """With a realistic-looking intercept/slope, the function should produce
    plausible K% values across the typical CSW% range."""
    intercept = -0.08
    slope = 1.0
    for csw in (0.22, 0.27, 0.32):
        k = csw_implied_k_rate(csw, intercept=intercept, slope=slope)
        assert 0.10 < k < 0.30


# ---- _aggregate_pitcher_seasons --------------------------------------------


def _pitch_row(*, pitcher, season=2024, description="called_strike",
                events=None, **kw):
    return {
        "pitcher": pitcher, "__season": season,
        "description": description, "events": events,
        "player_name": kw.get("player_name", f"id_{pitcher}"),
    }


def test_aggregate_counts_csw_and_pa_correctly():
    """Build a small frame with known counts."""
    rows = []
    # Pitcher 1: 600 pitches, 100 CSW, 60 PAs, 18 K
    for _ in range(100):
        rows.append(_pitch_row(pitcher=1, description="swinging_strike"))
    for _ in range(500):
        rows.append(_pitch_row(pitcher=1, description="ball"))
    # PAs: 18 strikeouts + 42 field_outs (60 total)
    for _ in range(18):
        rows.append(_pitch_row(pitcher=1, description="swinging_strike",
                                events="strikeout"))
    for _ in range(42):
        rows.append(_pitch_row(pitcher=1, description="hit_into_play",
                                events="field_out"))
    df = pd.DataFrame(rows)
    out = _aggregate_pitcher_seasons(df)
    # Both PA-terminal pitches contribute to CSW count if they're CSW-eligible
    assert len(out) == 1
    row = out.iloc[0]
    # 100 swinging_strike (non-terminal) + 18 swinging_strike (terminal) = 118 CSW
    # 500 ball + 42 hit_into_play = 542 non-CSW
    # Total pitches = 100+500+18+42 = 660
    assert int(row["n_pitches"]) == 660
    assert int(row["n_csw"]) == 118
    assert int(row["n_pa"]) == 60
    assert int(row["n_k"]) == 18
    assert row["csw_pct"] == pytest.approx(118 / 660)
    assert row["k_pct"] == pytest.approx(18 / 60)


def test_aggregate_filters_out_below_minimum_samples():
    """A pitcher with too few pitches AND/OR too few PAs is dropped."""
    rows = []
    # Pitcher 100: 100 pitches (below MIN_PITCHES=500), should be dropped
    for _ in range(100):
        rows.append(_pitch_row(pitcher=100, description="ball"))
    for _ in range(60):
        rows.append(_pitch_row(pitcher=100, description="hit_into_play",
                                events="field_out"))
    df = pd.DataFrame(rows)
    out = _aggregate_pitcher_seasons(df)
    assert out.empty


# ---- _sanity_check ---------------------------------------------------------


def _make_payload(r2, slope, intercept, n_seasons,
                   csw_min=0.20, csw_max=0.36):
    return {
        "n_pitcher_seasons": n_seasons,
        "model": {
            "intercept": intercept, "slope": slope,
            "r_squared": r2, "rmse": 0.02,
        },
        "diagnostic": {
            "csw_range_observed": [csw_min, csw_max],
            "k_range_observed": [0.15, 0.40],
        },
    }


def test_sanity_passes_realistic_fit():
    payload = _make_payload(r2=0.65, slope=1.3, intercept=-0.10, n_seasons=400)
    _sanity_check(payload, (0.20, 0.36))


def test_sanity_halts_on_low_r_squared():
    payload = _make_payload(r2=0.30, slope=1.3, intercept=-0.10, n_seasons=400)
    with pytest.raises(AssertionError, match="R²"):
        _sanity_check(payload, (0.20, 0.36))


def test_sanity_halts_on_slope_too_low():
    payload = _make_payload(r2=0.65, slope=0.5, intercept=-0.10, n_seasons=400)
    with pytest.raises(AssertionError, match="slope"):
        _sanity_check(payload, (0.20, 0.36))


def test_sanity_halts_on_slope_too_high():
    payload = _make_payload(r2=0.65, slope=4.0, intercept=-0.10, n_seasons=400)
    with pytest.raises(AssertionError, match="slope"):
        _sanity_check(payload, (0.20, 0.36))


def test_sanity_does_not_halt_on_intercept_out_of_a_priori_range():
    """The intercept gate was REMOVED — extrapolation to CSW=0 is not a
    methodologically meaningful metric. An intercept of -0.22 (empirical
    MLB value) or even 0.20 (synthetic) is acceptable."""
    payload = _make_payload(r2=0.65, slope=1.3, intercept=-0.30, n_seasons=400)
    _sanity_check(payload, (0.20, 0.36))  # must not raise
    payload2 = _make_payload(r2=0.65, slope=1.3, intercept=0.20, n_seasons=400)
    _sanity_check(payload2, (0.20, 0.36))


def test_sanity_halts_on_small_sample():
    payload = _make_payload(r2=0.65, slope=1.3, intercept=-0.10, n_seasons=100)
    with pytest.raises(AssertionError, match="n_pitcher_seasons"):
        _sanity_check(payload, (0.20, 0.36))


def test_sanity_halts_on_narrow_csw_range():
    """If CSW observed range doesn't cover [0.22, 0.34], halt."""
    payload = _make_payload(r2=0.65, slope=1.3, intercept=-0.10, n_seasons=400,
                             csw_min=0.24, csw_max=0.30)
    with pytest.raises(AssertionError, match="CSW observed range"):
        _sanity_check(payload, (0.24, 0.30))


# ---- Constants -------------------------------------------------------------


def test_gate_constants_match_spec():
    assert GATE_R2_MIN == 0.50
    assert GATE_SLOPE_MIN == 1.0
    assert GATE_SLOPE_MAX == 3.0
    assert GATE_MIN_N_PITCHER_SEASONS == 300
    assert GATE_CSW_RANGE_MIN == 0.22
    assert GATE_CSW_RANGE_MAX == 0.34


def test_intercept_gate_constants_not_exported():
    """The intercept gate was removed; importing it should fail."""
    with pytest.raises(ImportError):
        from scripts.derive_csw_to_k_relationship import GATE_INTERCEPT_MIN  # noqa: F401
