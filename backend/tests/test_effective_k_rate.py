"""Phase 4c-v2 Step 2/6: tests for the effective K rate blend."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.projection.effective_k_rate import (
    DEFAULT_K_PRIOR_PA,
    EffectiveKRateResult,
    P_K_CEIL,
    P_K_FLOOR,
    compute_effective_k_rate,
    load_csw_to_k_relationship,
)


# Step 1 derived: K% = -0.2209 + 1.6346 * CSW%
LIVE_INTERCEPT = -0.2209
LIVE_SLOPE = 1.6346


# ---- Standard blend cases --------------------------------------------------


def test_well_established_pitcher_observed_dominates():
    """500 PAs at K=0.28 with CSW=0.30 -> blend leans heavily on observed."""
    r = compute_effective_k_rate(
        observed_k_rate=0.28, observed_n_pa=500,
        csw_pct=0.30,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    # csw_implied = -0.2209 + 1.6346 * 0.30 = 0.2694
    assert abs(r.csw_implied_k_rate - (-0.2209 + 1.6346 * 0.30)) < 1e-9
    # w_observed = 500 / (500 + 150) = 0.7692
    assert abs(r.blend_weight_observed - 500 / 650) < 1e-9
    # effective = 0.7692*0.28 + 0.2308*0.2694 = 0.2766
    expected = (500 / 650) * 0.28 + (150 / 650) * (-0.2209 + 1.6346 * 0.30)
    assert abs(r.effective_k_rate - expected) < 1e-9
    assert r.confidence == "observed_dominant"


def test_early_season_balanced_blend():
    """80 PAs at K=0.30, CSW=0.32 -> balanced (w_observed ≈ 0.35)."""
    r = compute_effective_k_rate(
        observed_k_rate=0.30, observed_n_pa=80,
        csw_pct=0.32,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert abs(r.blend_weight_observed - 80 / 230) < 1e-9
    assert r.confidence == "balanced"


def test_csw_dominant_when_observed_sample_tiny():
    """20 PAs vs 150 prior -> w_observed = 20/170 = 0.118 → csw_dominant."""
    r = compute_effective_k_rate(
        observed_k_rate=0.40, observed_n_pa=20,
        csw_pct=0.30,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert r.confidence == "csw_dominant"
    # effective should land much closer to csw_implied than to observed
    csw_implied = -0.2209 + 1.6346 * 0.30
    assert abs(r.effective_k_rate - csw_implied) < abs(r.effective_k_rate - 0.40)


# ---- Edge cases ------------------------------------------------------------


def test_observed_none_falls_back_to_csw_only():
    """No observed PAs but CSW available -> use csw_implied."""
    r = compute_effective_k_rate(
        observed_k_rate=None, observed_n_pa=0,
        csw_pct=0.31,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    csw_implied = -0.2209 + 1.6346 * 0.31
    assert abs(r.effective_k_rate - csw_implied) < 1e-9
    assert r.observed_k_rate is None
    assert r.blend_weight_observed == 0.0
    assert r.confidence == "csw_only_fallback"


def test_csw_none_uses_observed_only():
    """Observed available, CSW missing -> use observed (clipped)."""
    r = compute_effective_k_rate(
        observed_k_rate=0.27, observed_n_pa=500,
        csw_pct=None,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert r.effective_k_rate == 0.27
    assert r.csw_implied_k_rate is None
    assert r.blend_weight_observed == 1.0
    assert r.confidence == "observed_only_no_csw"


def test_both_none_returns_none():
    r = compute_effective_k_rate(
        observed_k_rate=None, observed_n_pa=0,
        csw_pct=None,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is None


# ---- Clipping --------------------------------------------------------------


def test_effective_k_clipped_at_floor():
    """An extremely-low blend (e.g., observed K=0.01) clips to P_K_FLOOR=0.05."""
    r = compute_effective_k_rate(
        observed_k_rate=0.01, observed_n_pa=10_000,  # enormous weight on observed
        csw_pct=0.05,                                  # CSW also extremely low
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert r.effective_k_rate == P_K_FLOOR


def test_effective_k_clipped_at_ceiling():
    """An extreme high blend clips to P_K_CEIL=0.50."""
    r = compute_effective_k_rate(
        observed_k_rate=0.65, observed_n_pa=10_000,
        csw_pct=0.50,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert r.effective_k_rate == P_K_CEIL


def test_csw_only_path_also_clips():
    """CSW-only fallback respects floor/ceil clipping."""
    r = compute_effective_k_rate(
        observed_k_rate=None, observed_n_pa=0,
        csw_pct=0.05,  # csw_implied = -0.22 + 1.63*0.05 = -0.139 -> clip to floor
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    assert r.effective_k_rate == P_K_FLOOR


# ---- Skenes-style case (the canonical motivating example) ------------------


def test_skenes_style_case_matches_spec():
    """observed_K=0.331, CSW=0.294, n=200 PAs (from the spec).

    csw_implied_k = -0.2209 + 1.6346 * 0.294 = 0.2596 (approx 0.260)
    w_observed = 200 / (200 + 150) = 0.5714
    effective = 0.5714 * 0.331 + 0.4286 * 0.2596 = 0.301
    confidence: "balanced"
    """
    r = compute_effective_k_rate(
        observed_k_rate=0.331, observed_n_pa=200,
        csw_pct=0.294,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert r is not None
    csw_implied = -0.2209 + 1.6346 * 0.294
    assert abs(r.csw_implied_k_rate - csw_implied) < 1e-9
    assert abs(r.blend_weight_observed - 200 / 350) < 1e-9
    expected = (200/350) * 0.331 + (150/350) * csw_implied
    assert abs(r.effective_k_rate - expected) < 1e-9
    # Approximately 0.301 per spec
    assert 0.298 < r.effective_k_rate < 0.305
    # Confidence: 0.5714 is in [0.25, 0.75] → balanced
    assert r.confidence == "balanced"
    # Sanity: effective sits between observed and csw_implied
    assert csw_implied < r.effective_k_rate < r.observed_k_rate


# ---- Configurable k_prior_pa -----------------------------------------------


def test_smaller_k_prior_shifts_weight_to_observed():
    """A small k_prior_pa (e.g. 50) makes 100-PA pitcher observed-dominant."""
    r = compute_effective_k_rate(
        observed_k_rate=0.30, observed_n_pa=100,
        csw_pct=0.25,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
        k_prior_pa=50,
    )
    assert r is not None
    assert abs(r.blend_weight_observed - 100 / 150) < 1e-9
    # With k_prior=50, w_observed = 0.667 → still balanced (not > 0.75)


def test_default_k_prior_is_150():
    assert DEFAULT_K_PRIOR_PA == 150


# ---- load_csw_to_k_relationship --------------------------------------------


def test_load_csw_to_k_relationship_returns_live_values():
    """The Step 1 output file should load and produce the expected
    intercept/slope (within rounding)."""
    intercept, slope = load_csw_to_k_relationship()
    # From Step 1: K% = -0.2209 + 1.6346 * CSW%
    assert abs(intercept - (-0.2209)) < 1e-3
    assert abs(slope - 1.6346) < 1e-3


def test_load_csw_to_k_relationship_raises_on_missing(tmp_path):
    missing = tmp_path / "no_such_file.json"
    with pytest.raises(FileNotFoundError, match="derive_csw_to_k_relationship"):
        load_csw_to_k_relationship(missing)


def test_load_csw_to_k_relationship_raises_on_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": {"intercept": 0.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        load_csw_to_k_relationship(bad)


# ---- Result dataclass schema -----------------------------------------------


def test_result_dataclass_fields_are_complete():
    """The result has every field the consumer needs for debugging."""
    r = compute_effective_k_rate(
        observed_k_rate=0.25, observed_n_pa=400,
        csw_pct=0.28,
        csw_to_k_intercept=LIVE_INTERCEPT, csw_to_k_slope=LIVE_SLOPE,
    )
    assert isinstance(r, EffectiveKRateResult)
    for attr in (
        "effective_k_rate", "observed_k_rate", "observed_n_pa",
        "csw_implied_k_rate", "blend_weight_observed", "confidence",
    ):
        assert hasattr(r, attr), f"missing field: {attr}"
