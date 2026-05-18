"""Phase 4c design-matrix sanity tests.

These regression-test the structural fix for the failed first attempt:
condition-number check, no-NaN check, zero-variance check, delta-centered
check. If a future change reintroduces collinear features into the design
matrix, these halt the fit before any cycles are wasted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.design_matrix import (
    DEFAULT_CONDITION_THRESHOLD,
    DesignMatrixError,
    check_design_matrix,
    reparameterize_bf,
    reparameterize_kpa,
)


# ---- check_design_matrix ---------------------------------------------------


def test_check_passes_on_well_conditioned_matrix():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "feat_a": rng.normal(0, 1, 100),
        "feat_b": rng.normal(0, 1, 100),
        "feat_c": rng.normal(0, 1, 100),
    })
    diag = check_design_matrix(X, name="test")
    assert diag["n_rows"] == 100
    assert diag["n_columns"] == 3
    assert diag["condition_number"] < 5  # truly random data is near-orthogonal


def test_check_halts_on_nan():
    X = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(DesignMatrixError, match="NaN"):
        check_design_matrix(X, name="test")


def test_check_halts_on_zero_variance():
    X = pd.DataFrame({
        "constant": [1.0, 1.0, 1.0, 1.0],
        "varies": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(DesignMatrixError, match=r"zero-variance.*constant"):
        check_design_matrix(X, name="test")


def test_check_halts_on_collinearity():
    """The Phase 4c first-attempt failure: two columns expressing the same
    information. Condition number explodes."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 200)
    X = pd.DataFrame({
        "a": base,
        # Nearly identical column with tiny perturbation
        "b": base + rng.normal(0, 0.001, 200),
    })
    with pytest.raises(DesignMatrixError, match="condition number"):
        check_design_matrix(X, name="test", condition_threshold=100)


def test_phase4c_first_attempt_pattern_is_fixed_by_reparam():
    """The Phase 4c first attempt failed because pitcher_k_pct and league_k_pct
    expressed the same level information. The condition-number check ALONE
    wouldn't have caught it (the matrix wasn't strictly singular — it was
    nearly redundant in a way that produced unstable but technically-valid
    OLS solutions). The real fix is the re-parameterization: replace the
    level pair with (baseline, delta-from-baseline), which gives the columns
    orthogonal interpretations.

    This test reproduces the offending input shape and verifies the
    reparameterized output is well-conditioned.
    """
    rng = np.random.default_rng(42)
    n = 500
    league_k = rng.choice([0.2254, 0.2200, 0.2218, 0.2160], size=n)
    pitcher_k = league_k + rng.normal(0, 0.02, n)
    features = pd.DataFrame({
        "league_k_pct_vs_hand": league_k,
        "pitcher_k_pct_season_shrunk": pitcher_k,
    })
    out = reparameterize_kpa(features)
    # After reparam: baseline + delta. Delta is centered around 0 (since
    # pitcher_k is league_k plus zero-mean noise) and uncorrelated with the
    # baseline by construction.
    diag = check_design_matrix(out, name="reparam_ok")
    assert diag["condition_number"] < 50


def test_check_passes_after_reparameterization():
    """The same data but with delta-centering should pass."""
    rng = np.random.default_rng(42)
    n = 500
    seasons = rng.choice([2024, 2025], size=n)
    hands = rng.choice(["R", "L"], size=n)
    league_k = np.where(
        (seasons == 2024) & (hands == "R"), 0.2254,
        np.where(
            (seasons == 2024) & (hands == "L"), 0.2200,
            np.where((seasons == 2025) & (hands == "R"), 0.2218, 0.2160),
        ),
    )
    pitcher_k = league_k + rng.normal(0, 0.02, n)
    features = pd.DataFrame({
        "league_k_pct_vs_hand": league_k,
        "pitcher_k_pct_season_shrunk": pitcher_k,
    })
    X_reparam = reparameterize_kpa(features)
    # Should have league_k baseline + pitcher_delta, no longer collinear.
    check_design_matrix(X_reparam, name="kpa_reparam")  # no raise


# ---- reparameterize_kpa ----------------------------------------------------


def test_reparam_kpa_creates_pitcher_delta():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225, 0.220],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200, 0.225],
    })
    out = reparameterize_kpa(features)
    assert "pitcher_k_pct_delta" in out.columns
    np.testing.assert_allclose(
        out["pitcher_k_pct_delta"].to_numpy(), [0.025, -0.025, 0.005], atol=1e-9,
    )


def test_reparam_kpa_30d_delta_uses_pitcher_season():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200],
        "pitcher_k_pct_30d_blended": [0.260, 0.190],
    })
    out = reparameterize_kpa(features)
    assert "pitcher_k_pct_30d_delta" in out.columns
    # 0.260 - 0.250 = 0.010; 0.190 - 0.200 = -0.010
    assert list(out["pitcher_k_pct_30d_delta"]) == [pytest.approx(0.010), pytest.approx(-0.010)]


def test_reparam_kpa_park_factor_to_log():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225],
        "park_k_factor": [1.10],
    })
    out = reparameterize_kpa(features)
    assert "log_park_k_factor" in out.columns
    assert out["log_park_k_factor"].iloc[0] == pytest.approx(np.log(1.10))


def test_reparam_kpa_raises_on_missing_league_baseline():
    features = pd.DataFrame({
        "pitcher_k_pct_season_shrunk": [0.225],
    })
    with pytest.raises(DesignMatrixError, match="league_k_pct_vs_hand"):
        reparameterize_kpa(features)


# ---- reparameterize_bf -----------------------------------------------------


def test_reparam_bf_30d_delta_from_season():
    features = pd.DataFrame({
        "pitcher_pitches_per_pa_season": [3.85, 4.00],
        "pitcher_pitches_per_pa_30d": [3.90, 3.95],
    })
    out = reparameterize_bf(features)
    assert "pitcher_pitches_per_pa_30d_delta" in out.columns
    assert list(out["pitcher_pitches_per_pa_30d_delta"]) == [
        pytest.approx(0.05), pytest.approx(-0.05),
    ]


def test_reparam_bf_park_factor_to_log():
    features = pd.DataFrame({"park_run_environment_factor": [1.0, 1.12]})
    out = reparameterize_bf(features)
    assert "log_park_run_factor" in out.columns
    assert out["log_park_run_factor"].iloc[0] == 0.0
    assert out["log_park_run_factor"].iloc[1] == pytest.approx(np.log(1.12))


# ---- Delta-mean centering check --------------------------------------------


def test_delta_columns_must_be_centered():
    """Delta columns should average to ~0 within each (season, hand) cell."""
    X = pd.DataFrame({
        "pitcher_k_pct_delta": [0.10, -0.10, 0.10, -0.10],
        # Non-zero column to satisfy condition-number; small magnitude so it
        # doesn't dominate
        "varies": [1.0, 2.0, 3.0, 4.0],
        "season": [2024, 2024, 2025, 2025],
        "p_throws": ["R", "R", "R", "R"],
    })
    # Within season=2024 hand=R: mean of pitcher_k_pct_delta = 0 (passes)
    # Within season=2025 hand=R: mean = 0 (passes)
    check_design_matrix(
        X[["pitcher_k_pct_delta", "varies"]],
        delta_columns=["pitcher_k_pct_delta"],
        delta_group_cols=None,  # global mean check would be 0
        name="test",
    )


def test_check_halts_on_bad_delta_centering():
    """If a delta column has a clear non-zero mean within a cell, halt.

    Use random noise for the auxiliary column so the condition-number check
    doesn't trip first; the delta-centering check is what we're testing.
    """
    rng = np.random.default_rng(7)
    n_per_cell = 50
    deltas = np.concatenate([
        rng.normal(0.10, 0.02, n_per_cell),   # 2024 cell off-center by +0.10
        rng.normal(-0.05, 0.02, n_per_cell),  # 2025 cell off-center by -0.05
    ])
    aux = rng.normal(0, 1, 2 * n_per_cell)
    X = pd.DataFrame({
        "pitcher_k_pct_delta": deltas,
        "varies": aux,
        "season": [2024] * n_per_cell + [2025] * n_per_cell,
    })
    with pytest.raises(DesignMatrixError, match="not centered"):
        check_design_matrix(
            X[["pitcher_k_pct_delta", "varies"]],
            delta_columns=["pitcher_k_pct_delta"],
            delta_group_cols=None,  # global mean check (still > tolerance)
            delta_mean_tolerance=0.01,
            name="test",
        )
