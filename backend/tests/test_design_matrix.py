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
    ZERO_VARIANCE_THRESHOLD,
    DesignMatrixError,
    InsufficientSampleError,
    ReparameterizedDesign,
    check_design_matrix,
    drop_nan_rows,
    reparameterize_bf,
    reparameterize_kpa,
    require_min_sample,
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
    diag = check_design_matrix(out.matrix, name="reparam_ok")
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
    design = reparameterize_kpa(features)
    # Should have league_k baseline + pitcher_delta, no longer collinear.
    check_design_matrix(design.matrix, name="kpa_reparam")  # no raise


# ---- reparameterize_kpa ----------------------------------------------------


def test_reparam_kpa_creates_pitcher_delta():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225, 0.220],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200, 0.225],
    })
    out = reparameterize_kpa(features)
    assert "pitcher_k_pct_delta" in out.matrix.columns
    np.testing.assert_allclose(
        out.matrix["pitcher_k_pct_delta"].to_numpy(), [0.025, -0.025, 0.005], atol=1e-9,
    )


def test_reparam_kpa_30d_delta_uses_pitcher_season():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200],
        "pitcher_k_pct_30d_blended": [0.260, 0.190],
    })
    out = reparameterize_kpa(features)
    assert "pitcher_k_pct_30d_delta" in out.matrix.columns
    assert list(out.matrix["pitcher_k_pct_30d_delta"]) == [
        pytest.approx(0.010), pytest.approx(-0.010),
    ]


def test_reparam_kpa_park_factor_to_log():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225],
        "park_k_factor": [1.10, 0.95],
    })
    out = reparameterize_kpa(features)
    assert "log_park_k_factor" in out.matrix.columns
    np.testing.assert_allclose(
        out.matrix["log_park_k_factor"].to_numpy(),
        [np.log(1.10), np.log(0.95)],
    )


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
    assert "pitcher_pitches_per_pa_30d_delta" in out.matrix.columns
    np.testing.assert_allclose(
        out.matrix["pitcher_pitches_per_pa_30d_delta"].to_numpy(),
        [0.05, -0.05],
    )


def test_reparam_bf_park_factor_to_log():
    features = pd.DataFrame({"park_run_environment_factor": [1.0, 1.12]})
    out = reparameterize_bf(features)
    # log(1.0) is 0 — that column would have zero variance if all values were
    # 1.0. With one non-1.0 value here, it survives the zero-variance drop.
    assert "log_park_run_factor" in out.matrix.columns
    assert out.matrix["log_park_run_factor"].iloc[0] == 0.0
    assert out.matrix["log_park_run_factor"].iloc[1] == pytest.approx(np.log(1.12))


# ---- Zero-variance column drop (Fix B for the smoke halt) ------------------


def test_reparam_kpa_drops_zero_variance_umpire_column():
    """umpire_k_factor=1.0 for every row -> log=0 for every row -> dropped."""
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225, 0.220, 0.220],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200, 0.225, 0.230],
        "umpire_k_zone_factor": [1.0, 1.0, 1.0, 1.0],
        "park_k_factor": [1.10, 0.95, 1.05, 0.97],  # varies
    })
    out = reparameterize_kpa(features)
    assert "log_umpire_k_factor" in out.dropped_columns
    assert "log_umpire_k_factor" not in out.matrix.columns
    # Park K factor survives (it varies).
    assert "log_park_k_factor" in out.matrix.columns


def test_reparam_bf_drops_zero_variance_columns():
    features = pd.DataFrame({
        "pitcher_pitches_per_pa_season": [3.85, 4.00, 3.90, 4.10],
        "pitcher_pitches_per_pa_30d": [3.85, 4.00, 3.90, 4.10],  # identical
        "park_run_environment_factor": [1.0, 1.0, 1.0, 1.0],     # all neutral
        "park_k_factor": [1.10, 0.95, 1.05, 0.97],
    })
    out = reparameterize_bf(features)
    # 30d_delta is identically 0 -> dropped
    assert "pitcher_pitches_per_pa_30d_delta" in out.dropped_columns
    # log_park_run_factor is identically 0 -> dropped
    assert "log_park_run_factor" in out.dropped_columns
    # park_k_factor varies -> retained
    assert "log_park_k_factor" in out.matrix.columns


def test_zero_variance_threshold_boundary():
    """std=1e-7 is dropped, std=1e-5 is retained."""
    rng = np.random.default_rng(0)
    n = 100
    # baseline col so reparameterize_bf has something to compute against
    base = rng.normal(4.0, 0.5, n)
    # Build a delta column with controlled std after subtraction
    features_low = pd.DataFrame({
        "pitcher_pitches_per_pa_season": base,
        "pitcher_pitches_per_pa_30d": base + rng.normal(0, 1e-7, n),  # std ~1e-7
    })
    out_low = reparameterize_bf(features_low)
    assert "pitcher_pitches_per_pa_30d_delta" in out_low.dropped_columns

    features_hi = pd.DataFrame({
        "pitcher_pitches_per_pa_season": base,
        "pitcher_pitches_per_pa_30d": base + rng.normal(0, 1e-4, n),  # std ~1e-4
    })
    out_hi = reparameterize_bf(features_hi)
    assert "pitcher_pitches_per_pa_30d_delta" in out_hi.matrix.columns
    assert "pitcher_pitches_per_pa_30d_delta" not in out_hi.dropped_columns


def test_reparam_returns_reparameterized_design_with_metadata():
    features = pd.DataFrame({
        "league_k_pct_vs_hand": [0.225, 0.225],
        "pitcher_k_pct_season_shrunk": [0.250, 0.200],
    })
    out = reparameterize_kpa(features)
    assert isinstance(out, ReparameterizedDesign)
    assert isinstance(out.matrix, pd.DataFrame)
    assert isinstance(out.dropped_columns, list)


def test_check_design_matrix_halts_only_on_fully_empty_after_drops():
    """Individual column drops shouldn't halt; only an entirely empty matrix
    (zero columns) trips check_design_matrix."""
    # Build a features frame where every reparameterized column becomes
    # zero variance (all factors at neutral).
    features = pd.DataFrame({
        "park_run_environment_factor": [1.0, 1.0, 1.0],
        "park_k_factor": [1.0, 1.0, 1.0],
    })
    out = reparameterize_bf(features)
    # Both log columns dropped -> matrix has 0 columns.
    assert out.matrix.shape[1] == 0
    # check_design_matrix should halt on this empty matrix.
    with pytest.raises(DesignMatrixError, match="empty"):
        check_design_matrix(out.matrix, name="all_dropped")


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


# ---- drop_nan_rows ---------------------------------------------------------


def test_drop_nan_rows_drops_rows_not_columns():
    """When a delta value is NaN, the ROW is dropped, not the column.

    This is the Phase 4c Fix A invariant: undefined deltas are treated as
    missing-required-feature (drop the game) rather than median-filled or
    silently coerced to zero.
    """
    X_bf = pd.DataFrame({
        "ip_per_start": [5.0, 5.5, 6.0, 4.8],
        "p_per_pa_30d_delta": [0.05, np.nan, -0.10, 0.02],
    })
    X_kpa = pd.DataFrame({
        "pitcher_k_delta": [0.01, 0.02, np.nan, -0.01],
        "park_log": [0.0, 0.05, 0.0, 0.0],
    })
    [bf_clean, kpa_clean], drops = drop_nan_rows(X_bf, X_kpa)

    # Two rows had NaN somewhere (row 1 in BF, row 2 in KPA). Both dropped.
    assert len(bf_clean) == 2
    assert len(kpa_clean) == 2
    # Same rows survived in both matrices (alignment preserved).
    assert list(bf_clean["ip_per_start"]) == [5.0, 4.8]
    assert list(kpa_clean["pitcher_k_delta"]) == [0.01, -0.01]

    # Drop counts reported per offending column, not aggregated.
    assert drops["p_per_pa_30d_delta"] == 1
    assert drops["pitcher_k_delta"] == 1
    # Columns with no NaN don't appear in drops at all.
    assert "ip_per_start" not in drops
    assert "park_log" not in drops


def test_drop_nan_rows_preserves_all_columns():
    """Verify drop_nan_rows is row-only, never column-pruning."""
    X = pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    [cleaned], drops = drop_nan_rows(X)
    assert list(cleaned.columns) == ["a", "b"]  # both columns retained
    assert len(cleaned) == 2


def test_drop_nan_rows_returns_empty_for_empty_input():
    cleaned, drops = drop_nan_rows()
    assert cleaned == []
    assert drops == {}


def test_drop_nan_rows_rejects_mismatched_lengths():
    X_a = pd.DataFrame({"a": [1.0, 2.0]})
    X_b = pd.DataFrame({"b": [3.0, 4.0, 5.0]})
    with pytest.raises(ValueError, match="same row count"):
        drop_nan_rows(X_a, X_b)


def test_drop_nan_rows_no_nans_returns_input_unchanged():
    X_bf = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_kpa = pd.DataFrame({"b": [0.1, 0.2, 0.3]})
    [bf, kpa], drops = drop_nan_rows(X_bf, X_kpa)
    assert len(bf) == 3
    assert len(kpa) == 3
    assert drops == {}


# ---- require_min_sample ----------------------------------------------------


def test_require_min_sample_passes_when_sufficient():
    require_min_sample(2000, 2000)  # exactly the threshold — passes
    require_min_sample(5000, 100)


def test_require_min_sample_halts_with_clear_error():
    with pytest.raises(InsufficientSampleError, match="50 < 100"):
        require_min_sample(50, 100)


def test_require_min_sample_includes_context_in_message():
    with pytest.raises(InsufficientSampleError, match="post-NaN-drop"):
        require_min_sample(10, 100, context="post-NaN-drop fit pool")


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
