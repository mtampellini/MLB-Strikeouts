"""Phase 4c design-matrix utilities: re-parameterization + sanity checks.

The Phase 4c (first attempt) failure was structural collinearity between
`pitcher_k_pct_season` and `league_k_pct_vs_hand` — two features
expressing overlapping level information. The fix is to centerthe pitcher
column against the league baseline so the two columns carry orthogonal
signals.

This module provides:
- :func:`reparameterize`: take a feature DataFrame and return the
  delta-centered design matrix.
- :func:`check_design_matrix`: assert variance, no-NaN, and condition
  number < threshold (default 100).

Sanity checks raise :class:`DesignMatrixError` on failure — fail-fast so
the overnight rerun halts immediately if any pre-fit condition is wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReparameterizedDesign:
    """Output of a reparameterize_* call.

    ``matrix`` is the design matrix. ``dropped_columns`` lists features that
    were built but then dropped because they had std < ZERO_VARIANCE_THRESHOLD
    (no signal to fit). The fit script surfaces ``dropped_columns`` in its
    JSON output so future projection code knows which features weren't
    actually fit and should be treated as neutral (factor=1.0, log-odds=0).
    """

    matrix: pd.DataFrame
    dropped_columns: list[str] = field(default_factory=list)


ZERO_VARIANCE_THRESHOLD = 1e-6


def _drop_zero_variance(df: pd.DataFrame, *, context: str) -> ReparameterizedDesign:
    """Drop columns with std < ZERO_VARIANCE_THRESHOLD; log each removal.

    Self-healing by construction: a column entirely composed of neutral
    values (log(1.0)=0 for every row when its underlying factor file is
    placeholder all-1.0) returns std=0 and is dropped. When real data
    eventually lands and the column gets non-zero variance, it re-enters
    the design matrix automatically — no code change required.
    """
    if df.empty:
        return ReparameterizedDesign(matrix=df, dropped_columns=[])
    stds = df.std(numeric_only=True)
    dropped: list[str] = []
    for col, s in stds.items():
        if pd.isna(s) or s < ZERO_VARIANCE_THRESHOLD:
            dropped.append(str(col))
            logger.warning(
                "%s: dropped column %r from fit (std=%.3g, feature data "
                "is uniformly neutral — no signal to fit)",
                context, col, 0.0 if pd.isna(s) else float(s),
            )
    if dropped:
        df = df.drop(columns=dropped)
    return ReparameterizedDesign(matrix=df, dropped_columns=dropped)


class DesignMatrixError(AssertionError):
    """Design matrix violates a pre-fit sanity check."""


class InsufficientSampleError(AssertionError):
    """Sample size after NaN drops is below the minimum required for fit."""


# Default condition number threshold. The Phase 4c first attempt had
# implicit condition >> 10000 — the fit was numerically degenerate. A
# threshold of 100 is conservative: well-conditioned regression problems
# typically have condition numbers in the single digits to low hundreds.
DEFAULT_CONDITION_THRESHOLD = 100.0
DEFAULT_DELTA_MEAN_TOLERANCE = 0.01


def drop_nan_rows(
    *matrices: pd.DataFrame,
) -> tuple[list[pd.DataFrame], dict[str, int]]:
    """Drop rows where ANY column in ANY matrix is NaN.

    Treats undefined deltas (e.g. ``30d - season_baseline`` when the
    baseline is missing) as missing-required-feature — same discipline as
    every other missing-data case in the pipeline. No median fill, no
    column drops, no fabrication; only row drops.

    All matrices must have the same row count; the same rows are dropped
    from each so they stay aligned.

    Returns:
        (cleaned_matrices, drop_counts) where drop_counts maps column name
        to count of rows that had NaN in that column.
    """
    if not matrices:
        return [], {}
    n = len(matrices[0])
    for m in matrices:
        if len(m) != n:
            raise ValueError(
                f"drop_nan_rows: matrices must have same row count, "
                f"got {[len(x) for x in matrices]}"
            )

    drop_counts: dict[str, int] = {}
    mask = pd.Series([False] * n)
    for m in matrices:
        m_reset = m.reset_index(drop=True)
        for col in m_reset.columns:
            col_nan = m_reset[col].isna()
            n_col = int(col_nan.sum())
            if n_col > 0:
                drop_counts[col] = n_col
                mask = mask | col_nan

    keep = ~mask
    cleaned = [m.reset_index(drop=True)[keep.values].reset_index(drop=True) for m in matrices]
    return cleaned, drop_counts


def require_min_sample(n: int, min_required: int, *, context: str = "") -> None:
    """Halt with :class:`InsufficientSampleError` if ``n < min_required``."""
    if n < min_required:
        suffix = f" for {context}" if context else ""
        raise InsufficientSampleError(
            f"Insufficient sample{suffix} after NaN drops "
            f"({n} < {min_required}). Fit cannot proceed."
        )


def reparameterize_kpa(features: pd.DataFrame) -> ReparameterizedDesign:
    """Return a K|PA design matrix that's well-conditioned by construction.

    Replaces collinear pairs with their centered versions:

    - pitcher_k_pct_season_shrunk  vs  league_k_pct_vs_hand
        -> pitcher_k_pct_delta (centered),  league_k_pct_vs_hand (baseline)
    - pitcher_k_pct_30d_blended    vs  pitcher_k_pct_season_shrunk
        -> pitcher_k_pct_30d_delta (centered on the pitcher's own season)

    Other features stay as-is but log-transformed when appropriate
    (multiplicative factors like park_k_factor, umpire_k_factor become
    log-space additive terms).
    """
    out = pd.DataFrame(index=features.index)

    # League baseline must be present.
    if "league_k_pct_vs_hand" not in features.columns:
        raise DesignMatrixError(
            "reparameterize_kpa: league_k_pct_vs_hand column missing"
        )

    out["league_k_pct_vs_hand"] = features["league_k_pct_vs_hand"]

    if "pitcher_k_pct_season_shrunk" in features.columns:
        out["pitcher_k_pct_delta"] = (
            features["pitcher_k_pct_season_shrunk"]
            - features["league_k_pct_vs_hand"]
        )

    if "pitcher_k_pct_30d_blended" in features.columns:
        if "pitcher_k_pct_season_shrunk" in features.columns:
            out["pitcher_k_pct_30d_delta"] = (
                features["pitcher_k_pct_30d_blended"]
                - features["pitcher_k_pct_season_shrunk"]
            )
        else:
            out["pitcher_k_pct_30d_delta"] = (
                features["pitcher_k_pct_30d_blended"]
                - features["league_k_pct_vs_hand"]
            )

    # CSW / chase-whiff / putaway: centered on league anchors. The Phase 3c
    # constants live in features_kpa.py; we hard-code the same values here.
    if "pitcher_csw_pct_30d" in features.columns:
        out["pitcher_csw_pct_30d_delta"] = features["pitcher_csw_pct_30d"] - 0.28
    if "pitcher_chase_whiff_pct_30d" in features.columns:
        out["pitcher_chase_whiff_pct_30d_delta"] = (
            features["pitcher_chase_whiff_pct_30d"] - 0.22
        )
    if "pitcher_putaway_pitch_concentration" in features.columns:
        out["pitcher_putaway_pct_delta"] = (
            features["pitcher_putaway_pitch_concentration"] - 0.40
        )

    # Velocity trend is already a Z-score, centered at 0.
    if "pitcher_velocity_trend_3starts" in features.columns:
        out["pitcher_velocity_trend_z"] = features["pitcher_velocity_trend_3starts"]

    # Lineup features: center on the league anchor passed in via the same
    # league_k_pct_vs_hand column where appropriate.
    if "lineup_k_pct_vs_hand" in features.columns:
        out["lineup_k_pct_delta"] = (
            features["lineup_k_pct_vs_hand"] - features["league_k_pct_vs_hand"]
        )
    # Zone contact and chase need their own league anchors. The features_kpa
    # module derives these per-pitcher-hand; for the design matrix we'll use
    # the row-level builder output minus its anchor (passed in adjacent cols).
    if "lineup_zone_contact_pct" in features.columns and "league_zone_contact_anchor" in features.columns:
        out["lineup_zone_contact_delta"] = (
            features["lineup_zone_contact_pct"] - features["league_zone_contact_anchor"]
        )
    if "lineup_chase_rate" in features.columns and "league_chase_anchor" in features.columns:
        out["lineup_chase_delta"] = (
            features["lineup_chase_rate"] - features["league_chase_anchor"]
        )

    # Multiplicative factors in log space (centered at 0 for factor=1.0).
    if "park_k_factor" in features.columns:
        out["log_park_k_factor"] = np.log(features["park_k_factor"].clip(lower=1e-6))
    if "umpire_k_zone_factor" in features.columns:
        out["log_umpire_k_factor"] = np.log(features["umpire_k_zone_factor"].clip(lower=1e-6))

    return _drop_zero_variance(out, context="reparameterize_kpa")


def reparameterize_bf(features: pd.DataFrame) -> ReparameterizedDesign:
    """Return a well-conditioned E[BF] design matrix.

    The original spec sign-flipped on pitcher_pa_per_start_season. With the
    full feature set, the analogous fix is to express the 30d form as a
    delta from the pitcher's own season (form vs baseline), so the columns
    don't redundantly express the same level.
    """
    out = pd.DataFrame(index=features.index)

    if "pitcher_ip_per_start_30d_shrunk" in features.columns:
        out["pitcher_ip_per_start_30d_shrunk"] = features["pitcher_ip_per_start_30d_shrunk"]

    if "pitcher_pitches_per_pa_season" in features.columns:
        out["pitcher_pitches_per_pa_season"] = features["pitcher_pitches_per_pa_season"]
        if "pitcher_pitches_per_pa_30d" in features.columns:
            out["pitcher_pitches_per_pa_30d_delta"] = (
                features["pitcher_pitches_per_pa_30d"]
                - features["pitcher_pitches_per_pa_season"]
            )

    if "lineup_obp_vs_hand" in features.columns:
        if "league_obp_anchor" in features.columns:
            out["lineup_obp_delta"] = (
                features["lineup_obp_vs_hand"] - features["league_obp_anchor"]
            )
        else:
            out["lineup_obp_vs_hand"] = features["lineup_obp_vs_hand"]

    if "park_run_environment_factor" in features.columns:
        out["log_park_run_factor"] = np.log(
            features["park_run_environment_factor"].clip(lower=1e-6)
        )

    # park_k_factor is meaningful for BF too (K-friendly parks -> faster
    # innings -> fewer BF). Include if present.
    if "park_k_factor" in features.columns:
        out["log_park_k_factor"] = np.log(features["park_k_factor"].clip(lower=1e-6))

    return _drop_zero_variance(out, context="reparameterize_bf")


def check_design_matrix(
    X: pd.DataFrame,
    *,
    name: str = "design_matrix",
    condition_threshold: float = DEFAULT_CONDITION_THRESHOLD,
    delta_columns: Iterable[str] | None = None,
    delta_group_cols: Iterable[str] | None = None,
    delta_mean_tolerance: float = DEFAULT_DELTA_MEAN_TOLERANCE,
) -> dict:
    """Pre-fit sanity check on a design matrix.

    Halts (via :class:`DesignMatrixError`) on:
    - any NaN
    - any zero-variance column
    - condition number > ``condition_threshold``
    - any ``delta_columns`` whose mean within any (``delta_group_cols``)
      cell is more than ``delta_mean_tolerance`` away from zero

    Returns a diagnostic dict for logging.
    """
    if X.empty:
        raise DesignMatrixError(f"{name}: empty design matrix")

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        raise DesignMatrixError(f"{name}: NaN in columns {nan_cols}")

    stds = X.std()
    zero_var = stds[stds < 1e-9].index.tolist()
    if zero_var:
        raise DesignMatrixError(
            f"{name}: zero-variance columns {zero_var} — drop or fix the feature"
        )

    # Condition number on the standardized design matrix (avoids
    # interpretation pitfalls when columns have different scales).
    X_std = (X - X.mean()) / X.std()
    cond = float(np.linalg.cond(X_std.to_numpy()))
    if cond > condition_threshold:
        raise DesignMatrixError(
            f"{name}: condition number {cond:.1f} > {condition_threshold} "
            f"— likely collinear columns"
        )

    if delta_columns:
        for col in delta_columns:
            if col not in X.columns:
                continue
            if delta_group_cols:
                means = X.groupby(list(delta_group_cols))[col].mean()
                bad = means[means.abs() > delta_mean_tolerance]
                if not bad.empty:
                    raise DesignMatrixError(
                        f"{name}: delta column {col!r} not centered within "
                        f"groups ({len(bad)} cells exceed |mean| > "
                        f"{delta_mean_tolerance}): first offender: "
                        f"{bad.head(1).to_dict()}"
                    )
            else:
                m = float(X[col].mean())
                if abs(m) > delta_mean_tolerance:
                    raise DesignMatrixError(
                        f"{name}: delta column {col!r} not centered "
                        f"(global mean={m:.4f}, tolerance={delta_mean_tolerance})"
                    )

    return {
        "n_rows": int(len(X)),
        "n_columns": int(X.shape[1]),
        "columns": list(X.columns),
        "condition_number": round(cond, 2),
        "min_std": round(float(stds.min()), 6),
        "max_std": round(float(stds.max()), 6),
    }
