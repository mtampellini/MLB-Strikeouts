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
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DesignMatrixError(AssertionError):
    """Design matrix violates a pre-fit sanity check."""


# Default condition number threshold. The Phase 4c first attempt had
# implicit condition >> 10000 — the fit was numerically degenerate. A
# threshold of 100 is conservative: well-conditioned regression problems
# typically have condition numbers in the single digits to low hundreds.
DEFAULT_CONDITION_THRESHOLD = 100.0
DEFAULT_DELTA_MEAN_TOLERANCE = 0.01


def reparameterize_kpa(features: pd.DataFrame) -> pd.DataFrame:
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

    return out


def reparameterize_bf(features: pd.DataFrame) -> pd.DataFrame:
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

    return out


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
