"""Phase 4c-v2 Step 1/6: Derive the empirical CSW% -> K% relationship.

CSW% is a well-established K% predictor in pitching analytics — pitchers
who generate more called-or-swinging strikes per pitch generate more
strikeouts per PA. Step 2 will use this linear function to compute a
CSW-implied K rate that gets blended with observed K% to produce the
effective K rate that flows into the log5 offset.

Gates are STRUCTURAL only (sample size, R^2, slope range, observed CSW
range coverage). The intercept is geometric extrapolation to CSW=0 — no
real pitcher operates there, so the intercept value is not a
methodologically meaningful metric and does NOT halt the derivation.
This follows the project-wide rule: empirical-magnitude priors emit WARN,
not halt; structural priors emit halt.

Output: backend/data/processed/csw_to_k_relationship.json

CLI:
    python -m scripts.derive_csw_to_k_relationship
    python -m scripts.derive_csw_to_k_relationship --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "csw_to_k_relationship.json"
DIAGNOSTIC_FILE = PROCESSED_DIR / "csw_to_k_diagnostic.json"

# Event taxonomy
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
NON_PA_EVENTS = frozenset({
    "truncated_pa", "caught_stealing_2b", "caught_stealing_3b",
    "caught_stealing_home", "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home", "stolen_base_2b", "stolen_base_3b",
    "stolen_base_home", "wild_pitch", "passed_ball", "balk",
})
CSW_DESCRIPTIONS = frozenset({
    "called_strike", "swinging_strike", "swinging_strike_blocked",
})

# Filters
MIN_PITCHES = 500
MIN_PA = 50

# Sanity gates. Structural only — see module docstring on why the intercept
# gate is intentionally absent.
GATE_R2_MIN = 0.50
GATE_SLOPE_MIN = 1.0
GATE_SLOPE_MAX = 3.0
GATE_MIN_N_PITCHER_SEASONS = 300
GATE_CSW_RANGE_MIN = 0.22
GATE_CSW_RANGE_MAX = 0.34

# Manual spot-check pitchers (2024)
SPOT_CHECK = {
    694973: ("Paul Skenes", (0.32, 0.38)),
    664285: ("Framber Valdez", (0.21, 0.26)),
    # Patrick Corbin - reliably low K% workhorse, good "low-K control"
    571578: ("Patrick Corbin", (0.14, 0.19)),
}


# ---- Load ------------------------------------------------------------------


def _load_seasons(seasons: list[int]) -> pd.DataFrame:
    from pybaseball import statcast
    frames = []
    for season in seasons:
        logger.info("loading season %d", season)
        df = statcast(start_dt=f"{season}-03-15", end_dt=f"{season}-11-30")
        df = df[df["game_type"] == "R"].copy()
        df["__season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- Aggregation -----------------------------------------------------------


def _aggregate_pitcher_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """For each (pitcher, season), compute n_pitches, n_pa, n_csw, n_k.

    Returns DataFrame with one row per (pitcher, season) meeting the
    minimum-sample filter.
    """
    df = df.copy()
    df["is_csw"] = df["description"].isin(CSW_DESCRIPTIONS)
    # PA-terminal rows only for K% / n_PA
    df["is_pa_terminal"] = df["events"].notna() & ~df["events"].isin(NON_PA_EVENTS)
    df["is_k"] = df["events"].isin(K_EVENTS)
    agg = df.groupby(["pitcher", "__season"]).agg(
        n_pitches=("is_csw", "size"),
        n_csw=("is_csw", "sum"),
        n_pa=("is_pa_terminal", "sum"),
        n_k=("is_k", "sum"),
        name=("player_name", "first"),
    ).reset_index()
    # Filter to minimum sample
    agg = agg[(agg["n_pitches"] >= MIN_PITCHES) & (agg["n_pa"] >= MIN_PA)].copy()
    agg["csw_pct"] = agg["n_csw"] / agg["n_pitches"]
    agg["k_pct"] = agg["n_k"] / agg["n_pa"]
    return agg


# ---- Weighted linear regression --------------------------------------------


def fit_weighted_linear(
    x: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> dict:
    """Closed-form weighted least squares for y = a + b*x.

    Returns dict with intercept, slope, r_squared, rmse, fitted_values, residuals.
    """
    w_sum = float(w.sum())
    x_bar = float((w * x).sum() / w_sum)
    y_bar = float((w * y).sum() / w_sum)
    s_xx = float((w * (x - x_bar) ** 2).sum())
    s_xy = float((w * (x - x_bar) * (y - y_bar)).sum())
    slope = s_xy / s_xx if s_xx > 0 else 0.0
    intercept = y_bar - slope * x_bar
    yhat = intercept + slope * x
    residuals = y - yhat
    ss_res = float((w * residuals ** 2).sum())
    ss_tot = float((w * (y - y_bar) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / w_sum) if w_sum > 0 else 0.0
    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared": r2,
        "rmse": rmse,
        "fitted_values": yhat,
        "residuals": residuals,
    }


def csw_implied_k_rate(csw_pct: float, intercept: float, slope: float) -> float:
    """The exported function. Step 2 calls this to compute CSW-implied K rate."""
    return intercept + slope * csw_pct


# ---- Sanity checks ---------------------------------------------------------


def _sanity_check(payload: dict, csw_observed_range: tuple[float, float]) -> None:
    model = payload["model"]
    r2 = model["r_squared"]
    slope = model["slope"]
    intercept = model["intercept"]
    n_seasons = payload["n_pitcher_seasons"]

    if r2 < GATE_R2_MIN:
        raise AssertionError(
            f"R² {r2:.4f} < {GATE_R2_MIN} — CSW% should explain at least half of K% variance"
        )
    if slope < GATE_SLOPE_MIN:
        raise AssertionError(
            f"slope {slope:.4f} < {GATE_SLOPE_MIN} (1pp CSW should map to >1pp K)"
        )
    if slope > GATE_SLOPE_MAX:
        raise AssertionError(
            f"slope {slope:.4f} > {GATE_SLOPE_MAX} (unrealistic upper bound)"
        )
    if n_seasons < GATE_MIN_N_PITCHER_SEASONS:
        raise AssertionError(
            f"n_pitcher_seasons {n_seasons} < {GATE_MIN_N_PITCHER_SEASONS}"
        )
    csw_lo, csw_hi = csw_observed_range
    if csw_lo > GATE_CSW_RANGE_MIN or csw_hi < GATE_CSW_RANGE_MAX:
        raise AssertionError(
            f"CSW observed range [{csw_lo:.4f}, {csw_hi:.4f}] does not cover "
            f"required [{GATE_CSW_RANGE_MIN}, {GATE_CSW_RANGE_MAX}]"
        )
    # Intercept is reported but NOT gated — it is geometric extrapolation to
    # CSW=0 (no real pitcher operates there). Only the observed CSW range
    # matters methodologically.
    logger.info(
        "intercept = %.4f (extrapolation to CSW=0; not methodologically "
        "meaningful, only the [%.4f, %.4f] observed CSW range is evaluated)",
        intercept, csw_lo, csw_hi,
    )
    logger.info("sanity checks pass: R²=%.4f, slope=%.4f, n=%d",
                r2, slope, n_seasons)


def _spot_check(
    pitcher_seasons: pd.DataFrame, intercept: float, slope: float,
) -> list[dict]:
    """Predict K% from CSW% for spot-check pitchers' 2024 numbers.

    Returns a list of dicts with name, observed K%, predicted K%, and a WARN
    flag if predicted is off by >0.04.
    """
    out = []
    for pid, (name, expected_range) in SPOT_CHECK.items():
        sub = pitcher_seasons[
            (pitcher_seasons["pitcher"] == pid)
            & (pitcher_seasons["__season"] == 2024)
        ]
        if sub.empty:
            logger.warning("spot-check: %s (%d) 2024 not in sample", name, pid)
            continue
        row = sub.iloc[0]
        observed_k = float(row["k_pct"])
        predicted_k = csw_implied_k_rate(float(row["csw_pct"]), intercept, slope)
        lo, hi = expected_range
        in_range = lo <= predicted_k <= hi
        off_by = abs(observed_k - predicted_k)
        msg = (
            f"spot-check: {name} (2024) observed K%={observed_k:.4f}, "
            f"predicted={predicted_k:.4f} (CSW%={float(row['csw_pct']):.4f}); "
            f"expected_pred_range [{lo}, {hi}] -> {'IN' if in_range else 'OUT'}; "
            f"|observed - predicted|={off_by:.4f}"
        )
        if not in_range or off_by > 0.04:
            logger.warning("WARN: %s", msg)
        else:
            logger.info(msg)
        out.append({
            "pitcher_id": pid, "name": name,
            "observed_k_pct": round(observed_k, 4),
            "predicted_k_pct": round(predicted_k, 4),
            "expected_predicted_range": list(expected_range),
            "in_range": bool(in_range),
            "obs_pred_delta": round(off_by, 4),
        })
    return out


# ---- Diagnostic ------------------------------------------------------------


def _build_diagnostic(
    pitcher_seasons: pd.DataFrame, fit: dict, spot_checks: list[dict],
) -> dict:
    """Top residuals + scatter for downstream review."""
    df = pitcher_seasons.copy()
    df["fitted_k_pct"] = fit["fitted_values"]
    df["residual"] = fit["residuals"]
    df = df.sort_values("residual")

    def _row_to_dict(r):
        return {
            "pitcher_id": int(r["pitcher"]),
            "name": str(r.get("name") or f"id_{int(r['pitcher'])}"),
            "season": int(r["__season"]),
            "n_pitches": int(r["n_pitches"]),
            "n_pa": int(r["n_pa"]),
            "csw_pct": round(float(r["csw_pct"]), 4),
            "k_pct": round(float(r["k_pct"]), 4),
            "fitted_k_pct": round(float(r["fitted_k_pct"]), 4),
            "residual": round(float(r["residual"]), 4),
        }

    top_neg = df.head(10).apply(_row_to_dict, axis=1).tolist()  # K% LOWER than CSW predicts
    top_pos = df.tail(10).iloc[::-1].apply(_row_to_dict, axis=1).tolist()  # K% HIGHER

    top_pa = df.sort_values("n_pa", ascending=False).head(20).apply(
        _row_to_dict, axis=1
    ).tolist()

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scatter_top_20_by_pa": top_pa,
        "top_negative_residuals_k_below_csw": top_neg,
        "top_positive_residuals_k_above_csw": top_pos,
        "spot_check_predictions": spot_checks,
    }


# ---- Build payload ---------------------------------------------------------


def _build_payload(pitcher_seasons: pd.DataFrame, fit: dict, seasons: list[int]) -> dict:
    csw_min = float(pitcher_seasons["csw_pct"].min())
    csw_max = float(pitcher_seasons["csw_pct"].max())
    k_min = float(pitcher_seasons["k_pct"].min())
    k_max = float(pitcher_seasons["k_pct"].max())
    residuals = fit["residuals"]
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "weighted_linear_regression_csw_to_k",
        "seasons_used": list(seasons),
        "n_pitcher_seasons": int(len(pitcher_seasons)),
        "min_pitches_filter": MIN_PITCHES,
        "min_pa_filter": MIN_PA,
        "model": {
            "intercept": round(float(fit["intercept"]), 6),
            "slope": round(float(fit["slope"]), 6),
            "r_squared": round(float(fit["r_squared"]), 6),
            "rmse": round(float(fit["rmse"]), 6),
        },
        "diagnostic": {
            "n_pitcher_seasons_above_csw_threshold": int(
                (pitcher_seasons["csw_pct"] >= 0.30).sum()
            ),
            "csw_range_observed": [round(csw_min, 4), round(csw_max, 4)],
            "k_range_observed": [round(k_min, 4), round(k_max, 4)],
            "residual_quantiles": {
                "10": round(float(np.quantile(residuals, 0.10)), 4),
                "50": round(float(np.quantile(residuals, 0.50)), 4),
                "90": round(float(np.quantile(residuals, 0.90)), 4),
            },
        },
    }


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = _load_seasons(args.seasons)
    logger.info("loaded %d Statcast pitches across %s", len(df), args.seasons)

    pitcher_seasons = _aggregate_pitcher_seasons(df)
    logger.info(
        "%d pitcher-seasons pass min-pitches=%d / min-PA=%d",
        len(pitcher_seasons), MIN_PITCHES, MIN_PA,
    )

    x = pitcher_seasons["csw_pct"].to_numpy()
    y = pitcher_seasons["k_pct"].to_numpy()
    w = pitcher_seasons["n_pa"].to_numpy().astype(float)
    fit = fit_weighted_linear(x, y, w)

    payload = _build_payload(pitcher_seasons, fit, args.seasons)
    spot_checks = _spot_check(pitcher_seasons, fit["intercept"], fit["slope"])

    csw_range = tuple(payload["diagnostic"]["csw_range_observed"])
    _sanity_check(payload, csw_range)

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUTPUT_FILE)

    diagnostic = _build_diagnostic(pitcher_seasons, fit, spot_checks)
    DIAGNOSTIC_FILE.write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8",
    )
    logger.info("wrote %s", DIAGNOSTIC_FILE)

    logger.info(
        "=== Model: K%% = %.4f + %.4f * CSW%%, R²=%.4f, RMSE=%.4f ===",
        fit["intercept"], fit["slope"], fit["r_squared"], fit["rmse"],
    )
    logger.info("CSW range observed: [%.4f, %.4f]", csw_range[0], csw_range[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
