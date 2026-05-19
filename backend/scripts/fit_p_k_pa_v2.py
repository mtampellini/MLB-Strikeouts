"""Phase 4c-v2: per-PA logistic regression for P(K|PA) with log5 offset.

Structurally different from Phase 4c (failed):
- Target: per-PA strikeout (0/1), Bernoulli
- Offset: logit(log5_prior) - the well-established matchup baseline,
  forced coefficient = 1.0
- Regressors learn RESIDUAL log-odds adjustments only:
  - 5 main pitcher features (CSW delta, K delta, velocity Z, chase-whiff,
    log park K)
  - 19 archetype x TTO interaction dummies (5 x 4 - 1 reference cell)
- Sample: per-PA from 2024-2025, walk-forward (2024 train, 2025 test)

This file is SEPARATE from fit_feature_coefficients.py. That script remains
intact for any future BF refits. P(K|PA) fitting now lives here.

CLI:
    python -m scripts.fit_p_k_pa_v2 --smoke         # 1000 PAs, no fit
    python -m scripts.fit_p_k_pa_v2 --full          # full universe (overnight)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.projection.features_kpa import K_EVENTS, compute_p_k_pa
from src.projection.features_per_batter import per_batter_k_pct_vs_hand
from src.projection.inputs import (
    PADistribution,
    ParkKFactorsByHand,
    PitcherArchetype,
    ProjectionContext,
    TTOMultipliers,
)
from src.projection.projector import (
    ARCHETYPE_UNKNOWN_FALLBACK,
    LEAGUE_K_PCT_HARD_FALLBACK,
    _compute_log5,
    _compute_pitcher_effective_k_rate,
    _effective_batter_hand,
    _league_k_pct_vs_hand,
    _resolve_archetype,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "p_k_pa_coefficients.json"

# Same hard filters as Phase 4c (career IP / season IP / SP-game)
STARTER_MIN_PA = 12
SMOKE_PA_TARGET = 1000

# Archetypes (5) and TTO buckets (1..4). Reference cell: (Balanced, TTO=1).
ARCHETYPES = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced")
TTO_BUCKETS = (1, 2, 3, 4)
REFERENCE_ARCH = "Balanced"
REFERENCE_TTO = 1

# Main pitcher regressor features expected from compute_p_k_pa.used_features.
#
# Phase 4c-v2 Step 5 (CSW-blended log5 offset):
#
# CSW% information now flows through the OFFSET via the effective K rate
# blend (Steps 1-4). The regression therefore has nothing to learn about
# pitcher K-skill itself — that signal lives in the offset. We keep only
# features that are structurally orthogonal to K skill:
#
# - pitcher_velocity_trend_z: form vs baseline, not absolute K level
# - log_park_k_factor_by_hand: contextual / venue effect
#
# Plus the 19 archetype-TTO interaction terms (matchup decay, orthogonal
# to skill level). pitcher_csw_pct_season_delta and
# pitcher_chase_whiff_pct_30d_delta are both K-skill correlated and have
# been removed from the regressor set — they belong inside the offset (CSW
# via the blend) or are now redundant with what the offset captures.
MAIN_REGRESSORS = (
    "pitcher_velocity_trend_z",
    "log_park_k_factor_by_hand",
)

# Sign expectations per spec (all POSITIVE).
EXPECTED_SIGN_POSITIVE = set(MAIN_REGRESSORS)

# Gate thresholds
GATE_MIN_TRAIN_PA = 50_000
GATE_MIN_TEST_PA = 30_000
GATE_MIN_SMOKE_PA = 800   # 1000-sample can lose some rows after NaN drops
GATE_MAX_CONDITION = 100.0
GATE_MAX_Z_SCORE = 50.0
GATE_MIN_PSEUDO_R2 = 0.005
GATE_MIN_RATE_R2 = 0.05
GATE_MIN_BOOTSTRAP_STABILITY = 0.95
GATE_LEAKAGE_DELTA = 0.005

N_BOOTSTRAP = 200


# ---- Statcast load ---------------------------------------------------------


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


# ---- Static lookups --------------------------------------------------------


def _load_static_tables() -> dict:
    """Load archetypes, TTO multipliers, PA distribution, park K factors."""
    archetypes_blob = json.loads(
        (PROCESSED_DIR / "pitcher_archetypes.json").read_text(encoding="utf-8")
    )
    tto_table = TTOMultipliers.from_json(PROCESSED_DIR / "tto_multipliers.json")
    pa_dist = PADistribution.from_json(PROCESSED_DIR / "pa_distribution_by_bf.json")
    park_path = PROCESSED_DIR / "park_k_factors.json"
    return {
        "archetypes_blob": archetypes_blob,
        "tto_table": tto_table,
        "pa_dist": pa_dist,
        "park_path": park_path,
    }


# ---- Per-PA row builder ----------------------------------------------------


def _resolve_archetype_for_pitcher(
    archetypes_blob: dict, pitcher_id: int, season: int,
) -> str:
    """Apply the fallback chain (current_season -> rolling_30d -> prev_season
    -> 'Balanced' sentinel)."""
    pa = PitcherArchetype.from_archetypes_lookup(archetypes_blob, pitcher_id, season)
    if pa.archetype == "unknown":
        return ARCHETYPE_UNKNOWN_FALLBACK
    return pa.archetype


def _assign_tto(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Within each (game_pk, pitcher), assign TTO per PA via cumcount on batter."""
    df = df_pa.sort_values(
        ["game_pk", "pitcher", "inning", "at_bat_number"], kind="mergesort"
    ).copy()
    df["tto"] = df.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1
    df["tto"] = df["tto"].clip(upper=4).astype(int)
    return df


def _is_sp_game(group: pd.DataFrame) -> bool:
    """Same SP filter we used in derive_tto_multipliers + derive_pa_distribution."""
    if group["inning"].min() != 1:
        return False
    if group["batter"].nunique() < 9:
        return False
    OUT_EVENTS = {
        "strikeout", "strikeout_double_play",
        "field_out", "force_out", "fielders_choice_out",
        "grounded_into_double_play", "double_play", "triple_play",
        "sac_fly", "sac_fly_double_play",
        "sac_bunt", "sac_bunt_double_play", "other_out",
    }
    return int(group["events"].isin(OUT_EVENTS).sum()) >= 12


def _filter_sp_pas(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Drop non-starter games."""
    sp_keys = []
    for (game_pk, pitcher), g in df_pa.groupby(["game_pk", "pitcher"], sort=False):
        if _is_sp_game(g):
            sp_keys.append((game_pk, pitcher))
    if not sp_keys:
        return df_pa.iloc[0:0].copy()
    keys = pd.DataFrame(sp_keys, columns=["game_pk", "pitcher"])
    return df_pa.merge(keys, on=["game_pk", "pitcher"], how="inner")


def _build_per_pa_rows(
    df_all: pd.DataFrame,
    statics: dict,
    *,
    smoke_target: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the per-PA design DataFrame.

    ``df_all`` is the FULL pitch-level frame (with non-terminal pitches),
    used to build the pitcher/batter caches that feature builders need
    (CSW, chase-whiff, velocity all consume pitch-level data, not just
    PA-terminal). The per-PA iteration uses the PA-terminal subset.
    """
    from dataclasses import replace
    from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID
    from scripts.fit_feature_coefficients import _build_lineup_for_game
    from scripts.historical_bundle import (
        GameRecord, build_batter_cache, build_bundle, build_pitcher_cache,
    )

    # PA-terminal subset for SP filter + TTO + iteration
    df_pa = df_all[df_all["events"].notna()].copy()
    df_pa = _assign_tto(df_pa)
    df_pa = _filter_sp_pas(df_pa)
    if df_pa.empty:
        logger.warning("no SP PAs found in input")
        return pd.DataFrame()

    if smoke_target is not None:
        # Pick games at random (stratified by season) until we hit ~smoke_target PAs.
        rng = np.random.default_rng(seed)
        game_ids = (
            df_pa.groupby(["__season", "game_pk", "pitcher"]).size().reset_index()
        )
        game_ids = game_ids.rename(columns={0: "n_pa"})
        game_ids = game_ids.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
        game_ids["cum_pa"] = game_ids["n_pa"].cumsum()
        keep = game_ids[game_ids["cum_pa"] <= smoke_target + 50]
        if len(keep) == 0:
            keep = game_ids.iloc[:1]
        df_pa = df_pa.merge(
            keep[["game_pk", "pitcher"]], on=["game_pk", "pitcher"], how="inner",
        )
        logger.info("smoke: kept %d games -> %d PAs", len(keep), len(df_pa))

    logger.info("building pitcher + batter caches (full pitch-level data)")
    pitcher_cache = build_pitcher_cache(df_all)
    batter_cache = build_batter_cache(df_all)

    ctx = ProjectionContext.from_default_paths()
    archetypes_blob = statics["archetypes_blob"]
    tto_table = statics["tto_table"]
    pa_dist = statics["pa_dist"]
    park_path = statics["park_path"]

    rows: list[dict] = []
    n_games_skipped = 0
    n_pas_skipped_per_pa = 0

    # Iterate by (season, game_pk, pitcher) for stable order
    game_groups = df_pa.groupby(["__season", "game_pk", "pitcher"], sort=False)
    n_groups = len(game_groups)
    log_every = max(1, n_groups // 20)

    for i, ((season, game_pk, pitcher_id), g) in enumerate(game_groups):
        if i % log_every == 0:
            logger.info(
                "  per-PA expansion: game %d / %d (%d rows so far)",
                i + 1, n_groups, len(rows),
            )

        first_row = g.iloc[0]
        home_team = str(first_row.get("home_team"))
        away_team = str(first_row.get("away_team"))
        venue_id = TEAM_TO_VENUE_ID.get(home_team)
        if venue_id is None:
            n_games_skipped += 1
            continue
        inning_topbot = str(first_row.get("inning_topbot") or "Top")
        is_home = inning_topbot == "Top"
        pitcher_team = home_team if is_home else away_team
        opposing_team = away_team if is_home else home_team
        pitcher_hand = str(first_row["p_throws"]) if first_row.get("p_throws") in ("L", "R") else None
        if pitcher_hand is None:
            n_games_skipped += 1
            continue
        game_date = pd.to_datetime(str(first_row["game_date"])[:10]).date()

        lineup = _build_lineup_for_game(g, int(pitcher_id))
        if len(lineup) < 9:
            n_games_skipped += 1
            continue

        record = GameRecord(
            season=int(season), game_pk=int(game_pk),
            game_date=game_date, pitcher_id=int(pitcher_id),
            pitcher_hand=pitcher_hand,
            pitcher_team=pitcher_team, opposing_team=opposing_team,
            venue_id=int(venue_id), is_home=is_home,
            opposing_batters=lineup,
            observed_bf=int(len(g)),
            observed_k=int(g["events"].isin(K_EVENTS).sum()),
        )
        bundle = build_bundle(record, pitcher_cache, batter_cache)

        pa_lookup = PitcherArchetype.from_archetypes_lookup(
            archetypes_blob, int(pitcher_id), int(season),
        )
        park_lookup = ParkKFactorsByHand.from_json_lookup(park_path, int(venue_id))
        bundle = replace(
            bundle,
            pitcher_archetype=pa_lookup,
            tto_multipliers=tto_table,
            park_k_factors_by_hand=park_lookup,
            pa_distribution=pa_dist,
        )

        # Game-level features
        kpa = compute_p_k_pa(bundle, ctx)
        if kpa.skipped:
            n_games_skipped += 1
            continue
        feats_game = kpa.used_features
        # Check all main regressors available
        if any(k not in feats_game for k in MAIN_REGRESSORS):
            n_games_skipped += 1
            continue

        # Phase 4c-v2 Step 5: use CSW-blended effective K rate as the
        # pitcher input to log5 (instead of plain observed K%). CSW%
        # information now flows into the OFFSET; the regression won't see
        # CSW% as a competing regressor at all.
        effective_k, _ = _compute_pitcher_effective_k_rate(bundle, ctx)
        if effective_k is None:
            n_games_skipped += 1
            continue
        pitcher_k = effective_k

        # Per-batter K rates
        per_batter_k = per_batter_k_pct_vs_hand(bundle)
        # Sentinel for no league averages
        if 0 in per_batter_k:
            n_games_skipped += 1
            continue

        archetype = _resolve_archetype(bundle)

        # Build a fast lookup: batter_id -> (handedness, batting_order)
        batter_hand_map = {
            b.mlbam_id: b.handedness for b in bundle.opposing_lineup.batters
        }

        # Iterate PAs in this game (already SP-filtered and TTO-tagged)
        for _, pa in g.iterrows():
            batter_id = int(pa["batter"]) if pd.notna(pa["batter"]) else None
            if batter_id is None:
                n_pas_skipped_per_pa += 1
                continue
            batter_hand = batter_hand_map.get(batter_id)
            eff_hand = (
                "L" if batter_hand == "S" and pitcher_hand == "R"
                else "R" if batter_hand == "S" and pitcher_hand == "L"
                else batter_hand
            )
            if eff_hand not in ("L", "R"):
                # Pinch hitter or unmapped — skip
                n_pas_skipped_per_pa += 1
                continue
            league_k = _league_k_pct_vs_hand(ctx, eff_hand, pitcher_hand)
            if league_k is None:
                n_pas_skipped_per_pa += 1
                continue

            entry = per_batter_k.get(batter_id)
            if entry is not None and entry[0] is not None:
                batter_k_val = entry[0]
            else:
                batter_k_val = league_k

            log5 = _compute_log5(batter_k_val, pitcher_k, league_k)
            log5 = max(1e-6, min(1 - 1e-6, log5))
            logit_log5 = math.log(log5 / (1 - log5))

            tto = int(pa["tto"])
            y = 1 if pa.get("events") in K_EVENTS else 0

            row = {
                "season": int(season),
                "game_pk": int(game_pk),
                "pitcher_id": int(pitcher_id),
                "batter_id": batter_id,
                "tto": tto,
                "y": y,
                "logit_log5_prior": logit_log5,
                "archetype": archetype,
                "p_throws": pitcher_hand,
                "home_team": home_team,
            }
            for col in MAIN_REGRESSORS:
                row[col] = float(feats_game[col])
            rows.append(row)

    logger.info(
        "per-PA expansion done: %d rows, %d games skipped, %d PAs skipped per-row",
        len(rows), n_games_skipped, n_pas_skipped_per_pa,
    )
    return pd.DataFrame(rows)


# ---- Archetype-TTO interaction dummies -------------------------------------


def _add_archetype_tto_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add 19 archetype-TTO interaction columns. Reference cell:
    (Balanced, TTO=1) is omitted."""
    out = df.copy()
    cols: list[str] = []
    for arch in ARCHETYPES:
        for tto in TTO_BUCKETS:
            if arch == REFERENCE_ARCH and tto == REFERENCE_TTO:
                continue
            col = f"arch_{arch}_tto_{tto}"
            out[col] = ((out["archetype"] == arch) & (out["tto"] == tto)).astype(int)
            cols.append(col)
    return out, cols


# ---- Fit + diagnostics -----------------------------------------------------


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score the columns; return (X_std, mean, std). Constant columns get std=1."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std_safe, mean, std_safe


def _drop_zero_variance_cols(
    X: np.ndarray, col_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop columns with std < 1e-12 (zero variance — e.g. archetype-TTO
    cells with no observations in this sample). Returns
    (X_trimmed, kept_names, dropped_names).
    """
    std = X.std(axis=0, ddof=0)
    keep_mask = std >= 1e-12
    kept = [c for c, m in zip(col_names, keep_mask) if m]
    dropped = [c for c, m in zip(col_names, keep_mask) if not m]
    return X[:, keep_mask], kept, dropped


def _condition_number(X: np.ndarray) -> float:
    """Condition number on the standardized design matrix (after dropping
    zero-variance columns)."""
    std = X.std(axis=0, ddof=0)
    keep_mask = std >= 1e-12
    if keep_mask.sum() == 0:
        return float("inf")
    X_trim = X[:, keep_mask]
    Xs, _, _ = _standardize(X_trim)
    try:
        svd = np.linalg.svd(Xs, compute_uv=False)
        if svd.min() == 0:
            return float("inf")
        return float(svd.max() / svd.min())
    except np.linalg.LinAlgError:
        return float("inf")


def _fit_logit_with_offset(
    X: np.ndarray, y: np.ndarray, offset: np.ndarray, *, regularize: bool = False,
) -> "object":
    """statsmodels Logit fit with offset. Returns the fit Result object."""
    import statsmodels.api as sm
    X_const = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        if regularize:
            model = sm.Logit(y, X_const, offset=offset)
            return model.fit_regularized(
                method="l1", alpha=0.01, maxiter=400, disp=0,
            )
        model = sm.Logit(y, X_const, offset=offset)
        return model.fit(method="newton", maxiter=200, disp=0)


def _predict_with_offset(
    result, X: np.ndarray, offset: np.ndarray,
) -> np.ndarray:
    import statsmodels.api as sm
    X_const = sm.add_constant(X, has_constant="add")
    return np.asarray(result.predict(exog=X_const, offset=offset))


def _mcfadden_pseudo_r2_out(
    y: np.ndarray, p_hat: np.ndarray, p_null: float,
) -> float:
    """Out-of-sample McFadden's pseudo-R^2 vs constant-mean null."""
    eps = 1e-9
    p_hat = np.clip(p_hat, eps, 1 - eps)
    p_null = max(eps, min(1 - eps, p_null))
    ll_full = float(np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))
    ll_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
    if abs(ll_null) < eps:
        return 0.0
    return 1 - (ll_full / ll_null)


def _rate_space_r2(
    df: pd.DataFrame, p_hat: np.ndarray, weighted: bool = True,
) -> float:
    """Aggregate per-PA predictions to game level, compute K-rate R^2."""
    df = df.copy()
    df["p_hat"] = p_hat
    by_game = df.groupby(["game_pk", "pitcher_id"]).agg(
        observed_k=("y", "sum"), observed_pa=("y", "size"),
        predicted_k=("p_hat", "sum"),
    ).reset_index()
    by_game["observed_rate"] = by_game["observed_k"] / by_game["observed_pa"]
    by_game["predicted_rate"] = by_game["predicted_k"] / by_game["observed_pa"]
    if weighted:
        w = by_game["observed_pa"].to_numpy()
        w_mean = float(np.average(by_game["observed_rate"], weights=w))
        ss_res = float(np.sum(w * (by_game["observed_rate"] - by_game["predicted_rate"]) ** 2))
        ss_tot = float(np.sum(w * (by_game["observed_rate"] - w_mean) ** 2))
    else:
        mean = by_game["observed_rate"].mean()
        ss_res = float(((by_game["observed_rate"] - by_game["predicted_rate"]) ** 2).sum())
        ss_tot = float(((by_game["observed_rate"] - mean) ** 2).sum())
    if ss_tot < 1e-9:
        return 0.0
    return 1 - (ss_res / ss_tot)


def _bootstrap_sign_stability(
    X: np.ndarray, y: np.ndarray, offset: np.ndarray, col_names: list[str],
    *, n_boot: int = N_BOOTSTRAP, seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap-resample, refit, track sign consistency + 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(X)
    coef_samples: dict[str, list[float]] = {c: [] for c in col_names}
    successes = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            res = _fit_logit_with_offset(X[idx], y[idx], offset[idx])
            # statsmodels Logit params: [intercept, *features]
            params = np.asarray(res.params)[1:]
            for i, c in enumerate(col_names):
                coef_samples[c].append(float(params[i]))
            successes += 1
        except Exception:
            continue
    out: dict[str, dict[str, float]] = {}
    for c, samples in coef_samples.items():
        if not samples:
            out[c] = {"stability": 0.0, "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        arr = np.asarray(samples)
        n_pos = int((arr > 0).sum())
        n_neg = int((arr < 0).sum())
        stability = max(n_pos, n_neg) / len(samples)
        out[c] = {
            "stability": round(stability, 4),
            "ci_low": round(float(np.quantile(arr, 0.025)), 6),
            "ci_high": round(float(np.quantile(arr, 0.975)), 6),
            "n_resamples": len(samples),
        }
    out["_meta"] = {"n_successful_resamples": successes, "n_attempted": n_boot}
    return out


def _leakage_shuffle(
    df: pd.DataFrame, X_cols: list[str], offset_col: str, target_col: str,
    *, train_year: int = 2024, test_year: int = 2025,
    seed: int = 42,
) -> tuple[float, float]:
    """Refit on shuffled season labels; out-of-sample R^2 should drop.

    Returns (real_r2_out, shuffled_r2_out).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df_shuffled = df.copy()
    df_shuffled["season"] = rng.permutation(df_shuffled["season"].to_numpy())

    real_r2 = _train_test_rate_r2(df, X_cols, offset_col, target_col,
                                   train_year=train_year, test_year=test_year)
    shuffled_r2 = _train_test_rate_r2(df_shuffled, X_cols, offset_col, target_col,
                                       train_year=train_year, test_year=test_year)
    return real_r2, shuffled_r2


def _train_test_rate_r2(
    df: pd.DataFrame, X_cols: list[str], offset_col: str, target_col: str,
    *, train_year: int, test_year: int,
) -> float:
    train = df[df["season"] == train_year]
    test = df[df["season"] == test_year]
    if train.empty or test.empty:
        return 0.0
    X_train = train[X_cols].to_numpy()
    y_train = train[target_col].to_numpy()
    o_train = train[offset_col].to_numpy()
    X_test = test[X_cols].to_numpy()
    o_test = test[offset_col].to_numpy()
    try:
        res = _fit_logit_with_offset(X_train, y_train, o_train)
    except Exception as exc:
        logger.warning("leakage shuffle: fit failed: %s", exc)
        return 0.0
    p_hat = _predict_with_offset(res, X_test, o_test)
    return _rate_space_r2(test, p_hat, weighted=True)


# ---- Smoke ----------------------------------------------------------------


def _run_smoke() -> dict:
    """Validate the per-PA pipeline on 1000 PAs. Skip the fit."""
    logger.info("=== SMOKE MODE: 1000-PA pipeline validation ===")
    statics = _load_static_tables()
    df_all = _load_seasons([2025])

    df = _build_per_pa_rows(df_all, statics, smoke_target=SMOKE_PA_TARGET)
    if len(df) < GATE_MIN_SMOKE_PA:
        logger.error("smoke: only %d PA rows after expansion (min %d)", len(df), GATE_MIN_SMOKE_PA)
        return {"smoke_pass": False, "n_pa": len(df)}

    df, interaction_cols = _add_archetype_tto_dummies(df)
    X_cols = list(MAIN_REGRESSORS) + interaction_cols
    X = df[X_cols].to_numpy()
    offset = df["logit_log5_prior"].to_numpy()

    # NaN check
    nan_per_col = pd.DataFrame(X, columns=X_cols).isna().sum().to_dict()
    nan_cols = [c for c, n in nan_per_col.items() if n > 0]
    if nan_cols:
        logger.warning("smoke: NaN cols present (will drop rows): %s", nan_cols)
        keep = ~pd.DataFrame(X, columns=X_cols).isna().any(axis=1).to_numpy()
        df = df.iloc[keep].reset_index(drop=True)
        X = X[keep]
        offset = offset[keep]

    # In smoke samples, some archetype-TTO cells may have 0 observations
    # (e.g., 4 Offspeed-heavy games × no TTO=4 in any of them). Drop those
    # zero-variance columns before computing condition number — full mode
    # will have all cells populated.
    X_trim, kept_cols, dropped_cols = _drop_zero_variance_cols(X, X_cols)
    cond = _condition_number(X)
    payload = {
        "smoke_pass": cond < GATE_MAX_CONDITION and len(df) >= GATE_MIN_SMOKE_PA,
        "n_pa": int(len(df)),
        "n_games": int(df["game_pk"].nunique()),
        "design_matrix_shape": [int(X.shape[0]), int(X.shape[1])],
        "design_matrix_shape_post_zero_var_drop": [int(X_trim.shape[0]), int(X_trim.shape[1])],
        "zero_variance_columns_dropped": dropped_cols,
        "X_columns": X_cols,
        "main_regressors": list(MAIN_REGRESSORS),
        "interaction_terms_count": len(interaction_cols),
        "condition_number_standardized": round(cond, 3),
        "offset_min": round(float(offset.min()), 4),
        "offset_max": round(float(offset.max()), 4),
        "offset_mean": round(float(offset.mean()), 4),
        "k_rate_in_smoke_sample": round(float(df["y"].mean()), 4),
    }

    logger.info("=== Smoke results ===")
    for k, v in payload.items():
        if isinstance(v, list) and len(v) > 6:
            logger.info("  %s: (%d items)", k, len(v))
        else:
            logger.info("  %s: %s", k, v)

    if payload["smoke_pass"]:
        logger.info("Smoke run complete: per-PA pipeline validated, fit skipped")
    else:
        logger.error("Smoke run FAILED: see condition number / sample size above")
    return payload


# ---- Full fit -------------------------------------------------------------


def _run_full() -> dict:
    """Full fit on 2024-2025 SP PA universe."""
    logger.info("=== FULL MODE: per-PA fit on 2024-2025 ===")
    statics = _load_static_tables()
    df_all = _load_seasons([2024, 2025])

    df = _build_per_pa_rows(df_all, statics)
    df, interaction_cols = _add_archetype_tto_dummies(df)
    X_cols = list(MAIN_REGRESSORS) + interaction_cols

    # NaN drops at row level
    before = len(df)
    df = df.dropna(subset=X_cols + ["logit_log5_prior", "y"]).reset_index(drop=True)
    logger.info("dropped %d rows with NaN regressors -> %d", before - len(df), len(df))

    train_df = df[df["season"] == 2024]
    test_df = df[df["season"] == 2025]
    logger.info("train PAs: %d, test PAs: %d", len(train_df), len(test_df))

    gates_passed: list[str] = []
    gates_failed: list[str] = []
    warnings_list: list[str] = []

    if len(train_df) < GATE_MIN_TRAIN_PA:
        gates_failed.append(f"n_train {len(train_df)} < {GATE_MIN_TRAIN_PA}")
    else:
        gates_passed.append("n_train >= min")
    if len(test_df) < GATE_MIN_TEST_PA:
        gates_failed.append(f"n_test {len(test_df)} < {GATE_MIN_TEST_PA}")
    else:
        gates_passed.append("n_test >= min")
    if gates_failed:
        return _emit_payload(gates_passed, gates_failed, warnings_list,
                              {"n_train": len(train_df), "n_test": len(test_df)})

    X_train_full = train_df[X_cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=float)
    o_train = train_df["logit_log5_prior"].to_numpy(dtype=float)
    X_test_full = test_df[X_cols].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=float)
    o_test = test_df["logit_log5_prior"].to_numpy(dtype=float)

    # Drop zero-variance columns. In the full universe every archetype-TTO
    # cell should be populated, but defensively guard against an empty cell
    # caused by a filter we didn't anticipate.
    X_train, kept_cols, dropped_cols = _drop_zero_variance_cols(X_train_full, X_cols)
    if dropped_cols:
        warnings_list.append(
            f"dropped zero-variance columns: {dropped_cols}"
        )
        # Apply the same column mask to test
        keep_mask = [c in kept_cols for c in X_cols]
        X_test = X_test_full[:, keep_mask]
    else:
        X_test = X_test_full
    X_cols = kept_cols

    cond = _condition_number(X_train)
    logger.info("condition number (standardized): %.2f", cond)
    if cond >= GATE_MAX_CONDITION:
        gates_failed.append(f"condition_number {cond:.2f} >= {GATE_MAX_CONDITION}")
        return _emit_payload(gates_passed, gates_failed, warnings_list,
                              {"condition_number": cond})
    gates_passed.append("condition_number < 100")

    logger.info("fitting Logit with offset (newton, maxiter=200)")
    try:
        result = _fit_logit_with_offset(X_train, y_train, o_train)
    except Exception as exc:
        gates_failed.append(f"fit did not converge: {exc}")
        return _emit_payload(gates_passed, gates_failed, warnings_list, {})
    gates_passed.append("convergence")

    # Coefficient table (params[0] = intercept, then features)
    params = np.asarray(result.params)
    bse = np.asarray(result.bse)
    coef_intercept = float(params[0])
    coef_feats = params[1:]
    bse_feats = bse[1:]
    z_scores = coef_feats / np.where(bse_feats > 0, bse_feats, 1e-12)

    if np.any(np.abs(z_scores) > GATE_MAX_Z_SCORE):
        offenders = [
            (c, float(z)) for c, z in zip(X_cols, z_scores)
            if abs(z) > GATE_MAX_Z_SCORE
        ]
        gates_failed.append(f"extreme z-scores (collinearity): {offenders}")
    else:
        gates_passed.append("z-scores < 50")

    # Predictions
    p_hat_in = _predict_with_offset(result, X_train, o_train)
    p_hat_out = _predict_with_offset(result, X_test, o_test)

    # Pseudo-R^2
    p_mean_train = float(y_train.mean())
    p_mean_test = float(y_test.mean())
    pseudo_in = 1 - (result.llf / result.llnull)
    pseudo_out = _mcfadden_pseudo_r2_out(y_test, p_hat_out, p_mean_test)
    logger.info("pseudo R^2 in=%.4f, out=%.4f", pseudo_in, pseudo_out)
    if pseudo_out < GATE_MIN_PSEUDO_R2:
        gates_failed.append(f"pseudo_r2_out {pseudo_out:.4f} < {GATE_MIN_PSEUDO_R2}")
    else:
        gates_passed.append("pseudo_r2_out >= min")

    # Rate-space R^2
    r2_in_w = _rate_space_r2(train_df, p_hat_in, weighted=True)
    r2_out_w = _rate_space_r2(test_df, p_hat_out, weighted=True)
    r2_in_u = _rate_space_r2(train_df, p_hat_in, weighted=False)
    r2_out_u = _rate_space_r2(test_df, p_hat_out, weighted=False)
    logger.info("rate-space R^2 (weighted): in=%.4f, out=%.4f", r2_in_w, r2_out_w)
    logger.info("rate-space R^2 (unweighted): in=%.4f, out=%.4f", r2_in_u, r2_out_u)
    if r2_out_w < GATE_MIN_RATE_R2:
        gates_failed.append(f"rate_space_r2_out_weighted {r2_out_w:.4f} < {GATE_MIN_RATE_R2}")
    else:
        gates_passed.append("rate_space_r2_out >= min")

    # Sign expectations on the 5 main pitcher features
    sign_table: dict[str, dict] = {}
    sign_violations = []
    for i, col in enumerate(X_cols):
        sign_table[col] = {
            "value": round(float(coef_feats[i]), 6),
            "std_err": round(float(bse_feats[i]), 6),
            "z_score": round(float(z_scores[i]), 4),
            "p_value": round(float(result.pvalues[i + 1]), 6),
        }
        if col in EXPECTED_SIGN_POSITIVE and coef_feats[i] <= 0:
            sign_violations.append(col)
    if sign_violations:
        gates_failed.append(f"sign expectations violated: {sign_violations}")
    else:
        gates_passed.append("main feature signs match")

    # Bootstrap stability (200 resamples)
    logger.info("bootstrap stability (200 resamples)")
    bootstrap = _bootstrap_sign_stability(X_train, y_train, o_train, X_cols)
    main_unstable: list[tuple[str, float]] = []
    for col in MAIN_REGRESSORS:
        stab = bootstrap.get(col, {}).get("stability", 0.0)
        sign_table[col]["bootstrap_sign_stability"] = stab
        sign_table[col]["bootstrap_ci_low"] = bootstrap[col].get("ci_low")
        sign_table[col]["bootstrap_ci_high"] = bootstrap[col].get("ci_high")
        if stab < GATE_MIN_BOOTSTRAP_STABILITY:
            main_unstable.append((col, stab))
        elif stab < 0.98:
            warnings_list.append(f"bootstrap borderline: {col} stability={stab:.3f}")
    # Carry interaction stability too (informational)
    for col in interaction_cols:
        stab = bootstrap.get(col, {}).get("stability", 0.0)
        sign_table[col]["bootstrap_sign_stability"] = stab
        sign_table[col]["bootstrap_ci_low"] = bootstrap[col].get("ci_low")
        sign_table[col]["bootstrap_ci_high"] = bootstrap[col].get("ci_high")
    if main_unstable:
        gates_failed.append(f"main features unstable under bootstrap: {main_unstable}")
    else:
        gates_passed.append("main feature bootstrap stability >= 95%")

    # Archetype-TTO interaction average for TTO>1
    higher_tto_coefs = [
        coef_feats[i] for i, c in enumerate(X_cols)
        if c.startswith("arch_") and not c.endswith("_tto_1")
    ]
    avg_higher = float(np.mean(higher_tto_coefs)) if higher_tto_coefs else 0.0
    if avg_higher < 0:
        gates_passed.append("archetype-TTO higher-TTO average negative")
    else:
        warnings_list.append(
            f"archetype-TTO higher-TTO average is {avg_higher:.4f} (expected negative)"
        )

    # Leakage shuffle
    logger.info("leakage shuffle diagnostic")
    real_r2, shuf_r2 = _leakage_shuffle(df, X_cols, "logit_log5_prior", "y")
    leakage_delta = real_r2 - shuf_r2
    logger.info("leakage shuffle: real=%.4f, shuffled=%.4f, delta=%.4f",
                real_r2, shuf_r2, leakage_delta)
    if leakage_delta < GATE_LEAKAGE_DELTA:
        gates_failed.append(
            f"leakage shuffle delta {leakage_delta:.4f} < {GATE_LEAKAGE_DELTA}"
        )
    else:
        gates_passed.append("leakage shuffle delta acceptable")

    payload = {
        "model_type": "logistic_regression_per_pa_with_log5_offset",
        "sample_fit": False,
        "n_train_pa": int(len(train_df)),
        "n_test_pa": int(len(test_df)),
        "fit_method": "newton",
        "regularization": "none",
        "condition_number": round(cond, 4),
        "pseudo_r2_mcfadden_in": round(float(pseudo_in), 6),
        "pseudo_r2_mcfadden_out": round(float(pseudo_out), 6),
        "rate_space_r2_weighted_in": round(r2_in_w, 6),
        "rate_space_r2_weighted_out": round(r2_out_w, 6),
        "rate_space_r2_unweighted_in": round(r2_in_u, 6),
        "rate_space_r2_unweighted_out": round(r2_out_u, 6),
        "leakage_shuffle_r2_out": round(shuf_r2, 6),
        "leakage_shuffle_delta": round(leakage_delta, 6),
        "intercept": coef_intercept,
        "coefficients": sign_table,
        "bootstrap_meta": bootstrap.get("_meta", {}),
        "gates_passed": gates_passed,
        "gates_failed": gates_failed,
        "warnings": warnings_list,
    }
    return _emit_payload(gates_passed, gates_failed, warnings_list, payload)


def _emit_payload(
    gates_passed: list[str], gates_failed: list[str],
    warnings_list: list[str], extra: dict,
) -> dict:
    payload = dict(extra)
    payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload.setdefault("gates_passed", gates_passed)
    payload.setdefault("gates_failed", gates_failed)
    payload.setdefault("warnings", warnings_list)
    payload["model_type"] = payload.get("model_type", "logistic_regression_per_pa_with_log5_offset")
    return payload


# ---- CLI ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                         help="1000-PA pipeline validation (no fit)")
    parser.add_argument("--full", action="store_true",
                         help="full 2024-2025 fit (long runtime)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.smoke:
        payload = _run_smoke()
        out_path = PROCESSED_DIR / "p_k_pa_coefficients_smoke.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote %s", out_path)
        return 0 if payload.get("smoke_pass") else 1

    if args.full:
        payload = _run_full()
        OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote %s", OUTPUT_FILE)
        if payload.get("gates_failed"):
            logger.error("FIT FAILED gates: %s", payload["gates_failed"])
            return 1
        logger.info("Fit complete, all gates passed.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
