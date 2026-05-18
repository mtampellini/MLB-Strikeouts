"""Phase 4c (sample fit): fit feature coefficients on a stratified 1000-game
sample from 2024-2025.

This session's sample fit makes several deliberate simplifications versus
the spec to fit in the available time:

- Pitcher's "season K%" is the FULL-SEASON aggregate (with a small leakage:
  game G's K count contributes ~1/30 to the pitcher's denominator). Strict
  AsOfContext (cumulative-up-to-G-1) is deferred to the overnight rerun.
- Lineup K% is the leaguewide-vs-hand average for the season, not the
  opposing team's per-game lineup. Strict per-batter aggregation is
  deferred to the overnight rerun.
- 30-day rolling features (ip_per_start_30d, pitches_per_pa_30d, etc.) are
  NOT in the sample fit. They require per-pitcher rolling windows which
  are O(N²) in the naive pandas loop; vectorizing them and including them
  in the design matrix is the overnight rerun's job.
- Pitch-level features (CSW%, chase-whiff%, velocity trend, putaway,
  zone-contact, chase-rate) require pitch-level (not PA-terminal) data
  and full per-pitcher slicing. Deferred.
- Weather and umpire features require boxscore lookups not in this script.
  Deferred.

The features WE DO fit are the dominant K-rate signals: pitcher's
own K%, park K factor, league-vs-hand baseline. This gives the model
a real calibration of its biggest knobs.

The output JSON marks ``sample_fit: true`` and lists every gap as
``missing_features``. The projector logs a "SAMPLE FIT — NOT PRODUCTION"
warning when it loads coefficients with that flag set.

CLI:
    python -m scripts.fit_feature_coefficients --sample 1000
    python -m scripts.fit_feature_coefficients --full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
STARTER_MIN_PA = 12


# These are the features we fit in THIS sample. Others remain at Phase 3
# placeholder coefficients in features_bf.py / features_kpa.py.
BF_FEATURES = (
    "pitcher_pa_per_start_season",  # proxy for ip_per_start
    "park_k_factor",                # park run-environment proxy
)
KPA_FEATURES = (
    "pitcher_k_pct_season",
    "league_k_pct_vs_hand",
    "park_k_factor",
)


def _load_seasons(seasons: list[int]) -> pd.DataFrame:
    from pybaseball import statcast  # type: ignore

    frames = []
    for season in seasons:
        logger.info("loading season %d", season)
        df = statcast(start_dt=f"{season}-03-15", end_dt=f"{season}-11-30")
        df = df[df["game_type"] == "R"].copy()
        df["__season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _aggregate(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["__is_k"] = pa["events"].isin(K_EVENTS)
    pg = pa.groupby(
        ["__season", "game_pk", "pitcher", "home_team", "p_throws"], dropna=True
    ).agg(
        observed_pa=("__is_k", "count"),
        observed_k=("__is_k", "sum"),
    ).reset_index()
    pg = pg[pg["observed_pa"] >= STARTER_MIN_PA]
    return pg


def _attach_features(
    pg: pd.DataFrame,
    pa: pd.DataFrame,
    park_factors: dict[int, float],
) -> pd.DataFrame:
    """Vectorized feature attachment using full-season aggregates."""
    pg = pg.copy()

    # Pitcher full-season K%, PA/start (full-season aggregate per pitcher-season)
    pa["__is_k"] = pa["events"].isin(K_EVENTS)
    pitcher_season = (
        pa.groupby(["__season", "pitcher"])
        .agg(
            season_k=("__is_k", "sum"),
            season_pa=("__is_k", "count"),
        )
        .reset_index()
    )
    pitcher_season["pitcher_k_pct_season"] = (
        pitcher_season["season_k"] / pitcher_season["season_pa"]
    )
    # PA per start = season_pa / number of starts that pitcher had this season
    pitcher_starts = (
        pg.groupby(["__season", "pitcher"]).size().reset_index(name="n_starts")
    )
    pitcher_season = pitcher_season.merge(pitcher_starts, on=["__season", "pitcher"])
    pitcher_season["pitcher_pa_per_start_season"] = (
        pitcher_season["season_pa"] / pitcher_season["n_starts"]
    )
    pg = pg.merge(
        pitcher_season[[
            "__season", "pitcher",
            "pitcher_k_pct_season", "pitcher_pa_per_start_season",
        ]],
        on=["__season", "pitcher"],
        how="inner",
    )

    # Leaguewide K% vs pitcher's hand for the season
    league_by_hand = (
        pa.groupby(["__season", "p_throws"])
        .agg(league_k=("__is_k", "sum"), league_pa=("__is_k", "count"))
        .reset_index()
    )
    league_by_hand["league_k_pct_vs_hand"] = (
        league_by_hand["league_k"] / league_by_hand["league_pa"]
    )
    pg = pg.merge(
        league_by_hand[["__season", "p_throws", "league_k_pct_vs_hand"]],
        on=["__season", "p_throws"],
        how="left",
    )

    # Park K factor
    from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID
    pg["venue_id"] = pg["home_team"].map(TEAM_TO_VENUE_ID)
    pg["park_k_factor"] = pg["venue_id"].map(park_factors).fillna(1.0)

    # Targets
    pg["observed_bf"] = pg["observed_pa"].astype(float)
    pg["observed_k_rate"] = pg["observed_k"] / pg["observed_pa"]

    return pg


def _stratify(pg: pd.DataFrame, park_factors: dict[int, float]) -> pd.DataFrame:
    pg = pg.copy()
    pg["pitcher_quartile"] = pg.groupby("__season")["pitcher_k_pct_season"].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates="drop")
    )

    def park_type(vid):
        f = park_factors.get(vid, 1.0)
        if f < 0.97:
            return "K_suppressor"
        if f > 1.03:
            return "K_friendly"
        return "Neutral"

    pg["park_type"] = pg["venue_id"].map(park_type).fillna("Neutral")
    pg["stratum"] = (
        pg["__season"].astype(str) + "_"
        + pg["pitcher_quartile"].astype(str) + "_"
        + pg["park_type"]
    )
    return pg


def _stratified_sample(pg: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    strata = pg["stratum"].unique()
    per_stratum = max(1, n // len(strata))
    rng = np.random.default_rng(seed)
    sampled = []
    for s in strata:
        sub = pg[pg["stratum"] == s]
        take = min(len(sub), per_stratum)
        sampled.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(sampled, ignore_index=True)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    yhat = Xb @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return beta[1:], float(beta[0]), r2


def _fit_weighted_logit_ols(
    X: np.ndarray, y_rate: np.ndarray, weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    eps = 1e-6
    y_clipped = np.clip(y_rate, eps, 1 - eps)
    y_logit = np.log(y_clipped / (1 - y_clipped))
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    W = np.diag(weights)
    XtWX = Xb.T @ W @ Xb
    XtWy = Xb.T @ W @ y_logit
    beta = np.linalg.solve(XtWX, XtWy)
    yhat = Xb @ beta
    w_mean_y = float((weights * y_logit).sum() / weights.sum())
    ss_res = float((weights * (y_logit - yhat) ** 2).sum())
    ss_tot = float((weights * (y_logit - w_mean_y) ** 2).sum())
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-9))
    return beta[1:], float(beta[0]), r2


def _leakage_shuffle_test(
    train: pd.DataFrame, test: pd.DataFrame,
    feature_cols: list[str], target_col: str,
    weight_col: str | None = None, logit: bool = False,
    seed: int = 1234,
) -> tuple[float, float]:
    """Shuffle 2024/2025 labels and refit. Returns (real_r2_out, shuffled_r2_out).

    If the shuffled R² is comparable to the real one, our temporal structure
    isn't carrying signal — that's a leakage smell.
    """
    rng = np.random.default_rng(seed)
    all_games = pd.concat([train, test], ignore_index=True)
    perm = rng.permutation(len(all_games))
    n_train = len(train)
    shuffled_train = all_games.iloc[perm[:n_train]]
    shuffled_test = all_games.iloc[perm[n_train:]]

    def fit_and_r2(tr, te):
        Xtr = tr[feature_cols].to_numpy(dtype=float)
        ytr = tr[target_col].to_numpy(dtype=float)
        Xte = te[feature_cols].to_numpy(dtype=float)
        yte = te[target_col].to_numpy(dtype=float)
        if logit:
            wtr = tr[weight_col].to_numpy(dtype=float)
            coefs, intercept, _ = _fit_weighted_logit_ols(Xtr, ytr, wtr)
            eps = 1e-6
            y_logit = np.log(np.clip(yte, eps, 1 - eps) / (1 - np.clip(yte, eps, 1 - eps)))
            yhat = intercept + Xte @ coefs
            wte = te[weight_col].to_numpy(dtype=float)
            ss_res = float((wte * (y_logit - yhat) ** 2).sum())
            mean = float((wte * y_logit).sum() / wte.sum())
            ss_tot = float((wte * (y_logit - mean) ** 2).sum())
            return 1.0 - (ss_res / max(ss_tot, 1e-9))
        else:
            coefs, intercept, _ = _fit_ols(Xtr, ytr)
            yhat = intercept + Xte @ coefs
            ss_res = float(((yte - yhat) ** 2).sum())
            ss_tot = float(((yte - yte.mean()) ** 2).sum())
            return 1.0 - (ss_res / max(ss_tot, 1e-9))

    real = fit_and_r2(train, test)
    shuf = fit_and_r2(shuffled_train, shuffled_test)
    return real, shuf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    park_factors_blob = json.loads(
        (PROCESSED_DIR / "park_k_factors.json").read_text(encoding="utf-8")
    )
    park_factors = {
        int(k): float(v["factor"]) for k, v in park_factors_blob["factors"].items()
    }

    pa = _load_seasons(args.seasons)
    pa = pa.dropna(subset=["events", "stand", "p_throws", "pitcher"]).copy()
    logger.info("loaded %d PAs", len(pa))

    pg = _aggregate(pa)
    logger.info("starter-games: %d", len(pg))

    pg = _attach_features(pg, pa, park_factors)
    pg = _stratify(pg, park_factors)

    fit_pool = pg[pg["__season"].isin(args.seasons)].copy()
    all_features = sorted(set(BF_FEATURES) | set(KPA_FEATURES))
    fit_pool = fit_pool.dropna(
        subset=all_features + ["observed_bf", "observed_k_rate"]
    )
    logger.info("fit pool: %d games", len(fit_pool))

    if args.full:
        sample = fit_pool
    else:
        sample = _stratified_sample(fit_pool, n=args.sample)
    logger.info(
        "sample: %d games (%d strata)",
        len(sample), sample["stratum"].nunique(),
    )

    train = sample[sample["__season"] == 2024]
    test = sample[sample["__season"] == 2025]
    logger.info(
        "walk-forward: train=%d (2024) | test=%d (2025)",
        len(train), len(test),
    )

    # ---- E[BF] fit ---------------------------------------------------------
    bf_coefs, bf_intercept, bf_r2_in = _fit_ols(
        train[list(BF_FEATURES)].to_numpy(dtype=float),
        train["observed_bf"].to_numpy(dtype=float),
    )
    yhat_test_bf = (
        bf_intercept
        + test[list(BF_FEATURES)].to_numpy(dtype=float) @ bf_coefs
    )
    yte_bf = test["observed_bf"].to_numpy(dtype=float)
    bf_r2_out = 1.0 - (
        ((yte_bf - yhat_test_bf) ** 2).sum()
        / max(((yte_bf - yte_bf.mean()) ** 2).sum(), 1e-9)
    )

    # ---- P(K|PA) fit -------------------------------------------------------
    kpa_coefs, kpa_intercept, kpa_r2_in = _fit_weighted_logit_ols(
        train[list(KPA_FEATURES)].to_numpy(dtype=float),
        train["observed_k_rate"].to_numpy(dtype=float),
        train["observed_pa"].to_numpy(dtype=float),
    )
    Xte = test[list(KPA_FEATURES)].to_numpy(dtype=float)
    yte = test["observed_k_rate"].to_numpy(dtype=float)
    wte = test["observed_pa"].to_numpy(dtype=float)
    eps = 1e-6
    y_logit = np.log(np.clip(yte, eps, 1 - eps) / (1 - np.clip(yte, eps, 1 - eps)))
    yhat = kpa_intercept + Xte @ kpa_coefs
    ss_res = float((wte * (y_logit - yhat) ** 2).sum())
    mean = float((wte * y_logit).sum() / wte.sum())
    ss_tot = float((wte * (y_logit - mean) ** 2).sum())
    kpa_r2_out = 1.0 - (ss_res / max(ss_tot, 1e-9))

    # ---- Leakage shuffle test ---------------------------------------------
    bf_real, bf_shuffled = _leakage_shuffle_test(
        train, test, list(BF_FEATURES), "observed_bf",
    )
    kpa_real, kpa_shuffled = _leakage_shuffle_test(
        train, test, list(KPA_FEATURES), "observed_k_rate",
        weight_col="observed_pa", logit=True,
    )

    # Output coefficient files
    bf_payload = {
        "sample_fit": not args.full,
        "n_games_train": int(len(train)),
        "n_games_test": int(len(test)),
        "in_sample_r2": round(bf_r2_in, 4),
        "out_of_sample_r2": round(bf_r2_out, 4),
        "intercept": round(bf_intercept, 4),
        "coefficients": {
            f: round(float(c), 4) for f, c in zip(BF_FEATURES, bf_coefs)
        },
        "leakage_shuffle": {
            "real_out_of_sample_r2": round(bf_real, 4),
            "shuffled_out_of_sample_r2": round(bf_shuffled, 4),
            "drop_under_shuffle": round(bf_real - bf_shuffled, 4),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_features": [
            "pitcher_ip_per_start_30d_shrunk - 30d rolling deferred to overnight",
            "pitcher_pitches_per_pa_season - requires pitch-level aggregation",
            "pitcher_pitches_per_pa_30d - same",
            "lineup_obp_vs_hand - per-batter aggregation deferred",
            "weather_run_environment - needs boxscore weather lookups",
            "days_rest_bucket - needs per-pitcher game-date chains",
            "team_bullpen_short_hook_indicator - data source pending",
        ],
    }
    kpa_payload = {
        "sample_fit": not args.full,
        "n_games_train": int(len(train)),
        "n_games_test": int(len(test)),
        "in_sample_r2_logit": round(kpa_r2_in, 4),
        "out_of_sample_r2_logit": round(kpa_r2_out, 4),
        "intercept_logit": round(kpa_intercept, 4),
        "coefficients_logit": {
            f: round(float(c), 4) for f, c in zip(KPA_FEATURES, kpa_coefs)
        },
        "leakage_shuffle": {
            "real_out_of_sample_r2_logit": round(kpa_real, 4),
            "shuffled_out_of_sample_r2_logit": round(kpa_shuffled, 4),
            "drop_under_shuffle": round(kpa_real - kpa_shuffled, 4),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_features": [
            "pitcher_k_pct_30d_blended - 30d rolling deferred to overnight",
            "pitcher_csw_pct_30d - needs pitch-level data, deferred",
            "pitcher_chase_whiff_pct_30d - same",
            "pitcher_velocity_trend_3starts - same",
            "pitcher_putaway_pitch_concentration - same",
            "lineup_k_pct_vs_hand - per-batter aggregation deferred",
            "lineup_zone_contact_pct - pitch-level, deferred",
            "lineup_chase_rate - same",
            "umpire_k_zone_factor - needs boxscore lookups",
        ],
    }

    (PROCESSED_DIR / "feature_coefficients_bf.json").write_text(
        json.dumps(bf_payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    (PROCESSED_DIR / "feature_coefficients_kpa.json").write_text(
        json.dumps(kpa_payload, indent=2, sort_keys=True), encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info(
        "E[BF] fit: in-sample R^2=%.4f, out-of-sample R^2=%.4f",
        bf_r2_in, bf_r2_out,
    )
    logger.info("  intercept: %+.4f", bf_intercept)
    for f, c in zip(BF_FEATURES, bf_coefs):
        logger.info("  %-40s  %+.4f", f, c)
    logger.info(
        "  leakage shuffle: real=%.4f, shuffled=%.4f, drop=%.4f",
        bf_real, bf_shuffled, bf_real - bf_shuffled,
    )

    logger.info(
        "P(K|PA) fit (logit): in-sample R^2=%.4f, out-of-sample R^2=%.4f",
        kpa_r2_in, kpa_r2_out,
    )
    logger.info("  intercept: %+.4f", kpa_intercept)
    for f, c in zip(KPA_FEATURES, kpa_coefs):
        logger.info("  %-40s  %+.4f", f, c)
    logger.info(
        "  leakage shuffle: real=%.4f, shuffled=%.4f, drop=%.4f",
        kpa_real, kpa_shuffled, kpa_real - kpa_shuffled,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
