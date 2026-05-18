"""Phase 4c (re-parameterized): fit feature coefficients for E[BF] and P(K|PA).

Calls the SAME production feature builders (compute_e_bf, compute_p_k_pa) on
historical-bundle objects constructed from cached Statcast — eliminates the
fit-time/projection-time divergence risk that prompted the first attempt's
scope reduction.

Design matrix is re-parameterized (delta-from-baseline for collinear pairs)
in :mod:`scripts.design_matrix` before fit. Sanity checks
(:func:`scripts.design_matrix.check_design_matrix`) halt the script
BEFORE the OLS solver runs if:

- Any NaN columns
- Any zero-variance columns
- Condition number > 100 (catches the structural collinearity that broke
  the first attempt)
- Any delta column not centered within (season, hand) cells

The first attempt's gate failures (out-of-sample R² = -1.53 on K|PA, sign
flip on pa_per_start, leakage-shuffle inversion) all trace back to the
condition-number violation. Catching that pre-fit prevents the wasted
overnight cycle.

CLI:
    # In-session smoke (~50 games, only validates script runs end-to-end
    # — NOT a real fit; coefficients should be ignored).
    python -m scripts.fit_feature_coefficients --sample 50 --smoke

    # Overnight rerun on user's machine (full universe, real fit):
    python -m scripts.fit_feature_coefficients --full
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
STARTER_MIN_PA = 12


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


def _enumerate_starter_games(pitches: pd.DataFrame) -> pd.DataFrame:
    """Identify (season, game_pk, pitcher_id, ...) tuples for starter games.

    Starter = pitcher who faced >= STARTER_MIN_PA in the game.
    Captures observed BF and K + the metadata needed to build a bundle.
    """
    pa = pitches.dropna(subset=["events", "pitcher"]).copy()
    pa["__is_k"] = pa["events"].isin(K_EVENTS)
    pg = pa.groupby(
        ["__season", "game_pk", "pitcher", "home_team", "p_throws"], dropna=True
    ).agg(
        observed_pa=("__is_k", "count"),
        observed_k=("__is_k", "sum"),
        game_date=("game_date", "first"),
    ).reset_index()
    pg = pg[pg["observed_pa"] >= STARTER_MIN_PA].copy()
    pg["game_date"] = pd.to_datetime(pg["game_date"].astype(str).str[:10]).dt.date
    return pg


def _build_lineup_for_game(
    pa_in_game: pd.DataFrame, pitcher_id: int
) -> dict[int, tuple[str | None, int]]:
    """For one pitcher-game, identify opposing batters + their batting order.

    Order = order of first appearance against the pitcher in this game.
    """
    pa_for_pitcher = pa_in_game[pa_in_game["pitcher"] == pitcher_id].copy()
    pa_for_pitcher = pa_for_pitcher.sort_values(["at_bat_number"])
    lineup: dict[int, tuple[str | None, int]] = {}
    order = 1
    for _, row in pa_for_pitcher.iterrows():
        bid = int(row["batter"]) if pd.notna(row["batter"]) else None
        if bid is None or bid in lineup:
            continue
        hand = row.get("stand")
        if hand not in ("L", "R", "S"):
            hand = None
        lineup[bid] = (hand, order)
        order += 1
        if order > 9:
            break
    return lineup


def _stratified_sample(pg: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample N games stratified by (season, pitcher_k_quartile, park_type).

    park_type comes from Phase 4b factor file; pitcher_k_quartile is computed
    per-season from full-season K%.
    """
    pg = pg.copy()
    season_k = pg.groupby(["__season", "pitcher"]).agg(
        k=("observed_k", "sum"), pa=("observed_pa", "sum"),
    ).reset_index()
    season_k["season_k_pct"] = season_k["k"] / season_k["pa"]
    season_k["pitcher_quartile"] = season_k.groupby("__season")["season_k_pct"].transform(
        lambda x: pd.qcut(x, 4, labels=False, duplicates="drop")
    )
    pg = pg.merge(
        season_k[["__season", "pitcher", "pitcher_quartile"]],
        on=["__season", "pitcher"],
        how="left",
    )

    park_factors_blob = json.loads(
        (PROCESSED_DIR / "park_k_factors.json").read_text(encoding="utf-8")
    )
    park_factors = {
        int(k): float(v["factor"]) for k, v in park_factors_blob["factors"].items()
    }
    from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID

    pg["venue_id"] = pg["home_team"].map(TEAM_TO_VENUE_ID)

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
        + pg["pitcher_quartile"].fillna(-1).astype(int).astype(str) + "_"
        + pg["park_type"]
    )

    strata = pg["stratum"].unique()
    per_stratum = max(1, n // len(strata))
    rng = np.random.default_rng(seed)
    sampled = []
    for s in strata:
        sub = pg[pg["stratum"] == s]
        take = min(len(sub), per_stratum)
        if take > 0:
            sampled.append(sub.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    return pd.concat(sampled, ignore_index=True)


def _build_feature_rows(
    sample: pd.DataFrame,
    pitches_all: pd.DataFrame,
    pitcher_cache,
    batter_cache,
    ctx,
    park_factors: dict[int, float],
    league_avgs_lookup,
) -> pd.DataFrame:
    """For each sample game, build a bundle and call production feature builders.

    Returns a DataFrame with one row per game containing the feature values
    captured from ``used_features`` plus observed targets.
    """
    from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID
    from scripts.historical_bundle import GameRecord, build_bundle
    from src.projection.features_bf import compute_e_bf
    from src.projection.features_kpa import (
        _lineup_chase_anchor,
        _lineup_zone_contact_anchor,
        _league_anchor_k_pct,
        compute_p_k_pa,
    )

    # Group pitches by game_pk once so per-game lineup lookups are fast.
    pa_by_game = (
        pitches_all.dropna(subset=["events", "pitcher", "batter"])
        .groupby("game_pk")
    )

    # pandas itertuples() mangles attribute names that start with double
    # underscores (Python identifier rules), so rename the internal __season
    # column for clean attribute access in the loop.
    sample = sample.rename(columns={"__season": "season"})

    rows: list[dict] = []
    skipped = 0
    for game in sample.itertuples():
        try:
            pa_in_game = pa_by_game.get_group(int(game.game_pk))
        except KeyError:
            skipped += 1
            continue
        lineup = _build_lineup_for_game(pa_in_game, int(game.pitcher))
        if len(lineup) < 6:
            # Pitcher faced fewer than 6 distinct batters — not a starter
            # outing worth fitting.
            skipped += 1
            continue

        venue_id = TEAM_TO_VENUE_ID.get(game.home_team)
        if venue_id is None:
            skipped += 1
            continue

        record = GameRecord(
            season=int(game.season),
            game_pk=int(game.game_pk),
            game_date=game.game_date,
            pitcher_id=int(game.pitcher),
            pitcher_hand=str(game.p_throws),
            pitcher_team="UNK",  # not needed for feature computation
            opposing_team="UNK",
            venue_id=int(venue_id),
            is_home=False,
            opposing_batters=lineup,
            observed_bf=int(game.observed_pa),
            observed_k=int(game.observed_k),
        )
        bundle = build_bundle(record, pitcher_cache, batter_cache)

        bf = compute_e_bf(bundle, ctx)
        kpa = compute_p_k_pa(bundle, ctx)
        if bf.skipped or kpa.skipped:
            skipped += 1
            continue

        # Capture league anchors as columns so reparameterize can center.
        league_k_anchor = _league_anchor_k_pct(bundle, ctx)
        league_zc_anchor = _lineup_zone_contact_anchor(bundle, ctx)
        league_chase_anchor = _lineup_chase_anchor(bundle, ctx)

        row = {
            **{k: v for k, v in bf.used_features.items() if isinstance(v, (int, float))},
            **{k: v for k, v in kpa.used_features.items() if isinstance(v, (int, float))},
            "league_k_pct_vs_hand": league_k_anchor,
            "league_zone_contact_anchor": league_zc_anchor,
            "league_chase_anchor": league_chase_anchor,
            "observed_bf": float(record.observed_bf),
            "observed_k": int(record.observed_k),
            "observed_pa": int(record.observed_bf),
            "observed_k_rate": record.observed_k / record.observed_bf,
            "season": record.season,
            "p_throws": record.pitcher_hand,
        }
        rows.append(row)

    logger.info(
        "feature extraction: %d kept, %d skipped (lineup/venue/builder skips)",
        len(rows), skipped,
    )
    return pd.DataFrame(rows)


def _check_min_sample_unless_smoke(
    n: int, min_rows: int, *, smoke: bool, context: str = "",
) -> bool:
    """Run :func:`require_min_sample` unless in smoke mode.

    Returns True if the check was performed (and passed; raises otherwise),
    False if it was skipped due to smoke. Smoke runs validate pipeline
    shape and don't run OLS, so the minimum-sample gate doesn't apply.
    """
    from scripts.design_matrix import require_min_sample

    if smoke:
        return False
    require_min_sample(n, min_rows, context=context)
    return True


def _fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    yhat = Xb @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return beta[1:], float(beta[0]), r2


def _weighted_logit_r2(
    y_rate: np.ndarray, yhat_logit: np.ndarray, weights: np.ndarray,
) -> float:
    eps = 1e-6
    y_clipped = np.clip(y_rate, eps, 1 - eps)
    y_logit = np.log(y_clipped / (1 - y_clipped))
    w_mean = float((weights * y_logit).sum() / weights.sum())
    ss_res = float((weights * (y_logit - yhat_logit) ** 2).sum())
    ss_tot = float((weights * (y_logit - w_mean) ** 2).sum())
    return 1.0 - (ss_res / max(ss_tot, 1e-9))


def _rate_r2(
    y_rate: np.ndarray,
    yhat_rate: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        ss_res = float(((y_rate - yhat_rate) ** 2).sum())
        ss_tot = float(((y_rate - y_rate.mean()) ** 2).sum())
    else:
        w_mean = float((weights * y_rate).sum() / weights.sum())
        ss_res = float((weights * (y_rate - yhat_rate) ** 2).sum())
        ss_tot = float((weights * (y_rate - w_mean) ** 2).sum())
    return 1.0 - (ss_res / max(ss_tot, 1e-9))


def _bootstrap_sign_consistency(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray | None,
    col_names: list[str],
    *,
    n_boot: int = 200,
    seed: int = 42,
    logit: bool = True,
) -> dict[str, dict[str, float]]:
    """Bootstrap-resample training data, refit, count coefficient signs.

    Stable coefficients should keep the same sign in >95% of bootstraps.
    Coefficients with bimodal sign distributions reveal collinearity-
    sensitive or noisy fits. 200 resamples × an OLS fit is fast (<1s).
    """
    rng = np.random.default_rng(seed)
    n = len(X_train)
    pos_counts = np.zeros(len(col_names))
    neg_counts = np.zeros(len(col_names))
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        X_b = X_train[idx]
        y_b = y_train[idx]
        w_b = weights[idx] if weights is not None else None
        try:
            if logit:
                coefs, _, _ = _fit_weighted_logit_ols(X_b, y_b, w_b)
            else:
                coefs, _, _ = _fit_ols(X_b, y_b)
        except np.linalg.LinAlgError:
            # Singular matrix on this resample — skip it
            continue
        pos_counts += (coefs > 1e-9).astype(int)
        neg_counts += (coefs < -1e-9).astype(int)
    total = pos_counts + neg_counts
    return {
        col: {
            "pct_positive": round(float(pos_counts[i] / n_boot), 4),
            "pct_negative": round(float(neg_counts[i] / n_boot), 4),
            # Sign consistency = max(pct_positive, pct_negative). 1.0 = always
            # same sign; 0.5 = bimodal.
            "consistency": round(
                float(max(pos_counts[i], neg_counts[i]) / max(total[i], 1)), 4,
            ),
        }
        for i, col in enumerate(col_names)
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full universe of 2024-2025 starter games (overnight rerun).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: build features but skip the fit. Useful for "
             "validating end-to-end pipeline without overnight commitment.",
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument(
        "--min-rows", type=int, default=None,
        help=(
            "Minimum post-NaN-drop sample size. Defaults to 2000 for --full, "
            "100 otherwise. Halt with InsufficientSampleError if violated."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from scripts.design_matrix import (
        check_design_matrix,
        drop_nan_rows,
        reparameterize_bf,
        reparameterize_kpa,
        require_min_sample,
    )
    from scripts.historical_bundle import build_batter_cache, build_pitcher_cache
    from src.projection.inputs import ProjectionContext

    # Static lookups
    park_factors_blob = json.loads(
        (PROCESSED_DIR / "park_k_factors.json").read_text(encoding="utf-8")
    )
    park_factors = {
        int(k): float(v["factor"]) for k, v in park_factors_blob["factors"].items()
    }
    ctx = ProjectionContext.from_default_paths()

    # Load + cache Statcast
    pitches_all = _load_seasons(args.seasons)
    pitches_all = pitches_all.dropna(subset=["pitcher", "batter", "game_pk"]).copy()
    logger.info("loaded %d pitches", len(pitches_all))

    pitcher_cache = build_pitcher_cache(pitches_all)
    batter_cache = build_batter_cache(pitches_all)
    logger.info(
        "caches: %d pitchers, %d batters",
        len(pitcher_cache.pitches), len(batter_cache.pas),
    )

    starter_games = _enumerate_starter_games(pitches_all)
    starter_games = starter_games[starter_games["__season"].isin([2024, 2025])]
    logger.info("starter games (2024+2025): %d", len(starter_games))

    if args.full:
        sample = starter_games
    else:
        sample = _stratified_sample(starter_games, n=args.sample)
    logger.info("sample: %d games", len(sample))

    rows_df = _build_feature_rows(
        sample, pitches_all, pitcher_cache, batter_cache, ctx,
        park_factors, league_avgs_lookup=None,
    )
    if rows_df.empty:
        raise SystemExit("no usable games in sample — investigate skip rate")

    # Re-parameterize design matrices. NaN propagation is intentional:
    # when a delta's baseline is missing (early season, sparse prior data,
    # etc.), the delta is undefined and we treat it as missing-required-
    # feature — drop the row, no median fill.
    #
    # Zero-variance columns (e.g. log_umpire_k_factor when the umpire
    # factor file is placeholder all-1.0) are dropped at reparameterization
    # time and tracked in the dropped_columns metadata. Self-healing: when
    # real data lands, the column reappears automatically.
    bf_design = reparameterize_bf(rows_df)
    kpa_design = reparameterize_kpa(rows_df)
    X_bf, X_kpa = bf_design.matrix, kpa_design.matrix

    if bf_design.dropped_columns:
        logger.info(
            "E[BF] dropped zero-variance columns: %s", bf_design.dropped_columns
        )
    if kpa_design.dropped_columns:
        logger.info(
            "P(K|PA) dropped zero-variance columns: %s", kpa_design.dropped_columns
        )

    n_before = len(rows_df)
    logger.info("sample before NaN drop: %d games", n_before)

    (X_bf, X_kpa, rows_df), drop_counts = drop_nan_rows(X_bf, X_kpa, rows_df)
    for col, n_drop in sorted(drop_counts.items()):
        logger.info("  dropped %d rows due to NaN in %s", n_drop, col)

    n_after = len(rows_df)
    pct_retained = (100.0 * n_after / n_before) if n_before else 0.0
    logger.info(
        "sample after NaN drop: %d games (%.1f%% retained)",
        n_after, pct_retained,
    )

    # Per (season, hand) cell counts so we can see if drops cluster anywhere.
    # Format: "post-drop cell counts: 2024 L=N, 2024 R=N, 2025 L=N, 2025 R=N"
    # Diagnostic only — cells below 50 get a WARNING but do not halt the fit
    # (overall-sample threshold via require_min_sample is the gate).
    if "season" in rows_df.columns and "p_throws" in rows_df.columns and n_after > 0:
        cell_counts = rows_df.groupby(["season", "p_throws"]).size()
        formatted = ", ".join(
            f"{int(season)} {hand}={int(count)}"
            for (season, hand), count in cell_counts.items()
        )
        logger.info("post-drop cell counts: %s", formatted)
        for (season, hand), count in cell_counts.items():
            if int(count) < 50:
                logger.warning(
                    "cell (season=%d, hand=%s) has only %d rows after drops "
                    "— per-hand effects may be unstable",
                    int(season), hand, int(count),
                )

    # Minimum-sample threshold. Default depends on mode. In --smoke we skip
    # the check entirely — smoke validates the pipeline shape, not the fit's
    # statistical adequacy, and OLS isn't run anyway.
    min_rows = args.min_rows
    if min_rows is None:
        min_rows = 2000 if args.full else 100
    if not _check_min_sample_unless_smoke(
        n_after, min_rows, smoke=args.smoke, context="post-NaN-drop fit pool"
    ):
        logger.info("--smoke: skipping require_min_sample check")

    # Sanity-check both BEFORE the fit. Halts on any violation.
    bf_diag = check_design_matrix(X_bf, name="E[BF]")
    kpa_diag = check_design_matrix(X_kpa, name="P(K|PA)")
    logger.info("E[BF] design matrix: %s", bf_diag)
    logger.info("P(K|PA) design matrix: %s", kpa_diag)

    if args.smoke:
        logger.info("Smoke run complete: pipeline validated, OLS skipped")
        return 0

    # Walk-forward split
    rows_df["__row"] = range(len(rows_df))
    train_mask = rows_df["season"] == 2024
    test_mask = rows_df["season"] == 2025
    train = rows_df[train_mask].reset_index(drop=True)
    test = rows_df[test_mask].reset_index(drop=True)
    X_bf_train = reparameterize_bf(train).matrix.to_numpy(dtype=float)
    X_bf_test = reparameterize_bf(test).matrix.to_numpy(dtype=float)
    X_kpa_train = reparameterize_kpa(train).matrix.to_numpy(dtype=float)
    X_kpa_test = reparameterize_kpa(test).matrix.to_numpy(dtype=float)

    bf_coefs, bf_intercept, bf_r2_in = _fit_ols(
        X_bf_train, train["observed_bf"].to_numpy(dtype=float)
    )
    yhat = bf_intercept + X_bf_test @ bf_coefs
    yte = test["observed_bf"].to_numpy(dtype=float)
    bf_r2_out = 1.0 - (
        ((yte - yhat) ** 2).sum() / max(((yte - yte.mean()) ** 2).sum(), 1e-9)
    )

    y_train_rate = train["observed_k_rate"].to_numpy(dtype=float)
    y_test_rate = test["observed_k_rate"].to_numpy(dtype=float)
    w_train = train["observed_pa"].to_numpy(dtype=float)
    w_test = test["observed_pa"].to_numpy(dtype=float)

    kpa_coefs, kpa_intercept, kpa_r2_in_logit = _fit_weighted_logit_ols(
        X_kpa_train, y_train_rate, w_train,
    )

    # K|PA out-of-sample R² in logit space (Task 1 — missing branch)
    yhat_train_logit = kpa_intercept + X_kpa_train @ kpa_coefs
    yhat_test_logit = kpa_intercept + X_kpa_test @ kpa_coefs
    kpa_r2_out_logit = _weighted_logit_r2(y_test_rate, yhat_test_logit, w_test)

    # K|PA rate-space R² (Task 2 — apples-to-apples with original gate spec)
    yhat_train_rate = 1.0 / (1.0 + np.exp(-yhat_train_logit))
    yhat_test_rate = 1.0 / (1.0 + np.exp(-yhat_test_logit))
    kpa_r2_in_rate = _rate_r2(y_train_rate, yhat_train_rate)
    kpa_r2_out_rate = _rate_r2(y_test_rate, yhat_test_rate)
    kpa_r2_in_rate_weighted = _rate_r2(y_train_rate, yhat_train_rate, w_train)
    kpa_r2_out_rate_weighted = _rate_r2(y_test_rate, yhat_test_rate, w_test)

    # Bootstrap sign stability (Task 3 — diagnose the sign flips)
    bf_cols = list(reparameterize_bf(train).matrix.columns)
    kpa_cols = list(reparameterize_kpa(train).matrix.columns)
    logger.info("running 200-iter bootstrap for BF + K|PA sign stability...")
    bf_bootstrap = _bootstrap_sign_consistency(
        X_bf_train,
        train["observed_bf"].to_numpy(dtype=float),
        weights=None,
        col_names=bf_cols,
        n_boot=200, seed=42, logit=False,
    )
    kpa_bootstrap = _bootstrap_sign_consistency(
        X_kpa_train, y_train_rate, w_train,
        col_names=kpa_cols,
        n_boot=200, seed=42, logit=True,
    )
    bf_payload = {
        "sample_fit": not args.full,
        "n_games_train": int(len(train)),
        "n_games_test": int(len(test)),
        "in_sample_r2": round(bf_r2_in, 4),
        "out_of_sample_r2": round(bf_r2_out, 4),
        "intercept": round(bf_intercept, 4),
        "coefficients": {
            f: round(float(c), 4) for f, c in zip(bf_cols, bf_coefs)
        },
        "design_matrix_diagnostics": bf_diag,
        "dropped_zero_variance": bf_design.dropped_columns,
        "bootstrap_sign_consistency": bf_bootstrap,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    kpa_payload = {
        "sample_fit": not args.full,
        "n_games_train": int(len(train)),
        "n_games_test": int(len(test)),
        "in_sample_r2_logit": round(kpa_r2_in_logit, 4),
        "out_of_sample_r2_logit": round(kpa_r2_out_logit, 4),
        "logit_r2_delta": round(kpa_r2_in_logit - kpa_r2_out_logit, 4),
        "rate_space_r2": {
            "in_sample": round(kpa_r2_in_rate, 4),
            "out_of_sample": round(kpa_r2_out_rate, 4),
            "in_sample_weighted": round(kpa_r2_in_rate_weighted, 4),
            "out_of_sample_weighted": round(kpa_r2_out_rate_weighted, 4),
        },
        "intercept_logit": round(kpa_intercept, 4),
        "coefficients_logit": {
            f: round(float(c), 4) for f, c in zip(kpa_cols, kpa_coefs)
        },
        "design_matrix_diagnostics": kpa_diag,
        "dropped_zero_variance": kpa_design.dropped_columns,
        "bootstrap_sign_consistency": kpa_bootstrap,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (PROCESSED_DIR / "feature_coefficients_bf.json").write_text(
        json.dumps(bf_payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    (PROCESSED_DIR / "feature_coefficients_kpa.json").write_text(
        json.dumps(kpa_payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    logger.info(
        "E[BF] fit: in-sample R^2=%.4f, out-of-sample R^2=%.4f",
        bf_r2_in, bf_r2_out,
    )
    logger.info(
        "P(K|PA) fit (logit): in-sample R^2=%.4f, out-of-sample R^2=%.4f, "
        "delta=%.4f",
        kpa_r2_in_logit, kpa_r2_out_logit,
        kpa_r2_in_logit - kpa_r2_out_logit,
    )
    logger.info(
        "P(K|PA) fit (rate space): in=%.4f / out=%.4f | "
        "weighted in=%.4f / out=%.4f",
        kpa_r2_in_rate, kpa_r2_out_rate,
        kpa_r2_in_rate_weighted, kpa_r2_out_rate_weighted,
    )
    logger.info("bootstrap sign consistency (200 resamples):")
    logger.info("  E[BF]:")
    for col, info in bf_bootstrap.items():
        logger.info(
            "    %-40s  pos=%.2f  neg=%.2f  consistency=%.2f",
            col, info["pct_positive"], info["pct_negative"], info["consistency"],
        )
    logger.info("  P(K|PA):")
    for col, info in kpa_bootstrap.items():
        logger.info(
            "    %-40s  pos=%.2f  neg=%.2f  consistency=%.2f",
            col, info["pct_positive"], info["pct_negative"], info["consistency"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
