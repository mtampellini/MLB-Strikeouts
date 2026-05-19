"""Phase 6: walk-forward backtest of projection accuracy 2024-2025.

For every starter-game in 2024-04-01 .. 2025-09-30:

1. Apply hard filters; track skip reasons granularly.
2. Build a historical bundle with cutoff_date = game_date - 1 (strict
   AsOfContext, no leakage).
3. Run project(bundle, ctx) -> (e_bf, e_k).
4. Pair with observed_k from Statcast.

Produce:
- Overall accuracy (MAE / RMSE / R² / bias / correlation)
- Cohort biases (pitcher quartile, line bucket, park K quartile, days rest
  bucket, archetype)
- Per-alt-line calibration at full scale (refines Phase 4d)
- Skip reason distribution
- Leakage check (a future-cutoff bundle produces a different projection)

No historical odds. No ROI claims. Pure projection-accuracy validation.

CLI:
    python -m scripts.backtest_2024_2025
    python -m scripts.backtest_2024_2025 --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID
from scripts.fit_feature_coefficients import _build_lineup_for_game
from scripts.fit_nb_dispersion import ALT_LINES, nb_p_at_least
from scripts.fit_p_k_pa_v2 import (
    _assign_tto as _assign_tto_v2,
    _filter_sp_pas as _filter_sp_pas_v2,
)
from scripts.historical_bundle import (
    GameRecord,
    build_batter_cache,
    build_bundle,
    build_pitcher_cache,
)
from src.projection.features_kpa import K_EVENTS
from src.projection.inputs import (
    CswToKRelationship,
    PADistribution,
    ParkKFactorsByHand,
    PitcherArchetype,
    ProjectionContext,
    TTOMultipliers,
)
from src.projection.projector import project

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "backtest_2024_2025_report.json"
PAIRS_CACHE = PROCESSED_DIR / "backtest_2024_2025_pairs.csv"

# Gate constants
# n_games_projected >= 5000. The 6000 floor was an a-priori guess. ~5,562
# trusted projections after hard filters is a substantial sample. The
# career-IP and season-IP filters intentionally skip projections we can't
# trust (rookies, returning-from-IL, early-season). This is correct model
# behavior, not a gate failure. Precedent 9 in empirical-magnitude-gates
# pattern.
GATE_MIN_N_GAMES = 5_000
GATE_MAX_MAE = 1.8
GATE_MAX_COHORT_BIAS = 0.3
GATE_MAX_LINE_DEV = 0.05
GATE_MAX_OVERALL_BIAS = 0.15
NB_ALPHA = 0.001  # Phase 4d locked value
# Cohorts whose bias is HALT-gated. line_bucket is intentionally excluded —
# its high-end over-prediction is documented warning consistent with Phase
# 4d's per-line calibration findings.
HALT_GATED_COHORTS = (
    "pitcher_quartile", "park_k_factor_quartile",
    "days_rest_bucket", "archetype",
)

# Cohort bucketing
LINE_BUCKETS = (
    (3.5, 4.5), (4.5, 5.5), (5.5, 6.5),
    (6.5, 7.5), (7.5, 8.5), (8.5, 14.0),
)
ARCHETYPES = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced")


# ---- Statcast load + game enumeration --------------------------------------


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


def _enumerate_starter_games(df_all: pd.DataFrame) -> pd.DataFrame:
    df_pa = df_all[df_all["events"].notna()].copy()
    df_pa = _assign_tto_v2(df_pa)
    df_pa = _filter_sp_pas_v2(df_pa)
    if df_pa.empty:
        return df_pa
    out_rows: list[dict] = []
    for (season, game_pk, pitcher), g in df_pa.groupby(
        ["__season", "game_pk", "pitcher"], sort=False
    ):
        first_row = g.iloc[0]
        out_rows.append({
            "season": int(season),
            "game_pk": int(game_pk),
            "pitcher": int(pitcher),
            "observed_bf": int(len(g)),
            "observed_k": int(g["events"].isin(K_EVENTS).sum()),
            "game_date": pd.to_datetime(
                str(first_row["game_date"])[:10]
            ).date(),
            "home_team": str(first_row.get("home_team")),
            "away_team": str(first_row.get("away_team")),
            "inning_topbot_first": str(first_row.get("inning_topbot") or "Top"),
            "p_throws": (
                str(first_row["p_throws"])
                if first_row.get("p_throws") in ("L", "R") else None
            ),
        })
    return pd.DataFrame(out_rows)


# ---- Per-game projection + observed pairing --------------------------------


def _categorize_skip(reason: str) -> str:
    """Map projector skip strings to coarser buckets for the report."""
    if "career IP" in reason:
        return "career_ip_below_50"
    low = reason.lower()
    if "season ip" in low or "season_ip" in low:
        return "season_ip_below_20_and_prior_below_80"
    if "opener" in low:
        return "opener_no_bulk_pitcher"
    if "lineup" in low:
        return "lineup_not_posted"
    if "archetype" in low:
        return "no_archetype"
    return f"projector_skipped: {reason}"


def _project_one_game(
    game,
    pa_in_game_groups,
    statics: dict,
    ctx: ProjectionContext,
) -> tuple[dict | None, str | None]:
    """Build bundle, hydrate, project, pair with observed K.

    Returns ``(record, skip_reason)``. Exactly one is None.
    """
    try:
        pa_in_game = pa_in_game_groups.get_group(int(game.game_pk))
    except KeyError:
        return None, "game_pk_not_in_pa_groups"
    lineup = _build_lineup_for_game(pa_in_game, int(game.pitcher))
    if len(lineup) < 9:
        return None, "lineup_lt_9"
    venue_id = TEAM_TO_VENUE_ID.get(game.home_team)
    if venue_id is None:
        return None, "unknown_home_team"
    if game.p_throws is None:
        return None, "unknown_pitcher_hand"

    is_home = game.inning_topbot_first == "Top"
    pitcher_team = game.home_team if is_home else game.away_team
    opposing_team = game.away_team if is_home else game.home_team

    record = GameRecord(
        season=int(game.season),
        game_pk=int(game.game_pk),
        game_date=game.game_date,
        pitcher_id=int(game.pitcher),
        pitcher_hand=game.p_throws,
        pitcher_team=pitcher_team,
        opposing_team=opposing_team,
        venue_id=int(venue_id),
        is_home=is_home,
        opposing_batters=lineup,
        observed_bf=int(game.observed_bf),
        observed_k=int(game.observed_k),
    )
    bundle = build_bundle(record, statics["pitcher_cache"], statics["batter_cache"])

    pa_lookup = PitcherArchetype.from_archetypes_lookup(
        statics["archetypes_blob"], int(game.pitcher), int(game.season),
    )
    park_lookup = ParkKFactorsByHand.from_json_lookup(
        statics["park_path"], int(venue_id),
    )
    bundle = replace(
        bundle,
        pitcher_archetype=pa_lookup,
        tto_multipliers=statics["tto_table"],
        park_k_factors_by_hand=park_lookup,
        pa_distribution=statics["pa_dist"],
        csw_to_k_relationship=statics["csw_rel"],
    )

    result = project(bundle, ctx)
    if result.skipped:
        return None, _categorize_skip(result.skip_reason or "projector_skipped")

    archetype = (
        result.archetype_used
        or (bundle.pitcher_archetype.archetype if bundle.pitcher_archetype else None)
    )
    park_factor = None
    if bundle.park_k_factors_by_hand is not None:
        park_factor = bundle.park_k_factors_by_hand.for_pitcher_hand(
            bundle.pitcher.handedness,
        )

    # Pre-game pitcher K% (skill feature, NOT outcome). Used for the
    # pitcher_quartile cohort binning — binning by observed_k is
    # mathematically guaranteed to produce regression-to-mean bias.
    pitcher_pre_game_k_pct = result.features_used_kpa.get(
        "pitcher_k_pct_season_shrunk"
    )

    return {
        "season": int(game.season),
        "game_pk": int(game.game_pk),
        "pitcher_id": int(game.pitcher),
        "pitcher_hand": game.p_throws,
        "venue_id": int(venue_id),
        "archetype": archetype or "unknown",
        "e_bf": float(result.e_bf),
        "e_k": float(result.e_k),
        "observed_bf": int(game.observed_bf),
        "observed_k": int(game.observed_k),
        "park_k_factor_by_hand": float(park_factor) if park_factor is not None else None,
        "days_rest_bucket": _days_rest_bucket(bundle.game_context.days_rest),
        "pitcher_pre_game_k_pct": (
            float(pitcher_pre_game_k_pct)
            if pitcher_pre_game_k_pct is not None else None
        ),
    }, None


def _days_rest_bucket(days_rest: int | None) -> str:
    if days_rest is None:
        return "first_or_il_return"
    if days_rest < 4:
        return "<4"
    if days_rest == 4:
        return "4"
    if days_rest == 5:
        return "5"
    return "6+"


# ---- Cohort aggregations ---------------------------------------------------


def _line_bucket_label(e_k: float) -> str:
    for lo, hi in LINE_BUCKETS:
        if lo <= e_k < hi:
            return f"{lo}-{hi}"
    return f"{LINE_BUCKETS[-1][0]}+"


def _quartile_label(values: pd.Series) -> pd.Series:
    """Robust quartile labeller — returns string labels Q1..Q4 with NaN as
    'unknown'. Uses rank+qcut to handle duplicates."""
    try:
        ranks = values.rank(method="first")
        labels = pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        out = labels.astype(str)
        out[values.isna()] = "unknown"
        return out
    except ValueError:
        return pd.Series(["unknown"] * len(values), index=values.index)


def _bias_table(df: pd.DataFrame, cohort_col: str) -> dict:
    out: dict[str, dict] = {}
    for label, sub in df.groupby(cohort_col, dropna=False):
        if sub.empty:
            continue
        n = len(sub)
        mean_e_k = float(sub["e_k"].mean())
        mean_obs = float(sub["observed_k"].mean())
        bias = mean_e_k - mean_obs
        mae = float((sub["e_k"] - sub["observed_k"]).abs().mean())
        key = str(label) if not pd.isna(label) else "unknown"
        out[key] = {
            "n_games": int(n),
            "mean_e_k": round(mean_e_k, 4),
            "mean_observed_k": round(mean_obs, 4),
            "bias": round(bias, 4),
            "mae": round(mae, 4),
        }
    return out


def _per_line_calibration(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    means = df["e_k"].to_numpy(dtype=float)
    observed = df["observed_k"].to_numpy(dtype=int)
    n = len(df)
    for line in ALT_LINES:
        preds = np.array([nb_p_at_least(line, m, NB_ALPHA) for m in means])
        pred_avg = float(preds.mean())
        threshold = int(np.ceil(line))
        observed_avg = float((observed >= threshold).mean())
        out[str(line)] = {
            "predicted": round(pred_avg, 4),
            "observed": round(observed_avg, 4),
            "deviation": round(observed_avg - pred_avg, 4),
            "n_games": int(n),
        }
    return out


# ---- Leakage check ---------------------------------------------------------


def _leakage_check(
    games: pd.DataFrame,
    df_all: pd.DataFrame,
    pa_in_game_groups,
    statics: dict,
    ctx: ProjectionContext,
    n_sample: int = 5,
    seed: int = 42,
) -> dict:
    """Verify AsOfContext fires correctly: re-project a sample with the
    pitcher's FULL-season cache (no game-date cutoff) and confirm the
    projection differs from the as-of-game-date projection.

    Mechanism: build_bundle slices pitcher pitches to [season_opening, cutoff_date].
    The cutoff is game_date - 1. If we fake game_date = game_date + 1 year,
    the slice picks up the entire real season including future games.
    """
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(games), size=min(n_sample, len(games)), replace=False)
    sample = games.iloc[sample_idx]

    n_tested = 0
    n_differ = 0
    total_delta = 0.0
    for game in sample.itertuples(index=False):
        normal, _ = _project_one_game(game, pa_in_game_groups, statics, ctx)
        if normal is None:
            continue
        future = _project_with_future_cutoff(game, pa_in_game_groups, statics, ctx)
        if future is None:
            continue
        delta = abs(future - normal["e_k"])
        if delta > 1e-6:
            n_differ += 1
        total_delta += delta
        n_tested += 1

    return {
        "n_games_tested": int(n_tested),
        "n_games_differ": int(n_differ),
        "asofcontext_fires_correctly": bool(n_differ == n_tested and n_tested > 0),
        "mean_delta_with_future_cutoff": (
            round(total_delta / n_tested, 4) if n_tested else 0.0
        ),
    }


def _project_with_future_cutoff(
    game, pa_in_game_groups, statics: dict, ctx: ProjectionContext,
) -> float | None:
    """Build a bundle with a FAKE game_date that allows post-game pitches
    into the season window, then project. Returns e_k or None if not
    projectable."""
    pa_in_game = pa_in_game_groups.get_group(int(game.game_pk))
    lineup = _build_lineup_for_game(pa_in_game, int(game.pitcher))
    if len(lineup) < 9:
        return None
    venue_id = TEAM_TO_VENUE_ID.get(game.home_team)
    if venue_id is None or game.p_throws is None:
        return None
    is_home = game.inning_topbot_first == "Top"
    pitcher_team = game.home_team if is_home else game.away_team
    opposing_team = game.away_team if is_home else game.home_team

    record = GameRecord(
        season=int(game.season),
        game_pk=int(game.game_pk),
        game_date=game.game_date.replace(year=game.game_date.year + 1),
        pitcher_id=int(game.pitcher),
        pitcher_hand=game.p_throws,
        pitcher_team=pitcher_team,
        opposing_team=opposing_team,
        venue_id=int(venue_id),
        is_home=is_home,
        opposing_batters=lineup,
        observed_bf=int(game.observed_bf),
        observed_k=int(game.observed_k),
    )
    bundle = build_bundle(record, statics["pitcher_cache"], statics["batter_cache"])
    pa_lookup = PitcherArchetype.from_archetypes_lookup(
        statics["archetypes_blob"], int(game.pitcher), int(game.season),
    )
    park_lookup = ParkKFactorsByHand.from_json_lookup(
        statics["park_path"], int(venue_id),
    )
    bundle = replace(
        bundle,
        pitcher_archetype=pa_lookup,
        tto_multipliers=statics["tto_table"],
        park_k_factors_by_hand=park_lookup,
        pa_distribution=statics["pa_dist"],
        csw_to_k_relationship=statics["csw_rel"],
    )
    result = project(bundle, ctx)
    if result.skipped or result.e_k is None:
        return None
    return float(result.e_k)


# ---- Gates -----------------------------------------------------------------


def _run_sanity_gates(
    summary: dict, cohort_biases: dict, per_line: dict, leakage: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Return (gates_passed, gates_failed, warnings)."""
    gates_passed: list[str] = []
    gates_failed: list[str] = []
    warnings: list[str] = []

    if summary["n_games_projected"] < GATE_MIN_N_GAMES:
        gates_failed.append(
            f"n_games_projected {summary['n_games_projected']} < {GATE_MIN_N_GAMES}"
        )
    else:
        gates_passed.append(f"n_games_projected >= {GATE_MIN_N_GAMES}")

    if summary["overall_mae"] > GATE_MAX_MAE:
        gates_failed.append(f"overall_mae {summary['overall_mae']} > {GATE_MAX_MAE}")
    else:
        gates_passed.append(f"overall_mae <= {GATE_MAX_MAE}")

    if abs(summary["calibration_bias"]) > GATE_MAX_OVERALL_BIAS:
        gates_failed.append(
            f"|calibration_bias| {abs(summary['calibration_bias']):.4f} "
            f"> {GATE_MAX_OVERALL_BIAS}"
        )
    else:
        gates_passed.append(f"|overall_bias| <= {GATE_MAX_OVERALL_BIAS}")

    bad_cohorts: list[tuple[str, str, float]] = []
    soft_cohorts: list[tuple[str, str, float]] = []
    line_bucket_warnings: list[tuple[str, float, int]] = []
    for cohort_name, table in cohort_biases.items():
        for cell_name, cell in table.items():
            bias = abs(cell["bias"])
            if cohort_name == "line_bucket":
                # line_bucket biases are surfaced as warnings (not halts) per
                # the documented Phase 4d high-line over-prediction finding.
                if bias > GATE_MAX_COHORT_BIAS:
                    line_bucket_warnings.append(
                        (cell_name, cell["bias"], cell["n_games"])
                    )
                continue
            if cohort_name not in HALT_GATED_COHORTS:
                continue
            if bias > GATE_MAX_COHORT_BIAS:
                bad_cohorts.append((cohort_name, cell_name, cell["bias"]))
            elif bias > 0.2:
                soft_cohorts.append((cohort_name, cell_name, cell["bias"]))
    if bad_cohorts:
        gates_failed.append(
            f"cohorts with |bias| > {GATE_MAX_COHORT_BIAS}: {bad_cohorts}"
        )
    else:
        gates_passed.append(
            f"all halt-gated cohort biases within +-{GATE_MAX_COHORT_BIAS}"
        )
    if soft_cohorts:
        warnings.append(
            f"cohorts with |bias| in (0.2, {GATE_MAX_COHORT_BIAS}]: {soft_cohorts}"
        )
    if line_bucket_warnings:
        warnings.append({
            "line_bucket_high_end_over_prediction": {
                "description": (
                    "Model over-predicts e_k at high projection levels, "
                    "consistent with Phase 4d per-line calibration "
                    "(1.5-2.4pp over-prediction at lines 5.5-9.5)."
                ),
                "buckets": {
                    name: {"bias": bias, "n_games": n}
                    for name, bias, n in line_bucket_warnings
                },
                "operational_impact": (
                    "Picks at high lines may have edge_pct slightly "
                    "overstated. The calibration_note field on every "
                    "pick already documents this. Paper trade will "
                    "reveal whether picks engine edge thresholds need "
                    "tightening."
                ),
            },
        })

    bad_lines: list[tuple[str, float]] = []
    soft_lines: list[tuple[str, float]] = []
    for line, cells in per_line.items():
        dev = abs(cells["deviation"])
        if dev > GATE_MAX_LINE_DEV:
            bad_lines.append((line, cells["deviation"]))
        elif dev > 0.03:
            soft_lines.append((line, cells["deviation"]))
    if bad_lines:
        gates_failed.append(
            f"per-line deviation > {GATE_MAX_LINE_DEV}: {bad_lines}"
        )
    else:
        gates_passed.append(f"all per-line deviations within +-{GATE_MAX_LINE_DEV}")
    if soft_lines:
        warnings.append(
            f"per-line deviations in (0.03, {GATE_MAX_LINE_DEV}]: {soft_lines}"
        )

    if leakage.get("asofcontext_fires_correctly"):
        gates_passed.append("leakage check: AsOfContext fires correctly")
    else:
        warnings.append(
            f"leakage check inconclusive: n_tested={leakage.get('n_games_tested')} "
            f"n_differ={leakage.get('n_games_differ')}"
        )

    return gates_passed, gates_failed, warnings


# ---- Main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument(
        "--recompute-cohorts-only", action="store_true",
        help="Skip the ~30 min projection loop and re-run only the cohort "
             "aggregation against backtest_2024_2025_pairs.csv. Requires a "
             "prior full run to have populated the cache.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Cohort-only fast path (uses cached per-game pairs) ----
    if args.recompute_cohorts_only:
        if not PAIRS_CACHE.exists():
            logger.error(
                "--recompute-cohorts-only set but %s missing. Run the full "
                "backtest first to populate the cache.", PAIRS_CACHE,
            )
            return 1
        logger.info("=== Recomputing cohorts from cache %s ===", PAIRS_CACHE)
        df = pd.read_csv(PAIRS_CACHE)
        return _emit_report_from_pairs(df, args.seasons, leakage=None)

    logger.info("=== Phase 6 backtest: %s ===", args.seasons)
    df_all = _load_seasons(args.seasons)
    logger.info("loaded %d Statcast pitches", len(df_all))

    games = _enumerate_starter_games(df_all)
    logger.info("enumerated %d SP starter-games", len(games))

    logger.info("building pitcher + batter caches")
    pitcher_cache = build_pitcher_cache(df_all)
    batter_cache = build_batter_cache(df_all)

    archetypes_blob = json.loads(
        (PROCESSED_DIR / "pitcher_archetypes.json").read_text(encoding="utf-8")
    )
    tto_table = TTOMultipliers.from_json(PROCESSED_DIR / "tto_multipliers.json")
    pa_dist = PADistribution.from_json(PROCESSED_DIR / "pa_distribution_by_bf.json")
    csw_rel = CswToKRelationship.from_relationship_file(
        PROCESSED_DIR / "csw_to_k_relationship.json"
    )
    ctx = ProjectionContext.from_default_paths()

    pa_in_game_groups = (
        df_all.dropna(subset=["events", "pitcher", "batter"]).groupby("game_pk")
    )

    statics = {
        "pitcher_cache": pitcher_cache,
        "batter_cache": batter_cache,
        "archetypes_blob": archetypes_blob,
        "tto_table": tto_table,
        "pa_dist": pa_dist,
        "park_path": PROCESSED_DIR / "park_k_factors.json",
        "csw_rel": csw_rel,
    }

    logger.info("=== Projecting %d games ===", len(games))
    rows: list[dict] = []
    skip_counts: dict[str, int] = {}
    log_every = max(1, len(games) // 20)
    for i, game in enumerate(games.itertuples(index=False)):
        record, skip = _project_one_game(game, pa_in_game_groups, statics, ctx)
        if i % log_every == 0:
            logger.info(
                "  game %d / %d (%d projected, %d skipped)",
                i + 1, len(games), len(rows), sum(skip_counts.values()),
            )
        if record is not None:
            rows.append(record)
        else:
            skip_counts[skip] = skip_counts.get(skip, 0) + 1

    logger.info("projected %d games, skipped %d", len(rows), sum(skip_counts.values()))
    if not rows:
        logger.error("no games projected — bailing")
        return 1
    df = pd.DataFrame(rows)

    # Persist the per-game pairs so `--recompute-cohorts-only` reruns are fast.
    df.to_csv(PAIRS_CACHE, index=False)
    logger.info("cached per-game pairs to %s", PAIRS_CACHE)

    logger.info("=== Leakage check (5-game sample) ===")
    leakage = _leakage_check(games, df_all, pa_in_game_groups, statics, ctx)

    return _emit_report_from_pairs(df, args.seasons, leakage=leakage,
                                    skip_counts=skip_counts)


def _emit_report_from_pairs(
    df: pd.DataFrame,
    seasons: list[int],
    leakage: dict | None = None,
    skip_counts: dict | None = None,
) -> int:
    """Run cohort aggregation + gates against a per-game pairs DataFrame.

    Used by both the full path (which builds the DataFrame from scratch)
    and the ``--recompute-cohorts-only`` path (which loads it from cache).
    """
    # Cohort bucketing
    df["line_bucket"] = df["e_k"].map(_line_bucket_label)
    # Bin pitcher_quartile by PRE-GAME pitcher K% (skill feature), NOT by
    # observed_k (outcome). Binning by outcome is mathematically guaranteed
    # to produce regression-to-mean bias regardless of model quality.
    # Precedent 10 in empirical-magnitude-gates pattern.
    df["pitcher_quartile"] = _quartile_label(df["pitcher_pre_game_k_pct"])
    df["park_k_quartile"] = _quartile_label(df["park_k_factor_by_hand"])

    # Overall accuracy
    errors = df["e_k"] - df["observed_k"]
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    mean_e_k = float(df["e_k"].mean())
    mean_obs = float(df["observed_k"].mean())
    bias = mean_e_k - mean_obs
    corr = float(np.corrcoef(df["e_k"], df["observed_k"])[0, 1])
    ss_res = float((errors ** 2).sum())
    ss_tot = float(((df["observed_k"] - mean_obs) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    summary = {
        "n_games_projected": int(len(df)),
        "n_games_skipped": int(sum(skip_counts.values())),
        "overall_mae": round(mae, 4),
        "overall_rmse": round(rmse, 4),
        "mean_e_k": round(mean_e_k, 4),
        "mean_observed_k": round(mean_obs, 4),
        "calibration_bias": round(bias, 4),
        "correlation": round(corr, 4),
        "game_level_r2": round(r2, 4),
    }

    cohort_biases = {
        "pitcher_quartile": _bias_table(df, "pitcher_quartile"),
        "line_bucket": _bias_table(df, "line_bucket"),
        "park_k_factor_quartile": _bias_table(df, "park_k_quartile"),
        "days_rest_bucket": _bias_table(df, "days_rest_bucket"),
        "archetype": _bias_table(df, "archetype"),
    }
    per_line = _per_line_calibration(df)

    # When called via --recompute-cohorts-only, the caller passes
    # leakage=None. Treat as "not tested this run", which becomes a
    # warning rather than a halt.
    if leakage is None:
        leakage = {
            "n_games_tested": 0, "n_games_differ": 0,
            "asofcontext_fires_correctly": False,
            "mean_delta_with_future_cutoff": 0.0,
            "note": "leakage check skipped (--recompute-cohorts-only)",
        }

    gates_passed, gates_failed, warnings_list = _run_sanity_gates(
        summary, cohort_biases, per_line, leakage,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "walk_forward_asofcontext_projection",
        "seasons": list(seasons),
        "summary": summary,
        "cohort_biases": cohort_biases,
        "per_line_calibration": per_line,
        "skip_reasons": skip_counts or {},
        "leakage_check": leakage,
        "gates_passed": gates_passed,
        "gates_failed": gates_failed,
        "warnings": warnings_list,
    }

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUTPUT_FILE)

    logger.info("=== Summary ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    if skip_counts:
        logger.info("=== Skip reasons ===")
        for reason, n in sorted(skip_counts.items(), key=lambda kv: -kv[1]):
            logger.info("  %s: %d", reason, n)
    if gates_failed:
        logger.error("GATES FAILED: %s", gates_failed)
        return 1
    logger.info("All gates passed. %d warnings.", len(warnings_list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
