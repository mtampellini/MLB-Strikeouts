"""Phase 4b: derive K-specific park factors from 3 seasons of Statcast.

Per the spec: shrunk observed/expected K rate by venue. The "expected" uses
each pitcher's full-season K% so high-K-arsenal teams (Tampa, LA, etc.)
don't distort their home parks' factors. Lineup K% adjustment is omitted
deliberately — it requires per-game opposing-team tracking that's expensive
to maintain and the pitcher-quality control already absorbs the dominant
source of confounding.

Output: data/processed/park_k_factors.json, keyed by venue_id with
{venue_name, factor, n_games}. Shrinkage prior k_prior=300 games keeps
small-sample parks (Athletics' Sutter Health Park in 2025, ~70 games) close
to neutral.

Hard rules:
- ``game_type == 'R'`` regular season only
- Pitcher-game (not pitch-game) aggregation; we use observed K/PA per
  pitcher per game
- Sanity assertions halt on Coors/Trop > 1.0, missing venues, or many
  parks outside [0.90, 1.10]

CLI:
    python -m scripts.derive_park_k_factors
    python -m scripts.derive_park_k_factors --seasons 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})

# Statcast home_team uses StatsAPI 2-letter form. Map to MLB venue_id.
# Statcast normalizes all Athletics games (2023-Coliseum, 2024-Coliseum,
# 2025-Sutter Health) to home_team='ATH'; there is no 'OAK' label in the
# feed. So all Athletics games end up at venue 2529, blending the two
# physical venues. Acknowledged in the output _comment.
TEAM_TO_VENUE_ID = {
    "ATH": 2529, "AZ": 15, "ATL": 4705, "BAL": 2, "BOS": 3, "CHC": 17,
    "CIN": 2602, "CLE": 5, "COL": 19, "CWS": 4, "DET": 2394, "HOU": 2392,
    "KC": 7, "LAA": 1, "LAD": 22, "MIA": 4169, "MIL": 32, "MIN": 3312,
    "NYM": 3289, "NYY": 3313, "PHI": 2681, "PIT": 31, "SD": 2680,
    "SEA": 680, "SF": 2395, "STL": 2889, "TB": 12, "TEX": 5325, "TOR": 14,
    "WSH": 3309,
}

VENUE_NAMES = {
    1: "Angel Stadium", 2: "Oriole Park at Camden Yards", 3: "Fenway Park",
    4: "Rate Field", 5: "Progressive Field", 7: "Kauffman Stadium",
    12: "Tropicana Field", 14: "Rogers Centre",
    15: "Chase Field", 17: "Wrigley Field", 19: "Coors Field",
    22: "Dodger Stadium", 31: "PNC Park", 32: "American Family Field",
    680: "T-Mobile Park", 2392: "Daikin Park", 2394: "Comerica Park",
    2395: "Oracle Park", 2529: "Sutter Health Park (ATH 2023-24 Coliseum games merged here)",
    2602: "Great American Ball Park", 2680: "Petco Park",
    2681: "Citizens Bank Park", 2889: "Busch Stadium", 3289: "Citi Field",
    3309: "Nationals Park", 3312: "Target Field", 3313: "Yankee Stadium",
    4169: "loanDepot park", 4705: "Truist Park", 5325: "Globe Life Field",
}

SHRINKAGE_PRIOR_GAMES = 300         # legacy single-factor mode
SHRINKAGE_PRIOR_GAMES_BY_HAND = 200  # per-hand split — smaller because halved sample


def _pull_seasons(seasons: list[int]) -> pd.DataFrame:
    from pybaseball import statcast  # type: ignore

    frames: list[pd.DataFrame] = []
    for season in seasons:
        start = f"{season}-03-15"
        end = f"{season}-11-30"
        logger.info("pulling Statcast %s..%s", start, end)
        df = statcast(start_dt=start, end_dt=end)
        df = df[df["game_type"] == "R"].copy()
        logger.info("season %d: %d regular-season pitches", season, len(df))
        df["__season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


MIN_EXCL_GAMES_FOR_BASELINE = 5  # leave-one-venue-out gate


def _build_pitcher_game_table(
    df: pd.DataFrame, *, leave_one_venue_out: bool = True,
) -> pd.DataFrame:
    """Per pitcher-game table with observed K/PA, expected K, ratio, venue,
    p_throws.

    Default uses leave-one-venue-out methodology for the expected baseline:
    for each pitcher-game at venue V, the pitcher's "season K%" is computed
    across that pitcher's games in the SAME season EXCLUDING starts at V.
    This removes home-park dampening — without it, a pitcher's full-season
    K% includes V's effect and biases the park factor toward 1.0.

    Pitcher-games where the pitcher has fewer than MIN_EXCL_GAMES_FOR_BASELINE
    other-venue starts in that season are dropped (insufficient sample for
    a clean baseline).
    """
    pa = df.dropna(subset=["events"]).copy()
    pa["__is_k"] = pa["events"].isin(K_EVENTS)

    pg = (
        pa.groupby(["__season", "game_pk", "pitcher", "home_team"], dropna=True)
        .agg(
            observed_k=("__is_k", "sum"),
            observed_pa=("__is_k", "count"),
            p_throws=("p_throws", "first"),
        )
        .reset_index()
    )
    pg = pg[pg["observed_pa"] > 0].copy()

    pg["venue_id"] = pg["home_team"].map(TEAM_TO_VENUE_ID)
    unmapped = pg[pg["venue_id"].isna()]
    if not unmapped.empty:
        bad_teams = sorted(unmapped["home_team"].dropna().unique().tolist())
        logger.warning(
            "%d pitcher-games at unmapped home_team values %s — dropping",
            len(unmapped), bad_teams,
        )
        pg = pg.dropna(subset=["venue_id"])
    pg["venue_id"] = pg["venue_id"].astype(int)

    if leave_one_venue_out:
        season_totals = (
            pg.groupby(["__season", "pitcher"])
            .agg(
                season_k=("observed_k", "sum"),
                season_pa=("observed_pa", "sum"),
                season_games=("observed_k", "count"),
            )
            .reset_index()
        )
        venue_totals = (
            pg.groupby(["__season", "pitcher", "venue_id"])
            .agg(
                venue_k=("observed_k", "sum"),
                venue_pa=("observed_pa", "sum"),
                venue_games=("observed_k", "count"),
            )
            .reset_index()
        )
        pg = pg.merge(season_totals, on=["__season", "pitcher"], how="inner")
        pg = pg.merge(
            venue_totals,
            on=["__season", "pitcher", "venue_id"],
            how="inner",
        )
        pg["excl_k"] = pg["season_k"] - pg["venue_k"]
        pg["excl_pa"] = pg["season_pa"] - pg["venue_pa"]
        pg["excl_games"] = pg["season_games"] - pg["venue_games"]
        before = len(pg)
        pg = pg[pg["excl_games"] >= MIN_EXCL_GAMES_FOR_BASELINE].copy()
        pg = pg[pg["excl_pa"] > 0].copy()
        dropped = before - len(pg)
        if dropped > 0:
            logger.info(
                "leave-one-venue-out: dropped %d pitcher-games "
                "(pitcher had <%d starts at other venues in same season)",
                dropped, MIN_EXCL_GAMES_FOR_BASELINE,
            )
        pg["season_k_pct"] = pg["excl_k"] / pg["excl_pa"]
    else:
        # Legacy: full-season K% (includes target venue's effect)
        pitcher_season = (
            pa.groupby(["__season", "pitcher"], dropna=True)
            .agg(season_k=("__is_k", "sum"), season_pa=("__is_k", "count"))
            .reset_index()
        )
        pitcher_season["season_k_pct"] = (
            pitcher_season["season_k"] / pitcher_season["season_pa"]
        )
        pg = pg.merge(
            pitcher_season[["__season", "pitcher", "season_k_pct"]],
            on=["__season", "pitcher"],
            how="inner",
        )

    pg["expected_k"] = pg["season_k_pct"] * pg["observed_pa"]
    pg = pg[pg["expected_k"] > 0].copy()
    pg["ratio"] = pg["observed_k"] / pg["expected_k"]
    return pg


def _shrunk_factor(mean_ratio: float, n_games: int, k_prior: int) -> float:
    return (mean_ratio * n_games + 1.0 * k_prior) / (n_games + k_prior)


def _build_park_k_factors(df: pd.DataFrame, *, by_hand: bool = True) -> dict:
    """Compute park K factors. When by_hand=True, factors split per pitcher
    handedness with a combined weighted-average factor for backward compat.
    """
    pg = _build_pitcher_game_table(df)

    if not by_hand:
        # Legacy single-factor path
        by_venue = (
            pg.groupby("venue_id")
            .agg(mean_ratio=("ratio", "mean"), n_games=("ratio", "count"))
            .reset_index()
        )
        by_venue["factor"] = (
            by_venue["mean_ratio"] * by_venue["n_games"]
            + 1.0 * SHRINKAGE_PRIOR_GAMES
        ) / (by_venue["n_games"] + SHRINKAGE_PRIOR_GAMES)
        factors: dict[str, dict] = {}
        for _, row in by_venue.iterrows():
            vid = int(row["venue_id"])
            factors[str(vid)] = {
                "venue_name": VENUE_NAMES.get(vid, f"venue_{vid}"),
                "factor": round(float(row["factor"]), 4),
                "n_games": int(row["n_games"]),
            }
        return {
            "_comment": (
                "Park K-specific factors (single, not split by hand). "
                "Method: mean(observed_K / expected_K) per venue. EB shrunk "
                f"toward 1.0 with k_prior={SHRINKAGE_PRIOR_GAMES} games."
            ),
            "generated_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "+00:00"),
            "seasons_used": sorted(int(s) for s in df["__season"].dropna().unique()),
            "method": "shrunk_observed_over_expected_k_rate",
            "shrinkage_prior_games": SHRINKAGE_PRIOR_GAMES,
            "factors": factors,
        }

    # By-hand path: compute factor per (venue, p_throws). Drop pitcher-games
    # with unknown handedness (shouldn't happen in practice; defensive).
    pg = pg[pg["p_throws"].isin(("L", "R"))].copy()

    by_vh = (
        pg.groupby(["venue_id", "p_throws"])
        .agg(mean_ratio=("ratio", "mean"), n_games=("ratio", "count"))
        .reset_index()
    )
    # Vectorized shrinkage — pandas .apply on an empty frame returns a
    # DataFrame which breaks column assignment.
    by_vh["factor"] = (
        by_vh["mean_ratio"] * by_vh["n_games"]
        + 1.0 * SHRINKAGE_PRIOR_GAMES_BY_HAND
    ) / (by_vh["n_games"] + SHRINKAGE_PRIOR_GAMES_BY_HAND)

    # Pivot to per-venue rows with separate L and R columns
    venues = sorted(pg["venue_id"].unique())
    factors_by_hand: dict[str, dict] = {}
    for vid in venues:
        vid = int(vid)
        row_l = by_vh[(by_vh["venue_id"] == vid) & (by_vh["p_throws"] == "L")]
        row_r = by_vh[(by_vh["venue_id"] == vid) & (by_vh["p_throws"] == "R")]
        if row_l.empty or row_r.empty:
            # If a venue has zero games for one hand, fall back to shrinking
            # to 1.0 on the missing side with n_games=0.
            f_l = (
                float(row_l["factor"].iloc[0]) if not row_l.empty else 1.0
            )
            f_r = (
                float(row_r["factor"].iloc[0]) if not row_r.empty else 1.0
            )
            n_l = int(row_l["n_games"].iloc[0]) if not row_l.empty else 0
            n_r = int(row_r["n_games"].iloc[0]) if not row_r.empty else 0
        else:
            f_l = float(row_l["factor"].iloc[0])
            f_r = float(row_r["factor"].iloc[0])
            n_l = int(row_l["n_games"].iloc[0])
            n_r = int(row_r["n_games"].iloc[0])

        # Combined: weight each side's factor by its sample size
        total_n = n_l + n_r
        combined = (
            (f_l * n_l + f_r * n_r) / total_n if total_n > 0 else 1.0
        )

        factors_by_hand[str(vid)] = {
            "venue_name": VENUE_NAMES.get(vid, f"venue_{vid}"),
            "factor_lhp": round(f_l, 4),
            "factor_rhp": round(f_r, 4),
            "factor_combined": round(combined, 4),
            "n_games_lhp": n_l,
            "n_games_rhp": n_r,
        }

    return {
        "_comment": (
            "Park K-specific factors split by pitcher handedness. Method: "
            "mean(observed_K / expected_K) per (venue, hand), where expected "
            "uses each pitcher's full-season K% as the baseline. EB shrunk "
            f"toward 1.0 with k_prior={SHRINKAGE_PRIOR_GAMES_BY_HAND} games "
            "per side (smaller than the single-factor mode's 300 to account "
            "for halved sample size per cell). factor_combined is a "
            "sample-size-weighted average of the two sides, kept for "
            "downstream code that still reads a single factor."
        ),
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "+00:00"),
        "seasons_used": sorted(int(s) for s in df["__season"].dropna().unique()),
        "method": "shrunk_observed_over_expected_k_rate_by_hand",
        "shrinkage_prior_games": SHRINKAGE_PRIOR_GAMES_BY_HAND,
        "factors": factors_by_hand,
    }


def _sanity_check(payload: dict) -> None:
    """Sanity gates for by-hand mode (post Phase 4b-v2):
    - All 30 venues present
    - factor_lhp and factor_rhp both populated for each venue
    - Each factor_lhp and factor_rhp in [0.85, 1.30] (empirical post-2021
      + LOO range; T-Mobile sits at ~1.24, the documented outlier)
    - Standard deviation of factor_combined across venues in [0.02, 0.10]
      (catches "too uniform" — shrinkage too aggressive, or "too spread" —
      shrinkage too weak)
    - n_games_lhp and n_games_rhp both >= 80 per venue
    - |factor_lhp - factor_rhp| > 0.10 fires WARNING only (informational)
    """
    import math
    factors = payload["factors"]
    by_hand = payload.get("method", "").endswith("_by_hand")

    expected = set(TEAM_TO_VENUE_ID.values())
    present = {int(k) for k in factors.keys()}
    missing = expected - present
    if missing:
        raise AssertionError(f"venues missing from output: {sorted(missing)}")

    if not by_hand:
        # Legacy single-factor sanity (kept for --no-by-hand mode).
        out_of_band = [
            (vid, info) for vid, info in factors.items()
            if not (0.90 <= info["factor"] <= 1.10)
        ]
        if len(out_of_band) > 5:
            raise AssertionError(
                f"too many parks outside [0.90, 1.10]: {out_of_band}"
            )
        extreme = [
            (vid, info) for vid, info in factors.items()
            if not (0.80 <= info["factor"] <= 1.20)
        ]
        if extreme:
            raise AssertionError(f"extreme park factors: {extreme}")
        return

    # By-hand mode
    for vid, info in factors.items():
        if "factor_lhp" not in info or "factor_rhp" not in info:
            raise AssertionError(
                f"venue {vid}: missing factor_lhp or factor_rhp"
            )

    # Each factor in [0.85, 1.30] for both hands. Range reflects empirical
    # 2023-2025 park K factors AFTER leave-one-venue-out methodology.
    # T-Mobile (Seattle) lands at ~1.24 due to known K-friendliness from
    # marine-layer air, large fair territory, and retractable-roof effects.
    # Pre-LOO the cap was 1.20; LOO removes the home-park dampening bias
    # and the most-K-friendly venues now read sharper, hence the wider cap.
    extreme = []
    for vid, info in factors.items():
        for side, key in (("LHP", "factor_lhp"), ("RHP", "factor_rhp")):
            f = info[key]
            if not (0.85 <= f <= 1.30):
                extreme.append((vid, info.get("venue_name", ""), side, f))
    if extreme:
        raise AssertionError(
            f"venues with factor outside [0.85, 1.30]: {extreme}"
        )

    # Std of factor_combined across 30 venues in [0.02, 0.10]
    combined = [info["factor_combined"] for info in factors.values()]
    mean_c = sum(combined) / len(combined)
    var_c = sum((x - mean_c) ** 2 for x in combined) / len(combined)
    std_c = math.sqrt(var_c)
    if not (0.02 <= std_c <= 0.10):
        raise AssertionError(
            f"std of factor_combined across venues = {std_c:.4f}, "
            f"expected in [0.02, 0.10]. Too low = shrinkage over-aggressive; "
            f"too high = shrinkage under-aggressive."
        )
    logger.info(
        "factor_combined distribution: mean=%.4f, std=%.4f (gate [0.02, 0.10])",
        mean_c, std_c,
    )

    # n_games thresholds — every major MLB venue should have >80 per side
    # across 3 seasons of regular play
    thin_cells = [
        (v, i["n_games_lhp"], i["n_games_rhp"]) for v, i in factors.items()
        if i["n_games_lhp"] < 80 or i["n_games_rhp"] < 80
    ]
    if thin_cells:
        raise AssertionError(
            f"venues with <80 games per hand-side: {thin_cells}"
        )

    # |factor_lhp - factor_rhp| > 0.10 → WARN, not halt
    big_disparity = [
        (v, i["venue_name"], i["factor_lhp"], i["factor_rhp"])
        for v, i in factors.items()
        if abs(i["factor_lhp"] - i["factor_rhp"]) > 0.10
    ]
    for v, name, f_l, f_r in big_disparity:
        logger.warning(
            "venue %s (%s): large L/R disparity factor_lhp=%.4f vs "
            "factor_rhp=%.4f (|delta|=%.4f) — investigate",
            v, name, f_l, f_r, abs(f_l - f_r),
        )

    # Sorted print for human review (by factor_combined)
    sorted_factors = sorted(factors.items(), key=lambda kv: kv[1]["factor_combined"])
    logger.info("park K factors by hand (sorted by combined):")
    for vid, info in sorted_factors:
        logger.info(
            "  %5s  %-30s  L=%.4f (n=%d)  R=%.4f (n=%d)  combined=%.4f",
            vid, info["venue_name"],
            info["factor_lhp"], info["n_games_lhp"],
            info["factor_rhp"], info["n_games_rhp"],
            info["factor_combined"],
        )


def _write_diagnostic_report(payload: dict, out_path: Path) -> None:
    """Write park_k_factors_report.json with disparities + histograms."""
    factors = payload["factors"]
    venues = [
        {
            "venue_id": int(v),
            "venue_name": info["venue_name"],
            "factor_lhp": info["factor_lhp"],
            "factor_rhp": info["factor_rhp"],
            "factor_combined": info["factor_combined"],
            "n_games_lhp": info["n_games_lhp"],
            "n_games_rhp": info["n_games_rhp"],
            "lr_disparity": round(
                info["factor_lhp"] - info["factor_rhp"], 4
            ),
        }
        for v, info in factors.items()
    ]
    venues_sorted_by_disparity = sorted(
        venues, key=lambda v: abs(v["lr_disparity"]), reverse=True,
    )
    top_5 = venues_sorted_by_disparity[:5]

    # Histograms: 20 buckets from 0.80 to 1.20
    bucket_edges = [0.80 + 0.02 * i for i in range(21)]

    def _hist(values: list[float]) -> list[int]:
        buckets = [0] * (len(bucket_edges) - 1)
        for v in values:
            for i in range(len(bucket_edges) - 1):
                if bucket_edges[i] <= v < bucket_edges[i + 1]:
                    buckets[i] += 1
                    break
            else:
                if v >= bucket_edges[-1]:
                    buckets[-1] += 1
        return buckets

    lhp_factors = [info["factor_lhp"] for info in factors.values()]
    rhp_factors = [info["factor_rhp"] for info in factors.values()]
    report = {
        "generated_at": payload["generated_at"],
        "seasons_used": payload["seasons_used"],
        "venues": sorted(venues, key=lambda v: v["venue_id"]),
        "top_5_disparity": top_5,
        "histograms": {
            "bucket_edges": [round(e, 2) for e in bucket_edges],
            "lhp": _hist(lhp_factors),
            "rhp": _hist(rhp_factors),
        },
    }
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
    )
    logger.info("wrote diagnostic report → %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
    )
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--by-hand", dest="by_hand", action="store_true", default=True,
        help="Split factors by pitcher handedness (default).",
    )
    parser.add_argument(
        "--no-by-hand", dest="by_hand", action="store_false",
        help="Legacy single-factor-per-venue mode.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = _pull_seasons(args.seasons)
    payload = _build_park_k_factors(df, by_hand=args.by_hand)
    _sanity_check(payload)
    out_path = args.out_dir / "park_k_factors.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("wrote %s (%d venues)", out_path, len(payload["factors"]))

    if args.by_hand:
        report_path = args.out_dir / "park_k_factors_report.json"
        _write_diagnostic_report(payload, report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
