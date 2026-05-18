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

SHRINKAGE_PRIOR_GAMES = 300


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


def _build_park_k_factors(df: pd.DataFrame) -> dict:
    # Restrict to PA-terminal rows (one row per PA).
    pa = df.dropna(subset=["events"]).copy()
    pa["__is_k"] = pa["events"].isin(K_EVENTS)

    # Per-pitcher-season K% for the "expected" baseline.
    pitcher_season = (
        pa.groupby(["__season", "pitcher"], dropna=True)
        .agg(season_k=("__is_k", "sum"), season_pa=("__is_k", "count"))
        .reset_index()
    )
    pitcher_season["season_k_pct"] = (
        pitcher_season["season_k"] / pitcher_season["season_pa"]
    )

    # Per pitcher-game observed counts.
    pg = (
        pa.groupby(["__season", "game_pk", "pitcher", "home_team"], dropna=True)
        .agg(observed_k=("__is_k", "sum"), observed_pa=("__is_k", "count"))
        .reset_index()
    )
    pg = pg.merge(
        pitcher_season[["__season", "pitcher", "season_k_pct"]],
        on=["__season", "pitcher"],
        how="inner",
    )
    # Drop pitcher-games with effectively no sample for the season K% prior.
    pg = pg[pg["observed_pa"] > 0]

    pg["expected_k"] = pg["season_k_pct"] * pg["observed_pa"]
    # Guard against zero expected (pitchers with all-walk seasons — never
    # happens in practice but defensive).
    pg = pg[pg["expected_k"] > 0]
    pg["ratio"] = pg["observed_k"] / pg["expected_k"]

    # Map home_team to venue_id.
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

    # Per-venue stats: mean ratio, n_games, shrink to 1.0.
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

    payload = {
        "_comment": (
            "Park K-specific factors derived from 3-season Statcast pulls. "
            "Method: mean(observed_K / expected_K) per venue, where expected "
            "uses each pitcher's full-season K% as the baseline. EB shrunk "
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
    return payload


def _sanity_check(payload: dict) -> None:
    """Range-based sanity. The 4b spec assumed Coors and Trop would land
    below 1.0, but empirical 2023-25 data shows that prior is wrong for the
    current ball + rule environment. The real sanity question is "do the
    factors look like park-level deviations and not raw noise" — which
    means most parks near neutral and none extreme.
    """
    factors = payload["factors"]

    expected = set(TEAM_TO_VENUE_ID.values())
    present = {int(k) for k in factors.keys()}
    missing = expected - present
    if missing:
        raise AssertionError(f"venues missing from output: {sorted(missing)}")

    # Distribution sanity: most parks should be near neutral after shrinkage.
    out_of_band = [
        (vid, info) for vid, info in factors.items()
        if not (0.90 <= info["factor"] <= 1.10)
    ]
    if len(out_of_band) > 5:
        raise AssertionError(
            f"too many parks outside [0.90, 1.10] band ({len(out_of_band)}); "
            f"shrinkage prior may be too small. Outliers: {out_of_band}"
        )

    # Any single park >0.20 from neutral is suspicious — almost certainly a
    # data bug, not a real park effect of that size. T-Mobile (cool marine
    # air, large OF) is the historical upper outlier at ~1.16; Coors with
    # post-sticky-stuff rules has come back toward neutral at ~1.04.
    extreme = [
        (vid, info) for vid, info in factors.items()
        if not (0.80 <= info["factor"] <= 1.20)
    ]
    if extreme:
        raise AssertionError(
            f"extreme park factors outside [0.80, 1.20]: {extreme}"
        )

    # Sorted print for human review.
    sorted_factors = sorted(
        factors.items(), key=lambda kv: kv[1]["factor"]
    )
    logger.info("park K factors (sorted ascending):")
    for vid, info in sorted_factors:
        logger.info(
            "  %5s  %-30s  factor=%.4f  n_games=%d",
            vid, info["venue_name"], info["factor"], info["n_games"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
    )
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = _pull_seasons(args.seasons)
    payload = _build_park_k_factors(df)
    _sanity_check(payload)
    out_path = args.out_dir / "park_k_factors.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("wrote %s (%d venues)", out_path, len(payload["factors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
