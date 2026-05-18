"""Phase 4a: derive leaguewide rate splits by (batter_hand, pitcher_hand).

Pulls every regular-season Statcast pitch for the requested season(s) and
computes splits for RR / RL / LR / LL / SR / SL plus an overall ``ALL`` line.

Hard rules applied:
- ``game_type == 'R'`` filter (regular season only).
- No imputation. Pitches with missing handedness or zone are dropped at the
  source for that metric only (a pitch missing ``stand`` still contributes
  to chase rate if ``zone`` and ``description`` are present, just not to
  any handedness split).
- Sanity assertions halt the script. Halting is intentional — a leagueK%
  outside [0.21, 0.24] means the upstream feed is broken, not that we should
  ship a bad number.

Output: ``backend/data/processed/league_averages_{season}.json``. The 2025
file overwrites the Phase 3a placeholder.

CLI:
    python -m scripts.derive_league_averages              # both 2024 and 2025
    python -m scripts.derive_league_averages --season 2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Statcast event taxonomy
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
WALK_EVENTS = frozenset({"walk", "intent_walk"})
HBP_EVENTS = frozenset({"hit_by_pitch"})
SF_EVENTS = frozenset({"sac_fly", "sac_fly_double_play"})
SH_EVENTS = frozenset({"sac_bunt", "sac_bunt_double_play"})

# Pitch description taxonomy
SWING_DESCRIPTIONS = frozenset({
    "foul", "foul_bunt", "foul_pitchout", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "swinging_strike", "swinging_strike_blocked",
    "missed_bunt",
})
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked"})

IN_ZONE_ZONES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
OUT_OF_ZONE_ZONES = frozenset({11, 12, 13, 14})

# Sanity ranges
LEAGUE_K_PCT_RANGE = (0.21, 0.24)
# MLB handedness distribution at ~160k PA/season: RR ~38%, LR ~30%, RL ~22%,
# LL ~10%. Note: Statcast's `stand` field always carries the live batting
# side (a switch-hitter LHB facing a RHP shows up as "L", not "S"). So
# switch hitters are already correctly distributed across the four R/L
# splits — there is no separate "SR"/"SL" split to compute from Statcast.
# LeagueAverages.lookup() handles the S->opposite-of-pitcher resolution at
# query time.
MIN_PA_COMMON_SPLIT = 30_000   # RR, LR
MIN_PA_RARER_SPLIT = 12_000    # RL, LL


def _pull_season(season: int) -> pd.DataFrame:
    """Pull a full regular season of Statcast pitches.

    Pybaseball chunks internally and caches per-day, so the first pull is
    slow (~15 min for a full season) and re-runs are near-instant.
    """
    from pybaseball import statcast  # type: ignore

    start = f"{season}-03-15"  # covers spring-into-regular transition
    end = f"{season}-11-30"    # covers postseason; we filter to game_type='R'
    logger.info("pulling Statcast %s..%s (cold pull is ~15min)", start, end)
    df = statcast(start_dt=start, end_dt=end)
    df = df[df["game_type"] == "R"].copy()
    logger.info("season %d: %d regular-season pitches", season, len(df))
    return df


def _pa_terminal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per PA — the row carrying the terminal ``events`` value.

    Statcast emits one row per pitch; the terminal pitch of a PA is the one
    with a non-null ``events`` value.
    """
    return df.dropna(subset=["events"]).copy()


def _split_label(stand: str, p_throws: str) -> str | None:
    """Return ``RR``/``RL``/``LR``/``LL``, or None for unrecognized hands.

    Switch hitters surface in ``stand`` as their live side (L vs RHP,
    R vs LHP) and naturally land in the LR/RL buckets.
    """
    if stand not in ("L", "R"):
        return None
    if p_throws not in ("L", "R"):
        return None
    return f"{stand}{p_throws}"


def _compute_split(rows_pa: pd.DataFrame, rows_pitch: pd.DataFrame) -> dict:
    """Compute the four rate stats + n_pa for one (batter, pitcher) split."""
    n_pa = len(rows_pa)
    if n_pa == 0:
        return {
            "k_pct": None, "obp": None, "zone_contact_pct": None,
            "chase_rate": None, "n_pa": 0,
        }

    events = rows_pa["events"]
    k = events.isin(K_EVENTS).sum()
    hits = events.isin(HIT_EVENTS).sum()
    walks = events.isin(WALK_EVENTS).sum()
    hbps = events.isin(HBP_EVENTS).sum()
    sh = events.isin(SH_EVENTS).sum()
    sf = events.isin(SF_EVENTS).sum()

    k_pct = float(k) / n_pa
    # OBP per MLB rule: (H + BB + HBP) / (AB + BB + HBP + SF).
    # AB = PA - BB - HBP - SF - SH, so AB + BB + HBP + SF = PA - SH.
    obp_denom = n_pa - sh
    obp = float(hits + walks + hbps) / obp_denom if obp_denom > 0 else None

    # Zone-contact and chase use ALL pitches in this split (not just PA-terminal)
    in_zone = rows_pitch["zone"].isin(IN_ZONE_ZONES)
    ooz = rows_pitch["zone"].isin(OUT_OF_ZONE_ZONES)
    swung = rows_pitch["description"].isin(SWING_DESCRIPTIONS)
    whiffed = rows_pitch["description"].isin(WHIFF_DESCRIPTIONS)

    in_zone_swings = (in_zone & swung).sum()
    in_zone_contact = (in_zone & swung & ~whiffed).sum()
    ooz_pitches = ooz.sum()
    ooz_swings = (ooz & swung).sum()

    zone_contact_pct = (
        float(in_zone_contact) / in_zone_swings if in_zone_swings > 0 else None
    )
    chase_rate = (
        float(ooz_swings) / ooz_pitches if ooz_pitches > 0 else None
    )

    return {
        "k_pct": round(k_pct, 4) if k_pct is not None else None,
        "obp": round(obp, 4) if obp is not None else None,
        "zone_contact_pct": (
            round(zone_contact_pct, 4) if zone_contact_pct is not None else None
        ),
        "chase_rate": round(chase_rate, 4) if chase_rate is not None else None,
        "n_pa": int(n_pa),
    }


def _build_payload(season: int, df: pd.DataFrame) -> dict:
    """Compose the JSON output for one season."""
    pa_rows = _pa_terminal_rows(df)
    # Attach split label to both pitch- and PA-level frames
    df = df.copy()
    pa_rows = pa_rows.copy()
    df["__split"] = [
        _split_label(s, p) for s, p in zip(df["stand"], df["p_throws"])
    ]
    pa_rows["__split"] = [
        _split_label(s, p) for s, p in zip(pa_rows["stand"], pa_rows["p_throws"])
    ]
    valid_pa = pa_rows.dropna(subset=["__split"])
    valid_pitch = df.dropna(subset=["__split"])

    splits: dict[str, dict] = {}
    for key in ("RR", "RL", "LR", "LL"):
        splits[key] = _compute_split(
            valid_pa[valid_pa["__split"] == key],
            valid_pitch[valid_pitch["__split"] == key],
        )

    overall = _compute_split(valid_pa, valid_pitch)

    return {
        "_note": (
            "Statcast 'stand' always reflects the live batting side. "
            "Switch hitters facing RHP show up as L (in LR); facing LHP as "
            "R (in RL). LeagueAverages.lookup() applies the same resolution "
            "for queries on switch hitters at projection time."
        ),
        "season": season,
        "generated_at": (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "+00:00")
        ),
        "n_pa": int(len(valid_pa)),
        "splits": splits,
        "all": overall,
    }


def _sanity_check(payload: dict) -> None:
    """Halt the script on any sanity violation (intentional: bad data = bad model)."""
    season = payload["season"]
    overall_k = payload["all"]["k_pct"]
    if overall_k is None:
        raise AssertionError(f"season {season}: overall K% is None")
    if not (LEAGUE_K_PCT_RANGE[0] <= overall_k <= LEAGUE_K_PCT_RANGE[1]):
        raise AssertionError(
            f"season {season}: overall K% {overall_k} outside expected range "
            f"{LEAGUE_K_PCT_RANGE}. Halting — upstream feed may be broken."
        )

    rr = payload["splits"]["RR"]["k_pct"]
    lr = payload["splits"]["LR"]["k_pct"]
    if rr is None or lr is None:
        raise AssertionError(f"season {season}: RR or LR K% is None")
    if not (rr > lr):
        raise AssertionError(
            f"season {season}: RR K% ({rr}) is not greater than LR K% ({lr}). "
            f"Same-hand matchups should produce more Ks — investigate."
        )

    # n_pa thresholds — reflect MLB hand distribution
    for split, mn in (
        ("RR", MIN_PA_COMMON_SPLIT), ("LR", MIN_PA_COMMON_SPLIT),
        ("RL", MIN_PA_RARER_SPLIT), ("LL", MIN_PA_RARER_SPLIT),
    ):
        n = payload["splits"][split]["n_pa"]
        if n < mn:
            raise AssertionError(
                f"season {season}: split {split} has only {n} PA, "
                f"expected >= {mn}"
            )

    logger.info(
        "season %d sanity checks pass: overall K%%=%.4f, RR=%.4f, LR=%.4f",
        season, overall_k, rr, lr,
    )


def derive_season(season: int) -> dict:
    df = _pull_season(season)
    payload = _build_payload(season, df)
    _sanity_check(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, action="append", default=None,
        help="Season(s) to derive. Default: 2024 and 2025.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROCESSED_DIR,
        help="Where to write league_averages_{season}.json",
    )
    args = parser.parse_args()
    seasons = args.season or [2024, 2025]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        logger.info("=== season %d ===", season)
        payload = derive_season(season)
        out_path = args.out_dir / f"league_averages_{season}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        logger.info("wrote %s", out_path)

    # If 2025 was derived, overwrite the Phase 3a placeholder (which sits at
    # the same path — replacing the placeholder is the intent of Phase 4a).
    return 0


if __name__ == "__main__":
    sys.exit(main())
