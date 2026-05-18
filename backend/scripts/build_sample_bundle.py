"""Phase 2c canonical bundle builder.

Pulls Probables, StatsAPI (lineups/weather/umpire/context), Statcast (pitcher
+ each opposing batter), and Odds for a single (date, pitcher) tuple and
merges them into the canonical JSON shape that Phase 3 feature engineering
will consume.

Usage:
    python -m scripts.build_sample_bundle --date YYYY-MM-DD --pitcher-mlbam-id NNNN

Hard rules (every rule is enforced; violation -> non-zero exit):
1. No median fill, no synthetic data.
2. No DataFrames in the output JSON; pitch-level data is list-of-dicts.
3. ISO 8601 timestamps with offset, UTC where ambiguous.
4. Field names match the contract exactly (case-sensitive).
5. If pitcher unresolvable / game not found / Statcast pull fails, raise and
   write nothing.
6. Odds-unavailable is NOT a fatal error: the market section uses the
   canonical missing-state representation.

Contract details and validation rules live in scripts/bundle_validator.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from src.data.odds_client import (
    CrossRepoKeyBleedError,
    MissingSnapshotError,
    OddsAPIClient,
    PitcherProp,
)
from src.data.probables_client import (
    ProbablePitcher,
    ProbablesClient,
    normalize_team_abbr,
)
from src.data.statcast_client import StatcastClient
from src.data.statsapi_client import (
    BatterEntry,
    GameContext,
    GameWeather,
    Lineup,
    LineupNotPostedError,
    StatsAPIClient,
    UmpireAssignment,
)

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "1.0"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

# Statcast columns we surface in the bundle. The full pybaseball response has
# ~118 columns; we keep the subset Phase 3 features will need (see
# backend/README.md feature list). Trimming brings the bundle from ~34 MB to
# ~5 MB without losing anything load-bearing. Phase 3 can request more by
# extending this list and re-running the bundle script.
BUNDLE_STATCAST_COLUMNS: tuple[str, ...] = (
    "game_pk", "game_date", "pitcher", "batter",
    "at_bat_number", "pitch_number",
    "pitch_type", "pitch_name",
    "description", "events", "type",
    "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_speed", "release_spin_rate", "release_pos_x", "release_pos_z",
    "stand", "p_throws",
    "balls", "strikes", "outs_when_up", "inning",
    "bb_type", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle",
)

# Domes / retractable roofs likely indoor. Phase 3 will replace this with a
# proper park-metadata lookup; for the Phase 2c sample we just flag the four
# always-indoor venues by StatsAPI venue_id.
DOME_VENUE_IDS = {12, 14, 2392, 4705, 5000}  # Trop, ATL/AAS, MIA, MIN, ARI (approx)


# -------- Helpers ------------------------------------------------------------


def df_to_records(df, columns: Iterable[str] = BUNDLE_STATCAST_COLUMNS) -> list[dict]:
    """Convert a pandas DataFrame to a JSON-clean list of dicts.

    - Columns trimmed to ``columns`` (BUNDLE_STATCAST_COLUMNS by default).
      Missing columns are emitted as ``None``.
    - NaN -> None
    - Timestamps -> ISO strings (date-only for game_date, full ISO for datetimes)
    - Numpy scalar types -> native Python types
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    col_list = list(columns)
    available = [c for c in col_list if c in df.columns]
    missing = [c for c in col_list if c not in df.columns]
    sub = df[available]
    records: list[dict] = []
    for row in sub.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for col, val in row.items():
            clean[col] = _scrub_value(val)
        for m in missing:
            clean[m] = None
        records.append(clean)
    return records


def _scrub_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float):
        return None if math.isnan(val) else val
    if isinstance(val, (int, str, bool)):
        return val
    # pandas / numpy scalars
    if hasattr(val, "item"):
        try:
            unwrapped = val.item()
            if isinstance(unwrapped, float) and math.isnan(unwrapped):
                return None
            return unwrapped
        except (ValueError, AttributeError):
            pass
    # Timestamps / datetimes / dates
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def iso_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "+00:00")
    )


def normalize_first_pitch(ctx_iso: str | None) -> str | None:
    """StatsAPI returns Zulu-suffix datetimes. Convert to +00:00 form."""
    if not ctx_iso:
        return None
    if ctx_iso.endswith("Z"):
        return ctx_iso[:-1] + "+00:00"
    return ctx_iso


def find_game_pk_and_side(
    target_date: date, pitcher_id: int
) -> tuple[int, bool, str, str]:
    """Look up game_pk and home/away side for the pitcher's game.

    Returns (game_pk, is_home, team_abbr, opp_team_abbr). Uses StatsAPI
    schedule with the probablePitcher+team hydrate.
    """
    params = {
        "sportId": 1,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher,team",
    }
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule", params=params, timeout=30
    )
    resp.raise_for_status()
    sched = resp.json()
    for d in sched.get("dates", []):
        if d.get("date") != target_date.isoformat():
            continue
        for game in d.get("games", []):
            for side in ("home", "away"):
                pp = ((game.get("teams") or {}).get(side) or {}).get("probablePitcher")
                if pp and pp.get("id") == pitcher_id:
                    home_team = normalize_team_abbr(
                        ((game["teams"]["home"] or {}).get("team") or {}).get("abbreviation")
                        or ""
                    )
                    away_team = normalize_team_abbr(
                        ((game["teams"]["away"] or {}).get("team") or {}).get("abbreviation")
                        or ""
                    )
                    team_abbr = home_team if side == "home" else away_team
                    opp_abbr = away_team if side == "home" else home_team
                    return int(game["gamePk"]), side == "home", team_abbr, opp_abbr
    raise RuntimeError(
        f"build_sample_bundle: pitcher {pitcher_id} is not listed as a probable "
        f"for any game on {target_date}"
    )


def compute_days_rest(pitches_30d_records: list[dict], game_date: date) -> int | None:
    """Days between game_date and the pitcher's most recent appearance.

    Looks at game_date column of the 30-day pitcher Statcast. Returns None
    when no prior appearance is in the window (rookie, return from IL, etc.)
    — no median fill.
    """
    prev_dates: list[date] = []
    for row in pitches_30d_records:
        gd = row.get("game_date")
        if isinstance(gd, str) and len(gd) >= 10:
            try:
                d = date.fromisoformat(gd[:10])
            except ValueError:
                continue
            if d < game_date:
                prev_dates.append(d)
    if not prev_dates:
        return None
    return (game_date - max(prev_dates)).days


def opener_detection_result(probable: ProbablePitcher) -> str:
    """Phase 2c proxy: use the FG opener flag.

    Full opener_detection (which requires season-IP / rotation-slot / market
    inputs) is wired in Phase 3. For the sample bundle, ``bulk_pitcher`` is
    the right answer whenever FG flagged opener+primary; otherwise
    ``no_override``.
    """
    return "bulk_pitcher" if probable.opener_flagged else "no_override"


# -------- Section builders ---------------------------------------------------


def build_metadata(
    *,
    game_date: date,
    cutoff_date: date,
    pitcher_id: int,
    pitcher_name: str,
    game_pk: int,
) -> dict:
    return {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": iso_utc_now(),
        "game_date": game_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "pitcher_mlbam_id": int(pitcher_id),
        "pitcher_name": pitcher_name,
        "game_pk": int(game_pk),
    }


def build_pitcher_section(
    *,
    probable: ProbablePitcher,
    team_abbr: str,
    pitches_30d: list[dict],
    pitches_season: list[dict],
    pitches_prior_year: list[dict],
) -> dict:
    return {
        "mlbam_id": probable.pitcher_mlbam_id,
        "name": probable.pitcher_name,
        "handedness": probable.pitcher_hand,
        "team_abbr": team_abbr,
        "source": probable.source,
        "fg_opener_flag": bool(probable.opener_flagged),
        "fg_primary_pitcher_flag": bool(probable.opener_flagged),
        "opener_detection_result": opener_detection_result(probable),
        "statcast_pitches_30d": pitches_30d,
        "statcast_pitches_season": pitches_season,
        "statcast_pitches_prior_year": pitches_prior_year,
    }


def build_opposing_lineup_section(
    *,
    opp_team_abbr: str,
    lineup: Lineup | None,
    batter_pa_records: list[tuple[BatterEntry, list[dict]]],
) -> dict:
    if lineup is None or not lineup.batters:
        return {
            "team_abbr": opp_team_abbr,
            "batters": [],
            "lineup_posted": False,
        }
    batters: list[dict] = []
    for batter, pa_records in batter_pa_records:
        batters.append(
            {
                "mlbam_id": batter.batter_mlbam_id,
                "name": batter.name,
                "batting_order": batter.batting_order_spot,
                "handedness": batter.bats,
                "position": batter.position_code,
                "statcast_pa_season": pa_records,
            }
        )
    return {
        "team_abbr": opp_team_abbr,
        "batters": batters,
        "lineup_posted": True,
    }


def build_game_context_section(
    *,
    context: GameContext,
    weather: GameWeather,
    umpire: UmpireAssignment,
    days_rest: int | None,
) -> dict:
    is_dome = (
        context.venue_id in DOME_VENUE_IDS
        or (weather.condition or "").lower() == "dome"
    )
    return {
        "venue_id": context.venue_id,
        "venue_name": context.venue_name,
        "is_dome": bool(is_dome),
        "weather": {
            "temp_f": weather.temp_f,
            "wind_speed_mph": weather.wind_speed_mph,
            "wind_direction": weather.wind_direction,
            # StatsAPI doesn't surface humidity; canonical empty -> null.
            "humidity_pct": None,
            "conditions": weather.condition,
        },
        "umpire_name": umpire.home_plate_umpire_name,
        "umpire_id": umpire.home_plate_umpire_id,
        "first_pitch_iso": normalize_first_pitch(context.game_datetime_iso),
        "days_rest": days_rest,
    }


def build_market_section(*, game_date: date, game_pk: int) -> dict:
    """Try live odds, fall back to committed snapshot, else canonical missing.

    Order:
    1. Live mode via OddsAPIClient.fetch(cutoff_date=game_date).
    2. Historical mode (committed snapshot for game_date).
    3. Canonical missing state.
    """
    empty_market = {
        "fanduel": {"available": False, "lines": []},
        "draftkings": {"available": False, "lines": []},
        "snapshot_timestamp": None,
        "snapshot_source": "missing",
    }
    # Try live mode.
    try:
        client = OddsAPIClient()
        events = client.fetch(cutoff_date=game_date)
        return _select_event_for_game(events, game_pk, source="live")
    except CrossRepoKeyBleedError as exc:
        logger.warning("market: live mode disabled (%s); trying committed snapshot", exc)
    except Exception as exc:
        logger.warning("market: live mode failed (%s); trying committed snapshot", exc)

    # Try historical snapshot.
    try:
        client = OddsAPIClient(api_key="placeholder-for-historical")
    except CrossRepoKeyBleedError:
        client = None
    if client is not None:
        try:
            events = client._read_historical(game_date)
            return _select_event_for_game(events, game_pk, source="committed")
        except MissingSnapshotError:
            logger.warning("market: no committed snapshot for %s", game_date)
        except Exception as exc:
            logger.warning("market: snapshot read failed (%s)", exc)

    return empty_market


def _select_event_for_game(events, game_pk: int, *, source: str) -> dict:
    """Build the market dict for the chosen game from a list of EventOdds.

    The Odds API's event_id and StatsAPI's game_pk don't share an ID space;
    we match by commence_time + team names if possible, otherwise we just
    take the first event whose home/away teams match the schedule we used.
    For Phase 2c, we don't have schedule team-names handy here; treat the
    event list as unfiltered and emit canonical missing if nothing matches.
    """
    if not events:
        return {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None,
            "snapshot_source": "missing",
        }
    # Without a robust event->game_pk mapping (Phase 7 problem), we
    # surface the first event. The bundle's metadata.game_pk records the
    # canonical game; the market section's commence_time records what we
    # actually pulled. If they don't match the caller will see it.
    event = events[0]
    by_book = {"fanduel": [], "draftkings": []}
    for prop in event.pitcher_props:
        if prop.book in by_book:
            for ln in prop.lines:
                by_book[prop.book].append(
                    {"line": ln.line, "side": ln.side, "price": int(ln.price)}
                )
    return {
        "fanduel": {
            "available": bool(by_book["fanduel"]),
            "lines": by_book["fanduel"],
        },
        "draftkings": {
            "available": bool(by_book["draftkings"]),
            "lines": by_book["draftkings"],
        },
        "snapshot_timestamp": event.commence_time or None,
        "snapshot_source": source,
    }


# -------- Orchestration ------------------------------------------------------


def build_bundle(target_date: date, pitcher_id: int) -> dict:
    cutoff_date = target_date - timedelta(days=1)

    # 1. Discover the game.
    game_pk, is_home, team_abbr, opp_team_abbr = find_game_pk_and_side(
        target_date, pitcher_id
    )
    logger.info(
        "game_pk=%d %s @ %s (pitcher's team: %s)",
        game_pk, team_abbr if not is_home else opp_team_abbr, team_abbr if is_home else opp_team_abbr,
        team_abbr,
    )

    # 2. Probables (Phase 2a). Match the pitcher.
    probables_client = ProbablesClient()
    probables = probables_client.fetch(cutoff_date=target_date)
    probable = next(
        (p for p in probables if p.pitcher_mlbam_id == pitcher_id), None
    )
    if probable is None:
        raise RuntimeError(
            f"build_sample_bundle: pitcher {pitcher_id} not found in FanGraphs/StatsAPI "
            f"probables for {target_date}. Try a different (date, pitcher) pair."
        )

    # 3. StatsAPI sections.
    statsapi = StatsAPIClient()
    try:
        home_lineup, away_lineup = statsapi.get_lineups(game_pk, cutoff_date=target_date)
    except LineupNotPostedError:
        home_lineup = None
        away_lineup = None
    weather = statsapi.get_weather(game_pk, cutoff_date=target_date)
    umpire = statsapi.get_umpire(game_pk, cutoff_date=target_date)
    context = statsapi.get_game_context(game_pk, cutoff_date=target_date)
    opp_lineup = (
        away_lineup if is_home else home_lineup
    )

    # 4. Statcast for pitcher (3 windows).
    statcast = StatcastClient()
    logger.info("pulling pitcher 30d statcast (cutoff=%s)", cutoff_date)
    pitches_30d_df = statcast.get_pitcher_pitches(
        pitcher_id, cutoff_date=cutoff_date, days_back=30
    )
    logger.info("pulling pitcher season-to-date statcast")
    pitches_season_df = statcast.get_pitcher_pitches(
        pitcher_id, cutoff_date=cutoff_date, days_back=60
    )
    prior_year_cutoff = date(target_date.year - 1, 10, 31)
    logger.info("pulling pitcher prior-year (%s) statcast", prior_year_cutoff)
    pitches_prior_df = statcast.get_pitcher_pitches(
        pitcher_id, cutoff_date=prior_year_cutoff, days_back=220
    )

    pitches_30d = df_to_records(pitches_30d_df)
    pitches_season = df_to_records(pitches_season_df)
    pitches_prior_year = df_to_records(pitches_prior_df)

    # 5. Statcast for each opposing batter (season-to-date).
    batter_pa_records: list[tuple[BatterEntry, list[dict]]] = []
    if opp_lineup is not None:
        for batter in opp_lineup.batters:
            logger.info(
                "pulling batter %s (%d) season-to-date statcast",
                batter.name, batter.batter_mlbam_id,
            )
            try:
                pa_df = statcast.get_batter_pitches(
                    batter.batter_mlbam_id, cutoff_date=cutoff_date, days_back=60
                )
            except Exception as exc:
                raise RuntimeError(
                    f"build_sample_bundle: statcast pull failed for batter "
                    f"{batter.name} ({batter.batter_mlbam_id}): {exc}"
                ) from exc
            batter_pa_records.append((batter, df_to_records(pa_df)))

    # 6. Days rest from pitcher's recent appearances.
    days_rest = compute_days_rest(pitches_30d, target_date)

    # 7. Market (odds).
    market = build_market_section(game_date=target_date, game_pk=game_pk)

    # 8. Assemble bundle.
    bundle = {
        "metadata": build_metadata(
            game_date=target_date,
            cutoff_date=cutoff_date,
            pitcher_id=pitcher_id,
            pitcher_name=probable.pitcher_name,
            game_pk=game_pk,
        ),
        "pitcher": build_pitcher_section(
            probable=probable,
            team_abbr=team_abbr,
            pitches_30d=pitches_30d,
            pitches_season=pitches_season,
            pitches_prior_year=pitches_prior_year,
        ),
        "opposing_lineup": build_opposing_lineup_section(
            opp_team_abbr=opp_team_abbr,
            lineup=opp_lineup,
            batter_pa_records=batter_pa_records,
        ),
        "game_context": build_game_context_section(
            context=context,
            weather=weather,
            umpire=umpire,
            days_rest=days_rest,
        ),
        "market": market,
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument(
        "--pitcher-mlbam-id",
        required=True,
        type=int,
        help="MLBAM player id of the starting pitcher to bundle",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: data/phase2c_sample_bundle_{date}_{id}.json)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write the bundle even if it fails the contract check (debug only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    target_date = date.fromisoformat(args.date)
    pitcher_id = int(args.pitcher_mlbam_id)

    bundle = build_bundle(target_date, pitcher_id)

    from scripts.bundle_validator import validate

    issues = validate(bundle)
    if issues:
        print("VALIDATION FAILED:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        if not args.skip_validation:
            print(
                "Refusing to write bundle. Use --skip-validation to write anyway "
                "(debug only).",
                file=sys.stderr,
            )
            return 2

    out_path = (
        Path(args.out)
        if args.out
        else OUT_DIR / f"phase2c_sample_bundle_{target_date}_{pitcher_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"wrote bundle -> {out_path}")
    print(
        f"  pitcher: {bundle['pitcher']['name']} ({pitcher_id})  "
        f"team: {bundle['pitcher']['team_abbr']}  hand: {bundle['pitcher']['handedness']}"
    )
    print(
        f"  pitches 30d: {len(bundle['pitcher']['statcast_pitches_30d'])}  "
        f"season: {len(bundle['pitcher']['statcast_pitches_season'])}  "
        f"prior_yr: {len(bundle['pitcher']['statcast_pitches_prior_year'])}"
    )
    print(
        f"  opposing lineup posted: {bundle['opposing_lineup']['lineup_posted']}  "
        f"batters: {len(bundle['opposing_lineup']['batters'])}"
    )
    print(
        f"  game: {bundle['game_context']['venue_name']}  "
        f"ump: {bundle['game_context']['umpire_name']}  "
        f"days_rest: {bundle['game_context']['days_rest']}"
    )
    print(f"  market source: {bundle['market']['snapshot_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
