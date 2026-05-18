"""Historical bundle builder for the Phase 4c fit.

Builds :class:`ProjectionBundle` objects from bulk-cached Statcast data,
calling the SAME production feature builders downstream so fit-time and
projection-time feature semantics can't diverge.

Two caches are constructed once and reused per-game:

- ``PitcherCache``: per pitcher_id, two DataFrames — full pitch-level rows
  (for CSW/chase/velocity/putaway features) and PA-terminal rows (for K%,
  pitches/PA). Each sorted by game_date.
- ``BatterCache``: per batter_id, full pitch-level rows seen as the batter
  (for lineup zone-contact/chase rate features).

Per-game bundle construction is then ~O(1) DataFrame slices + dataclass
construction, well under 100ms each — making the 5000-game overnight rerun
tractable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from src.projection.inputs import (
    BatterInputs,
    BookMarket,
    BundleMetadata,
    GameContext,
    Market,
    OpposingLineup,
    PitcherInputs,
    ProjectionBundle,
    WeatherInputs,
)

logger = logging.getLogger(__name__)


# Spring training cutoff per season. We won't include spring games in the
# "season-to-date" slice; opening day in March/April is the boundary.
def _season_opening(year: int) -> date:
    return date(year, 3, 27)


def _season_end(year: int) -> date:
    return date(year, 11, 30)


@dataclass
class PitcherCache:
    """Per-pitcher Statcast data. Keys are pitcher_id."""

    # pitcher_id -> full pitch-level DataFrame (sorted by game_date)
    pitches: dict[int, pd.DataFrame]
    # pitcher_id -> PA-terminal-only DataFrame (sorted by game_date)
    pa_terminal: dict[int, pd.DataFrame]


@dataclass
class BatterCache:
    """Per-batter Statcast data. Keys are batter_id."""

    # batter_id -> full pitch-level DataFrame seen by this batter (sorted by date)
    pas: dict[int, pd.DataFrame]


def build_pitcher_cache(pitches_all: pd.DataFrame) -> PitcherCache:
    """Group bulk Statcast by pitcher; build per-pitcher dictionaries."""
    pitches_all = pitches_all.copy()
    # Ensure game_date is sortable
    pitches_all["game_date"] = pitches_all["game_date"].astype(str).str[:10]

    pitches: dict[int, pd.DataFrame] = {}
    pa_terminal: dict[int, pd.DataFrame] = {}
    for pid, g in pitches_all.groupby("pitcher", dropna=True):
        g_sorted = g.sort_values("game_date")
        pitches[int(pid)] = g_sorted
        pa_terminal[int(pid)] = g_sorted.dropna(subset=["events"])
    return PitcherCache(pitches=pitches, pa_terminal=pa_terminal)


def build_batter_cache(pitches_all: pd.DataFrame) -> BatterCache:
    """Group bulk Statcast by batter."""
    pitches_all = pitches_all.copy()
    pitches_all["game_date"] = pitches_all["game_date"].astype(str).str[:10]
    pas: dict[int, pd.DataFrame] = {}
    for bid, g in pitches_all.groupby("batter", dropna=True):
        pas[int(bid)] = g.sort_values("game_date")
    return BatterCache(pas=pas)


def _slice_by_date(df: pd.DataFrame, start_iso: str, end_iso: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    return df[(df["game_date"] >= start_iso) & (df["game_date"] <= end_iso)]


def _rows_to_dicts(df: pd.DataFrame, columns: Iterable[str]) -> tuple[dict, ...]:
    """Convert a DataFrame slice to a frozen tuple of dicts, trimmed to columns."""
    if df is None or len(df) == 0:
        return ()
    keep = [c for c in columns if c in df.columns]
    sub = df[keep]
    return tuple(sub.to_dict(orient="records"))


# Columns we keep in each bundle's statcast lists. Matches the Phase 2c
# trim — Phase 3 feature builders only need these.
PITCH_LEVEL_COLUMNS = (
    "game_pk", "game_date", "pitcher", "batter",
    "at_bat_number", "pitch_number",
    "pitch_type", "pitch_name",
    "description", "events", "type",
    "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_speed", "release_spin_rate",
    "stand", "p_throws", "strikes",
)


@dataclass
class GameRecord:
    """Aggregated info needed to build a bundle for one starter-game."""

    season: int
    game_pk: int
    game_date: date
    pitcher_id: int
    pitcher_hand: str
    pitcher_team: str
    opposing_team: str
    venue_id: int
    is_home: bool
    # batter_id -> (handedness, batting_order)
    opposing_batters: dict[int, tuple[str | None, int]]
    # Observed for regression target
    observed_bf: int
    observed_k: int


def build_bundle(
    game: GameRecord,
    pitcher_cache: PitcherCache,
    batter_cache: BatterCache,
) -> ProjectionBundle:
    """Construct a ProjectionBundle for one historical game.

    Pitcher's 30d window: [game_date - 30, game_date - 1].
    Pitcher's season window: [opening_day(game.year), game_date - 1].
    Pitcher's prior-year window: [opening_day(year-1), season_end(year-1)].
    Each opposing batter's season window: [opening_day(game.year), game_date - 1].
    """
    cutoff = game.game_date - timedelta(days=1)
    cutoff_iso = cutoff.isoformat()
    season_start = _season_opening(game.game_date.year).isoformat()
    window_30d_start = (game.game_date - timedelta(days=30)).isoformat()
    prior_start = _season_opening(game.game_date.year - 1).isoformat()
    prior_end = _season_end(game.game_date.year - 1).isoformat()

    p_pitches = pitcher_cache.pitches.get(game.pitcher_id, pd.DataFrame())
    p30 = _slice_by_date(p_pitches, window_30d_start, cutoff_iso)
    p_season = _slice_by_date(p_pitches, season_start, cutoff_iso)
    p_prior = _slice_by_date(p_pitches, prior_start, prior_end)

    batters: list[BatterInputs] = []
    sorted_batters = sorted(
        game.opposing_batters.items(),
        key=lambda kv: kv[1][1],  # batting_order
    )
    for bid, (hand, order) in sorted_batters[:9]:
        b_pas = batter_cache.pas.get(int(bid), pd.DataFrame())
        b_season = _slice_by_date(b_pas, season_start, cutoff_iso)
        batters.append(BatterInputs(
            mlbam_id=int(bid),
            name=f"id_{bid}",  # historical fit doesn't need names
            batting_order=int(order),
            handedness=hand if hand in ("L", "R", "S") else None,
            position=None,
            statcast_pa_season=_rows_to_dicts(b_season, PITCH_LEVEL_COLUMNS),
        ))

    return ProjectionBundle(
        metadata=BundleMetadata(
            bundle_version="1.0",
            generated_at="historical-fit",
            game_date=game.game_date,
            cutoff_date=cutoff,
            pitcher_mlbam_id=game.pitcher_id,
            pitcher_name=f"id_{game.pitcher_id}",
            game_pk=game.game_pk,
        ),
        pitcher=PitcherInputs(
            mlbam_id=game.pitcher_id,
            name=f"id_{game.pitcher_id}",
            handedness=game.pitcher_hand,
            team_abbr=game.pitcher_team,
            source="historical-actual",
            fg_opener_flag=False,
            fg_primary_pitcher_flag=False,
            opener_detection_result="no_override",
            statcast_pitches_30d=_rows_to_dicts(p30, PITCH_LEVEL_COLUMNS),
            statcast_pitches_season=_rows_to_dicts(p_season, PITCH_LEVEL_COLUMNS),
            statcast_pitches_prior_year=_rows_to_dicts(p_prior, PITCH_LEVEL_COLUMNS),
        ),
        opposing_lineup=OpposingLineup(
            team_abbr=game.opposing_team,
            batters=tuple(batters),
            lineup_posted=True,
        ),
        game_context=GameContext(
            venue_id=game.venue_id,
            venue_name=None,
            is_dome=False,  # we don't know without boxscore; defaults to outdoor
            weather=WeatherInputs(
                temp_f=None, wind_speed_mph=None, wind_direction=None,
                humidity_pct=None, conditions=None,
            ),
            umpire_name=None, umpire_id=None,
            first_pitch_iso=None,
            days_rest=None,  # computed downstream from pitches_30d if needed
        ),
        market=Market(
            fanduel=BookMarket(available=False, lines=()),
            draftkings=BookMarket(available=False, lines=()),
            snapshot_timestamp=None,
            snapshot_source="missing",
        ),
    )
