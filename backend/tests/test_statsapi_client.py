"""Tests for StatsAPIClient (lineups, weather, umpire, game context)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data.as_of_context import LeakageError
from src.data.statsapi_client import (
    BatterEntry,
    GameContext,
    GameWeather,
    Lineup,
    LineupNotPostedError,
    StatsAPIClient,
    UmpireAssignment,
    parse_game_context,
    parse_lineup,
    parse_umpire,
    parse_weather,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "statsapi"


@pytest.fixture
def live_feed() -> dict:
    return json.loads(
        (FIXTURES_DIR / "feed_live_822982.json").read_text(encoding="utf-8")
    )


# -------- Parser tests ------------------------------------------------------


def test_parse_lineup_home(live_feed):
    lineup = parse_lineup(live_feed, is_home=True)
    assert lineup is not None
    assert lineup.is_home is True
    assert lineup.team_abbr == "TB"
    assert lineup.game_pk == 822982
    # 9 batters, ordered 1..9
    assert len(lineup.batters) == 9
    spots = [b.batting_order_spot for b in lineup.batters]
    assert spots == list(range(1, 10))
    # First-spot batter has handedness resolved from gameData.players
    first = lineup.batters[0]
    assert first.bats in {"L", "R", "S"}
    assert first.batter_mlbam_id > 0
    assert first.name


def test_parse_lineup_away(live_feed):
    lineup = parse_lineup(live_feed, is_home=False)
    assert lineup is not None
    assert lineup.is_home is False
    assert len(lineup.batters) == 9


def test_parse_lineup_missing_returns_none():
    """An empty boxscore (game not yet posted) yields None for the parser."""
    feed = {
        "gamePk": 999,
        "gameData": {},
        "liveData": {"boxscore": {"teams": {"home": {"battingOrder": []}, "away": {"battingOrder": []}}}},
    }
    assert parse_lineup(feed, is_home=True) is None
    assert parse_lineup(feed, is_home=False) is None


def test_parse_weather_dome(live_feed):
    weather = parse_weather(live_feed)
    assert weather.condition == "Dome"
    assert weather.temp_f == 72.0
    # Dome: "0 mph, None" parses to speed=0, direction=None.
    assert weather.wind_speed_mph == 0.0
    assert weather.wind_direction is None


def test_parse_weather_outdoor_with_wind():
    feed = {
        "gamePk": 100,
        "gameData": {"weather": {"condition": "Sunny", "temp": "78", "wind": "12 mph, Out To LF"}},
    }
    w = parse_weather(feed)
    assert w.condition == "Sunny"
    assert w.temp_f == 78.0
    assert w.wind_speed_mph == 12.0
    assert w.wind_direction == "Out To LF"


def test_parse_weather_missing_fields_returns_none_values():
    w = parse_weather({"gamePk": 1, "gameData": {}})
    assert w.condition is None
    assert w.temp_f is None
    assert w.wind_speed_mph is None


def test_parse_umpire(live_feed):
    ump = parse_umpire(live_feed)
    assert ump.home_plate_umpire_id == 594151
    assert ump.home_plate_umpire_name == "Ramon De Jesus"


def test_parse_umpire_missing_returns_none_fields():
    feed = {"gamePk": 1, "liveData": {"boxscore": {"officials": []}}}
    ump = parse_umpire(feed)
    assert ump.home_plate_umpire_id is None
    assert ump.home_plate_umpire_name is None


def test_parse_game_context(live_feed):
    ctx = parse_game_context(live_feed)
    assert ctx.venue_id == 12  # Tropicana Field
    assert ctx.venue_name == "Tropicana Field"
    assert ctx.game_date == date(2026, 5, 17)


# -------- Client (orchestration) -------------------------------------------


def test_get_lineups_returns_pair(live_feed):
    client = StatsAPIClient(fetch_feed=lambda pk: live_feed)
    home, away = client.get_lineups(822982, cutoff_date=date(2026, 5, 17))
    assert isinstance(home, Lineup) and isinstance(away, Lineup)
    assert home.is_home is True
    assert away.is_home is False


def test_get_lineups_raises_when_not_posted():
    feed = {
        "gamePk": 1,
        "gameData": {"datetime": {"officialDate": "2026-05-17"}},
        "liveData": {"boxscore": {"teams": {"home": {"battingOrder": []}, "away": {"battingOrder": []}}}},
    }
    client = StatsAPIClient(fetch_feed=lambda pk: feed)
    with pytest.raises(LineupNotPostedError):
        client.get_lineups(1, cutoff_date=date(2026, 5, 17))


def test_get_weather_and_umpire_and_context(live_feed):
    client = StatsAPIClient(fetch_feed=lambda pk: live_feed)
    cutoff = date(2026, 5, 17)
    w = client.get_weather(822982, cutoff_date=cutoff)
    assert w.condition == "Dome"
    ump = client.get_umpire(822982, cutoff_date=cutoff)
    assert ump.home_plate_umpire_name == "Ramon De Jesus"
    ctx = client.get_game_context(822982, cutoff_date=cutoff)
    assert ctx.venue_id == 12


def test_cutoff_check_fires_on_future_dated_feed(live_feed):
    """If we ask for cutoff < game_date, the leakage walker must trip."""
    client = StatsAPIClient(fetch_feed=lambda pk: live_feed)
    # game_date is 2026-05-17; cutoff 2026-05-16 should raise.
    with pytest.raises(LeakageError):
        client.get_weather(822982, cutoff_date=date(2026, 5, 16))


def test_live_mode_warns_then_succeeds(live_feed):
    import warnings

    client = StatsAPIClient(fetch_feed=lambda pk: live_feed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        w = client.get_weather(822982, cutoff_date=None)
    assert any("LIVE MODE" in str(c.message) for c in captured)
    assert w.condition == "Dome"
