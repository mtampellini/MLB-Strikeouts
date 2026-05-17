"""StatsAPIClient: lineups, weather, umpire, game/venue context.

Probables come from :mod:`probables_client`, not from here.

All endpoints hit the MLB StatsAPI live-feed
(``/api/v1.1/game/{game_pk}/feed/live``), which is immutable once the game
goes Final. For pre-game/live games, lineups are present in the feed once the
team posts them (typically 2-3 hours before first pitch); before that,
:meth:`get_lineups` raises :class:`LineupNotPostedError`.

Park metadata beyond ``venue_id``/``venue_name`` (altitude, dimensions,
roof-state) is precomputed elsewhere (Phase 3) and merged in by the feature
pipeline; this client only surfaces the StatsAPI fields verbatim.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from .as_of_context import AsOfClient

logger = logging.getLogger(__name__)

STATSAPI_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


class LineupNotPostedError(LookupError):
    """The lineup for the requested game has not yet been posted."""


# -------- Canonical output shapes --------------------------------------------


@dataclass(frozen=True)
class BatterEntry:
    batter_mlbam_id: int
    name: str
    batting_order_spot: int  # 1..9
    bats: str | None  # 'L' | 'R' | 'S'
    position_code: str | None  # e.g. 'CF', 'SS', 'DH'


@dataclass(frozen=True)
class Lineup:
    game_pk: int
    team_abbr: str
    is_home: bool
    batters: tuple[BatterEntry, ...]


@dataclass(frozen=True)
class GameWeather:
    game_pk: int
    temp_f: float | None
    wind_speed_mph: float | None
    wind_direction: str | None  # e.g. 'Out To LF', 'In From CF', 'None'
    condition: str | None  # 'Sunny', 'Cloudy', 'Dome', ...


@dataclass(frozen=True)
class UmpireAssignment:
    game_pk: int
    home_plate_umpire_id: int | None
    home_plate_umpire_name: str | None


@dataclass(frozen=True)
class GameContext:
    game_pk: int
    game_date: date
    venue_id: int | None
    venue_name: str | None
    home_team_abbr: str | None
    away_team_abbr: str | None
    game_datetime_iso: str | None


# -------- Parsers (pure, testable in isolation) -----------------------------


def _player_block(feed: dict[str, Any], player_id: int) -> dict[str, Any]:
    return ((feed.get("gameData") or {}).get("players") or {}).get(
        f"ID{player_id}"
    ) or {}


def parse_lineup(feed: dict[str, Any], *, is_home: bool) -> Lineup | None:
    side = "home" if is_home else "away"
    boxscore = (feed.get("liveData") or {}).get("boxscore") or {}
    team_block = (boxscore.get("teams") or {}).get(side) or {}
    order = team_block.get("battingOrder") or []
    if not order:
        return None
    # The boxscore team block omits `abbreviation`; gameData.teams has it.
    gd_team = ((feed.get("gameData") or {}).get("teams") or {}).get(side) or {}
    team_info = team_block.get("team") or {}
    team_abbr = (
        gd_team.get("abbreviation")
        or team_info.get("abbreviation")
        or team_info.get("name", "")
    )
    game_pk = (feed.get("gamePk")) or (
        (feed.get("gameData") or {}).get("game", {}) or {}
    ).get("pk")

    players_block = team_block.get("players") or {}
    batters: list[BatterEntry] = []
    for spot, pid in enumerate(order, start=1):
        box_player = players_block.get(f"ID{pid}") or {}
        person = box_player.get("person") or {}
        position = box_player.get("position") or {}
        gd_player = _player_block(feed, pid)
        bat_side = (gd_player.get("batSide") or {}).get("code")
        batters.append(
            BatterEntry(
                batter_mlbam_id=int(pid),
                name=person.get("fullName") or gd_player.get("fullName") or "",
                batting_order_spot=spot,
                bats=bat_side,
                position_code=position.get("abbreviation"),
            )
        )

    return Lineup(
        game_pk=int(game_pk) if game_pk else 0,
        team_abbr=team_abbr,
        is_home=is_home,
        batters=tuple(batters),
    )


_WIND_RE = re.compile(
    r"^\s*(?P<speed>\d+(?:\.\d+)?)\s*mph\s*,\s*(?P<dir>.+?)\s*$",
    re.IGNORECASE,
)


def parse_weather(feed: dict[str, Any]) -> GameWeather:
    weather = (feed.get("gameData") or {}).get("weather") or {}
    game_pk = feed.get("gamePk") or (
        (feed.get("gameData") or {}).get("game") or {}
    ).get("pk", 0)

    temp_f = _safe_float(weather.get("temp"))
    wind_speed: float | None = None
    wind_dir: str | None = None
    wind_raw = weather.get("wind")
    if wind_raw:
        m = _WIND_RE.match(str(wind_raw))
        if m:
            wind_speed = _safe_float(m.group("speed"))
            wind_dir = m.group("dir").strip()
            if wind_dir.lower() == "none":
                wind_dir = None
        else:
            # Best-effort: still keep the raw string so we don't lose info.
            wind_dir = str(wind_raw)

    return GameWeather(
        game_pk=int(game_pk),
        temp_f=temp_f,
        wind_speed_mph=wind_speed,
        wind_direction=wind_dir,
        condition=weather.get("condition") or None,
    )


def parse_umpire(feed: dict[str, Any]) -> UmpireAssignment:
    boxscore = (feed.get("liveData") or {}).get("boxscore") or {}
    officials = boxscore.get("officials") or []
    game_pk = feed.get("gamePk") or 0
    for off in officials:
        if (off.get("officialType") or "").lower() == "home plate":
            ump = off.get("official") or {}
            return UmpireAssignment(
                game_pk=int(game_pk),
                home_plate_umpire_id=ump.get("id"),
                home_plate_umpire_name=ump.get("fullName"),
            )
    return UmpireAssignment(
        game_pk=int(game_pk),
        home_plate_umpire_id=None,
        home_plate_umpire_name=None,
    )


def parse_game_context(feed: dict[str, Any]) -> GameContext:
    game_data = feed.get("gameData") or {}
    game = game_data.get("game") or {}
    venue = game_data.get("venue") or {}
    datetime_block = game_data.get("datetime") or {}
    teams = game_data.get("teams") or {}
    home_team = (teams.get("home") or {}).get("abbreviation")
    away_team = (teams.get("away") or {}).get("abbreviation")
    game_pk = feed.get("gamePk") or game.get("pk", 0)
    game_date_iso = datetime_block.get("officialDate") or datetime_block.get(
        "dateTime", ""
    )[:10]
    try:
        game_date_obj = date.fromisoformat(game_date_iso) if game_date_iso else date.min
    except ValueError:
        game_date_obj = date.min

    return GameContext(
        game_pk=int(game_pk),
        game_date=game_date_obj,
        venue_id=venue.get("id"),
        venue_name=venue.get("name"),
        home_team_abbr=home_team,
        away_team_abbr=away_team,
        game_datetime_iso=datetime_block.get("dateTime"),
    )


def _safe_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# -------- Default HTTP fetcher (injectable) ---------------------------------


def _default_fetch_feed(game_pk: int) -> dict[str, Any]:
    import requests

    resp = requests.get(STATSAPI_FEED_URL.format(game_pk=game_pk), timeout=30)
    resp.raise_for_status()
    return resp.json()


# -------- The client --------------------------------------------------------


class StatsAPIClient(AsOfClient):
    """Live + historical access to lineups, weather, umpire, game context.

    Multi-endpoint: subclasses of :class:`AsOfClient` typically implement a
    single :meth:`_fetch`. Here we override _fetch with a stub and route each
    named method through :meth:`AsOfClient._asof_pre` / :meth:`_asof_post`.
    """

    def __init__(
        self,
        *,
        fetch_feed: Callable[[int], dict[str, Any]] | None = None,
    ) -> None:
        self._fetch_feed = fetch_feed or _default_fetch_feed

    def _fetch(self, cutoff_date: date | None, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "StatsAPIClient has named endpoints; call get_lineups, get_weather, "
            "get_umpire, or get_game_context."
        )

    # ---- Lineups -----------------------------------------------------------

    def get_lineups(
        self,
        game_pk: int,
        *,
        cutoff_date: date | None,
    ) -> tuple[Lineup, Lineup]:
        """Return (home_lineup, away_lineup) for a game.

        Raises :class:`LineupNotPostedError` if either side is missing — the
        pipeline should defer to a later cron fire.
        """
        self._asof_pre(cutoff_date)
        feed = self._fetch_feed(game_pk)
        home = parse_lineup(feed, is_home=True)
        away = parse_lineup(feed, is_home=False)
        if home is None or away is None:
            raise LineupNotPostedError(
                f"StatsAPIClient: lineup not posted for game_pk={game_pk} "
                f"(home={home is not None}, away={away is not None})"
            )
        result = (home, away)
        self._asof_post(_lineups_payload_for_check(result, feed), cutoff_date)
        return result

    # ---- Weather -----------------------------------------------------------

    def get_weather(
        self, game_pk: int, *, cutoff_date: date | None
    ) -> GameWeather:
        self._asof_pre(cutoff_date)
        feed = self._fetch_feed(game_pk)
        result = parse_weather(feed)
        self._asof_post(_weather_payload_for_check(result, feed), cutoff_date)
        return result

    # ---- Umpire ------------------------------------------------------------

    def get_umpire(
        self, game_pk: int, *, cutoff_date: date | None
    ) -> UmpireAssignment:
        self._asof_pre(cutoff_date)
        feed = self._fetch_feed(game_pk)
        result = parse_umpire(feed)
        self._asof_post(_umpire_payload_for_check(result, feed), cutoff_date)
        return result

    # ---- Game context ------------------------------------------------------

    def get_game_context(
        self, game_pk: int, *, cutoff_date: date | None
    ) -> GameContext:
        self._asof_pre(cutoff_date)
        feed = self._fetch_feed(game_pk)
        result = parse_game_context(feed)
        self._asof_post(_context_payload_for_check(result, feed), cutoff_date)
        return result


# -------- Helpers ------------------------------------------------------------


def _lineups_payload_for_check(
    result: tuple[Lineup, Lineup], feed: dict[str, Any]
) -> dict[str, Any]:
    """Payload to feed the AsOf leakage walker.

    The walker looks for date-like strings. We pass the full feed so any
    post-cutoff timestamp (e.g. a game played after cutoff) is caught.
    """
    return {
        "game_date": (feed.get("gameData") or {}).get("datetime", {}).get(
            "officialDate"
        ),
        "result": [
            {
                "game_pk": ln.game_pk,
                "team_abbr": ln.team_abbr,
                "batter_ids": [b.batter_mlbam_id for b in ln.batters],
            }
            for ln in result
        ],
    }


def _weather_payload_for_check(
    result: GameWeather, feed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "game_date": (feed.get("gameData") or {}).get("datetime", {}).get(
            "officialDate"
        ),
        "result": asdict(result),
    }


def _umpire_payload_for_check(
    result: UmpireAssignment, feed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "game_date": (feed.get("gameData") or {}).get("datetime", {}).get(
            "officialDate"
        ),
        "result": asdict(result),
    }


def _context_payload_for_check(
    result: GameContext, feed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "game_date": result.game_date.isoformat()
        if result.game_date != date.min
        else None,
        "result_datetime_iso": result.game_datetime_iso,
    }
