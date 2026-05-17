"""ProbablesClient: FanGraphs RosterResource primary, MLB StatsAPI fallback.

For live / recent dates, scrapes FanGraphs and falls back to StatsAPI. FanGraphs
already flags opener days via ``team.opener`` + ``team.primaryPitcher`` (the
bulk reliever). We trust that flag and pass the bulk pitcher straight through.
The separate :mod:`opener_detection` module catches cases FG missed (low-volume
probables, market signals).

For backtest dates, :meth:`get_historical_probables` reconstructs the probables
from the boxscore's actual starter. This is an explicit shortcut documented in
``backend/README.md`` — the historical "listed probable" isn't accessible, so
we use the conservative real-world answer (whoever actually started).

Cross-source disagreements (FG says A, StatsAPI says B) are logged to
``backend/data/raw/probables_disagreements.jsonl``. FG always wins.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .as_of_context import AsOfClient
from .player_resolver import load_cache, resolve_player, save_cache

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROBABLES_CACHE_DIR = DATA_RAW_DIR / "probables"
DISAGREEMENT_LOG = DATA_RAW_DIR / "probables_disagreements.jsonl"

FANGRAPHS_URL = "https://www.fangraphs.com/roster-resource/probables-grid"
STATSAPI_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
STATSAPI_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_FG_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>', re.DOTALL
)

# StatsAPI and FanGraphs disagree on a handful of abbreviations; normalize
# both sides to the FG-style 3-letter form for cross-source keying.
_TEAM_ABBR_TO_CANONICAL = {
    "TB": "TBR",
    "WSH": "WSN",
    "AZ": "ARI",
    "CWS": "CHW",
    "KC": "KCR",
    "SF": "SFG",
    "SD": "SDP",
}


def normalize_team_abbr(abbr: str) -> str:
    return _TEAM_ABBR_TO_CANONICAL.get(abbr, abbr)


@dataclass(frozen=True)
class ProbablePitcher:
    game_date: date
    team_abbr: str
    opponent_abbr: str
    is_home: bool
    pitcher_mlbam_id: int
    pitcher_name: str
    pitcher_hand: str | None  # 'L' / 'R'
    source: str  # 'fangraphs' | 'statsapi' | 'historical-actual'
    confidence: float
    opener_flagged: bool = False
    fangraphs_player_id: str | None = None
    game_pk: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["game_date"] = self.game_date.isoformat()
        return d


# -------- Parsers (pure functions, testable in isolation) ---------------------


def parse_fangraphs_html(html: str) -> list[dict[str, Any]]:
    """Pull the games list out of FanGraphs' embedded __NEXT_DATA__ JSON."""
    m = _FG_NEXT_DATA_RE.search(html)
    if not m:
        raise ValueError("FanGraphs HTML: __NEXT_DATA__ script tag not found")
    data = json.loads(m.group(1))
    try:
        queries = data["props"]["pageProps"]["dehydratedState"]["queries"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"FanGraphs HTML: unexpected JSON shape ({exc})")
    if not queries:
        raise ValueError("FanGraphs HTML: empty queries list")
    games = queries[0]["state"]["data"].get("games")
    if games is None:
        raise ValueError("FanGraphs HTML: no games key in dehydrated data")
    return games


def extract_fangraphs_probables(
    games: Iterable[dict[str, Any]],
    target_date: date,
) -> list[dict[str, Any]]:
    """For every team-row on ``target_date``, emit a probable record.

    FanGraphs renders each game twice (one row per team). We keep both rows so
    we can produce a ProbablePitcher record per team-game. The caller decides
    whether to dedupe by (date, home_team, away_team).
    """
    target_iso = target_date.isoformat()
    out: list[dict[str, Any]] = []
    for g in games:
        if g.get("gameDate") != target_iso:
            continue
        team_block = g.get("team") or {}
        opp_block = g.get("opponent") or {}
        team_abbr = g.get("abbName") or ""
        opp_abbr = opp_block.get("abbName") or ""
        opener = team_block.get("opener")
        primary = team_block.get("primaryPitcher")
        sp = team_block.get("sp")

        if opener and primary:
            pitcher_blob = primary
            opener_flagged = True
            confidence = 0.9
        elif opener and not primary:
            # FG knows it's an opener day but doesn't name the bulk pitcher.
            # Skip: we'd rather drop the game than guess.
            out.append(
                {
                    "skip": True,
                    "reason": "fangraphs: opener flagged but no primaryPitcher",
                    "game_date": target_date,
                    "team_abbr": normalize_team_abbr(team_abbr),
                    "opponent_abbr": normalize_team_abbr(opp_abbr),
                    "is_home": bool(g.get("isHome")),
                }
            )
            continue
        elif sp:
            pitcher_blob = sp
            opener_flagged = False
            confidence = 1.0
        else:
            # No probable announced yet.
            continue

        out.append(
            {
                "skip": False,
                "game_date": target_date,
                "team_abbr": normalize_team_abbr(team_abbr),
                "opponent_abbr": normalize_team_abbr(opp_abbr),
                "is_home": bool(g.get("isHome")),
                "pitcher_name": pitcher_blob.get("name", ""),
                "pitcher_hand": pitcher_blob.get("throws"),
                "fangraphs_player_id": str(pitcher_blob.get("playerId") or ""),
                "opener_flagged": opener_flagged,
                "confidence": confidence,
            }
        )
    return out


def parse_statsapi_schedule(
    schedule: dict[str, Any],
    target_date: date,
) -> list[dict[str, Any]]:
    """For each game on ``target_date``, emit a probable record per side."""
    out: list[dict[str, Any]] = []
    for date_entry in schedule.get("dates", []):
        if date_entry.get("date") != target_date.isoformat():
            continue
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            for side in ("home", "away"):
                team = (game.get("teams") or {}).get(side) or {}
                team_info = team.get("team") or {}
                opp = (game.get("teams") or {}).get(
                    "away" if side == "home" else "home"
                ) or {}
                opp_info = opp.get("team") or {}
                prob = team.get("probablePitcher")
                if not prob:
                    continue
                out.append(
                    {
                        "game_date": target_date,
                        "game_pk": game_pk,
                        "team_abbr": normalize_team_abbr(
                            team_info.get("abbreviation")
                            or team_info.get("name", "")
                        ),
                        "opponent_abbr": normalize_team_abbr(
                            opp_info.get("abbreviation")
                            or opp_info.get("name", "")
                        ),
                        "is_home": side == "home",
                        "pitcher_name": prob.get("fullName", ""),
                        "pitcher_mlbam_id": prob.get("id"),
                        "pitcher_hand": None,
                    }
                )
    return out


def parse_boxscore_actual_starters(
    feed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract actual starting pitchers from a StatsAPI live-feed boxscore.

    Used by :meth:`ProbablesClient.get_historical_probables`. The "starter" is
    the first pitcher in each team's ``pitchers`` array — this is StatsAPI's
    canonical ordering. The conservative shortcut for backtest mode.
    """
    out: list[dict[str, Any]] = []
    live = feed.get("liveData") or {}
    boxscore = live.get("boxscore") or {}
    teams = boxscore.get("teams") or {}
    game_pk = (feed.get("gamePk")) or (feed.get("gameData", {}) or {}).get(
        "game", {}
    ).get("pk")
    game_date_str = (
        ((feed.get("gameData") or {}).get("datetime") or {}).get("officialDate")
    )
    game_date = date.fromisoformat(game_date_str) if game_date_str else None

    for side in ("home", "away"):
        team_block = teams.get(side) or {}
        team_info = team_block.get("team") or {}
        pitchers = team_block.get("pitchers") or []
        if not pitchers:
            continue
        starter_id = pitchers[0]
        players = team_block.get("players") or {}
        player_block = players.get(f"ID{starter_id}") or {}
        person = player_block.get("person") or {}
        out.append(
            {
                "game_date": game_date,
                "game_pk": game_pk,
                "team_abbr": team_info.get("abbreviation") or team_info.get("name", ""),
                "is_home": side == "home",
                "pitcher_mlbam_id": int(starter_id),
                "pitcher_name": person.get("fullName", ""),
                "pitcher_hand": (
                    (player_block.get("person") or {}).get("pitchHand", {}) or {}
                ).get("code"),
            }
        )
    return out


# -------- Default HTTP fetchers (injectable) ---------------------------------


def _default_fetch_fangraphs(target_date: date) -> str:
    import requests  # local import keeps test imports cheap

    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html"}
    resp = requests.get(FANGRAPHS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def _default_fetch_statsapi_schedule(target_date: date) -> dict[str, Any]:
    import requests

    params = {
        "sportId": 1,
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher,team",
    }
    resp = requests.get(STATSAPI_SCHEDULE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _default_fetch_statsapi_feed(game_pk: int) -> dict[str, Any]:
    import requests

    resp = requests.get(
        STATSAPI_BOXSCORE_URL.format(game_pk=game_pk), timeout=30
    )
    resp.raise_for_status()
    return resp.json()


# -------- The client ---------------------------------------------------------


class ProbablesClient(AsOfClient):
    """Probables for live/recent dates. See module docstring."""

    def __init__(
        self,
        *,
        fetch_fangraphs: Callable[[date], str] | None = None,
        fetch_statsapi_schedule: Callable[[date], dict[str, Any]] | None = None,
        fetch_statsapi_feed: Callable[[int], dict[str, Any]] | None = None,
        player_lookup_fn: Callable[[str, str], object] | None = None,
        cache_dir: Path | None = None,
        player_cache_path: Path | None = None,
    ) -> None:
        self._fetch_fg = fetch_fangraphs or _default_fetch_fangraphs
        self._fetch_sched = fetch_statsapi_schedule or _default_fetch_statsapi_schedule
        self._fetch_feed = fetch_statsapi_feed or _default_fetch_statsapi_feed
        self._player_lookup_fn = player_lookup_fn
        self._cache_dir = cache_dir or PROBABLES_CACHE_DIR
        self._player_cache_path = player_cache_path
        self._player_cache = load_cache(path=player_cache_path)
        self._player_cache_dirty = False

    # ---- AsOfClient interface -----------------------------------------------

    def _fetch(
        self, cutoff_date: date | None, **kwargs: Any
    ) -> list[ProbablePitcher]:
        target = cutoff_date or date.today()
        try:
            return self._fetch_via_fangraphs(target)
        except Exception as exc:
            logger.warning(
                "ProbablesClient: FanGraphs path failed (%s); falling back to StatsAPI",
                exc,
            )
            return self._fetch_via_statsapi(target)

    # ---- Live / recent path -------------------------------------------------

    def _fetch_via_fangraphs(self, target: date) -> list[ProbablePitcher]:
        html = self._fetch_fg(target)
        self._cache_fangraphs_html(target, html)
        games = parse_fangraphs_html(html)
        fg_rows = extract_fangraphs_probables(games, target)

        # Also pull StatsAPI for the cross-check.
        try:
            sched = self._fetch_sched(target)
            sa_rows = parse_statsapi_schedule(sched, target)
        except Exception as exc:
            logger.warning("ProbablesClient: StatsAPI cross-check unavailable: %s", exc)
            sa_rows = []

        sa_by_key = {(r["team_abbr"], r["is_home"]): r for r in sa_rows}

        out: list[ProbablePitcher] = []
        for row in fg_rows:
            if row.get("skip"):
                logger.info(
                    "ProbablesClient: skipping %s vs %s on %s: %s",
                    row["team_abbr"], row["opponent_abbr"],
                    row["game_date"], row["reason"],
                )
                continue
            key = (row["team_abbr"], row["is_home"])
            sa_row = sa_by_key.get(key)
            confidence = float(row["confidence"])

            if sa_row and sa_row.get("pitcher_name"):
                if _names_match(row["pitcher_name"], sa_row["pitcher_name"]):
                    confidence = min(1.0, confidence + 0.05)
                else:
                    self._log_disagreement(
                        row["game_date"],
                        sa_row.get("game_pk"),
                        row["team_abbr"],
                        row["pitcher_name"],
                        sa_row["pitcher_name"],
                    )

            mlbam_id = self._resolve_id(row["pitcher_name"])
            if mlbam_id is None:
                logger.warning(
                    "ProbablesClient: dropping %s (cannot resolve MLBAM id)",
                    row["pitcher_name"],
                )
                continue

            out.append(
                ProbablePitcher(
                    game_date=row["game_date"],
                    team_abbr=row["team_abbr"],
                    opponent_abbr=row["opponent_abbr"],
                    is_home=row["is_home"],
                    pitcher_mlbam_id=mlbam_id,
                    pitcher_name=row["pitcher_name"],
                    pitcher_hand=row["pitcher_hand"],
                    source="fangraphs",
                    confidence=confidence,
                    opener_flagged=row["opener_flagged"],
                    fangraphs_player_id=row.get("fangraphs_player_id"),
                    game_pk=(sa_row or {}).get("game_pk"),
                )
            )

        if self._player_cache_dirty:
            save_cache(self._player_cache, path=self._player_cache_path)
            self._player_cache_dirty = False

        return out

    # ---- Fallback path ------------------------------------------------------

    def _fetch_via_statsapi(self, target: date) -> list[ProbablePitcher]:
        sched = self._fetch_sched(target)
        rows = parse_statsapi_schedule(sched, target)
        out: list[ProbablePitcher] = []
        for row in rows:
            mlbam_id = row.get("pitcher_mlbam_id")
            if not mlbam_id:
                continue
            out.append(
                ProbablePitcher(
                    game_date=row["game_date"],
                    team_abbr=row["team_abbr"],
                    opponent_abbr=row["opponent_abbr"],
                    is_home=row["is_home"],
                    pitcher_mlbam_id=int(mlbam_id),
                    pitcher_name=row["pitcher_name"],
                    pitcher_hand=row.get("pitcher_hand"),
                    source="statsapi",
                    confidence=0.8,
                    game_pk=row.get("game_pk"),
                )
            )
        return out

    # ---- Backtest path ------------------------------------------------------

    def get_historical_probables(self, game_date: date) -> list[ProbablePitcher]:
        """Reconstruct probables from actual game starters.

        BACKTEST ONLY. Issues a loud warning. The boxscore's first pitcher per
        side is treated as the probable. Opener detection is effectively
        disabled for backtests — we accept this conservative simplification.
        """
        import warnings

        warnings.warn(
            "ProbablesClient: BACKTEST MODE — actual starter used as probable proxy. "
            "Opener detection is limited in this mode.",
            stacklevel=2,
        )
        logger.warning(
            "ProbablesClient: BACKTEST MODE — using actual starters from boxscore for %s",
            game_date,
        )

        sched = self._fetch_sched(game_date)
        out: list[ProbablePitcher] = []
        for date_entry in sched.get("dates", []):
            if date_entry.get("date") != game_date.isoformat():
                continue
            for game in date_entry.get("games", []):
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue
                feed = self._fetch_feed(game_pk)
                rows = parse_boxscore_actual_starters(feed)
                # Stamp source and resolve opponent abbr from the schedule.
                home_team = (
                    ((game.get("teams") or {}).get("home") or {}).get("team") or {}
                ).get("abbreviation") or ""
                away_team = (
                    ((game.get("teams") or {}).get("away") or {}).get("team") or {}
                ).get("abbreviation") or ""
                for row in rows:
                    opp = away_team if row["is_home"] else home_team
                    out.append(
                        ProbablePitcher(
                            game_date=row["game_date"] or game_date,
                            team_abbr=row["team_abbr"],
                            opponent_abbr=opp,
                            is_home=row["is_home"],
                            pitcher_mlbam_id=row["pitcher_mlbam_id"],
                            pitcher_name=row["pitcher_name"],
                            pitcher_hand=row.get("pitcher_hand"),
                            source="historical-actual",
                            confidence=0.5,
                            game_pk=game_pk,
                        )
                    )
        return out

    # ---- helpers ------------------------------------------------------------

    def _resolve_id(self, name: str) -> int | None:
        before = dict(self._player_cache)
        mid = resolve_player(
            name,
            cache=self._player_cache,
            lookup_fn=self._player_lookup_fn,
        )
        if self._player_cache != before:
            self._player_cache_dirty = True
        return mid

    def _cache_fangraphs_html(self, target: date, html: str) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M", time.gmtime())
            path = self._cache_dir / f"fangraphs_{target.isoformat()}_{stamp}.html"
            path.write_text(html, encoding="utf-8")
        except OSError as exc:
            logger.warning("ProbablesClient: failed to cache FG HTML: %s", exc)

    def _log_disagreement(
        self,
        game_date: date,
        game_pk: int | None,
        team_abbr: str,
        fg_name: str,
        sa_name: str,
    ) -> None:
        try:
            DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "game_date": game_date.isoformat(),
                "game_pk": game_pk,
                "team_abbr": team_abbr,
                "fangraphs_pitcher": fg_name,
                "statsapi_pitcher": sa_name,
                "decision": "fangraphs",
            }
            with DISAGREEMENT_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("ProbablesClient: failed to log disagreement: %s", exc)


# -------- Name comparison utility --------------------------------------------


def _names_match(a: str, b: str) -> bool:
    """Loose equality: strip accents/suffixes/case, compare."""
    from .player_resolver import normalize_name

    return normalize_name(a) == normalize_name(b)


# -------- CLI entry point ----------------------------------------------------


def _cli() -> None:
    """``python -m src.data.probables_client --date today``

    Live-fetches today's probables and writes them to
    ``backend/data/raw/probables/today.json`` for review.
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    parser.add_argument(
        "--out",
        default=str(PROBABLES_CACHE_DIR / "today.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    target = date.today() if args.date == "today" else date.fromisoformat(args.date)
    client = ProbablesClient()
    probables = client.fetch(cutoff_date=target)
    payload = {
        "as_of_date": target.isoformat(),
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "count": len(probables),
        "probables": [p.to_dict() for p in probables],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(probables)} probables -> {out_path}")
    for p in probables[:5]:
        flag = " [OPENER]" if p.opener_flagged else ""
        # Strip accents for console output to avoid cp1252 encoding errors on
        # Windows; the JSON output preserves the original UTF-8.
        from .player_resolver import _strip_accents
        name = _strip_accents(p.pitcher_name)
        print(
            f"  {p.team_abbr:3s} {'vs' if p.is_home else '@ '} {p.opponent_abbr:3s}  "
            f"{name:25s}  ({p.pitcher_hand or '?'})  "
            f"src={p.source} conf={p.confidence:.2f}{flag}"
        )
    if len(probables) > 5:
        print(f"  ... +{len(probables) - 5} more")


if __name__ == "__main__":
    _cli()
