"""OddsAPIClient: The Odds API wrapper for ``pitcher_strikeouts_alternate``.

Books are hardcoded to FanDuel + DraftKings — if a third book slips through the
API response the allowlist filter drops it and logs a warning.

Snapshot persistence (this IS our dataset):
- After every successful event-odds fetch, the raw response is written to
  ``backend/data/odds/YYYY-MM-DD/HHMM_event-{event_id}.json``.
- HHMM is the current UTC minute. Same minute = overwrite (idempotent). A
  later minute creates a separate snapshot so we can track intraday line
  movement.

Historical mode (cutoff_date < today):
- Reads snapshots from ``backend/data/odds/{cutoff_date}/`` and returns the
  latest snapshot per event. If no snapshot exists for that date, raises
  ``FileNotFoundError`` — we never fabricate odds.

Budget tracking:
- The Odds API returns ``x-requests-remaining`` on every response. Below the
  configured floor (default 50), :meth:`request` refuses further calls in
  this session by raising :class:`BudgetExhaustedError`.

Devig:
- Multiplicative: ``p_true = p_book / (p_over + p_under)``. When only one
  side is offered, the prop is "one-sided" — the line is captured but flagged
  ``one_sided=True`` and the picks layer skips it.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .as_of_context import AsOfClient
from .player_resolver import load_cache, resolve_player, save_cache

logger = logging.getLogger(__name__)

DATA_ODDS_DIR = Path(__file__).resolve().parents[2] / "data" / "odds"
ALLOWED_BOOKS = frozenset({"fanduel", "draftkings"})
SPORT_KEY = "baseball_mlb"
MARKET_KEY = "pitcher_strikeouts_alternate"
REQUESTS_REMAINING_HEADER = "x-requests-remaining"
DEFAULT_BUDGET_FLOOR = 50


class BudgetExhaustedError(RuntimeError):
    """API request budget dipped below the configured floor."""


class MissingSnapshotError(FileNotFoundError):
    """No committed snapshot exists for the requested historical date."""


# -------- Canonical output shapes --------------------------------------------


@dataclass(frozen=True)
class PitcherLine:
    line: float
    side: str  # 'Over' | 'Under'
    price: int  # American odds


@dataclass(frozen=True)
class PitcherProp:
    pitcher_name: str
    pitcher_mlbam_id: int | None
    book: str
    lines: tuple[PitcherLine, ...]


@dataclass(frozen=True)
class EventOdds:
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    pitcher_props: tuple[PitcherProp, ...]
    snapshot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "commence_time": self.commence_time,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "pitcher_props": [
                {
                    "pitcher_name": p.pitcher_name,
                    "pitcher_mlbam_id": p.pitcher_mlbam_id,
                    "book": p.book,
                    "lines": [asdict(ln) for ln in p.lines],
                }
                for p in self.pitcher_props
            ],
            "snapshot_path": self.snapshot_path,
        }


# -------- Parsing -------------------------------------------------------------


def parse_event_odds_response(
    payload: Mapping[str, Any],
    *,
    resolve_id: Callable[[str], int | None] | None = None,
) -> EventOdds:
    """Convert The Odds API raw event-odds payload to canonical EventOdds.

    Drops bookmakers not in the FD/DK allowlist (with a warning). Drops props
    where the pitcher name can't be resolved to an MLBAM id (caller-injectable
    via ``resolve_id``).
    """
    event_id = payload.get("id", "")
    commence_time = payload.get("commence_time", "")
    home_team = payload.get("home_team", "")
    away_team = payload.get("away_team", "")

    props: list[PitcherProp] = []
    for book in payload.get("bookmakers", []):
        book_key = book.get("key", "")
        if book_key not in ALLOWED_BOOKS:
            logger.warning(
                "OddsAPIClient: dropping disallowed book %r from event %s",
                book_key, event_id,
            )
            continue
        for market in book.get("markets", []):
            if market.get("key") != MARKET_KEY:
                continue
            by_pitcher: dict[str, list[PitcherLine]] = {}
            for oc in market.get("outcomes", []):
                name = oc.get("description") or oc.get("name") or ""
                side = oc.get("name") if oc.get("description") else "Over"
                price = oc.get("price")
                point = oc.get("point")
                if not name or price is None or point is None:
                    continue
                try:
                    price_int = int(price)
                except (TypeError, ValueError):
                    continue
                try:
                    line_val = float(point)
                except (TypeError, ValueError):
                    continue
                by_pitcher.setdefault(name, []).append(
                    PitcherLine(line=line_val, side=str(side), price=price_int)
                )

            for pitcher_name, lines in by_pitcher.items():
                mlbam_id = resolve_id(pitcher_name) if resolve_id else None
                if resolve_id and mlbam_id is None:
                    logger.warning(
                        "OddsAPIClient: cannot resolve %r to MLBAM id; dropping prop",
                        pitcher_name,
                    )
                    continue
                props.append(
                    PitcherProp(
                        pitcher_name=pitcher_name,
                        pitcher_mlbam_id=mlbam_id,
                        book=book_key,
                        lines=tuple(sorted(lines, key=lambda ln: (ln.line, ln.side))),
                    )
                )

    return EventOdds(
        event_id=event_id,
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
        pitcher_props=tuple(props),
    )


# -------- Devig ---------------------------------------------------------------


def american_to_implied(price: int) -> float:
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def devig_two_sided(over_price: int, under_price: int) -> dict[str, float]:
    p_over = american_to_implied(over_price)
    p_under = american_to_implied(under_price)
    total = p_over + p_under
    if total <= 0:
        raise ValueError("devig_two_sided: non-positive implied total")
    return {
        "over_true": p_over / total,
        "under_true": p_under / total,
        "vig": total - 1.0,
    }


# -------- The client ----------------------------------------------------------


class OddsAPIClient(AsOfClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        budget_floor: int = DEFAULT_BUDGET_FLOOR,
        fetch_events: Callable[[str], tuple[list[dict[str, Any]], dict[str, str]]] | None = None,
        fetch_event_odds: (
            Callable[[str, str], tuple[dict[str, Any], dict[str, str]]] | None
        ) = None,
        snapshot_dir: Path | None = None,
        player_lookup_fn: Callable[[str, str], object] | None = None,
        clock: Callable[[], datetime] | None = None,
        player_cache_path: Path | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ODDS_API_KEY")
        self._budget_floor = budget_floor
        self._fetch_events = fetch_events or self._default_fetch_events
        self._fetch_event_odds = fetch_event_odds or self._default_fetch_event_odds
        self._snapshot_dir = snapshot_dir or DATA_ODDS_DIR
        self._player_lookup_fn = player_lookup_fn
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._player_cache_path = player_cache_path
        self._player_cache = load_cache(path=player_cache_path)
        self._player_cache_dirty = False
        self._budget_exhausted = False
        self._last_requests_remaining: int | None = None

    # ---- AsOfClient interface ------------------------------------------------

    def _fetch(self, cutoff_date: date | None, **kwargs: Any) -> list[EventOdds]:
        today = self._clock().date()
        target = cutoff_date or today
        if cutoff_date is None or cutoff_date >= today:
            return self._fetch_live(target, dry_run=bool(kwargs.get("dry_run")))
        return self._read_historical(target)

    # ---- Live mode -----------------------------------------------------------

    def _fetch_live(self, target: date, *, dry_run: bool) -> list[EventOdds]:
        if not self._api_key and not dry_run:
            raise RuntimeError(
                "OddsAPIClient: no API key configured (set ODDS_API_KEY)"
            )
        events, headers = self._fetch_events(self._api_key or "")
        self._update_budget(headers)

        out: list[EventOdds] = []
        for event in events:
            event_id = event.get("id")
            if not event_id:
                continue
            # Skip events whose game start is past the cutoff date
            ct = event.get("commence_time", "")
            event_date = _commence_date(ct)
            if event_date and event_date > target:
                continue
            if dry_run:
                out.append(
                    EventOdds(
                        event_id=event_id,
                        commence_time=ct,
                        home_team=event.get("home_team", ""),
                        away_team=event.get("away_team", ""),
                        pitcher_props=(),
                    )
                )
                continue
            self._guard_budget()
            payload, headers = self._fetch_event_odds(self._api_key or "", event_id)
            self._update_budget(headers)
            snapshot_path = self._persist_snapshot(target, event_id, payload)
            parsed = parse_event_odds_response(
                payload, resolve_id=self._resolve_id
            )
            out.append(
                EventOdds(
                    event_id=parsed.event_id or event_id,
                    commence_time=parsed.commence_time or ct,
                    home_team=parsed.home_team or event.get("home_team", ""),
                    away_team=parsed.away_team or event.get("away_team", ""),
                    pitcher_props=parsed.pitcher_props,
                    snapshot_path=str(snapshot_path) if snapshot_path else None,
                )
            )

        if self._player_cache_dirty:
            save_cache(self._player_cache, path=self._player_cache_path)
            self._player_cache_dirty = False

        return out

    # ---- Historical mode -----------------------------------------------------

    def _read_historical(self, target: date) -> list[EventOdds]:
        day_dir = self._snapshot_dir / target.isoformat()
        if not day_dir.exists():
            raise MissingSnapshotError(
                f"OddsAPIClient: no snapshots committed for {target} at {day_dir}"
            )
        latest_by_event: dict[str, tuple[str, Path]] = {}
        for path in day_dir.glob("*_event-*.json"):
            stem = path.stem
            try:
                stamp, _, rest = stem.partition("_event-")
            except ValueError:
                continue
            event_id = rest
            prev = latest_by_event.get(event_id)
            if prev is None or stamp > prev[0]:
                latest_by_event[event_id] = (stamp, path)

        if not latest_by_event:
            raise MissingSnapshotError(
                f"OddsAPIClient: snapshot directory for {target} is empty"
            )

        out: list[EventOdds] = []
        for event_id, (_, path) in sorted(latest_by_event.items()):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "OddsAPIClient: skipping corrupt snapshot %s: %s", path, exc
                )
                continue
            parsed = parse_event_odds_response(payload, resolve_id=self._resolve_id)
            out.append(
                EventOdds(
                    event_id=parsed.event_id or event_id,
                    commence_time=parsed.commence_time,
                    home_team=parsed.home_team,
                    away_team=parsed.away_team,
                    pitcher_props=parsed.pitcher_props,
                    snapshot_path=str(path),
                )
            )

        if self._player_cache_dirty:
            save_cache(self._player_cache, path=self._player_cache_path)
            self._player_cache_dirty = False

        return out

    # ---- Posted-K-prop helper for the opener trigger -------------------------

    def posted_k_prop_pitchers(
        self, events: Iterable[EventOdds]
    ) -> set[int]:
        """Return the set of MLBAM IDs that have any posted K prop on FD/DK."""
        ids: set[int] = set()
        for ev in events:
            for prop in ev.pitcher_props:
                if prop.pitcher_mlbam_id is not None:
                    ids.add(prop.pitcher_mlbam_id)
        return ids

    # ---- Snapshot helpers ---------------------------------------------------

    def _persist_snapshot(
        self, target: date, event_id: str, payload: Mapping[str, Any]
    ) -> Path | None:
        try:
            day_dir = self._snapshot_dir / target.isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._clock().strftime("%H%M")
            path = day_dir / f"{stamp}_event-{event_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return path
        except OSError as exc:
            logger.warning("OddsAPIClient: failed to write snapshot: %s", exc)
            return None

    # ---- Budget helpers -----------------------------------------------------

    def _update_budget(self, headers: Mapping[str, str]) -> None:
        if not headers:
            return
        raw = headers.get(REQUESTS_REMAINING_HEADER) or headers.get(
            REQUESTS_REMAINING_HEADER.title()
        )
        if raw is None:
            return
        try:
            self._last_requests_remaining = int(raw)
        except (TypeError, ValueError):
            return
        logger.info(
            "OddsAPIClient: requests-remaining=%d (floor=%d)",
            self._last_requests_remaining, self._budget_floor,
        )
        if self._last_requests_remaining < self._budget_floor:
            self._budget_exhausted = True
            logger.warning(
                "OddsAPIClient: budget below floor (%d < %d); refusing further calls",
                self._last_requests_remaining, self._budget_floor,
            )

    def _guard_budget(self) -> None:
        if self._budget_exhausted:
            raise BudgetExhaustedError(
                f"OddsAPIClient: requests-remaining "
                f"{self._last_requests_remaining} below floor {self._budget_floor}"
            )

    # ---- HTTP defaults (injectable) -----------------------------------------

    @staticmethod
    def _default_fetch_events(
        api_key: str,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        import requests

        url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
        params = {"apiKey": api_key}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), dict(resp.headers)

    @staticmethod
    def _default_fetch_event_odds(
        api_key: str, event_id: str
    ) -> tuple[dict[str, Any], dict[str, str]]:
        import requests

        url = (
            f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/"
            f"{event_id}/odds"
        )
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": MARKET_KEY,
            "bookmakers": "fanduel,draftkings",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json(), dict(resp.headers)

    # ---- Player resolution --------------------------------------------------

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


# -------- Module helpers ------------------------------------------------------


def _commence_date(commence_time: str) -> date | None:
    if not commence_time:
        return None
    try:
        return datetime.fromisoformat(
            commence_time.replace("Z", "+00:00")
        ).astimezone(timezone.utc).date()
    except ValueError:
        return None


# -------- CLI entry point ----------------------------------------------------


def _cli() -> None:
    """``python -m src.data.odds_client --date today [--dry-run]``

    ``--dry-run`` lists which events exist on the slate without consuming any
    per-event odds requests (one /events call is still made — that one is
    free on the free tier as of 2025).
    """
    import argparse

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hit /events only — do not pull per-event odds",
    )
    parser.add_argument(
        "--out", default=None, help="Output JSON path; default depends on mode"
    )
    args = parser.parse_args()

    target = date.today() if args.date == "today" else date.fromisoformat(args.date)
    client = OddsAPIClient()
    events = client.fetch(cutoff_date=target, dry_run=args.dry_run)

    out_path = Path(
        args.out
        or (
            DATA_ODDS_DIR
            / f"_summary_{target.isoformat()}{'_dryrun' if args.dry_run else ''}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of_date": target.isoformat(),
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "dry_run": args.dry_run,
        "events": [ev.to_dict() for ev in events],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pitchers = sorted(
        {p.pitcher_name for ev in events for p in ev.pitcher_props}
    )
    print(f"wrote {len(events)} events → {out_path}")
    if args.dry_run:
        print(f"  dry-run: per-event odds not fetched")
    else:
        print(f"  {len(pitchers)} pitchers with posted K props on FD/DK:")
        for n in pitchers:
            print(f"    {n}")


if __name__ == "__main__":
    _cli()
