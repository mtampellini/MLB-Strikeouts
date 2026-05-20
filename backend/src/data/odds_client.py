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
# Floor is 100 (not 50): a fresh 500-budget month has less slack to play with
# than HR-Picks, which has been running for months and has stable usage.
DEFAULT_BUDGET_FLOOR = 100

# Env var name is intentionally distinct from HR-Picks' ODDS_API_KEY so the
# repos can't accidentally share a key. The startup check in __init__ raises
# if the strikeouts key is missing OR matches the HR key value.
ENV_KEY_STRIKEOUTS = "ODDS_API_KEY_STRIKEOUTS"
ENV_KEY_HR = "ODDS_API_KEY"


class CrossRepoKeyBleedError(RuntimeError):
    """The strikeouts API key is missing, or equals the HR repo's key."""


class BudgetExhaustedError(RuntimeError):
    """API request budget dipped below the configured floor."""


class QuotaExhaustedError(RuntimeError):
    """401 OUT_OF_USAGE_CREDITS — the monthly quota for the active key is
    spent. Distinct from BudgetExhaustedError (which is our local pre-flight
    floor); both trigger key rotation in the multi-key failover loop.

    Auth failures that would equally affect every key (invalid key, IP
    block) are NOT mapped to this error — they propagate as
    :class:`OddsAPIError` so we don't silently burn through every key
    chasing the same problem."""


class OddsAPIError(RuntimeError):
    """Non-quota Odds API failure (HTTP 4xx/5xx other than 401-out-of-credits)."""


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
        api_key: str | list[str] | None = None,
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
        self._api_keys: list[str] = _resolve_api_keys(api_key)
        if not self._api_keys:
            raise CrossRepoKeyBleedError(
                f"OddsAPIClient: env var {ENV_KEY_STRIKEOUTS!r} is not set. "
                f"This repo uses a dedicated Odds API key separate from "
                f"{ENV_KEY_HR!r} (the HR-Picks key). See backend/README.md."
            )
        self._key_idx: int = 0
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

    # ---- Multi-key failover -------------------------------------------------

    @property
    def _api_key(self) -> str:
        """Currently active API key. Rotates on quota / budget exhaustion."""
        return self._api_keys[self._key_idx]

    @property
    def active_key_index(self) -> int:
        return self._key_idx

    @property
    def num_keys(self) -> int:
        return len(self._api_keys)

    def _advance_key(self) -> bool:
        """Move to the next configured key. Returns False if there is no
        next key (caller should propagate the exhaustion error). Resets the
        budget tracker — each key has its own monthly quota."""
        if self._key_idx + 1 >= len(self._api_keys):
            return False
        logger.warning(
            "OddsAPIClient: key index %d exhausted (remaining=%s); "
            "rotating to key index %d (of %d configured)",
            self._key_idx, self._last_requests_remaining,
            self._key_idx + 1, len(self._api_keys),
        )
        self._key_idx += 1
        self._budget_exhausted = False
        self._last_requests_remaining = None
        return True

    # ---- Budget pre-flight ---------------------------------------------------

    def budget_status(self) -> tuple[int | None, int, bool]:
        """Return ``(remaining, floor, ok_to_fetch)`` for pre-flight checks.

        ``ok_to_fetch`` is True iff a recent response has carried an
        ``x-requests-remaining`` header AND the value is at or above the floor.
        Before any request is made ``remaining`` is None and ``ok_to_fetch``
        is False — the conservative answer to "don't know yet."

        Phase 7 pipeline calls this after the initial ``/events`` probe to
        gate the per-event odds pulls.
        """
        remaining = self._last_requests_remaining
        ok = (
            remaining is not None
            and remaining >= self._budget_floor
            and not self._budget_exhausted
        )
        return (remaining, self._budget_floor, ok)

    # ---- AsOfClient interface ------------------------------------------------

    def _fetch(self, cutoff_date: date | None, **kwargs: Any) -> list[EventOdds]:
        today = self._clock().date()
        target = cutoff_date or today
        if cutoff_date is None or cutoff_date >= today:
            return self._fetch_live(target, dry_run=bool(kwargs.get("dry_run")))
        return self._read_historical(target)

    # ---- Live mode -----------------------------------------------------------

    def _fetch_live(self, target: date, *, dry_run: bool) -> list[EventOdds]:
        events, headers = self._call_with_rotation(
            lambda: self._fetch_events(self._api_key or "")
        )
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
            # If all keys exhaust, BudgetExhaustedError / QuotaExhaustedError
            # propagates up — build_market_section already maps any exception
            # to canonical empty-market state, which is the right downstream
            # signal for every unfetched event.
            payload, headers = self._call_with_rotation(
                lambda eid=event_id: self._fetch_event_odds(self._api_key or "", eid)
            )
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

    def _call_with_rotation(self, call):
        """Run an HTTP-issuing callable, rotating to the next key on
        BudgetExhaustedError (our local floor tripped) or QuotaExhaustedError
        (API returned 401 OUT_OF_USAGE_CREDITS).

        Pre-flight: if the active key's budget already tripped on a prior
        request in this client's lifetime, rotate BEFORE attempting the call —
        no point spending a roundtrip just to learn what we already know.

        Raises the original exhaustion error if no more keys remain."""
        while True:
            try:
                self._guard_budget()
            except BudgetExhaustedError as exc:
                if not self._advance_key():
                    raise
                continue
            try:
                return call()
            except (BudgetExhaustedError, QuotaExhaustedError) as exc:
                if not self._advance_key():
                    raise
                continue

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
        _raise_for_quota_or_status(resp, url)
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
        _raise_for_quota_or_status(resp, url)
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


# Scan up to this many indexed env vars: ODDS_API_KEY_STRIKEOUTS,
# _STRIKEOUTS_2, ... _STRIKEOUTS_5. Five is enough headroom for hand-managed
# free-tier rotation; raise if needed.
_MAX_ENV_KEYS = 5


def _resolve_api_keys(api_key: str | list[str] | None) -> list[str]:
    """Build the ordered key list for OddsAPIClient.

    Inputs accepted, in priority order:
      1. ``api_key`` constructor arg: ``str`` (comma-separated OK) or ``list[str]``.
      2. Env vars ``ODDS_API_KEY_STRIKEOUTS``, ``ODDS_API_KEY_STRIKEOUTS_2``,
         ..., ``ODDS_API_KEY_STRIKEOUTS_{N}``. A comma-separated value is also
         accepted (split here).

    Cross-repo bleed guard fires only on the FIRST key resolved from env: if
    ``ODDS_API_KEY_STRIKEOUTS`` equals ``ODDS_API_KEY`` (HR's key value),
    raise. Indexed strikeouts keys (``_STRIKEOUTS_2`` etc.) are NOT checked
    against HR — they're under the dedicated namespace.

    Empties/duplicates removed while preserving order — secrets that aren't
    configured in CI come through as empty strings."""
    if api_key is not None:
        raw = api_key if isinstance(api_key, list) else [api_key]
    else:
        primary = os.environ.get(ENV_KEY_STRIKEOUTS, "")
        # Cross-repo bleed guard on the primary slot only.
        if primary:
            hr_key = os.environ.get(ENV_KEY_HR, "")
            if hr_key and primary == hr_key:
                raise CrossRepoKeyBleedError(
                    f"OddsAPIClient: {ENV_KEY_STRIKEOUTS!r} equals "
                    f"{ENV_KEY_HR!r}. This repo must use a separate API key — "
                    f"sharing depletes the HR-Picks budget. Check your .env."
                )
        raw = [primary]
        for i in range(2, _MAX_ENV_KEYS + 1):
            raw.append(os.environ.get(f"{ENV_KEY_STRIKEOUTS}_{i}", ""))

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not item:
            continue
        for tok in item.split(","):
            tok = tok.strip()
            if tok and tok not in seen:
                out.append(tok)
                seen.add(tok)
    return out


def _is_out_of_credits(resp) -> bool:
    """Return True iff the body indicates monthly quota exhaustion.

    The Odds API returns 401 with JSON body containing
    ``"error_code": "OUT_OF_USAGE_CREDITS"`` when the key's monthly budget
    is gone. Distinguishing this from other 401s (e.g. invalid key) matters —
    we only rotate on quota exhaustion, not on auth failures that would
    affect every key equally."""
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error_code") == "OUT_OF_USAGE_CREDITS":
            return True
    except (ValueError, AttributeError):
        pass
    return "OUT_OF_USAGE_CREDITS" in (resp.text or "")


def _raise_for_quota_or_status(resp, url: str) -> None:
    """Replacement for ``raise_for_status`` that surfaces 401 OUT_OF_USAGE_CREDITS
    as :class:`QuotaExhaustedError` so the rotation loop can act on it. All
    other non-2xx responses propagate as :class:`OddsAPIError`."""
    if resp.status_code == 401 and _is_out_of_credits(resp):
        raise QuotaExhaustedError(
            f"401 OUT_OF_USAGE_CREDITS from {url}: {resp.text[:200]}"
        )
    if not resp.ok:
        raise OddsAPIError(f"{resp.status_code} from {url}: {resp.text[:200]}")


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
