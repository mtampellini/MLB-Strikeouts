"""StatcastClient: pitch-by-pitch data via pybaseball, cached locally.

Two access patterns:

- ``get_game_pitches(game_pk, game_date, *, cutoff_date)`` — every pitch
  thrown in a single game. Cached at
  ``data/raw/statcast/{YYYY}/game_{game_pk}.parquet``.
- ``get_pitcher_pitches(pitcher_id, *, cutoff_date, days_back)`` — every
  pitch a specific pitcher threw in the trailing window. Backed by
  pybaseball.statcast_pitcher (direct query is faster than walking games
  when you only need one pitcher). Cached at
  ``data/raw/statcast/pitchers/{pitcher_id}.parquet`` with incremental
  refresh.
- ``get_batter_pitches`` — symmetric, ``data/raw/statcast/batters/``.

Cutoff enforcement: every returned DataFrame is filtered to
``game_date <= cutoff_date`` before being handed back. The AsOfClient
runtime check walks the result and double-confirms.

Empty results (no games in window, no Statcast for a player) return an
empty DataFrame, not None — the feature pipeline can then run its own
skip logic without nil checks.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .as_of_context import AsOfClient

logger = logging.getLogger(__name__)

STATCAST_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "statcast"
)

# Columns we always require downstream (Phase 3 features). pybaseball's
# Statcast response carries far more — we keep the full set but enumerate
# essentials so a parser regression is loud.
ESSENTIAL_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "game_date",
    "pitcher",
    "batter",
    "pitch_type",
    "description",
    "events",
    "zone",
    "release_speed",
    "plate_x",
    "plate_z",
    "stand",
    "p_throws",
)


# -------- Default fetchers (injectable) -------------------------------------


def _default_fetch_pitcher(start: date, end: date, pitcher_id: int) -> pd.DataFrame:
    """Real pybaseball call. Wrapped so tests can swap it."""
    from pybaseball import statcast_pitcher  # type: ignore

    df = statcast_pitcher(start.isoformat(), end.isoformat(), pitcher_id)
    return _normalize_dataframe(df)


def _default_fetch_batter(start: date, end: date, batter_id: int) -> pd.DataFrame:
    from pybaseball import statcast_batter  # type: ignore

    df = statcast_batter(start.isoformat(), end.isoformat(), batter_id)
    return _normalize_dataframe(df)


def _default_fetch_game(game_date: date, game_pk: int) -> pd.DataFrame:
    """Pull one day of Statcast and filter to a specific game.

    pybaseball.statcast(start_dt, end_dt) returns every pitch in the date
    range. For a single game we still hit a one-day window since pybaseball
    has no by-game endpoint; the cache then keys by game_pk for reuse.
    """
    from pybaseball import statcast  # type: ignore

    df = statcast(game_date.isoformat(), game_date.isoformat())
    df = _normalize_dataframe(df)
    if df.empty:
        return df
    return df[df["game_pk"] == game_pk].reset_index(drop=True)


def _normalize_dataframe(df: Any) -> pd.DataFrame:
    """Coerce pybaseball output to a stable contract."""
    if df is None:
        return pd.DataFrame(columns=ESSENTIAL_COLUMNS)
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=ESSENTIAL_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=ESSENTIAL_COLUMNS)
    if "game_date" in df.columns:
        df["game_date"] = df["game_date"].astype(str).str[:10]
    return df


# -------- The client --------------------------------------------------------


class StatcastClient(AsOfClient):
    def __init__(
        self,
        *,
        fetch_pitcher: Callable[[date, date, int], pd.DataFrame] | None = None,
        fetch_batter: Callable[[date, date, int], pd.DataFrame] | None = None,
        fetch_game: Callable[[date, int], pd.DataFrame] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._fetch_pitcher = fetch_pitcher or _default_fetch_pitcher
        self._fetch_batter = fetch_batter or _default_fetch_batter
        self._fetch_game = fetch_game or _default_fetch_game
        self._cache_dir = cache_dir or STATCAST_CACHE_DIR

    def _fetch(self, cutoff_date: date | None, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "StatcastClient has named endpoints; call get_pitcher_pitches, "
            "get_batter_pitches, or get_game_pitches."
        )

    # ---- Per-game cache ----------------------------------------------------

    def get_game_pitches(
        self,
        game_pk: int,
        game_date: date,
        *,
        cutoff_date: date | None,
    ) -> pd.DataFrame:
        self._asof_pre(cutoff_date)
        if cutoff_date is not None and game_date > cutoff_date:
            raise ValueError(
                f"StatcastClient.get_game_pitches: game_date {game_date} > "
                f"cutoff_date {cutoff_date}"
            )
        path = self._game_cache_path(game_pk, game_date)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = self._fetch_game(game_date, game_pk)
            self._persist(df, path)
        self._asof_post(_df_payload(df), cutoff_date)
        return df

    # ---- Per-pitcher window ------------------------------------------------

    def get_pitcher_pitches(
        self,
        pitcher_id: int,
        *,
        cutoff_date: date | None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        return self._get_player_pitches(
            player_id=pitcher_id,
            cutoff_date=cutoff_date,
            days_back=days_back,
            kind="pitchers",
            fetch_fn=self._fetch_pitcher,
        )

    def get_batter_pitches(
        self,
        batter_id: int,
        *,
        cutoff_date: date | None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        return self._get_player_pitches(
            player_id=batter_id,
            cutoff_date=cutoff_date,
            days_back=days_back,
            kind="batters",
            fetch_fn=self._fetch_batter,
        )

    # ---- Shared per-player logic ------------------------------------------

    def _get_player_pitches(
        self,
        *,
        player_id: int,
        cutoff_date: date | None,
        days_back: int,
        kind: str,
        fetch_fn: Callable[[date, date, int], pd.DataFrame],
    ) -> pd.DataFrame:
        self._asof_pre(cutoff_date)
        end = cutoff_date or date.today()
        start = end - timedelta(days=days_back)

        path = self._player_cache_path(kind, player_id)
        fetched_through = self._read_fetched_through(path)
        cached = pd.read_parquet(path) if path.exists() else None

        if cached is not None and fetched_through is not None:
            if fetched_through >= end:
                # Cache already covers the requested window. Use as-is.
                df = cached
            else:
                next_start = fetched_through + timedelta(days=1)
                fresh = fetch_fn(next_start, end, player_id)
                df = pd.concat([cached, fresh], ignore_index=True)
                dedup_keys = ["game_pk", "pitcher", "batter"]
                if "pitch_number" in df.columns:
                    dedup_keys.append("pitch_number")
                df = df.drop_duplicates(subset=dedup_keys).reset_index(drop=True)
                self._persist(df, path, fetched_through=end)
        else:
            df = fetch_fn(start, end, player_id)
            self._persist(df, path, fetched_through=end)

        # Slice to the requested window and never return rows past cutoff.
        df = df[
            (df["game_date"] >= start.isoformat())
            & (df["game_date"] <= end.isoformat())
        ].reset_index(drop=True)
        self._asof_post(_df_payload(df), cutoff_date)
        return df

    def _read_fetched_through(self, parquet_path: Path) -> date | None:
        meta_path = parquet_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
            return date.fromisoformat(blob["fetched_through"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def _write_fetched_through(
        self, parquet_path: Path, fetched_through: date
    ) -> None:
        meta_path = parquet_path.with_suffix(".meta.json")
        try:
            meta_path.write_text(
                json.dumps({"fetched_through": fetched_through.isoformat()}),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "StatcastClient: failed to write meta sidecar %s: %s",
                meta_path, exc,
            )

    # ---- Cache helpers ----------------------------------------------------

    def _game_cache_path(self, game_pk: int, game_date: date) -> Path:
        return (
            self._cache_dir
            / "games"
            / str(game_date.year)
            / f"game_{game_pk}.parquet"
        )

    def _player_cache_path(self, kind: str, player_id: int) -> Path:
        return self._cache_dir / kind / f"{player_id}.parquet"

    def _persist(
        self,
        df: pd.DataFrame,
        path: Path,
        *,
        fetched_through: date | None = None,
    ) -> None:
        if df is None or df.empty:
            # Still write a marker so we don't re-fetch on every call. An empty
            # parquet round-trips cleanly through pandas.
            df = pd.DataFrame(columns=ESSENTIAL_COLUMNS)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            if fetched_through is not None:
                self._write_fetched_through(path, fetched_through)
        except OSError as exc:
            logger.warning(
                "StatcastClient: failed to persist cache at %s: %s", path, exc
            )


# -------- AsOf payload helper -----------------------------------------------


def _df_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Hand the AsOf leakage walker just the game_date column.

    Walking the whole DataFrame would be O(rows × walker overhead) and these
    frames are big. The max game_date string is sufficient to catch leakage.
    """
    if df is None or df.empty or "game_date" not in df.columns:
        return {}
    return {"max_game_date": str(df["game_date"].max())}
