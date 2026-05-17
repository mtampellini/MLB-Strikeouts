"""Player name → MLBAM ID resolution with a committed local cache.

The cache lives at ``backend/data/raw/player_id_cache.json``. The file IS
committed to the repo — it shaves significant time off cold starts and
keeps tests deterministic even when pybaseball is unreachable.

Handles common name variants:
- Accents (José Berríos)
- Suffixes (Jr., Sr., II, III, IV, V)
- Trailing whitespace

When pybaseball returns no match, the function returns ``None`` and the
caller MUST skip the prop / pitcher. No guessing.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "player_id_cache.json"
)

_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Strip accents, lowercase, drop suffix, collapse whitespace."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def load_cache(path: Path | None = None) -> dict[str, int]:
    p = path or CACHE_PATH
    if not p.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except (json.JSONDecodeError, OSError, ValueError):
        logger.warning("player_resolver: cache at %s is corrupt; starting fresh", p)
        return {}


def save_cache(cache: dict[str, int], path: Path | None = None) -> None:
    p = path or CACHE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
    )


def _default_lookup(first: str, last: str):
    """Real pybaseball lookup. Wrapped so tests can swap it.

    Strips accents from both the first and last name before calling
    pybaseball (Chadwick stores ASCII names). Uses ``fuzzy=False`` because a
    fuzzy "close enough" match for an unknown name returns a wrong-pitcher's
    ID — silently fetching the wrong pitcher's Statcast data would be a
    correctness disaster. Better to return no match and skip the prop.
    """
    from pybaseball import playerid_lookup  # type: ignore

    return playerid_lookup(
        _strip_accents(last), _strip_accents(first), fuzzy=False
    )


def _strip_accents(s: str) -> str:
    nfd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfd if not unicodedata.combining(ch))


def resolve_player(
    name: str,
    *,
    cache: dict[str, int] | None = None,
    lookup_fn: Callable[[str, str], object] | None = None,
) -> int | None:
    """Resolve a player name to an MLBAM ID. Returns None when no match.

    ``cache`` is mutated in place when a network lookup succeeds. The caller
    is responsible for persisting the cache via :func:`save_cache`.
    """
    if cache is None:
        cache = load_cache()
    lookup_fn = lookup_fn or _default_lookup

    key = normalize_name(name)
    if key in cache:
        return cache[key]

    parts = name.split()
    if len(parts) < 2:
        logger.warning("player_resolver: malformed name %r", name)
        return None

    # Strip suffix token from the raw split as well (covers "Cal Ripken Jr.").
    if _SUFFIX_RE.fullmatch(" " + parts[-1]):
        parts = parts[:-1]
    if len(parts) < 2:
        logger.warning("player_resolver: name has no surname after suffix strip: %r", name)
        return None

    first = parts[0]
    last = " ".join(parts[1:])

    try:
        df = lookup_fn(first, last)
    except Exception as exc:  # pybaseball wraps a wide variety of upstream errors
        logger.warning("player_resolver: lookup failed for %r: %s", name, exc)
        return None

    mlbam_id = _extract_mlbam_id(df, name)
    if mlbam_id is None:
        return None

    cache[key] = mlbam_id
    return mlbam_id


def _extract_mlbam_id(df, name: str) -> int | None:
    if df is None:
        logger.warning("player_resolver: no result row for %r", name)
        return None
    if hasattr(df, "empty"):
        if df.empty:
            logger.warning("player_resolver: no result row for %r", name)
            return None
        row = df.iloc[0]
        try:
            mid = int(row["key_mlbam"])
        except (KeyError, TypeError, ValueError):
            logger.warning("player_resolver: row missing key_mlbam for %r", name)
            return None
        return mid if mid > 0 else None
    if isinstance(df, list) and df:
        try:
            return int(df[0]["key_mlbam"])
        except (KeyError, TypeError, ValueError):
            return None
    return None
