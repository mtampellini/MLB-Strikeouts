"""Tests for player name → MLBAM id resolution."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import player_resolver as pr


def test_normalize_strips_accents():
    assert pr.normalize_name("José Berríos") == "jose berrios"


def test_normalize_strips_suffix():
    assert pr.normalize_name("Cal Ripken Jr.") == "cal ripken"
    assert pr.normalize_name("Ken Griffey III") == "ken griffey"


def test_normalize_collapses_whitespace():
    assert pr.normalize_name("  Aaron   Nola  ") == "aaron nola"


def test_resolve_uses_cache_hit():
    cache = {"aaron nola": 605400}
    # lookup_fn should NOT be called because the cache hits.
    def fail(*a, **k):
        raise AssertionError("lookup_fn was called on a cache hit")

    assert pr.resolve_player("Aaron Nola", cache=cache, lookup_fn=fail) == 605400


def test_resolve_writes_back_to_cache():
    cache: dict[str, int] = {}

    def fake_lookup(first, last):
        return pd.DataFrame([{"key_mlbam": 592450}])

    out = pr.resolve_player("Aaron Judge", cache=cache, lookup_fn=fake_lookup)
    assert out == 592450
    assert cache["aaron judge"] == 592450


def test_resolve_handles_accent_via_cache():
    """The cache key is normalized — JóSé → jose."""
    cache: dict[str, int] = {}

    def fake_lookup(first, last):
        assert last in ("Berríos", "Berrios")
        return pd.DataFrame([{"key_mlbam": 605244}])

    assert pr.resolve_player("José Berríos", cache=cache, lookup_fn=fake_lookup) == 605244
    # Subsequent lookup with a different accent style hits cache.
    assert pr.resolve_player("Jose Berrios", cache=cache, lookup_fn=fail_if_called) == 605244


def fail_if_called(*a, **k):
    raise AssertionError("lookup_fn was called when cache should have hit")


def test_resolve_unknown_name_returns_none(caplog):
    def empty_lookup(first, last):
        return pd.DataFrame()

    out = pr.resolve_player("Nonexistent Player", cache={}, lookup_fn=empty_lookup)
    assert out is None


def test_resolve_handles_lookup_exception(caplog):
    def angry_lookup(first, last):
        raise RuntimeError("upstream lookup is on fire")

    out = pr.resolve_player("Aaron Nola", cache={}, lookup_fn=angry_lookup)
    assert out is None


def test_resolve_handles_suffix_in_lookup():
    cache: dict[str, int] = {}
    captured = {}

    def fake_lookup(first, last):
        captured["first"] = first
        captured["last"] = last
        return pd.DataFrame([{"key_mlbam": 12345}])

    pr.resolve_player("Vladimir Guerrero Jr.", cache=cache, lookup_fn=fake_lookup)
    # The suffix should have been stripped before the network lookup.
    assert captured["last"] == "Guerrero"


def test_resolve_malformed_single_name_returns_none():
    out = pr.resolve_player("Madison", cache={}, lookup_fn=fail_if_called)
    assert out is None


def test_cache_round_trip(tmp_path: Path):
    cache = {"aaron nola": 605400, "aaron judge": 592450}
    p = tmp_path / "cache.json"
    pr.save_cache(cache, path=p)
    loaded = pr.load_cache(path=p)
    assert loaded == cache
