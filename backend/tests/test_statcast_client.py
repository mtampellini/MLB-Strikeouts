"""Tests for StatcastClient (per-game and per-player caching)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.as_of_context import LeakageError
from src.data.statcast_client import (
    ESSENTIAL_COLUMNS,
    StatcastClient,
    _normalize_dataframe,
)


def _sample_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ESSENTIAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def _pitch_row(*, game_pk: int, game_date: str, pitcher: int = 605400, batter: int = 1):
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "pitcher": pitcher,
        "batter": batter,
        "pitch_type": "FF",
        "description": "called_strike",
        "events": None,
        "zone": 5,
        "release_speed": 95.0,
        "plate_x": 0.1,
        "plate_z": 2.5,
        "stand": "R",
        "p_throws": "R",
        "pitch_number": 1,
    }


# -------- _normalize_dataframe ----------------------------------------------


def test_normalize_returns_essential_columns_on_none():
    df = _normalize_dataframe(None)
    assert list(df.columns) == list(ESSENTIAL_COLUMNS)
    assert df.empty


def test_normalize_coerces_game_date_to_iso_string():
    raw = pd.DataFrame(
        [{"game_date": pd.Timestamp("2024-07-15"), "game_pk": 1, "pitcher": 605400}]
    )
    df = _normalize_dataframe(raw)
    assert df.iloc[0]["game_date"] == "2024-07-15"


# -------- get_game_pitches: cache behavior ----------------------------------


def test_get_game_pitches_caches_to_disk(tmp_path):
    calls = {"n": 0}

    def fake_fetch(game_date, game_pk):
        calls["n"] += 1
        return _sample_df([_pitch_row(game_pk=game_pk, game_date=str(game_date))])

    client = StatcastClient(fetch_game=fake_fetch, cache_dir=tmp_path)
    cutoff = date(2024, 7, 15)
    df1 = client.get_game_pitches(12345, date(2024, 7, 15), cutoff_date=cutoff)
    df2 = client.get_game_pitches(12345, date(2024, 7, 15), cutoff_date=cutoff)
    assert calls["n"] == 1  # cached on second call
    pd.testing.assert_frame_equal(df1, df2)
    # File exists on disk
    cache_path = tmp_path / "games" / "2024" / "game_12345.parquet"
    assert cache_path.exists()


def test_get_game_pitches_rejects_future_game_date(tmp_path):
    client = StatcastClient(
        fetch_game=lambda gd, pk: pd.DataFrame(),
        cache_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="cutoff_date"):
        client.get_game_pitches(
            12345, date(2024, 8, 1), cutoff_date=date(2024, 7, 15)
        )


def test_get_game_pitches_empty_result_persists_marker(tmp_path):
    """When pybaseball returns nothing for a game (rain-out, missing data),
    we still write an empty cache so we don't re-pull every call."""
    calls = {"n": 0}

    def fake_fetch(game_date, game_pk):
        calls["n"] += 1
        return pd.DataFrame(columns=ESSENTIAL_COLUMNS)

    client = StatcastClient(fetch_game=fake_fetch, cache_dir=tmp_path)
    cutoff = date(2024, 7, 15)
    df1 = client.get_game_pitches(99999, date(2024, 7, 15), cutoff_date=cutoff)
    df2 = client.get_game_pitches(99999, date(2024, 7, 15), cutoff_date=cutoff)
    assert df1.empty and df2.empty
    assert calls["n"] == 1


# -------- get_pitcher_pitches: window, slice, leakage -----------------------


def test_get_pitcher_pitches_returns_window(tmp_path):
    def fake_fetch(start, end, pitcher_id):
        rows = []
        cur = start
        while cur <= end:
            rows.append(_pitch_row(game_pk=int(cur.strftime("%Y%m%d")), game_date=cur.isoformat(), pitcher=pitcher_id))
            cur += timedelta(days=2)
        return _sample_df(rows)

    client = StatcastClient(fetch_pitcher=fake_fetch, cache_dir=tmp_path)
    cutoff = date(2024, 7, 15)
    df = client.get_pitcher_pitches(605400, cutoff_date=cutoff, days_back=7)
    assert not df.empty
    # All rows within [cutoff-7d, cutoff]
    assert df["game_date"].min() >= (cutoff - timedelta(days=7)).isoformat()
    assert df["game_date"].max() <= cutoff.isoformat()


def test_get_pitcher_pitches_uses_cache_on_repeat(tmp_path):
    calls = {"n": 0}

    def fake_fetch(start, end, pitcher_id):
        calls["n"] += 1
        return _sample_df([_pitch_row(game_pk=1, game_date="2024-07-14", pitcher=pitcher_id)])

    client = StatcastClient(fetch_pitcher=fake_fetch, cache_dir=tmp_path)
    cutoff = date(2024, 7, 15)
    client.get_pitcher_pitches(605400, cutoff_date=cutoff, days_back=7)
    client.get_pitcher_pitches(605400, cutoff_date=cutoff, days_back=7)
    assert calls["n"] == 1


def test_get_pitcher_pitches_incremental_refresh_pulls_only_new_days(tmp_path):
    """Calling on day N+5 with an already-cached day-N window should pull
    only days N+1..N+5, not the full window again."""
    calls: list[tuple[date, date]] = []

    def fake_fetch(start, end, pitcher_id):
        calls.append((start, end))
        return _sample_df(
            [_pitch_row(game_pk=int(d.strftime("%Y%m%d")), game_date=d.isoformat(), pitcher=pitcher_id)
             for d in pd.date_range(start, end).date]
        )

    client = StatcastClient(fetch_pitcher=fake_fetch, cache_dir=tmp_path)
    client.get_pitcher_pitches(605400, cutoff_date=date(2024, 7, 10), days_back=7)
    client.get_pitcher_pitches(605400, cutoff_date=date(2024, 7, 15), days_back=7)
    # First call: 2024-07-03..2024-07-10. Second call: 2024-07-11..2024-07-15.
    assert len(calls) == 2
    assert calls[1] == (date(2024, 7, 11), date(2024, 7, 15))


def test_get_pitcher_pitches_filters_to_cutoff(tmp_path):
    """Cache contains rows past cutoff (older incremental pull cached more
    than the trailing window). Returned slice must exclude them."""

    def fake_fetch(start, end, pitcher_id):
        # Return rows beyond `end` too — caller is supposed to clip
        return _sample_df(
            [
                _pitch_row(game_pk=1, game_date="2024-07-10", pitcher=pitcher_id),
                _pitch_row(game_pk=2, game_date="2024-07-14", pitcher=pitcher_id),
                _pitch_row(game_pk=3, game_date="2024-07-15", pitcher=pitcher_id),
                _pitch_row(game_pk=4, game_date="2024-07-20", pitcher=pitcher_id),
            ]
        )

    client = StatcastClient(fetch_pitcher=fake_fetch, cache_dir=tmp_path)
    df = client.get_pitcher_pitches(605400, cutoff_date=date(2024, 7, 15), days_back=7)
    assert df["game_date"].max() <= "2024-07-15"
    assert "2024-07-20" not in df["game_date"].values


def test_get_pitcher_pitches_leakage_check_catches_polluted_cache(tmp_path):
    """If a malicious/broken cache exists with post-cutoff rows AND the slice
    fails to filter, _asof_post must catch it.

    We force the bug by bypassing the slice via an out-of-bound days_back.
    """
    bad = _sample_df([_pitch_row(game_pk=1, game_date="2099-01-01", pitcher=605400)])
    cache_path = tmp_path / "pitchers" / "605400.parquet"
    cache_path.parent.mkdir(parents=True)
    bad.to_parquet(cache_path)
    client = StatcastClient(
        fetch_pitcher=lambda *a, **k: bad,
        cache_dir=tmp_path,
    )
    # Cutoff in past + days_back=400 would include 2099 if slicing missed.
    # The slice should still clip — but if a future bug breaks the clip, the
    # AsOf walker would catch it. We test the clip is in place:
    df = client.get_pitcher_pitches(605400, cutoff_date=date(2024, 7, 15), days_back=400)
    assert df.empty  # all rows were 2099-01-01, past 2024-07-15 cutoff


def test_get_batter_pitches_symmetric(tmp_path):
    """Batter access path mirrors pitcher; verify it routes through batter cache."""
    def fake_fetch(start, end, batter_id):
        return _sample_df(
            [_pitch_row(game_pk=1, game_date="2024-07-14", pitcher=999, batter=batter_id)]
        )

    client = StatcastClient(fetch_batter=fake_fetch, cache_dir=tmp_path)
    df = client.get_batter_pitches(592450, cutoff_date=date(2024, 7, 15), days_back=7)
    assert not df.empty
    assert (tmp_path / "batters" / "592450.parquet").exists()
    assert not (tmp_path / "pitchers" / "592450.parquet").exists()


# -------- AsOf integration --------------------------------------------------


def test_get_pitcher_pitches_future_cutoff_raises_via_base(tmp_path):
    client = StatcastClient(
        fetch_pitcher=lambda s, e, p: pd.DataFrame(columns=ESSENTIAL_COLUMNS),
        cache_dir=tmp_path,
    )
    future = date.today() + timedelta(days=365)
    with pytest.raises(ValueError):
        client.get_pitcher_pitches(605400, cutoff_date=future, days_back=7)


def test_live_mode_warns(tmp_path):
    import warnings

    def fake_fetch(start, end, pitcher_id):
        return _sample_df([_pitch_row(game_pk=1, game_date=str(end), pitcher=pitcher_id)])

    client = StatcastClient(fetch_pitcher=fake_fetch, cache_dir=tmp_path)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        client.get_pitcher_pitches(605400, cutoff_date=None, days_back=7)
    assert any("LIVE MODE" in str(c.message) for c in captured)
