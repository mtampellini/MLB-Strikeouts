"""Tests for the AsOfClient base class.

The base class is the single load-bearing primitive against leakage. If these
tests pass, every subsequent client we build inherits the same guarantee.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from src.data.as_of_context import AsOfClient, LeakageError


class _DummyClient(AsOfClient):
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def _fetch(self, cutoff_date, **kwargs):
        return self._payload


def test_subclass_without_fetch_cannot_instantiate():
    class _BrokenClient(AsOfClient):
        pass

    with pytest.raises(TypeError):
        _BrokenClient()  # type: ignore[abstract]


def test_clean_payload_passes_cutoff_check():
    client = _DummyClient({"game_date": "2024-05-01", "name": "Aaron Nola"})
    result = client.fetch(cutoff_date=date(2024, 6, 1))
    assert result["name"] == "Aaron Nola"


def test_post_cutoff_iso_date_raises_leakage_error():
    client = _DummyClient({"records": [{"date": "2024-06-15"}]})
    with pytest.raises(LeakageError):
        client.fetch(cutoff_date=date(2024, 6, 1))


def test_post_cutoff_datetime_raises_leakage_error():
    client = _DummyClient(
        {"as_of": datetime(2024, 6, 15, 23, 0, tzinfo=timezone.utc)}
    )
    with pytest.raises(LeakageError):
        client.fetch(cutoff_date=date(2024, 6, 1))


def test_nested_post_cutoff_record_raises_leakage_error():
    client = _DummyClient(
        [
            {"name": "ok", "game_date": "2024-05-01"},
            {
                "name": "buried",
                "nested": {"deep": [{"game_date": "2024-07-01"}]},
            },
        ]
    )
    with pytest.raises(LeakageError):
        client.fetch(cutoff_date=date(2024, 6, 1))


def test_live_mode_logs_warning_and_skips_check():
    client = _DummyClient({"future_date": "2099-01-01"})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = client.fetch(cutoff_date=None)
    assert any("LIVE MODE" in str(w.message) for w in captured)
    assert result["future_date"] == "2099-01-01"  # leakage check skipped


def test_future_cutoff_raises_value_error():
    client = _DummyClient([])
    future = date.today() + timedelta(days=1)
    with pytest.raises(ValueError):
        client.fetch(cutoff_date=future)


def test_non_date_cutoff_raises_type_error():
    client = _DummyClient([])
    with pytest.raises(TypeError):
        client.fetch(cutoff_date="2024-06-01")  # type: ignore[arg-type]


def test_strings_that_arent_dates_dont_trip_check():
    """A pitcher name that happens to contain digits should not look like a date."""
    client = _DummyClient(
        {"pitcher": "Mike Trout", "park": "Yankee Stadium 2020-something"}
    )
    # Should NOT raise — the "2020-something" string isn't ISO parseable past head.
    client.fetch(cutoff_date=date(2024, 6, 1))


def test_runtime_check_disabled_via_env(monkeypatch):
    monkeypatch.setenv("ASOF_DISABLE_RUNTIME_CHECK", "1")
    client = _DummyClient({"date": "2099-01-01"})
    # Should NOT raise even though the date is in the future.
    client.fetch(cutoff_date=date(2024, 6, 1))
