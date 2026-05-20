"""Tests for OddsAPIClient (allowlist, snapshots, devig, budget, historical)."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.odds_client import (
    ALLOWED_BOOKS,
    BudgetExhaustedError,
    CrossRepoKeyBleedError,
    DEFAULT_BUDGET_FLOOR,
    ENV_KEY_HR,
    ENV_KEY_STRIKEOUTS,
    EventOdds,
    MissingSnapshotError,
    OddsAPIClient,
    PitcherProp,
    QuotaExhaustedError,
    american_to_implied,
    devig_two_sided,
    parse_event_odds_response,
)


def _id_resolver():
    table = {
        "Drew Rasmussen": 656876,
        "Eury Pérez": 691587,
    }

    def resolve(name):
        return table.get(name)

    return resolve


# -------- parse_event_odds_response -----------------------------------------


def test_parse_drops_caesars(sample_event_odds_payload):
    ev = parse_event_odds_response(
        sample_event_odds_payload, resolve_id=_id_resolver()
    )
    assert all(p.book in ALLOWED_BOOKS for p in ev.pitcher_props)
    books = {p.book for p in ev.pitcher_props}
    assert "caesars" not in books


def test_parse_collects_alt_lines_per_pitcher(sample_event_odds_payload):
    ev = parse_event_odds_response(
        sample_event_odds_payload, resolve_id=_id_resolver()
    )
    rasmussen_fd = [
        p for p in ev.pitcher_props
        if p.pitcher_name == "Drew Rasmussen" and p.book == "fanduel"
    ]
    assert len(rasmussen_fd) == 1
    lines = rasmussen_fd[0].lines
    # 3 Over + 2 Under = 5 alt entries
    assert len(lines) == 5
    sides = {ln.side for ln in lines}
    assert sides == {"Over", "Under"}


def test_parse_drops_unresolvable_pitcher(sample_event_odds_payload):
    """When the resolver returns None, the prop is dropped (no synthetic IDs)."""
    def reject_all(name):
        return None

    ev = parse_event_odds_response(sample_event_odds_payload, resolve_id=reject_all)
    assert ev.pitcher_props == ()


def test_parse_handles_missing_resolver(sample_event_odds_payload):
    """Without a resolver we keep props but with id=None (for unit-testable parsing)."""
    ev = parse_event_odds_response(sample_event_odds_payload, resolve_id=None)
    assert all(p.pitcher_mlbam_id is None for p in ev.pitcher_props)


# -------- Devig --------------------------------------------------------------


def test_american_to_implied_positive():
    assert abs(american_to_implied(100) - 0.5) < 1e-9


def test_american_to_implied_negative():
    assert abs(american_to_implied(-110) - 0.5238095) < 1e-6


def test_devig_two_sided_symmetric_market():
    """A pair priced at -110 / -110 should devig to ~50/50."""
    result = devig_two_sided(-110, -110)
    assert abs(result["over_true"] - 0.5) < 1e-9
    assert abs(result["under_true"] - 0.5) < 1e-9
    assert result["vig"] > 0


def test_devig_two_sided_asymmetric():
    """-180 / +145 paired line — devig must yield a coherent split summing to 1."""
    result = devig_two_sided(-180, 145)
    assert abs(result["over_true"] + result["under_true"] - 1.0) < 1e-9


def test_devig_rejects_invalid_prices():
    with pytest.raises(ValueError):
        devig_two_sided(0, 0)


# -------- Live-mode fetch ----------------------------------------------------


def _stub_events_payload():
    return [
        {
            "id": "evt-1",
            "commence_time": "2026-05-17T23:05:00Z",
            "home_team": "Rays",
            "away_team": "Marlins",
        }
    ]


def _stub_headers(remaining: int = 400) -> dict[str, str]:
    return {"x-requests-remaining": str(remaining)}


def test_live_mode_writes_snapshot(
    sample_event_odds_payload, tmp_path, monkeypatch
):
    client = OddsAPIClient(
        api_key="test-key",
        fetch_events=lambda key: (_stub_events_payload(), _stub_headers(300)),
        fetch_event_odds=lambda key, eid: (sample_event_odds_payload, _stub_headers(299)),
        snapshot_dir=tmp_path,
        player_lookup_fn=None,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    # Inject an id resolver via cache.
    monkeypatch.setattr(
        client, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n)
    )
    events = client.fetch(cutoff_date=date(2026, 5, 17))
    assert len(events) == 1
    snap = events[0].snapshot_path
    assert snap and Path(snap).exists()
    assert "1130_event-evt-1.json" in snap


def test_snapshot_idempotent_within_same_minute(
    sample_event_odds_payload, tmp_path, monkeypatch
):
    same_minute = datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc)

    def make_client(payload):
        c = OddsAPIClient(
            api_key="x",
            fetch_events=lambda key: (_stub_events_payload(), _stub_headers()),
            fetch_event_odds=lambda key, eid: (payload, _stub_headers()),
            snapshot_dir=tmp_path,
            player_lookup_fn=None,
            clock=lambda: same_minute,
        )
        monkeypatch.setattr(c, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n))
        return c

    c1 = make_client(sample_event_odds_payload)
    c1.fetch(cutoff_date=date(2026, 5, 17))
    files1 = sorted(p.name for p in (tmp_path / "2026-05-17").glob("*.json"))

    # Same minute → same file (idempotent overwrite).
    c2 = make_client(sample_event_odds_payload)
    c2.fetch(cutoff_date=date(2026, 5, 17))
    files2 = sorted(p.name for p in (tmp_path / "2026-05-17").glob("*.json"))
    assert files1 == files2


def test_snapshot_creates_new_file_for_new_minute(
    sample_event_odds_payload, tmp_path, monkeypatch
):
    def make_client(minute: int):
        c = OddsAPIClient(
            api_key="x",
            fetch_events=lambda key: (_stub_events_payload(), _stub_headers()),
            fetch_event_odds=lambda key, eid: (sample_event_odds_payload, _stub_headers()),
            snapshot_dir=tmp_path,
            player_lookup_fn=None,
            clock=lambda: datetime(2026, 5, 17, 11, minute, tzinfo=timezone.utc),
        )
        monkeypatch.setattr(c, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n))
        return c

    make_client(30).fetch(cutoff_date=date(2026, 5, 17))
    make_client(45).fetch(cutoff_date=date(2026, 5, 17))
    files = sorted(p.name for p in (tmp_path / "2026-05-17").glob("*.json"))
    assert len(files) == 2


def test_budget_refusal_below_floor(
    sample_event_odds_payload, tmp_path, monkeypatch
):
    # First /events call returns headers showing only 30 remaining (< floor 50).
    client = OddsAPIClient(
        api_key="x",
        fetch_events=lambda key: (_stub_events_payload() * 2, _stub_headers(30)),
        fetch_event_odds=lambda key, eid: (sample_event_odds_payload, _stub_headers(29)),
        snapshot_dir=tmp_path,
        budget_floor=50,
        player_lookup_fn=None,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(client, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n))
    with pytest.raises(BudgetExhaustedError):
        client.fetch(cutoff_date=date(2026, 5, 17))


def test_dry_run_skips_event_odds_calls(tmp_path):
    odds_calls = {"n": 0}

    def bomb(key, eid):
        odds_calls["n"] += 1
        raise AssertionError("event-odds endpoint should not be hit on dry run")

    client = OddsAPIClient(
        api_key="x",
        fetch_events=lambda key: (_stub_events_payload(), _stub_headers()),
        fetch_event_odds=bomb,
        snapshot_dir=tmp_path,
        player_lookup_fn=None,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    events = client.fetch(cutoff_date=date(2026, 5, 17), dry_run=True)
    assert len(events) == 1
    assert events[0].pitcher_props == ()
    assert odds_calls["n"] == 0


# -------- Historical mode ----------------------------------------------------


def test_historical_mode_reads_committed_snapshot(
    sample_event_odds_payload, tmp_path, monkeypatch
):
    target = date(2024, 7, 15)
    day_dir = tmp_path / target.isoformat()
    day_dir.mkdir(parents=True)
    (day_dir / "1100_event-evt-1.json").write_text(
        json.dumps(sample_event_odds_payload), encoding="utf-8"
    )
    (day_dir / "1500_event-evt-1.json").write_text(
        json.dumps(sample_event_odds_payload), encoding="utf-8"
    )

    client = OddsAPIClient(
        api_key="x",
        snapshot_dir=tmp_path,
        player_lookup_fn=None,
        clock=lambda: datetime(2024, 12, 31, tzinfo=timezone.utc),  # "today" >> target
    )
    monkeypatch.setattr(client, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n))
    events = client.fetch(cutoff_date=target)
    assert len(events) == 1
    # The 1500 snapshot wins over 1100.
    assert events[0].snapshot_path is not None
    assert "1500_event-evt-1.json" in events[0].snapshot_path


def test_historical_missing_snapshot_raises(tmp_path):
    client = OddsAPIClient(
        api_key="x",
        snapshot_dir=tmp_path,
        player_lookup_fn=None,
        clock=lambda: datetime(2024, 12, 31, tzinfo=timezone.utc),
    )
    with pytest.raises(MissingSnapshotError):
        client.fetch(cutoff_date=date(2024, 7, 15))


# -------- API key resolution + cross-repo bleed guard -----------------------


def test_missing_strikeouts_key_raises(monkeypatch):
    """If ODDS_API_KEY_STRIKEOUTS isn't set, __init__ refuses to build a client.

    Phase 2 spec: cross-repo key bleed must not happen silently.
    """
    monkeypatch.delenv(ENV_KEY_STRIKEOUTS, raising=False)
    monkeypatch.delenv(ENV_KEY_HR, raising=False)
    with pytest.raises(CrossRepoKeyBleedError, match=ENV_KEY_STRIKEOUTS):
        OddsAPIClient()


def test_strikeouts_key_equals_hr_key_raises(monkeypatch):
    monkeypatch.setenv(ENV_KEY_STRIKEOUTS, "shared-value")
    monkeypatch.setenv(ENV_KEY_HR, "shared-value")
    with pytest.raises(CrossRepoKeyBleedError, match="equals"):
        OddsAPIClient()


def test_strikeouts_key_distinct_from_hr_passes(monkeypatch):
    monkeypatch.setenv(ENV_KEY_STRIKEOUTS, "strikeouts-key")
    monkeypatch.setenv(ENV_KEY_HR, "hr-key")
    client = OddsAPIClient()  # no raise
    assert client._api_key == "strikeouts-key"


def test_explicit_api_key_bypasses_env_check(monkeypatch):
    """Explicit api_key= injection always wins, even with no env set."""
    monkeypatch.delenv(ENV_KEY_STRIKEOUTS, raising=False)
    monkeypatch.delenv(ENV_KEY_HR, raising=False)
    client = OddsAPIClient(api_key="explicit-test-key")
    assert client._api_key == "explicit-test-key"


# -------- Budget floor + budget_status --------------------------------------


def test_default_budget_floor_is_100():
    assert DEFAULT_BUDGET_FLOOR == 100


def test_default_budget_floor_refuses_at_99(sample_event_odds_payload, tmp_path, monkeypatch):
    """At default floor (100), a remaining count of 99 trips refusal."""
    client = OddsAPIClient(
        api_key="x",
        fetch_events=lambda key: (_stub_events_payload() * 2, _stub_headers(99)),
        fetch_event_odds=lambda key, eid: (sample_event_odds_payload, _stub_headers(98)),
        snapshot_dir=tmp_path,
        player_lookup_fn=None,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(client, "_resolve_id", lambda n: 1)
    with pytest.raises(BudgetExhaustedError):
        client.fetch(cutoff_date=date(2026, 5, 17))


# -------- Multi-key failover ------------------------------------------------


def test_multi_key_resolution_from_indexed_env_vars(monkeypatch):
    """_resolve_api_keys scans ODDS_API_KEY_STRIKEOUTS through _STRIKEOUTS_5."""
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS",   "key1")
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS_2", "key2")
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS_3", "key3")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    client = OddsAPIClient()
    assert client.num_keys == 3
    assert client._api_keys == ["key1", "key2", "key3"]
    assert client._api_key == "key1"
    assert client.active_key_index == 0


def test_multi_key_dedup_preserves_order(monkeypatch):
    """Duplicates across slots collapse; first occurrence wins."""
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS",   "k")
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS_2", "k")
    monkeypatch.setenv("ODDS_API_KEY_STRIKEOUTS_3", "k2")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    client = OddsAPIClient()
    assert client._api_keys == ["k", "k2"]


def test_failover_advances_to_next_key_on_budget_exhausted(
    sample_event_odds_payload, tmp_path, monkeypatch,
):
    """First key's events fetch succeeds (returning budget=5 < floor=100),
    but the post-fetch budget trip causes rotation before the per-event
    odds fetches. Second key handles the per-event work."""
    events_keys, odds_keys = [], []
    def events_stub(key):
        events_keys.append(key)
        # k1 returns budget=5 (below floor) → trips _budget_exhausted; k2 healthy.
        rem = 5 if key == "k1" else 500
        return _stub_events_payload(), _stub_headers(rem)
    def event_odds_stub(key, eid):
        odds_keys.append(key)
        return sample_event_odds_payload, _stub_headers(495)
    client = OddsAPIClient(
        api_key=["k1", "k2"],
        fetch_events=events_stub,
        fetch_event_odds=event_odds_stub,
        snapshot_dir=tmp_path,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(client, "_resolve_id", lambda n: 1)
    out = client.fetch(cutoff_date=date(2026, 5, 17))
    # Events fetch consumed k1's last request; per-event fetches rotated to k2.
    assert events_keys == ["k1"]
    assert all(k == "k2" for k in odds_keys)
    assert client.active_key_index == 1
    assert len(out) > 0


def test_failover_advances_on_quota_exhausted_401(
    sample_event_odds_payload, tmp_path, monkeypatch,
):
    """First key's per-event fetch raises QuotaExhaustedError (simulating 401
    OUT_OF_USAGE_CREDITS); rotation falls through to second key."""
    calls = {"events": 0, "odds": 0}
    def events_stub(key):
        calls["events"] += 1
        return _stub_events_payload(), _stub_headers(500)
    def event_odds_stub(key, eid):
        calls["odds"] += 1
        if key == "k1":
            raise QuotaExhaustedError("simulated 401 OUT_OF_USAGE_CREDITS on k1")
        return sample_event_odds_payload, _stub_headers(495)
    client = OddsAPIClient(
        api_key=["k1", "k2"],
        fetch_events=events_stub,
        fetch_event_odds=event_odds_stub,
        snapshot_dir=tmp_path,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(client, "_resolve_id", lambda n: 1)
    out = client.fetch(cutoff_date=date(2026, 5, 17))
    assert client.active_key_index == 1
    assert len(out) > 0


def test_failover_raises_when_all_keys_exhausted(
    sample_event_odds_payload, tmp_path, monkeypatch,
):
    """Every key trips quota: error propagates, not silently swallowed."""
    def events_stub(key):
        raise QuotaExhaustedError(f"simulated quota out on {key}")
    client = OddsAPIClient(
        api_key=["k1", "k2"],
        fetch_events=events_stub,
        fetch_event_odds=lambda k, eid: (sample_event_odds_payload, _stub_headers(0)),
        snapshot_dir=tmp_path,
        clock=lambda: datetime(2026, 5, 17, 11, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(QuotaExhaustedError):
        client.fetch(cutoff_date=date(2026, 5, 17))
    # Walked through both keys before giving up.
    assert client.active_key_index == 1


def test_is_out_of_credits_detects_json_error_code():
    """_is_out_of_credits matches the 401 OUT_OF_USAGE_CREDITS body shape."""
    from src.data.odds_client import _is_out_of_credits
    class Resp:
        def __init__(self, j, t=""):
            self._j = j; self.text = t
        def json(self): return self._j
    assert _is_out_of_credits(Resp({"error_code": "OUT_OF_USAGE_CREDITS"})) is True
    assert _is_out_of_credits(Resp({"error_code": "INVALID_KEY"})) is False
    # Fallback text match
    assert _is_out_of_credits(Resp({}, t="OUT_OF_USAGE_CREDITS in body")) is True


def test_budget_status_before_any_fetch_returns_none_remaining():
    client = OddsAPIClient(api_key="x")
    remaining, floor, ok = client.budget_status()
    assert remaining is None
    assert floor == DEFAULT_BUDGET_FLOOR
    assert ok is False


def test_budget_status_above_floor_returns_ok():
    client = OddsAPIClient(api_key="x")
    client._update_budget({"x-requests-remaining": "250"})
    remaining, floor, ok = client.budget_status()
    assert remaining == 250
    assert floor == 100
    assert ok is True


def test_budget_status_below_floor_returns_not_ok():
    client = OddsAPIClient(api_key="x")
    client._update_budget({"x-requests-remaining": "75"})
    remaining, floor, ok = client.budget_status()
    assert remaining == 75
    assert ok is False


def test_budget_status_respects_custom_floor():
    client = OddsAPIClient(api_key="x", budget_floor=200)
    client._update_budget({"x-requests-remaining": "150"})
    remaining, floor, ok = client.budget_status()
    assert floor == 200
    assert ok is False  # 150 < 200


# -------- posted_k_prop_pitchers --------------------------------------------


def test_posted_k_prop_pitchers_set(sample_event_odds_payload, monkeypatch):
    client = OddsAPIClient(api_key="x", player_lookup_fn=None)
    monkeypatch.setattr(client, "_resolve_id", lambda n: {"Drew Rasmussen": 656876, "Eury Pérez": 691587}.get(n))
    ev = parse_event_odds_response(
        sample_event_odds_payload, resolve_id=client._resolve_id
    )
    ids = client.posted_k_prop_pitchers([ev])
    assert ids == {656876, 691587}
