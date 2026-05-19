"""Phase 7: tests for the picks pipeline orchestrator (run_daily_picks)."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from scripts.run_daily_picks import (
    _categorize_skips,
    _merge_picks_with_snapshots,
    _same_minute_bucket,
    _snapshot_dict_from_pick,
    compute_clv_pct,
    run_pipeline,
)


# ---- CLV computation -------------------------------------------------------


def test_clv_positive_for_over_when_line_moves_against_book():
    """Over pick: first_seen book p=0.50, closing p=0.45 (book less
    confident over hits). CLV = 0.50 - 0.45 = +0.05."""
    assert compute_clv_pct(0.50, 0.45, "Over") == pytest.approx(0.05)


def test_clv_negative_for_over_when_book_more_confident():
    """Over pick: first_seen 0.50, closing 0.55 (book MORE confident).
    CLV = -0.05 (line moved against us)."""
    assert compute_clv_pct(0.50, 0.55, "Over") == pytest.approx(-0.05)


def test_clv_under_sign_flipped():
    """Under pick: first_seen 0.50, closing 0.55 (book more confident
    over hits = book less confident under hits). For UNDER pick we
    actually want negative correlation: book becoming less confident
    means our line moved in our favor.

    spec: clv_pct = first_seen_market_p - closing_market_p, sign flipped
    for Under.

    Under, first_seen 0.50, closing 0.55:
    raw = 0.50 - 0.55 = -0.05
    flipped: +0.05 (positive CLV for the under bettor)
    """
    assert compute_clv_pct(0.50, 0.55, "Under") == pytest.approx(0.05)


def test_clv_zero_when_no_movement():
    assert compute_clv_pct(0.50, 0.50, "Over") == 0.0
    assert compute_clv_pct(0.50, 0.50, "Under") == 0.0


# ---- Idempotent merge -----------------------------------------------------


def _make_pick(
    pick_id: str = "abc123", line: float = 5.5, side: str = "Over",
    odds: int = -110, market_p: float = 0.524, model_p: float = 0.640,
    edge: float = 0.221, ev: float = 5.5, tier: str = "primary",
    rank: int = 1, book: str = "fanduel",
) -> dict:
    return {
        "pick_id": pick_id,
        "pitcher_mlbam_id": 656876,
        "pitcher_name": "Drew Rasmussen",
        "game_pk": 822982,
        "game_date": "2026-05-17",
        "line": line, "side": side, "book": book,
        "american_odds": odds,
        "market_p": market_p,
        "edge_pct": edge,
        "ev_pct": ev,
        "model_p": model_p,
        "model_e_k": 5.85,
        "model_e_bf": 25.6,
        "tier": tier,
        "rank_in_tier": rank,
        "devig_source": "two_sided",
        "pitcher_archetype": "Power-FF",
        "park_k_factor_by_hand": 1.07,
        "calibration_note": "...",
        "generated_at": "2026-05-19T11:00:00+00:00",
    }


def test_merge_first_run_creates_new_records():
    """First run for a fresh day: all picks are new."""
    new_picks = [_make_pick("a"), _make_pick("b", odds=+200)]
    run_time = datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc)
    merged = _merge_picks_with_snapshots(new_picks, existing_ledger=None,
                                          run_time=run_time)
    assert len(merged) == 2
    for pick in merged:
        assert pick["first_seen_at"] == run_time.isoformat(timespec="seconds")
        assert pick["first_seen_odds"] == pick["american_odds"]
        assert pick["clv_pct"] == 0.0  # no movement on first snapshot
        assert len(pick["snapshots"]) == 1
        assert pick["active"] is True


def test_merge_second_run_appends_snapshot():
    """Second run with the same pick_id: appends to snapshot history,
    updates closing_*, preserves first_seen_*."""
    run_1 = datetime(2026, 5, 19, 11, 0, 0, tzinfo=timezone.utc)
    run_2 = datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc)
    pick_v1 = _make_pick(odds=-110, market_p=0.524, edge=0.221)
    pick_v2 = _make_pick(odds=-125, market_p=0.556, edge=0.150)

    first_merge = _merge_picks_with_snapshots(
        [pick_v1], existing_ledger=None, run_time=run_1,
    )
    existing_ledger = {"primary": first_merge}

    second_merge = _merge_picks_with_snapshots(
        [pick_v2], existing_ledger=existing_ledger, run_time=run_2,
    )
    assert len(second_merge) == 1
    p = second_merge[0]
    assert p["first_seen_odds"] == -110           # unchanged
    assert p["first_seen_market_p"] == 0.524      # unchanged
    assert p["closing_american_odds"] == -125     # updated
    assert p["closing_market_p"] == 0.556         # updated
    assert len(p["snapshots"]) == 2               # appended
    # CLV: first_seen 0.524, closing 0.556, Over → 0.524 - 0.556 = -0.032
    assert p["clv_pct"] == pytest.approx(-0.032, abs=1e-9)


def test_merge_idempotent_within_same_minute():
    """Two runs within the same HH:MM produce only ONE snapshot entry
    (idempotent)."""
    run_1 = datetime(2026, 5, 19, 11, 0, 0, tzinfo=timezone.utc)
    run_1b = datetime(2026, 5, 19, 11, 0, 30, tzinfo=timezone.utc)  # same minute
    pick = _make_pick()

    first_merge = _merge_picks_with_snapshots(
        [pick], existing_ledger=None, run_time=run_1,
    )
    existing = {"primary": first_merge}
    second_merge = _merge_picks_with_snapshots(
        [pick], existing_ledger=existing, run_time=run_1b,
    )
    assert len(second_merge[0]["snapshots"]) == 1  # no duplicate


def test_merge_pick_removed_from_slate_marked_inactive():
    """A pick that was on the slate at 11am but vanishes by 3pm stays in
    the ledger (with history) but is marked active=False."""
    run_1 = datetime(2026, 5, 19, 11, 0, 0, tzinfo=timezone.utc)
    run_2 = datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc)
    pick_a = _make_pick("a")
    pick_b = _make_pick("b", odds=+200)

    first_merge = _merge_picks_with_snapshots(
        [pick_a, pick_b], existing_ledger=None, run_time=run_1,
    )
    existing = {"primary": first_merge}
    second_merge = _merge_picks_with_snapshots(
        [pick_a], existing_ledger=existing, run_time=run_2,
    )
    by_id = {p["pick_id"]: p for p in second_merge}
    assert by_id["a"]["active"] is True
    assert by_id["b"]["active"] is False  # dropped from active slate
    assert len(by_id["b"]["snapshots"]) == 1  # history preserved


def test_merge_preserves_tier_promotions():
    """If a pick was shadow at 11am and primary at 3pm, the latest tier
    wins on tier-output but snapshots show the evolution."""
    run_1 = datetime(2026, 5, 19, 11, 0, 0, tzinfo=timezone.utc)
    run_2 = datetime(2026, 5, 19, 15, 0, 0, tzinfo=timezone.utc)
    shadow_pick = _make_pick(edge=0.15, tier="shadow", rank=1)
    promoted = _make_pick(edge=0.25, tier="primary", rank=2)

    first = _merge_picks_with_snapshots(
        [shadow_pick], existing_ledger=None, run_time=run_1,
    )
    existing = {"shadow": first}
    second = _merge_picks_with_snapshots(
        [promoted], existing_ledger=existing, run_time=run_2,
    )
    assert second[0]["tier"] == "primary"
    assert second[0]["rank_in_tier"] == 2
    assert second[0]["snapshots"][0]["tier"] == "shadow"
    assert second[0]["snapshots"][1]["tier"] == "primary"


# ---- _same_minute_bucket ---------------------------------------------------


def test_same_minute_bucket_true_for_same_hhmm():
    assert _same_minute_bucket(
        "2026-05-19T11:00:30+00:00", "2026-05-19T11:00:45+00:00",
    )


def test_same_minute_bucket_false_for_different_hhmm():
    assert not _same_minute_bucket(
        "2026-05-19T11:00:30+00:00", "2026-05-19T11:01:00+00:00",
    )


def test_same_minute_bucket_handles_none():
    assert not _same_minute_bucket(None, "2026-05-19T11:00:00+00:00")


# ---- Snapshot dict shape --------------------------------------------------


def test_snapshot_dict_captures_time_varying_fields():
    pick = _make_pick()
    run_time = datetime(2026, 5, 19, 11, 0, 0, tzinfo=timezone.utc)
    snap = _snapshot_dict_from_pick(pick, run_time)
    for field in ("run_time", "american_odds", "market_p", "edge_pct",
                   "ev_pct", "model_p", "model_e_k", "devig_source",
                   "tier", "rank_in_tier"):
        assert field in snap


# ---- Dry-run smoke (no API hits) -------------------------------------------


# ---- Skip categorization ---------------------------------------------------


def test_categorize_skips_buckets_transient_separately():
    skipped = [
        {"reason": "lineup_not_posted", "is_transient": True},
        {"reason": "lineup_not_posted", "is_transient": True},
        {"reason": "projector_skipped: hard_filter: career_ip ...",
         "is_transient": False},
        {"reason": "no_market_data_at_either_book", "is_transient": False},
    ]
    breakdown = _categorize_skips(skipped)
    assert breakdown["transient_lineup_not_posted"] == 2
    assert breakdown["permanent_projector_skipped"] == 1
    assert breakdown["permanent_no_market_data_at_either_book"] == 1


def test_categorize_skips_empty_returns_empty_dict():
    assert _categorize_skips([]) == {}


def test_categorize_skips_defaults_to_permanent_when_flag_missing():
    """Pre-amendment skip records without is_transient should be
    classified as permanent (safer default — won't be confused with
    lineup-pending)."""
    skipped = [{"reason": "no_market_data_at_either_book"}]
    breakdown = _categorize_skips(skipped)
    assert "permanent_no_market_data_at_either_book" in breakdown


# ---- Dry-run smoke (no API hits) -------------------------------------------


def test_run_pipeline_dry_run_empty_slate(tmp_path, monkeypatch):
    """dry_run=True with bundles=[] skips writes and returns a summary."""
    monkeypatch.setattr(
        "scripts.run_daily_picks.DATA_PICKS_DIR", tmp_path,
    )
    summary = run_pipeline(
        date(2026, 5, 19), dry_run=True, bundles=[],
    )
    assert summary["dry_run"] is True
    assert summary["n_bundles"] == 0
    # No files written
    assert not (tmp_path / "2026-05-19").exists()


def test_run_pipeline_writes_ledger_on_non_dry_run(tmp_path, monkeypatch):
    """Non-dry-run with empty slate still creates the day dir + empty ledger."""
    monkeypatch.setattr(
        "scripts.run_daily_picks.DATA_PICKS_DIR", tmp_path,
    )
    run_pipeline(date(2026, 5, 19), dry_run=False, bundles=[])
    day_dir = tmp_path / "2026-05-19"
    assert day_dir.exists()
    assert (day_dir / "picks.json").exists()
    assert (day_dir / "all_picks_debug.json").exists()
    assert (day_dir / "run_log.json").exists()
    snapshots_dir = day_dir / "snapshots"
    assert snapshots_dir.exists()


def test_run_pipeline_idempotent_two_runs_same_minute(tmp_path, monkeypatch):
    """Two pipeline calls in the same minute produce ONE run_log entry per
    snapshot bucket — the snapshot file isn't overwritten."""
    monkeypatch.setattr(
        "scripts.run_daily_picks.DATA_PICKS_DIR", tmp_path,
    )
    target = date(2026, 5, 19)
    run_pipeline(target, dry_run=False, bundles=[])
    run_pipeline(target, dry_run=False, bundles=[])
    snapshots = list((tmp_path / "2026-05-19" / "snapshots").glob("*.json"))
    # Both runs hit the same HHMM bucket (within a second), so only one
    # snapshot file exists.
    assert len(snapshots) == 1
    # But run_log captures both runs
    log = json.loads((tmp_path / "2026-05-19" / "run_log.json").read_text())
    assert len(log["runs"]) == 2
