"""Phase 7: tests for the daily settlement orchestrator (settle_results)."""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from scripts.settle_results import (
    grade_pick,
    profit_loss_units,
    settle_day,
    settle_pick,
    update_calibration_tracker,
    update_performance_ledger,
)


# ---- grade_pick ------------------------------------------------------------


def test_grade_over_half_integer_win():
    assert grade_pick("Over", 5.5, 7) == "win"
    assert grade_pick("Over", 5.5, 6) == "win"


def test_grade_over_half_integer_loss():
    assert grade_pick("Over", 5.5, 4) == "loss"
    assert grade_pick("Over", 5.5, 5) == "loss"


def test_grade_under_half_integer_win():
    assert grade_pick("Under", 5.5, 4) == "win"
    assert grade_pick("Under", 5.5, 5) == "win"


def test_grade_under_half_integer_loss():
    assert grade_pick("Under", 5.5, 6) == "loss"


def test_grade_integer_line_push():
    """K == integer line is a push."""
    assert grade_pick("Over", 5.0, 5) == "push"
    assert grade_pick("Under", 5.0, 5) == "push"


def test_grade_integer_line_win_loss():
    assert grade_pick("Over", 5.0, 6) == "win"
    assert grade_pick("Over", 5.0, 4) == "loss"
    assert grade_pick("Under", 5.0, 4) == "win"
    assert grade_pick("Under", 5.0, 6) == "loss"


def test_grade_pitcher_did_not_start():
    """Missing observed_k -> pitcher_did_not_start."""
    assert grade_pick("Over", 5.5, None) == "pitcher_did_not_start"


def test_grade_invalid_side():
    with pytest.raises(ValueError):
        grade_pick("Push", 5.5, 6)


# ---- profit_loss_units -----------------------------------------------------


def test_pl_loss_is_minus_one():
    assert profit_loss_units(-110, "loss") == -1.0
    assert profit_loss_units(+150, "loss") == -1.0


def test_pl_push_is_zero():
    assert profit_loss_units(-110, "push") == 0.0
    assert profit_loss_units(+150, "push") == 0.0


def test_pl_pitcher_did_not_start_is_zero():
    assert profit_loss_units(-110, "pitcher_did_not_start") == 0.0


def test_pl_no_game_is_zero():
    assert profit_loss_units(-110, "no_game") == 0.0


def test_pl_win_negative_odds():
    """-110 win = 100/110 ≈ 0.909 units."""
    assert profit_loss_units(-110, "win") == pytest.approx(100 / 110, abs=1e-9)


def test_pl_win_positive_odds():
    """+150 win = 1.50 units."""
    assert profit_loss_units(+150, "win") == pytest.approx(1.5, abs=1e-9)


# ---- settle_pick (integration of grade + P/L) ------------------------------


def _make_pick_with_history() -> dict:
    return {
        "pick_id": "p1",
        "pitcher_mlbam_id": 656876,
        "pitcher_name": "Drew Rasmussen",
        "game_pk": 822982,
        "game_date": "2026-05-17",
        "line": 5.5, "side": "Over", "book": "fanduel",
        "tier": "primary",
        "first_seen_odds": -110,
        "first_seen_market_p": 0.524,
        "closing_american_odds": -125,
        "closing_market_p": 0.556,
        "clv_pct": -0.032,
        "model_p": 0.640,
        "model_e_k": 5.85,
        "snapshots": [],
    }


def test_settle_pick_win_at_first_seen_returns_correct_pl():
    pick = _make_pick_with_history()
    settled = settle_pick(pick, observed_k=7)
    assert settled["result"] == "win"
    assert settled["observed_k"] == 7
    # First-seen -110 win: 100/110 ≈ 0.9091
    assert settled["profit_loss_first_seen"] == pytest.approx(100 / 110, abs=1e-4)
    # Closing -125 win: 100/125 = 0.8000
    assert settled["profit_loss_closing"] == pytest.approx(100 / 125, abs=1e-4)


def test_settle_pick_loss():
    pick = _make_pick_with_history()
    settled = settle_pick(pick, observed_k=3)
    assert settled["result"] == "loss"
    assert settled["profit_loss_first_seen"] == -1.0
    assert settled["profit_loss_closing"] == -1.0


def test_settle_pick_no_start():
    pick = _make_pick_with_history()
    settled = settle_pick(pick, observed_k=None)
    assert settled["result"] == "pitcher_did_not_start"
    assert settled["profit_loss_first_seen"] == 0.0


# ---- update_performance_ledger --------------------------------------------


def test_performance_ledger_appends_resolved_picks():
    resolved = [
        {**_make_pick_with_history(), "pick_id": "a", "result": "win",
         "profit_loss_first_seen": 0.9091, "profit_loss_closing": 0.8},
        {**_make_pick_with_history(), "pick_id": "b", "result": "loss",
         "profit_loss_first_seen": -1.0, "profit_loss_closing": -1.0,
         "tier": "shadow"},
    ]
    ledger = update_performance_ledger(resolved, existing_blob=None)
    assert ledger["total_picks_resolved"] == 2
    assert ledger["by_tier"]["primary"]["wins"] == 1
    assert ledger["by_tier"]["shadow"]["losses"] == 1
    assert len(ledger["all_picks"]) == 2


def test_performance_ledger_dedups_by_pick_id():
    resolved = [
        {**_make_pick_with_history(), "pick_id": "a", "result": "win",
         "profit_loss_first_seen": 0.9091, "profit_loss_closing": 0.8},
    ]
    first = update_performance_ledger(resolved, existing_blob=None)
    # Re-settle: the same pick shouldn't be appended twice.
    second = update_performance_ledger(resolved, existing_blob=first)
    assert len(second["all_picks"]) == 1


def test_performance_ledger_by_tier_aggregates():
    """Aggregates compute win rate, ROI%, and mean CLV correctly."""
    resolved = [
        {**_make_pick_with_history(), "pick_id": "a",
         "result": "win", "profit_loss_first_seen": 0.9091,
         "profit_loss_closing": 0.9091, "clv_pct": 0.02},
        {**_make_pick_with_history(), "pick_id": "b",
         "result": "loss", "profit_loss_first_seen": -1.0,
         "profit_loss_closing": -1.0, "clv_pct": 0.01},
    ]
    ledger = update_performance_ledger(resolved, existing_blob=None)
    primary = ledger["by_tier"]["primary"]
    assert primary["n"] == 2
    assert primary["wins"] == 1
    assert primary["losses"] == 1
    assert primary["win_rate"] == 50.0
    # ROI = (0.9091 + -1.0) / 2 * 100 = -4.545%
    assert primary["roi_pct"] == pytest.approx(-4.545, abs=0.01)
    # Mean CLV = (0.02 + 0.01) / 2 * 100 = 1.5%
    assert primary["mean_clv_pct"] == pytest.approx(1.5, abs=0.01)


# ---- update_calibration_tracker -------------------------------------------


def test_calibration_tracker_accumulates_over_lines():
    resolved = [
        # Over picks at line 5.5: predicted 0.60, observed: 1 win, 1 loss = 50%
        {**_make_pick_with_history(), "pick_id": "a",
         "line": 5.5, "side": "Over", "model_p": 0.60, "result": "win"},
        {**_make_pick_with_history(), "pick_id": "b",
         "line": 5.5, "side": "Over", "model_p": 0.60, "result": "loss"},
    ]
    cal = update_calibration_tracker(resolved, existing_blob=None)
    assert "5.5" in cal["by_line"]
    entry = cal["by_line"]["5.5"]
    assert entry["n_picks_over"] == 2
    assert entry["observed_p_over_mean"] == 0.5  # 1 win out of 2
    assert entry["predicted_p_over_mean"] == 0.6
    # Deviation = (observed - predicted) * 100 = -10pp
    assert entry["deviation_pp"] == pytest.approx(-10.0, abs=0.01)


def test_calibration_tracker_skips_under_picks():
    """Per-line calibration anchored to P(K>=line); Under picks tracked
    elsewhere (or skipped)."""
    resolved = [
        {**_make_pick_with_history(), "pick_id": "a",
         "line": 5.5, "side": "Under", "model_p": 0.40, "result": "win"},
    ]
    cal = update_calibration_tracker(resolved, existing_blob=None)
    # Under pick is tracked in ID dedup but not in by_line aggregation
    assert "5.5" not in cal["by_line"]
    assert "a" in cal["tracked_pick_ids"]


def test_calibration_tracker_dedups_pick_ids_across_runs():
    resolved = [
        {**_make_pick_with_history(), "pick_id": "a",
         "line": 5.5, "side": "Over", "model_p": 0.60, "result": "win"},
    ]
    first = update_calibration_tracker(resolved, existing_blob=None)
    second = update_calibration_tracker(resolved, existing_blob=first)
    # n_picks_tracked should stay at 1
    assert second["n_picks_tracked"] == 1
    assert second["by_line"]["5.5"]["n_picks_over"] == 1


# ---- settle_day no-ledger short-circuit -----------------------------------


def test_settle_day_no_ledger_short_circuits(tmp_path, monkeypatch):
    """If the target date has no ledger, settle returns gracefully."""
    monkeypatch.setattr(
        "scripts.settle_results.DATA_PICKS_DIR", tmp_path,
    )
    summary = settle_day(date(2026, 5, 19), dry_run=True)
    assert summary["n_picks_settled"] == 0
    assert summary["reason"] == "no_ledger"


def test_settle_day_dry_run_reads_but_skips_writes(tmp_path, monkeypatch):
    """Dry-run reads the ledger and computes results in memory; doesn't
    update the performance ledger or calibration tracker on disk."""
    monkeypatch.setattr(
        "scripts.settle_results.DATA_PICKS_DIR", tmp_path,
    )
    # Avoid hitting the real MLB Stats API
    monkeypatch.setattr(
        "scripts.settle_results._fetch_observed_k", lambda pid, gp: 7,
    )
    day_dir = tmp_path / "2026-05-19"
    day_dir.mkdir(parents=True)
    pick = _make_pick_with_history()
    pick["game_date"] = "2026-05-19"
    (day_dir / "all_picks_debug.json").write_text(
        json.dumps({"primary": [pick], "secondary": [], "shadow": []}),
        encoding="utf-8",
    )
    summary = settle_day(date(2026, 5, 19), dry_run=True)
    assert summary["n_picks_settled"] == 1
    assert summary["n_wins"] == 1
    # No performance ledger written (dry-run)
    assert not (tmp_path / "performance_ledger.json").exists()
