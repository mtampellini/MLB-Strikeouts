"""Tests for the pure-logic opener detection module."""
from __future__ import annotations

import pytest

from src.data.opener_detection import (
    BulkPitcherCandidate,
    BulkPitcherResult,
    NoOverride,
    PitcherSeasonStats,
    SkipResult,
    TeamRotationSlot,
    check_opener,
    trigger_low_volume_probable,
    trigger_market_signal,
    trigger_recent_opener_at_slot,
)


COLE = 543037
TAMPA_OPENER = 9999
BULK_RELIEVER = 8888
ANOTHER_RELIEVER = 7777


def _normal_starter_stats(pid: int = COLE) -> PitcherSeasonStats:
    return PitcherSeasonStats(pitcher_mlbam_id=pid, season_ip=180.0, starts=29)


def _opener_stats(pid: int = TAMPA_OPENER) -> PitcherSeasonStats:
    # 8 IP across 9 outings → 0.89 IP/start
    return PitcherSeasonStats(pitcher_mlbam_id=pid, season_ip=8.0, starts=9)


def _strong_bulk_candidate(pid: int = BULK_RELIEVER) -> BulkPitcherCandidate:
    return BulkPitcherCandidate(
        pitcher_mlbam_id=pid, season_ip=42.0, recent_bulk_relief_ip=5.0
    )


# -------- Trigger helpers ----------------------------------------------------


def test_trigger_low_volume_fires_on_opener_stats():
    assert trigger_low_volume_probable(_opener_stats()) is not None


def test_trigger_low_volume_does_not_fire_on_normal_starter():
    assert trigger_low_volume_probable(_normal_starter_stats()) is None


def test_trigger_low_volume_requires_both_conditions():
    # Many short outings but high IP total → not an opener (e.g. bullpen game starter).
    stats = PitcherSeasonStats(pitcher_mlbam_id=1, season_ip=80.0, starts=50)
    assert trigger_low_volume_probable(stats) is None
    # High IP/start but very low total IP (just back from IL).
    stats2 = PitcherSeasonStats(pitcher_mlbam_id=1, season_ip=20.0, starts=4)
    assert trigger_low_volume_probable(stats2) is None


def test_trigger_recent_opener_at_slot_fires():
    slot = TeamRotationSlot(last_starter_id=TAMPA_OPENER, last_starter_ip=1.0, days_ago=5)
    assert trigger_recent_opener_at_slot(slot) is not None


def test_trigger_recent_opener_at_slot_does_not_fire_for_old_or_long_outing():
    far_back = TeamRotationSlot(last_starter_id=COLE, last_starter_ip=1.0, days_ago=14)
    assert trigger_recent_opener_at_slot(far_back) is None
    long_outing = TeamRotationSlot(last_starter_id=COLE, last_starter_ip=6.0, days_ago=5)
    assert trigger_recent_opener_at_slot(long_outing) is None


def test_trigger_recent_opener_at_slot_handles_none():
    assert trigger_recent_opener_at_slot(None) is None


def test_trigger_market_signal_fires():
    reason = trigger_market_signal(
        listed_probable_id=TAMPA_OPENER,
        posted_k_prop_pitchers={BULK_RELIEVER, COLE},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
    )
    assert reason is not None and "market signal" in reason


def test_trigger_market_signal_does_not_fire_when_listed_has_prop():
    reason = trigger_market_signal(
        listed_probable_id=COLE,
        posted_k_prop_pitchers={COLE},
        team_pitchers={COLE, BULK_RELIEVER},
    )
    assert reason is None


def test_trigger_market_signal_does_not_fire_when_no_teammates_have_props():
    reason = trigger_market_signal(
        listed_probable_id=TAMPA_OPENER,
        posted_k_prop_pitchers={COLE},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
    )
    assert reason is None


# -------- check_opener composite -------------------------------------------


def test_check_opener_normal_starter_returns_no_override():
    decision = check_opener(
        listed_probable_id=COLE,
        listed_probable_stats=_normal_starter_stats(),
        team_rotation_slot=None,
        posted_k_prop_pitchers={COLE},
        team_pitchers={COLE, BULK_RELIEVER},
        bulk_candidates=[_strong_bulk_candidate()],
    )
    assert isinstance(decision, NoOverride)


def test_check_opener_low_volume_with_bulk_returns_bulk_result():
    decision = check_opener(
        listed_probable_id=TAMPA_OPENER,
        listed_probable_stats=_opener_stats(),
        team_rotation_slot=None,
        posted_k_prop_pitchers={BULK_RELIEVER},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
        bulk_candidates=[_strong_bulk_candidate()],
    )
    assert isinstance(decision, BulkPitcherResult)
    assert decision.use_pitcher == BULK_RELIEVER
    # Two triggers fired (low-volume AND market signal) → higher confidence.
    assert decision.confidence >= 0.85


def test_check_opener_with_no_bulk_candidate_returns_skip():
    decision = check_opener(
        listed_probable_id=TAMPA_OPENER,
        listed_probable_stats=_opener_stats(),
        team_rotation_slot=None,
        posted_k_prop_pitchers={BULK_RELIEVER},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
        bulk_candidates=[],
    )
    assert isinstance(decision, SkipResult)
    assert "no bulk pitcher" in decision.reason


def test_check_opener_with_multiple_bulk_candidates_returns_skip():
    decision = check_opener(
        listed_probable_id=TAMPA_OPENER,
        listed_probable_stats=_opener_stats(),
        team_rotation_slot=None,
        posted_k_prop_pitchers={BULK_RELIEVER},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER, ANOTHER_RELIEVER},
        bulk_candidates=[
            _strong_bulk_candidate(),
            BulkPitcherCandidate(
                pitcher_mlbam_id=ANOTHER_RELIEVER, season_ip=20.0, recent_bulk_relief_ip=4.0
            ),
        ],
    )
    assert isinstance(decision, SkipResult)
    assert "multiple bulk candidates" in decision.reason


def test_check_opener_rejects_underqualified_bulk_candidates():
    """A reliever with too few IP or too little recent bulk-relief workload
    must not satisfy the bulk-pitcher filter.
    """
    decision = check_opener(
        listed_probable_id=TAMPA_OPENER,
        listed_probable_stats=_opener_stats(),
        team_rotation_slot=None,
        posted_k_prop_pitchers={BULK_RELIEVER},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
        bulk_candidates=[
            BulkPitcherCandidate(
                pitcher_mlbam_id=BULK_RELIEVER, season_ip=10.0, recent_bulk_relief_ip=5.0
            ),  # season_ip too low
            BulkPitcherCandidate(
                pitcher_mlbam_id=ANOTHER_RELIEVER, season_ip=30.0, recent_bulk_relief_ip=1.0
            ),  # bulk relief too low
        ],
    )
    assert isinstance(decision, SkipResult)


def test_check_opener_recent_opener_slot_triggers_override():
    decision = check_opener(
        listed_probable_id=TAMPA_OPENER,
        listed_probable_stats=_normal_starter_stats(pid=TAMPA_OPENER),  # IP looks fine
        team_rotation_slot=TeamRotationSlot(
            last_starter_id=99, last_starter_ip=1.0, days_ago=4
        ),
        posted_k_prop_pitchers={TAMPA_OPENER},
        team_pitchers={TAMPA_OPENER, BULK_RELIEVER},
        bulk_candidates=[_strong_bulk_candidate()],
    )
    assert isinstance(decision, BulkPitcherResult)
    assert "opener at this slot" in decision.reason
