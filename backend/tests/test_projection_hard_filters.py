"""Phase 3c: hard filter unit tests (run pre-feature)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.projection.hard_filters import (
    CAREER_IP_FLOOR,
    PRIOR_YEAR_IP_FLOOR,
    SEASON_IP_FLOOR,
    check_pre_feature_filters,
)
from src.projection.inputs import ProjectionBundle


def _bundle(**overrides) -> ProjectionBundle:
    base = {
        "metadata": {
            "bundle_version": "1.0", "generated_at": "2026-05-17T23:00:00+00:00",
            "game_date": "2026-05-17", "cutoff_date": "2026-05-16",
            "pitcher_mlbam_id": 656876, "pitcher_name": "X", "game_pk": 1,
        },
        "pitcher": {
            "mlbam_id": 656876, "name": "X", "handedness": "R", "team_abbr": "TBR",
            "source": "fangraphs", "fg_opener_flag": False,
            "fg_primary_pitcher_flag": False,
            "opener_detection_result": "no_override",
            # Default: plenty of IP (1 PA / 4.3 IP -> 300 PA = ~70 IP)
            "statcast_pitches_30d": [],
            "statcast_pitches_season": [
                {"game_pk": 5000 + (i // 25), "at_bat_number": (i % 25) + 1,
                 "pitch_number": 1} for i in range(120)
            ],  # 120 PA = ~28 IP
            "statcast_pitches_prior_year": [
                {"game_pk": 4000 + (i // 30), "at_bat_number": (i % 30) + 1,
                 "pitch_number": 1} for i in range(400)
            ],  # 400 PA = ~93 IP
        },
        "opposing_lineup": {
            "team_abbr": "MIA",
            "batters": [
                {"mlbam_id": 1, "name": "B", "batting_order": 1,
                 "handedness": "R", "position": "DH", "statcast_pa_season": []}
            ],
            "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12, "venue_name": "Trop", "is_dome": True,
            "weather": {"temp_f": 72.0, "wind_speed_mph": 0.0,
                        "wind_direction": None, "humidity_pct": None,
                        "conditions": "Dome"},
            "umpire_name": "X", "umpire_id": 100,
            "first_pitch_iso": "2026-05-17T16:15:00+00:00", "days_rest": 5,
        },
        "market": {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None, "snapshot_source": "missing",
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return ProjectionBundle.from_dict(base)


def test_normal_pitcher_passes_all_filters():
    assert check_pre_feature_filters(_bundle()) is None


def test_career_ip_below_floor_skips():
    """Total PA < 50 * 4.3 = ~215 across season + prior_year"""
    bundle = _bundle(pitcher={
        "statcast_pitches_30d": [],
        "statcast_pitches_season": [
            {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1}
        ],
        "statcast_pitches_prior_year": [
            {"game_pk": 2, "at_bat_number": i + 1, "pitch_number": 1}
            for i in range(50)  # ~12 IP only
        ],
    })
    reason = check_pre_feature_filters(bundle)
    assert reason is not None
    assert "career IP" in reason


def test_low_season_and_low_prior_year_skips():
    """Season IP < 20 AND prior_year IP < 80 -> skip."""
    bundle = _bundle(pitcher={
        "statcast_pitches_season": [
            {"game_pk": 1, "at_bat_number": i + 1, "pitch_number": 1}
            for i in range(50)  # ~12 IP
        ],
        "statcast_pitches_prior_year": [
            {"game_pk": 2, "at_bat_number": i + 1, "pitch_number": 1}
            for i in range(250)  # ~58 IP - below the 80 prior floor
        ],
    })
    reason = check_pre_feature_filters(bundle)
    assert reason is not None
    assert "season IP" in reason


def test_low_season_but_strong_prior_passes():
    """Season IP < 20 BUT prior_year IP >= 80 should still pass.

    PA count uses distinct (game_pk, at_bat_number); each row gets a unique
    at_bat_number so the count reflects actual PA volume.
    """
    bundle = _bundle(pitcher={
        "statcast_pitches_season": [
            {"game_pk": 1, "at_bat_number": i + 1, "pitch_number": 1}
            for i in range(30)  # 30 PA = ~7 IP
        ],
        "statcast_pitches_prior_year": [
            {"game_pk": 2 + (i // 30), "at_bat_number": (i % 30) + 1,
             "pitch_number": 1, "events": "field_out"}
            for i in range(500)  # 500 PA across ~17 games = ~116 IP
        ],
    })
    assert check_pre_feature_filters(bundle) is None


def test_lineup_not_posted_skips():
    bundle = _bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    reason = check_pre_feature_filters(bundle)
    assert reason is not None
    assert "lineup not posted" in reason


def test_opener_skip_result_skips():
    bundle = _bundle(pitcher={"opener_detection_result": "skip"})
    reason = check_pre_feature_filters(bundle)
    assert reason is not None
    assert "opener" in reason
