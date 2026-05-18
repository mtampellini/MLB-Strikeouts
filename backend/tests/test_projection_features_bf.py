"""Phase 3b tests: E[BF] feature builders + composition."""
from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

import pytest

from src.projection.features_bf import (
    ANCHOR_IP_PER_START,
    BASELINE_BF,
    BF_CEIL,
    BF_FLOOR,
    DAYS_REST_BF_ADJUSTMENT,
    K_PRIOR_IP_PER_START_STARTS,
    EBFResult,
    compute_e_bf,
    days_rest_bucket,
    lineup_obp_vs_hand,
    park_run_environment_factor,
    pitcher_ip_per_start_30d_shrunk,
    pitcher_pitches_per_pa_30d,
    pitcher_pitches_per_pa_season,
    team_bullpen_short_hook_indicator,
    weather_run_environment,
)
from src.projection.inputs import (
    HandednessAverages,
    LeagueAverages,
    ParkFactors,
    ProjectionBundle,
    ProjectionContext,
    UmpireKFactors,
)


# ---- Fixtures --------------------------------------------------------------


def _ctx_default() -> ProjectionContext:
    return ProjectionContext.from_default_paths()


def _pitch_row(*, game_pk, at_bat_number, pitch_number=1, events=None, p_throws="R", **extras):
    row = {
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "events": events,
        "p_throws": p_throws,
    }
    row.update(extras)
    return row


def _make_pitcher_rows(*, n_starts: int, pa_per_start: int, pitches_per_pa: float = 4.0) -> list[dict]:
    """Build a synthetic pitches list with stable PA/game structure."""
    rows = []
    pitches_per_pa_int = max(1, int(round(pitches_per_pa)))
    for s in range(n_starts):
        game_pk = 1000 + s
        for ab in range(1, pa_per_start + 1):
            for pn in range(1, pitches_per_pa_int + 1):
                rows.append(
                    _pitch_row(
                        game_pk=game_pk,
                        at_bat_number=ab,
                        pitch_number=pn,
                        events="strikeout" if pn == pitches_per_pa_int else None,
                        p_throws="R",
                    )
                )
    return rows


def _make_bundle(**overrides) -> ProjectionBundle:
    """Minimal bundle for E[BF] testing. Pass overrides to mutate specifics."""
    base = {
        "metadata": {
            "bundle_version": "1.0",
            "generated_at": "2026-05-17T23:00:00+00:00",
            "game_date": "2026-05-17",
            "cutoff_date": "2026-05-16",
            "pitcher_mlbam_id": 656876,
            "pitcher_name": "Drew Rasmussen",
            "game_pk": 822982,
        },
        "pitcher": {
            "mlbam_id": 656876,
            "name": "Drew Rasmussen",
            "handedness": "R",
            "team_abbr": "TBR",
            "source": "fangraphs",
            "fg_opener_flag": False,
            "fg_primary_pitcher_flag": False,
            "opener_detection_result": "no_override",
            "statcast_pitches_30d": _make_pitcher_rows(n_starts=5, pa_per_start=22, pitches_per_pa=4.0),
            "statcast_pitches_season": _make_pitcher_rows(n_starts=9, pa_per_start=22, pitches_per_pa=4.0),
            "statcast_pitches_prior_year": _make_pitcher_rows(n_starts=29, pa_per_start=23, pitches_per_pa=4.1),
        },
        "opposing_lineup": {
            "team_abbr": "MIA",
            "batters": [
                {
                    "mlbam_id": 100 + i,
                    "name": f"Batter {i}",
                    "batting_order": i,
                    "handedness": "R",
                    "position": "DH",
                    "statcast_pa_season": [
                        # 100 same-hand PAs with 30 on-base (OBP=.300)
                        _pitch_row(
                            game_pk=5000 + j,
                            at_bat_number=j,
                            pitch_number=1,
                            events="single" if j < 30 else "field_out",
                            p_throws="R",
                        )
                        for j in range(100)
                    ],
                }
                for i in range(1, 10)
            ],
            "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12,
            "venue_name": "Tropicana Field",
            "is_dome": True,
            "weather": {
                "temp_f": 72.0, "wind_speed_mph": 0.0,
                "wind_direction": None, "humidity_pct": None,
                "conditions": "Dome",
            },
            "umpire_name": "Ramon De Jesus",
            "umpire_id": 594151,
            "first_pitch_iso": "2026-05-17T16:15:00+00:00",
            "days_rest": 5,
        },
        "market": {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None,
            "snapshot_source": "missing",
        },
    }
    _deep_merge(base, overrides)
    return ProjectionBundle.from_dict(base)


def _deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


# ---- Feature 1: ip_per_start_30d_shrunk -----------------------------------


def test_ip_per_start_30d_shrinks_toward_season():
    """30d=4 starts with 22 PA/start (~5.12 IP/start), season=9 starts with same → no
    discrepancy → shrunk value ~equal to both. Sanity: returns within plausible range."""
    bundle = _make_bundle()
    ctx = _ctx_default()
    value, miss = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    assert miss is None
    assert 4.0 < value < 6.5


def test_ip_per_start_uses_season_when_30d_empty():
    bundle = _make_bundle(
        pitcher={"statcast_pitches_30d": []}
    )
    ctx = _ctx_default()
    value, miss = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    assert miss is None
    assert value is not None


def test_ip_per_start_falls_back_to_prior_year_when_no_season():
    bundle = _make_bundle(
        pitcher={"statcast_pitches_30d": [], "statcast_pitches_season": []}
    )
    ctx = _ctx_default()
    value, miss = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    # prior_year is populated → use it as prior, no 30d → return prior directly
    assert miss is None
    assert value is not None


def test_ip_per_start_returns_none_when_all_empty():
    bundle = _make_bundle(
        pitcher={
            "statcast_pitches_30d": [],
            "statcast_pitches_season": [],
            "statcast_pitches_prior_year": [],
        }
    )
    ctx = _ctx_default()
    value, miss = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    assert value is None
    assert miss is not None
    assert "no" in miss.lower()


def test_ip_shrinkage_pulls_toward_prior_for_small_sample():
    """30d sample has 1 start of 30 PA (huge), season prior has 9 starts of 22 PA.
    Shrunk should be pulled toward the season anchor."""
    big_30d = _make_pitcher_rows(n_starts=1, pa_per_start=30, pitches_per_pa=3.5)
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": big_30d})
    ctx = _ctx_default()
    raw_30d = 30 / 4.3   # ~6.98 IP for the one start
    raw_season = 22 / 4.3  # ~5.12 IP/start anchor
    value, _ = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    # With k_prior=5, 1 start of 6.98 vs 5 prior of 5.12:
    # shrunk = (1*6.98 + 5*5.12) / 6 ≈ 5.43
    assert raw_season < value < raw_30d
    assert abs(value - ((1 * raw_30d + K_PRIOR_IP_PER_START_STARTS * raw_season) / (1 + K_PRIOR_IP_PER_START_STARTS))) < 0.001


# ---- Feature 2: pitches_per_pa_season --------------------------------------


def test_pitches_per_pa_season_happy_path():
    bundle = _make_bundle()  # season has 9*22*4 = 792 pitches across 198 PA -> 4.0
    ctx = _ctx_default()
    value, miss = pitcher_pitches_per_pa_season(bundle, ctx)
    assert miss is None
    assert abs(value - 4.0) < 0.01


def test_pitches_per_pa_season_shrinks_when_small_sample():
    """Season has <200 pitches AND a prior_year anchor — must shrink."""
    small_season = _make_pitcher_rows(n_starts=2, pa_per_start=10, pitches_per_pa=5.0)
    bundle = _make_bundle(pitcher={"statcast_pitches_season": small_season})
    ctx = _ctx_default()
    value, _ = pitcher_pitches_per_pa_season(bundle, ctx)
    raw_season_rate = 5.0
    prior_pitches = 29 * 23 * 4  # rough estimate; prior config
    # Prior rate = pitches_per_pa from prior_year rows we created (~4.0)
    assert 4.0 < value < 5.0   # pulled toward prior


def test_pitches_per_pa_season_returns_none_if_no_data():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [],
        "statcast_pitches_prior_year": [],
    })
    value, miss = pitcher_pitches_per_pa_season(bundle, _ctx_default())
    assert value is None
    assert miss is not None


# ---- Feature 3: pitches_per_pa_30d -----------------------------------------


def test_pitches_per_pa_30d_blends_with_season():
    bundle = _make_bundle()
    ctx = _ctx_default()
    value, miss = pitcher_pitches_per_pa_30d(bundle, ctx)
    assert miss is None
    assert abs(value - 4.0) < 0.05


def test_pitches_per_pa_30d_returns_none_when_no_recent_pa():
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": []})
    value, miss = pitcher_pitches_per_pa_30d(bundle, _ctx_default())
    assert value is None
    assert miss is not None


# ---- Feature 4: lineup_obp_vs_hand -----------------------------------------


def test_lineup_obp_vs_hand_happy_path():
    bundle = _make_bundle()  # 9 batters, each .300 OBP, all vs R, pitcher R
    ctx = _ctx_default()
    value, miss = lineup_obp_vs_hand(bundle, ctx)
    assert miss is None
    assert abs(value - 0.300) < 0.005


def test_lineup_obp_skips_when_lineup_not_posted():
    bundle = _make_bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    value, miss = lineup_obp_vs_hand(bundle, _ctx_default())
    assert value is None
    assert miss == "lineup not posted"


def test_lineup_obp_returns_none_when_pitcher_hand_unknown():
    bundle = _make_bundle(pitcher={"handedness": None})
    value, miss = lineup_obp_vs_hand(bundle, _ctx_default())
    assert value is None
    assert "handedness" in miss


def test_lineup_obp_shrinks_small_sample_batters_to_league_avg():
    """A batter with <50 same-hand PAs should pull toward league avg."""
    # One batter with 10 PAs all on-base → raw OBP=1.0. Shrunk should be close to league avg.
    base = _make_bundle()
    # Replace first batter with a tiny-sample one
    bundle_dict = base.to_dict()
    bundle_dict["opposing_lineup"]["batters"][0]["statcast_pa_season"] = [
        _pitch_row(game_pk=9000 + j, at_bat_number=1, pitch_number=1, events="walk", p_throws="R")
        for j in range(10)
    ]
    bundle = ProjectionBundle.from_dict(bundle_dict)
    ctx = _ctx_default()
    value, _ = lineup_obp_vs_hand(bundle, ctx)
    # Without shrinkage that batter would be 1.0 → lineup OBP would jump.
    # With shrinkage to league avg (~0.305), lineup stays close to 0.300.
    assert value < 0.5  # nowhere near 1.0


def test_lineup_obp_uses_league_avg_for_zero_pa_batter():
    base = _make_bundle()
    bundle_dict = base.to_dict()
    bundle_dict["opposing_lineup"]["batters"][0]["statcast_pa_season"] = []
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, _ = lineup_obp_vs_hand(bundle, _ctx_default())
    assert value is not None
    # Sane range
    assert 0.25 < value < 0.40


# ---- Feature 5: park_run_environment_factor --------------------------------


def test_park_factor_happy_path():
    bundle = _make_bundle()  # Tropicana, venue_id=12 → 1.0 in our placeholder
    ctx = _ctx_default()
    value, miss = park_run_environment_factor(bundle, ctx)
    assert miss is None
    assert value == 1.0


def test_park_factor_coors_above_neutral():
    bundle = _make_bundle(game_context={"venue_id": 19})  # Coors
    ctx = _ctx_default()
    value, _ = park_run_environment_factor(bundle, ctx)
    assert value > 1.0


def test_park_factor_unknown_venue_returns_none():
    bundle = _make_bundle(game_context={"venue_id": 99999})
    value, miss = park_run_environment_factor(bundle, _ctx_default())
    assert value is None
    assert "venue_id=99999" in miss


# ---- Feature 6: weather_run_environment ------------------------------------


def test_weather_dome_returns_neutral():
    bundle = _make_bundle()  # Trop, is_dome=True
    value, miss = weather_run_environment(bundle, _ctx_default())
    assert miss is None
    assert value == 1.0


def test_weather_outdoor_wind_out_increases_factor():
    bundle = _make_bundle(game_context={
        "is_dome": False,
        "weather": {"temp_f": 75.0, "wind_speed_mph": 12.0,
                    "wind_direction": "Out To CF", "humidity_pct": None,
                    "conditions": "Sunny"},
    })
    value, miss = weather_run_environment(bundle, _ctx_default())
    assert miss is None
    assert value > 1.0


def test_weather_outdoor_wind_in_decreases_factor():
    bundle = _make_bundle(game_context={
        "is_dome": False,
        "weather": {"temp_f": 65.0, "wind_speed_mph": 10.0,
                    "wind_direction": "In From CF", "humidity_pct": None,
                    "conditions": "Cloudy"},
    })
    value, _ = weather_run_environment(bundle, _ctx_default())
    assert value < 1.0


def test_weather_missing_temp_returns_none():
    bundle = _make_bundle(game_context={
        "is_dome": False,
        "weather": {"temp_f": None, "wind_speed_mph": 5.0,
                    "wind_direction": "Out To LF", "humidity_pct": None,
                    "conditions": None},
    })
    value, miss = weather_run_environment(bundle, _ctx_default())
    assert value is None
    assert "weather" in miss


def test_weather_missing_wind_returns_none():
    bundle = _make_bundle(game_context={
        "is_dome": False,
        "weather": {"temp_f": 70.0, "wind_speed_mph": None,
                    "wind_direction": "Out To LF", "humidity_pct": None,
                    "conditions": None},
    })
    value, miss = weather_run_environment(bundle, _ctx_default())
    assert value is None


# ---- Feature 7: days_rest_bucket -------------------------------------------


def test_days_rest_bucket_categories():
    cases = [
        (3, "<4"),
        (4, "4"),
        (5, "5"),
        (6, "6+"),
        (10, "6+"),
        (None, "first_start_or_il_return"),
    ]
    for dr, expected in cases:
        bundle = _make_bundle(game_context={"days_rest": dr})
        bucket, miss = days_rest_bucket(bundle, _ctx_default())
        assert miss is None
        assert bucket == expected


# ---- Feature 8: bullpen_short_hook (deferred) ------------------------------


def test_bullpen_short_hook_returns_neutral_placeholder():
    bundle = _make_bundle()
    value, miss = team_bullpen_short_hook_indicator(bundle, _ctx_default())
    assert value == 0.0
    assert miss is None


# ---- E[BF] composition -----------------------------------------------------


def test_compute_e_bf_happy_path():
    bundle = _make_bundle()
    result = compute_e_bf(bundle, _ctx_default())
    assert result.skipped is False
    assert result.skip_reason is None
    assert result.e_bf is not None
    assert BF_FLOOR <= result.e_bf <= BF_CEIL
    # Should include all four required features
    for key in (
        "pitcher_ip_per_start_30d_shrunk",
        "pitcher_pitches_per_pa_season",
        "lineup_obp_vs_hand",
        "park_run_environment_factor",
    ):
        assert key in result.used_features


def test_compute_e_bf_required_missing_skips():
    bundle = _make_bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    result = compute_e_bf(bundle, _ctx_default())
    assert result.skipped is True
    assert "lineup_obp_vs_hand" in result.skip_reason
    assert result.e_bf is None


def test_compute_e_bf_unknown_park_skips():
    bundle = _make_bundle(game_context={"venue_id": 99999})
    result = compute_e_bf(bundle, _ctx_default())
    assert result.skipped is True
    assert "park_run_environment_factor" in result.skip_reason


def test_compute_e_bf_optional_missing_does_not_skip():
    """No 30d data → pitches_per_pa_30d is missing → should still compute."""
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": []})
    result = compute_e_bf(bundle, _ctx_default())
    # IP/start shrinkage falls back; pitches_per_pa_30d is None and is optional.
    assert result.skipped is False
    assert "pitcher_pitches_per_pa_30d" not in result.used_features


def test_compute_e_bf_clipping_engages_at_floor():
    """Extreme low IP, low OBP, depressed park — clip should hold the floor."""
    # Construct a pitcher with very short outings (3 PA/start = ~0.7 IP/start)
    tiny = _make_pitcher_rows(n_starts=5, pa_per_start=3, pitches_per_pa=4.0)
    bundle = _make_bundle(
        pitcher={
            "statcast_pitches_30d": tiny,
            "statcast_pitches_season": tiny * 2,
            "statcast_pitches_prior_year": tiny * 3,
        },
    )
    result = compute_e_bf(bundle, _ctx_default())
    assert result.skipped is False
    assert result.e_bf >= BF_FLOOR


def test_compute_e_bf_clipping_engages_at_ceiling():
    """Extreme high IP/start should clip to ceiling 32."""
    huge = _make_pitcher_rows(n_starts=5, pa_per_start=45, pitches_per_pa=4.0)
    bundle = _make_bundle(pitcher={
        "statcast_pitches_30d": huge,
        "statcast_pitches_season": huge * 2,
    })
    result = compute_e_bf(bundle, _ctx_default())
    assert result.skipped is False
    assert result.e_bf <= BF_CEIL


def test_compute_e_bf_uses_days_rest_adjustment():
    """The days_rest bucket value should appear in used_features."""
    bundle = _make_bundle(game_context={"days_rest": 4})
    result = compute_e_bf(bundle, _ctx_default())
    assert result.used_features["days_rest_bucket"] == "4"
