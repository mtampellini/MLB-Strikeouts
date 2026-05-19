"""Phase 3c tests: P(K|PA) builders + log-odds composition.

Updated for Phase 3-v2c-iii feature redesign:
- Removed tests for dropped builders (pitcher_k_pct_30d_blended,
  pitcher_csw_pct_30d, pitcher_putaway_pitch_concentration,
  lineup_k_pct_vs_hand, lineup_zone_contact_pct, lineup_chase_rate,
  park_k_factor).
- Added tests for new builders (pitcher_csw_pct_season,
  pitcher_csw_pct_season_delta, pitcher_archetype_feature,
  log_park_k_factor_by_hand).
- Composition tests updated to expect the new feature set.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from src.projection.features_kpa import (
    K_EVENTS,
    P_K_PA_CEIL,
    P_K_PA_FLOOR,
    PKPAResult,
    compute_p_k_pa,
    log_park_k_factor_by_hand,
    pitcher_archetype_feature,
    pitcher_chase_whiff_pct_30d,
    pitcher_csw_pct_season,
    pitcher_csw_pct_season_delta,
    pitcher_k_pct_season_shrunk,
    pitcher_velocity_trend_3starts,
    umpire_k_zone_factor,
)
from src.projection.inputs import ProjectionBundle, ProjectionContext


def _ctx() -> ProjectionContext:
    return ProjectionContext.from_default_paths()


def _pitch(*, game_pk=1, at_bat_number=1, pitch_number=1, **kw):
    base = {
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "p_throws": "R",
        "stand": "R",
        "pitch_type": "FF",
        "description": None,
        "events": None,
        "zone": 5,
        "release_speed": 95.0,
        "strikes": 0,
    }
    base.update(kw)
    return base


def _pa_pitches(*, game_pk, at_bat_number, k_event: bool, p_throws="R", **kw):
    """Build 4 pitches for one PA; final pitch's events is set if k_event=True."""
    rows = []
    for i in range(1, 4):
        rows.append(_pitch(game_pk=game_pk, at_bat_number=at_bat_number,
                            pitch_number=i, description="called_strike",
                            strikes=min(i, 2), p_throws=p_throws, **kw))
    last = _pitch(game_pk=game_pk, at_bat_number=at_bat_number, pitch_number=4,
                  description="swinging_strike" if k_event else "hit_into_play",
                  strikes=2,
                  events="strikeout" if k_event else "field_out",
                  p_throws=p_throws, **kw)
    rows.append(last)
    return rows


def _make_pitcher_pa_rows(*, n_pa: int, k_rate: float = 0.25, p_throws="R"):
    rows = []
    n_k = int(n_pa * k_rate)
    for i in range(n_pa):
        is_k = i < n_k
        rows.extend(_pa_pitches(game_pk=1000 + (i // 25),
                                at_bat_number=(i % 25) + 1,
                                k_event=is_k, p_throws=p_throws))
    return rows


def _make_bundle(**overrides) -> ProjectionBundle:
    base = {
        "metadata": {
            "bundle_version": "1.0",
            "generated_at": "2025-05-17T23:00:00+00:00",
            "game_date": "2025-05-17",
            "cutoff_date": "2025-05-16",
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
            "statcast_pitches_30d": _make_pitcher_pa_rows(n_pa=120, k_rate=0.25),
            "statcast_pitches_season": _make_pitcher_pa_rows(n_pa=200, k_rate=0.25),
            "statcast_pitches_prior_year": _make_pitcher_pa_rows(n_pa=720, k_rate=0.24),
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
                    "statcast_pa_season": _make_pitcher_pa_rows(n_pa=100, k_rate=0.225, p_throws="R"),
                }
                for i in range(1, 10)
            ],
            "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12,
            "venue_name": "Tropicana Field",
            "is_dome": True,
            "weather": {"temp_f": 72.0, "wind_speed_mph": 0.0,
                        "wind_direction": None, "humidity_pct": None,
                        "conditions": "Dome"},
            "umpire_name": "Ramon De Jesus",
            "umpire_id": 594151,
            "first_pitch_iso": "2025-05-17T16:15:00+00:00",
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


# ---- pitcher_k_pct_season_shrunk -------------------------------------------


def test_pitcher_k_pct_season_returns_shrunk_value():
    bundle = _make_bundle()
    value, miss = pitcher_k_pct_season_shrunk(bundle, _ctx())
    assert miss is None
    assert 0.20 < value < 0.27


def test_pitcher_k_pct_season_returns_none_when_empty():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [], "statcast_pitches_prior_year": [],
    })
    value, miss = pitcher_k_pct_season_shrunk(bundle, _ctx())
    assert value is None
    assert miss is not None


def test_pitcher_k_pct_season_falls_back_to_prior_year():
    bundle = _make_bundle(pitcher={"statcast_pitches_season": []})
    value, miss = pitcher_k_pct_season_shrunk(bundle, _ctx())
    assert miss is None
    assert 0.20 < value < 0.30


# ---- pitcher_csw_pct_season (NEW PRIMARY) -----------------------------------


def test_pitcher_csw_pct_season_happy_path():
    """Bundle has 200 PA × 4 pitches ≈ 800 season pitches with CSW-eligible
    descriptions; should land in the season-shrunk branch."""
    bundle = _make_bundle()
    value, miss = pitcher_csw_pct_season(bundle, _ctx())
    assert miss is None
    assert 0.0 < value < 1.0


def test_pitcher_csw_pct_season_falls_back_to_prior_year():
    """Empty season pitches but a fat prior-year window should return the
    prior-year rate directly."""
    bundle = _make_bundle(pitcher={"statcast_pitches_season": []})
    value, miss = pitcher_csw_pct_season(bundle, _ctx())
    assert miss is None
    assert 0.0 < value < 1.0


def test_pitcher_csw_pct_season_returns_none_when_both_empty():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [], "statcast_pitches_prior_year": [],
    })
    value, miss = pitcher_csw_pct_season(bundle, _ctx())
    assert value is None
    assert "insufficient" in miss


def test_pitcher_csw_pct_season_returns_none_with_tiny_samples():
    """Season has < MIN_PITCHES_CSW_SEASON (200) AND prior has < MIN_PITCHES_CSW_PRIOR (800)."""
    tiny = _make_pitcher_pa_rows(n_pa=20, k_rate=0.25)  # ~80 pitches
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": tiny,
        "statcast_pitches_prior_year": tiny,
    })
    value, miss = pitcher_csw_pct_season(bundle, _ctx())
    assert value is None
    assert "insufficient" in miss


# ---- pitcher_csw_pct_season_delta ------------------------------------------


def test_pitcher_csw_pct_season_delta_is_pitcher_minus_league():
    """Delta should equal pitcher CSW% - league CSW% (within float epsilon).

    The synthetic test fixture's CSW% is artificially high (~75%) because the
    pitch construction puts called_strike on every non-terminal pitch, so the
    delta will be a large positive number. We don't bound the magnitude here —
    we just verify the identity delta == pitcher_csw - league_csw.
    """
    bundle = _make_bundle()
    ctx = _ctx()
    csw, csw_miss = pitcher_csw_pct_season(bundle, ctx)
    delta, miss = pitcher_csw_pct_season_delta(bundle, ctx)
    if csw_miss is not None:
        pytest.skip("pitcher_csw_pct_season returned None for this fixture")
    assert csw is not None
    # If 2025 league_averages has csw_pct populated, delta should equal csw - league_csw
    if miss is None:
        from src.projection.features_kpa import _league_csw_anchor
        league_csw = _league_csw_anchor(bundle, ctx)
        assert league_csw is not None
        assert abs(delta - (csw - league_csw)) < 1e-9


def test_pitcher_csw_pct_season_delta_propagates_csw_missing():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [], "statcast_pitches_prior_year": [],
    })
    delta, miss = pitcher_csw_pct_season_delta(bundle, _ctx())
    assert delta is None
    assert "insufficient" in miss


# ---- pitcher_archetype_feature ---------------------------------------------


def test_pitcher_archetype_feature_returns_archetype_when_present():
    """Bundle with a populated PitcherArchetype returns the archetype name."""
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["pitcher_archetype"] = {
        "archetype": "Power-FF", "season": 2025,
        "fastball_pct": 0.6, "four_seam_pct": 0.4, "sinker_pct": 0.2,
        "cutter_pct": 0.0, "breaking_pct": 0.25, "offspeed_pct": 0.15,
        "n_starts": 25, "confidence": "current_season",
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, miss = pitcher_archetype_feature(bundle, _ctx())
    assert miss is None
    assert value == "Power-FF"


def test_pitcher_archetype_feature_none_when_missing():
    """Bundle without pitcher_archetype field -> (None, reason)."""
    bundle = _make_bundle()
    value, miss = pitcher_archetype_feature(bundle, _ctx())
    assert value is None
    assert "no pitcher_archetype" in miss


def test_pitcher_archetype_feature_none_when_unknown_sentinel():
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["pitcher_archetype"] = {
        "archetype": "unknown", "season": 2025,
        "fastball_pct": 0.0, "four_seam_pct": 0.0, "sinker_pct": 0.0,
        "cutter_pct": 0.0, "breaking_pct": 0.0, "offspeed_pct": 0.0,
        "n_starts": 0, "confidence": "unknown",
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, miss = pitcher_archetype_feature(bundle, _ctx())
    assert value is None
    assert "unknown" in miss


# ---- Chase-whiff -----------------------------------------------------------


def test_pitcher_chase_whiff_with_ooz_pitches():
    """Add some OOZ swings to the 30d window."""
    ooz_rows = []
    for i in range(20):
        ooz_rows.append(_pitch(game_pk=5000, at_bat_number=i + 1, pitch_number=1,
                                description="swinging_strike", zone=12))
        ooz_rows.append(_pitch(game_pk=5000, at_bat_number=i + 1, pitch_number=2,
                                description="foul", zone=11))
    bundle = _make_bundle(pitcher={"statcast_pitches_30d":
                                    _make_pitcher_pa_rows(n_pa=100, k_rate=0.25) + ooz_rows})
    value, miss = pitcher_chase_whiff_pct_30d(bundle, _ctx())
    assert miss is None
    assert value is not None


def test_pitcher_chase_whiff_returns_none_no_ooz_swings():
    """All pitches are in-zone -> no chase swings -> None."""
    no_ooz = []
    for i in range(50):
        no_ooz.append(_pitch(game_pk=1, at_bat_number=i + 1, pitch_number=1,
                              description="called_strike", zone=5))
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": no_ooz})
    value, miss = pitcher_chase_whiff_pct_30d(bundle, _ctx())
    assert value is None


# ---- Velocity trend --------------------------------------------------------


def test_velocity_trend_returns_z_score():
    """Construct a pitcher whose recent 3 starts have velo down vs season."""
    season_fb = [_pitch(game_pk=1000 + i, at_bat_number=1, pitch_number=1,
                         pitch_type="FF", release_speed=95.0 + (i % 4) * 0.5)
                 for i in range(50)]
    recent_fb = [_pitch(game_pk=2000 + (i // 12), at_bat_number=(i % 12) + 1,
                         pitch_number=1, pitch_type="FF", release_speed=93.5)
                 for i in range(36)]
    bundle = _make_bundle(pitcher={
        "statcast_pitches_30d": recent_fb,
        "statcast_pitches_season": season_fb,
    })
    value, miss = pitcher_velocity_trend_3starts(bundle, _ctx())
    assert miss is None
    assert value < 0  # velocity is down -> negative Z


def test_velocity_trend_returns_none_few_fastballs():
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": []})
    value, miss = pitcher_velocity_trend_3starts(bundle, _ctx())
    assert value is None
    assert "fastballs" in miss


# ---- log_park_k_factor_by_hand ---------------------------------------------


def test_log_park_k_factor_by_hand_uses_rhp_split_for_right_handed_pitcher():
    """RHP -> uses factor_rhp from bundle.park_k_factors_by_hand."""
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["park_k_factors_by_hand"] = {
        "venue_id": 12, "venue_name": "Tropicana Field",
        "factor_lhp": 1.10, "factor_rhp": 1.05,
        "factor_combined": 1.07, "n_games_lhp": 100, "n_games_rhp": 100,
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, miss = log_park_k_factor_by_hand(bundle, _ctx())
    assert miss is None
    assert abs(value - math.log(1.05)) < 1e-9


def test_log_park_k_factor_by_hand_uses_lhp_split_for_left_handed_pitcher():
    bundle_dict = _make_bundle(pitcher={"handedness": "L"}).to_dict()
    bundle_dict["park_k_factors_by_hand"] = {
        "venue_id": 12, "venue_name": "Tropicana Field",
        "factor_lhp": 1.10, "factor_rhp": 1.05,
        "factor_combined": 1.07, "n_games_lhp": 100, "n_games_rhp": 100,
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, miss = log_park_k_factor_by_hand(bundle, _ctx())
    assert miss is None
    assert abs(value - math.log(1.10)) < 1e-9


def test_log_park_k_factor_by_hand_falls_back_to_legacy_when_no_by_hand():
    """Legacy bundle without park_k_factors_by_hand uses ctx.park_k_factors."""
    bundle = _make_bundle()
    assert bundle.park_k_factors_by_hand is None
    value, miss = log_park_k_factor_by_hand(bundle, _ctx())
    # Tropicana is in park_k_factors.json — should resolve
    assert miss is None
    assert value is not None


def test_log_park_k_factor_by_hand_unknown_venue_returns_none():
    """Unknown venue AND no by_hand field -> None."""
    bundle = _make_bundle(game_context={"venue_id": 99999})
    value, miss = log_park_k_factor_by_hand(bundle, _ctx())
    assert value is None
    assert "park_k_factors" in miss


# ---- Umpire ----------------------------------------------------------------


def test_umpire_k_factor_missing_returns_one():
    bundle = _make_bundle()
    value, miss = umpire_k_zone_factor(bundle, _ctx())
    assert miss is None
    assert value == 1.0


# ---- compute_p_k_pa composition --------------------------------------------


def test_compute_p_k_pa_happy_path():
    bundle = _make_bundle()
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is False
    assert result.skip_reason is None
    assert result.p_k_pa is not None
    assert P_K_PA_FLOOR <= result.p_k_pa <= P_K_PA_CEIL
    # New feature set must appear
    for k in ("pitcher_csw_pct_season", "log_park_k_factor_by_hand",
              "league_k_pct_vs_hand"):
        assert k in result.used_features, f"missing required feature: {k}"


def test_compute_p_k_pa_does_not_emit_dropped_features():
    """Dropped builders must NOT appear in used_features."""
    bundle = _make_bundle()
    result = compute_p_k_pa(bundle, _ctx())
    dropped = {
        "pitcher_k_pct_30d_blended", "pitcher_csw_pct_30d",
        "pitcher_putaway_pitch_concentration",
        "lineup_k_pct_vs_hand", "lineup_zone_contact_pct",
        "lineup_chase_rate", "park_k_factor",
    }
    for name in dropped:
        assert name not in result.used_features, (
            f"dropped feature {name!r} leaked into used_features"
        )


def test_compute_p_k_pa_skips_when_csw_missing():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [], "statcast_pitches_prior_year": [],
    })
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is True
    assert "pitcher_csw_pct_season" in result.skip_reason


def test_compute_p_k_pa_skips_when_park_missing():
    """Unknown venue AND no park_k_factors_by_hand -> skip."""
    bundle = _make_bundle(game_context={"venue_id": 99999})
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is True
    assert "log_park_k_factor_by_hand" in result.skip_reason


def test_compute_p_k_pa_clips_at_floor():
    """Very low K rate pitcher -> result clipped at floor."""
    low_k_rows = _make_pitcher_pa_rows(n_pa=300, k_rate=0.05)
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": low_k_rows,
        "statcast_pitches_prior_year": low_k_rows,
        "statcast_pitches_30d": low_k_rows[:60],
    })
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is False
    assert result.p_k_pa >= P_K_PA_FLOOR


def test_compute_p_k_pa_clips_at_ceiling():
    """Very high K rate pitcher -> result clipped at ceiling."""
    high_k_rows = _make_pitcher_pa_rows(n_pa=300, k_rate=0.55)
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": high_k_rows,
        "statcast_pitches_prior_year": high_k_rows,
        "statcast_pitches_30d": high_k_rows[:60],
    })
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is False
    assert result.p_k_pa <= P_K_PA_CEIL


def test_compute_p_k_pa_carries_pitcher_archetype_in_used_features():
    """When the bundle carries a pitcher_archetype, the side-channel feature
    surfaces in used_features (but does NOT contribute to logit composition)."""
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["pitcher_archetype"] = {
        "archetype": "Power-FF", "season": 2025,
        "fastball_pct": 0.6, "four_seam_pct": 0.4, "sinker_pct": 0.2,
        "cutter_pct": 0.0, "breaking_pct": 0.25, "offspeed_pct": 0.15,
        "n_starts": 25, "confidence": "current_season",
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is False
    assert result.used_features.get("pitcher_archetype") == "Power-FF"
