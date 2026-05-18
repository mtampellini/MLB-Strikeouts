"""Phase 3c tests: P(K|PA) builders + log-odds composition."""
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
    lineup_chase_rate,
    lineup_k_pct_vs_hand,
    lineup_zone_contact_pct,
    park_k_factor,
    pitcher_chase_whiff_pct_30d,
    pitcher_csw_pct_30d,
    pitcher_k_pct_30d_blended,
    pitcher_k_pct_season_shrunk,
    pitcher_putaway_pitch_concentration,
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


# ---- pitcher_k_pct_30d_blended ---------------------------------------------


def test_pitcher_k_pct_30d_blends_with_season():
    bundle = _make_bundle()
    value, miss = pitcher_k_pct_30d_blended(bundle, _ctx())
    assert miss is None
    assert 0.20 < value < 0.30


def test_pitcher_k_pct_30d_returns_none_when_no_30d():
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": []})
    value, miss = pitcher_k_pct_30d_blended(bundle, _ctx())
    assert value is None


# ---- CSW ------------------------------------------------------------------


def test_pitcher_csw_pct_30d_happy():
    bundle = _make_bundle()
    value, miss = pitcher_csw_pct_30d(bundle, _ctx())
    assert miss is None
    assert 0.0 < value < 1.0


def test_pitcher_csw_returns_none_no_pitches():
    bundle = _make_bundle(pitcher={"statcast_pitches_30d": []})
    value, miss = pitcher_csw_pct_30d(bundle, _ctx())
    assert value is None


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
    # 3 recent starts (game_pks 2000-2002), 12 fastballs each = 36 total.
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


# ---- Putaway concentration -------------------------------------------------


def test_putaway_concentration_returns_share():
    """A pitcher who throws their slider 60% of 2K counts has concentration 0.6."""
    two_strike_pitches = []
    for i in range(60):
        two_strike_pitches.append(_pitch(game_pk=1, at_bat_number=i + 1, pitch_number=3,
                                          strikes=2, pitch_type="SL"))
    for i in range(40):
        two_strike_pitches.append(_pitch(game_pk=1, at_bat_number=i + 1 + 60,
                                          pitch_number=3, strikes=2, pitch_type="FF"))
    bundle = _make_bundle(pitcher={"statcast_pitches_season": two_strike_pitches})
    value, miss = pitcher_putaway_pitch_concentration(bundle, _ctx())
    assert miss is None
    assert abs(value - 0.6) < 0.001


def test_putaway_returns_none_no_two_strike():
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": [], "statcast_pitches_prior_year": [],
    })
    value, miss = pitcher_putaway_pitch_concentration(bundle, _ctx())
    assert value is None


# ---- Lineup features -------------------------------------------------------


def test_lineup_k_pct_vs_hand_happy():
    bundle = _make_bundle()
    value, miss = lineup_k_pct_vs_hand(bundle, _ctx())
    assert miss is None
    assert 0.20 < value < 0.30


def test_lineup_k_pct_vs_hand_skips_when_no_lineup():
    bundle = _make_bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    value, miss = lineup_k_pct_vs_hand(bundle, _ctx())
    assert value is None


def test_lineup_zone_contact_happy():
    bundle = _make_bundle()
    value, miss = lineup_zone_contact_pct(bundle, _ctx())
    # Our test data has all PAs ending in non-swing or swing-strike — value
    # depends on the SWING_DESCRIPTIONS classification. As long as it returns
    # something in [0, 1] the math is sound.
    assert miss is None
    assert 0.0 <= value <= 1.0


def test_lineup_chase_rate_happy_with_ooz_swings():
    """Inject OOZ pitches into one batter so chase rate has a value."""
    bundle_dict = _make_bundle().to_dict()
    ooz_rows = []
    for i in range(20):
        ooz_rows.append(_pitch(game_pk=8000, at_bat_number=i + 1, pitch_number=1,
                                description="swinging_strike", zone=12, p_throws="R"))
        ooz_rows.append(_pitch(game_pk=8000, at_bat_number=i + 1, pitch_number=2,
                                description="ball", zone=11, p_throws="R"))
    bundle_dict["opposing_lineup"]["batters"][0]["statcast_pa_season"] = ooz_rows
    bundle = ProjectionBundle.from_dict(bundle_dict)
    value, miss = lineup_chase_rate(bundle, _ctx())
    assert miss is None
    assert 0.0 <= value <= 1.0


# ---- Park / umpire ---------------------------------------------------------


def test_park_k_factor_known_venue():
    """Tropicana (venue 12) is present in park_k_factors.json. With Phase 4b
    derived factors the value is no longer exactly 1.0; just assert it's in
    a sane range and not None.
    """
    bundle = _make_bundle()
    value, miss = park_k_factor(bundle, _ctx())
    assert miss is None
    assert value is not None
    assert 0.80 <= value <= 1.20


def test_park_k_factor_unknown_venue_returns_none():
    bundle = _make_bundle(game_context={"venue_id": 99999})
    value, miss = park_k_factor(bundle, _ctx())
    assert value is None
    assert "park_k_factors" in miss


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
    for k in ("pitcher_k_pct_season_shrunk", "lineup_k_pct_vs_hand", "park_k_factor"):
        assert k in result.used_features


def test_compute_p_k_pa_skips_when_required_missing():
    bundle = _make_bundle(game_context={"venue_id": 99999})  # unknown park
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is True
    assert "park_k_factor" in result.skip_reason
    assert result.p_k_pa is None


def test_compute_p_k_pa_clips_at_floor():
    """Force an artificial scenario where unclipped result < 0.10."""
    # Use a pitcher with very low K rate and a lineup that's hard to strike out.
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
    """Pitcher with very high K rate -> ceiling at 0.45."""
    high_k_rows = _make_pitcher_pa_rows(n_pa=300, k_rate=0.55)
    bundle = _make_bundle(pitcher={
        "statcast_pitches_season": high_k_rows,
        "statcast_pitches_prior_year": high_k_rows,
        "statcast_pitches_30d": high_k_rows[:60],
    })
    result = compute_p_k_pa(bundle, _ctx())
    assert result.skipped is False
    assert result.p_k_pa <= P_K_PA_CEIL


def test_compute_p_k_pa_lineup_high_k_increases_projection():
    """Same pitcher, higher-K lineup -> higher projected K rate."""
    bundle_normal = _make_bundle()
    high_k_batter_rows = _make_pitcher_pa_rows(n_pa=100, k_rate=0.40, p_throws="R")
    bundle_dict = bundle_normal.to_dict()
    for batter in bundle_dict["opposing_lineup"]["batters"]:
        batter["statcast_pa_season"] = high_k_batter_rows
    bundle_high = ProjectionBundle.from_dict(bundle_dict)

    r_normal = compute_p_k_pa(bundle_normal, _ctx())
    r_high = compute_p_k_pa(bundle_high, _ctx())
    assert r_high.p_k_pa > r_normal.p_k_pa
