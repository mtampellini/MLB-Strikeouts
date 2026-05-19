"""Phase 3-v2c-ii: tests for per-batter feature builders."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.projection.features_per_batter import (
    DEFAULT_K_PRIOR,
    DEFAULT_MIN_CURRENT_N,
    DEFAULT_MIN_PRIOR_N,
    SENTINEL_NO_LEAGUE_AVGS_KEY,
    SENTINEL_NO_LEAGUE_AVGS_REASON,
    _effective_batter_hand,
    _load_league_avgs,
    _shrunk_batter_rate,
    per_batter_chase_rate,
    per_batter_k_pct_vs_hand,
    per_batter_obp_vs_hand,
    per_batter_zone_contact_pct,
)
from src.projection.inputs import ProjectionBundle

SAMPLE_BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase3_v2c_i_sample_bundle_2026-05-17_656876.json"
)


# ---- _shrunk_batter_rate ---------------------------------------------------


def test_shrink_current_season_path():
    """observed_n >= min_current_n -> shrink observed to league."""
    rate, reason = _shrunk_batter_rate(
        observed_rate=0.25, observed_n=100,
        prior_year_rate=None, prior_year_n=0,
        league_rate=0.22,
    )
    assert reason == "current_season_shrunk_to_league"
    # Posterior = (0.25*100 + 0.22*80) / 180 = (25 + 17.6)/180 = 0.2367
    assert abs(rate - (0.25*100 + 0.22*80) / 180) < 1e-9


def test_shrink_prior_year_fallback_when_current_below_threshold():
    """observed_n < 50 but prior_year_n >= 200 -> two-step chain."""
    rate, reason = _shrunk_batter_rate(
        observed_rate=0.15, observed_n=30,
        prior_year_rate=0.20, prior_year_n=400,
        league_rate=0.22,
    )
    assert reason == "prior_year_chain_to_league"
    # Step1: (0.15*30 + 0.20*80) / 110 = (4.5 + 16) / 110 = 0.1864
    step1 = (0.15*30 + 0.20*80) / (30 + 80)
    # Step2: (step1 * 110 + 0.22 * 80) / 190
    expected = (step1 * 110 + 0.22 * 80) / (110 + 80)
    assert abs(rate - expected) < 1e-9


def test_shrink_prior_year_only_zero_current_observations():
    """observed_n=0 but prior_year_n large enough -> step1 = prior."""
    rate, reason = _shrunk_batter_rate(
        observed_rate=None, observed_n=0,
        prior_year_rate=0.20, prior_year_n=400,
        league_rate=0.22,
    )
    assert reason == "prior_year_chain_to_league"
    # step1 = 0.20 (prior), step1_n=80; step2 = (0.20*80 + 0.22*80) / 160 = 0.21
    expected = (0.20 * 80 + 0.22 * 80) / 160
    assert abs(rate - expected) < 1e-9


def test_shrink_small_sample_league_fallback():
    """observed_n < min AND prior_year_n < min -> return league_rate."""
    rate, reason = _shrunk_batter_rate(
        observed_rate=0.40, observed_n=20,  # not enough
        prior_year_rate=0.35, prior_year_n=50,  # also not enough
        league_rate=0.22,
    )
    assert reason == "small_sample_league_fallback"
    assert rate == 0.22


def test_shrink_no_league_rate_returns_none():
    rate, reason = _shrunk_batter_rate(
        observed_rate=0.25, observed_n=200,
        prior_year_rate=None, prior_year_n=0,
        league_rate=None,
    )
    assert rate is None
    assert reason == "no rate data available"


# ---- _effective_batter_hand ------------------------------------------------


def test_switch_hitter_vs_right_handed_pitcher():
    assert _effective_batter_hand("S", "R") == "L"


def test_switch_hitter_vs_left_handed_pitcher():
    assert _effective_batter_hand("S", "L") == "R"


def test_lr_batter_pass_through():
    assert _effective_batter_hand("L", "R") == "L"
    assert _effective_batter_hand("R", "L") == "R"


def test_unknown_handedness_returns_none():
    assert _effective_batter_hand(None, "R") is None
    assert _effective_batter_hand("Z", "R") is None


# ---- Fixtures --------------------------------------------------------------


def _pa_pitch_rows(
    *, game_pk: int = 1, abn_start: int = 1, p_throws: str = "R",
    events: str = "field_out", zone: int = 5, description: str = "hit_into_play",
    n_pitches: int = 1,
) -> list[dict]:
    """Construct synthetic pitch rows for a single PA."""
    rows = []
    for i in range(n_pitches):
        is_last = (i == n_pitches - 1)
        rows.append({
            "game_pk": game_pk, "at_bat_number": abn_start,
            "pitch_number": i + 1, "p_throws": p_throws,
            "events": events if is_last else None,
            "zone": zone, "description": description,
        })
    return rows


def _make_batter_dict(
    mlbam_id: int, name: str, batting_order: int, handedness: str,
    n_k: int = 0, n_bb: int = 0, n_singles: int = 0, n_outs: int = 0,
    p_throws: str = "R",
) -> dict:
    """Build a batter dict with the given PA outcomes (one row per PA — using
    only terminal pitches because that's what _pa_event_rows extracts)."""
    rows: list[dict] = []
    abn = 1
    for _ in range(n_k):
        rows.extend(_pa_pitch_rows(
            game_pk=1, abn_start=abn, p_throws=p_throws,
            events="strikeout", zone=5, description="swinging_strike",
        ))
        abn += 1
    for _ in range(n_bb):
        rows.extend(_pa_pitch_rows(
            game_pk=1, abn_start=abn, p_throws=p_throws,
            events="walk", zone=12, description="ball",
        ))
        abn += 1
    for _ in range(n_singles):
        rows.extend(_pa_pitch_rows(
            game_pk=1, abn_start=abn, p_throws=p_throws,
            events="single", zone=5, description="hit_into_play",
        ))
        abn += 1
    for _ in range(n_outs):
        rows.extend(_pa_pitch_rows(
            game_pk=1, abn_start=abn, p_throws=p_throws,
            events="field_out", zone=5, description="hit_into_play",
        ))
        abn += 1
    return {
        "mlbam_id": mlbam_id, "name": name,
        "batting_order": batting_order, "handedness": handedness,
        "position": "1B", "statcast_pa_season": rows,
    }


@pytest.fixture
def synthetic_bundle_dict() -> dict:
    """Bundle with a 9-batter MIA lineup vs RHP Drew Rasmussen.

    Mix of sample sizes so each shrinkage branch fires somewhere.
    """
    batters = [
        # Slot 1: switch hitter, 100 same-hand PAs (above threshold)
        _make_batter_dict(669364, "Xavier Edwards", 1, "S",
                          n_k=20, n_bb=10, n_singles=20, n_outs=50),
        # Slot 2: lefty, 80 PAs (above threshold)
        _make_batter_dict(665487, "Otto Lopez", 2, "L",
                          n_k=15, n_bb=8, n_singles=15, n_outs=42),
        # Slot 3: righty, 100 PAs
        _make_batter_dict(606115, "Kyle Stowers", 3, "R",
                          n_k=25, n_bb=10, n_singles=18, n_outs=47),
        # Slot 4: lefty, 75 PAs
        _make_batter_dict(593428, "Eric Wagaman", 4, "L",
                          n_k=18, n_bb=6, n_singles=12, n_outs=39),
        # Slot 5: switch, 60 PAs
        _make_batter_dict(657557, "Heriberto Hernandez", 5, "S",
                          n_k=15, n_bb=5, n_singles=10, n_outs=30),
        # Slot 6: righty, 55 PAs
        _make_batter_dict(696166, "Connor Norby", 6, "R",
                          n_k=14, n_bb=4, n_singles=9, n_outs=28),
        # Slot 7: lefty, 50 PAs (right at threshold)
        _make_batter_dict(671213, "Liam Hicks", 7, "L",
                          n_k=12, n_bb=4, n_singles=8, n_outs=26),
        # Slot 8: switch, 30 PAs (below threshold → league fallback)
        _make_batter_dict(700001, "Switch SmallSample", 8, "S",
                          n_k=8, n_bb=2, n_singles=4, n_outs=16),
        # Slot 9: rookie, 10 PAs (small sample → league fallback)
        _make_batter_dict(701234, "Joe Mack", 9, "L",
                          n_k=3, n_bb=1, n_singles=2, n_outs=4),
    ]
    return {
        "metadata": {
            "bundle_version": "1.0",
            "generated_at": "2025-05-17T23:00:00+00:00",
            "game_date": "2025-05-17",
            "cutoff_date": "2025-05-16",
            "pitcher_mlbam_id": 656876, "pitcher_name": "Drew Rasmussen",
            "game_pk": 822982,
        },
        "pitcher": {
            "mlbam_id": 656876, "name": "Drew Rasmussen",
            "handedness": "R", "team_abbr": "TBR", "source": "fangraphs",
            "fg_opener_flag": False, "fg_primary_pitcher_flag": False,
            "opener_detection_result": "no_override",
            "statcast_pitches_30d": [],
            "statcast_pitches_season": [],
            "statcast_pitches_prior_year": [],
        },
        "opposing_lineup": {
            "team_abbr": "MIA", "batters": batters, "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12, "venue_name": "Tropicana Field", "is_dome": True,
            "weather": {
                "temp_f": 72.0, "wind_speed_mph": 0.0, "wind_direction": None,
                "humidity_pct": None, "conditions": "Dome",
            },
            "umpire_name": "Ramon De Jesus", "umpire_id": 594151,
            "first_pitch_iso": "2025-05-17T16:15:00+00:00", "days_rest": 6,
        },
        "market": {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None, "snapshot_source": "missing",
        },
    }


@pytest.fixture
def synthetic_bundle(synthetic_bundle_dict):
    return ProjectionBundle.from_dict(synthetic_bundle_dict)


# ---- Happy path: 9-batter lineup, full output ------------------------------


def test_per_batter_k_pct_returns_9_entries(synthetic_bundle):
    out = per_batter_k_pct_vs_hand(synthetic_bundle)
    assert len(out) == 9
    # Every entry has a numeric rate (synthetic data ensures coverage)
    for bid, (val, reason) in out.items():
        assert val is not None, f"batter {bid}: rate is None ({reason})"
        assert 0.0 <= val <= 1.0


def test_per_batter_obp_returns_9_entries(synthetic_bundle):
    out = per_batter_obp_vs_hand(synthetic_bundle)
    assert len(out) == 9
    for bid, (val, reason) in out.items():
        assert val is not None
        assert 0.0 <= val <= 1.0


def test_per_batter_zone_contact_returns_9_entries(synthetic_bundle):
    out = per_batter_zone_contact_pct(synthetic_bundle)
    assert len(out) == 9
    for bid, (val, reason) in out.items():
        assert val is not None


def test_per_batter_chase_returns_9_entries(synthetic_bundle):
    out = per_batter_chase_rate(synthetic_bundle)
    assert len(out) == 9
    for bid, (val, reason) in out.items():
        assert val is not None


# ---- Switch hitter handling ------------------------------------------------


def test_switch_hitter_uses_lr_split_vs_right_pitcher(synthetic_bundle):
    """Slot 1 (Edwards) is a switch hitter vs RHP — should resolve to LR cell
    and get a per-batter K% in the right ballpark."""
    out = per_batter_k_pct_vs_hand(synthetic_bundle)
    val, reason = out[669364]
    # Switch hitter with 100 PAs above threshold — should hit current_season path
    assert reason == "current_season_shrunk_to_league"
    assert val is not None


# ---- Small-sample fallback -------------------------------------------------


def test_small_sample_batter_falls_back_to_league(synthetic_bundle):
    """Slot 9 has 10 PAs — below min_current_n (50), no prior-year data in
    the contract, so league fallback fires."""
    out = per_batter_k_pct_vs_hand(synthetic_bundle)
    val, reason = out[701234]  # Joe Mack at slot 9
    assert reason == "small_sample_league_fallback"
    # The league fallback K% comes from league_averages
    assert val is not None
    assert 0.15 < val < 0.35  # plausible league K% range


def test_threshold_batter_at_50_uses_current_path(synthetic_bundle):
    """Slot 7 has exactly 50 PAs — boundary, should use current path."""
    out = per_batter_k_pct_vs_hand(synthetic_bundle)
    val, reason = out[671213]
    assert reason == "current_season_shrunk_to_league"


# ---- Lineup not posted -----------------------------------------------------


def test_returns_empty_dict_when_lineup_not_posted(synthetic_bundle_dict):
    blob = copy.deepcopy(synthetic_bundle_dict)
    blob["opposing_lineup"]["lineup_posted"] = False
    blob["opposing_lineup"]["batters"] = []
    bundle = ProjectionBundle.from_dict(blob)

    assert per_batter_k_pct_vs_hand(bundle) == {}
    assert per_batter_zone_contact_pct(bundle) == {}
    assert per_batter_chase_rate(bundle) == {}
    assert per_batter_obp_vs_hand(bundle) == {}


# ---- Pitcher handedness unknown returns empty dict -------------------------


def test_returns_empty_dict_when_pitcher_handedness_unknown(synthetic_bundle_dict):
    blob = copy.deepcopy(synthetic_bundle_dict)
    blob["pitcher"]["handedness"] = None
    bundle = ProjectionBundle.from_dict(blob)
    assert per_batter_k_pct_vs_hand(bundle) == {}


# ---- League averages missing -> sentinel -----------------------------------


def test_missing_league_averages_returns_sentinel(monkeypatch, synthetic_bundle):
    """If _load_league_avgs returns None (file missing), the builder emits
    the {0: (None, "league_averages not in bundle")} sentinel."""
    import src.projection.features_per_batter as mod

    # Bypass the lru_cache by patching the function
    monkeypatch.setattr(mod, "_load_league_avgs", lambda season: None)

    out = per_batter_k_pct_vs_hand(synthetic_bundle)
    assert out == {SENTINEL_NO_LEAGUE_AVGS_KEY: (None, SENTINEL_NO_LEAGUE_AVGS_REASON)}


# ---- Per-handedness-cell missing -> entry-level None -----------------------


def test_unknown_batter_handedness_entry_marks_none(synthetic_bundle_dict):
    """A batter with unknown handedness produces a (None, reason) entry, not
    a missing dict key."""
    blob = copy.deepcopy(synthetic_bundle_dict)
    # Slot 9: blank handedness (unresolvable)
    blob["opposing_lineup"]["batters"][8]["handedness"] = None
    bundle = ProjectionBundle.from_dict(blob)

    out = per_batter_k_pct_vs_hand(bundle)
    assert 701234 in out
    val, reason = out[701234]
    assert val is None
    assert "league average" in reason


# ---- Integration with the Phase 3-v2c-i sample bundle ----------------------


def test_integration_with_phase3_v2c_i_sample_bundle():
    """Run all 4 builders against the real Drew Rasmussen vs MIA sample."""
    if not SAMPLE_BUNDLE.exists():
        pytest.skip(f"{SAMPLE_BUNDLE} not present")
    bundle = ProjectionBundle.from_json(SAMPLE_BUNDLE)
    # Skip if lineup wasn't posted (rare but defensible)
    if not bundle.opposing_lineup.lineup_posted:
        pytest.skip("sample bundle has no posted lineup")
    assert len(bundle.opposing_lineup.batters) == 9

    for builder, name in (
        (per_batter_k_pct_vs_hand, "k_pct"),
        (per_batter_zone_contact_pct, "zone_contact"),
        (per_batter_chase_rate, "chase"),
        (per_batter_obp_vs_hand, "obp"),
    ):
        out = builder(bundle)
        # Either we get 9 entries (success path) or 1 entry (sentinel for
        # missing league_avgs). The real bundle should produce 9.
        assert len(out) == 9 or out.get(SENTINEL_NO_LEAGUE_AVGS_KEY) is not None, (
            f"{name}: unexpected dict shape {out}"
        )
        if len(out) == 9:
            # At least one batter must produce a numeric value — Marlins lineup
            # is not entirely composed of rookies.
            numeric_entries = [
                v for v, _ in out.values() if v is not None
            ]
            assert numeric_entries, (
                f"{name}: every batter returned None — that's not the Marlins"
            )


# ---- Default constants -----------------------------------------------------


def test_default_shrinkage_constants_match_spec():
    assert DEFAULT_K_PRIOR == 80
    assert DEFAULT_MIN_CURRENT_N == 50
    assert DEFAULT_MIN_PRIOR_N == 200


def test_sentinel_constants_exposed():
    assert SENTINEL_NO_LEAGUE_AVGS_KEY == 0
    assert "league_averages" in SENTINEL_NO_LEAGUE_AVGS_REASON
