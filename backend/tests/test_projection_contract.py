"""Phase 3-v2c-i: tests for the additive contract extensions.

Covers:
- Loading bundles with the four new fields populated.
- Loading legacy bundles (no new fields) without raising.
- PitcherArchetype fallback chain.
- ParkKFactorsByHand lookup.
- Validator behavior on new fields (and silent acceptance of legacy bundles).
- Phase 4c regression: E[BF] fit still works on a legacy bundle.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.projection.inputs import (
    ARCHETYPE_CONFIDENCE_VALUES,
    CswToKRelationship,
    PADistribution,
    PADistributionCell,
    ParkKFactorsByHand,
    PitcherArchetype,
    ProjectionBundle,
    ProjectionContext,
    TTOMultiplierCell,
    TTOMultipliers,
    VALID_ARCHETYPES,
)


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARCHETYPES_PATH = PROCESSED_DIR / "pitcher_archetypes.json"
TTO_PATH = PROCESSED_DIR / "tto_multipliers.json"
PARK_K_PATH = PROCESSED_DIR / "park_k_factors.json"
PA_DIST_PATH = PROCESSED_DIR / "pa_distribution_by_bf.json"

LEGACY_SAMPLE_BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase2c_sample_bundle_2026-05-17_656876.json"
)


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def legacy_bundle_dict() -> dict:
    """A minimal legacy bundle (no Phase 3-v2c-i extension fields)."""
    return {
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
            "statcast_pitches_30d": [],
            "statcast_pitches_season": [],
            "statcast_pitches_prior_year": [],
        },
        "opposing_lineup": {
            "team_abbr": "MIA",
            "batters": [{
                "mlbam_id": 669364, "name": "Xavier Edwards",
                "batting_order": 1, "handedness": "S",
                "position": "2B", "statcast_pa_season": [],
            }],
            "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12, "venue_name": "Tropicana Field",
            "is_dome": True,
            "weather": {
                "temp_f": 72.0, "wind_speed_mph": 0.0,
                "wind_direction": None, "humidity_pct": None,
                "conditions": "Dome",
            },
            "umpire_name": "Ramon De Jesus", "umpire_id": 594151,
            "first_pitch_iso": "2025-05-17T16:15:00+00:00",
            "days_rest": 6,
        },
        "market": {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None, "snapshot_source": "missing",
        },
    }


@pytest.fixture
def archetypes_blob() -> dict:
    return {
        "archetypes": {
            "656876": {
                "name": "Drew Rasmussen",
                "p_throws": "R",
                "by_season": {
                    "2024": {
                        "archetype": "Power-FF",
                        "fastball_pct": 0.61, "four_seam_pct": 0.40,
                        "sinker_pct": 0.10, "cutter_pct": 0.11,
                        "breaking_pct": 0.25, "offspeed_pct": 0.14,
                        "n_starts": 25, "n_pitches_all": 2300,
                        "total_pitches": 2300, "junk_pct": 0.0,
                    },
                    "2025": {
                        "archetype": "Sinker-ball",
                        "fastball_pct": 0.62, "four_seam_pct": 0.28,
                        "sinker_pct": 0.30, "cutter_pct": 0.04,
                        "breaking_pct": 0.24, "offspeed_pct": 0.14,
                        "n_starts": 8, "n_pitches_all": 800,
                        "total_pitches": 800, "junk_pct": 0.0,
                    },
                },
                "season_rolling_30d": {
                    "2025": {
                        "archetype": "Power-FF",
                        "fastball_pct": 0.60, "four_seam_pct": 0.38,
                        "sinker_pct": 0.12, "cutter_pct": 0.10,
                        "breaking_pct": 0.26, "offspeed_pct": 0.14,
                        "n_starts": 4, "as_of_date": "2025-08-31",
                    },
                },
            },
            "999998": {  # rolling-only pitcher (for fallback test)
                "name": "Test Rolling Only",
                "p_throws": "R",
                "by_season": {},
                "season_rolling_30d": {
                    "2025": {
                        "archetype": "Breaking-heavy",
                        "fastball_pct": 0.5, "four_seam_pct": 0.30,
                        "sinker_pct": 0.10, "cutter_pct": 0.10,
                        "breaking_pct": 0.40, "offspeed_pct": 0.10,
                        "n_starts": 3, "as_of_date": "2025-08-31",
                    }
                },
            },
            "999997": {  # previous-season-only (for prev-season fallback)
                "name": "Test Prev Year Only",
                "p_throws": "L",
                "by_season": {
                    "2024": {
                        "archetype": "Offspeed-heavy",
                        "fastball_pct": 0.50, "four_seam_pct": 0.20,
                        "sinker_pct": 0.20, "cutter_pct": 0.10,
                        "breaking_pct": 0.20, "offspeed_pct": 0.30,
                        "n_starts": 20, "n_pitches_all": 1800,
                    },
                },
                "season_rolling_30d": {},
            },
        }
    }


# ---- PitcherArchetype fallback chain ---------------------------------------


def test_archetype_current_season_hit(archetypes_blob):
    pa = PitcherArchetype.from_archetypes_lookup(archetypes_blob, 656876, 2025)
    assert pa.archetype == "Sinker-ball"
    assert pa.season == 2025
    assert pa.confidence == "current_season"
    assert pa.n_starts == 8


def test_archetype_rolling_30d_fallback(archetypes_blob):
    """Pitcher 999998 has no by_season — should fall back to rolling 30d."""
    pa = PitcherArchetype.from_archetypes_lookup(archetypes_blob, 999998, 2025)
    assert pa.archetype == "Breaking-heavy"
    assert pa.confidence == "rolling_30d"


def test_archetype_previous_season_fallback(archetypes_blob):
    """Pitcher 999997 has only 2024 by_season — game in 2025 should fall back."""
    pa = PitcherArchetype.from_archetypes_lookup(archetypes_blob, 999997, 2025)
    assert pa.archetype == "Offspeed-heavy"
    assert pa.season == 2024
    assert pa.confidence == "previous_season_fallback"


def test_archetype_unknown_sentinel_when_no_data(archetypes_blob):
    """Pitcher not in any season → unknown sentinel."""
    pa = PitcherArchetype.from_archetypes_lookup(archetypes_blob, 111111, 2025)
    assert pa.archetype == "unknown"
    assert pa.confidence == "unknown"
    assert pa.season == 2025


def test_archetype_confidence_set_matches_valid_values():
    """Every confidence the resolver can emit must be in the valid set."""
    for c in ("current_season", "rolling_30d", "previous_season_fallback", "unknown"):
        assert c in ARCHETYPE_CONFIDENCE_VALUES


def test_valid_archetypes_constant():
    assert set(VALID_ARCHETYPES) == {
        "Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced",
    }


# ---- ParkKFactorsByHand ----------------------------------------------------


def test_park_k_factors_by_hand_tropicana():
    """Tropicana Field (venue_id=12) should load with L/R-split factors."""
    pk = ParkKFactorsByHand.from_json_lookup(PARK_K_PATH, venue_id=12)
    assert pk is not None
    assert pk.venue_id == 12
    assert pk.venue_name == "Tropicana Field"
    assert 0.85 <= pk.factor_lhp <= 1.30
    assert 0.85 <= pk.factor_rhp <= 1.30
    # for_pitcher_hand selects the right split
    assert pk.for_pitcher_hand("L") == pk.factor_lhp
    assert pk.for_pitcher_hand("R") == pk.factor_rhp
    assert pk.for_pitcher_hand(None) == pk.factor_combined


def test_park_k_factors_by_hand_unknown_venue_returns_none():
    pk = ParkKFactorsByHand.from_json_lookup(PARK_K_PATH, venue_id=99999999)
    assert pk is None


def test_park_k_factors_by_hand_none_venue_returns_none():
    pk = ParkKFactorsByHand.from_json_lookup(PARK_K_PATH, venue_id=None)
    assert pk is None


# ---- TTOMultipliers --------------------------------------------------------


def test_tto_multipliers_loads_all_5_archetypes_x_4_tto():
    tto = TTOMultipliers.from_json(TTO_PATH)
    assert set(tto.by_archetype.keys()) == set(VALID_ARCHETYPES)
    for arch in VALID_ARCHETYPES:
        assert set(tto.by_archetype[arch].keys()) == {1, 2, 3, 4}
    assert set(tto.league_wide.keys()) == {1, 2, 3, 4}


def test_tto_multipliers_lookup_uses_archetype_when_known():
    tto = TTOMultipliers.from_json(TTO_PATH)
    cell = tto.lookup("Power-FF", 3)
    league_cell = tto.lookup("unknown", 3)
    assert cell.multiplier == tto.by_archetype["Power-FF"][3].multiplier
    assert league_cell.multiplier == tto.league_wide[3].multiplier
    # And the two should differ — Power-FF is not the league mean
    assert cell.multiplier != league_cell.multiplier


def test_tto_multipliers_lookup_falls_back_for_unknown_archetype():
    tto = TTOMultipliers.from_json(TTO_PATH)
    cell = tto.lookup(None, 2)
    assert cell.multiplier == tto.league_wide[2].multiplier


# ---- PADistribution --------------------------------------------------------


def test_pa_distribution_loads_full_bf_range():
    pa = PADistribution.from_json(PA_DIST_PATH)
    assert pa.bf_range_observed == (12, 35)
    assert pa.modal_bf == 23
    assert set(pa.by_bf.keys()) == set(range(12, 36))
    # Every BF entry has 9 slots
    for bf, slots in pa.by_bf.items():
        assert set(slots.keys()) == set(range(1, 10)), f"BF={bf} missing slots"


def test_pa_distribution_lookup_rounds_to_nearest():
    pa = PADistribution.from_json(PA_DIST_PATH)
    by_slot = pa.lookup(25.6)
    by_slot_at_26 = pa.by_bf[26]
    # Same slot 1 cell after rounding
    assert by_slot[1].total_pa == by_slot_at_26[1].total_pa


def test_pa_distribution_lookup_clamps_below_range():
    pa = PADistribution.from_json(PA_DIST_PATH)
    by_slot = pa.lookup(5.0)
    assert by_slot[1].total_pa == pa.by_bf[12][1].total_pa


# ---- ProjectionBundle: legacy bundle loads, fields stay None ---------------


def test_legacy_bundle_loads_without_new_fields(legacy_bundle_dict):
    bundle = ProjectionBundle.from_dict(legacy_bundle_dict)
    assert bundle.pitcher_archetype is None
    assert bundle.tto_multipliers is None
    assert bundle.park_k_factors_by_hand is None
    assert bundle.pa_distribution is None


def test_legacy_real_sample_bundle_loads():
    """The committed Phase 2c sample bundle predates Phase 3-v2c-i; it
    must continue to load with the new fields defaulting to None."""
    if not LEGACY_SAMPLE_BUNDLE.exists():
        pytest.skip(f"{LEGACY_SAMPLE_BUNDLE} not present")
    bundle = ProjectionBundle.from_json(LEGACY_SAMPLE_BUNDLE)
    # New fields may be None OR populated (after we regenerate the sample);
    # the key assertion is the loader didn't raise.
    assert bundle.metadata.pitcher_mlbam_id == 656876


# ---- ProjectionBundle: from_dict_with_static_data hydration ---------------


def test_hydrated_bundle_populates_all_four_fields(legacy_bundle_dict, tmp_path):
    """from_dict_with_static_data should populate all four extension fields
    from the static data files."""
    bundle = ProjectionBundle.from_dict_with_static_data(
        legacy_bundle_dict,
        archetypes_path=ARCHETYPES_PATH,
        tto_path=TTO_PATH,
        park_k_factors_path=PARK_K_PATH,
        pa_distribution_path=PA_DIST_PATH,
    )
    assert bundle.pitcher_archetype is not None
    assert bundle.pitcher_archetype.archetype in (
        "Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy",
        "Balanced", "unknown",
    )
    assert bundle.tto_multipliers is not None
    assert set(bundle.tto_multipliers.by_archetype.keys()) == set(VALID_ARCHETYPES)
    assert bundle.park_k_factors_by_hand is not None
    assert bundle.park_k_factors_by_hand.venue_id == 12
    assert bundle.pa_distribution is not None
    assert bundle.pa_distribution.modal_bf == 23


def test_hydrated_bundle_serializes_extension_fields(legacy_bundle_dict):
    """to_dict round-trip preserves the four extension fields."""
    bundle = ProjectionBundle.from_dict_with_static_data(
        legacy_bundle_dict,
        archetypes_path=ARCHETYPES_PATH,
        tto_path=TTO_PATH,
        park_k_factors_path=PARK_K_PATH,
        pa_distribution_path=PA_DIST_PATH,
    )
    blob = bundle.to_dict()
    assert "pitcher_archetype" in blob
    assert "tto_multipliers" in blob
    assert "park_k_factors_by_hand" in blob
    assert "pa_distribution" in blob

    # Round-trip back through from_dict
    bundle2 = ProjectionBundle.from_dict(blob)
    assert bundle2.pitcher_archetype.archetype == bundle.pitcher_archetype.archetype
    assert (bundle2.park_k_factors_by_hand.factor_lhp ==
            bundle.park_k_factors_by_hand.factor_lhp)


# ---- Validator: silent on legacy, validates new fields when present --------


def test_validator_silent_on_legacy_bundle(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    issues = validate(legacy_bundle_dict)
    assert issues == [], f"unexpected issues: {issues}"


def test_validator_passes_on_hydrated_bundle(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    bundle = ProjectionBundle.from_dict_with_static_data(
        legacy_bundle_dict,
        archetypes_path=ARCHETYPES_PATH,
        tto_path=TTO_PATH,
        park_k_factors_path=PARK_K_PATH,
        pa_distribution_path=PA_DIST_PATH,
    )
    issues = validate(bundle.to_dict())
    assert issues == [], f"hydrated bundle should pass validation, got: {issues}"


def test_validator_flags_bad_archetype(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    blob = copy.deepcopy(legacy_bundle_dict)
    blob["pitcher_archetype"] = {
        "archetype": "NotAnArchetype",
        "season": 2025,
        "fastball_pct": 0.5, "four_seam_pct": 0.3, "sinker_pct": 0.2,
        "cutter_pct": 0.0, "breaking_pct": 0.3, "offspeed_pct": 0.2,
        "n_starts": 10, "confidence": "current_season",
    }
    issues = validate(blob)
    assert any("archetype" in i for i in issues)


def test_validator_flags_park_factor_out_of_range(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    blob = copy.deepcopy(legacy_bundle_dict)
    blob["park_k_factors_by_hand"] = {
        "venue_id": 12, "venue_name": "Tropicana Field",
        "factor_lhp": 2.5,  # WAY out of [0.85, 1.30]
        "factor_rhp": 1.0,
        "factor_combined": 1.0,
        "n_games_lhp": 100, "n_games_rhp": 100,
    }
    issues = validate(blob)
    assert any("factor_lhp" in i and "outside" in i for i in issues)


# ---- Backward compatibility: Phase 4c E[BF] fit on legacy bundle -----------


def test_e_bf_compute_unchanged_on_legacy_bundle(legacy_bundle_dict):
    """Phase 4c E[BF] fit must continue to work on bundles WITHOUT the new
    fields. This is the additivity guarantee."""
    from src.projection.features_bf import compute_e_bf

    bundle = ProjectionBundle.from_dict(legacy_bundle_dict)
    # Sanity: no extension fields
    assert bundle.pitcher_archetype is None
    assert bundle.tto_multipliers is None

    ctx = ProjectionContext.from_default_paths()
    result = compute_e_bf(bundle, ctx)
    # The legacy minimal bundle has empty statcast windows — compute_e_bf
    # will return a result with skipped=True (no IP signal available).
    # The KEY assertion: it returns a result WITHOUT raising, and its
    # public surface (the EBFResult type) is unchanged.
    assert result is not None
    assert hasattr(result, "skipped")
    assert hasattr(result, "used_features")
    # features_used dict shape is the same as before — only the original
    # feature names appear (no Phase 3-v2c-i feature has leaked in).
    extension_feature_names = {
        "pitcher_archetype", "tto_multiplier", "pa_distribution",
        "park_k_factor_by_hand",
    }
    for name in result.used_features:
        assert name not in extension_feature_names, (
            f"Phase 3-v2c-i feature {name!r} leaked into E[BF] used_features "
            f"— that's a contract additivity violation"
        )


# ---- Schema constants ------------------------------------------------------


def test_tto_multiplier_cell_dataclass_shape():
    cell = TTOMultiplierCell(multiplier=0.9, k_rate=0.20, n_pa=1000, low_sample=False)
    assert cell.multiplier == 0.9
    blob = cell.to_dict()
    back = TTOMultiplierCell.from_dict(blob)
    assert back == cell


def test_pa_distribution_cell_dataclass_shape():
    cell = PADistributionCell(tto_1=1.0, tto_2=1.0, tto_3=0.5,
                               tto_4=0.0, total_pa=2.5)
    assert cell.total_pa == 2.5
    blob = cell.to_dict()
    back = PADistributionCell.from_dict(blob)
    assert back == cell


# ---- Phase 4c-v2 Step 4: csw_to_k_relationship -----------------------------


CSW_TO_K_PATH = PROCESSED_DIR / "csw_to_k_relationship.json"


def test_csw_to_k_relationship_dataclass_roundtrip():
    rel = CswToKRelationship(
        intercept=-0.22, slope=1.63, r_squared=0.6,
        method="weighted_linear_regression_csw_to_k",
    )
    blob = rel.to_dict()
    back = CswToKRelationship.from_dict(blob)
    assert back == rel


def test_csw_to_k_relationship_loads_from_live_file():
    """Step 1's live file should load into the dataclass cleanly."""
    if not CSW_TO_K_PATH.exists():
        pytest.skip(f"{CSW_TO_K_PATH} not present")
    rel = CswToKRelationship.from_relationship_file(CSW_TO_K_PATH)
    assert rel is not None
    assert abs(rel.intercept - (-0.2209)) < 1e-3
    assert abs(rel.slope - 1.6346) < 1e-3
    assert 0.0 < rel.r_squared < 1.0
    assert rel.method == "weighted_linear_regression_csw_to_k"


def test_csw_to_k_relationship_missing_file_returns_none(tmp_path):
    missing = tmp_path / "nope.json"
    assert CswToKRelationship.from_relationship_file(missing) is None


def test_legacy_bundle_loads_without_csw_to_k(legacy_bundle_dict):
    """A bundle predating Step 4 (no csw_to_k_relationship key) loads
    with the field as None — no error."""
    bundle = ProjectionBundle.from_dict(legacy_bundle_dict)
    assert bundle.csw_to_k_relationship is None


def test_hydrated_bundle_populates_csw_to_k(legacy_bundle_dict):
    """from_dict_with_static_data with csw_to_k_path populates the field."""
    bundle = ProjectionBundle.from_dict_with_static_data(
        legacy_bundle_dict,
        csw_to_k_path=CSW_TO_K_PATH,
    )
    if not CSW_TO_K_PATH.exists():
        # Skip if the live file isn't available
        assert bundle.csw_to_k_relationship is None
        return
    assert bundle.csw_to_k_relationship is not None
    assert bundle.csw_to_k_relationship.slope > 1.0


def test_bundle_to_dict_round_trips_csw_to_k(legacy_bundle_dict):
    """Bundle -> to_dict -> from_dict preserves csw_to_k_relationship."""
    if not CSW_TO_K_PATH.exists():
        pytest.skip(f"{CSW_TO_K_PATH} not present")
    bundle = ProjectionBundle.from_dict_with_static_data(
        legacy_bundle_dict,
        csw_to_k_path=CSW_TO_K_PATH,
    )
    assert bundle.csw_to_k_relationship is not None
    blob = bundle.to_dict()
    assert "csw_to_k_relationship" in blob
    bundle2 = ProjectionBundle.from_dict(blob)
    assert bundle2.csw_to_k_relationship == bundle.csw_to_k_relationship


def test_validator_passes_on_valid_csw_to_k(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    blob = copy.deepcopy(legacy_bundle_dict)
    blob["csw_to_k_relationship"] = {
        "intercept": -0.22, "slope": 1.63, "r_squared": 0.6,
        "method": "weighted_linear_regression_csw_to_k",
    }
    issues = validate(blob)
    assert issues == [], f"unexpected issues: {issues}"


def test_validator_passes_when_csw_to_k_absent(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    assert "csw_to_k_relationship" not in legacy_bundle_dict
    issues = validate(legacy_bundle_dict)
    assert issues == []


def test_validator_flags_slope_out_of_range(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    blob = copy.deepcopy(legacy_bundle_dict)
    blob["csw_to_k_relationship"] = {
        "intercept": -0.22, "slope": 10.0, "r_squared": 0.6,
        "method": "weighted_linear_regression_csw_to_k",
    }
    issues = validate(blob)
    assert any("slope" in i for i in issues)


def test_validator_flags_wrong_method(legacy_bundle_dict):
    from scripts.bundle_validator import validate
    blob = copy.deepcopy(legacy_bundle_dict)
    blob["csw_to_k_relationship"] = {
        "intercept": -0.22, "slope": 1.63, "r_squared": 0.6,
        "method": "polynomial_csw_to_k_v2",
    }
    issues = validate(blob)
    assert any("method" in i for i in issues)
