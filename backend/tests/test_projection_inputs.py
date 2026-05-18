"""Phase 3a tests: the typed input contract + loader + static-data helpers."""
from __future__ import annotations

import copy
import json
import warnings
from datetime import date
from pathlib import Path

import pytest

from src.projection.inputs import (
    BatterInputs,
    BookLine,
    BookMarket,
    BundleMetadata,
    ContractViolation,
    GameContext,
    HandednessAverages,
    LeagueAverages,
    Market,
    MLB_TEAM_ABBRS,
    OpposingLineup,
    ParkFactors,
    PitcherInputs,
    ProjectionBundle,
    UmpireKFactors,
    WeatherInputs,
)


SAMPLE_BUNDLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "phase2c_sample_bundle_2026-05-17_656876.json"
)


@pytest.fixture
def minimal_bundle() -> dict:
    """Hand-rolled minimal bundle that passes every contract check."""
    return {
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
            "statcast_pitches_30d": [],
            "statcast_pitches_season": [],
            "statcast_pitches_prior_year": [],
        },
        "opposing_lineup": {
            "team_abbr": "MIA",
            "batters": [
                {
                    "mlbam_id": 669364,
                    "name": "Xavier Edwards",
                    "batting_order": 1,
                    "handedness": "S",
                    "position": "2B",
                    "statcast_pa_season": [],
                }
            ],
            "lineup_posted": True,
        },
        "game_context": {
            "venue_id": 12,
            "venue_name": "Tropicana Field",
            "is_dome": True,
            "weather": {
                "temp_f": 72.0,
                "wind_speed_mph": 0.0,
                "wind_direction": None,
                "humidity_pct": None,
                "conditions": "Dome",
            },
            "umpire_name": "Ramon De Jesus",
            "umpire_id": 594151,
            "first_pitch_iso": "2026-05-17T16:15:00+00:00",
            "days_rest": 6,
        },
        "market": {
            "fanduel": {"available": False, "lines": []},
            "draftkings": {"available": False, "lines": []},
            "snapshot_timestamp": None,
            "snapshot_source": "missing",
        },
    }


# ---- Loader basics ---------------------------------------------------------


def test_from_dict_constructs_typed_struct(minimal_bundle):
    bundle = ProjectionBundle.from_dict(minimal_bundle)
    assert isinstance(bundle, ProjectionBundle)
    assert bundle.metadata.game_date == date(2026, 5, 17)
    assert bundle.metadata.cutoff_date == date(2026, 5, 16)
    assert bundle.pitcher.handedness == "R"
    assert bundle.pitcher.team_abbr == "TBR"
    assert bundle.opposing_lineup.lineup_posted is True
    assert len(bundle.opposing_lineup.batters) == 1
    assert bundle.opposing_lineup.batters[0].handedness == "S"
    assert bundle.game_context.is_dome is True
    assert bundle.market.snapshot_source == "missing"


def test_round_trip_preserves_shape(minimal_bundle):
    """Load -> to_dict -> JSON serialize -> compare to original."""
    bundle = ProjectionBundle.from_dict(minimal_bundle)
    round_tripped = bundle.to_dict()
    # JSON-roundtrip both sides so ordering and types are normalized.
    expected = json.loads(json.dumps(minimal_bundle, sort_keys=True))
    actual = json.loads(json.dumps(round_tripped, sort_keys=True))
    assert actual == expected


def test_from_json_loads_phase2c_sample_bundle():
    if not SAMPLE_BUNDLE_PATH.exists():
        pytest.skip("Phase 2c sample bundle not present")
    bundle = ProjectionBundle.from_json(SAMPLE_BUNDLE_PATH)
    # Spot-check against known sample
    assert bundle.metadata.pitcher_mlbam_id == 656876
    assert bundle.metadata.game_pk == 822982
    assert bundle.pitcher.team_abbr == "TBR"
    assert bundle.opposing_lineup.team_abbr == "MIA"
    assert bundle.opposing_lineup.lineup_posted is True
    assert len(bundle.opposing_lineup.batters) == 9
    assert bundle.game_context.venue_id == 12
    assert bundle.game_context.is_dome is True
    # Pitches are dicts (list[dict] per spec, NOT DataFrames)
    assert isinstance(bundle.pitcher.statcast_pitches_30d, tuple)
    assert all(isinstance(r, dict) for r in bundle.pitcher.statcast_pitches_30d)


def test_dataclass_is_frozen(minimal_bundle):
    bundle = ProjectionBundle.from_dict(minimal_bundle)
    with pytest.raises(Exception):  # FrozenInstanceError is a dataclass.FrozenInstanceError subclass of AttributeError
        bundle.metadata = None  # type: ignore[misc]


# ---- Validation rule 1: required top-level keys ---------------------------


def test_missing_top_level_key_raises(minimal_bundle):
    del minimal_bundle["market"]
    with pytest.raises(ContractViolation, match="market"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_non_dict_top_level_raises():
    with pytest.raises(ContractViolation):
        ProjectionBundle.from_dict([])  # type: ignore[arg-type]


# ---- Validation rule 2: cutoff < game_date ---------------------------------


def test_future_cutoff_raises_value_error(minimal_bundle):
    minimal_bundle["metadata"]["cutoff_date"] = "2026-05-17"  # == game_date
    with pytest.raises(ValueError, match="strictly before"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_cutoff_after_game_date_raises(minimal_bundle):
    minimal_bundle["metadata"]["cutoff_date"] = "2026-05-18"
    with pytest.raises(ValueError):
        ProjectionBundle.from_dict(minimal_bundle)


# ---- Validation rule 3: handedness consistency (warn, not raise) ----------


def test_handedness_mismatch_warns(minimal_bundle):
    minimal_bundle["pitcher"]["statcast_pitches_30d"] = [
        {"p_throws": "L", "game_pk": 1, "at_bat_number": 1, "pitch_number": 1}
    ]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        bundle = ProjectionBundle.from_dict(minimal_bundle)
    assert any("handedness mismatch" in str(w.message) for w in captured)
    # Top-level handedness wins; bundle still constructs.
    assert bundle.pitcher.handedness == "R"


def test_handedness_match_does_not_warn(minimal_bundle):
    minimal_bundle["pitcher"]["statcast_pitches_30d"] = [
        {"p_throws": "R", "game_pk": 1, "at_bat_number": 1, "pitch_number": 1}
    ]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ProjectionBundle.from_dict(minimal_bundle)
    assert not any("handedness mismatch" in str(w.message) for w in captured)


# ---- Validation rule 4: team_abbr allowlist --------------------------------


def test_invalid_pitcher_team_abbr_raises(minimal_bundle):
    minimal_bundle["pitcher"]["team_abbr"] = "ZZZ"
    with pytest.raises(ContractViolation, match="team_abbr"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_invalid_opposing_team_abbr_raises(minimal_bundle):
    minimal_bundle["opposing_lineup"]["team_abbr"] = "XXX"
    with pytest.raises(ContractViolation, match="team_abbr"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_allowlist_includes_30_teams_plus_oak_alias():
    assert len(MLB_TEAM_ABBRS) == 31  # 30 + OAK alias for ATH
    for t in ("ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL",
              "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM",
              "NYY", "ATH", "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR",
              "TEX", "TOR", "WSN"):
        assert t in MLB_TEAM_ABBRS


# ---- Validation rule 5: empty-state contract -------------------------------


def test_empty_batters_with_lineup_posted_true_raises(minimal_bundle):
    minimal_bundle["opposing_lineup"]["batters"] = []
    # lineup_posted stays True
    with pytest.raises(ContractViolation, match="lineup_posted=True"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_batters_with_lineup_posted_false_raises(minimal_bundle):
    minimal_bundle["opposing_lineup"]["lineup_posted"] = False
    # batters still has one entry
    with pytest.raises(ContractViolation, match="lineup_posted=False"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_book_available_false_with_lines_raises(minimal_bundle):
    minimal_bundle["market"]["fanduel"]["lines"] = [
        {"line": 5.5, "side": "Over", "price": 100}
    ]
    with pytest.raises(ContractViolation, match="available=False"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_book_available_true_with_lines_ok(minimal_bundle):
    minimal_bundle["market"]["fanduel"] = {
        "available": True,
        "lines": [{"line": 5.5, "side": "Over", "price": 100}],
    }
    bundle = ProjectionBundle.from_dict(minimal_bundle)
    assert bundle.market.fanduel.available is True
    assert len(bundle.market.fanduel.lines) == 1


# ---- Validation rule 6: no unsupported types -------------------------------


def test_pandas_timestamp_in_payload_raises(minimal_bundle):
    """If a pandas.Timestamp or numpy scalar leaks past the bundle builder,
    the loader catches it."""
    import pandas as pd

    minimal_bundle["metadata"]["generated_at"] = pd.Timestamp("2026-05-17T23:00:00Z")  # type: ignore[assignment]
    with pytest.raises(ContractViolation, match="unsupported"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_numpy_scalar_in_payload_raises(minimal_bundle):
    import numpy as np

    minimal_bundle["pitcher"]["mlbam_id"] = np.int64(656876)  # type: ignore[assignment]
    with pytest.raises(ContractViolation, match="unsupported"):
        ProjectionBundle.from_dict(minimal_bundle)


# ---- Enum / source validation ---------------------------------------------


def test_invalid_pitcher_source_raises(minimal_bundle):
    minimal_bundle["pitcher"]["source"] = "made_up"
    with pytest.raises(ContractViolation, match="pitcher.source"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_invalid_opener_result_raises(minimal_bundle):
    minimal_bundle["pitcher"]["opener_detection_result"] = "?"
    with pytest.raises(ContractViolation, match="opener_detection_result"):
        ProjectionBundle.from_dict(minimal_bundle)


def test_invalid_snapshot_source_raises(minimal_bundle):
    minimal_bundle["market"]["snapshot_source"] = "stale"
    with pytest.raises(ContractViolation, match="snapshot_source"):
        ProjectionBundle.from_dict(minimal_bundle)


# ---- LeagueAverages --------------------------------------------------------


def test_league_averages_loads_from_disk():
    path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed" / "league_averages_2025.json"
    )
    la = LeagueAverages.from_json(path)
    assert la.year == 2025
    assert 0.20 < la.r_vs_r.k_pct < 0.30


def test_league_averages_lookup_handedness_combos():
    la = _stub_league_averages()
    assert la.lookup("R", "R") is la.r_vs_r
    assert la.lookup("L", "R") is la.l_vs_r
    assert la.lookup("R", "L") is la.r_vs_l
    assert la.lookup("L", "L") is la.l_vs_l


def test_league_averages_switch_hitter_resolves_to_opposite():
    la = _stub_league_averages()
    # Switch hitter vs R pitcher -> use L split.
    assert la.lookup("S", "R") is la.l_vs_r
    # Switch hitter vs L pitcher -> use R split.
    assert la.lookup("S", "L") is la.r_vs_l


def test_league_averages_returns_none_for_unknown_hand():
    la = _stub_league_averages()
    assert la.lookup(None, "R") is None
    assert la.lookup("R", None) is None
    assert la.lookup("R", "X") is None


def _stub_league_averages() -> LeagueAverages:
    def avg(k):
        return HandednessAverages(k_pct=k, obp=0.3, zone_contact_pct=0.8, chase_rate=0.3)
    return LeagueAverages(
        year=2025,
        r_vs_r=avg(0.225),
        r_vs_l=avg(0.220),
        l_vs_r=avg(0.235),
        l_vs_l=avg(0.265),
    )


# ---- ParkFactors / UmpireKFactors -----------------------------------------


def test_park_factors_loads_from_disk():
    path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed" / "park_factors.json"
    )
    pf = ParkFactors.from_json(path)
    # Tropicana baseline (Phase 3a run-factor placeholder still at 1.0).
    assert pf.get(12) == 1.0
    assert pf.get(19) > 1.0      # Coors elevated
    assert pf.get(999999) is None


def test_park_k_factors_loads_from_disk():
    """Phase 4b park K factors are derived from 2023-25 Statcast. Tropicana
    came in slightly K-promoting (~1.07); T-Mobile the highest (~1.16);
    Nationals the lowest (~0.97). We assert structural shape and a sane
    range rather than exact values which will drift with overnight rerun.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed" / "park_k_factors.json"
    )
    pf = ParkFactors.from_json(path)
    trop = pf.get(12)
    assert trop is not None
    assert 0.85 <= trop <= 1.20
    assert pf.get(999999) is None


def test_umpire_k_factors_missing_defaults_to_1(tmp_path):
    path = tmp_path / "umps.json"
    path.write_text(json.dumps({"factors": {}}), encoding="utf-8")
    uk = UmpireKFactors.from_json(path)
    assert uk.get(594151) == 1.0  # unknown ump = neutral
    assert uk.get(None) == 1.0


def test_umpire_k_factors_returns_real_value_when_present(tmp_path):
    path = tmp_path / "umps.json"
    path.write_text(json.dumps({"factors": {"100": 1.05}}), encoding="utf-8")
    uk = UmpireKFactors.from_json(path)
    assert uk.get(100) == 1.05
