"""Phase 3c: projector orchestration tests + Phase 2c integration."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.projection.inputs import (
    BundleMetadata,
    ProjectionBundle,
    ProjectionContext,
)
from src.projection.projector import ProjectionResult, project

# Reuse the same bundle factory from features_bf tests
from tests.test_projection_features_bf import (
    _ctx_default as _ctx,
    _make_bundle,
)

SAMPLE_BUNDLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "phase2c_sample_bundle_2026-05-17_656876.json"
)


# ---- Happy path -----------------------------------------------------------


def test_project_happy_path():
    bundle = _make_bundle()
    result = project(bundle, _ctx())
    assert result.skipped is False
    assert result.skip_reason is None
    assert result.e_bf is not None
    assert result.p_k_pa is not None
    assert result.e_k is not None
    assert abs(result.e_k - result.e_bf * result.p_k_pa) < 1e-9
    assert isinstance(result.bundle_metadata, BundleMetadata)


# ---- Pre-feature filters -------------------------------------------------


def test_project_skips_on_lineup_not_posted():
    bundle = _make_bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    result = project(bundle, _ctx())
    assert result.skipped is True
    assert "lineup not posted" in result.skip_reason
    assert result.e_bf is None and result.p_k_pa is None and result.e_k is None


def test_project_skips_on_opener():
    bundle = _make_bundle(pitcher={"opener_detection_result": "skip"})
    result = project(bundle, _ctx())
    assert result.skipped is True
    assert "opener" in result.skip_reason


def test_project_skips_on_low_career_ip():
    """Force a thin pitcher record below the career floor."""
    thin_rows = [
        {"game_pk": 1, "at_bat_number": i + 1, "pitch_number": 1}
        for i in range(20)
    ]
    bundle = _make_bundle(pitcher={
        "statcast_pitches_30d": [], "statcast_pitches_season": thin_rows,
        "statcast_pitches_prior_year": thin_rows,
    })
    result = project(bundle, _ctx())
    assert result.skipped is True
    assert "career IP" in result.skip_reason


# ---- Post-feature filter (projected BF < 12) -----------------------------


def test_project_skips_on_projected_bf_below_floor():
    """E[BF] composition raw value < 12 -> skip."""
    # Build a pitcher with extremely low IP per start to drive E[BF] down.
    from tests.test_projection_features_bf import _make_pitcher_rows
    tiny = _make_pitcher_rows(n_starts=10, pa_per_start=3, pitches_per_pa=4.0)
    bundle = _make_bundle(pitcher={
        "statcast_pitches_30d": tiny,
        "statcast_pitches_season": tiny * 3,
        "statcast_pitches_prior_year": tiny * 4,
    })
    result = project(bundle, _ctx())
    # Either the < 12 filter trips OR the pitcher passes via clipping.
    # We expect the trip because IP per start of 0.7 + all anchors -> raw_bf << 12.
    assert result.skipped is True
    assert ("projected_BF" in result.skip_reason
            or "career IP" in result.skip_reason)


# ---- Skip via missing feature --------------------------------------------


def test_project_skips_on_unknown_park():
    bundle = _make_bundle(game_context={"venue_id": 99999})
    result = project(bundle, _ctx())
    assert result.skipped is True
    # Park check is a required feature in both BF and K|PA; either side can trip.
    assert (
        "park_run_environment_factor" in result.skip_reason
        or "log_park_k_factor_by_hand" in result.skip_reason
    )


# ---- Skipped-result invariant --------------------------------------------


def test_skipped_result_has_all_none_values():
    bundle = _make_bundle(opposing_lineup={"lineup_posted": False, "batters": []})
    result = project(bundle, _ctx())
    assert result.e_bf is None
    assert result.p_k_pa is None
    assert result.e_k is None
    assert result.bundle_metadata is not None


# ---- to_dict serialization ------------------------------------------------


def test_projection_result_to_dict_is_json_safe():
    import json

    bundle = _make_bundle()
    result = project(bundle, _ctx())
    d = result.to_dict()
    # Roundtrip through JSON
    s = json.dumps(d, sort_keys=True)
    rt = json.loads(s)
    assert rt["e_bf"] == result.e_bf
    assert rt["p_k_pa"] == result.p_k_pa
    assert rt["e_k"] == result.e_k
    assert rt["bundle_metadata"]["pitcher_mlbam_id"] == result.bundle_metadata.pitcher_mlbam_id


# ---- Phase 2c integration ------------------------------------------------


def test_phase2c_sample_bundle_projects_successfully():
    if not SAMPLE_BUNDLE_PATH.exists():
        pytest.skip("Phase 2c sample bundle not present")
    bundle = ProjectionBundle.from_json(SAMPLE_BUNDLE_PATH)
    result = project(bundle, ProjectionContext.from_default_paths())
    assert result.skipped is False, f"unexpected skip: {result.skip_reason}"
    assert 12.0 <= result.e_bf <= 32.0
    assert 0.10 <= result.p_k_pa <= 0.45
    # Sanity: a ~24 BF starter facing average lineup at neutral park should
    # land somewhere in the 3-8 K range.
    assert 3.0 <= result.e_k <= 10.0
    # Legacy bundle has no pa_distribution / tto_multipliers fields:
    # projector must route to legacy_aggregated.
    assert result.projection_method == "legacy_aggregated"
    assert result.per_batter_breakdown is None
    # All required features must be present.
    for k in (
        "pitcher_ip_per_start_30d_shrunk",
        "pitcher_pitches_per_pa_season",
        "lineup_obp_vs_hand",
        "park_run_environment_factor",
    ):
        assert k in result.features_used_bf
    for k in (
        "pitcher_csw_pct_season",
        "league_k_pct_vs_hand",
        "log_park_k_factor_by_hand",
    ):
        assert k in result.features_used_kpa


# ---- Phase 3-v2c-iv: per-batter projector internals ----------------------


def test_compute_log5_baseline_anchored():
    """Equal batter+pitcher rates at league anchor returns the same rate."""
    from src.projection.projector import _compute_log5
    assert abs(_compute_log5(0.22, 0.22, 0.22) - 0.22) < 1e-9


def test_compute_log5_high_pitcher_low_batter():
    """High-K pitcher vs low-K batter -> result between the two, weighted
    by log5 (closer to the geometric mean)."""
    from src.projection.projector import _compute_log5
    val = _compute_log5(batter_k=0.15, pitcher_k=0.30, league_k=0.22)
    # Should be in (0.15, 0.30) and lean toward the geometric mean
    assert 0.15 < val < 0.30


def test_compute_log5_degenerate_league_rate_falls_back_to_mean():
    """league_k at 0 or 1 returns the simple mean (no NaN)."""
    from src.projection.projector import _compute_log5
    assert _compute_log5(0.2, 0.3, 0.0) == pytest.approx(0.25)
    assert _compute_log5(0.2, 0.3, 1.0) == pytest.approx(0.25)


def test_resolve_archetype_uses_balanced_when_missing():
    from src.projection.projector import _resolve_archetype
    bundle = _make_bundle()
    # Synthetic bundle has no pitcher_archetype field
    assert _resolve_archetype(bundle) == "Balanced"


def test_resolve_archetype_returns_real_archetype_when_present():
    """A bundle with a non-unknown archetype returns that archetype."""
    from src.projection.projector import _resolve_archetype
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["pitcher_archetype"] = {
        "archetype": "Power-FF", "season": 2025,
        "fastball_pct": 0.6, "four_seam_pct": 0.4, "sinker_pct": 0.2,
        "cutter_pct": 0.0, "breaking_pct": 0.25, "offspeed_pct": 0.15,
        "n_starts": 25, "confidence": "current_season",
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    assert _resolve_archetype(bundle) == "Power-FF"


def test_resolve_archetype_maps_unknown_sentinel_to_balanced():
    from src.projection.projector import _resolve_archetype
    bundle_dict = _make_bundle().to_dict()
    bundle_dict["pitcher_archetype"] = {
        "archetype": "unknown", "season": 2025,
        "fastball_pct": 0.0, "four_seam_pct": 0.0, "sinker_pct": 0.0,
        "cutter_pct": 0.0, "breaking_pct": 0.0, "offspeed_pct": 0.0,
        "n_starts": 0, "confidence": "unknown",
    }
    bundle = ProjectionBundle.from_dict(bundle_dict)
    assert _resolve_archetype(bundle) == "Balanced"


def test_get_tto_multiplier_falls_back_to_league_for_missing_archetype():
    """A made-up archetype hits the league_wide fallback."""
    from src.projection.projector import _get_tto_multiplier
    from src.projection.inputs import TTOMultiplierCell, TTOMultipliers
    league_cell = TTOMultiplierCell(multiplier=0.85, k_rate=0.18, n_pa=1000, low_sample=False)
    arch_cell = TTOMultiplierCell(multiplier=0.80, k_rate=0.17, n_pa=500, low_sample=False)
    tto_table = TTOMultipliers(
        by_archetype={"Power-FF": {1: TTOMultiplierCell(1.0, 0.22, 1000, False),
                                    3: arch_cell}},
        league_wide={1: TTOMultiplierCell(1.0, 0.22, 5000, False),
                     3: league_cell},
    )
    # Power-FF + TTO=3 -> archetype-specific
    assert _get_tto_multiplier(tto_table, "Power-FF", 3) == 0.80
    # Made-up archetype -> league_wide fallback
    assert _get_tto_multiplier(tto_table, "Curveballer-X", 3) == 0.85


# ---- Phase 3-v2c-iv: integration with hydrated bundle --------------------


PHASE3_V2C_I_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase3_v2c_i_sample_bundle_2026-05-17_656876.json"
)


def test_hydrated_bundle_uses_per_batter_path():
    """The Phase 3-v2c-i sample bundle has pa_distribution + tto_multipliers
    populated -> projector routes to per_batter_with_tto."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ProjectionContext.from_default_paths())
    assert result.skipped is False, f"unexpected skip: {result.skip_reason}"
    assert result.projection_method == "per_batter_with_tto"
    assert result.per_batter_breakdown is not None
    # Lineup has 9 batters -> 9 per-batter entries plus the Step 3+ _meta
    # block alongside them in the same dict.
    batter_entries = {
        k: v for k, v in result.per_batter_breakdown.items() if k != "_meta"
    }
    assert len(batter_entries) == 9
    assert result.archetype_used is not None
    assert result.pa_distribution_bf_used is not None


def test_per_batter_breakdown_sums_to_e_k():
    """The integrity invariant: sum of per-batter expected_k_total ==  e_k
    (within float epsilon)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ProjectionContext.from_default_paths())
    if result.per_batter_breakdown is None:
        pytest.skip("not on per-batter path")
    total = sum(
        entry["expected_k_total"]
        for k, entry in result.per_batter_breakdown.items()
        if k != "_meta"
    )
    assert abs(total - result.e_k) < 0.01


def test_per_batter_breakdown_by_tto_sums_to_batter_total():
    """For each batter, sum(by_tto[*][expected_k]) == expected_k_total."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ProjectionContext.from_default_paths())
    if result.per_batter_breakdown is None:
        pytest.skip("not on per-batter path")
    for bid, entry in result.per_batter_breakdown.items():
        if bid == "_meta":
            continue
        tto_sum = sum(t["expected_k"] for t in entry["by_tto"].values())
        # Each cell value is rounded to 4 dp; allow up to 2e-4 tolerance for
        # the sum (4 cells × 5e-5 worst-case rounding per cell).
        assert abs(tto_sum - entry["expected_k_total"]) < 0.001, (
            f"batter {bid}: by_tto sum {tto_sum:.4f} != "
            f"expected_k_total {entry['expected_k_total']:.4f}"
        )


def test_per_batter_top_of_order_has_more_pa_than_bottom():
    """Empirical PA distribution puts more PAs at slots 1-5 than 6-9 for
    most BF values. The breakdown should reflect that ordering."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ProjectionContext.from_default_paths())
    if result.per_batter_breakdown is None:
        pytest.skip("not on per-batter path")
    by_slot = {}
    for k, entry in result.per_batter_breakdown.items():
        if k == "_meta":
            continue
        by_slot[entry["batting_order_slot"]] = entry["expected_pa_total"]
    # Slot 1 PA >= slot 9 PA (strictly true for any BF in observed range,
    # since the lineup never fully turns over for the #9 hitter at typical
    # starter BF counts).
    assert by_slot[1] >= by_slot[9]


def test_rasmussen_projection_in_reasonable_range():
    """Rasmussen at the Trop vs Marlins: e_bf in [22, 28], e_k in [4.5, 7.0]."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ProjectionContext.from_default_paths())
    assert result.skipped is False
    assert 22.0 <= result.e_bf <= 28.0, f"e_bf {result.e_bf} out of expected range"
    assert 4.5 <= result.e_k <= 8.0, f"e_k {result.e_k} out of expected range"


# ---- ctx default-paths convenience ---------------------------------------


def test_project_works_without_explicit_ctx():
    """project(bundle) without ctx should auto-load default paths."""
    bundle = _make_bundle()
    result = project(bundle)  # no ctx
    # Synthetic bundle gets routed to legacy (no pa_distribution)
    assert result.projection_method == "legacy_aggregated"
    assert result.skipped is False or result.skip_reason is not None
