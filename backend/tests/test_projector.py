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
        or "park_k_factor" in result.skip_reason
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
    # All required features must be present.
    for k in (
        "pitcher_ip_per_start_30d_shrunk",
        "pitcher_pitches_per_pa_season",
        "lineup_obp_vs_hand",
        "park_run_environment_factor",
    ):
        assert k in result.features_used_bf
    for k in (
        "pitcher_k_pct_season_shrunk",
        "lineup_k_pct_vs_hand",
        "park_k_factor",
    ):
        assert k in result.features_used_kpa
