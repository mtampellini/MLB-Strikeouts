"""Phase 4c-v2 Step 3/6: tests for the projector's effective-K-rate wiring."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.projection.effective_k_rate import load_csw_to_k_relationship
from src.projection.inputs import ProjectionBundle, ProjectionContext
from src.projection.projector import (
    _compute_pitcher_effective_k_rate,
    project,
)

PHASE3_V2C_I_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase3_v2c_i_sample_bundle_2026-05-17_656876.json"
)


def _ctx_with_csw() -> ProjectionContext:
    """Full default-paths ctx (loads CSW-to-K params)."""
    return ProjectionContext.from_default_paths()


def _ctx_without_csw() -> ProjectionContext:
    """Build a ctx with the CSW-to-K params explicitly nulled out — the
    legacy-context path."""
    ctx = ProjectionContext.from_default_paths()
    return replace(ctx, csw_to_k_intercept=None, csw_to_k_slope=None)


# ---- Legacy ctx (no CSW-to-K params) ---------------------------------------


def test_legacy_ctx_falls_back_to_disk_step4(monkeypatch):
    """Step 4 changed the resolver priority chain to: bundle > ctx > disk
    > None. With a ctx that has no CSW params, the projector now falls
    back to the disk file (csw_to_k_relationship.json), NOT immediately
    to the legacy log5 path. The legacy path only fires when all three
    sources are unavailable."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_without_csw()
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    # Bundle from Phase 3-v2c-i has no csw_to_k_relationship field, so
    # the resolver falls to ctx (empty) then disk (file exists).
    assert bundle.csw_to_k_relationship is None
    result = project(bundle, ctx)
    assert result.skipped is False
    meta = result.per_batter_breakdown.get("_meta")
    assert meta is not None
    # Disk fallback fired → blend is real, not legacy.
    assert meta["blend_confidence"] != "legacy_no_csw_blend"
    assert meta.get("fallback_to_composed_p_k_pa") is not True


def test_truly_legacy_path_when_all_three_sources_missing(monkeypatch):
    """When the bundle has no field, ctx has no params, AND the disk
    file is unreachable, the projector falls to the legacy composed
    P(K|PA) for log5."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    # Monkeypatch the disk loader to simulate a missing file.
    import src.projection.projector as projector_mod
    monkeypatch.setattr(
        projector_mod, "load_csw_to_k_relationship",
        lambda: (_ for _ in ()).throw(FileNotFoundError("simulated")),
    )
    ctx = _ctx_without_csw()
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    assert bundle.csw_to_k_relationship is None
    result = project(bundle, ctx)
    assert result.skipped is False
    meta = result.per_batter_breakdown.get("_meta")
    assert meta is not None
    assert meta["blend_confidence"] == "legacy_no_csw_blend"
    assert meta.get("fallback_to_composed_p_k_pa") is True


# ---- Full ctx (CSW-to-K params loaded) -------------------------------------


def test_full_ctx_uses_effective_k_rate():
    """The default ctx (from_default_paths) loads CSW-to-K params, so the
    blend kicks in. _meta should show a real effective rate + blend weight."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_with_csw()
    # The CSW-to-K file must exist for this assertion to be meaningful.
    if ctx.csw_to_k_intercept is None:
        pytest.skip("CSW-to-K relationship file not loaded into default ctx")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    assert result.skipped is False
    meta = result.per_batter_breakdown["_meta"]
    assert meta["blend_confidence"] in (
        "observed_dominant", "balanced", "csw_dominant",
        "csw_only_fallback", "observed_only_no_csw",
    )
    assert meta["pitcher_effective_k_rate"] is not None
    assert 0.05 <= meta["pitcher_effective_k_rate"] <= 0.50
    assert "fallback_to_composed_p_k_pa" not in meta


# ---- _meta block shape -----------------------------------------------------


def test_meta_block_has_all_expected_fields():
    """The _meta block must surface every component the consumer needs to
    audit the blend decision."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_with_csw()
    if ctx.csw_to_k_intercept is None:
        pytest.skip("CSW-to-K relationship file not loaded into default ctx")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    meta = result.per_batter_breakdown["_meta"]
    for field in (
        "pitcher_effective_k_rate", "pitcher_observed_k_rate",
        "pitcher_observed_n_pa", "pitcher_csw_pct",
        "pitcher_csw_implied_k_rate", "blend_weight_observed",
        "blend_confidence",
    ):
        assert field in meta, f"missing field: {field}"


def test_meta_blend_weight_in_valid_range():
    """blend_weight_observed must be in [0, 1] (when present)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_with_csw()
    if ctx.csw_to_k_intercept is None:
        pytest.skip("CSW-to-K relationship file not loaded into default ctx")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    w = result.per_batter_breakdown["_meta"]["blend_weight_observed"]
    assert 0.0 <= w <= 1.0


# ---- _compute_pitcher_effective_k_rate (helper) ----------------------------


def test_helper_returns_none_for_pitcher_with_no_data(synthetic_bundle_no_pitcher_data):
    """A pitcher with empty statcast windows: no observed K, no CSW% ->
    returns (None, meta) with blend_confidence='no_data_available' (or
    'legacy_no_csw_blend' on a ctx without CSW params)."""
    ctx = _ctx_with_csw()
    if ctx.csw_to_k_intercept is None:
        pytest.skip("CSW-to-K relationship file not loaded into default ctx")
    eff, meta = _compute_pitcher_effective_k_rate(synthetic_bundle_no_pitcher_data, ctx)
    assert eff is None
    assert meta["blend_confidence"] in ("no_data_available", "legacy_no_csw_blend")


def test_helper_legacy_path_when_all_sources_missing(monkeypatch):
    """When bundle has no field, ctx has no params, AND disk is mocked
    unreachable, the helper returns (None, legacy meta)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    import src.projection.projector as projector_mod
    monkeypatch.setattr(
        projector_mod, "load_csw_to_k_relationship",
        lambda: (_ for _ in ()).throw(FileNotFoundError("simulated")),
    )
    ctx = _ctx_without_csw()
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    eff, meta = _compute_pitcher_effective_k_rate(bundle, ctx)
    assert eff is None
    assert meta["blend_confidence"] == "legacy_no_csw_blend"


def test_helper_uses_bundle_field_when_present(monkeypatch):
    """Step 4 priority: when bundle.csw_to_k_relationship is populated,
    the helper uses it even if ctx is also populated (bundle wins)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    from dataclasses import replace
    from src.projection.inputs import CswToKRelationship

    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    # Inject a deliberately-different relationship into the bundle so we
    # can distinguish it from ctx / disk by the meta output.
    bundle = replace(
        bundle,
        csw_to_k_relationship=CswToKRelationship(
            intercept=0.0, slope=1.0, r_squared=0.6,
            method="weighted_linear_regression_csw_to_k",
        ),
    )
    ctx = _ctx_with_csw()
    eff, meta = _compute_pitcher_effective_k_rate(bundle, ctx)
    assert eff is not None
    # csw_implied_k = 0.0 + 1.0 * csw_pct = csw_pct itself
    csw_val = meta["pitcher_csw_pct"]
    expected_csw_implied = round(csw_val, 4)
    assert meta["pitcher_csw_implied_k_rate"] == expected_csw_implied


# ---- Round-trip through dict ----------------------------------------------


def test_projection_result_to_dict_round_trips_meta():
    """ProjectionResult.to_dict() -> json.dumps -> json.loads preserves the
    _meta block in per_batter_breakdown."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_with_csw()
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    blob = json.dumps(result.to_dict(), default=str)
    rt = json.loads(blob)
    assert "per_batter_breakdown" in rt
    assert "_meta" in rt["per_batter_breakdown"]
    assert rt["per_batter_breakdown"]["_meta"]["blend_confidence"] in (
        "observed_dominant", "balanced", "csw_dominant",
        "csw_only_fallback", "observed_only_no_csw",
        "legacy_no_csw_blend", "no_data_available",
    )


# ---- Rasmussen reference point ---------------------------------------------


def test_rasmussen_full_ctx_e_k_remains_in_research_range():
    """The blend shifts e_k a bit but should stay in the reasonable range
    for Rasmussen vs MIA at the Trop."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_with_csw()
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    assert 22.0 <= result.e_bf <= 28.0
    assert 4.5 <= result.e_k <= 8.0


# ---- Fixtures --------------------------------------------------------------


@pytest.fixture
def synthetic_bundle_no_pitcher_data():
    """A bundle where the pitcher has zero Statcast pitches in all windows
    (early-season, returning-from-IL, or just a feed gap)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    blob = json.loads(PHASE3_V2C_I_SAMPLE.read_text(encoding="utf-8"))
    blob = copy.deepcopy(blob)
    blob["pitcher"]["statcast_pitches_30d"] = []
    blob["pitcher"]["statcast_pitches_season"] = []
    blob["pitcher"]["statcast_pitches_prior_year"] = []
    return ProjectionBundle.from_dict(blob)
