"""Phase 4c-v2 Step 6/6: tests for the projector's fitted-coefficient wiring."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.projection.fitted_coefficients import (
    DEFAULT_PATH as FITTED_DEFAULT_PATH,
    load_fitted_kpa_coefficients,
)
from src.projection.inputs import ProjectionBundle, ProjectionContext
from src.projection.projector import project


PHASE3_V2C_I_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase3_v2c_i_sample_bundle_2026-05-17_656876.json"
)
STEP4_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "phase4c_v2_step4_sample_bundle_2026-05-17_656876.json"
)


def _ctx_with_fitted() -> ProjectionContext:
    """Default ctx (which loads fitted coefs from disk if present)."""
    return ProjectionContext.from_default_paths()


def _ctx_without_fitted() -> ProjectionContext:
    """Force the placeholder path by nulling fitted coefs on a default ctx."""
    ctx = ProjectionContext.from_default_paths()
    return replace(ctx, fitted_kpa_coefficients=None)


# ---- Path routing ----------------------------------------------------------


def test_projector_uses_fitted_when_ctx_has_coefs():
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    ctx = _ctx_with_fitted()
    assert ctx.fitted_kpa_coefficients is not None
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    assert result.skipped is False
    pmd = result.per_batter_breakdown["_meta"]["projection_method_details"]
    assert pmd["kpa_coefficients_source"] == "fitted"
    assert pmd["kpa_rate_space_r2_out"] > 0.05


def test_projector_uses_placeholder_when_ctx_has_no_fitted_coefs():
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    ctx = _ctx_without_fitted()
    assert ctx.fitted_kpa_coefficients is None
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, ctx)
    assert result.skipped is False
    pmd = result.per_batter_breakdown["_meta"]["projection_method_details"]
    assert pmd["kpa_coefficients_source"] == "placeholder"


def test_fitted_and_placeholder_paths_produce_close_but_distinct_e_k():
    """Same bundle, two ctx variants — e_k differs (fitted shifts log-odds)
    but stays close (the offset already absorbs the bulk of K skill)."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result_fitted = project(bundle, _ctx_with_fitted())
    result_placeholder = project(bundle, _ctx_without_fitted())
    # Both produce a valid e_k in the research range.
    assert 3.0 <= result_fitted.e_k <= 8.0
    assert 3.0 <= result_placeholder.e_k <= 8.0
    # They differ (fitted shifts the per-cell log-odds).
    assert result_fitted.e_k != result_placeholder.e_k


# ---- Coefficient application -----------------------------------------------


def test_fitted_archetype_tto_shifts_are_applied_per_cell():
    """The per-batter breakdown's by_tto entries should reflect the
    (archetype, TTO) shifts when the fitted path is active."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, _ctx_with_fitted())
    # TTO=3 cells should have lower p_k than TTO=1 cells for the same batter
    # (third-time-through penalty is captured by the archetype-TTO shifts).
    for k, entry in result.per_batter_breakdown.items():
        if k == "_meta":
            continue
        tto_1 = entry["by_tto"][1]
        tto_3 = entry["by_tto"][3]
        if tto_1["pa"] > 0 and tto_3["pa"] > 0:
            assert tto_3["p_k"] < tto_1["p_k"], (
                f"batter {k}: TTO=3 p_k {tto_3['p_k']} >= TTO=1 {tto_1['p_k']}"
            )


def test_game_log_odds_shift_recorded_in_meta():
    """When fitted path runs, _meta.projection_method_details surfaces the
    game-level log-odds shift for debugging."""
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, _ctx_with_fitted())
    pmd = result.per_batter_breakdown["_meta"]["projection_method_details"]
    assert "game_log_odds_shift" in pmd
    assert "fitted_intercept" in pmd


# ---- Round-trip + Rasmussen ------------------------------------------------


def test_rasmussen_step4_bundle_projects_with_fitted_coefs():
    """The Step 4 sample bundle (with embedded csw_to_k_relationship)
    projects cleanly with fitted coefficients applied."""
    if not STEP4_SAMPLE.exists():
        pytest.skip(f"{STEP4_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    bundle = ProjectionBundle.from_json(STEP4_SAMPLE)
    result = project(bundle, _ctx_with_fitted())
    assert result.skipped is False
    assert 22.0 <= result.e_bf <= 28.0
    assert 3.5 <= result.e_k <= 7.0
    pmd = result.per_batter_breakdown["_meta"]["projection_method_details"]
    assert pmd["kpa_coefficients_source"] == "fitted"


def test_to_dict_round_trips_projection_method_details():
    if not PHASE3_V2C_I_SAMPLE.exists():
        pytest.skip(f"{PHASE3_V2C_I_SAMPLE} not present")
    if not FITTED_DEFAULT_PATH.exists():
        pytest.skip("fitted KPA coefficients file not present")
    bundle = ProjectionBundle.from_json(PHASE3_V2C_I_SAMPLE)
    result = project(bundle, _ctx_with_fitted())
    blob = json.loads(json.dumps(result.to_dict(), default=str))
    meta = blob["per_batter_breakdown"]["_meta"]
    assert meta["projection_method_details"]["kpa_coefficients_source"] == "fitted"
