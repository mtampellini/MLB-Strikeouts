"""Phase 4c-v2 Step 6/6: tests for the FittedKPACoefficients loader."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.projection.fitted_coefficients import (
    DEFAULT_PATH,
    FittedKPACoefficients,
    load_fitted_kpa_coefficients,
)


def test_loader_reads_live_file():
    """The Step 5 --full output should load and have all the structure
    the projector expects."""
    if not DEFAULT_PATH.exists():
        pytest.skip(f"{DEFAULT_PATH} not present (run Step 5 --full first)")
    fit = load_fitted_kpa_coefficients()
    assert isinstance(fit, FittedKPACoefficients)
    # Main features must have a sign-correct positive value
    assert fit.pitcher_velocity_trend_z > 0
    assert fit.log_park_k_factor_by_hand > 0
    # Rate-space R² out should pass the Step 5 gate
    assert fit.rate_space_r2_out >= 0.05


def test_loader_archetype_tto_has_all_5_archetypes():
    """The fit emits coefficients for arch_<each>_tto_<1..4> except the
    (Balanced, TTO=1) reference cell."""
    if not DEFAULT_PATH.exists():
        pytest.skip(f"{DEFAULT_PATH} not present")
    fit = load_fitted_kpa_coefficients()
    expected = {"Power-FF", "Sinker-ball", "Breaking-heavy",
                "Offspeed-heavy", "Balanced"}
    assert set(fit.archetype_tto.keys()) == expected
    # Non-Balanced archetypes carry TTO 1..4
    for arch in expected - {"Balanced"}:
        assert set(fit.archetype_tto[arch].keys()) == {1, 2, 3, 4}, (
            f"{arch} missing TTO buckets: {sorted(fit.archetype_tto[arch])}"
        )
    # Balanced carries TTO 2..4 only (TTO=1 is the absolute reference cell)
    assert set(fit.archetype_tto["Balanced"].keys()) == {2, 3, 4}


def test_get_archetype_tto_shift_returns_zero_for_reference_cell():
    if not DEFAULT_PATH.exists():
        pytest.skip(f"{DEFAULT_PATH} not present")
    fit = load_fitted_kpa_coefficients()
    assert fit.get_archetype_tto_shift("Balanced", 1) == 0.0


def test_get_archetype_tto_shift_returns_fitted_value():
    if not DEFAULT_PATH.exists():
        pytest.skip(f"{DEFAULT_PATH} not present")
    fit = load_fitted_kpa_coefficients()
    # Power-FF TTO=3 should be negative (third time through penalty)
    val = fit.get_archetype_tto_shift("Power-FF", 3)
    assert val < 0


def test_get_archetype_tto_shift_returns_zero_for_unknown_archetype():
    """Defensive: an archetype not in the fit (shouldn't happen since the
    projector maps 'unknown' to 'Balanced' upstream) returns 0.0."""
    if not DEFAULT_PATH.exists():
        pytest.skip(f"{DEFAULT_PATH} not present")
    fit = load_fitted_kpa_coefficients()
    assert fit.get_archetype_tto_shift("UnknownArchetype-X", 2) == 0.0


def test_loader_raises_filenotfounderror_on_missing(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="fit_p_k_pa_v2"):
        load_fitted_kpa_coefficients(missing)


def test_loader_raises_value_error_when_gates_failed(tmp_path):
    """A fit with gates_failed != [] must refuse to load — shipping a
    broken fit silently is the failure mode this guards against."""
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({
            "intercept": 0.0,
            "model_type": "logistic_regression_per_pa_with_log5_offset",
            "rate_space_r2_weighted_out": 0.10,
            "leakage_shuffle_delta": 0.05,
            "generated_at": "2026-05-19T00:00:00Z",
            "gates_failed": ["some_gate_failed"],
            "coefficients": {
                "pitcher_velocity_trend_z": {"value": 0.1},
                "log_park_k_factor_by_hand": {"value": 0.5},
            },
        }), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gates_failed"):
        load_fitted_kpa_coefficients(bad)


def test_loader_raises_on_missing_main_feature(tmp_path):
    bad = tmp_path / "missing.json"
    bad.write_text(
        json.dumps({
            "intercept": 0.0,
            "gates_failed": [],
            "coefficients": {
                "pitcher_velocity_trend_z": {"value": 0.1},
                # missing log_park_k_factor_by_hand
            },
        }), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed"):
        load_fitted_kpa_coefficients(bad)


def test_loader_handles_archetype_names_with_hyphens(tmp_path):
    """Power-FF, Sinker-ball etc. contain hyphens. Loader must split keys
    on '_tto_' (not the first '_') to handle this correctly."""
    good = tmp_path / "ok.json"
    good.write_text(
        json.dumps({
            "intercept": 0.05,
            "model_type": "logistic_regression_per_pa_with_log5_offset",
            "rate_space_r2_weighted_out": 0.14,
            "leakage_shuffle_delta": 0.05,
            "generated_at": "2026-05-19T00:00:00Z",
            "gates_failed": [],
            "coefficients": {
                "pitcher_velocity_trend_z": {"value": 0.1},
                "log_park_k_factor_by_hand": {"value": 0.9},
                "arch_Power-FF_tto_3": {"value": -0.23},
                "arch_Sinker-ball_tto_2": {"value": -0.13},
            },
        }), encoding="utf-8",
    )
    fit = load_fitted_kpa_coefficients(good)
    assert fit.get_archetype_tto_shift("Power-FF", 3) == -0.23
    assert fit.get_archetype_tto_shift("Sinker-ball", 2) == -0.13
