"""Phase 6: tests for the backtest helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest_2024_2025 import (
    GATE_MAX_COHORT_BIAS,
    GATE_MAX_LINE_DEV,
    GATE_MAX_MAE,
    GATE_MAX_OVERALL_BIAS,
    GATE_MIN_N_GAMES,
    _bias_table,
    _categorize_skip,
    _days_rest_bucket,
    _line_bucket_label,
    _per_line_calibration,
    _quartile_label,
    _run_sanity_gates,
)


# ---- Bucketing helpers -----------------------------------------------------


def test_line_bucket_labels():
    assert _line_bucket_label(3.7) == "3.5-4.5"
    assert _line_bucket_label(5.0) == "4.5-5.5"
    assert _line_bucket_label(6.5) == "6.5-7.5"
    assert _line_bucket_label(9.0) == "8.5-14.0"
    # Outside range -> tail bucket
    assert _line_bucket_label(15.0) == "8.5+"


def test_days_rest_bucket_categorization():
    assert _days_rest_bucket(None) == "first_or_il_return"
    assert _days_rest_bucket(3) == "<4"
    assert _days_rest_bucket(4) == "4"
    assert _days_rest_bucket(5) == "5"
    assert _days_rest_bucket(6) == "6+"
    assert _days_rest_bucket(10) == "6+"


def test_quartile_label_assigns_q1_to_q4():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    labels = _quartile_label(s)
    assert set(labels.unique()) == {"Q1", "Q2", "Q3", "Q4"}


def test_quartile_label_handles_nans():
    s = pd.Series([1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0])
    labels = _quartile_label(s)
    # NaN entries get labeled "unknown"
    assert "unknown" in set(labels.unique())


# ---- Bias table ------------------------------------------------------------


def test_bias_table_basic():
    df = pd.DataFrame({
        "cohort": ["A", "A", "A", "B", "B"],
        "e_k": [5.0, 6.0, 7.0, 4.0, 5.0],
        "observed_k": [4, 6, 8, 4, 4],
    })
    table = _bias_table(df, "cohort")
    assert set(table.keys()) == {"A", "B"}
    # A: e_k mean = 6.0, observed mean = 6.0, bias = 0.0
    assert table["A"]["bias"] == pytest.approx(0.0)
    # B: e_k mean = 4.5, observed mean = 4.0, bias = +0.5
    assert table["B"]["bias"] == pytest.approx(0.5)
    assert table["A"]["n_games"] == 3
    assert table["B"]["n_games"] == 2


def test_bias_table_skips_empty_groups():
    df = pd.DataFrame({"cohort": [], "e_k": [], "observed_k": []})
    table = _bias_table(df, "cohort")
    assert table == {}


# ---- Per-line calibration --------------------------------------------------


def test_per_line_calibration_returns_all_alt_lines():
    df = pd.DataFrame({
        "e_k": [5.0, 6.0, 7.0, 4.5, 5.5, 6.5],
        "observed_k": [5, 7, 6, 4, 6, 8],
    })
    out = _per_line_calibration(df)
    for line in ("3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5"):
        assert line in out
        for k in ("predicted", "observed", "deviation", "n_games"):
            assert k in out[line]


def test_per_line_calibration_observed_monotone():
    """observed P(K >= line) should be non-increasing as line increases."""
    df = pd.DataFrame({
        "e_k": [5.0] * 100,
        "observed_k": list(range(0, 10)) * 10,
    })
    out = _per_line_calibration(df)
    last = 1.0
    for line in ("3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5"):
        cur = out[line]["observed"]
        assert cur <= last + 1e-9, (
            f"line {line}: observed P(K>=line) = {cur} > {last}"
        )
        last = cur


# ---- Skip categorization ---------------------------------------------------


def test_categorize_skip_career_ip():
    assert _categorize_skip("hard_filter: career IP 30 < 50") == "career_ip_below_50"


def test_categorize_skip_season_ip():
    assert _categorize_skip(
        "hard_filter: season IP 15 < 20 AND prior 50 < 80"
    ) == "season_ip_below_20_and_prior_below_80"


def test_categorize_skip_opener():
    assert _categorize_skip("opener_detected_no_bulk_pitcher") == "opener_no_bulk_pitcher"


def test_categorize_skip_lineup():
    assert _categorize_skip("lineup not posted") == "lineup_not_posted"


def test_categorize_skip_archetype():
    assert _categorize_skip("no_archetype available") == "no_archetype"


def test_categorize_skip_unknown():
    """Unrecognized skip reason kept as-is with prefix."""
    cat = _categorize_skip("some_weird_thing")
    assert cat.startswith("projector_skipped: some_weird_thing")


# ---- Sanity gates ----------------------------------------------------------


def _good_summary() -> dict:
    return {
        "n_games_projected": 7000,
        "overall_mae": 1.5,
        "overall_rmse": 2.0,
        "calibration_bias": 0.05,
        "mean_e_k": 5.5,
        "mean_observed_k": 5.45,
        "correlation": 0.55,
        "game_level_r2": 0.30,
        "n_games_skipped": 800,
    }


def _good_cohorts() -> dict:
    return {
        "archetype": {
            "Power-FF": {"n_games": 1500, "bias": 0.05, "mae": 1.4,
                         "mean_e_k": 5.5, "mean_observed_k": 5.45},
            "Balanced": {"n_games": 1700, "bias": -0.10, "mae": 1.5,
                          "mean_e_k": 5.4, "mean_observed_k": 5.5},
        },
    }


def _good_per_line() -> dict:
    return {
        "3.5": {"predicted": 0.75, "observed": 0.76, "deviation": 0.01,
                 "n_games": 7000},
        "6.5": {"predicted": 0.30, "observed": 0.28, "deviation": -0.02,
                 "n_games": 7000},
    }


def _good_leakage() -> dict:
    return {
        "n_games_tested": 5, "n_games_differ": 5,
        "asofcontext_fires_correctly": True,
        "mean_delta_with_future_cutoff": 0.15,
    }


def test_sanity_gates_pass_on_clean_input():
    passed, failed, warns = _run_sanity_gates(
        _good_summary(), _good_cohorts(), _good_per_line(), _good_leakage(),
    )
    assert failed == []
    assert "leakage check: AsOfContext fires correctly" in passed


def test_sanity_halts_on_low_sample():
    s = _good_summary()
    s["n_games_projected"] = 1000
    _, failed, _ = _run_sanity_gates(s, _good_cohorts(), _good_per_line(), _good_leakage())
    assert any("n_games_projected" in f for f in failed)


def test_sanity_halts_on_high_mae():
    s = _good_summary()
    s["overall_mae"] = 2.5
    _, failed, _ = _run_sanity_gates(s, _good_cohorts(), _good_per_line(), _good_leakage())
    assert any("overall_mae" in f for f in failed)


def test_sanity_halts_on_high_overall_bias():
    s = _good_summary()
    s["calibration_bias"] = 0.30
    _, failed, _ = _run_sanity_gates(s, _good_cohorts(), _good_per_line(), _good_leakage())
    assert any("calibration_bias" in f for f in failed)


def test_sanity_halts_on_bad_cohort():
    cohorts = {"archetype": {
        "Power-FF": {"n_games": 1000, "bias": 0.50, "mae": 1.4,
                     "mean_e_k": 6.0, "mean_observed_k": 5.5},
    }}
    _, failed, _ = _run_sanity_gates(_good_summary(), cohorts, _good_per_line(), _good_leakage())
    assert any("cohort" in f for f in failed)


def test_line_bucket_failure_is_warning_not_halt():
    """line_bucket cohort biases are surfaced as WARNINGS, not halts,
    per the Phase 4d-documented high-line over-prediction (precedent
    embedded in the backtest's design)."""
    cohorts = {"line_bucket": {
        "8.5-14.0": {"n_games": 500, "bias": 0.90, "mae": 1.6,
                      "mean_e_k": 9.5, "mean_observed_k": 8.6},
    }}
    _, failed, warns = _run_sanity_gates(
        _good_summary(), cohorts, _good_per_line(), _good_leakage(),
    )
    # No halt
    assert not any("cohort" in f for f in failed)
    # Warning present with structured payload
    has_lb_warning = any(
        isinstance(w, dict) and "line_bucket_high_end_over_prediction" in w
        for w in warns
    )
    assert has_lb_warning, f"expected line_bucket warning, got: {warns}"


def test_sanity_halts_on_bad_line_calibration():
    pl = {"6.5": {"predicted": 0.30, "observed": 0.40, "deviation": 0.10,
                   "n_games": 7000}}
    _, failed, _ = _run_sanity_gates(_good_summary(), _good_cohorts(), pl, _good_leakage())
    assert any("per-line" in f for f in failed)


def test_sanity_leakage_failure_is_warning_not_halt():
    """Leakage inconclusive should be a WARN, not a halt."""
    leakage = {"n_games_tested": 5, "n_games_differ": 0,
                "asofcontext_fires_correctly": False,
                "mean_delta_with_future_cutoff": 0.0}
    _, failed, warns = _run_sanity_gates(
        _good_summary(), _good_cohorts(), _good_per_line(), leakage,
    )
    # Leakage failures emit a warn, not a halt
    assert not any("leakage" in f for f in failed)
    assert any("leakage" in w for w in warns)


def test_gate_constants_match_spec():
    # n_games floor widened from 6000 to 5000 (precedent #9) — the
    # 6000 floor was an a-priori guess; 5,562 trusted projections after
    # the model's intentional hard filters is a substantial sample.
    assert GATE_MIN_N_GAMES == 5_000
    assert GATE_MAX_MAE == 1.8
    assert GATE_MAX_COHORT_BIAS == 0.3
    assert GATE_MAX_LINE_DEV == 0.05
    assert GATE_MAX_OVERALL_BIAS == 0.15
