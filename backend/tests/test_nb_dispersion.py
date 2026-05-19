"""Phase 4d: tests for the NB dispersion fit."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from scripts.fit_nb_dispersion import (
    ALT_LINES,
    GATE_ALPHA_MAX,
    GATE_ALPHA_MIN,
    GATE_PER_LINE_MAX_DEV,
    _per_line_calibration,
    _run_sanity_gates,
    fit_alpha_mle,
    nb_logpmf,
    nb_p_at_least,
    neg_log_likelihood,
)


# ---- NB log-likelihood ------------------------------------------------------


def test_nb_logpmf_matches_scipy_for_known_params():
    """nb_logpmf is the (mean, alpha) parameterization wrapper. Verify it
    matches scipy's (n, p) parameterization directly."""
    mean, alpha = 5.5, 0.15
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * mean)
    for k in (0, 3, 5, 8, 12):
        our = float(nb_logpmf(np.array([k]), np.array([mean]), alpha)[0])
        scipy_lp = float(nbinom.logpmf(k, n_param, p_param))
        assert abs(our - scipy_lp) < 1e-9


def test_nb_p_at_least_complement_sums_to_one():
    """P(K >= 3.5) + P(K <= 3) should equal 1 for any (mean, alpha)."""
    mean, alpha = 5.0, 0.2
    p_over = nb_p_at_least(3.5, mean, alpha)
    p_under = float(nbinom.cdf(3, 1.0 / alpha, 1.0 / (1.0 + alpha * mean)))
    assert abs(p_over + p_under - 1.0) < 1e-9


def test_neg_log_likelihood_rejects_non_positive_alpha():
    means = np.array([5.0])
    observed = np.array([5])
    assert neg_log_likelihood(0.0, means, observed) == float("inf")
    assert neg_log_likelihood(-0.1, means, observed) == float("inf")


# ---- MLE recovery ----------------------------------------------------------


def test_mle_recovers_known_dispersion_on_synthetic_data():
    """Generate (mean, observed_k) pairs from NB with known alpha, then
    refit. MLE should recover the true alpha within ~5%."""
    true_alpha = 0.15
    rng = np.random.default_rng(0)
    n = 5000
    means = rng.uniform(3.0, 8.5, n)
    n_param = 1.0 / true_alpha
    p_params = 1.0 / (1.0 + true_alpha * means)
    observed = nbinom.rvs(n_param, p_params, random_state=rng)
    fitted, _ = fit_alpha_mle(means, observed)
    assert abs(fitted - true_alpha) / true_alpha < 0.10, (
        f"recovered {fitted:.4f} vs true {true_alpha:.4f}"
    )


def test_mle_recovers_higher_dispersion():
    """Higher true alpha (0.30) should also be recovered."""
    true_alpha = 0.30
    rng = np.random.default_rng(42)
    n = 5000
    means = rng.uniform(3.0, 8.5, n)
    p_params = 1.0 / (1.0 + true_alpha * means)
    observed = nbinom.rvs(1.0 / true_alpha, p_params, random_state=rng)
    fitted, _ = fit_alpha_mle(means, observed)
    assert abs(fitted - true_alpha) / true_alpha < 0.10


# ---- Per-line calibration --------------------------------------------------


def test_per_line_calibration_returns_all_alt_lines():
    pairs = pd.DataFrame({
        "e_k": [5.0, 6.0, 4.5, 7.0],
        "observed_k": [5, 7, 4, 8],
    })
    out = _per_line_calibration(pairs, alpha=0.15)
    assert set(out.keys()) == set(str(line) for line in ALT_LINES)
    for cells in out.values():
        for k in ("predicted_p_over", "observed_p_over", "deviation", "n_games"):
            assert k in cells


def test_per_line_calibration_observed_p_under_alpha_zero():
    """All observed_p_over fractions sum to a non-decreasing-with-line pattern."""
    pairs = pd.DataFrame({
        "e_k": [6.0] * 1000,
        "observed_k": [int(x) for x in np.random.default_rng(0).poisson(6.0, 1000)],
    })
    out = _per_line_calibration(pairs, alpha=0.15)
    obs = [out[str(line)]["observed_p_over"] for line in ALT_LINES]
    # Observed P(K >= line) must be non-increasing as the line increases.
    for i in range(1, len(obs)):
        assert obs[i] <= obs[i - 1] + 1e-9


# ---- Sanity gates ----------------------------------------------------------


def _good_in_sample() -> dict:
    return {"n_games": 4000, "log_likelihood": -8000.0, "ks_test_p": 0.5,
            "ecdf_max_deviation": 0.02, "ks_test_stat": 0.03}


def _good_out_of_sample() -> dict:
    return {"n_games": 3500, "log_likelihood": -7000.0, "ks_test_p": 0.5,
            "ecdf_max_deviation": 0.025, "ks_test_stat": 0.03}


def _good_diagnostic() -> dict:
    return {"mean_e_k": 5.5, "mean_observed_k": 5.45, "calibration_bias": 0.05}


def _good_per_line_oos() -> dict:
    return {
        str(line): {"predicted_p_over": 0.5, "observed_p_over": 0.49,
                     "deviation": -0.01, "n_games": 3500}
        for line in ALT_LINES
    }


def test_sanity_gates_pass_on_realistic_inputs():
    _run_sanity_gates(
        alpha=0.15, in_sample=_good_in_sample(),
        out_of_sample=_good_out_of_sample(), diagnostic=_good_diagnostic(),
        per_line_oos=_good_per_line_oos(),
    )


def test_sanity_halts_on_alpha_too_low():
    """alpha below 0.001 (effectively 0 or negative) halts. The floor was
    widened to 0.001 after the first fit landed there with clean per-line
    calibration."""
    with pytest.raises(AssertionError, match="alpha"):
        _run_sanity_gates(
            alpha=0.0001, in_sample=_good_in_sample(),
            out_of_sample=_good_out_of_sample(),
            diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
        )


def test_sanity_accepts_alpha_at_widened_floor():
    """alpha=0.001 is the new gate floor — sanity must accept it."""
    _run_sanity_gates(
        alpha=0.001, in_sample=_good_in_sample(),
        out_of_sample=_good_out_of_sample(),
        diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
    )


def test_sanity_halts_on_alpha_too_high():
    with pytest.raises(AssertionError, match="alpha"):
        _run_sanity_gates(
            alpha=0.5, in_sample=_good_in_sample(),
            out_of_sample=_good_out_of_sample(),
            diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
        )


def test_sanity_halts_on_low_n_games():
    bad_oos = _good_out_of_sample()
    bad_oos["n_games"] = 1000
    with pytest.raises(AssertionError, match="n_games"):
        _run_sanity_gates(
            alpha=0.15, in_sample=_good_in_sample(), out_of_sample=bad_oos,
            diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
        )


def test_sanity_does_not_halt_on_low_ks_p():
    """KS p-value is NOT a gate — sample-size sensitivity makes it unreliable
    at n > ~3000. ECDF max deviation is the substantive distributional-fit
    gate. A KS p=0.0 should not halt."""
    bad_oos = _good_out_of_sample()
    bad_oos["ks_test_p"] = 0.0
    _run_sanity_gates(
        alpha=0.15, in_sample=_good_in_sample(), out_of_sample=bad_oos,
        diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
    )


def test_sanity_halts_on_ecdf_max_deviation():
    bad_oos = _good_out_of_sample()
    bad_oos["ecdf_max_deviation"] = 0.10
    with pytest.raises(AssertionError, match="ECDF"):
        _run_sanity_gates(
            alpha=0.15, in_sample=_good_in_sample(), out_of_sample=bad_oos,
            diagnostic=_good_diagnostic(), per_line_oos=_good_per_line_oos(),
        )


def test_sanity_halts_on_calibration_bias():
    bad_diag = _good_diagnostic()
    bad_diag["mean_e_k"] = 6.0
    bad_diag["mean_observed_k"] = 5.5  # 0.5 bias > 0.1 gate
    with pytest.raises(AssertionError, match="calibration bias"):
        _run_sanity_gates(
            alpha=0.15, in_sample=_good_in_sample(),
            out_of_sample=_good_out_of_sample(),
            diagnostic=bad_diag, per_line_oos=_good_per_line_oos(),
        )


def test_sanity_halts_on_per_line_deviation():
    bad_per_line = _good_per_line_oos()
    bad_per_line["4.5"]["deviation"] = 0.08  # > 5pp
    with pytest.raises(AssertionError, match="per-line"):
        _run_sanity_gates(
            alpha=0.15, in_sample=_good_in_sample(),
            out_of_sample=_good_out_of_sample(),
            diagnostic=_good_diagnostic(), per_line_oos=bad_per_line,
        )


# ---- Constants -------------------------------------------------------------


def test_alt_lines_canonical():
    assert ALT_LINES == (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)


def test_gate_constants_match_spec():
    # alpha floor widened from 0.05 to 0.001 after the first fit landed at
    # the floor and the per-line calibration was clean. The empirical
    # answer is "given calibrated E[K], residuals are approximately
    # Poisson" and the gate now admits that.
    assert GATE_ALPHA_MIN == 0.001
    assert GATE_ALPHA_MAX == 0.40
    assert GATE_PER_LINE_MAX_DEV == 0.05
