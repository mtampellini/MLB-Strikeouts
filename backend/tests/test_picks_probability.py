"""Phase 5: tests for probability.py."""
from __future__ import annotations

import math

import pytest
from scipy.stats import nbinom

from src.picks.probability import (
    model_probabilities_for_pitcher,
    p_k_geq,
    p_k_leq,
)


# Phase 4d locked alpha
ALPHA = 0.001


def test_p_k_geq_matches_scipy_at_half_integer_line():
    """For line=5.5, P(K>=5.5) means K must be 6 or more — i.e. 1 - CDF(5)."""
    e_k, alpha = 5.5, ALPHA
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * e_k)
    expected = float(1.0 - nbinom.cdf(5, n_param, p_param))
    assert abs(p_k_geq(5.5, e_k, alpha) - expected) < 1e-9


def test_p_k_leq_matches_scipy_at_half_integer_line():
    """For line=5.5, P(K<=5.5) means K is 5 or fewer — CDF(5)."""
    e_k, alpha = 5.5, ALPHA
    expected = float(nbinom.cdf(5, 1.0 / alpha, 1.0 / (1.0 + alpha * e_k)))
    assert abs(p_k_leq(5.5, e_k, alpha) - expected) < 1e-9


def test_half_integer_lines_sum_to_one():
    """For ANY half-integer line and any (e_k, alpha), p_over + p_under == 1
    (no probability mass at the line itself)."""
    for line in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5):
        for e_k in (3.0, 5.5, 8.0):
            for alpha in (0.001, 0.05, 0.20):
                p_over = p_k_geq(line, e_k, alpha)
                p_under = p_k_leq(line, e_k, alpha)
                assert abs(p_over + p_under - 1.0) < 1e-9, (
                    f"line={line} e_k={e_k} alpha={alpha}: "
                    f"p_over+p_under={p_over + p_under}"
                )


def test_integer_lines_have_push_mass():
    """For integer lines, p_over + p_under + P(K==line) == 1. Verify that
    the missing mass is the push probability."""
    line = 5.0
    e_k = 5.5
    alpha = 0.001
    n_param = 1.0 / alpha
    p_param = 1.0 / (1.0 + alpha * e_k)
    p_over = p_k_geq(line, e_k, alpha)  # P(K >= 5)
    p_under = p_k_leq(line, e_k, alpha)  # P(K <= 4)
    p_push = float(nbinom.pmf(5, n_param, p_param))
    assert abs(p_over + p_under + p_push - 1.0) < 1e-9


def test_p_over_decreasing_in_line():
    """At fixed (e_k, alpha), P(K >= line) is monotone decreasing in line."""
    e_k, alpha = 6.0, 0.001
    last = 1.0
    for line in (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5):
        cur = p_k_geq(line, e_k, alpha)
        assert cur < last, f"line={line}: p_over={cur} >= prev {last}"
        last = cur


def test_p_over_increasing_in_e_k():
    """At fixed line and alpha, P(K >= line) is monotone increasing in mean."""
    line, alpha = 6.5, 0.001
    last = 0.0
    for e_k in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
        cur = p_k_geq(line, e_k, alpha)
        assert cur > last, f"e_k={e_k}: p_over={cur} <= prev {last}"
        last = cur


def test_model_probabilities_for_pitcher_returns_all_requested_lines():
    out = model_probabilities_for_pitcher(
        e_k=5.5, alpha=ALPHA, lines=(3.5, 5.5, 7.5),
    )
    assert set(out.keys()) == {3.5, 5.5, 7.5}
    for line, (p_over, p_under) in out.items():
        assert 0.0 < p_over < 1.0
        assert 0.0 < p_under < 1.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="alpha"):
        p_k_geq(5.5, 6.0, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        p_k_geq(5.5, 6.0, alpha=-0.1)
    with pytest.raises(ValueError, match="mean"):
        p_k_geq(5.5, 0.0, alpha=0.001)
