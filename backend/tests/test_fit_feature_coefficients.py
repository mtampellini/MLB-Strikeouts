"""Tests for the Phase 4c fit script's pure-logic helpers.

The end-to-end fit is validated via the `--smoke` CLI run, not unit tests.
These tests cover the small gating helpers where unit testing is cheap and
the behavior is invariant across data.
"""
from __future__ import annotations

import pytest

from scripts.design_matrix import InsufficientSampleError
from scripts.fit_feature_coefficients import _check_min_sample_unless_smoke


def test_smoke_mode_skips_min_sample_check():
    """In smoke mode the min-sample gate is bypassed entirely — the smoke
    purpose is pipeline validation, not statistical adequacy."""
    result = _check_min_sample_unless_smoke(10, 100, smoke=True)
    assert result is False  # check skipped


def test_smoke_skip_does_not_raise_even_with_zero_rows():
    """Smoke must remain a clean exit path regardless of sample size."""
    assert _check_min_sample_unless_smoke(0, 2000, smoke=True) is False


def test_non_smoke_mode_enforces_min_sample():
    """The require_min_sample gate runs and raises on too few rows when
    smoke is False. This is the real-fit path."""
    with pytest.raises(InsufficientSampleError, match="10 < 100"):
        _check_min_sample_unless_smoke(10, 100, smoke=False)


def test_non_smoke_mode_passes_when_sample_sufficient():
    """When sample meets the threshold, the check returns True and runs
    without raising."""
    assert _check_min_sample_unless_smoke(2000, 2000, smoke=False) is True
    assert _check_min_sample_unless_smoke(5000, 100, smoke=False) is True


def test_context_message_in_error():
    """When the gate fires in non-smoke mode, the error carries the context
    string so the operator sees what the gate was guarding."""
    with pytest.raises(InsufficientSampleError, match="post-NaN-drop"):
        _check_min_sample_unless_smoke(
            10, 100, smoke=False, context="post-NaN-drop fit pool"
        )
