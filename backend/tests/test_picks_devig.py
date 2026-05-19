"""Phase 5: tests for devig.py."""
from __future__ import annotations

import pytest

from src.picks.devig import (
    MAX_IMPUTATION_DISTANCE,
    american_to_decimal,
    devig_with_imputation,
    implied_from_american,
    multiplicative_devig,
    nearest_paired_line_imputation,
)


# ---- American <-> decimal --------------------------------------------------


def test_american_to_decimal_negative():
    assert american_to_decimal(-110) == pytest.approx(1.0 + 100 / 110)
    assert american_to_decimal(-200) == pytest.approx(1.50)
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=0.001)


def test_american_to_decimal_positive():
    assert american_to_decimal(+100) == pytest.approx(2.0)
    assert american_to_decimal(+150) == pytest.approx(2.5)
    assert american_to_decimal(+250) == pytest.approx(3.5)


def test_american_to_decimal_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_implied_from_american_negative():
    assert implied_from_american(-110) == pytest.approx(110 / 210, abs=1e-9)
    assert implied_from_american(-200) == pytest.approx(2 / 3, abs=1e-9)


def test_implied_from_american_positive():
    assert implied_from_american(+100) == pytest.approx(0.5)
    assert implied_from_american(+150) == pytest.approx(100 / 250, abs=1e-9)


# ---- Multiplicative devig --------------------------------------------------


def test_multiplicative_devig_symmetric():
    """At -110/-110 (typical 4.5% vig market), both true probs are 0.5."""
    over_p = implied_from_american(-110)
    under_p = implied_from_american(-110)
    true_over, true_under = multiplicative_devig(over_p, under_p)
    assert true_over == pytest.approx(0.5, abs=1e-9)
    assert true_under == pytest.approx(0.5, abs=1e-9)
    assert true_over + true_under == pytest.approx(1.0, abs=1e-9)


def test_multiplicative_devig_asymmetric():
    """At -130/+110 the over implies ~56.5% raw, under implies ~47.6% raw.
    Devig produces ~54.3% / 45.7%."""
    over_p = implied_from_american(-130)
    under_p = implied_from_american(+110)
    true_over, true_under = multiplicative_devig(over_p, under_p)
    assert true_over > 0.50
    assert true_under < 0.50
    assert true_over + true_under == pytest.approx(1.0, abs=1e-9)


def test_multiplicative_devig_zero_sum_raises():
    with pytest.raises(ValueError):
        multiplicative_devig(0.0, 0.0)


# ---- Nearest-paired-line imputation ----------------------------------------


def test_imputation_uses_nearest_paired_line():
    """One-sided line 6.5 with a paired 5.5 line should impute via 5.5's vig."""
    lines = {
        5.5: {"Over": -110, "Under": -110},  # paired, 4.55% vig
        6.5: {"Over": +150},                  # one-sided, missing Under
    }
    imputed = nearest_paired_line_imputation(lines, 6.5, "Under")
    assert imputed is not None
    # The paired 5.5 has vig = (110/210 + 110/210) - 1 ≈ 0.0476
    # Existing 6.5 Over at +150 has raw p = 100/250 = 0.4
    # Imputed under raw p = (1 + 0.0476) - 0.4 ≈ 0.6476
    assert imputed == pytest.approx(0.6476, abs=0.001)


def test_imputation_returns_none_when_no_paired_line_within_distance():
    """Paired line at 4.0 vs target 6.5 -> 2.5 distance, exceeds 1.5 cap."""
    lines = {
        4.0: {"Over": -110, "Under": -110},  # paired but 2.5 K away
        6.5: {"Over": +150},
    }
    assert nearest_paired_line_imputation(lines, 6.5, "Under") is None


def test_imputation_prefers_closest_paired_line():
    """Two paired lines available; the closer one wins."""
    lines = {
        5.5: {"Over": -110, "Under": -110},     # 1.0 K away
        7.5: {"Over": -120, "Under": +100},     # 1.0 K away (tied)
        6.5: {"Over": +150},                     # target
    }
    # Sorted by abs(line - target_line) — ties broken by sort stability.
    # Either could be picked, but the function must return a valid number.
    imputed = nearest_paired_line_imputation(lines, 6.5, "Under")
    assert imputed is not None
    assert 0.0 <= imputed <= 1.0


def test_imputation_target_line_not_in_dict_returns_none():
    lines = {5.5: {"Over": -110, "Under": -110}}
    assert nearest_paired_line_imputation(lines, 7.5, "Under") is None


def test_max_imputation_distance_is_15():
    assert MAX_IMPUTATION_DISTANCE == 1.5


# ---- devig_with_imputation: top-level --------------------------------------


def test_devig_two_sided_returns_two_sided_source():
    lines = {5.5: {"Over": -110, "Under": -110}}
    result = devig_with_imputation(lines, 5.5, "Over")
    assert result is not None
    p, source = result
    assert source == "two_sided"
    assert p == pytest.approx(0.5, abs=1e-9)


def test_devig_one_sided_imputed_when_paired_exists():
    lines = {
        5.5: {"Over": -110, "Under": -110},
        6.5: {"Over": +150},
    }
    # Asking for Over at 6.5 — needs to impute the missing Under to devig.
    result = devig_with_imputation(lines, 6.5, "Over")
    assert result is not None
    p, source = result
    assert source == "imputed_nearest_pair"
    # Over+Under should equal 1.0 in the devigged frame
    over_p_raw = implied_from_american(+150)  # 0.4
    imputed_under_raw = nearest_paired_line_imputation(lines, 6.5, "Under")
    expected = over_p_raw / (over_p_raw + imputed_under_raw)
    assert p == pytest.approx(expected, abs=1e-9)


def test_devig_missing_side_imputed():
    """Asking for the side that's NOT posted should impute the side."""
    lines = {
        5.5: {"Over": -110, "Under": -110},
        6.5: {"Over": +150},  # only Over posted
    }
    # Asking for Under at 6.5: this side is missing, impute its raw, then devig.
    result = devig_with_imputation(lines, 6.5, "Under")
    assert result is not None
    p, source = result
    assert source == "imputed_nearest_pair"
    assert 0.0 < p < 1.0


def test_devig_returns_none_when_no_paired_line():
    """One-sided line with no paired line within distance — can't devig."""
    lines = {6.5: {"Over": +150}}  # only this line, no paired
    assert devig_with_imputation(lines, 6.5, "Under") is None
    assert devig_with_imputation(lines, 6.5, "Over") is None


def test_devig_returns_none_when_target_line_missing():
    lines = {5.5: {"Over": -110, "Under": -110}}
    assert devig_with_imputation(lines, 9.5, "Over") is None
