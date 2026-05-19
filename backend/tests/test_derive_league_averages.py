"""Phase 4a: unit tests for the league-averages derivation script.

The script's heavy work (the Statcast pull) is tested by hand running it
against the real feed. These tests cover the pure functions: split
labeling, split computation, and sanity assertions.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.derive_league_averages import (
    LEAGUE_K_PCT_RANGE,
    _build_payload,
    _compute_split,
    _pa_terminal_rows,
    _sanity_check,
    _split_label,
)


def _row(*, events=None, description="ball", zone=5, stand="R", p_throws="R",
         game_pk=1, at_bat_number=1, pitch_number=1):
    return {
        "events": events, "description": description, "zone": zone,
        "stand": stand, "p_throws": p_throws, "game_pk": game_pk,
        "at_bat_number": at_bat_number, "pitch_number": pitch_number,
        "game_type": "R",
    }


# ---- _split_label ----------------------------------------------------------


def test_split_label_recognizes_four_combos():
    assert _split_label("R", "R") == "RR"
    assert _split_label("R", "L") == "RL"
    assert _split_label("L", "R") == "LR"
    assert _split_label("L", "L") == "LL"


def test_split_label_rejects_switch_stand():
    """Statcast 'stand' never carries 'S' — the live batting side is what shows."""
    assert _split_label("S", "R") is None


def test_split_label_rejects_garbage():
    assert _split_label("X", "R") is None
    assert _split_label("R", "Z") is None
    assert _split_label(None, "R") is None  # type: ignore[arg-type]


# ---- _compute_split --------------------------------------------------------


def test_compute_split_handles_empty_input():
    out = _compute_split(pd.DataFrame(), pd.DataFrame())
    assert out == {
        "k_pct": None, "obp": None, "zone_contact_pct": None,
        "chase_rate": None, "csw_pct": None, "n_pa": 0,
    }


def test_compute_split_basic_rates():
    """Build a tiny in-memory frame with known counts and verify rates."""
    # 10 PAs: 3 strikeouts, 2 walks, 1 single, 4 field_outs.
    # K% = 0.30, OBP = (1+2+0)/10 = 0.30 (no SF or SH).
    pa_rows = pd.DataFrame([
        _row(events="strikeout"), _row(events="strikeout"), _row(events="strikeout"),
        _row(events="walk"), _row(events="walk"),
        _row(events="single"),
        _row(events="field_out"), _row(events="field_out"),
        _row(events="field_out"), _row(events="field_out"),
    ])
    # 5 in-zone pitches: 2 swings, 1 whiff. zone_contact_pct = 1/2 = 0.5
    # 5 OOZ pitches: 2 swings. chase_rate = 2/5 = 0.4
    pitches = pd.DataFrame([
        _row(description="hit_into_play", zone=4),  # in-zone, contact
        _row(description="swinging_strike", zone=5),  # in-zone, whiff
        _row(description="called_strike", zone=6),    # in-zone, no swing
        _row(description="ball", zone=4),             # in-zone, no swing
        _row(description="ball", zone=4),             # in-zone, no swing
        _row(description="ball", zone=11),            # OOZ, no swing
        _row(description="ball", zone=12),            # OOZ, no swing
        _row(description="ball", zone=13),            # OOZ, no swing
        _row(description="swinging_strike", zone=14), # OOZ, swing (chase, whiff)
        _row(description="foul", zone=11),            # OOZ, swing (chase, contact)
    ])
    out = _compute_split(pa_rows, pitches)
    assert out["n_pa"] == 10
    assert abs(out["k_pct"] - 0.30) < 0.001
    assert abs(out["obp"] - 0.30) < 0.001
    assert abs(out["zone_contact_pct"] - 0.5) < 0.001
    assert abs(out["chase_rate"] - 0.4) < 0.001


def test_compute_split_obp_excludes_sac_bunt():
    """OBP denom = PA - SH. Sac bunts shouldn't penalize OBP."""
    pa_rows = pd.DataFrame([
        _row(events="single"),
        _row(events="field_out"),
        _row(events="sac_bunt"),
    ])
    # Empty pitches df still needs the expected schema for the zone/desc filters.
    pitches = pd.DataFrame(columns=["zone", "description"])
    out = _compute_split(pa_rows, pitches)
    # PA = 3, SH = 1 -> denom = 2. OBP = 1/2 = 0.5.
    assert abs(out["obp"] - 0.5) < 0.001


# ---- _sanity_check ---------------------------------------------------------


def _sample_payload(*, season=2024, overall_k=0.225, rr_k=0.235, lr_k=0.225,
                    common_n=50000, rarer_n=20000):
    """Build a minimal payload with the shape _sanity_check expects."""
    def split(n_pa, k_pct):
        return {
            "k_pct": k_pct, "obp": 0.31, "zone_contact_pct": 0.85,
            "chase_rate": 0.29, "n_pa": n_pa,
        }
    return {
        "season": season,
        "all": split(common_n + rarer_n, overall_k),
        "splits": {
            "RR": split(common_n, rr_k),
            "LR": split(common_n, lr_k),
            "RL": split(rarer_n, 0.22),
            "LL": split(rarer_n, 0.24),
        },
    }


def test_sanity_check_passes_on_realistic_payload():
    _sanity_check(_sample_payload())  # no raise


def test_sanity_check_halts_on_out_of_range_k_pct():
    bad = _sample_payload(overall_k=0.30)
    with pytest.raises(AssertionError, match="outside expected range"):
        _sanity_check(bad)


def test_sanity_check_halts_when_rr_le_lr():
    """Same-hand matchups MUST produce more Ks than opposite-hand."""
    bad = _sample_payload(rr_k=0.21, lr_k=0.23)
    with pytest.raises(AssertionError, match="not greater than LR"):
        _sanity_check(bad)


def test_sanity_check_halts_on_thin_common_split():
    bad = _sample_payload(common_n=20_000)  # below 30k threshold
    with pytest.raises(AssertionError, match=r"split RR"):
        _sanity_check(bad)


def test_sanity_check_halts_on_thin_rarer_split():
    bad = _sample_payload(rarer_n=8_000)  # below 12k threshold
    with pytest.raises(AssertionError, match=r"split (RL|LL)"):
        _sanity_check(bad)
