"""Phase 3-v2a: tests for pitcher archetype classification."""
from __future__ import annotations

import pytest

from scripts.derive_pitcher_archetypes import (
    ARCHETYPES,
    FASTBALL_THRESHOLD,
    MAX_JUNK_PCT,
    MIN_PITCHES,
    MIN_STARTS,
    _classify,
    _mix_from_counts,
    _sanity_check,
    _skip_reason,
)


def _mix(*, fb=0.50, ff=0.30, si=0.15, fc=0.05, br=0.30, os=0.20,
         junk_pct=0.0, n_pitches_all=2000) -> dict:
    return {
        "fastball_pct": fb, "four_seam_pct": ff, "sinker_pct": si,
        "cutter_pct": fc, "breaking_pct": br, "offspeed_pct": os,
        "total_pitches": int(n_pitches_all * (1 - junk_pct)),
        "junk_pct": junk_pct, "n_pitches_all": n_pitches_all,
    }


# ---- _classify (rule order) ------------------------------------------------


def test_power_ff_rule():
    """fastball >= 55% AND four_seam >= 35% → Power-FF."""
    assert _classify(_mix(fb=0.60, ff=0.45, si=0.10, fc=0.05, br=0.30, os=0.10)) == "Power-FF"


def test_sinker_ball_rule():
    """fastball >= 55% AND sinker >= 25% → Sinker-ball."""
    assert _classify(_mix(fb=0.60, ff=0.25, si=0.30, fc=0.05, br=0.25, os=0.15)) == "Sinker-ball"


def test_sinker_ball_wins_when_ff_below_threshold():
    """Edge case from spec: FF=30, SI=30, fastball=60 → Sinker-ball wins
    because FF threshold (35) isn't hit but SI threshold (25) is."""
    assert _classify(_mix(fb=0.60, ff=0.30, si=0.30, fc=0.0, br=0.30, os=0.10)) == "Sinker-ball"


def test_power_ff_wins_when_both_ff_and_si_hit_thresholds():
    """First match wins. Power-FF rule comes before Sinker-ball rule."""
    assert _classify(_mix(fb=0.70, ff=0.40, si=0.25, fc=0.05, br=0.20, os=0.10)) == "Power-FF"


def test_breaking_heavy():
    """40% breaking, 50% fastball, 10% offspeed → Breaking-heavy."""
    assert _classify(_mix(fb=0.50, ff=0.30, si=0.15, fc=0.05, br=0.40, os=0.10)) == "Breaking-heavy"


def test_offspeed_heavy():
    """28% changeup-or-splitter, 55% fastball (split FF/SI/FC below thresholds),
    17% breaking → Offspeed-heavy."""
    assert _classify(_mix(fb=0.55, ff=0.20, si=0.20, fc=0.15, br=0.17, os=0.28)) == "Offspeed-heavy"


def test_balanced_when_no_thresholds_met():
    """50% fastball (split FF/SI/FC below thresholds), 30% breaking, 20%
    offspeed → Balanced."""
    assert _classify(_mix(fb=0.50, ff=0.25, si=0.15, fc=0.10, br=0.30, os=0.20)) == "Balanced"


def test_breaking_heavy_wins_over_offspeed_heavy_at_overlapping_thresholds():
    """Breaking rule comes before offspeed rule."""
    assert _classify(_mix(fb=0.30, ff=0.15, si=0.10, fc=0.05, br=0.40, os=0.30)) == "Breaking-heavy"


# ---- _mix_from_counts ------------------------------------------------------


def test_mix_from_counts_basic():
    counts = {"FF": 500, "SI": 200, "SL": 200, "CH": 100}  # 1000 total, no junk
    m = _mix_from_counts(counts)
    assert m["total_pitches"] == 1000
    assert m["fastball_pct"] == 0.7
    assert m["four_seam_pct"] == 0.5
    assert m["sinker_pct"] == 0.2
    assert m["breaking_pct"] == 0.2
    assert m["offspeed_pct"] == 0.1
    assert m["junk_pct"] == 0.0


def test_mix_from_counts_excludes_junk_from_denominator():
    """Junk pitches don't count toward fastball/breaking/offspeed percentages."""
    counts = {"FF": 600, "SL": 300, "KN": 100}  # 1000 total, 100 junk
    m = _mix_from_counts(counts)
    assert m["total_pitches"] == 900  # excludes junk
    assert m["fastball_pct"] == round(600 / 900, 4)
    assert m["breaking_pct"] == round(300 / 900, 4)
    assert m["junk_pct"] == 0.1
    assert m["n_pitches_all"] == 1000


def test_mix_from_counts_empty():
    m = _mix_from_counts({})
    assert m["total_pitches"] == 0
    assert m["fastball_pct"] == 0.0


# ---- _skip_reason ----------------------------------------------------------


def test_insufficient_starts_skip():
    mix = _mix(n_pitches_all=2000, junk_pct=0.0)
    assert _skip_reason(mix, n_starts=3) == "insufficient_starts"
    assert _skip_reason(mix, n_starts=MIN_STARTS) is None


def test_insufficient_pitches_skip():
    mix = _mix(n_pitches_all=400, junk_pct=0.0)
    assert _skip_reason(mix, n_starts=20) == "insufficient_pitches"
    mix2 = _mix(n_pitches_all=MIN_PITCHES, junk_pct=0.0)
    assert _skip_reason(mix2, n_starts=20) is None


def test_high_junk_skip():
    mix = _mix(n_pitches_all=2000, junk_pct=0.40)
    assert _skip_reason(mix, n_starts=20) == "high_junk"
    mix2 = _mix(n_pitches_all=2000, junk_pct=MAX_JUNK_PCT - 0.01)
    assert _skip_reason(mix2, n_starts=20) is None


def test_starts_filter_evaluated_first():
    """If both starts and pitches are insufficient, starts message is what
    callers see (consistency with the rule order in _skip_reason)."""
    mix = _mix(n_pitches_all=100, junk_pct=0.0)
    assert _skip_reason(mix, n_starts=2) == "insufficient_starts"


# ---- _sanity_check ---------------------------------------------------------


def _payload_with_summary(season_counts: dict[int, dict[str, int]],
                          per_pitcher_mixes: list[tuple[int, int, dict]] | None = None,
                          ) -> dict:
    archetypes: dict[str, dict] = {}
    if per_pitcher_mixes:
        for pid, season, mix in per_pitcher_mixes:
            entry = archetypes.setdefault(str(pid), {"by_season": {}, "season_rolling_30d": {}})
            entry["by_season"][str(season)] = {"archetype": "Balanced", **mix}
    return {
        "summary": {"by_season": {str(s): c for s, c in season_counts.items()}},
        "archetypes": archetypes,
        "seasons": list(season_counts.keys()),
    }


def _balanced_majority(extra: dict[str, int] | None = None) -> dict[str, int]:
    """Realistic distribution: 5 archetypes, Balanced largest at <50%."""
    base = {
        "Power-FF": 50, "Sinker-ball": 30, "Breaking-heavy": 25,
        "Offspeed-heavy": 15, "Balanced": 80, "Skipped": 50,
    }  # total classified = 200, Balanced = 40%
    if extra:
        base.update(extra)
    return base


def test_sanity_halt_on_empty_archetype():
    payload = _payload_with_summary({2024: _balanced_majority({"Offspeed-heavy": 0})})
    with pytest.raises(AssertionError, match="zero pitchers"):
        _sanity_check(payload)


def test_sanity_halt_category_over_50_pct():
    """A single category over 50% of classified pitchers should halt —
    suggests thresholds are too loose."""
    payload = _payload_with_summary({2024: {
        "Power-FF": 200, "Sinker-ball": 30, "Breaking-heavy": 20,
        "Offspeed-heavy": 10, "Balanced": 50, "Skipped": 30,
    }})  # Power-FF = 200/310 = 64.5%
    with pytest.raises(AssertionError, match="suggests thresholds are too loose"):
        _sanity_check(payload)


def test_sanity_halt_on_category_under_5_pct():
    """An archetype with <5% of classified universe should halt —
    suggests thresholds are too tight."""
    payload = _payload_with_summary({2024: {
        # Total classified = 300. Offspeed = 5, which is 1.7%.
        "Power-FF": 80, "Sinker-ball": 60, "Breaking-heavy": 80,
        "Offspeed-heavy": 5, "Balanced": 75, "Skipped": 30,
    }})
    with pytest.raises(AssertionError, match="suggests thresholds are too tight"):
        _sanity_check(payload)


def test_sanity_passes_realistic_distribution():
    """Empirically-grounded distribution: Breaking-heavy may be the modal
    bucket (modern MLB), every archetype >=5%, none >50%."""
    realistic_2024 = {
        "Power-FF": 62, "Sinker-ball": 39, "Breaking-heavy": 70,
        "Offspeed-heavy": 30, "Balanced": 48, "Skipped": 50,
    }
    payload = _payload_with_summary(
        {2024: realistic_2024, 2025: realistic_2024},
        per_pitcher_mixes=[
            # Sums to 1.00 — within [0.98, 1.02]
            (1, 2024, _mix(fb=0.55, br=0.30, os=0.15, n_pitches_all=2000)),
            (1, 2025, _mix(fb=0.55, br=0.30, os=0.15, n_pitches_all=2000)),
        ],
    )
    _sanity_check(payload)


def test_sanity_halt_on_pitcher_pct_sum_out_of_range():
    """A pitcher whose pct sum is implausibly low — flags a bug."""
    payload = _payload_with_summary(
        {2024: _balanced_majority()},
        per_pitcher_mixes=[
            (1, 2024, _mix(fb=0.20, br=0.15, os=0.10, n_pitches_all=2000)),  # sum = 0.45
        ],
    )
    with pytest.raises(AssertionError, match="pct sum"):
        _sanity_check(payload)


# ---- Output schema ---------------------------------------------------------


def test_archetype_constants_all_5_present():
    assert set(ARCHETYPES) == {
        "Power-FF", "Sinker-ball", "Breaking-heavy",
        "Offspeed-heavy", "Balanced",
    }
