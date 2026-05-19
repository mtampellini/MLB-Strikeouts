"""Phase 3-v2b: tests for TTO multiplier derivation."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.derive_tto_multipliers import (
    ARCHETYPES,
    K_EVENTS,
    LEAGUE_TTO3_RANGE,
    MAX_TTO,
    SHRINKAGE_PRIOR_PA,
    TTO_BUCKETS,
    _aggregate_cells,
    _assign_tto,
    _attach_archetype,
    _build_archetype_multipliers,
    _build_league_wide,
    _filter_sp_pas,
    _sanity_check,
    _shrunk_multiplier,
)


def _pa_row(
    *, game_pk: int, pitcher: int, batter: int, inning: int,
    at_bat_number: int, events: str = "field_out", season: int = 2025,
) -> dict:
    return {
        "game_pk": game_pk, "pitcher": pitcher, "batter": batter,
        "inning": inning, "at_bat_number": at_bat_number,
        "events": events, "__season": season,
    }


# ---- _assign_tto -----------------------------------------------------------


def test_assign_tto_first_9_distinct_batters_all_tto_1():
    """9 distinct batters in one game produce TTO=1 for all of them."""
    rows = [_pa_row(game_pk=1, pitcher=100, batter=b, inning=(b - 1) // 3 + 1,
                    at_bat_number=b)
            for b in range(1, 10)]
    df = pd.DataFrame(rows)
    out = _assign_tto(df)
    assert (out["tto"] == 1).all()


def test_assign_tto_second_pass_is_tto_2():
    """18-PA game with same 9 batters facing same pitcher twice produces
    TTO=1 for the first 9 PAs and TTO=2 for the next 9."""
    rows = []
    abn = 1
    for cycle in range(2):
        for b in range(1, 10):
            rows.append(_pa_row(game_pk=1, pitcher=100, batter=b,
                                inning=cycle * 3 + (b - 1) // 3 + 1,
                                at_bat_number=abn))
            abn += 1
    df = pd.DataFrame(rows)
    out = _assign_tto(df).sort_values("at_bat_number")
    assert list(out["tto"]) == [1] * 9 + [2] * 9


def test_assign_tto_caps_at_max():
    """A batter appearing 6 times against same pitcher caps at MAX_TTO=4."""
    rows = [_pa_row(game_pk=1, pitcher=100, batter=1, inning=i + 1,
                    at_bat_number=i + 1)
            for i in range(6)]
    df = pd.DataFrame(rows)
    out = _assign_tto(df).sort_values("at_bat_number")
    assert list(out["tto"]) == [1, 2, 3, 4, 4, 4]


def test_assign_tto_pitcher_change_does_not_carry_over():
    """If batter B faces pitcher A then later pitcher B, the count for
    pitcher B resets — TTO is per-pitcher."""
    rows = [
        _pa_row(game_pk=1, pitcher=100, batter=1, inning=1, at_bat_number=1),
        _pa_row(game_pk=1, pitcher=100, batter=1, inning=4, at_bat_number=10),
        _pa_row(game_pk=1, pitcher=200, batter=1, inning=7, at_bat_number=20),
    ]
    df = pd.DataFrame(rows)
    out = _assign_tto(df).sort_values("at_bat_number")
    # Pitcher 100 sees batter 1 twice: TTO=1, TTO=2
    # Pitcher 200 sees batter 1 once: TTO=1
    assert list(out.loc[out["pitcher"] == 100, "tto"]) == [1, 2]
    assert list(out.loc[out["pitcher"] == 200, "tto"]) == [1]


def test_assign_tto_across_different_games_resets():
    """Same pitcher, same batter, different game_pk → TTO restarts at 1."""
    rows = [
        _pa_row(game_pk=1, pitcher=100, batter=1, inning=1, at_bat_number=1),
        _pa_row(game_pk=1, pitcher=100, batter=1, inning=4, at_bat_number=10),
        _pa_row(game_pk=2, pitcher=100, batter=1, inning=1, at_bat_number=1),
    ]
    df = pd.DataFrame(rows)
    out = _assign_tto(df).sort_values(["game_pk", "at_bat_number"])
    assert list(out["tto"]) == [1, 2, 1]


# ---- _filter_sp_pas --------------------------------------------------------


def _sp_game_rows(game_pk: int, pitcher: int) -> list[dict]:
    """Construct a synthetic SP game: pitcher faces 12 distinct batters,
    records 12 outs across 4 innings (inning 1..4), all field_out events."""
    rows = []
    abn = 1
    for inn in range(1, 5):
        for slot in range(1, 4):  # 3 batters per inning, 1-9 cycling
            batter = ((inn - 1) * 3 + slot - 1) % 9 + 1  # batters 1..9
            rows.append(_pa_row(
                game_pk=game_pk, pitcher=pitcher, batter=batter,
                inning=inn, at_bat_number=abn,
                events="field_out",
            ))
            abn += 1
    # Last 3 rows already give 12 distinct? No — 4 innings * 3 batters = 12 PAs,
    # but batters cycle so it's 9 distinct. That satisfies SP_MIN_DISTINCT_BATTERS.
    return rows


def test_filter_sp_pas_keeps_valid_starter():
    rows = _sp_game_rows(game_pk=1, pitcher=100)
    df = pd.DataFrame(rows)
    out = _filter_sp_pas(df)
    assert len(out) == len(df)


def test_filter_sp_pas_drops_reliever_who_did_not_start_inning_1():
    """A pitcher whose first PA is in inning 6 is not the starter."""
    rows = [_pa_row(game_pk=1, pitcher=200, batter=b, inning=6,
                    at_bat_number=20 + b, events="field_out")
            for b in range(1, 13)]
    df = pd.DataFrame(rows)
    out = _filter_sp_pas(df)
    assert len(out) == 0


def test_filter_sp_pas_drops_short_outing():
    """A pitcher who got pulled after 2 batters (no 9 distinct) doesn't count."""
    rows = [_pa_row(game_pk=1, pitcher=100, batter=b, inning=1,
                    at_bat_number=b, events="field_out")
            for b in range(1, 3)]
    df = pd.DataFrame(rows)
    out = _filter_sp_pas(df)
    assert len(out) == 0


# ---- _attach_archetype -----------------------------------------------------


def test_attach_archetype_filters_unknown_pitchers():
    rows = [
        _pa_row(game_pk=1, pitcher=100, batter=1, inning=1, at_bat_number=1),  # known
        _pa_row(game_pk=1, pitcher=999, batter=1, inning=1, at_bat_number=2),  # unknown
    ]
    df = pd.DataFrame(rows)
    arch_map = {100: {2025: "Power-FF"}}
    out, n_skipped = _attach_archetype(df, arch_map)
    assert len(out) == 1
    assert n_skipped == 1
    assert out.iloc[0]["archetype"] == "Power-FF"


def test_attach_archetype_filters_pitcher_with_missing_season():
    """Pitcher classified in 2024 but PA is 2025 — skip."""
    rows = [_pa_row(game_pk=1, pitcher=100, batter=1, inning=1,
                    at_bat_number=1, season=2025)]
    df = pd.DataFrame(rows)
    arch_map = {100: {2024: "Power-FF"}}
    out, n_skipped = _attach_archetype(df, arch_map)
    assert len(out) == 0
    assert n_skipped == 1


# ---- _shrunk_multiplier ----------------------------------------------------


def test_shrunk_multiplier_zero_observations_returns_1():
    """With 0 observations, posterior == baseline → multiplier = 1.00."""
    assert _shrunk_multiplier(0, 0, baseline_rate=0.22, prior_pa=500) == pytest.approx(1.0)


def test_shrunk_multiplier_large_sample_approaches_observed():
    """With many observations, posterior approaches observed rate."""
    # 100,000 PAs at rate 0.18 vs baseline 0.22 → multiplier should be near 0.18/0.22
    raw_ratio = 0.18 / 0.22
    mult = _shrunk_multiplier(18000, 100000, baseline_rate=0.22, prior_pa=500)
    assert abs(mult - raw_ratio) < 0.005


def test_shrunk_multiplier_small_sample_pulled_toward_1():
    """With 100 observations at rate 0.10 vs baseline 0.22, shrinkage pulls toward 1.00."""
    raw_ratio = 0.10 / 0.22  # ~0.455
    mult = _shrunk_multiplier(10, 100, baseline_rate=0.22, prior_pa=500)
    assert mult > raw_ratio  # pulled UP toward 1.00
    assert mult < 1.0        # but not all the way


# ---- _aggregate_cells ------------------------------------------------------


def test_aggregate_cells_counts_k_correctly():
    rows = []
    # Power-FF, TTO=1: 8 PAs, 2 K
    for i in range(8):
        rows.append({"archetype": "Power-FF", "tto": 1,
                     "events": "strikeout" if i < 2 else "field_out"})
    # Balanced, TTO=1: 5 PAs, 1 K
    for i in range(5):
        rows.append({"archetype": "Balanced", "tto": 1,
                     "events": "strikeout" if i < 1 else "field_out"})
    df = pd.DataFrame(rows)
    out = _aggregate_cells(df)
    pwr = out[(out["archetype"] == "Power-FF") & (out["tto"] == 1)].iloc[0]
    assert pwr["k_count"] == 2
    assert pwr["pa_count"] == 8
    assert pwr["k_rate"] == pytest.approx(0.25)


# ---- _build_archetype_multipliers ------------------------------------------


def test_low_sample_cell_flagged_and_neutralized():
    """Cell with fewer than 200 PAs gets multiplier=1.00 and low_sample=True."""
    rows = []
    # Power-FF TTO=1: 5000 PAs, 1100 K → baseline 0.22
    rows.append({"archetype": "Power-FF", "tto": 1, "k_count": 1100,
                 "pa_count": 5000, "k_rate": 0.22})
    # Power-FF TTO=2: only 150 PAs (below 200)
    rows.append({"archetype": "Power-FF", "tto": 2, "k_count": 25,
                 "pa_count": 150, "k_rate": 0.1667})
    # Power-FF TTO=3,4: 1000 PAs each, K rate dropping
    rows.append({"archetype": "Power-FF", "tto": 3, "k_count": 200,
                 "pa_count": 1000, "k_rate": 0.20})
    rows.append({"archetype": "Power-FF", "tto": 4, "k_count": 180,
                 "pa_count": 1000, "k_rate": 0.18})
    # Fill in other archetypes with full data so we can build at all
    for arch in ARCHETYPES:
        if arch == "Power-FF":
            continue
        for t in TTO_BUCKETS:
            rows.append({"archetype": arch, "tto": t, "k_count": 220,
                         "pa_count": 1000, "k_rate": 0.22})
    cells = pd.DataFrame(rows)
    out, low_sample = _build_archetype_multipliers(cells)
    assert out["Power-FF"]["tto_2"]["multiplier"] == 1.0
    assert out["Power-FF"]["tto_2"]["low_sample"] is True
    assert any(c["archetype"] == "Power-FF" and c["tto"] == 2 for c in low_sample)
    # Adequate-sample cell is NOT flagged
    assert out["Power-FF"]["tto_3"]["low_sample"] is False


# ---- _sanity_check ---------------------------------------------------------


def _payload_with_archetypes(arch_mults: dict[str, dict[int, float]],
                              league_mults: dict[int, float]) -> dict:
    by_archetype = {}
    for arch in ARCHETYPES:
        block = {}
        for t in TTO_BUCKETS:
            mult = arch_mults.get(arch, {}).get(t, 1.0 if t == 1 else 0.92)
            block[f"tto_{t}"] = {"multiplier": mult, "k_rate": 0.22,
                                 "n_pa": 1000, "low_sample": False}
        by_archetype[arch] = block
    league_wide = {}
    for t in TTO_BUCKETS:
        league_wide[f"tto_{t}"] = {"multiplier": league_mults[t],
                                    "k_rate": 0.22, "n_pa": 5000}
    return {
        "by_archetype": by_archetype, "league_wide": league_wide,
        "summary": {}, "seasons_used": [2024, 2025],
    }


def test_sanity_passes_realistic():
    payload = _payload_with_archetypes(
        arch_mults={a: {1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84} for a in ARCHETYPES},
        league_mults={1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84},
    )
    _sanity_check(payload)


def test_sanity_halts_when_tto_1_not_one():
    payload = _payload_with_archetypes(
        arch_mults={"Power-FF": {1: 0.99, 2: 0.96, 3: 0.90, 4: 0.84}},
        league_mults={1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84},
    )
    with pytest.raises(AssertionError, match="tto_1 multiplier"):
        _sanity_check(payload)


def test_sanity_halts_on_non_monotonic_league():
    """If TTO=2 > TTO=3 fails (e.g. TTO=3 mult higher than TTO=2)."""
    payload = _payload_with_archetypes(
        arch_mults={a: {1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84} for a in ARCHETYPES},
        league_mults={1: 1.0, 2: 0.92, 3: 0.95, 4: 0.85},  # tto_3 > tto_2
    )
    with pytest.raises(AssertionError, match="not monotonically decreasing"):
        _sanity_check(payload)


def test_sanity_halts_on_max_multiplier_exceeded():
    payload = _payload_with_archetypes(
        arch_mults={"Power-FF": {1: 1.0, 2: 1.10, 3: 0.90, 4: 0.84}},
        league_mults={1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84},
    )
    with pytest.raises(AssertionError, match="exceeds max"):
        _sanity_check(payload)


def test_sanity_halts_on_min_multiplier_breached():
    payload = _payload_with_archetypes(
        arch_mults={"Power-FF": {1: 1.0, 2: 0.96, 3: 0.60, 4: 0.84}},
        league_mults={1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84},
    )
    with pytest.raises(AssertionError, match="below min"):
        _sanity_check(payload)


def test_sanity_halts_on_league_tto3_out_of_research_range():
    """League TTO=3 multiplier outside [0.85, 0.95] halts."""
    payload = _payload_with_archetypes(
        arch_mults={a: {1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84} for a in ARCHETYPES},
        league_mults={1: 1.0, 2: 0.96, 3: 0.97, 4: 0.84},  # tto_3 too high
    )
    # tto_3=0.97 > tto_2=0.96 also fails monotonicity, which fires first
    with pytest.raises(AssertionError):
        _sanity_check(payload)


def test_sanity_halts_when_archetype_missing():
    payload = _payload_with_archetypes(
        arch_mults={a: {1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84} for a in ARCHETYPES},
        league_mults={1: 1.0, 2: 0.96, 3: 0.90, 4: 0.84},
    )
    del payload["by_archetype"]["Balanced"]
    with pytest.raises(AssertionError, match="missing archetypes"):
        _sanity_check(payload)


# ---- Output schema ---------------------------------------------------------


def test_archetypes_constant_matches_phase_3v2a():
    assert set(ARCHETYPES) == {
        "Power-FF", "Sinker-ball", "Breaking-heavy",
        "Offspeed-heavy", "Balanced",
    }


def test_tto_buckets_constant_is_1_through_4():
    assert TTO_BUCKETS == (1, 2, 3, 4)
    assert MAX_TTO == 4


def test_k_events_includes_both_k_outcomes():
    assert "strikeout" in K_EVENTS
    assert "strikeout_double_play" in K_EVENTS


def test_league_tto3_range_is_k_specific_not_wOBA():
    """K-rate TTO research clusters TTO=3 in [0.80, 0.88]; the gate's
    [0.78, 0.92] envelopes that. Was previously [0.85, 0.95] (wOBA-based),
    which would not pass the empirical 2023-2025 K-rate data."""
    lo, hi = LEAGUE_TTO3_RANGE
    assert lo == 0.78
    assert hi == 0.92
    # Cross-check: 0.85 (the old gate floor) is well inside the new range;
    # 0.81 (observed empirical value) is also inside.
    assert lo < 0.81 < hi
    assert lo < 0.85 < hi
