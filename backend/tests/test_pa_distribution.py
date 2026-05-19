"""Phase 3-v2c-0: tests for PA distribution derivation."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.derive_pa_distribution import (
    BF_MAX,
    BF_MIN,
    MIN_SMOOTHING_THRESHOLD,
    SLOTS,
    TTO_BUCKETS,
    _aggregate_cell_pa,
    _analytical_distribution,
    _attach_slot_and_drop_pinch_hits,
    _build_distributions,
    _per_game_bf,
    _sanity_check,
    _smoothed_at_bf,
    lookup_distribution,
)
from scripts.derive_tto_multipliers import _assign_tto


def _pa_row(
    *, game_pk: int, pitcher: int, batter: int, inning: int,
    at_bat_number: int, events: str = "field_out", season: int = 2025,
) -> dict:
    return {
        "game_pk": game_pk, "pitcher": pitcher, "batter": batter,
        "inning": inning, "at_bat_number": at_bat_number,
        "events": events, "__season": season,
    }


def _synthetic_sp_game(
    game_pk: int, pitcher: int, n_cycles: int, partial_after: int = 0,
) -> pd.DataFrame:
    """Build a synthetic SP game where pitcher faces batters 1-9 cyclically.

    n_cycles full times through the order, plus `partial_after` extra batters
    in the next cycle (slots 1..partial_after).
    """
    rows = []
    abn = 1
    inning = 1
    for cycle in range(n_cycles):
        for slot in range(1, 10):
            rows.append(_pa_row(
                game_pk=game_pk, pitcher=pitcher, batter=slot,
                inning=cycle * 3 + (slot - 1) // 3 + 1,
                at_bat_number=abn,
            ))
            abn += 1
    for slot in range(1, partial_after + 1):
        rows.append(_pa_row(
            game_pk=game_pk, pitcher=pitcher, batter=slot,
            inning=n_cycles * 3 + (slot - 1) // 3 + 1,
            at_bat_number=abn,
        ))
        abn += 1
    return pd.DataFrame(rows)


# ---- _attach_slot_and_drop_pinch_hits --------------------------------------


def test_slot_assignment_first_9_distinct_get_slots_1_to_9():
    df = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=1)
    df = _assign_tto(df)
    out = _attach_slot_and_drop_pinch_hits(df)
    assert sorted(out["slot"].unique()) == list(range(1, 10))


def test_pinch_hit_game_is_dropped_entirely():
    """If batter 10 (slot > 9) appears, the whole game is dropped."""
    df = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=2)
    pinch_row = pd.DataFrame([_pa_row(
        game_pk=1, pitcher=100, batter=10, inning=7, at_bat_number=19,
    )])
    df = pd.concat([df, pinch_row], ignore_index=True)
    df = _assign_tto(df)
    out = _attach_slot_and_drop_pinch_hits(df)
    assert len(out) == 0


def test_slot_is_per_game_not_per_pitcher_across_games():
    """Same batter id in different games keeps slot 1 in each."""
    df_a = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=2)
    df_b = _synthetic_sp_game(game_pk=2, pitcher=100, n_cycles=2)
    df = pd.concat([df_a, df_b], ignore_index=True)
    df = _assign_tto(df)
    out = _attach_slot_and_drop_pinch_hits(df)
    # Batter 1 should be slot 1 in both games
    assert (out.loc[out["batter"] == 1, "slot"] == 1).all()


# ---- _per_game_bf ----------------------------------------------------------


def test_per_game_bf_counts_pas():
    df = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=2, partial_after=6)
    df = _assign_tto(df)
    df = _attach_slot_and_drop_pinch_hits(df)
    bf = _per_game_bf(df)
    assert bf.iloc[0]["total_bf"] == 24  # 2 full cycles + 6


# ---- Synthetic 9-batter game produces tto_1=1.0 ----------------------------


def test_synthetic_9_pa_game_yields_tto_1_one_per_slot():
    """1 cycle = 9 PAs, all slots have tto_1=1.0 and zero elsewhere."""
    df = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=1)
    # Need to satisfy SP filter (9 distinct batters, 12 outs, min inning 1).
    # 1 cycle only gives 9 outs — boost to 12 by adding 3 more PAs from the
    # cycle continuation. Or: bypass SP filter and just test the math.
    df = _assign_tto(df)
    # Skip SP filter for unit test; directly attach slot + agg
    df = _attach_slot_and_drop_pinch_hits(df)
    bf = _per_game_bf(df)
    cells = _aggregate_cell_pa(df, bf)
    # BF=9, every slot should have tto_1 PA count = 1
    bf9 = cells[cells["total_bf"] == 9]
    for slot in SLOTS:
        row = bf9[(bf9["slot"] == slot) & (bf9["tto"] == 1)]
        assert int(row.iloc[0]["pa_count"]) == 1


def test_synthetic_27_pa_game_yields_full_tto_3_coverage():
    """3 cycles = 27 PAs, every slot should have tto_1=tto_2=tto_3=1.0."""
    df = _synthetic_sp_game(game_pk=1, pitcher=100, n_cycles=3)
    df = _assign_tto(df)
    df = _attach_slot_and_drop_pinch_hits(df)
    bf = _per_game_bf(df)
    cells = _aggregate_cell_pa(df, bf)
    bf27 = cells[cells["total_bf"] == 27]
    for slot in SLOTS:
        for t in (1, 2, 3):
            row = bf27[(bf27["slot"] == slot) & (bf27["tto"] == t)]
            assert int(row.iloc[0]["pa_count"]) == 1, (
                f"slot {slot} tto {t} missing in 27-PA game"
            )


# ---- Smoothing -------------------------------------------------------------


def test_smoothing_applied_for_low_n_bf():
    """If BF=15 has only 20 games (below threshold 50), smoothing kicks in."""
    rows = []
    # 20 games at BF=15 (insufficient on its own)
    for g in range(20):
        rows.extend(_synthetic_sp_game(
            game_pk=1000 + g, pitcher=100, n_cycles=1, partial_after=6,
        ).to_dict("records"))
    # 60 games at BF=14 (sufficient) for the smoothing window to pull from
    for g in range(60):
        rows.extend(_synthetic_sp_game(
            game_pk=2000 + g, pitcher=100, n_cycles=1, partial_after=5,
        ).to_dict("records"))
    # 50 games at BF=16
    for g in range(50):
        rows.extend(_synthetic_sp_game(
            game_pk=3000 + g, pitcher=100, n_cycles=1, partial_after=7,
        ).to_dict("records"))
    df = pd.DataFrame(rows)
    df = _assign_tto(df)
    df = _attach_slot_and_drop_pinch_hits(df)
    bf = _per_game_bf(df)
    cells = _aggregate_cell_pa(df, bf)
    cells = cells[(cells["total_bf"] >= 13) & (cells["total_bf"] <= 17)]

    slot_dict, n_raw, n_effective, was_smoothed = _smoothed_at_bf(cells, 15)
    assert was_smoothed is True
    assert n_raw == 20  # raw at BF=15
    # Smoothed value uses BF=14,15,16 → 60+20+50 = 130 games
    assert n_effective == 130


def test_smoothing_skipped_for_high_n_bf():
    rows = []
    for g in range(100):
        rows.extend(_synthetic_sp_game(
            game_pk=1000 + g, pitcher=100, n_cycles=2, partial_after=5,
        ).to_dict("records"))
    df = pd.DataFrame(rows)
    df = _assign_tto(df)
    df = _attach_slot_and_drop_pinch_hits(df)
    bf = _per_game_bf(df)
    cells = _aggregate_cell_pa(df, bf)
    slot_dict, n_raw, n_effective, was_smoothed = _smoothed_at_bf(cells, 23)
    assert was_smoothed is False
    assert n_raw == 100
    assert n_effective == 100


# ---- _analytical_distribution ----------------------------------------------


def test_analytical_at_bf_27_is_perfect_3_pa_each():
    d = _analytical_distribution(27)
    for slot in SLOTS:
        assert d[str(slot)]["tto_1"] == 1.0
        assert d[str(slot)]["tto_2"] == 1.0
        assert d[str(slot)]["tto_3"] == 1.0
        assert d[str(slot)]["tto_4"] == 0.0
        assert d[str(slot)]["total_pa"] == 3.0


def test_analytical_at_bf_24_is_2pt67_partial_tto_3():
    d = _analytical_distribution(24)
    for slot in SLOTS:
        assert d[str(slot)]["tto_1"] == 1.0
        assert d[str(slot)]["tto_2"] == 1.0
        assert d[str(slot)]["tto_3"] == round(24/9 - 2, 4)
        assert d[str(slot)]["tto_4"] == 0.0


# ---- _sanity_check ---------------------------------------------------------


def _well_formed_distribution(bf: int, n_games: int = 1000) -> dict:
    """A synthetic well-formed distribution matching analytical (which passes
    all sanity gates)."""
    analytical = _analytical_distribution(bf)
    return {"n_games": n_games, "smoothed": False, "by_slot": analytical}


def _full_payload(distributions: dict) -> dict:
    return {
        "distributions": distributions,
        "summary": {"bf_range_observed": [BF_MIN, BF_MAX]},
    }


def test_sanity_passes_when_all_bf_well_formed():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)
    }
    _sanity_check(_full_payload(distributions))


def test_sanity_halts_when_bf_missing():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX)
    }
    # Missing BF_MAX
    with pytest.raises(AssertionError, match="missing BF"):
        _sanity_check(_full_payload(distributions))


def test_sanity_halts_when_slot_sum_neq_bf():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)
    }
    # Break BF=24: zero out slot 9
    distributions["24"]["by_slot"]["9"] = {
        f"tto_{t}": 0.0 for t in TTO_BUCKETS
    } | {"total_pa": 0.0}
    with pytest.raises(AssertionError, match="slot-sum"):
        _sanity_check(_full_payload(distributions))


def test_sanity_halts_when_tto_1_not_monotonic():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)
    }
    # Break BF=24: bump slot 5 tto_1 to 1.5
    distributions["24"]["by_slot"]["5"]["tto_1"] = 1.5
    with pytest.raises(AssertionError, match="exceeds 1.0|monotonic"):
        _sanity_check(_full_payload(distributions))


def test_sanity_halts_when_cell_exceeds_one():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)
    }
    distributions["24"]["by_slot"]["1"]["tto_2"] = 1.5
    with pytest.raises(AssertionError, match="exceeds 1.0"):
        _sanity_check(_full_payload(distributions))


def test_sanity_halts_when_bf_lt_27_slot9_tto3_too_high():
    distributions = {
        str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)
    }
    # BF=20: slot 1 has tto_3=0 in analytical; bump slot 9 tto_3 to 0.5
    distributions["20"]["by_slot"]["9"]["tto_3"] = 0.5
    with pytest.raises(AssertionError, match="order shouldn't fully turn over"):
        _sanity_check(_full_payload(distributions))


# ---- lookup_distribution ---------------------------------------------------


def test_lookup_rounds_to_nearest_integer():
    distributions = {str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)}
    # 25.6 → 26
    result = lookup_distribution(distributions, 25.6)
    assert result is distributions["26"]


def test_lookup_clamps_to_range():
    distributions = {str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)}
    # 5 → BF_MIN
    assert lookup_distribution(distributions, 5) is distributions[str(BF_MIN)]
    # 100 → BF_MAX
    assert lookup_distribution(distributions, 100) is distributions[str(BF_MAX)]


def test_lookup_half_rounds_to_even_or_up():
    """Standard Python rounding (banker's rounding): 25.5 -> 26."""
    distributions = {str(b): _well_formed_distribution(b) for b in range(BF_MIN, BF_MAX + 1)}
    # round(25.5) = 26 (banker's rounds half to even)
    result = lookup_distribution(distributions, 25.5)
    assert result is distributions["26"]


# ---- Output schema ---------------------------------------------------------


def test_bf_constants_are_12_to_35():
    assert BF_MIN == 12
    assert BF_MAX == 35


def test_smoothing_threshold_is_50():
    assert MIN_SMOOTHING_THRESHOLD == 50


def test_tto_buckets_constant_matches_phase_3v2b():
    assert TTO_BUCKETS == (1, 2, 3, 4)
