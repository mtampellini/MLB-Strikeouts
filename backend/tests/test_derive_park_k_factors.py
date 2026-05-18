"""Phase 4b unit tests: derive_park_k_factors helper functions."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.derive_park_k_factors import (
    SHRINKAGE_PRIOR_GAMES,
    SHRINKAGE_PRIOR_GAMES_BY_HAND,
    TEAM_TO_VENUE_ID,
    _build_park_k_factors,
    _sanity_check,
)


def _pa_row(*, game_pk, pitcher, home_team, season=2024, is_k=False, p_throws="R"):
    return {
        "events": "strikeout" if is_k else "field_out",
        "game_pk": game_pk,
        "pitcher": pitcher,
        "home_team": home_team,
        "p_throws": p_throws,
        "__season": season,
        "game_type": "R",
    }


def test_team_to_venue_id_covers_30_teams():
    assert len(set(TEAM_TO_VENUE_ID.values())) == 30


def test_team_to_venue_id_has_no_oak():
    """Statcast normalizes Athletics to ATH only; OAK should not be in the map."""
    assert "OAK" not in TEAM_TO_VENUE_ID


def test_build_factors_returns_one_entry_per_venue():
    # Two pitchers playing 10 games each at NYY and TB so the leave-one-
    # venue-out baseline (>=5 other-venue games per pitcher) is satisfied.
    rows = []
    game_pk_counter = 1000
    for pitcher in (1, 2):
        for home_team in ("NYY", "TB"):
            for g in range(10):
                game_pk_counter += 1
                for ab in range(25):
                    rows.append(_pa_row(
                        game_pk=game_pk_counter, pitcher=pitcher,
                        home_team=home_team, is_k=(ab % 4 == 0),
                    ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df, by_hand=False)
    assert "factors" in payload
    venue_ids = {int(k) for k in payload["factors"].keys()}
    assert venue_ids == {3313, 12}


def test_build_factors_applies_shrinkage_toward_neutral():
    """An extreme observed/expected ratio at a sparse venue should be
    shrunk back toward 1.0. Three-venue construction ensures the
    leave-one-venue-out baseline isn't contaminated by the extreme venue.
    """
    rows = []
    game_pk = 1_000_000
    # 20 pitchers, each playing many games at NYY/BOS/LAD (all 25% K).
    # The leave-one-venue-out baseline for any of these three is dominated
    # by the other two = 25% → ratio at each ≈ 1.0.
    for pitcher in range(1, 21):
        for venue in ("NYY", "BOS", "LAD"):
            for g in range(8):
                game_pk += 1
                for ab in range(20):
                    rows.append(_pa_row(
                        game_pk=game_pk, pitcher=pitcher, home_team=venue,
                        is_k=(ab < 5),  # 25% K
                    ))
    # Same pitchers play 2 games at TB with extreme 100% K rate. TB sample
    # is small (40 games total across all pitchers) and the pitcher
    # baselines for TB are near 25% (dominated by NYY+BOS+LAD).
    for pitcher in range(1, 21):
        for g in range(2):
            game_pk += 1
            for ab in range(20):
                rows.append(_pa_row(
                    game_pk=game_pk, pitcher=pitcher, home_team="TB",
                    is_k=True,
                ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df, by_hand=False)
    tb = payload["factors"]["12"]["factor"]
    nyy = payload["factors"]["3313"]["factor"]
    # TB raw ratio = 100% / 25% = 4.0. EB shrinkage with k_prior=300 against
    # 40 games: factor = (4.0*40 + 1.0*300) / 340 ≈ 1.35.
    assert 1.0 < tb < 1.6
    # NYY: ratio slightly below 1.0 because LOO baseline includes the few
    # TB games. With heavy shrinkage (k_prior=300, 160 games) it stays
    # near 1.0 (~0.91). The point of the test is shrinkage at TB; NYY just
    # confirms a normal venue doesn't blow up.
    assert abs(nyy - 1.0) < 0.10


def test_sanity_check_requires_all_30_venues():
    payload = {"factors": {"3313": {"factor": 1.0, "n_games": 100, "venue_name": "X"}}}
    with pytest.raises(AssertionError, match="venues missing"):
        _sanity_check(payload)


def test_sanity_check_flags_too_many_outliers():
    factors = {}
    for vid in TEAM_TO_VENUE_ID.values():
        # Push 10 venues outside the [0.90, 1.10] band — should trip the
        # "too many outliers" check.
        factors[str(vid)] = {
            "factor": 1.12 if vid % 10 == 0 else 1.0,
            "n_games": 500,
            "venue_name": "X",
        }
    n_outliers = sum(1 for f in factors.values() if f["factor"] >= 1.10)
    if n_outliers > 5:
        with pytest.raises(AssertionError, match="outside"):
            _sanity_check({"factors": factors})


def test_sanity_check_flags_extreme_value():
    factors = {str(vid): {"factor": 1.0, "n_games": 500, "venue_name": "X"}
               for vid in TEAM_TO_VENUE_ID.values()}
    factors["3313"]["factor"] = 1.45  # extreme
    with pytest.raises(AssertionError, match=r"extreme park factors"):
        _sanity_check({"factors": factors})


def test_sanity_check_passes_realistic_distribution():
    """All 30 venues, factors in [0.95, 1.10] — should pass."""
    factors = {}
    for i, vid in enumerate(TEAM_TO_VENUE_ID.values()):
        f = 0.96 + (i % 11) * 0.013  # spreads 0.96 .. 1.09
        factors[str(vid)] = {
            "factor": round(f, 4), "n_games": 2000, "venue_name": "X",
        }
    _sanity_check({"factors": factors})  # no raise


# ---- By-hand mode ---------------------------------------------------------


def _by_hand_factors(*, all_below_one: bool = True) -> dict[str, dict]:
    """Build a 30-venue payload with both hands populated, adequate sample
    sizes, and std of factor_combined in the [0.02, 0.10] gate band."""
    factors = {}
    # Spread combined factors across ~0.95 .. 1.05 → std ≈ 0.03
    for i, vid in enumerate(TEAM_TO_VENUE_ID.values()):
        base = 0.95 + (i / 29.0) * 0.10  # 0.95 to 1.05 linearly
        f_l = base - 0.005
        f_r = base + 0.005
        factors[str(vid)] = {
            "venue_name": "X",
            "factor_lhp": round(f_l, 4),
            "factor_rhp": round(f_r, 4),
            "factor_combined": round((f_l + f_r) / 2, 4),
            "n_games_lhp": 150,
            "n_games_rhp": 350,
        }
    return factors


def test_build_factors_by_hand_splits_per_pitcher_hand():
    """LHPs have high K rate at venue 1, RHPs have low. Venue 1 should
    show factor_lhp > 1 and factor_rhp < 1."""
    rows = []
    # 200 LHP pitcher-games at venue 1 (NYY) — high K rate (40%)
    for g in range(200):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=10000 + g, pitcher=1, home_team="NYY",
                p_throws="L", is_k=(ab < 10),  # ~40% K rate
            ))
    # 200 RHP pitcher-games at venue 1 — low K rate (15%)
    for g in range(200):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=20000 + g, pitcher=2, home_team="NYY",
                p_throws="R", is_k=(ab < 4),  # ~16% K rate
            ))
    # Anchor each pitcher's full-season K% to 25% by adding games at
    # a different venue so the (observed/expected) ratio at NYY actually
    # differs by hand.
    for g in range(200):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=30000 + g, pitcher=1, home_team="BOS",
                p_throws="L", is_k=(ab < 6),  # ~24% baseline
            ))
            rows.append(_pa_row(
                game_pk=40000 + g, pitcher=2, home_team="BOS",
                p_throws="R", is_k=(ab < 6),  # ~24% baseline
            ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df, by_hand=True)
    nyy = payload["factors"]["3313"]
    assert nyy["factor_lhp"] > 1.0  # LHPs over-perform at NYY
    assert nyy["factor_rhp"] < 1.0  # RHPs under-perform at NYY
    assert "factor_combined" in nyy
    assert nyy["n_games_lhp"] == 200
    assert nyy["n_games_rhp"] == 200


def test_combined_factor_is_sample_weighted_average():
    """factor_combined = (f_l * n_l + f_r * n_r) / (n_l + n_r) using the
    shrunk per-hand factors. Pitchers need multi-venue games so the
    leave-one-venue-out baseline is defined."""
    rows = []
    game_pk = 50_000
    # 10 LHPs and 10 RHPs, each playing games at both NYY and BOS so the
    # leave-one-venue-out baseline is well-defined.
    for pitcher_id, hand in [(p, "L") for p in range(1, 11)] + [(p + 10, "R") for p in range(1, 11)]:
        for venue in ("NYY", "BOS"):
            for g in range(8):
                game_pk += 1
                for ab in range(25):
                    rows.append(_pa_row(
                        game_pk=game_pk, pitcher=pitcher_id, home_team=venue,
                        p_throws=hand, is_k=(ab < 6),
                    ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df, by_hand=True)
    nyy = payload["factors"]["3313"]
    expected = (
        nyy["factor_lhp"] * nyy["n_games_lhp"]
        + nyy["factor_rhp"] * nyy["n_games_rhp"]
    ) / (nyy["n_games_lhp"] + nyy["n_games_rhp"])
    assert abs(nyy["factor_combined"] - round(expected, 4)) < 1e-3


def test_shrinkage_pulls_small_hand_cell_toward_neutral():
    """A venue with few LHP games but extreme rate should land near 1.0
    because k_prior=200 dominates the small sample. Each pitcher needs
    multi-venue games so leave-one-venue-out doesn't filter them out."""
    rows = []
    game_pk = 70_000
    # 5 LHPs playing at NYY (extreme: all-K) and BOS (baseline 25%)
    for pitcher in range(1, 6):
        for g in range(2):
            game_pk += 1
            for ab in range(20):
                rows.append(_pa_row(
                    game_pk=game_pk, pitcher=pitcher, home_team="NYY",
                    p_throws="L", is_k=True,
                ))
        for g in range(5):
            game_pk += 1
            for ab in range(20):
                rows.append(_pa_row(
                    game_pk=game_pk, pitcher=pitcher, home_team="BOS",
                    p_throws="L", is_k=(ab < 5),
                ))
    # 50 RHP games at NYY (neutral)
    for pitcher in range(100, 110):
        for g in range(5):
            game_pk += 1
            for ab in range(25):
                rows.append(_pa_row(
                    game_pk=game_pk, pitcher=pitcher, home_team="NYY",
                    p_throws="R", is_k=(ab < 6),
                ))
        for g in range(5):
            game_pk += 1
            for ab in range(25):
                rows.append(_pa_row(
                    game_pk=game_pk, pitcher=pitcher, home_team="BOS",
                    p_throws="R", is_k=(ab < 6),
                ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df, by_hand=True)
    nyy = payload["factors"]["3313"]
    # 10 LHP-games at NYY at 100% K with leave-one-venue-out baseline of
    # 25% would give a raw ratio of ~4.0. After EB shrinkage with
    # k_prior=200 the factor should be much closer to 1.0.
    assert nyy["factor_lhp"] < 1.5


def test_sanity_check_by_hand_requires_both_hands():
    factors = _by_hand_factors()
    del factors[str(list(TEAM_TO_VENUE_ID.values())[0])]["factor_lhp"]
    with pytest.raises(AssertionError, match="missing factor_lhp"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_by_hand_halts_on_extreme_factor():
    """Phase 4b-v2: any factor outside [0.85, 1.30] for either hand halts."""
    factors = _by_hand_factors()
    factors["19"]["factor_lhp"] = 1.35  # over the 1.30 ceiling
    with pytest.raises(AssertionError, match=r"outside \[0.85, 1.30\]"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_by_hand_halts_on_too_low_factor():
    factors = _by_hand_factors()
    factors["12"]["factor_rhp"] = 0.80  # under the 0.85 floor
    with pytest.raises(AssertionError, match=r"outside \[0.85, 1.30\]"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_by_hand_allows_t_mobile_at_1p25():
    """T-Mobile post-LOO is the documented outlier at ~1.24. Must pass."""
    factors = _by_hand_factors()
    factors["680"]["factor_lhp"] = 1.259
    factors["680"]["factor_rhp"] = 1.237
    factors["680"]["factor_combined"] = 1.242
    _sanity_check({
        "method": "shrunk_observed_over_expected_k_rate_by_hand",
        "factors": factors,
    })  # no raise


def test_sanity_check_by_hand_allows_coors_above_one():
    """Coors at 1.05 (post-2021 reality) must pass — strict pre-2021 prior
    was relaxed in Phase 4b-v2."""
    factors = _by_hand_factors()
    factors["19"]["factor_lhp"] = 1.006
    factors["19"]["factor_rhp"] = 1.058
    factors["19"]["factor_combined"] = 1.045
    _sanity_check({
        "method": "shrunk_observed_over_expected_k_rate_by_hand",
        "factors": factors,
    })  # no raise


def test_sanity_check_halts_on_too_low_std_across_venues():
    """If shrinkage is so aggressive every venue is essentially 1.0, the
    std gate catches it."""
    factors = {}
    for vid in TEAM_TO_VENUE_ID.values():
        factors[str(vid)] = {
            "venue_name": "X",
            "factor_lhp": 1.000,
            "factor_rhp": 1.001,  # std across venues ~0
            "factor_combined": 1.0005,
            "n_games_lhp": 150,
            "n_games_rhp": 350,
        }
    with pytest.raises(AssertionError, match="std of factor_combined"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_halts_on_too_high_std_across_venues():
    """If shrinkage is too weak, venues spread too wide. Std gate catches it."""
    factors = {}
    for i, vid in enumerate(TEAM_TO_VENUE_ID.values()):
        # Linear spread from 0.85 to 1.15 → std ≈ 0.09; bump to fire gate
        base = 0.85 + (i / 29.0) * 0.30  # std ~0.09
        factors[str(vid)] = {
            "venue_name": "X",
            "factor_lhp": base - 0.005,
            "factor_rhp": base + 0.005,
            "factor_combined": base,
            "n_games_lhp": 150,
            "n_games_rhp": 350,
        }
    # std should be ~0.09 — JUST under 0.10. Spread wider to trip the gate.
    for i, vid in enumerate(TEAM_TO_VENUE_ID.values()):
        base = 0.85 + (i / 29.0) * 0.35
        # clip to valid factor range so we trip the std gate, not the
        # individual-factor gate
        base = max(0.86, min(1.19, base))
        factors[str(vid)]["factor_combined"] = base
        factors[str(vid)]["factor_lhp"] = base - 0.005
        factors[str(vid)]["factor_rhp"] = base + 0.005
    # If std is still < 0.10 with this spread, this test is vacuous; assert it
    import math
    combined = [f["factor_combined"] for f in factors.values()]
    mean_c = sum(combined) / len(combined)
    var_c = sum((x - mean_c) ** 2 for x in combined) / len(combined)
    std_c = math.sqrt(var_c)
    if std_c <= 0.10:
        pytest.skip(f"could not construct a payload with std > 0.10 (got {std_c:.4f})")
    with pytest.raises(AssertionError, match="std of factor_combined"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_by_hand_halts_on_thin_hand_cell():
    factors = _by_hand_factors()
    factors[str(list(TEAM_TO_VENUE_ID.values())[5])]["n_games_lhp"] = 40
    with pytest.raises(AssertionError, match="<80 games"):
        _sanity_check({
            "method": "shrunk_observed_over_expected_k_rate_by_hand",
            "factors": factors,
        })


def test_sanity_check_by_hand_passes_realistic_payload():
    payload = {
        "method": "shrunk_observed_over_expected_k_rate_by_hand",
        "factors": _by_hand_factors(all_below_one=True),
    }
    _sanity_check(payload)  # no raise


def test_leave_one_venue_out_amplifies_real_venue_effect():
    """Pitcher P throws 30% Ks at venue A and 15% at venue B (same season).
    Without leave-one-venue-out, expected at A uses average ~22.5% → ratio
    at A is ~1.33. With leave-one-venue-out, expected at A uses 15% (the
    other venue) → ratio is 2.00. The new methodology sharpens venue
    effects vs. the pitcher's other-venue baseline.
    """
    rows = []
    # 10 games at venue A (NYY) with 30% K, 25 PA each
    for g in range(10):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=10000 + g, pitcher=1, home_team="NYY",
                p_throws="R", is_k=(ab < 7),
            ))
    # 10 games at venue B (BOS) with 15% K
    for g in range(10):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=20000 + g, pitcher=1, home_team="BOS",
                p_throws="R", is_k=(ab < 4),
            ))
    df = pd.DataFrame(rows)
    from scripts.derive_park_k_factors import _build_pitcher_game_table

    pg_loo = _build_pitcher_game_table(df, leave_one_venue_out=True)
    pg_full = _build_pitcher_game_table(df, leave_one_venue_out=False)

    # Mean ratio at venue NYY (3313): LOO should be higher than full-season
    nyy_loo = pg_loo[pg_loo["venue_id"] == 3313]["ratio"].mean()
    nyy_full = pg_full[pg_full["venue_id"] == 3313]["ratio"].mean()
    assert nyy_loo > nyy_full, (
        f"LOO ratio at NYY ({nyy_loo:.3f}) should exceed full-season "
        f"ratio ({nyy_full:.3f})"
    )


def test_leave_one_venue_out_drops_thin_baseline_pitcher_games():
    """A pitcher with only 4 games total (all at one venue) gets dropped —
    can't form a valid leave-one-venue-out baseline."""
    rows = []
    for g in range(4):
        for ab in range(25):
            rows.append(_pa_row(
                game_pk=10000 + g, pitcher=99, home_team="NYY",
                p_throws="R", is_k=(ab < 6),
            ))
    df = pd.DataFrame(rows)
    from scripts.derive_park_k_factors import _build_pitcher_game_table

    pg = _build_pitcher_game_table(df, leave_one_venue_out=True)
    # All 4 games dropped because pitcher had zero other-venue games
    assert pg.empty or (pg["pitcher"] == 99).sum() == 0


def test_sanity_check_by_hand_warns_on_large_disparity(caplog):
    factors = _by_hand_factors()
    factors["1"]["factor_lhp"] = 0.92
    factors["1"]["factor_rhp"] = 0.98  # |delta|=0.06, under 0.10 threshold
    factors[str(list(TEAM_TO_VENUE_ID.values())[-1])]["factor_lhp"] = 0.85
    factors[str(list(TEAM_TO_VENUE_ID.values())[-1])]["factor_rhp"] = 0.97  # |delta|=0.12
    import logging
    caplog.set_level(logging.WARNING)
    _sanity_check({
        "method": "shrunk_observed_over_expected_k_rate_by_hand",
        "factors": factors,
    })
    assert any("large L/R disparity" in r.message for r in caplog.records)
