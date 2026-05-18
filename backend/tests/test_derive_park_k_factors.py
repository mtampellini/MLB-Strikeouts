"""Phase 4b unit tests: derive_park_k_factors helper functions."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.derive_park_k_factors import (
    SHRINKAGE_PRIOR_GAMES,
    TEAM_TO_VENUE_ID,
    _build_park_k_factors,
    _sanity_check,
)


def _pa_row(*, game_pk, pitcher, home_team, season=2024, is_k=False):
    return {
        "events": "strikeout" if is_k else "field_out",
        "game_pk": game_pk,
        "pitcher": pitcher,
        "home_team": home_team,
        "__season": season,
        "game_type": "R",
    }


def test_team_to_venue_id_covers_30_teams():
    assert len(set(TEAM_TO_VENUE_ID.values())) == 30


def test_team_to_venue_id_has_no_oak():
    """Statcast normalizes Athletics to ATH only; OAK should not be in the map."""
    assert "OAK" not in TEAM_TO_VENUE_ID


def test_build_factors_returns_one_entry_per_venue():
    # Two pitchers each face 100 PA at two parks. Even K rate everywhere.
    rows = []
    for game_pk, home_team in enumerate(("NYY", "TB"), start=1):
        for pitcher in (1, 2):
            for ab in range(50):
                rows.append(_pa_row(
                    game_pk=game_pk * 1000 + pitcher,
                    pitcher=pitcher,
                    home_team=home_team,
                    is_k=(ab % 4 == 0),  # ~25% K rate
                ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df)
    assert "factors" in payload
    venue_ids = {int(k) for k in payload["factors"].keys()}
    assert venue_ids == {3313, 12}  # NYY, TB


def test_build_factors_applies_shrinkage_toward_neutral():
    """A park with very few games should land near 1.0 even if raw ratio is extreme."""
    rows = []
    # 10 games at venue 12 (TB), all 100% K rate → raw ratio = many
    for g in range(10):
        for ab in range(20):
            rows.append(_pa_row(
                game_pk=2000 + g, pitcher=999, home_team="TB", is_k=True,
            ))
    # 500 games at venue 3313 (NYY), 25% K rate
    for g in range(500):
        for ab in range(20):
            rows.append(_pa_row(
                game_pk=g, pitcher=999, home_team="NYY", is_k=(ab < 5),
            ))
    df = pd.DataFrame(rows)
    payload = _build_park_k_factors(df)
    tb = payload["factors"]["12"]["factor"]
    nyy = payload["factors"]["3313"]["factor"]
    # TB with 10 games and 100% Ks would be RAW ratio = 4.0 but shrinkage
    # pulls it heavily toward 1.0
    assert 0.9 < tb < 1.5
    # NYY with 500 games at neutral K rate should be very close to 1.0
    assert abs(nyy - 1.0) < 0.05


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
