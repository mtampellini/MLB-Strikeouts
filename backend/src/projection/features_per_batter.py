"""Phase 3-v2c-ii: per-batter feature builders for the P(K|PA) rewrite.

Each builder returns ``dict[batter_mlbam_id, (rate | None, missing_reason | None)]``.
The projector consumes the dict to iterate the lineup per batter rather than
working with a lineup-aggregated scalar.

This module ONLY contains the per-batter shape. The existing aggregated
``lineup_*`` builders in :mod:`features_bf` and :mod:`features_kpa` remain
unchanged and are still consumed by E[BF] (Phase 4c).

Builders:

- :func:`per_batter_k_pct_vs_hand`
- :func:`per_batter_zone_contact_pct`
- :func:`per_batter_chase_rate`
- :func:`per_batter_obp_vs_hand`

Shared :func:`_shrunk_batter_rate` helper implements the Bayesian shrinkage
chain: current season -> prior year (when available) -> league average.

The current Phase 3-v2c-i contract does not carry per-batter prior-year
Statcast windows, so the prior-year arm of the chain is testable in
isolation via the helper but is dormant in builder integration until a
future additive contract update adds the window. The chain is wired so
that wiring it up later is a one-line change in each builder.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .features_bf import (
    BATTING_ORDER_PA_WEIGHTS,
    ON_BASE_EVENTS,
    _pa_event_rows,
)
from .features_kpa import (
    IN_ZONE_ZONES,
    K_EVENTS,
    OUT_OF_ZONE_ZONES,
    SWING_DESCRIPTIONS,
    SWINGING_STRIKE_DESCRIPTIONS,
)
from .inputs import (
    HandednessAverages,
    LeagueAverages,
    ProjectionBundle,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Shrinkage defaults. Match the spec: k_prior=80, min_current_n=50,
# min_prior_n=200.
DEFAULT_K_PRIOR = 80
DEFAULT_MIN_CURRENT_N = 50
DEFAULT_MIN_PRIOR_N = 200

# Sentinel returned when league_averages can't be loaded (catastrophic — the
# projector consumer should treat this as a contract gap, not a per-batter
# missing field).
SENTINEL_NO_LEAGUE_AVGS_KEY = 0
SENTINEL_NO_LEAGUE_AVGS_REASON = "league_averages not in bundle"


# ---- League averages loader (cached, lazy) ---------------------------------


@lru_cache(maxsize=8)
def _load_league_avgs(season: int) -> LeagueAverages | None:
    """Lazy load ``league_averages_<season>.json``. Returns None if missing.

    Cached per-season because Phase 4a regenerates this file at most yearly.
    """
    path = PROCESSED_DIR / f"league_averages_{season}.json"
    if not path.exists():
        logger.warning(
            "per-batter features: league_averages file not found at %s", path,
        )
        return None
    try:
        return LeagueAverages.from_json(path)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "per-batter features: failed to load league_averages_%d.json: %s",
            season, exc,
        )
        return None


# ---- Shrinkage helper ------------------------------------------------------


def _shrunk_batter_rate(
    observed_rate: float | None,
    observed_n: int,
    prior_year_rate: float | None,
    prior_year_n: int,
    league_rate: float | None,
    k_prior: int = DEFAULT_K_PRIOR,
    min_current_n: int = DEFAULT_MIN_CURRENT_N,
    min_prior_n: int = DEFAULT_MIN_PRIOR_N,
) -> tuple[float | None, str | None]:
    """Three-step Bayesian shrinkage chain.

    1. ``observed_n >= min_current_n`` -> shrink observed toward league with
       ``k_prior`` prior. Confidence: ``current_season_shrunk_to_league``.
    2. ``prior_year_n >= min_prior_n`` -> shrink current toward prior with
       ``k_prior``, then shrink the result toward league with ``k_prior``.
       Confidence: ``prior_year_chain_to_league``.
    3. ``league_rate`` available -> return league rate.
       Confidence: ``small_sample_league_fallback``.
    4. Nothing available -> ``(None, "no rate data available")``.
    """
    if league_rate is None:
        return None, "no rate data available"

    if observed_n >= min_current_n and observed_rate is not None:
        shrunk = (observed_rate * observed_n + league_rate * k_prior) / (
            observed_n + k_prior
        )
        return shrunk, "current_season_shrunk_to_league"

    if prior_year_n >= min_prior_n and prior_year_rate is not None:
        # Step A: blend current observation (or 0 weight) toward prior-year rate
        if observed_n > 0 and observed_rate is not None:
            step1_rate = (observed_rate * observed_n + prior_year_rate * k_prior) / (
                observed_n + k_prior
            )
        else:
            step1_rate = prior_year_rate
        step1_n = observed_n + k_prior
        # Step B: blend step1 toward league average
        step2 = (step1_rate * step1_n + league_rate * k_prior) / (step1_n + k_prior)
        return step2, "prior_year_chain_to_league"

    return league_rate, "small_sample_league_fallback"


# ---- Common helpers --------------------------------------------------------


def _effective_batter_hand(
    batter_hand: str | None, pitcher_hand: str
) -> str | None:
    """Resolve switch hitters: S vs R -> L, S vs L -> R. Returns None for
    unknown handedness."""
    if not batter_hand:
        return None
    if batter_hand == "S":
        return "L" if pitcher_hand == "R" else "R"
    if batter_hand in ("L", "R"):
        return batter_hand
    return None


def _same_hand_pa_rows(batter, pitcher_hand: str) -> list[dict]:
    """Return PA-terminal rows for the batter filtered to same-hand pitcher
    matchups."""
    return _pa_event_rows(
        r for r in batter.statcast_pa_season
        if r.get("p_throws") == pitcher_hand
    )


def _same_hand_pitch_rows(batter, pitcher_hand: str) -> list[dict]:
    """Return all pitch rows (not PA-terminal) for same-hand matchups."""
    return [
        r for r in batter.statcast_pa_season
        if r.get("p_throws") == pitcher_hand
    ]


def _zone_swings_and_contacts(rows: Iterable[dict]) -> tuple[int, int]:
    """Return (contacts, in_zone_swings) for zone-contact computation."""
    swings = 0
    contacts = 0
    for r in rows:
        try:
            z = int(r["zone"]) if r.get("zone") is not None else None
        except (TypeError, ValueError):
            continue
        if z not in IN_ZONE_ZONES:
            continue
        desc = r.get("description")
        if desc not in SWING_DESCRIPTIONS:
            continue
        swings += 1
        if desc not in SWINGING_STRIKE_DESCRIPTIONS:
            contacts += 1
    return contacts, swings


def _ooz_pitches_and_swings(rows: Iterable[dict]) -> tuple[int, int]:
    """Return (swings_at_ooz, ooz_pitches) for chase-rate computation."""
    swings = 0
    ooz = 0
    for r in rows:
        try:
            z = int(r["zone"]) if r.get("zone") is not None else None
        except (TypeError, ValueError):
            continue
        if z not in OUT_OF_ZONE_ZONES:
            continue
        ooz += 1
        if r.get("description") in SWING_DESCRIPTIONS:
            swings += 1
    return swings, ooz


def _validate_lineup_and_get_avgs(
    bundle: ProjectionBundle,
) -> tuple[LeagueAverages | None, str | None]:
    """Common preflight: ensure the lineup is posted and league_averages
    loaded. Returns (league_avgs, gating_reason). If gating_reason is non-None,
    the builder should bail out with the appropriate signal.
    """
    if not bundle.opposing_lineup.lineup_posted:
        return None, "lineup_not_posted"
    pitcher_hand = bundle.pitcher.handedness
    if pitcher_hand not in ("L", "R"):
        return None, "pitcher_handedness_unknown"
    season = bundle.metadata.game_date.year
    avgs = _load_league_avgs(season)
    if avgs is None:
        # Try fallback to previous season (Phase 4a's most recent file)
        avgs = _load_league_avgs(season - 1)
        if avgs is None:
            return None, "league_averages_not_in_bundle"
    return avgs, None


def _empty_lineup_or_sentinel(
    gating_reason: str,
) -> dict[int, tuple[float | None, str | None]]:
    """Translate gating reason to the appropriate empty/sentinel dict shape.

    - lineup not posted -> {} (projector handles empty)
    - league_averages missing -> {0: (None, "league_averages not in bundle")}
      sentinel signaling a contract gap
    - pitcher handedness unknown -> {} (lineup-based features are undefined)
    """
    if gating_reason == "league_averages_not_in_bundle":
        return {SENTINEL_NO_LEAGUE_AVGS_KEY: (None, SENTINEL_NO_LEAGUE_AVGS_REASON)}
    return {}


# ---- Per-batter builders ---------------------------------------------------


def per_batter_k_pct_vs_hand(
    bundle: ProjectionBundle,
) -> dict[int, tuple[float | None, str | None]]:
    """Per-batter K% in PAs against same-hand pitchers.

    Returns dict keyed by batter_mlbam_id with (k_pct, reason). Reason is the
    confidence flag from the shrinkage chain (or the missing-data reason).
    """
    avgs, gate = _validate_lineup_and_get_avgs(bundle)
    if gate is not None:
        return _empty_lineup_or_sentinel(gate)

    pitcher_hand = bundle.pitcher.handedness  # already validated L/R
    out: dict[int, tuple[float | None, str | None]] = {}
    for batter in bundle.opposing_lineup.batters:
        cell = avgs.lookup(batter.handedness, pitcher_hand)
        if cell is None:
            out[batter.mlbam_id] = (
                None, "no league average for this handedness pair",
            )
            continue
        rows = _same_hand_pa_rows(batter, pitcher_hand)
        n_pa = len(rows)
        k_count = sum(1 for r in rows if r.get("events") in K_EVENTS)
        observed_rate = k_count / n_pa if n_pa > 0 else None
        rate, reason = _shrunk_batter_rate(
            observed_rate=observed_rate,
            observed_n=n_pa,
            prior_year_rate=None,
            prior_year_n=0,
            league_rate=cell.k_pct,
        )
        out[batter.mlbam_id] = (rate, reason)
    return out


def per_batter_zone_contact_pct(
    bundle: ProjectionBundle,
) -> dict[int, tuple[float | None, str | None]]:
    """Per-batter zone-contact rate (contact / in-zone swings) vs same hand."""
    avgs, gate = _validate_lineup_and_get_avgs(bundle)
    if gate is not None:
        return _empty_lineup_or_sentinel(gate)

    pitcher_hand = bundle.pitcher.handedness
    out: dict[int, tuple[float | None, str | None]] = {}
    for batter in bundle.opposing_lineup.batters:
        cell = avgs.lookup(batter.handedness, pitcher_hand)
        if cell is None:
            out[batter.mlbam_id] = (
                None, "no league average for this handedness pair",
            )
            continue
        rows = _same_hand_pitch_rows(batter, pitcher_hand)
        contacts, swings = _zone_swings_and_contacts(rows)
        observed_rate = contacts / swings if swings > 0 else None
        rate, reason = _shrunk_batter_rate(
            observed_rate=observed_rate,
            observed_n=swings,
            prior_year_rate=None,
            prior_year_n=0,
            league_rate=cell.zone_contact_pct,
        )
        out[batter.mlbam_id] = (rate, reason)
    return out


def per_batter_chase_rate(
    bundle: ProjectionBundle,
) -> dict[int, tuple[float | None, str | None]]:
    """Per-batter chase rate (out-of-zone swings / out-of-zone pitches)
    vs same hand."""
    avgs, gate = _validate_lineup_and_get_avgs(bundle)
    if gate is not None:
        return _empty_lineup_or_sentinel(gate)

    pitcher_hand = bundle.pitcher.handedness
    out: dict[int, tuple[float | None, str | None]] = {}
    for batter in bundle.opposing_lineup.batters:
        cell = avgs.lookup(batter.handedness, pitcher_hand)
        if cell is None:
            out[batter.mlbam_id] = (
                None, "no league average for this handedness pair",
            )
            continue
        rows = _same_hand_pitch_rows(batter, pitcher_hand)
        swings, ooz = _ooz_pitches_and_swings(rows)
        observed_rate = swings / ooz if ooz > 0 else None
        rate, reason = _shrunk_batter_rate(
            observed_rate=observed_rate,
            observed_n=ooz,
            prior_year_rate=None,
            prior_year_n=0,
            league_rate=cell.chase_rate,
        )
        out[batter.mlbam_id] = (rate, reason)
    return out


def per_batter_obp_vs_hand(
    bundle: ProjectionBundle,
) -> dict[int, tuple[float | None, str | None]]:
    """Per-batter OBP in PAs against same-hand pitchers."""
    avgs, gate = _validate_lineup_and_get_avgs(bundle)
    if gate is not None:
        return _empty_lineup_or_sentinel(gate)

    pitcher_hand = bundle.pitcher.handedness
    out: dict[int, tuple[float | None, str | None]] = {}
    for batter in bundle.opposing_lineup.batters:
        cell = avgs.lookup(batter.handedness, pitcher_hand)
        if cell is None:
            out[batter.mlbam_id] = (
                None, "no league average for this handedness pair",
            )
            continue
        rows = _same_hand_pa_rows(batter, pitcher_hand)
        n_pa = len(rows)
        on_base = sum(1 for r in rows if r.get("events") in ON_BASE_EVENTS)
        observed_rate = on_base / n_pa if n_pa > 0 else None
        rate, reason = _shrunk_batter_rate(
            observed_rate=observed_rate,
            observed_n=n_pa,
            prior_year_rate=None,
            prior_year_n=0,
            league_rate=cell.obp,
        )
        out[batter.mlbam_id] = (rate, reason)
    return out
