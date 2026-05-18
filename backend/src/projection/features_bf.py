"""Phase 3b: E[BF] (expected batters faced) feature builders + composition.

Each public builder is a pure function:

    builder(bundle, ctx) -> (value, missing_reason)

`value` is None when data is missing; the caller must skip the pitcher
rather than fall back to a default. ``missing_reason`` is a short string for
logging.

E[BF] composition is additive in BF space:
    E[BF] = baseline_bf + sum(adjustments), clipped to [BF_FLOOR, BF_CEIL]

Calibration coefficients (the conversion of feature values into BF
adjustments) are first-pass placeholders. They will be re-derived in
Phase 4 from observed-BF-vs-feature regressions across 2024-2025 starts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

from .inputs import (
    BatterInputs,
    HandednessAverages,
    LeagueAverages,
    OpposingLineup,
    PitcherInputs,
    ProjectionBundle,
    ProjectionContext,
)

# ---- Magic numbers (calibrate in Phase 4) ----------------------------------

BASELINE_BF = 24.0           # leaguewide SP-game BF average
BF_FLOOR = 12.0              # per spec, projected_BF < 12 -> hard skip
BF_CEIL = 32.0

PA_PER_INNING_APPROX = 4.3   # leaguewide PA/IP, used for IP-from-PA rollups

# Feature -> BF adjustment slope. Each line is "every unit above a neutral
# anchor maps to this many BF". Anchors are league-average values.
COEFF_IP_PER_START = 3.0     # per IP above 5.0
ANCHOR_IP_PER_START = 5.0

COEFF_PITCHES_PER_PA = -1.5  # per pitch above 3.85 P/PA (efficient SPs face more BF)
ANCHOR_PITCHES_PER_PA = 3.85

COEFF_PITCHES_PER_PA_30D = -1.2  # same slope, slightly muted for short-window noise
ANCHOR_PITCHES_PER_PA_30D = 3.85

COEFF_LINEUP_OBP = 50.0      # per 1.0 in OBP-space; ~0.5 BF per 10 pts OBP
ANCHOR_LINEUP_OBP = 0.315    # league avg OBP

COEFF_PARK_RUN_FACTOR = 1.5  # per 1.0 in factor-space; high-offense park => more BF
ANCHOR_PARK_RUN_FACTOR = 1.0

COEFF_WEATHER = 1.5
ANCHOR_WEATHER = 1.0

DAYS_REST_BF_ADJUSTMENT = {
    "<4": -0.3,
    "4": -0.1,
    "5": 0.0,
    "6+": 0.2,
    "first_start_or_il_return": -0.5,
}

# Bayesian shrinkage prior strengths (in PA units unless noted)
K_PRIOR_IP_PER_START_STARTS = 5   # ~half a typical 30-day sample (in start units)
K_PRIOR_PITCHES_PER_PA = 200      # season pitches needed before we trust it
K_PRIOR_PITCHES_PER_PA_30D = 8    # form weight vs season prior

# Batting-order PA weights for lineup aggregation
BATTING_ORDER_PA_WEIGHTS = {
    1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2,
    6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8,
}

ON_BASE_EVENTS = frozenset({
    "walk", "hit_by_pitch", "single", "double", "triple", "home_run",
})

# Required-vs-optional feature membership for skip logic.
REQUIRED_FEATURES = (
    "pitcher_ip_per_start_30d_shrunk",
    "pitcher_pitches_per_pa_season",
    "lineup_obp_vs_hand",
    "park_run_environment_factor",
)


# ---- Helpers ---------------------------------------------------------------


def _count_pa(rows: Iterable[dict]) -> int:
    """Distinct (game_pk, at_bat_number) tuples."""
    seen: set[tuple[int, int]] = set()
    for r in rows:
        gp = r.get("game_pk")
        ab = r.get("at_bat_number")
        if gp is None or ab is None:
            continue
        seen.add((int(gp), int(ab)))
    return len(seen)


def _count_starts(rows: Iterable[dict]) -> int:
    """Distinct game_pks the pitcher appeared in (proxy for starts)."""
    return len({int(r["game_pk"]) for r in rows if r.get("game_pk") is not None})


def _count_pitches(rows: Iterable[dict]) -> int:
    return sum(1 for _ in rows)


def _ip_estimate(rows: Iterable[dict]) -> float:
    """Innings pitched, estimated from PA / 4.3."""
    pa = _count_pa(rows)
    return pa / PA_PER_INNING_APPROX


def _pa_event_rows(rows: Iterable[dict]) -> list[dict]:
    """Take one row per PA — the row with the highest pitch_number for each at-bat.

    Many features that key off ``events`` only want the terminal pitch of
    each PA (the only one carrying a non-null event).
    """
    by_ab: dict[tuple[int, int], dict] = {}
    for r in rows:
        gp = r.get("game_pk")
        ab = r.get("at_bat_number")
        if gp is None or ab is None:
            continue
        key = (int(gp), int(ab))
        prev = by_ab.get(key)
        pn = r.get("pitch_number") or 0
        prev_pn = (prev or {}).get("pitch_number") or 0
        if prev is None or pn > prev_pn:
            by_ab[key] = r
    return list(by_ab.values())


def _is_dome(bundle: ProjectionBundle) -> bool:
    return bundle.game_context.is_dome or (
        (bundle.game_context.weather.conditions or "").lower() == "dome"
    )


# ---- Feature builders ------------------------------------------------------


def pitcher_ip_per_start_30d_shrunk(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """IP/start last 30d, Bayesian-shrunk to season (or prior_year if season empty)."""
    p = bundle.pitcher
    starts_30d = _count_starts(p.statcast_pitches_30d)
    ip_30d = _ip_estimate(p.statcast_pitches_30d)
    ip_per_start_30d = (ip_30d / starts_30d) if starts_30d > 0 else None

    starts_season = _count_starts(p.statcast_pitches_season)
    ip_season = _ip_estimate(p.statcast_pitches_season)
    ip_per_start_season = (ip_season / starts_season) if starts_season > 0 else None

    starts_prior = _count_starts(p.statcast_pitches_prior_year)
    ip_prior = _ip_estimate(p.statcast_pitches_prior_year)
    ip_per_start_prior = (ip_prior / starts_prior) if starts_prior > 0 else None

    # Determine the shrinkage prior: season first, else prior_year.
    if ip_per_start_season is not None:
        prior_rate = ip_per_start_season
    elif ip_per_start_prior is not None:
        prior_rate = ip_per_start_prior
    else:
        return None, "no season or prior-year IP data"

    if ip_per_start_30d is None:
        # No 30d sample. Use the season/prior anchor directly.
        return prior_rate, None

    k = K_PRIOR_IP_PER_START_STARTS
    shrunk = (starts_30d * ip_per_start_30d + k * prior_rate) / (starts_30d + k)
    return shrunk, None


def pitcher_pitches_per_pa_season(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    season_pitches = _count_pitches(p.statcast_pitches_season)
    season_pa = _count_pa(p.statcast_pitches_season)
    prior_pitches = _count_pitches(p.statcast_pitches_prior_year)
    prior_pa = _count_pa(p.statcast_pitches_prior_year)

    season_rate = (season_pitches / season_pa) if season_pa > 0 else None
    prior_rate = (prior_pitches / prior_pa) if prior_pa > 0 else None

    if season_rate is None and prior_rate is None:
        return None, "no season or prior-year pitches"

    # Per spec: shrink to prior_year only when season_pitches < 200.
    if season_pitches < 200:
        if prior_rate is None:
            # Small season + no prior -> use what we have, no shrink.
            return season_rate, None
        if season_rate is None:
            return prior_rate, None
        k = K_PRIOR_PITCHES_PER_PA
        shrunk = (season_pitches + prior_rate * k) / (season_pa + k)
        return shrunk, None

    return season_rate, None


def pitcher_pitches_per_pa_30d(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    pitches_30d = _count_pitches(p.statcast_pitches_30d)
    pa_30d = _count_pa(p.statcast_pitches_30d)
    if pa_30d == 0:
        return None, "no PAs in 30d window"

    # Need a season anchor for the Bayesian blend.
    season_rate, _ = pitcher_pitches_per_pa_season(bundle, ctx)
    if season_rate is None:
        return None, "no season anchor for 30d blend"

    k = K_PRIOR_PITCHES_PER_PA_30D
    shrunk = (pitches_30d + season_rate * k) / (pa_30d + k)
    return shrunk, None


def lineup_obp_vs_hand(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Lineup-aggregated OBP vs same-hand pitcher, weighted by batting order."""
    if not bundle.opposing_lineup.lineup_posted:
        return None, "lineup not posted"
    pitcher_hand = bundle.pitcher.handedness
    if pitcher_hand not in ("L", "R"):
        return None, "pitcher handedness unknown"

    weights: list[float] = []
    weighted_obps: list[float] = []
    for batter in bundle.opposing_lineup.batters:
        avg = ctx.league_avgs.lookup(batter.handedness, pitcher_hand)
        if avg is None:
            continue
        # Filter batter's PAs to same-hand pitcher matchups
        same_hand_pas = _pa_event_rows(
            [r for r in batter.statcast_pa_season if r.get("p_throws") == pitcher_hand]
        )
        n_pa = len(same_hand_pas)
        if n_pa == 0:
            batter_obp = avg.obp  # league prior, no actual data
        else:
            on_base = sum(1 for r in same_hand_pas if r.get("events") in ON_BASE_EVENTS)
            raw_obp = on_base / n_pa
            if n_pa < 50:
                # Shrink to league average vs hand
                k = 50
                batter_obp = (on_base + avg.obp * k) / (n_pa + k)
            else:
                batter_obp = raw_obp
        w = BATTING_ORDER_PA_WEIGHTS.get(batter.batting_order, 4.0)
        weights.append(w)
        weighted_obps.append(batter_obp * w)

    if not weights:
        return None, "no batters with resolvable handedness"
    return sum(weighted_obps) / sum(weights), None


def park_run_environment_factor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    factor = ctx.park_run_factors.get(bundle.game_context.venue_id)
    if factor is None:
        return None, f"park not in run-factor file (venue_id={bundle.game_context.venue_id})"
    return factor, None


def weather_run_environment(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Composite of temp + wind effect on run scoring. Dome -> 1.0 neutral."""
    if _is_dome(bundle):
        return 1.0, None
    w = bundle.game_context.weather
    if w.temp_f is None or w.wind_speed_mph is None:
        return None, "weather temp_f or wind_speed_mph missing"

    # Temp effect: ~0.05% per °F above 70°F (mild — temperature matters less
    # than wind for run env at the major-league level).
    temp_factor = 1.0 + (w.temp_f - 70.0) * 0.0005

    # Wind: signed by direction.
    wind_signed = 0.0
    direction = (w.wind_direction or "").lower()
    if "out" in direction:
        wind_signed = w.wind_speed_mph
    elif "in" in direction:
        wind_signed = -w.wind_speed_mph
    wind_factor = 1.0 + wind_signed * 0.005

    return temp_factor * wind_factor, None


def days_rest_bucket(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[str | None, str | None]:
    dr = bundle.game_context.days_rest
    if dr is None:
        return "first_start_or_il_return", None
    if dr < 4:
        return "<4", None
    if dr == 4:
        return "4", None
    if dr == 5:
        return "5", None
    return "6+", None


def team_bullpen_short_hook_indicator(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Deferred to v1.1 — requires team-level historical hook patterns we
    don't have in the bundle. Returns 0.0 (neutral) so composition is unaffected.

    Add a real data source in a later phase (manager rotation logs,
    Statcast team-pitcher aggregations) and wire it in here.
    """
    return 0.0, None


# ---- Composition -----------------------------------------------------------


@dataclass(frozen=True)
class EBFResult:
    e_bf: float | None           # clipped to [BF_FLOOR, BF_CEIL]
    e_bf_raw: float | None       # pre-clip; projector uses this for the < 12 hard filter
    used_features: dict
    skipped: bool
    skip_reason: str | None


def _adjustment_ip_per_start(value: float) -> float:
    return (value - ANCHOR_IP_PER_START) * COEFF_IP_PER_START


def _adjustment_pitches_per_pa(value: float) -> float:
    return (value - ANCHOR_PITCHES_PER_PA) * COEFF_PITCHES_PER_PA


def _adjustment_pitches_per_pa_30d(value: float) -> float:
    return (value - ANCHOR_PITCHES_PER_PA_30D) * COEFF_PITCHES_PER_PA_30D


def _adjustment_lineup_obp(value: float) -> float:
    return (value - ANCHOR_LINEUP_OBP) * COEFF_LINEUP_OBP


def _adjustment_park_factor(value: float) -> float:
    return (value - ANCHOR_PARK_RUN_FACTOR) * COEFF_PARK_RUN_FACTOR


def _adjustment_weather(value: float) -> float:
    return (value - ANCHOR_WEATHER) * COEFF_WEATHER


def compute_e_bf(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> EBFResult:
    """Compose E[BF] from the additive feature set.

    Required features (any missing -> skipped=True): ip_per_start_30d_shrunk,
    pitches_per_pa_season, lineup_obp_vs_hand, park_run_environment_factor.
    Optional features can be missing without skipping; they're omitted from
    the sum and noted in used_features.
    """
    used: dict = {}

    # Required
    ip_val, ip_miss = pitcher_ip_per_start_30d_shrunk(bundle, ctx)
    if ip_val is None:
        return EBFResult(
            e_bf=None, e_bf_raw=None,
            used_features={"pitcher_ip_per_start_30d_shrunk": None},
            skipped=True, skip_reason=f"pitcher_ip_per_start_30d_shrunk: {ip_miss}",
        )
    used["pitcher_ip_per_start_30d_shrunk"] = ip_val

    pps_val, pps_miss = pitcher_pitches_per_pa_season(bundle, ctx)
    if pps_val is None:
        return EBFResult(
            e_bf=None, e_bf_raw=None, used_features=used,
            skipped=True, skip_reason=f"pitcher_pitches_per_pa_season: {pps_miss}",
        )
    used["pitcher_pitches_per_pa_season"] = pps_val

    lobp_val, lobp_miss = lineup_obp_vs_hand(bundle, ctx)
    if lobp_val is None:
        return EBFResult(
            e_bf=None, e_bf_raw=None, used_features=used,
            skipped=True, skip_reason=f"lineup_obp_vs_hand: {lobp_miss}",
        )
    used["lineup_obp_vs_hand"] = lobp_val

    park_val, park_miss = park_run_environment_factor(bundle, ctx)
    if park_val is None:
        return EBFResult(
            e_bf=None, e_bf_raw=None, used_features=used,
            skipped=True, skip_reason=f"park_run_environment_factor: {park_miss}",
        )
    used["park_run_environment_factor"] = park_val

    # Optional features — compute, include if present, otherwise skip silently.
    pps30_val, _ = pitcher_pitches_per_pa_30d(bundle, ctx)
    if pps30_val is not None:
        used["pitcher_pitches_per_pa_30d"] = pps30_val

    weather_val, _ = weather_run_environment(bundle, ctx)
    if weather_val is not None:
        used["weather_run_environment"] = weather_val

    rest_bucket, _ = days_rest_bucket(bundle, ctx)
    if rest_bucket is not None:
        used["days_rest_bucket"] = rest_bucket

    bullpen_val, _ = team_bullpen_short_hook_indicator(bundle, ctx)
    if bullpen_val is not None:
        used["team_bullpen_short_hook_indicator"] = bullpen_val

    # Sum adjustments.
    adjustments = (
        _adjustment_ip_per_start(used["pitcher_ip_per_start_30d_shrunk"])
        + _adjustment_pitches_per_pa(used["pitcher_pitches_per_pa_season"])
        + _adjustment_lineup_obp(used["lineup_obp_vs_hand"])
        + _adjustment_park_factor(used["park_run_environment_factor"])
    )
    if "pitcher_pitches_per_pa_30d" in used:
        adjustments += _adjustment_pitches_per_pa_30d(used["pitcher_pitches_per_pa_30d"])
    if "weather_run_environment" in used:
        adjustments += _adjustment_weather(used["weather_run_environment"])
    if "days_rest_bucket" in used:
        adjustments += DAYS_REST_BF_ADJUSTMENT.get(used["days_rest_bucket"], 0.0)
    if "team_bullpen_short_hook_indicator" in used:
        adjustments += used["team_bullpen_short_hook_indicator"]

    raw_e_bf = BASELINE_BF + adjustments
    e_bf = max(BF_FLOOR, min(BF_CEIL, raw_e_bf))

    return EBFResult(
        e_bf=e_bf,
        e_bf_raw=raw_e_bf,
        used_features=used,
        skipped=False,
        skip_reason=None,
    )
