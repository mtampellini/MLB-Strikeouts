"""Phase 3c: P(K|PA) feature builders + log-odds composition.

Same shape as :mod:`features_bf`: each builder is a pure function returning
``(value, missing_reason)``. The composition uses log-odds (not raw
multiplicative) so the result is guaranteed to stay in (0, 1), then clipped
to [P_K_PA_FLOOR, P_K_PA_CEIL].

Adjustments are log-odds shifts:
    logit(p) = logit(baseline) + Σ shifts
where each shift is ``log(rate / league_anchor)`` or a calibrated coefficient
× delta.

Required features (any missing -> skip): pitcher_k_pct_season_shrunk,
lineup_k_pct_vs_hand, park_k_factor.

Calibration coefficients are first-pass placeholders. Phase 4 will re-derive
them from the regression of observed K rates against feature values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .features_bf import (
    BATTING_ORDER_PA_WEIGHTS,
    _count_pa,
    _count_pitches,
    _is_dome,
    _pa_event_rows,
)
from .inputs import (
    HandednessAverages,
    LeagueAverages,
    ProjectionBundle,
    ProjectionContext,
)

# ---- Calibration constants -------------------------------------------------

P_K_PA_FLOOR = 0.10
P_K_PA_CEIL = 0.45

K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})

SWINGING_STRIKE_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked",
})
CSW_DESCRIPTIONS = frozenset({
    "called_strike", "swinging_strike", "swinging_strike_blocked",
})
# Descriptions that involve a bat swing (made contact OR whiffed).
SWING_DESCRIPTIONS = frozenset({
    "foul", "foul_bunt", "foul_pitchout", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "swinging_strike", "swinging_strike_blocked",
    "missed_bunt",
})
IN_ZONE_ZONES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
OUT_OF_ZONE_ZONES = frozenset({11, 12, 13, 14})
FASTBALL_TYPES = frozenset({"FF", "SI", "FC"})

# Bayesian shrink priors
K_PRIOR_K_PCT_SEASON = 80      # PA strength of prior_year prior
K_PRIOR_K_PCT_30D = 30
K_PRIOR_CSW = 100              # pitches (also the spec's threshold)
K_PRIOR_LINEUP_BATTER = 50     # PA

# Log-odds coefficients (per unit-delta above an anchor)
COEFF_CSW = 0.6                # log-odds shift per 1.0 in CSW% delta
COEFF_CHASE_WHIFF = 0.4
COEFF_VELOCITY_Z = 0.04        # per +1 Z fastball velo
COEFF_PUTAWAY = 0.5
COEFF_ZONE_CONTACT = -0.6      # higher contact = lower K
COEFF_CHASE_RATE = 0.4

ANCHOR_CSW = 0.28              # leaguewide CSW%
ANCHOR_PUTAWAY = 0.40          # share of pitcher's most-used 2-strike pitch type


# ---- Pitcher-side builders -------------------------------------------------


def _pitcher_k_pct(rows: Iterable[dict]) -> tuple[int, int]:
    """Return (k_count, pa_count) from a window of pitches."""
    pa_rows = _pa_event_rows(rows)
    pa = len(pa_rows)
    k = sum(1 for r in pa_rows if r.get("events") in K_EVENTS)
    return k, pa


def pitcher_k_pct_season_shrunk(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    k_s, pa_s = _pitcher_k_pct(p.statcast_pitches_season)
    k_p, pa_p = _pitcher_k_pct(p.statcast_pitches_prior_year)

    season_rate = (k_s / pa_s) if pa_s > 0 else None
    prior_rate = (k_p / pa_p) if pa_p > 0 else None

    if season_rate is None and prior_rate is None:
        return None, "no season or prior-year K data"
    if season_rate is None:
        return prior_rate, None
    if prior_rate is None:
        return season_rate, None
    k_prior = K_PRIOR_K_PCT_SEASON
    shrunk = (k_s + prior_rate * k_prior) / (pa_s + k_prior)
    return shrunk, None


def pitcher_k_pct_30d_blended(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    k_30d, pa_30d = _pitcher_k_pct(p.statcast_pitches_30d)
    if pa_30d == 0:
        return None, "no PAs in 30d window"
    season_rate, _ = pitcher_k_pct_season_shrunk(bundle, ctx)
    if season_rate is None:
        return None, "no season anchor for 30d blend"
    k_prior = K_PRIOR_K_PCT_30D
    shrunk = (k_30d + season_rate * k_prior) / (pa_30d + k_prior)
    return shrunk, None


def _csw_count(rows: Iterable[dict]) -> tuple[int, int]:
    pitches = 0
    csw = 0
    for r in rows:
        pitches += 1
        if r.get("description") in CSW_DESCRIPTIONS:
            csw += 1
    return csw, pitches


def pitcher_csw_pct_30d(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    csw_30d, pitches_30d = _csw_count(p.statcast_pitches_30d)
    if pitches_30d == 0:
        return None, "no pitches in 30d window"
    raw_30d = csw_30d / pitches_30d
    if pitches_30d >= K_PRIOR_CSW:
        return raw_30d, None
    csw_s, pitches_s = _csw_count(p.statcast_pitches_season)
    if pitches_s == 0:
        return raw_30d, None
    season_rate = csw_s / pitches_s
    k_prior = K_PRIOR_CSW
    shrunk = (csw_30d + season_rate * k_prior) / (pitches_30d + k_prior)
    return shrunk, None


def _chase_whiff_counts(rows: Iterable[dict]) -> tuple[int, int]:
    """Return (chase_whiffs, total_chases)."""
    whiffs = 0
    chases = 0
    for r in rows:
        zone = r.get("zone")
        try:
            zone_i = int(zone) if zone is not None else None
        except (TypeError, ValueError):
            continue
        if zone_i not in OUT_OF_ZONE_ZONES:
            continue
        desc = r.get("description")
        if desc not in SWING_DESCRIPTIONS:
            continue
        chases += 1
        if desc in SWINGING_STRIKE_DESCRIPTIONS:
            whiffs += 1
    return whiffs, chases


def pitcher_chase_whiff_pct_30d(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    p = bundle.pitcher
    whiffs_30d, chases_30d = _chase_whiff_counts(p.statcast_pitches_30d)
    if chases_30d == 0:
        return None, "no out-of-zone swings in 30d window"
    raw_30d = whiffs_30d / chases_30d
    if chases_30d >= K_PRIOR_CSW:
        return raw_30d, None
    whiffs_s, chases_s = _chase_whiff_counts(p.statcast_pitches_season)
    if chases_s == 0:
        return raw_30d, None
    season_rate = whiffs_s / chases_s
    k_prior = K_PRIOR_CSW
    shrunk = (whiffs_30d + season_rate * k_prior) / (chases_30d + k_prior)
    return shrunk, None


def pitcher_velocity_trend_3starts(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Z-score: (last 3 starts' fastball velo) - (season fastball mean) / season SD.

    Returns None if <30 fastballs in last 3 starts."""
    p = bundle.pitcher
    fastballs_recent = _fastball_velos_last_n_starts(p.statcast_pitches_30d, n=3)
    if len(fastballs_recent) < 30:
        return None, f"only {len(fastballs_recent)} fastballs in last 3 starts"

    fb_season = [r["release_speed"] for r in p.statcast_pitches_season
                 if r.get("pitch_type") in FASTBALL_TYPES
                 and isinstance(r.get("release_speed"), (int, float))]
    if not fb_season:
        # Try prior year.
        fb_season = [r["release_speed"] for r in p.statcast_pitches_prior_year
                     if r.get("pitch_type") in FASTBALL_TYPES
                     and isinstance(r.get("release_speed"), (int, float))]
    if not fb_season:
        return None, "no season or prior-year fastball velocity anchor"
    mean_season = sum(fb_season) / len(fb_season)
    var_season = sum((v - mean_season) ** 2 for v in fb_season) / max(1, len(fb_season) - 1)
    sd_season = math.sqrt(var_season) if var_season > 0 else None
    if not sd_season or sd_season < 0.1:
        return None, "season fastball velocity SD too small"
    mean_recent = sum(fastballs_recent) / len(fastballs_recent)
    z = (mean_recent - mean_season) / sd_season
    return z, None


def _fastball_velos_last_n_starts(rows, *, n: int) -> list[float]:
    by_game: dict[int, list[float]] = {}
    for r in rows:
        if r.get("pitch_type") not in FASTBALL_TYPES:
            continue
        rs = r.get("release_speed")
        if not isinstance(rs, (int, float)):
            continue
        gp = r.get("game_pk")
        if gp is None:
            continue
        by_game.setdefault(int(gp), []).append(float(rs))
    # last N starts = highest N game_pks (Statcast game_pks increase chronologically)
    last_n_games = sorted(by_game.keys())[-n:]
    out: list[float] = []
    for g in last_n_games:
        out.extend(by_game[g])
    return out


def pitcher_putaway_pitch_concentration(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Share of two-strike pitches that are the pitcher's most-used 2K pitch type."""
    p = bundle.pitcher
    season_rows = [r for r in p.statcast_pitches_season if _is_two_strike(r)]
    if len(season_rows) >= 50:
        return _putaway_share(season_rows), None
    # Shrink with prior year if season sample is small.
    prior_rows = [r for r in p.statcast_pitches_prior_year if _is_two_strike(r)]
    if not season_rows and not prior_rows:
        return None, "no two-strike pitches in season or prior year"
    if not prior_rows:
        return _putaway_share(season_rows), None
    # Take prior year's share as the anchor with k_prior=50 pitches.
    season_share = _putaway_share(season_rows) if season_rows else 0.0
    prior_share = _putaway_share(prior_rows)
    n_s = len(season_rows)
    k_prior = 50
    shrunk = (season_share * n_s + prior_share * k_prior) / (n_s + k_prior)
    return shrunk, None


def _is_two_strike(row: dict) -> bool:
    return row.get("strikes") == 2


def _putaway_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    counts: dict[str, int] = {}
    for r in rows:
        pt = r.get("pitch_type") or "UNK"
        counts[pt] = counts.get(pt, 0) + 1
    if not counts:
        return 0.0
    top = max(counts.values())
    return top / sum(counts.values())


# ---- Lineup builders -------------------------------------------------------


def _on_base_or_k_count(rows: list[dict]) -> tuple[int, int, int]:
    """For a list of PA-terminal rows return (n_pa, n_k, n_on_base)."""
    n = len(rows)
    k = sum(1 for r in rows if r.get("events") in K_EVENTS)
    return n, k, sum(1 for r in rows if r.get("events") in {
        "walk", "hit_by_pitch", "single", "double", "triple", "home_run",
    })


def lineup_k_pct_vs_hand(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    if not bundle.opposing_lineup.lineup_posted:
        return None, "lineup not posted"
    pitcher_hand = bundle.pitcher.handedness
    if pitcher_hand not in ("L", "R"):
        return None, "pitcher handedness unknown"

    weights, weighted = [], []
    for batter in bundle.opposing_lineup.batters:
        avg = ctx.league_avgs.lookup(batter.handedness, pitcher_hand)
        if avg is None:
            continue
        rows = _pa_event_rows(
            [r for r in batter.statcast_pa_season if r.get("p_throws") == pitcher_hand]
        )
        n_pa = len(rows)
        k_count = sum(1 for r in rows if r.get("events") in K_EVENTS)
        if n_pa == 0:
            batter_k_pct = avg.k_pct
        elif n_pa < K_PRIOR_LINEUP_BATTER:
            batter_k_pct = (k_count + avg.k_pct * K_PRIOR_LINEUP_BATTER) / (
                n_pa + K_PRIOR_LINEUP_BATTER
            )
        else:
            batter_k_pct = k_count / n_pa
        w = BATTING_ORDER_PA_WEIGHTS.get(batter.batting_order, 4.0)
        weights.append(w)
        weighted.append(batter_k_pct * w)

    if not weights:
        return None, "no batters with resolvable handedness"
    return sum(weighted) / sum(weights), None


def _zone_contact_per_batter(
    rows: Iterable[dict],
) -> tuple[int, int]:
    """Return (contacts, in_zone_swings)."""
    swings = 0
    contacts = 0
    for r in rows:
        zone = r.get("zone")
        try:
            z = int(zone) if zone is not None else None
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


def lineup_zone_contact_pct(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    if not bundle.opposing_lineup.lineup_posted:
        return None, "lineup not posted"
    pitcher_hand = bundle.pitcher.handedness
    if pitcher_hand not in ("L", "R"):
        return None, "pitcher handedness unknown"

    weights, weighted = [], []
    for batter in bundle.opposing_lineup.batters:
        avg = ctx.league_avgs.lookup(batter.handedness, pitcher_hand)
        if avg is None:
            continue
        rows = [
            r for r in batter.statcast_pa_season if r.get("p_throws") == pitcher_hand
        ]
        contacts, swings = _zone_contact_per_batter(rows)
        if swings == 0:
            batter_rate = avg.zone_contact_pct
        elif swings < K_PRIOR_LINEUP_BATTER:
            batter_rate = (contacts + avg.zone_contact_pct * K_PRIOR_LINEUP_BATTER) / (
                swings + K_PRIOR_LINEUP_BATTER
            )
        else:
            batter_rate = contacts / swings
        w = BATTING_ORDER_PA_WEIGHTS.get(batter.batting_order, 4.0)
        weights.append(w)
        weighted.append(batter_rate * w)

    if not weights:
        return None, "no batters with resolvable handedness"
    return sum(weighted) / sum(weights), None


def _chase_rate_per_batter(rows: Iterable[dict]) -> tuple[int, int]:
    """Return (swings_at_ooz, pitches_ooz)."""
    swings = 0
    ooz_pitches = 0
    for r in rows:
        zone = r.get("zone")
        try:
            z = int(zone) if zone is not None else None
        except (TypeError, ValueError):
            continue
        if z not in OUT_OF_ZONE_ZONES:
            continue
        ooz_pitches += 1
        if r.get("description") in SWING_DESCRIPTIONS:
            swings += 1
    return swings, ooz_pitches


def lineup_chase_rate(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    if not bundle.opposing_lineup.lineup_posted:
        return None, "lineup not posted"
    pitcher_hand = bundle.pitcher.handedness
    if pitcher_hand not in ("L", "R"):
        return None, "pitcher handedness unknown"

    weights, weighted = [], []
    for batter in bundle.opposing_lineup.batters:
        avg = ctx.league_avgs.lookup(batter.handedness, pitcher_hand)
        if avg is None:
            continue
        rows = [
            r for r in batter.statcast_pa_season if r.get("p_throws") == pitcher_hand
        ]
        swings, ooz = _chase_rate_per_batter(rows)
        if ooz == 0:
            batter_rate = avg.chase_rate
        elif ooz < K_PRIOR_LINEUP_BATTER:
            batter_rate = (swings + avg.chase_rate * K_PRIOR_LINEUP_BATTER) / (
                ooz + K_PRIOR_LINEUP_BATTER
            )
        else:
            batter_rate = swings / ooz
        w = BATTING_ORDER_PA_WEIGHTS.get(batter.batting_order, 4.0)
        weights.append(w)
        weighted.append(batter_rate * w)

    if not weights:
        return None, "no batters with resolvable handedness"
    return sum(weighted) / sum(weights), None


# ---- Park / umpire ---------------------------------------------------------


def park_k_factor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    factor = ctx.park_k_factors.get(bundle.game_context.venue_id)
    if factor is None:
        return None, f"venue_id={bundle.game_context.venue_id} not in park_k_factors"
    return factor, None


def umpire_k_zone_factor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    return ctx.umpire_k_factors.get(bundle.game_context.umpire_id), None


# ---- Composition -----------------------------------------------------------


def _logit(p: float) -> float:
    eps = 1e-9
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _league_anchor_k_pct(bundle: ProjectionBundle, ctx: ProjectionContext) -> float:
    """Leaguewide K% vs the pitcher's hand (or overall if unknown)."""
    hand = bundle.pitcher.handedness
    if hand == "R":
        # Average across batter-hands; use league mix (rough 60/40 R/L)
        return 0.6 * ctx.league_avgs.r_vs_r.k_pct + 0.4 * ctx.league_avgs.l_vs_r.k_pct
    if hand == "L":
        return 0.6 * ctx.league_avgs.r_vs_l.k_pct + 0.4 * ctx.league_avgs.l_vs_l.k_pct
    # Fallback: overall average across all four splits.
    return (
        ctx.league_avgs.r_vs_r.k_pct + ctx.league_avgs.r_vs_l.k_pct
        + ctx.league_avgs.l_vs_r.k_pct + ctx.league_avgs.l_vs_l.k_pct
    ) / 4


def _lineup_zone_contact_anchor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> float:
    hand = bundle.pitcher.handedness
    if hand == "R":
        return 0.6 * ctx.league_avgs.r_vs_r.zone_contact_pct + 0.4 * ctx.league_avgs.l_vs_r.zone_contact_pct
    if hand == "L":
        return 0.6 * ctx.league_avgs.r_vs_l.zone_contact_pct + 0.4 * ctx.league_avgs.l_vs_l.zone_contact_pct
    return (
        ctx.league_avgs.r_vs_r.zone_contact_pct + ctx.league_avgs.r_vs_l.zone_contact_pct
        + ctx.league_avgs.l_vs_r.zone_contact_pct + ctx.league_avgs.l_vs_l.zone_contact_pct
    ) / 4


def _lineup_chase_anchor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> float:
    hand = bundle.pitcher.handedness
    if hand == "R":
        return 0.6 * ctx.league_avgs.r_vs_r.chase_rate + 0.4 * ctx.league_avgs.l_vs_r.chase_rate
    if hand == "L":
        return 0.6 * ctx.league_avgs.r_vs_l.chase_rate + 0.4 * ctx.league_avgs.l_vs_l.chase_rate
    return (
        ctx.league_avgs.r_vs_r.chase_rate + ctx.league_avgs.r_vs_l.chase_rate
        + ctx.league_avgs.l_vs_r.chase_rate + ctx.league_avgs.l_vs_l.chase_rate
    ) / 4


@dataclass(frozen=True)
class PKPAResult:
    p_k_pa: float | None
    p_k_pa_raw: float | None  # pre-clip
    used_features: dict
    skipped: bool
    skip_reason: str | None


def compute_p_k_pa(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> PKPAResult:
    used: dict = {}

    # Required: pitcher_k_pct_season_shrunk
    base_val, base_miss = pitcher_k_pct_season_shrunk(bundle, ctx)
    if base_val is None:
        return PKPAResult(
            p_k_pa=None, p_k_pa_raw=None, used_features={},
            skipped=True, skip_reason=f"pitcher_k_pct_season_shrunk: {base_miss}",
        )
    used["pitcher_k_pct_season_shrunk"] = base_val

    # Required: lineup_k_pct_vs_hand
    lkp_val, lkp_miss = lineup_k_pct_vs_hand(bundle, ctx)
    if lkp_val is None:
        return PKPAResult(
            p_k_pa=None, p_k_pa_raw=None, used_features=used,
            skipped=True, skip_reason=f"lineup_k_pct_vs_hand: {lkp_miss}",
        )
    used["lineup_k_pct_vs_hand"] = lkp_val

    # Required: park_k_factor
    pkf_val, pkf_miss = park_k_factor(bundle, ctx)
    if pkf_val is None:
        return PKPAResult(
            p_k_pa=None, p_k_pa_raw=None, used_features=used,
            skipped=True, skip_reason=f"park_k_factor: {pkf_miss}",
        )
    used["park_k_factor"] = pkf_val

    # Optional features
    k30_val, _ = pitcher_k_pct_30d_blended(bundle, ctx)
    if k30_val is not None:
        used["pitcher_k_pct_30d_blended"] = k30_val
    csw_val, _ = pitcher_csw_pct_30d(bundle, ctx)
    if csw_val is not None:
        used["pitcher_csw_pct_30d"] = csw_val
    cw_val, _ = pitcher_chase_whiff_pct_30d(bundle, ctx)
    if cw_val is not None:
        used["pitcher_chase_whiff_pct_30d"] = cw_val
    velo_val, _ = pitcher_velocity_trend_3starts(bundle, ctx)
    if velo_val is not None:
        used["pitcher_velocity_trend_3starts"] = velo_val
    pa_val, _ = pitcher_putaway_pitch_concentration(bundle, ctx)
    if pa_val is not None:
        used["pitcher_putaway_pitch_concentration"] = pa_val
    zc_val, _ = lineup_zone_contact_pct(bundle, ctx)
    if zc_val is not None:
        used["lineup_zone_contact_pct"] = zc_val
    chase_val, _ = lineup_chase_rate(bundle, ctx)
    if chase_val is not None:
        used["lineup_chase_rate"] = chase_val
    ump_val, _ = umpire_k_zone_factor(bundle, ctx)
    if ump_val is not None:
        used["umpire_k_zone_factor"] = ump_val

    # Compose in log-odds space.
    # Blend pitcher baseline: 70% season, 30% 30d when both present.
    if "pitcher_k_pct_30d_blended" in used:
        baseline = 0.7 * base_val + 0.3 * used["pitcher_k_pct_30d_blended"]
    else:
        baseline = base_val
    logit_p = _logit(baseline)

    # Lineup K% vs hand: log ratio against league anchor (handedness-adjusted).
    anchor_k = _league_anchor_k_pct(bundle, ctx)
    if anchor_k > 0:
        logit_p += math.log(max(1e-6, used["lineup_k_pct_vs_hand"] / anchor_k))

    # Park K factor: already a multiplier centered at 1.0 -> log directly.
    logit_p += math.log(max(1e-6, used["park_k_factor"]))

    # Umpire K factor.
    if "umpire_k_zone_factor" in used:
        logit_p += math.log(max(1e-6, used["umpire_k_zone_factor"]))

    # CSW: linear delta above anchor.
    if "pitcher_csw_pct_30d" in used:
        logit_p += COEFF_CSW * (used["pitcher_csw_pct_30d"] - ANCHOR_CSW)

    # Chase-whiff: anchored at the same CSW reference for simplicity.
    if "pitcher_chase_whiff_pct_30d" in used:
        # league chase-whiff is ~22%; treat as ANCHOR_CSW - 6pp
        logit_p += COEFF_CHASE_WHIFF * (used["pitcher_chase_whiff_pct_30d"] - 0.22)

    # Velocity trend: per +1 Z.
    if "pitcher_velocity_trend_3starts" in used:
        logit_p += COEFF_VELOCITY_Z * used["pitcher_velocity_trend_3starts"]

    # Putaway concentration.
    if "pitcher_putaway_pitch_concentration" in used:
        logit_p += COEFF_PUTAWAY * (
            used["pitcher_putaway_pitch_concentration"] - ANCHOR_PUTAWAY
        )

    # Lineup zone contact (NEGATIVE coefficient: higher contact -> lower K).
    if "lineup_zone_contact_pct" in used:
        anchor_zc = _lineup_zone_contact_anchor(bundle, ctx)
        logit_p += COEFF_ZONE_CONTACT * (used["lineup_zone_contact_pct"] - anchor_zc)

    # Lineup chase rate.
    if "lineup_chase_rate" in used:
        anchor_chase = _lineup_chase_anchor(bundle, ctx)
        logit_p += COEFF_CHASE_RATE * (used["lineup_chase_rate"] - anchor_chase)

    raw = _sigmoid(logit_p)
    p_clipped = max(P_K_PA_FLOOR, min(P_K_PA_CEIL, raw))

    return PKPAResult(
        p_k_pa=p_clipped,
        p_k_pa_raw=raw,
        used_features=used,
        skipped=False,
        skip_reason=None,
    )
