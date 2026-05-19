"""Phase 3c: P(K|PA) feature builders + log-odds composition.

Same shape as :mod:`features_bf`: each builder is a pure function returning
``(value, missing_reason)``. The composition uses log-odds (not raw
multiplicative) so the result is guaranteed to stay in (0, 1), then clipped
to [P_K_PA_FLOOR, P_K_PA_CEIL].

Phase 3-v2c-iii rewrite
-----------------------
Drops the bootstrap-failing / sign-stable-wrong features from the prior
overnight fit (pitcher_k_pct_30d_delta, pitcher_csw_pct_30d_delta,
pitcher_putaway_pct_delta, lineup_chase_delta) and the aggregated lineup
features (lineup_k_pct_vs_hand, lineup_zone_contact_pct, lineup_chase_rate)
that are now computed per-batter by :mod:`features_per_batter`.

Promotes pitcher_csw_pct_season to PRIMARY K-skill feature. Adds
pitcher_archetype passthrough for downstream TTO multiplier lookup.
Park K factor now uses bundle.park_k_factors_by_hand with pitcher
handedness selection.

Survivors (bootstrap-stable in the prior fit):
- pitcher_k_pct_delta (1.00) — kept as SECONDARY K-rate signal
- pitcher_velocity_trend_z (97%) — kept
- pitcher_chase_whiff_pct_30d_delta (82%) — kept under observation
- league_k_pct_vs_hand (1.00) — anchor

Calibration coefficients remain first-pass placeholders. Phase 4c-v2 will
re-derive them from the regression of observed K rates against the new
feature values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .features_bf import (
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
K_PRIOR_K_PCT_SEASON = 80      # PA strength of prior_year prior for K%
K_PRIOR_CSW_SEASON = 400       # pitches; prior strength for season CSW%
MIN_PITCHES_CSW_SEASON = 200   # season pitches needed to trust observed CSW%
MIN_PITCHES_CSW_PRIOR = 800    # prior-year pitches required for prior-only fallback

# Log-odds coefficients (placeholders — replaced by Phase 4c-v2 fit)
COEFF_K_PCT_DELTA = 4.0        # primary K-rate delta (per unit raw rate)
COEFF_CSW_SEASON_DELTA = 4.0   # primary CSW delta (per unit raw rate)
COEFF_VELOCITY_Z = 0.04        # per +1 Z fastball velo
COEFF_CHASE_WHIFF = 0.4        # per unit raw rate above league anchor

# League CSW% used as a fallback anchor when league_averages.csw_pct is
# missing. Roughly the current MLB league average; close enough for placeholder
# composition until Phase 4a re-derives the actual csw_pct field per split.
LEAGUE_CSW_ANCHOR_FALLBACK = 0.28


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


def _csw_count(rows: Iterable[dict]) -> tuple[int, int]:
    pitches = 0
    csw = 0
    for r in rows:
        pitches += 1
        if r.get("description") in CSW_DESCRIPTIONS:
            csw += 1
    return csw, pitches


def pitcher_csw_pct_season(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Pitcher's season-to-date CSW% shrunk to prior-year.

    - If season pitches >= MIN_PITCHES_CSW_SEASON (200): shrink to prior-year
      CSW% with k_prior=K_PRIOR_CSW_SEASON (400).
    - Elif prior-year pitches >= MIN_PITCHES_CSW_PRIOR (800): use prior-year
      directly (mark with no missing_reason — prior is reliable on its own).
    - Else: return None with "insufficient CSW sample".
    """
    p = bundle.pitcher
    csw_s, pitches_s = _csw_count(p.statcast_pitches_season)
    csw_p, pitches_p = _csw_count(p.statcast_pitches_prior_year)

    if pitches_s >= MIN_PITCHES_CSW_SEASON:
        season_rate = csw_s / pitches_s
        prior_rate = csw_p / pitches_p if pitches_p > 0 else None
        if prior_rate is None:
            return season_rate, None
        k_prior = K_PRIOR_CSW_SEASON
        shrunk = (csw_s + prior_rate * k_prior) / (pitches_s + k_prior)
        return shrunk, None

    if pitches_p >= MIN_PITCHES_CSW_PRIOR:
        return csw_p / pitches_p, None

    return None, "insufficient CSW sample"


def _league_csw_anchor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> float | None:
    """League CSW% for the pitcher's hand, averaged across batter-hands by
    league mix (~60% R / 40% L). Falls back to LEAGUE_CSW_ANCHOR_FALLBACK
    when league_averages lacks csw_pct (pre-Phase-3-v2c-iii files)."""
    hand = bundle.pitcher.handedness
    if hand not in ("L", "R"):
        avgs = (ctx.league_avgs.r_vs_r, ctx.league_avgs.r_vs_l,
                ctx.league_avgs.l_vs_r, ctx.league_avgs.l_vs_l)
    elif hand == "R":
        avgs = (ctx.league_avgs.r_vs_r, ctx.league_avgs.l_vs_r)
        weights = (0.6, 0.4)
    else:
        avgs = (ctx.league_avgs.r_vs_l, ctx.league_avgs.l_vs_l)
        weights = (0.6, 0.4)

    csw_vals = [a.csw_pct for a in avgs if a.csw_pct is not None]
    if not csw_vals:
        return None
    if hand in ("L", "R") and len(csw_vals) == 2:
        return csw_vals[0] * weights[0] + csw_vals[1] * weights[1]
    return sum(csw_vals) / len(csw_vals)


def pitcher_csw_pct_season_delta(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Pitcher CSW% minus league CSW% (handedness-mixed). The delta
    re-parameterization removes the level-vs-individual collinearity that
    plagued the prior fit."""
    csw, csw_miss = pitcher_csw_pct_season(bundle, ctx)
    if csw is None:
        return None, csw_miss
    league = _league_csw_anchor(bundle, ctx)
    if league is None:
        return None, "league CSW not available"
    return csw - league, None


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
    if chases_30d >= 100:
        return raw_30d, None
    whiffs_s, chases_s = _chase_whiff_counts(p.statcast_pitches_season)
    if chases_s == 0:
        return raw_30d, None
    season_rate = whiffs_s / chases_s
    k_prior = 100
    shrunk = (whiffs_30d + season_rate * k_prior) / (chases_30d + k_prior)
    return shrunk, None


def pitcher_velocity_trend_3starts(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """Z-score: (last 3 starts' fastball velo) - (season fastball mean) / season SD."""
    p = bundle.pitcher
    fastballs_recent = _fastball_velos_last_n_starts(p.statcast_pitches_30d, n=3)
    if len(fastballs_recent) < 30:
        return None, f"only {len(fastballs_recent)} fastballs in last 3 starts"

    fb_season = [r["release_speed"] for r in p.statcast_pitches_season
                 if r.get("pitch_type") in FASTBALL_TYPES
                 and isinstance(r.get("release_speed"), (int, float))]
    if not fb_season:
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
    last_n_games = sorted(by_game.keys())[-n:]
    out: list[float] = []
    for g in last_n_games:
        out.extend(by_game[g])
    return out


def pitcher_archetype_feature(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[str | None, str | None]:
    """Passthrough of the pitcher's archetype for downstream TTO multiplier
    lookup. Reads :attr:`ProjectionBundle.pitcher_archetype`.

    Returns the archetype name as a string. Not a regression feature — the
    projector consumes it to pick a TTO multiplier per PA, not as an input
    to the logistic composition.

    Missing reasons:
    - "no pitcher_archetype in bundle" when the contract field is None
    - "archetype unknown" when the field is the unknown sentinel
    """
    pa = bundle.pitcher_archetype
    if pa is None:
        return None, "no pitcher_archetype in bundle"
    if pa.archetype == "unknown":
        return None, "archetype unknown"
    return pa.archetype, None


# ---- Park / umpire ---------------------------------------------------------


def log_park_k_factor_by_hand(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    """log(park K factor) using the L/R-split factor selected by pitcher hand.

    - bundle.park_k_factors_by_hand present: pick factor_lhp / factor_rhp /
      factor_combined based on bundle.pitcher.handedness.
    - bundle.park_k_factors_by_hand None (legacy bundle): fall back to the
      ctx.park_k_factors single-factor lookup. This is the only remaining
      place that reads the old shape, for backward compat with bundles that
      predate Phase 3-v2c-i.
    """
    pk = bundle.park_k_factors_by_hand
    if pk is not None:
        factor = pk.for_pitcher_hand(bundle.pitcher.handedness)
        return math.log(max(1e-6, factor)), None
    # Legacy fallback path.
    factor = ctx.park_k_factors.get(bundle.game_context.venue_id)
    if factor is None:
        return None, (
            f"venue_id={bundle.game_context.venue_id} not in park_k_factors "
            f"and bundle has no park_k_factors_by_hand"
        )
    return math.log(max(1e-6, factor)), None


def umpire_k_zone_factor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> tuple[float | None, str | None]:
    return ctx.umpire_k_factors.get(bundle.game_context.umpire_id), None


# ---- Anchors (kept for Phase 4c fit script consumption) --------------------
#
# These helpers are imported by scripts/fit_feature_coefficients.py to write
# league baseline columns into the design matrix (so reparameterize_kpa can
# center deltas). They remain accessible even though the public lineup_*
# builders that originally used them have been removed in Phase 3-v2c-iii.
# When Phase 4c-v2 runs, the design matrix will simply have fewer columns
# (no aggregated lineup features), and reparameterize_kpa will skip the
# obsolete delta entries.


def _league_anchor_k_pct(bundle: ProjectionBundle, ctx: ProjectionContext) -> float:
    """Leaguewide K% vs the pitcher's hand (or overall if unknown)."""
    hand = bundle.pitcher.handedness
    if hand == "R":
        return 0.6 * ctx.league_avgs.r_vs_r.k_pct + 0.4 * ctx.league_avgs.l_vs_r.k_pct
    if hand == "L":
        return 0.6 * ctx.league_avgs.r_vs_l.k_pct + 0.4 * ctx.league_avgs.l_vs_l.k_pct
    return (
        ctx.league_avgs.r_vs_r.k_pct + ctx.league_avgs.r_vs_l.k_pct
        + ctx.league_avgs.l_vs_r.k_pct + ctx.league_avgs.l_vs_l.k_pct
    ) / 4


def _lineup_zone_contact_anchor(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> float:
    """Legacy helper retained for the Phase 4c fit script. Returns league
    zone-contact% by pitcher hand (mixed batter hand)."""
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
    """Legacy helper retained for the Phase 4c fit script."""
    hand = bundle.pitcher.handedness
    if hand == "R":
        return 0.6 * ctx.league_avgs.r_vs_r.chase_rate + 0.4 * ctx.league_avgs.l_vs_r.chase_rate
    if hand == "L":
        return 0.6 * ctx.league_avgs.r_vs_l.chase_rate + 0.4 * ctx.league_avgs.l_vs_l.chase_rate
    return (
        ctx.league_avgs.r_vs_r.chase_rate + ctx.league_avgs.r_vs_l.chase_rate
        + ctx.league_avgs.l_vs_r.chase_rate + ctx.league_avgs.l_vs_l.chase_rate
    ) / 4


# ---- Composition -----------------------------------------------------------


def _logit(p: float) -> float:
    eps = 1e-9
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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
    """Compose P(K|PA) from the Phase 3-v2c-iii feature set.

    Baseline: league K% vs pitcher hand. Each feature contributes an
    additive log-odds shift. Placeholder coefficients until Phase 4c-v2.

    Required: pitcher_csw_pct_season (PRIMARY K-skill),
    league_k_pct_vs_hand (anchor), log_park_k_factor_by_hand.
    Optional: pitcher_k_pct_season_shrunk (becomes pitcher_k_pct_delta),
    pitcher_velocity_trend_3starts, pitcher_chase_whiff_pct_30d,
    umpire_k_zone_factor.
    Side-channel: pitcher_archetype_feature (carried in used_features for
    the projector; does NOT contribute to the logit composition).
    """
    used: dict = {}

    # Anchor: league K% vs hand
    league_k = _league_anchor_k_pct(bundle, ctx)
    used["league_k_pct_vs_hand"] = league_k

    # Required: pitcher_csw_pct_season — primary K-skill
    csw_val, csw_miss = pitcher_csw_pct_season(bundle, ctx)
    if csw_val is None:
        return PKPAResult(
            p_k_pa=None, p_k_pa_raw=None, used_features=used,
            skipped=True, skip_reason=f"pitcher_csw_pct_season: {csw_miss}",
        )
    used["pitcher_csw_pct_season"] = csw_val

    csw_delta_val, csw_delta_miss = pitcher_csw_pct_season_delta(bundle, ctx)
    if csw_delta_val is not None:
        used["pitcher_csw_pct_season_delta"] = csw_delta_val

    # Required: log_park_k_factor_by_hand
    lpk_val, lpk_miss = log_park_k_factor_by_hand(bundle, ctx)
    if lpk_val is None:
        return PKPAResult(
            p_k_pa=None, p_k_pa_raw=None, used_features=used,
            skipped=True, skip_reason=f"log_park_k_factor_by_hand: {lpk_miss}",
        )
    used["log_park_k_factor_by_hand"] = lpk_val

    # Optional: pitcher_k_pct_delta (secondary K-rate signal)
    k_season, _ = pitcher_k_pct_season_shrunk(bundle, ctx)
    if k_season is not None:
        used["pitcher_k_pct_season_shrunk"] = k_season
        used["pitcher_k_pct_delta"] = k_season - league_k

    # Optional: velocity Z trend
    velo_val, _ = pitcher_velocity_trend_3starts(bundle, ctx)
    if velo_val is not None:
        used["pitcher_velocity_trend_z"] = velo_val

    # Optional: chase-whiff delta from a 22% anchor (league chase-whiff is
    # ~22%; we'd use league_avgs here once Phase 4a derives it)
    cw_val, _ = pitcher_chase_whiff_pct_30d(bundle, ctx)
    if cw_val is not None:
        used["pitcher_chase_whiff_pct_30d_delta"] = cw_val - 0.22

    # Optional: umpire K factor
    ump_val, _ = umpire_k_zone_factor(bundle, ctx)
    if ump_val is not None:
        used["umpire_k_zone_factor"] = ump_val

    # Side-channel: pitcher archetype (for projector TTO lookup; not in logit)
    arch_val, _ = pitcher_archetype_feature(bundle, ctx)
    if arch_val is not None:
        used["pitcher_archetype"] = arch_val

    # ---- Compose in log-odds space ----
    logit_p = _logit(league_k)

    if "pitcher_k_pct_delta" in used:
        logit_p += COEFF_K_PCT_DELTA * used["pitcher_k_pct_delta"]

    if "pitcher_csw_pct_season_delta" in used:
        logit_p += COEFF_CSW_SEASON_DELTA * used["pitcher_csw_pct_season_delta"]

    logit_p += used["log_park_k_factor_by_hand"]

    if "umpire_k_zone_factor" in used:
        logit_p += math.log(max(1e-6, used["umpire_k_zone_factor"]))

    if "pitcher_velocity_trend_z" in used:
        logit_p += COEFF_VELOCITY_Z * used["pitcher_velocity_trend_z"]

    if "pitcher_chase_whiff_pct_30d_delta" in used:
        logit_p += COEFF_CHASE_WHIFF * used["pitcher_chase_whiff_pct_30d_delta"]

    raw = _sigmoid(logit_p)
    p_clipped = max(P_K_PA_FLOOR, min(P_K_PA_CEIL, raw))

    return PKPAResult(
        p_k_pa=p_clipped,
        p_k_pa_raw=raw,
        used_features=used,
        skipped=False,
        skip_reason=None,
    )
