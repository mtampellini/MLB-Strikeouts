"""Phase 3-v2c-iv: per-batter projector for E[K].

The projector iterates over (batter, TTO) cells when the bundle carries the
Phase 3-v2c-i additive fields (pa_distribution + tto_multipliers); otherwise
it falls back to the legacy aggregated E[BF] * P(K|PA) composition.

Per-cell K probability uses log5 with the league rate as the baseline, then
applies the (archetype, TTO) multiplier from Phase 3-v2b. Per-batter K%
fallbacks: if a batter has insufficient sample for the per-batter builder
to produce an individual rate, the cell uses the league average vs hand
(no batter signal, but cell still contributes).

The per_batter_breakdown surfaces every (batter, TTO) cell's contribution
for debug/visibility. Sum of per_batter_breakdown[*]["expected_k_total"]
equals the overall E[K] within float epsilon.

BOOK-AGNOSTIC. Imports from features_*, hard_filters, and inputs only;
never from src.picks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .features_bf import BF_FLOOR, EBFResult, compute_e_bf
from .features_kpa import PKPAResult, compute_p_k_pa
from .features_per_batter import (
    SENTINEL_NO_LEAGUE_AVGS_KEY,
    per_batter_k_pct_vs_hand,
)
from .hard_filters import check_pre_feature_filters
from .inputs import (
    BundleMetadata,
    HandednessAverages,
    PADistribution,
    ProjectionBundle,
    ProjectionContext,
    TTOMultipliers,
)

# Clip bounds for per-cell K probability (consistent with features_kpa).
P_K_CELL_FLOOR = 0.05
P_K_CELL_CEIL = 0.55

# Archetype to use when bundle.pitcher_archetype is None / "unknown".
# Balanced has TTO multipliers closest to league_wide, minimizing distortion
# when the true archetype is unknown.
ARCHETYPE_UNKNOWN_FALLBACK = "Balanced"

# Default league K% for the final "no signal at all" path (e.g., switch-hitter
# vs same hand without league_averages). Roughly the current MLB league mean.
LEAGUE_K_PCT_HARD_FALLBACK = 0.22


@dataclass(frozen=True)
class ProjectionResult:
    e_bf: float | None
    p_k_pa: float | None
    e_k: float | None
    skipped: bool
    skip_reason: str | None
    features_used_bf: dict
    features_used_kpa: dict
    bundle_metadata: BundleMetadata
    # Phase 3-v2c-iv additive fields (default values preserve backward compat).
    projection_method: str = "legacy_aggregated"
    per_batter_breakdown: dict | None = None
    archetype_used: str | None = None
    pa_distribution_bf_used: int | None = None

    def to_dict(self) -> dict:
        out: dict = {
            "e_bf": self.e_bf,
            "p_k_pa": self.p_k_pa,
            "e_k": self.e_k,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "features_used_bf": dict(self.features_used_bf),
            "features_used_kpa": dict(self.features_used_kpa),
            "bundle_metadata": self.bundle_metadata.to_dict(),
            "projection_method": self.projection_method,
        }
        if self.per_batter_breakdown is not None:
            # JSON keys must be strings; per_batter_breakdown is keyed by
            # batter mlbam_id (int).
            out["per_batter_breakdown"] = {
                str(bid): entry for bid, entry in self.per_batter_breakdown.items()
            }
        if self.archetype_used is not None:
            out["archetype_used"] = self.archetype_used
        if self.pa_distribution_bf_used is not None:
            out["pa_distribution_bf_used"] = self.pa_distribution_bf_used
        return out


# ---- Internal helpers ------------------------------------------------------


def _skipped(
    bundle: ProjectionBundle,
    reason: str,
    *,
    features_used_bf: dict | None = None,
    features_used_kpa: dict | None = None,
) -> ProjectionResult:
    return ProjectionResult(
        e_bf=None,
        p_k_pa=None,
        e_k=None,
        skipped=True,
        skip_reason=reason,
        features_used_bf=features_used_bf or {},
        features_used_kpa=features_used_kpa or {},
        bundle_metadata=bundle.metadata,
    )


def _resolve_archetype(bundle: ProjectionBundle) -> str:
    """Return the archetype string to use for TTO multiplier lookup.

    The fallback chain handled by the loader (current_season -> rolling_30d ->
    previous_season -> 'unknown') is implicit in :attr:`bundle.pitcher_archetype`.
    The final 'unknown' sentinel is mapped to :data:`ARCHETYPE_UNKNOWN_FALLBACK`
    here so the rest of the projector never sees 'unknown'.
    """
    pa = bundle.pitcher_archetype
    if pa is None or pa.archetype == "unknown":
        return ARCHETYPE_UNKNOWN_FALLBACK
    return pa.archetype


def _compute_log5(batter_k: float, pitcher_k: float, league_k: float) -> float:
    """Log5: project K rate when a batter with rate b faces a pitcher with
    rate p, given league baseline L.

        log5 = (b*p/L) / ( (b*p/L) + (1-b)*(1-p)/(1-L) )

    Degenerate guards keep us out of NaN territory when league_k is at 0 or 1.
    """
    if league_k <= 0 or league_k >= 1:
        return (batter_k + pitcher_k) / 2.0
    bp_l = batter_k * pitcher_k / league_k
    not_bp_l = (1 - batter_k) * (1 - pitcher_k) / (1 - league_k)
    denom = bp_l + not_bp_l
    if denom <= 0:
        return 0.0
    return bp_l / denom


def _effective_batter_hand(
    batter, pitcher_hand: str
) -> str | None:
    """L/R/S resolver: S vs R = L; S vs L = R. None for unknown."""
    h = batter.handedness
    if not h:
        return None
    if h == "S":
        return "L" if pitcher_hand == "R" else "R"
    return h if h in ("L", "R") else None


def _league_k_pct_vs_hand(
    ctx: ProjectionContext, batter_hand: str | None, pitcher_hand: str | None,
) -> float | None:
    cell = ctx.league_avgs.lookup(batter_hand, pitcher_hand)
    return cell.k_pct if cell is not None else None


def _get_tto_multiplier(
    tto_table: TTOMultipliers, archetype: str, tto: int,
) -> float:
    """Lookup (archetype, tto) multiplier. Fall back to league_wide when
    the archetype is missing from the table or the cell is empty."""
    if archetype in tto_table.by_archetype:
        cell = tto_table.by_archetype[archetype].get(tto)
        if cell is not None:
            return cell.multiplier
    return tto_table.league_wide[tto].multiplier


def _get_pa_distribution_for_bf(
    pa_dist: PADistribution, e_bf: float,
) -> tuple[dict, int]:
    """Round e_bf to nearest int, clamp to PA distribution's observed range,
    return (by_slot_dict, bf_used). by_slot_dict maps slot (int 1..9) to a
    PADistributionCell.
    """
    bf_int = int(round(e_bf))
    lo, hi = pa_dist.bf_range_observed
    bf_used = max(lo, min(hi, bf_int))
    # PADistribution.by_bf keys are ints (we cast at load time); fall back
    # to scanning if exact match is missing (shouldn't happen).
    if bf_used in pa_dist.by_bf:
        return pa_dist.by_bf[bf_used], bf_used
    # Defensive: pick the closest available
    nearest = min(pa_dist.by_bf.keys(), key=lambda k: abs(k - bf_used))
    return pa_dist.by_bf[nearest], nearest


def _compute_p_k_for_cell(
    bundle: ProjectionBundle,
    ctx: ProjectionContext,
    batter,
    tto: int,
    *,
    per_batter_k: dict,
    pitcher_k_rate: float,
    tto_table: TTOMultipliers,
    archetype: str,
) -> tuple[float, dict]:
    """Return (p_k_in_cell, debug_info).

    p_k_in_cell = clip(log5(batter_k, pitcher_k, league_k) * tto_multiplier).

    When the batter has no individual rate (per_batter_k entry is None),
    use the league average vs hand as the batter rate (cell still
    contributes to E[K]).
    """
    pitcher_hand = bundle.pitcher.handedness
    eff_hand = _effective_batter_hand(batter, pitcher_hand)
    league_k = _league_k_pct_vs_hand(ctx, eff_hand, pitcher_hand)

    # Resolve batter K rate, with fallback to league.
    entry = per_batter_k.get(batter.mlbam_id)
    if entry is not None and entry[0] is not None:
        batter_k_val = entry[0]
        k_source = "individual"
    else:
        batter_k_val = league_k if league_k is not None else LEAGUE_K_PCT_HARD_FALLBACK
        k_source = "league_fallback"

    league_for_log5 = (
        league_k if league_k is not None else LEAGUE_K_PCT_HARD_FALLBACK
    )
    log5_prior = _compute_log5(batter_k_val, pitcher_k_rate, league_for_log5)
    tto_mult = _get_tto_multiplier(tto_table, archetype, tto)
    p_k_cell = log5_prior * tto_mult
    p_k_cell = max(P_K_CELL_FLOOR, min(P_K_CELL_CEIL, p_k_cell))

    return p_k_cell, {
        "log5_prior": log5_prior,
        "tto_mult": tto_mult,
        "league_k": league_k,
        "batter_k_val": batter_k_val,
        "k_source": k_source,
    }


def _project_per_batter(
    bundle: ProjectionBundle,
    ctx: ProjectionContext,
    e_bf: float,
    pitcher_k_rate: float,
    archetype: str,
) -> tuple[float, dict, int]:
    """Run the per-(batter, TTO) projection loop.

    Returns (e_k_total, per_batter_breakdown, bf_used).
    """
    pa_dist = bundle.pa_distribution
    tto_table = bundle.tto_multipliers
    assert pa_dist is not None and tto_table is not None, (
        "_project_per_batter requires bundle.pa_distribution and "
        "bundle.tto_multipliers — caller should route to legacy"
    )

    by_slot_dist, bf_used = _get_pa_distribution_for_bf(pa_dist, e_bf)
    per_batter_k = per_batter_k_pct_vs_hand(bundle)
    # If per_batter_k is the sentinel ({0: (None, ...)}), the lookup at the
    # cell loop will fall back to league for every batter — that's the
    # intended degraded behavior, not a failure.

    per_batter_breakdown: dict = {}
    e_k_total = 0.0

    for batter in bundle.opposing_lineup.batters:
        slot = batter.batting_order
        cell = by_slot_dist.get(slot)
        if cell is None:
            continue
        by_tto: dict = {}
        expected_pa_total = 0.0
        expected_k_total = 0.0
        k_source_for_batter = "individual"  # may be overwritten by cell loop

        for tto in (1, 2, 3, 4):
            pa_count = getattr(cell, f"tto_{tto}")
            if pa_count <= 0:
                by_tto[tto] = {"pa": 0.0, "p_k": 0.0, "expected_k": 0.0}
                continue
            p_k, debug = _compute_p_k_for_cell(
                bundle, ctx, batter, tto,
                per_batter_k=per_batter_k,
                pitcher_k_rate=pitcher_k_rate,
                tto_table=tto_table,
                archetype=archetype,
            )
            k_source_for_batter = debug["k_source"]
            ek = pa_count * p_k
            by_tto[tto] = {
                "pa": round(pa_count, 4),
                "p_k": round(p_k, 4),
                "expected_k": round(ek, 4),
            }
            expected_pa_total += pa_count
            expected_k_total += ek

        per_batter_breakdown[batter.mlbam_id] = {
            "name": batter.name,
            "batting_order_slot": slot,
            "k_rate_source": k_source_for_batter,
            "expected_pa_total": round(expected_pa_total, 4),
            "expected_k_total": round(expected_k_total, 4),
            "by_tto": by_tto,
        }
        e_k_total += expected_k_total

    return e_k_total, per_batter_breakdown, bf_used


# ---- Public entry point ----------------------------------------------------


def project(
    bundle: ProjectionBundle,
    ctx: ProjectionContext | None = None,
) -> ProjectionResult:
    """Run the full projection pipeline.

    Routes to the per-batter path when the bundle carries pa_distribution
    and tto_multipliers (Phase 3-v2c-i contract); otherwise falls back to
    the legacy aggregated E[BF] * P(K|PA) composition.

    ``ctx`` defaults to ProjectionContext.from_default_paths() when omitted.
    """
    if ctx is None:
        ctx = ProjectionContext.from_default_paths()

    pre_skip = check_pre_feature_filters(bundle)
    if pre_skip:
        return _skipped(bundle, pre_skip)

    bf = compute_e_bf(bundle, ctx)
    if bf.skipped:
        return _skipped(
            bundle, f"e_bf: {bf.skip_reason}", features_used_bf=bf.used_features,
        )

    if bf.e_bf_raw is not None and bf.e_bf_raw < BF_FLOOR:
        return _skipped(
            bundle,
            f"hard_filter: projected_BF {bf.e_bf_raw:.2f} < {BF_FLOOR}",
            features_used_bf=bf.used_features,
        )

    kpa = compute_p_k_pa(bundle, ctx)
    if kpa.skipped:
        return _skipped(
            bundle, f"p_k_pa: {kpa.skip_reason}",
            features_used_bf=bf.used_features,
            features_used_kpa=kpa.used_features,
        )

    # Routing decision: new per-batter path requires BOTH pa_distribution
    # and tto_multipliers in the bundle AND a posted lineup.
    use_new_path = (
        bundle.pa_distribution is not None
        and bundle.tto_multipliers is not None
        and bundle.opposing_lineup.lineup_posted
        and len(bundle.opposing_lineup.batters) > 0
    )

    if use_new_path:
        archetype = _resolve_archetype(bundle)
        e_k, per_batter_breakdown, bf_used = _project_per_batter(
            bundle, ctx, bf.e_bf, kpa.p_k_pa, archetype,
        )
        # PA-weighted p_k_pa surface (so the existing p_k_pa field is still
        # meaningful in the per-batter path).
        total_pa = sum(
            entry["expected_pa_total"] for entry in per_batter_breakdown.values()
        )
        avg_p_k = e_k / total_pa if total_pa > 0 else kpa.p_k_pa

        return ProjectionResult(
            e_bf=bf.e_bf,
            p_k_pa=avg_p_k,
            e_k=e_k,
            skipped=False,
            skip_reason=None,
            features_used_bf=bf.used_features,
            features_used_kpa=kpa.used_features,
            bundle_metadata=bundle.metadata,
            projection_method="per_batter_with_tto",
            per_batter_breakdown=per_batter_breakdown,
            archetype_used=archetype,
            pa_distribution_bf_used=bf_used,
        )

    # Legacy aggregated path.
    e_k = bf.e_bf * kpa.p_k_pa
    return ProjectionResult(
        e_bf=bf.e_bf,
        p_k_pa=kpa.p_k_pa,
        e_k=e_k,
        skipped=False,
        skip_reason=None,
        features_used_bf=bf.used_features,
        features_used_kpa=kpa.used_features,
        bundle_metadata=bundle.metadata,
        projection_method="legacy_aggregated",
    )
