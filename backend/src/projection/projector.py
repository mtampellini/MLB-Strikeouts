"""Phase 3c: E[K] = E[BF] x P(K|PA) composition + projector orchestration.

The projector is the entry point. It enforces:
1. Pre-feature hard filters (career IP, season IP, lineup posted, opener skip).
2. E[BF] computation (Phase 3b).
3. Post-BF hard filter (projected_BF < 12 on the RAW pre-clip value).
4. P(K|PA) computation (Phase 3c features + log-odds composition).
5. Final E[K] = clipped E[BF] x clipped P(K|PA).

BOOK-AGNOSTIC. Imports from :mod:`features_bf`, :mod:`features_kpa`,
:mod:`hard_filters`, and :mod:`inputs` only — never from src.picks.
"""
from __future__ import annotations

from dataclasses import dataclass

from .features_bf import BF_FLOOR, EBFResult, compute_e_bf
from .features_kpa import PKPAResult, compute_p_k_pa
from .hard_filters import check_pre_feature_filters
from .inputs import BundleMetadata, ProjectionBundle, ProjectionContext


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

    def to_dict(self) -> dict:
        return {
            "e_bf": self.e_bf,
            "p_k_pa": self.p_k_pa,
            "e_k": self.e_k,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "features_used_bf": dict(self.features_used_bf),
            "features_used_kpa": dict(self.features_used_kpa),
            "bundle_metadata": self.bundle_metadata.to_dict(),
        }


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


def project(
    bundle: ProjectionBundle, ctx: ProjectionContext
) -> ProjectionResult:
    """Run the full projection pipeline. Returns ProjectionResult."""

    pre_skip = check_pre_feature_filters(bundle)
    if pre_skip:
        return _skipped(bundle, pre_skip)

    bf = compute_e_bf(bundle, ctx)
    if bf.skipped:
        return _skipped(
            bundle, f"e_bf: {bf.skip_reason}", features_used_bf=bf.used_features,
        )

    # Post-BF hard filter on the RAW (pre-clip) value: if the additive model
    # thinks the pitcher will face fewer than 12 batters, that's a bullpen
    # game / opener / pitcher we shouldn't price.
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
    )
