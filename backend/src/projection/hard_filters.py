"""Phase 3c hard filters.

Run BEFORE feature computation, on the bundle directly (with the exception of
the projected-BF check, which runs against the E[BF] result and lives in the
projector). Each filter returns a skip-reason string or None.

Per spec:
1. Career IP < 50 (estimated from total PAs in season + prior_year / 4.3).
2. (Season IP < 20) AND (Prior-year IP < 80).
3. Projected BF < 12 — implemented in :mod:`projector` because it needs the
   computed E[BF].
4. Lineup not posted.
5. Opener detected with no bulk pitcher (``opener_detection_result == "skip"``).

NOT here: "no posted K prop". That's a picks-layer concern; the projection
layer is book-agnostic.
"""
from __future__ import annotations

from .features_bf import PA_PER_INNING_APPROX, _count_pa
from .inputs import ProjectionBundle

CAREER_IP_FLOOR = 50
SEASON_IP_FLOOR = 20
PRIOR_YEAR_IP_FLOOR = 80


def check_pre_feature_filters(bundle: ProjectionBundle) -> str | None:
    """Return a skip reason if any pre-feature hard filter trips, else None."""

    p = bundle.pitcher

    season_pa = _count_pa(p.statcast_pitches_season)
    prior_pa = _count_pa(p.statcast_pitches_prior_year)
    season_ip = season_pa / PA_PER_INNING_APPROX
    prior_ip = prior_pa / PA_PER_INNING_APPROX
    career_ip = season_ip + prior_ip

    if career_ip < CAREER_IP_FLOOR:
        return (
            f"hard_filter: career IP < {CAREER_IP_FLOOR} "
            f"(estimated {career_ip:.1f} from season+prior_year PA/4.3)"
        )

    if season_ip < SEASON_IP_FLOOR and prior_ip < PRIOR_YEAR_IP_FLOOR:
        return (
            f"hard_filter: season IP < {SEASON_IP_FLOOR} AND prior_year IP < "
            f"{PRIOR_YEAR_IP_FLOOR} (estimated season={season_ip:.1f}, "
            f"prior={prior_ip:.1f})"
        )

    if not bundle.opposing_lineup.lineup_posted:
        return "hard_filter: opposing lineup not posted"

    if bundle.pitcher.opener_detection_result == "skip":
        return "hard_filter: opener detected, no bulk pitcher identified"

    return None
