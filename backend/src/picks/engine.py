"""Phase 5: picks engine orchestrator.

Glues:
- projection (E[K] per pitcher, via :func:`src.projection.projector.project`)
- NB(alpha) → P(K >= line) (probability.py)
- book odds → market_p via devig.py
- edge_pct + EV% (edge.py)
- three-tier classification + ranking (tiers.py)

OUTPUT-LAYER COUPLING. The engine imports projection internals freely; the
inverse (projection importing picks) is forbidden by the AST-walk test.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.projection.inputs import ProjectionBundle, ProjectionContext
from src.projection.projector import project

from .devig import devig_with_imputation
from .edge import edge_pct as compute_edge_pct, ev_pct as compute_ev_pct
from .probability import p_k_geq, p_k_leq
from .tiers import (
    PRIMARY_EDGE_THRESHOLD,
    SHADOW_EDGE_THRESHOLD,
    classify_pick,
    rank_primary_picks,
)

logger = logging.getLogger(__name__)

# Books we actually trade. Any other book in the snapshot is filtered out.
TRADED_BOOKS = ("fanduel", "draftkings")

# Phase 4d calibration note carried in every pick's payload (transparency).
CALIBRATION_NOTE = (
    "Phase 4d showed model over-predicts by 1.5-2.4pp at lines 5.5-9.5. "
    "True edge may be ~2pp less than reported edge_pct at high lines. "
    "No auto-correction applied; raw model probabilities published as-is."
)


@dataclass(frozen=True)
class PickResult:
    primary: list[dict]
    secondary: list[dict]
    shadow: list[dict]
    skipped: list[dict]
    metadata: dict = field(default_factory=dict)


def _pick_id(pitcher_id: int, game_pk: int, line: float, side: str, book: str) -> str:
    """Hash for dedup. Same (pitcher, game, line, side, book) -> same id
    across runs."""
    h = hashlib.sha256(
        f"{pitcher_id}|{game_pk}|{line}|{side}|{book}".encode("utf-8")
    ).hexdigest()
    return h[:16]


def _pitcher_book_lines_from_bundle(
    bundle: ProjectionBundle, book: str,
) -> dict[float, dict[str, int]]:
    """Extract ``{line: {side: american_odds}}`` for the given book from the
    bundle's market section."""
    book_section = getattr(bundle.market, book)
    if not book_section.available:
        return {}
    out: dict[float, dict[str, int]] = {}
    for ln in book_section.lines:
        out.setdefault(float(ln.line), {})[ln.side] = int(ln.price)
    return out


def _evaluate_pitcher_picks(
    bundle: ProjectionBundle, ctx: ProjectionContext, nb_alpha: float,
) -> tuple[list[dict], dict | None]:
    """Run project + per-line edge evaluation for one bundle.

    Returns ``(picks, skip_record)``. When the pitcher can't be projected at
    all, returns ``([], skip_record)``. When projection succeeded but no
    lines pass the shadow threshold, returns ``([], None)`` (no skip — just
    no picks).

    Transient pre-projection filter: if the opposing lineup has not been
    posted yet, we skip without projecting. The same pitcher gets
    re-evaluated on the next hourly run; once their lineup posts they
    generate picks normally. ``is_transient`` distinguishes this from
    permanent slate-day skips (career IP, no market data, projector
    failures).
    """
    if not bundle.opposing_lineup.lineup_posted:
        return [], {
            "pitcher_mlbam_id": bundle.metadata.pitcher_mlbam_id,
            "pitcher_name": bundle.metadata.pitcher_name,
            "game_pk": bundle.metadata.game_pk,
            "reason": "lineup_not_posted",
            "is_transient": True,
            "detail": "Awaiting official lineup posting; will re-evaluate next run.",
        }

    result = project(bundle, ctx)
    if result.skipped:
        return [], {
            "pitcher_mlbam_id": bundle.metadata.pitcher_mlbam_id,
            "pitcher_name": bundle.metadata.pitcher_name,
            "game_pk": bundle.metadata.game_pk,
            "reason": f"projector_skipped: {result.skip_reason}",
            "is_transient": False,
        }

    # Collect per-line market data per book
    book_lines = {book: _pitcher_book_lines_from_bundle(bundle, book)
                   for book in TRADED_BOOKS}
    if not any(book_lines.values()):
        return [], {
            "pitcher_mlbam_id": bundle.metadata.pitcher_mlbam_id,
            "pitcher_name": bundle.metadata.pitcher_name,
            "game_pk": bundle.metadata.game_pk,
            "reason": "no_market_data_at_either_book",
            "is_transient": False,
        }

    picks: list[dict] = []
    archetype = (
        result.archetype_used
        or (bundle.pitcher_archetype.archetype if bundle.pitcher_archetype else None)
    )
    park_factor_by_hand = None
    if bundle.park_k_factors_by_hand is not None:
        park_factor_by_hand = bundle.park_k_factors_by_hand.for_pitcher_hand(
            bundle.pitcher.handedness,
        )

    for book, lines in book_lines.items():
        for line_value, sides in lines.items():
            for side, american_odds in sides.items():
                # Devig (with imputation if one-sided)
                devig = devig_with_imputation(lines, line_value, side)
                if devig is None:
                    continue
                market_p, devig_source = devig
                # Model probability for this side
                if side == "Over":
                    model_p = p_k_geq(line_value, result.e_k, nb_alpha)
                else:
                    model_p = p_k_leq(line_value, result.e_k, nb_alpha)

                edge = compute_edge_pct(model_p, market_p)
                if edge < SHADOW_EDGE_THRESHOLD:
                    continue
                ev = compute_ev_pct(model_p, american_odds)
                pick = {
                    "pitcher_mlbam_id": bundle.metadata.pitcher_mlbam_id,
                    "pitcher_name": bundle.metadata.pitcher_name,
                    "game_pk": bundle.metadata.game_pk,
                    "game_date": bundle.metadata.game_date.isoformat(),
                    "venue_name": bundle.game_context.venue_name,
                    "line": line_value,
                    "side": side,
                    "book": book,
                    "american_odds": american_odds,
                    "model_p": round(model_p, 6),
                    "market_p": round(market_p, 6),
                    "edge_pct": round(edge, 6),
                    "ev_pct": round(ev, 6),
                    "devig_source": devig_source,
                    "model_e_k": round(result.e_k, 4),
                    "model_e_bf": round(result.e_bf, 4),
                    "pitcher_archetype": archetype,
                    "park_k_factor_by_hand": (
                        round(park_factor_by_hand, 4)
                        if park_factor_by_hand is not None else None
                    ),
                    "calibration_note": CALIBRATION_NOTE,
                    "pick_id": _pick_id(
                        bundle.metadata.pitcher_mlbam_id,
                        bundle.metadata.game_pk,
                        line_value, side, book,
                    ),
                }
                picks.append(pick)

    return picks, None


def generate_picks(
    bundles: Iterable[ProjectionBundle],
    *,
    ctx: ProjectionContext | None = None,
    nb_alpha: float = 0.001,
) -> PickResult:
    """Generate picks across a slate. See :class:`PickResult`."""
    if ctx is None:
        ctx = ProjectionContext.from_default_paths()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_picks: list[dict] = []
    skipped: list[dict] = []
    n_bundles = 0
    for bundle in bundles:
        n_bundles += 1
        picks, skip_record = _evaluate_pitcher_picks(bundle, ctx, nb_alpha)
        all_picks.extend(picks)
        if skip_record is not None:
            skipped.append(skip_record)

    primary_candidates = [p for p in all_picks if p["edge_pct"] >= PRIMARY_EDGE_THRESHOLD]
    ranked_primary_pool = rank_primary_picks(primary_candidates)

    primary: list[dict] = []
    secondary: list[dict] = []
    shadow: list[dict] = []
    # Walk ranked_primary_pool first so rank_in_tier is correctly assigned
    # for the primary classification.
    ranked_lookup = {p["pick_id"]: p for p in ranked_primary_pool}
    for raw in all_picks:
        in_ranked = ranked_lookup.get(raw["pick_id"])
        rank = in_ranked["rank_in_tier"] if in_ranked else 0
        tier = classify_pick(raw["edge_pct"], raw["american_odds"], rank)
        if tier == "none":
            continue
        pick = {**raw, "tier": tier,
                "rank_in_tier": rank if tier == "primary" else 0,
                "generated_at": generated_at}
        if tier == "primary":
            primary.append(pick)
        elif tier == "secondary":
            secondary.append(pick)
        elif tier == "shadow":
            shadow.append(pick)

    # Rank within secondary by edge_pct, shadow by edge_pct (informational)
    secondary = [{**p, "rank_in_tier": i + 1}
                 for i, p in enumerate(sorted(secondary, key=lambda x: -x["edge_pct"]))]
    shadow = [{**p, "rank_in_tier": i + 1}
              for i, p in enumerate(sorted(shadow, key=lambda x: -x["edge_pct"]))]
    primary = sorted(primary, key=lambda p: p["rank_in_tier"])

    metadata = {
        "generated_at": generated_at,
        "n_bundles": n_bundles,
        "n_total_picks_evaluated": len(all_picks),
        "n_primary": len(primary),
        "n_secondary": len(secondary),
        "n_shadow": len(shadow),
        "n_skipped": len(skipped),
        "nb_alpha": nb_alpha,
        "books_traded": list(TRADED_BOOKS),
    }
    return PickResult(
        primary=primary,
        secondary=secondary,
        shadow=shadow,
        skipped=skipped,
        metadata=metadata,
    )
