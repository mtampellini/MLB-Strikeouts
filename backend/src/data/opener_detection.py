"""Opener detection: pure logic, no I/O.

Run as a post-step on the probables output before the pitcher is handed to the
feature pipeline. Takes already-fetched inputs and returns one of:

- :class:`BulkPitcherResult` — opener detected, bulk pitcher identified.
- :class:`SkipResult`         — opener detected but no usable bulk pitcher.
- :class:`NoOverride`         — listed probable is the real starter.

Override triggers (any one fires):
1. Listed probable has <30 IP this season AND avg IP/outing < 2.
2. Team used an opener at this rotation slot in the last 7 days
   (last starter at this slot threw < 2 IP).
3. No K prop posted on FD/DK for listed probable but another team pitcher
   has one (market signal).

Bulk pitcher search (when an override fires):
- Candidate must have season IP >= 15 AND >= 3 IP in bulk relief in the
  last 7 days (bulk relief = relief outing of 3+ IP).
- Exactly one match → use them. Zero or multiple matches → SkipResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


# -------- Inputs --------------------------------------------------------------


@dataclass(frozen=True)
class PitcherSeasonStats:
    pitcher_mlbam_id: int
    season_ip: float
    starts: int

    @property
    def avg_ip_per_start(self) -> float:
        return self.season_ip / self.starts if self.starts > 0 else 0.0


@dataclass(frozen=True)
class TeamRotationSlot:
    """The team's most recent start at the same rotation position.

    The pipeline computes "rotation position" via days-since-last-start
    modulo team rotation length; this struct just carries the result so
    opener_detection can stay pure.
    """

    last_starter_id: int
    last_starter_ip: float
    days_ago: int


@dataclass(frozen=True)
class BulkPitcherCandidate:
    pitcher_mlbam_id: int
    season_ip: float
    recent_bulk_relief_ip: float


# -------- Outputs -------------------------------------------------------------


@dataclass(frozen=True)
class BulkPitcherResult:
    use_pitcher: int
    confidence: float
    reason: str


@dataclass(frozen=True)
class SkipResult:
    reason: str


@dataclass(frozen=True)
class NoOverride:
    pass


OpenerDecision = BulkPitcherResult | SkipResult | NoOverride


# -------- Trigger helpers (callables, so the pipeline can compose them) -------


def trigger_low_volume_probable(stats: PitcherSeasonStats) -> str | None:
    if stats.season_ip < 30 and stats.avg_ip_per_start < 2:
        return (
            f"low-volume probable: {stats.season_ip:.1f} season IP, "
            f"{stats.avg_ip_per_start:.2f} avg IP/start"
        )
    return None


def trigger_recent_opener_at_slot(slot: TeamRotationSlot | None) -> str | None:
    if slot is None:
        return None
    if slot.days_ago <= 7 and slot.last_starter_ip < 2:
        return (
            f"team used opener at this slot {slot.days_ago}d ago "
            f"(last starter threw {slot.last_starter_ip:.1f} IP)"
        )
    return None


def trigger_market_signal(
    listed_probable_id: int,
    posted_k_prop_pitchers: Iterable[int],
    team_pitchers: Iterable[int],
) -> str | None:
    posted = set(posted_k_prop_pitchers)
    if listed_probable_id in posted:
        return None
    teammates_with_prop = set(team_pitchers) & posted
    if not teammates_with_prop:
        return None
    return (
        f"market signal: no K prop for listed probable, "
        f"teammates with prop = {sorted(teammates_with_prop)}"
    )


# -------- Main entry point ----------------------------------------------------


def check_opener(
    listed_probable_id: int,
    listed_probable_stats: PitcherSeasonStats,
    team_rotation_slot: TeamRotationSlot | None,
    posted_k_prop_pitchers: Iterable[int],
    team_pitchers: Iterable[int],
    bulk_candidates: Sequence[BulkPitcherCandidate],
) -> OpenerDecision:
    triggers: list[str] = []
    for reason in (
        trigger_low_volume_probable(listed_probable_stats),
        trigger_recent_opener_at_slot(team_rotation_slot),
        trigger_market_signal(
            listed_probable_id, posted_k_prop_pitchers, team_pitchers
        ),
    ):
        if reason is not None:
            triggers.append(reason)

    if not triggers:
        return NoOverride()

    eligible = [
        c
        for c in bulk_candidates
        if c.season_ip >= 15 and c.recent_bulk_relief_ip >= 3
    ]
    trigger_str = "; ".join(triggers)

    if not eligible:
        return SkipResult(
            reason=f"opener triggers fired ({trigger_str}) but no bulk pitcher found"
        )
    if len(eligible) > 1:
        ids = sorted(c.pitcher_mlbam_id for c in eligible)
        return SkipResult(
            reason=(
                f"opener triggers fired ({trigger_str}) but multiple bulk "
                f"candidates: {ids}"
            )
        )

    bulk = eligible[0]
    confidence = 0.85 if len(triggers) >= 2 else 0.7
    return BulkPitcherResult(
        use_pitcher=bulk.pitcher_mlbam_id,
        confidence=confidence,
        reason=f"opener detected ({trigger_str}); using bulk pitcher {bulk.pitcher_mlbam_id}",
    )
