"""Phase 3a typed input contract.

The dataclasses here mirror the Phase 2c bundle JSON shape EXACTLY — no field
renaming, no normalization. The bundle builder is the only place where
heterogeneous source shapes get reconciled; from here downstream everything is
typed.

Loader: :meth:`ProjectionBundle.from_json`. Validation rules (raise on failure
unless noted):

1. Required top-level keys present.
2. ``metadata.cutoff_date < metadata.game_date`` (strict).
3. Handedness consistency: pitcher's Statcast rows' ``p_throws`` must equal
   the top-level ``pitcher.handedness``. On mismatch we warn loudly and trust
   the top-level value (it came from FG/StatsAPI, the rows came from a
   third-party feed) — never crash.
4. ``team_abbr`` for both the pitcher and the opposing lineup is in
   :data:`MLB_TEAM_ABBRS`.
5. Empty-state contract: ``lineup_posted=False`` requires ``batters=[]`` (and
   the inverse — posted lineup must have batters). ``book.available=False``
   requires ``lines=[]``.
6. No DataFrames, no numpy scalar types in the loaded JSON. Defense-in-depth
   against a regression in the bundle builder.

Phase 3a also exposes :class:`LeagueAverages`, :class:`ParkFactors`, and
:class:`UmpireKFactors` — wrappers around the static JSON files Phase 3b and
3c consume.
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class ContractViolation(Exception):
    """Bundle does not conform to the Phase 2c canonical contract."""


# 30 MLB teams in FG-style 3-letter form, matching the bundle's normalization.
# OAK is allowed as a legacy alias for ATH (the 2024 rebrand).
MLB_TEAM_ABBRS = frozenset({
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "OAK", "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR",
    "WSN",
})

ALLOWED_PITCHER_SOURCES = frozenset({"fangraphs", "statsapi", "historical-actual"})
ALLOWED_OPENER_RESULTS = frozenset({"no_override", "bulk_pitcher", "skip"})
ALLOWED_SNAPSHOT_SOURCES = frozenset({"live", "committed", "missing"})


# ---- Bundle dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class BundleMetadata:
    bundle_version: str
    generated_at: str
    game_date: date
    cutoff_date: date
    pitcher_mlbam_id: int
    pitcher_name: str
    game_pk: int

    @classmethod
    def from_dict(cls, d: dict) -> "BundleMetadata":
        return cls(
            bundle_version=d["bundle_version"],
            generated_at=d["generated_at"],
            game_date=date.fromisoformat(d["game_date"]),
            cutoff_date=date.fromisoformat(d["cutoff_date"]),
            pitcher_mlbam_id=int(d["pitcher_mlbam_id"]),
            pitcher_name=d["pitcher_name"],
            game_pk=int(d["game_pk"]),
        )

    def to_dict(self) -> dict:
        return {
            "bundle_version": self.bundle_version,
            "generated_at": self.generated_at,
            "game_date": self.game_date.isoformat(),
            "cutoff_date": self.cutoff_date.isoformat(),
            "pitcher_mlbam_id": self.pitcher_mlbam_id,
            "pitcher_name": self.pitcher_name,
            "game_pk": self.game_pk,
        }


@dataclass(frozen=True)
class PitcherInputs:
    mlbam_id: int
    name: str
    handedness: str | None
    team_abbr: str
    source: str
    fg_opener_flag: bool
    fg_primary_pitcher_flag: bool
    opener_detection_result: str
    statcast_pitches_30d: tuple[dict, ...]
    statcast_pitches_season: tuple[dict, ...]
    statcast_pitches_prior_year: tuple[dict, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "PitcherInputs":
        return cls(
            mlbam_id=int(d["mlbam_id"]),
            name=d["name"],
            handedness=d.get("handedness"),
            team_abbr=d["team_abbr"],
            source=d["source"],
            fg_opener_flag=bool(d["fg_opener_flag"]),
            fg_primary_pitcher_flag=bool(d["fg_primary_pitcher_flag"]),
            opener_detection_result=d["opener_detection_result"],
            statcast_pitches_30d=tuple(d.get("statcast_pitches_30d") or ()),
            statcast_pitches_season=tuple(d.get("statcast_pitches_season") or ()),
            statcast_pitches_prior_year=tuple(d.get("statcast_pitches_prior_year") or ()),
        )

    def to_dict(self) -> dict:
        return {
            "mlbam_id": self.mlbam_id,
            "name": self.name,
            "handedness": self.handedness,
            "team_abbr": self.team_abbr,
            "source": self.source,
            "fg_opener_flag": self.fg_opener_flag,
            "fg_primary_pitcher_flag": self.fg_primary_pitcher_flag,
            "opener_detection_result": self.opener_detection_result,
            "statcast_pitches_30d": list(self.statcast_pitches_30d),
            "statcast_pitches_season": list(self.statcast_pitches_season),
            "statcast_pitches_prior_year": list(self.statcast_pitches_prior_year),
        }


@dataclass(frozen=True)
class BatterInputs:
    mlbam_id: int
    name: str
    batting_order: int
    handedness: str | None  # L / R / S (switch)
    position: str | None
    statcast_pa_season: tuple[dict, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "BatterInputs":
        return cls(
            mlbam_id=int(d["mlbam_id"]),
            name=d["name"],
            batting_order=int(d["batting_order"]),
            handedness=d.get("handedness"),
            position=d.get("position"),
            statcast_pa_season=tuple(d.get("statcast_pa_season") or ()),
        )

    def to_dict(self) -> dict:
        return {
            "mlbam_id": self.mlbam_id,
            "name": self.name,
            "batting_order": self.batting_order,
            "handedness": self.handedness,
            "position": self.position,
            "statcast_pa_season": list(self.statcast_pa_season),
        }


@dataclass(frozen=True)
class OpposingLineup:
    team_abbr: str
    batters: tuple[BatterInputs, ...]
    lineup_posted: bool

    @classmethod
    def from_dict(cls, d: dict) -> "OpposingLineup":
        return cls(
            team_abbr=d["team_abbr"],
            batters=tuple(BatterInputs.from_dict(b) for b in (d.get("batters") or ())),
            lineup_posted=bool(d["lineup_posted"]),
        )

    def to_dict(self) -> dict:
        return {
            "team_abbr": self.team_abbr,
            "batters": [b.to_dict() for b in self.batters],
            "lineup_posted": self.lineup_posted,
        }


@dataclass(frozen=True)
class WeatherInputs:
    temp_f: float | None
    wind_speed_mph: float | None
    wind_direction: str | None
    humidity_pct: float | None
    conditions: str | None

    @classmethod
    def from_dict(cls, d: dict) -> "WeatherInputs":
        return cls(
            temp_f=_coerce_float(d.get("temp_f")),
            wind_speed_mph=_coerce_float(d.get("wind_speed_mph")),
            wind_direction=d.get("wind_direction"),
            humidity_pct=_coerce_float(d.get("humidity_pct")),
            conditions=d.get("conditions"),
        )

    def to_dict(self) -> dict:
        return {
            "temp_f": self.temp_f,
            "wind_speed_mph": self.wind_speed_mph,
            "wind_direction": self.wind_direction,
            "humidity_pct": self.humidity_pct,
            "conditions": self.conditions,
        }


@dataclass(frozen=True)
class GameContext:
    venue_id: int | None
    venue_name: str | None
    is_dome: bool
    weather: WeatherInputs
    umpire_name: str | None
    umpire_id: int | None
    first_pitch_iso: str | None
    days_rest: int | None

    @classmethod
    def from_dict(cls, d: dict) -> "GameContext":
        venue_id = d.get("venue_id")
        umpire_id = d.get("umpire_id")
        return cls(
            venue_id=int(venue_id) if venue_id is not None else None,
            venue_name=d.get("venue_name"),
            is_dome=bool(d["is_dome"]),
            weather=WeatherInputs.from_dict(d.get("weather") or {}),
            umpire_name=d.get("umpire_name"),
            umpire_id=int(umpire_id) if umpire_id is not None else None,
            first_pitch_iso=d.get("first_pitch_iso"),
            days_rest=int(d["days_rest"]) if d.get("days_rest") is not None else None,
        )

    def to_dict(self) -> dict:
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "is_dome": self.is_dome,
            "weather": self.weather.to_dict(),
            "umpire_name": self.umpire_name,
            "umpire_id": self.umpire_id,
            "first_pitch_iso": self.first_pitch_iso,
            "days_rest": self.days_rest,
        }


@dataclass(frozen=True)
class BookLine:
    line: float
    side: str  # "Over" or "Under"
    price: int  # American

    @classmethod
    def from_dict(cls, d: dict) -> "BookLine":
        return cls(line=float(d["line"]), side=d["side"], price=int(d["price"]))

    def to_dict(self) -> dict:
        return {"line": self.line, "side": self.side, "price": self.price}


@dataclass(frozen=True)
class BookMarket:
    available: bool
    lines: tuple[BookLine, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "BookMarket":
        return cls(
            available=bool(d["available"]),
            lines=tuple(BookLine.from_dict(ln) for ln in (d.get("lines") or ())),
        )

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "lines": [ln.to_dict() for ln in self.lines],
        }


@dataclass(frozen=True)
class Market:
    fanduel: BookMarket
    draftkings: BookMarket
    snapshot_timestamp: str | None
    snapshot_source: str

    @classmethod
    def from_dict(cls, d: dict) -> "Market":
        return cls(
            fanduel=BookMarket.from_dict(d["fanduel"]),
            draftkings=BookMarket.from_dict(d["draftkings"]),
            snapshot_timestamp=d.get("snapshot_timestamp"),
            snapshot_source=d["snapshot_source"],
        )

    def to_dict(self) -> dict:
        return {
            "fanduel": self.fanduel.to_dict(),
            "draftkings": self.draftkings.to_dict(),
            "snapshot_timestamp": self.snapshot_timestamp,
            "snapshot_source": self.snapshot_source,
        }


# ---- Phase 3-v2c-i: P(K|PA) rewrite static data wrappers --------------------
#
# These four dataclasses carry the additive contract extensions for the
# Phase 3-v2c projector rewrite. Each lives as an OPTIONAL field on
# ProjectionBundle; legacy bundles (Phase 4c E[BF] fit) ignore them entirely.


VALID_ARCHETYPES = frozenset({
    "Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced",
})
ARCHETYPE_CONFIDENCE_VALUES = frozenset({
    "current_season", "rolling_30d", "previous_season_fallback", "unknown",
})


@dataclass(frozen=True)
class PitcherArchetype:
    """Per-pitcher 5-category classification from Phase 3-v2a, resolved
    against the bundle's pitcher + game date.

    `confidence` records which lookup path succeeded — see `from_archetypes_lookup`
    for the fallback chain. archetype="unknown"/confidence="unknown" is the
    sentinel emitted when none of (current season, rolling 30d, previous
    season) is available.
    """

    archetype: str
    season: int
    fastball_pct: float
    four_seam_pct: float
    sinker_pct: float
    cutter_pct: float
    breaking_pct: float
    offspeed_pct: float
    n_starts: int
    confidence: str

    @classmethod
    def unknown(cls, season: int) -> "PitcherArchetype":
        return cls(
            archetype="unknown", season=season,
            fastball_pct=0.0, four_seam_pct=0.0, sinker_pct=0.0,
            cutter_pct=0.0, breaking_pct=0.0, offspeed_pct=0.0,
            n_starts=0, confidence="unknown",
        )

    @classmethod
    def from_archetypes_lookup(
        cls, archetypes_blob: dict, pitcher_mlbam_id: int, game_year: int,
    ) -> "PitcherArchetype":
        """Apply the fallback chain: current season → rolling 30d (same
        season) → previous season → unknown sentinel."""
        entry = (archetypes_blob.get("archetypes") or {}).get(str(pitcher_mlbam_id))
        if not entry:
            return cls.unknown(game_year)

        by_season = entry.get("by_season") or {}
        rolling = entry.get("season_rolling_30d") or {}

        cur = by_season.get(str(game_year))
        if cur and "archetype" in cur:
            return cls._from_info(cur, game_year, "current_season")

        cur_rolling = rolling.get(str(game_year))
        if cur_rolling and "archetype" in cur_rolling:
            return cls._from_info(cur_rolling, game_year, "rolling_30d")

        prev = by_season.get(str(game_year - 1))
        if prev and "archetype" in prev:
            return cls._from_info(prev, game_year - 1, "previous_season_fallback")

        return cls.unknown(game_year)

    @classmethod
    def _from_info(cls, info: dict, season: int, confidence: str) -> "PitcherArchetype":
        return cls(
            archetype=str(info["archetype"]),
            season=int(season),
            fastball_pct=float(info.get("fastball_pct") or 0.0),
            four_seam_pct=float(info.get("four_seam_pct") or 0.0),
            sinker_pct=float(info.get("sinker_pct") or 0.0),
            cutter_pct=float(info.get("cutter_pct") or 0.0),
            breaking_pct=float(info.get("breaking_pct") or 0.0),
            offspeed_pct=float(info.get("offspeed_pct") or 0.0),
            n_starts=int(info.get("n_starts") or 0),
            confidence=confidence,
        )

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "season": self.season,
            "fastball_pct": self.fastball_pct,
            "four_seam_pct": self.four_seam_pct,
            "sinker_pct": self.sinker_pct,
            "cutter_pct": self.cutter_pct,
            "breaking_pct": self.breaking_pct,
            "offspeed_pct": self.offspeed_pct,
            "n_starts": self.n_starts,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PitcherArchetype":
        return cls(
            archetype=str(d["archetype"]),
            season=int(d["season"]),
            fastball_pct=float(d["fastball_pct"]),
            four_seam_pct=float(d["four_seam_pct"]),
            sinker_pct=float(d["sinker_pct"]),
            cutter_pct=float(d["cutter_pct"]),
            breaking_pct=float(d["breaking_pct"]),
            offspeed_pct=float(d["offspeed_pct"]),
            n_starts=int(d["n_starts"]),
            confidence=str(d["confidence"]),
        )


@dataclass(frozen=True)
class TTOMultiplierCell:
    multiplier: float
    k_rate: float
    n_pa: int
    low_sample: bool

    @classmethod
    def from_dict(cls, d: dict) -> "TTOMultiplierCell":
        return cls(
            multiplier=float(d["multiplier"]),
            k_rate=float(d["k_rate"]),
            n_pa=int(d["n_pa"]),
            low_sample=bool(d.get("low_sample", False)),
        )

    def to_dict(self) -> dict:
        return {
            "multiplier": self.multiplier,
            "k_rate": self.k_rate,
            "n_pa": self.n_pa,
            "low_sample": self.low_sample,
        }


@dataclass(frozen=True)
class TTOMultipliers:
    """Full lookup table — 5 archetypes x 4 TTO buckets + league_wide
    fallback (TTO only)."""

    by_archetype: dict[str, dict[int, TTOMultiplierCell]]
    league_wide: dict[int, TTOMultiplierCell]

    @classmethod
    def from_json(cls, path: Path | str) -> "TTOMultipliers":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(blob)

    @classmethod
    def from_dict(cls, blob: dict) -> "TTOMultipliers":
        by_arch: dict[str, dict[int, TTOMultiplierCell]] = {}
        for arch, cells in (blob.get("by_archetype") or {}).items():
            inner: dict[int, TTOMultiplierCell] = {}
            for key, cell in cells.items():
                # keys arrive as "tto_1", "tto_2", ... — strip prefix
                t = int(key.split("_")[-1])
                inner[t] = TTOMultiplierCell.from_dict(cell)
            by_arch[arch] = inner
        lw: dict[int, TTOMultiplierCell] = {}
        for key, cell in (blob.get("league_wide") or {}).items():
            t = int(key.split("_")[-1])
            lw[t] = TTOMultiplierCell.from_dict(cell)
        return cls(by_archetype=by_arch, league_wide=lw)

    def to_dict(self) -> dict:
        return {
            "by_archetype": {
                arch: {f"tto_{t}": c.to_dict() for t, c in cells.items()}
                for arch, cells in self.by_archetype.items()
            },
            "league_wide": {
                f"tto_{t}": c.to_dict() for t, c in self.league_wide.items()
            },
        }

    def lookup(self, archetype: str | None, tto: int) -> TTOMultiplierCell:
        """Return the cell for (archetype, TTO). Falls back to league_wide
        when archetype is unknown / missing."""
        if archetype and archetype in self.by_archetype:
            cell = self.by_archetype[archetype].get(int(tto))
            if cell is not None:
                return cell
        return self.league_wide[int(tto)]


@dataclass(frozen=True)
class ParkKFactorsByHand:
    """Per-venue L/R-split park K factor from Phase 4b-v2."""

    venue_id: int
    venue_name: str
    factor_lhp: float
    factor_rhp: float
    factor_combined: float
    n_games_lhp: int
    n_games_rhp: int

    @classmethod
    def from_json_lookup(
        cls, path: Path | str, venue_id: int | None,
    ) -> "ParkKFactorsByHand | None":
        if venue_id is None:
            return None
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = (blob.get("factors") or {}).get(str(int(venue_id)))
        if not entry:
            return None
        return cls(
            venue_id=int(venue_id),
            venue_name=str(entry.get("venue_name") or ""),
            factor_lhp=float(entry.get("factor_lhp") or 1.0),
            factor_rhp=float(entry.get("factor_rhp") or 1.0),
            factor_combined=float(entry.get("factor_combined") or 1.0),
            n_games_lhp=int(entry.get("n_games_lhp") or 0),
            n_games_rhp=int(entry.get("n_games_rhp") or 0),
        )

    def to_dict(self) -> dict:
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "factor_lhp": self.factor_lhp,
            "factor_rhp": self.factor_rhp,
            "factor_combined": self.factor_combined,
            "n_games_lhp": self.n_games_lhp,
            "n_games_rhp": self.n_games_rhp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParkKFactorsByHand":
        return cls(
            venue_id=int(d["venue_id"]),
            venue_name=str(d["venue_name"]),
            factor_lhp=float(d["factor_lhp"]),
            factor_rhp=float(d["factor_rhp"]),
            factor_combined=float(d["factor_combined"]),
            n_games_lhp=int(d["n_games_lhp"]),
            n_games_rhp=int(d["n_games_rhp"]),
        )

    def for_pitcher_hand(self, hand: str | None) -> float:
        if hand == "L":
            return self.factor_lhp
        if hand == "R":
            return self.factor_rhp
        return self.factor_combined


@dataclass(frozen=True)
class PADistributionCell:
    tto_1: float
    tto_2: float
    tto_3: float
    tto_4: float
    total_pa: float

    @classmethod
    def from_dict(cls, d: dict) -> "PADistributionCell":
        return cls(
            tto_1=float(d["tto_1"]),
            tto_2=float(d["tto_2"]),
            tto_3=float(d["tto_3"]),
            tto_4=float(d["tto_4"]),
            total_pa=float(d["total_pa"]),
        )

    def to_dict(self) -> dict:
        return {
            "tto_1": self.tto_1, "tto_2": self.tto_2, "tto_3": self.tto_3,
            "tto_4": self.tto_4, "total_pa": self.total_pa,
        }


@dataclass(frozen=True)
class PADistribution:
    """Full PA distribution lookup table from Phase 3-v2c-0.

    by_bf: total_bf (int) → batting_order_slot (1..9, int) → cell.
    """

    by_bf: dict[int, dict[int, PADistributionCell]]
    bf_range_observed: tuple[int, int]
    modal_bf: int

    @classmethod
    def from_json(cls, path: Path | str) -> "PADistribution":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(blob)

    @classmethod
    def from_dict(cls, blob: dict) -> "PADistribution":
        """Accept BOTH wire formats:

        1. The raw `pa_distribution_by_bf.json` shape from derive_pa_distribution:
           ``{"distributions": {bf: {by_slot: {...}, n_games, smoothed}}, "summary": {...}}``
        2. The bundle-embedded shape from `to_dict()`:
           ``{"by_bf": {bf: {slot: cell}}, "bf_range_observed": [..], "modal_bf": N}``
        """
        by_bf: dict[int, dict[int, PADistributionCell]] = {}
        if "by_bf" in blob:
            # Bundle-embedded shape (from to_dict round-trip).
            for bf_str, slot_map_blob in (blob.get("by_bf") or {}).items():
                slot_map: dict[int, PADistributionCell] = {}
                for slot_str, cell in slot_map_blob.items():
                    slot_map[int(slot_str)] = PADistributionCell.from_dict(cell)
                by_bf[int(bf_str)] = slot_map
            rng = blob.get("bf_range_observed") or (0, 0)
            modal = blob.get("modal_bf") or 0
        else:
            # Raw derive_pa_distribution.py shape.
            for bf_str, entry in (blob.get("distributions") or {}).items():
                inner = (entry.get("by_slot") or {})
                slot_map = {}
                for slot_str, cell in inner.items():
                    slot_map[int(slot_str)] = PADistributionCell.from_dict(cell)
                by_bf[int(bf_str)] = slot_map
            rng = blob.get("summary", {}).get("bf_range_observed") or (0, 0)
            modal = blob.get("summary", {}).get("modal_bf") or 0
        return cls(
            by_bf=by_bf,
            bf_range_observed=(int(rng[0]), int(rng[1])),
            modal_bf=int(modal),
        )

    def to_dict(self) -> dict:
        return {
            "by_bf": {
                str(bf): {str(s): c.to_dict() for s, c in cells.items()}
                for bf, cells in self.by_bf.items()
            },
            "bf_range_observed": list(self.bf_range_observed),
            "modal_bf": self.modal_bf,
        }

    def lookup(self, projected_bf: float) -> dict[int, PADistributionCell]:
        """Round projected_bf to the nearest integer key (clamped to observed
        range) and return that BF's {slot -> cell} map."""
        keys = list(self.by_bf.keys())
        if not keys:
            return {}
        bf_int = int(round(projected_bf))
        lo, hi = min(keys), max(keys)
        bf_int = max(lo, min(hi, bf_int))
        return self.by_bf[bf_int]


@dataclass(frozen=True)
class ProjectionBundle:
    metadata: BundleMetadata
    pitcher: PitcherInputs
    opposing_lineup: OpposingLineup
    game_context: GameContext
    market: Market

    # Phase 3-v2c-i additive fields. All default None for backward compat.
    # Phase 4c E[BF] fit ignores these. Phase 3-v2c-iv projector requires
    # them populated for the new P(K|PA) path.
    pitcher_archetype: "PitcherArchetype | None" = None
    tto_multipliers: "TTOMultipliers | None" = None
    park_k_factors_by_hand: "ParkKFactorsByHand | None" = None
    pa_distribution: "PADistribution | None" = None

    # ---- Loader -----------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path | str) -> "ProjectionBundle":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(blob)

    @classmethod
    def from_dict(cls, blob: dict) -> "ProjectionBundle":
        _validate_top_level(blob)
        _validate_no_unsupported_types(blob)
        _validate_dates(blob)
        _validate_team_abbrs(blob)
        _validate_empty_states(blob)
        _validate_enums(blob)
        _check_handedness_consistency(blob)  # warn only

        # Phase 3-v2c-i: inline-embedded extension fields (additive).
        archetype_blob = blob.get("pitcher_archetype")
        pitcher_archetype = (
            PitcherArchetype.from_dict(archetype_blob)
            if archetype_blob is not None else None
        )
        tto_blob = blob.get("tto_multipliers")
        tto_multipliers = TTOMultipliers.from_dict(tto_blob) if tto_blob else None
        park_blob = blob.get("park_k_factors_by_hand")
        park_k_factors_by_hand = (
            ParkKFactorsByHand.from_dict(park_blob) if park_blob else None
        )
        pa_blob = blob.get("pa_distribution")
        pa_distribution = PADistribution.from_dict(pa_blob) if pa_blob else None

        return cls(
            metadata=BundleMetadata.from_dict(blob["metadata"]),
            pitcher=PitcherInputs.from_dict(blob["pitcher"]),
            opposing_lineup=OpposingLineup.from_dict(blob["opposing_lineup"]),
            game_context=GameContext.from_dict(blob["game_context"]),
            market=Market.from_dict(blob["market"]),
            pitcher_archetype=pitcher_archetype,
            tto_multipliers=tto_multipliers,
            park_k_factors_by_hand=park_k_factors_by_hand,
            pa_distribution=pa_distribution,
        )

    @classmethod
    def from_dict_with_static_data(
        cls,
        blob: dict,
        *,
        archetypes_path: "Path | str | None" = None,
        tto_path: "Path | str | None" = None,
        park_k_factors_path: "Path | str | None" = None,
        pa_distribution_path: "Path | str | None" = None,
    ) -> "ProjectionBundle":
        """Load a bundle and ALSO populate the four Phase 3-v2c-i extension
        fields from the static data files. Used by build_sample_bundle so
        downstream consumers see a fully-hydrated bundle. Missing file paths
        leave the corresponding field None (backward compat)."""
        bundle = cls.from_dict(blob)

        pitcher_archetype = bundle.pitcher_archetype
        if pitcher_archetype is None and archetypes_path is not None:
            try:
                archetypes_blob = json.loads(
                    Path(archetypes_path).read_text(encoding="utf-8")
                )
                pitcher_archetype = PitcherArchetype.from_archetypes_lookup(
                    archetypes_blob,
                    bundle.metadata.pitcher_mlbam_id,
                    bundle.metadata.game_date.year,
                )
            except FileNotFoundError:
                pitcher_archetype = None

        tto_multipliers = bundle.tto_multipliers
        if tto_multipliers is None and tto_path is not None:
            try:
                tto_multipliers = TTOMultipliers.from_json(tto_path)
            except FileNotFoundError:
                tto_multipliers = None

        park_k = bundle.park_k_factors_by_hand
        if park_k is None and park_k_factors_path is not None:
            park_k = ParkKFactorsByHand.from_json_lookup(
                park_k_factors_path, bundle.game_context.venue_id,
            )

        pa_distribution = bundle.pa_distribution
        if pa_distribution is None and pa_distribution_path is not None:
            try:
                pa_distribution = PADistribution.from_json(pa_distribution_path)
            except FileNotFoundError:
                pa_distribution = None

        # Re-construct with the populated fields (frozen dataclass)
        from dataclasses import replace
        return replace(
            bundle,
            pitcher_archetype=pitcher_archetype,
            tto_multipliers=tto_multipliers,
            park_k_factors_by_hand=park_k,
            pa_distribution=pa_distribution,
        )

    def to_dict(self) -> dict:
        out = {
            "metadata": self.metadata.to_dict(),
            "pitcher": self.pitcher.to_dict(),
            "opposing_lineup": self.opposing_lineup.to_dict(),
            "game_context": self.game_context.to_dict(),
            "market": self.market.to_dict(),
        }
        if self.pitcher_archetype is not None:
            out["pitcher_archetype"] = self.pitcher_archetype.to_dict()
        if self.tto_multipliers is not None:
            out["tto_multipliers"] = self.tto_multipliers.to_dict()
        if self.park_k_factors_by_hand is not None:
            out["park_k_factors_by_hand"] = self.park_k_factors_by_hand.to_dict()
        if self.pa_distribution is not None:
            out["pa_distribution"] = self.pa_distribution.to_dict()
        return out


# ---- Static-data wrappers (Phase 3a exposes; 3b/3c consume) ----------------


@dataclass(frozen=True)
class HandednessAverages:
    k_pct: float
    obp: float
    zone_contact_pct: float
    chase_rate: float
    # Phase 3-v2c-iii: league CSW% by hand split. Optional (default None) so
    # legacy league_averages files without this field continue to load.
    csw_pct: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "HandednessAverages":
        return cls(
            k_pct=float(d["k_pct"]),
            obp=float(d["obp"]),
            zone_contact_pct=float(d["zone_contact_pct"]),
            chase_rate=float(d["chase_rate"]),
            csw_pct=float(d["csw_pct"]) if d.get("csw_pct") is not None else None,
        )


@dataclass(frozen=True)
class LeagueAverages:
    """Leaguewide rate stats by (batter_hand, pitcher_hand) split."""

    year: int
    r_vs_r: HandednessAverages
    r_vs_l: HandednessAverages
    l_vs_r: HandednessAverages
    l_vs_l: HandednessAverages

    @classmethod
    def from_json(cls, path: Path | str) -> "LeagueAverages":
        """Load from JSON.

        Accepts either the Phase 4a key naming (RR / RL / LR / LL) or the
        legacy Phase 3a placeholder naming (RvR / RvL / LvR / LvL). The Phase
        4a `season` field is honored if present; otherwise we fall back to
        the legacy `year` field.
        """
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        splits = blob["splits"]

        def _get(short: str, long: str) -> HandednessAverages:
            if short in splits:
                return HandednessAverages.from_dict(splits[short])
            if long in splits:
                return HandednessAverages.from_dict(splits[long])
            raise KeyError(f"LeagueAverages: missing split {short!r} (or {long!r})")

        year = int(blob.get("season") or blob.get("year"))
        return cls(
            year=year,
            r_vs_r=_get("RR", "RvR"),
            r_vs_l=_get("RL", "RvL"),
            l_vs_r=_get("LR", "LvR"),
            l_vs_l=_get("LL", "LvL"),
        )

    def lookup(
        self, batter_hand: str | None, pitcher_hand: str | None
    ) -> HandednessAverages | None:
        """Look up the split. Switch hitters resolve to the opposite of the pitcher.

        Returns None if either hand is None — caller must skip the batter.
        """
        if not pitcher_hand or not batter_hand:
            return None
        if pitcher_hand not in ("L", "R"):
            return None
        if batter_hand == "S":
            batter_hand = "L" if pitcher_hand == "R" else "R"
        if batter_hand not in ("L", "R"):
            return None
        return {
            ("R", "R"): self.r_vs_r,
            ("R", "L"): self.r_vs_l,
            ("L", "R"): self.l_vs_r,
            ("L", "L"): self.l_vs_l,
        }[(batter_hand, pitcher_hand)]


@dataclass(frozen=True)
class ParkFactors:
    """venue_id -> factor (run env OR K, depending on which file)."""

    year: int
    factors: dict[int, float]

    @classmethod
    def from_json(cls, path: Path | str) -> "ParkFactors":
        """Load park factors.

        Accepts two shapes:
        1. Phase 3a placeholder: ``{"year": Y, "factors": {vid: float, ...}}``
        2. Phase 4b derived:     ``{"seasons_used": [...], "factors":
                                    {vid: {factor, n_games, venue_name}}}``

        ``year`` is honored when present, else falls back to the latest season
        in ``seasons_used`` (the operationally relevant one).
        """
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if "year" in blob:
            year = int(blob["year"])
        elif "seasons_used" in blob and blob["seasons_used"]:
            year = int(max(blob["seasons_used"]))
        else:
            year = 0  # unknown; loader caller decides if that matters

        raw = blob["factors"]
        factors: dict[int, float] = {}
        for k, v in raw.items():
            if isinstance(v, (int, float)):
                factors[int(k)] = float(v)
            elif isinstance(v, dict):
                # Prefer Phase 4b-v2's factor_combined (L/R-split schema),
                # fall back to Phase 4b's single factor.
                if "factor_combined" in v:
                    factors[int(k)] = float(v["factor_combined"])
                elif "factor" in v:
                    factors[int(k)] = float(v["factor"])
                else:
                    raise ValueError(
                        f"ParkFactors: entry for venue {k} has no "
                        f"'factor_combined' or 'factor' key: {v!r}"
                    )
            else:
                raise ValueError(f"ParkFactors: unexpected entry for venue {k}: {v!r}")
        return cls(year=year, factors=factors)

    def get(self, venue_id: int | None) -> float | None:
        if venue_id is None:
            return None
        return self.factors.get(int(venue_id))


@dataclass(frozen=True)
class UmpireKFactors:
    """umpire_id -> K-zone factor. Missing umpire defaults to 1.0 (neutral)."""

    factors: dict[int, float]

    @classmethod
    def from_json(cls, path: Path | str) -> "UmpireKFactors":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            factors={int(k): float(v) for k, v in (blob.get("factors") or {}).items()}
        )

    def get(self, umpire_id: int | None) -> float:
        if umpire_id is None:
            return 1.0
        return self.factors.get(int(umpire_id), 1.0)


@dataclass(frozen=True)
class ProjectionContext:
    """Bundle the static lookup tables that feature builders consume.

    Loaded once at the top of a projection run and passed through to every
    builder. Keeps each builder's signature flat (no kwarg explosion).
    """

    league_avgs: LeagueAverages
    park_run_factors: ParkFactors
    park_k_factors: ParkFactors
    umpire_k_factors: UmpireKFactors
    # Phase 4c-v2 Step 3/6: additive. Default None for backward compat —
    # callers that build a minimal ctx (older tests, etc.) still work; the
    # projector falls back to the composed P(K|PA) for its log5 input when
    # these are None.
    csw_to_k_intercept: float | None = None
    csw_to_k_slope: float | None = None

    @classmethod
    def from_default_paths(cls) -> "ProjectionContext":
        processed = Path(__file__).resolve().parents[2] / "data" / "processed"
        csw_intercept, csw_slope = _try_load_csw_to_k(processed)
        return cls(
            league_avgs=LeagueAverages.from_json(
                processed / "league_averages_2025.json"
            ),
            park_run_factors=ParkFactors.from_json(processed / "park_factors.json"),
            park_k_factors=ParkFactors.from_json(processed / "park_k_factors.json"),
            umpire_k_factors=UmpireKFactors.from_json(
                processed / "umpire_k_factors.json"
            ),
            csw_to_k_intercept=csw_intercept,
            csw_to_k_slope=csw_slope,
        )


# Module-level guard so the legacy-fallback warning fires once, not per call.
_CSW_LOAD_WARNED = False


def _try_load_csw_to_k(processed_dir: Path) -> tuple[float | None, float | None]:
    """Load the CSW-to-K intercept/slope from processed_dir, returning
    (None, None) with a one-shot warning if the file is missing.

    Phase 4c-v2 Step 3 wiring — this preserves backward compat for older
    contexts and test fixtures that built ctx before the CSW relationship
    existed.
    """
    global _CSW_LOAD_WARNED
    path = processed_dir / "csw_to_k_relationship.json"
    if not path.exists():
        if not _CSW_LOAD_WARNED:
            logger.warning(
                "CSW-to-K relationship not loaded (file %s missing); "
                "projector will use composed P(K|PA) for log5 (legacy behavior)",
                path,
            )
            _CSW_LOAD_WARNED = True
        return None, None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        model = blob.get("model") or {}
        return float(model.get("intercept")), float(model.get("slope"))
    except (OSError, ValueError, TypeError) as exc:
        if not _CSW_LOAD_WARNED:
            logger.warning(
                "CSW-to-K relationship at %s failed to parse (%s); "
                "projector will use composed P(K|PA) for log5",
                path, exc,
            )
            _CSW_LOAD_WARNED = True
        return None, None


# ---- Validators ------------------------------------------------------------


_REQUIRED_TOP_LEVEL = (
    "metadata", "pitcher", "opposing_lineup", "game_context", "market",
)


def _validate_top_level(blob: dict) -> None:
    if not isinstance(blob, dict):
        raise ContractViolation(f"top-level must be dict, got {type(blob).__name__}")
    for k in _REQUIRED_TOP_LEVEL:
        if k not in blob:
            raise ContractViolation(f"missing required top-level key: {k!r}")


def _validate_dates(blob: dict) -> None:
    md = blob["metadata"]
    try:
        gd = date.fromisoformat(md["game_date"])
        cd = date.fromisoformat(md["cutoff_date"])
    except (KeyError, ValueError) as exc:
        raise ContractViolation(f"metadata: bad date field ({exc})") from exc
    if cd >= gd:
        raise ValueError(
            f"metadata.cutoff_date ({cd}) must be strictly before "
            f"metadata.game_date ({gd})"
        )


def _validate_team_abbrs(blob: dict) -> None:
    for path, val in (
        ("pitcher.team_abbr", blob.get("pitcher", {}).get("team_abbr")),
        ("opposing_lineup.team_abbr", blob.get("opposing_lineup", {}).get("team_abbr")),
    ):
        if val not in MLB_TEAM_ABBRS:
            raise ContractViolation(
                f"{path}: {val!r} is not in MLB_TEAM_ABBRS allowlist"
            )


def _validate_empty_states(blob: dict) -> None:
    lineup = blob.get("opposing_lineup", {})
    posted = lineup.get("lineup_posted")
    batters = lineup.get("batters") or []
    if posted is True and not batters:
        raise ContractViolation(
            "opposing_lineup: lineup_posted=True but batters is empty"
        )
    if posted is False and batters:
        raise ContractViolation(
            "opposing_lineup: lineup_posted=False but batters is non-empty"
        )
    for book in ("fanduel", "draftkings"):
        b = blob.get("market", {}).get(book, {})
        if b.get("available") is False and b.get("lines"):
            raise ContractViolation(
                f"market.{book}: available=False but lines is non-empty"
            )


def _validate_enums(blob: dict) -> None:
    p = blob.get("pitcher", {})
    src = p.get("source")
    if src not in ALLOWED_PITCHER_SOURCES:
        raise ContractViolation(
            f"pitcher.source: {src!r} not in {sorted(ALLOWED_PITCHER_SOURCES)}"
        )
    odr = p.get("opener_detection_result")
    if odr not in ALLOWED_OPENER_RESULTS:
        raise ContractViolation(
            f"pitcher.opener_detection_result: {odr!r} not in "
            f"{sorted(ALLOWED_OPENER_RESULTS)}"
        )
    snap = blob.get("market", {}).get("snapshot_source")
    if snap not in ALLOWED_SNAPSHOT_SOURCES:
        raise ContractViolation(
            f"market.snapshot_source: {snap!r} not in "
            f"{sorted(ALLOWED_SNAPSHOT_SOURCES)}"
        )


def _check_handedness_consistency(blob: dict) -> None:
    """Warn if any pitcher Statcast row's p_throws disagrees with the top-level
    pitcher.handedness. Never raise — the top-level is authoritative."""
    pitcher_hand = (blob.get("pitcher") or {}).get("handedness")
    if not pitcher_hand:
        return
    for window in ("statcast_pitches_30d", "statcast_pitches_season", "statcast_pitches_prior_year"):
        rows = (blob.get("pitcher") or {}).get(window) or []
        for row in rows:
            row_hand = row.get("p_throws")
            if row_hand and row_hand != pitcher_hand:
                msg = (
                    f"handedness mismatch: pitcher.handedness={pitcher_hand!r} "
                    f"but row in {window} has p_throws={row_hand!r}. "
                    f"Trusting top-level value."
                )
                warnings.warn(msg, stacklevel=3)
                logger.warning(msg)
                return  # one warning per bundle is enough


def _validate_no_unsupported_types(obj: Any, *, path: str = "$") -> None:
    """Walk the loaded blob and refuse anything that isn't JSON-native.

    The bundle validator already checks this, but the loader is the user-facing
    entry point for Phase 3 features — better to fail loudly here than have a
    pandas Timestamp leak through into feature math.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return
    if isinstance(obj, (int, float)):
        # numpy scalars subclass int/float but have an `.item()` method we don't
        # want to silently accept. Reject if the type isn't exactly int/float.
        if type(obj) not in (int, float):
            raise ContractViolation(
                f"{path}: unsupported numeric subtype {type(obj).__name__}"
            )
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _validate_no_unsupported_types(v, path=f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _validate_no_unsupported_types(v, path=f"{path}.{k}")
        return
    raise ContractViolation(
        f"{path}: unsupported type {type(obj).__name__}"
    )


# ---- Helpers ---------------------------------------------------------------


def _coerce_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        # bool is a subclass of int — explicit reject so we don't coerce a
        # True/False that snuck in via a buggy serializer.
        raise ContractViolation(f"expected float, got bool: {x!r}")
    return float(x)
