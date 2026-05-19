"""Phase 2c bundle validator.

Validates a sample bundle JSON against the canonical contract. Used in two
ways:

1. By ``build_sample_bundle.py`` after writing, before declaring success.
2. As a standalone CLI: ``python -m scripts.bundle_validator <path>``.

Exit codes: 0 on pass, 1 on any contract violation (with details to stderr).

The contract is the source of truth for Phase 3. If a check here changes,
that's a contract change and downstream code must be reviewed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = (
    "metadata",
    "pitcher",
    "opposing_lineup",
    "game_context",
    "market",
)

REQUIRED_METADATA = (
    "bundle_version",
    "generated_at",
    "game_date",
    "cutoff_date",
    "pitcher_mlbam_id",
    "pitcher_name",
    "game_pk",
)

REQUIRED_PITCHER = (
    "mlbam_id",
    "name",
    "handedness",
    "team_abbr",
    "source",
    "fg_opener_flag",
    "fg_primary_pitcher_flag",
    "opener_detection_result",
    "statcast_pitches_30d",
    "statcast_pitches_season",
    "statcast_pitches_prior_year",
)

REQUIRED_OPPOSING_LINEUP = ("team_abbr", "batters", "lineup_posted")

REQUIRED_BATTER = (
    "mlbam_id",
    "name",
    "batting_order",
    "handedness",
    "position",
    "statcast_pa_season",
)

REQUIRED_GAME_CONTEXT = (
    "venue_id",
    "venue_name",
    "is_dome",
    "weather",
    "umpire_name",
    "umpire_id",
    "first_pitch_iso",
    "days_rest",
)

REQUIRED_WEATHER = (
    "temp_f",
    "wind_speed_mph",
    "wind_direction",
    "humidity_pct",
    "conditions",
)

REQUIRED_MARKET = ("fanduel", "draftkings", "snapshot_timestamp", "snapshot_source")

REQUIRED_BOOK = ("available", "lines")

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})$"
)


VALID_ARCHETYPES = (
    "Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy",
    "Balanced", "unknown",
)
VALID_ARCHETYPE_CONFIDENCE = (
    "current_season", "rolling_30d", "previous_season_fallback", "unknown",
)
PARK_K_FACTOR_MIN = 0.85
PARK_K_FACTOR_MAX = 1.30


def validate(bundle: dict) -> list[str]:
    issues: list[str] = []
    issues.extend(_check_required_keys(bundle, REQUIRED_TOP_LEVEL, where="root"))
    if "metadata" in bundle:
        issues.extend(_check_metadata(bundle["metadata"]))
    if "pitcher" in bundle:
        issues.extend(_check_pitcher(bundle["pitcher"]))
    if "opposing_lineup" in bundle:
        issues.extend(_check_opposing_lineup(bundle["opposing_lineup"]))
    if "game_context" in bundle:
        issues.extend(_check_game_context(bundle["game_context"]))
    if "market" in bundle:
        issues.extend(_check_market(bundle["market"]))
    # Phase 3-v2c-i optional extension fields. Only validate when present;
    # legacy bundles without them are intentionally accepted.
    if bundle.get("pitcher_archetype") is not None:
        issues.extend(_check_pitcher_archetype(bundle["pitcher_archetype"]))
    if bundle.get("tto_multipliers") is not None:
        issues.extend(_check_tto_multipliers(bundle["tto_multipliers"]))
    if bundle.get("park_k_factors_by_hand") is not None:
        issues.extend(_check_park_k_factors_by_hand(bundle["park_k_factors_by_hand"]))
    if bundle.get("pa_distribution") is not None:
        issues.extend(_check_pa_distribution(bundle["pa_distribution"]))
    issues.extend(_check_no_unsupported_types(bundle, path="$"))
    return issues


def _check_pitcher_archetype(blob: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(blob, dict):
        return ["pitcher_archetype: must be dict"]
    if blob.get("archetype") not in VALID_ARCHETYPES:
        out.append(
            f"pitcher_archetype.archetype: {blob.get('archetype')!r} not in "
            f"{VALID_ARCHETYPES}"
        )
    if blob.get("confidence") not in VALID_ARCHETYPE_CONFIDENCE:
        out.append(
            f"pitcher_archetype.confidence: {blob.get('confidence')!r} not in "
            f"{VALID_ARCHETYPE_CONFIDENCE}"
        )
    for k in ("fastball_pct", "four_seam_pct", "sinker_pct", "cutter_pct",
              "breaking_pct", "offspeed_pct"):
        v = blob.get(k)
        if not isinstance(v, (int, float)):
            out.append(f"pitcher_archetype.{k}: must be numeric")
    if not isinstance(blob.get("season"), int):
        out.append("pitcher_archetype.season: must be int")
    if not isinstance(blob.get("n_starts"), int):
        out.append("pitcher_archetype.n_starts: must be int")
    return out


def _check_tto_multipliers(blob: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(blob, dict):
        return ["tto_multipliers: must be dict"]
    by_archetype = blob.get("by_archetype") or {}
    expected = set(VALID_ARCHETYPES) - {"unknown"}
    if set(by_archetype.keys()) != expected:
        out.append(
            f"tto_multipliers.by_archetype: keys must be {sorted(expected)}, "
            f"got {sorted(by_archetype.keys())}"
        )
    for arch, cells in by_archetype.items():
        if not isinstance(cells, dict):
            out.append(f"tto_multipliers.by_archetype.{arch}: must be dict")
            continue
        for t in (1, 2, 3, 4):
            key = f"tto_{t}"
            if key not in cells:
                out.append(
                    f"tto_multipliers.by_archetype.{arch}: missing TTO bucket {t}"
                )
                continue
            cell = cells[key]
            if not isinstance(cell.get("multiplier"), (int, float)):
                out.append(
                    f"tto_multipliers.by_archetype.{arch}.{key}.multiplier: "
                    f"must be numeric"
                )
    league_wide = blob.get("league_wide") or {}
    for t in (1, 2, 3, 4):
        if f"tto_{t}" not in league_wide:
            out.append(f"tto_multipliers.league_wide: missing TTO bucket {t}")
    return out


def _check_park_k_factors_by_hand(blob: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(blob, dict):
        return ["park_k_factors_by_hand: must be dict"]
    for k in ("factor_lhp", "factor_rhp", "factor_combined"):
        v = blob.get(k)
        if not isinstance(v, (int, float)):
            out.append(f"park_k_factors_by_hand.{k}: must be numeric")
            continue
        if not (PARK_K_FACTOR_MIN <= v <= PARK_K_FACTOR_MAX):
            out.append(
                f"park_k_factors_by_hand.{k}: {v} outside "
                f"[{PARK_K_FACTOR_MIN}, {PARK_K_FACTOR_MAX}]"
            )
    if not isinstance(blob.get("venue_id"), int):
        out.append("park_k_factors_by_hand.venue_id: must be int")
    if not isinstance(blob.get("venue_name"), str):
        out.append("park_k_factors_by_hand.venue_name: must be str")
    return out


def _check_pa_distribution(blob: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(blob, dict):
        return ["pa_distribution: must be dict"]
    by_bf = blob.get("by_bf") or {}
    if not by_bf:
        out.append("pa_distribution.by_bf: must be non-empty")
        return out
    for bf_key, slots in by_bf.items():
        if not isinstance(slots, dict):
            out.append(f"pa_distribution.by_bf.{bf_key}: must be dict")
            continue
        # Spot-check: every cell's tto values sum to total_pa (±0.001)
        for slot_key, cell in slots.items():
            if not isinstance(cell, dict):
                out.append(
                    f"pa_distribution.by_bf.{bf_key}.{slot_key}: must be dict"
                )
                continue
            tto_sum = sum(
                float(cell.get(f"tto_{t}") or 0.0) for t in (1, 2, 3, 4)
            )
            total = float(cell.get("total_pa") or 0.0)
            if abs(tto_sum - total) > 0.001:
                out.append(
                    f"pa_distribution.by_bf.{bf_key}.{slot_key}: tto_sum "
                    f"{tto_sum:.4f} != total_pa {total:.4f}"
                )
    return out


def _check_required_keys(obj: dict, keys: tuple[str, ...], *, where: str) -> list[str]:
    out: list[str] = []
    if not isinstance(obj, dict):
        return [f"{where}: expected dict, got {type(obj).__name__}"]
    for k in keys:
        if k not in obj:
            out.append(f"{where}: missing required key {k!r}")
    return out


def _check_metadata(md: dict) -> list[str]:
    out = _check_required_keys(md, REQUIRED_METADATA, where="metadata")
    if "bundle_version" in md and md["bundle_version"] != "1.0":
        out.append(f"metadata.bundle_version: expected '1.0', got {md['bundle_version']!r}")
    if "game_date" in md and not _is_iso_date(md["game_date"]):
        out.append(f"metadata.game_date: not an ISO date: {md['game_date']!r}")
    if "cutoff_date" in md and not _is_iso_date(md["cutoff_date"]):
        out.append(f"metadata.cutoff_date: not an ISO date: {md['cutoff_date']!r}")
    if "generated_at" in md and not _is_iso_datetime(md["generated_at"]):
        out.append(f"metadata.generated_at: not an ISO datetime with offset: {md['generated_at']!r}")
    if "pitcher_mlbam_id" in md and not isinstance(md["pitcher_mlbam_id"], int):
        out.append("metadata.pitcher_mlbam_id: must be int")
    if "game_pk" in md and not isinstance(md["game_pk"], int):
        out.append("metadata.game_pk: must be int")
    return out


def _check_pitcher(pitcher: dict) -> list[str]:
    out = _check_required_keys(pitcher, REQUIRED_PITCHER, where="pitcher")
    if pitcher.get("handedness") not in ("L", "R", None):
        out.append(f"pitcher.handedness: must be 'L', 'R' (got {pitcher.get('handedness')!r})")
    if "source" in pitcher and pitcher["source"] not in (
        "fangraphs", "statsapi", "historical-actual",
    ):
        out.append(f"pitcher.source: unrecognized value {pitcher['source']!r}")
    for flag in ("fg_opener_flag", "fg_primary_pitcher_flag"):
        if flag in pitcher and not isinstance(pitcher[flag], bool):
            out.append(f"pitcher.{flag}: must be bool")
    if pitcher.get("opener_detection_result") not in (
        "no_override", "bulk_pitcher", "skip", None,
    ):
        out.append(
            f"pitcher.opener_detection_result: unrecognized "
            f"{pitcher.get('opener_detection_result')!r}"
        )
    for list_field in (
        "statcast_pitches_30d",
        "statcast_pitches_season",
        "statcast_pitches_prior_year",
    ):
        if list_field in pitcher and not isinstance(pitcher[list_field], list):
            out.append(
                f"pitcher.{list_field}: must be list (canonical empty is []), "
                f"got {type(pitcher[list_field]).__name__}"
            )
    return out


def _check_opposing_lineup(lineup: dict) -> list[str]:
    out = _check_required_keys(lineup, REQUIRED_OPPOSING_LINEUP, where="opposing_lineup")
    if "lineup_posted" in lineup and not isinstance(lineup["lineup_posted"], bool):
        out.append("opposing_lineup.lineup_posted: must be bool")
    if not isinstance(lineup.get("batters"), list):
        out.append("opposing_lineup.batters: must be list")
        return out
    for i, b in enumerate(lineup["batters"]):
        for k in REQUIRED_BATTER:
            if k not in b:
                out.append(f"opposing_lineup.batters[{i}]: missing {k!r}")
        if "handedness" in b and b["handedness"] not in ("L", "R", "S", None):
            out.append(
                f"opposing_lineup.batters[{i}].handedness: unrecognized {b['handedness']!r}"
            )
        if "batting_order" in b and not (
            isinstance(b["batting_order"], int) and 1 <= b["batting_order"] <= 9
        ):
            out.append(
                f"opposing_lineup.batters[{i}].batting_order: must be int 1..9"
            )
        if "statcast_pa_season" in b and not isinstance(b["statcast_pa_season"], list):
            out.append(
                f"opposing_lineup.batters[{i}].statcast_pa_season: must be list"
            )
    # If lineup_posted=False, batters must be []
    if lineup.get("lineup_posted") is False and lineup.get("batters"):
        out.append(
            "opposing_lineup: lineup_posted=False but batters is non-empty"
        )
    return out


def _check_game_context(ctx: dict) -> list[str]:
    out = _check_required_keys(ctx, REQUIRED_GAME_CONTEXT, where="game_context")
    if "weather" in ctx:
        if not isinstance(ctx["weather"], dict):
            out.append("game_context.weather: must be dict")
        else:
            out.extend(
                _check_required_keys(ctx["weather"], REQUIRED_WEATHER, where="game_context.weather")
            )
    if "is_dome" in ctx and not isinstance(ctx["is_dome"], bool):
        out.append("game_context.is_dome: must be bool")
    if "first_pitch_iso" in ctx and ctx["first_pitch_iso"] is not None:
        if not _is_iso_datetime(ctx["first_pitch_iso"]):
            out.append(
                f"game_context.first_pitch_iso: not ISO datetime with offset: "
                f"{ctx['first_pitch_iso']!r}"
            )
    return out


def _check_market(market: dict) -> list[str]:
    out = _check_required_keys(market, REQUIRED_MARKET, where="market")
    for book in ("fanduel", "draftkings"):
        if book not in market:
            continue
        section = market[book]
        if not isinstance(section, dict):
            out.append(f"market.{book}: must be dict")
            continue
        out.extend(_check_required_keys(section, REQUIRED_BOOK, where=f"market.{book}"))
        if "available" in section and not isinstance(section["available"], bool):
            out.append(f"market.{book}.available: must be bool")
        if "lines" in section and not isinstance(section["lines"], list):
            out.append(f"market.{book}.lines: must be list")
        # If available=False, lines must be []
        if section.get("available") is False and section.get("lines"):
            out.append(
                f"market.{book}: available=False but lines is non-empty"
            )
    if market.get("snapshot_source") not in ("live", "committed", "missing"):
        out.append(
            f"market.snapshot_source: must be one of "
            f"'live'/'committed'/'missing' (got {market.get('snapshot_source')!r})"
        )
    if market.get("snapshot_timestamp") is not None and not _is_iso_datetime(
        market["snapshot_timestamp"]
    ):
        out.append(
            f"market.snapshot_timestamp: not ISO datetime: "
            f"{market['snapshot_timestamp']!r}"
        )
    return out


def _check_no_unsupported_types(obj: Any, *, path: str) -> list[str]:
    """Walk the bundle and flag any non-JSON-native value.

    A DataFrame slipping through would land as ``"<class 'pandas...'>"`` after
    a default-str fallback, so we look for repr-ish strings AND any actual
    non-(str|int|float|bool|None|list|dict).
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return []
    if isinstance(obj, list):
        out: list[str] = []
        for i, v in enumerate(obj):
            out.extend(_check_no_unsupported_types(v, path=f"{path}[{i}]"))
        return out
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.extend(_check_no_unsupported_types(v, path=f"{path}.{k}"))
        return out
    return [
        f"{path}: unsupported type {type(obj).__name__} "
        f"(only list/dict/str/int/float/bool/null allowed)"
    ]


def _is_iso_date(s: Any) -> bool:
    return isinstance(s, str) and bool(ISO_DATE_RE.match(s))


def _is_iso_datetime(s: Any) -> bool:
    return isinstance(s, str) and bool(ISO_DATETIME_RE.match(s))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_path", type=Path)
    args = parser.parse_args()
    try:
        bundle = json.loads(args.bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not load bundle: {exc}", file=sys.stderr)
        return 1
    issues = validate(bundle)
    if issues:
        for i in issues:
            print(f"FAIL: {i}", file=sys.stderr)
        return 1
    print(f"ok: {args.bundle_path} matches the Phase 2c contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
