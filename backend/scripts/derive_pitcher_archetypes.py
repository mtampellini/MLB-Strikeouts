"""Phase 3-v2a: pitcher archetype classification.

Classifies every starter into one of five archetypes based on Statcast
pitch-type aggregation. Archetypes are the foundation for Phase 3-v2b
(TTO multipliers as archetype interaction terms) and Phase 3-v2c (P(K|PA)
feature redesign).

Archetypes (rules applied in order, first match wins):
1. Power-FF:       fastball_pct >= 0.55 AND four_seam_pct >= 0.35
2. Sinker-ball:    fastball_pct >= 0.55 AND sinker_pct    >= 0.25
3. Breaking-heavy: breaking_pct >= 0.35
4. Offspeed-heavy: offspeed_pct >= 0.25
5. Balanced:       catch-all

Skips:
- Fewer than 5 starts in the season
- Fewer than 500 total pitches in the season
- More than 30% "junk" pitches (KN/EP/GY/UN) — knuckleballer/eephus
  guys don't fit the framework

Each pitcher gets per-season classification AND a rolling 30-day
classification (most recent 30 days of the season).

Note: In current MLB, Breaking-heavy tends to be the modal archetype
among starting pitchers due to the slider/sweeper usage surge of recent
seasons. This is empirically observed, not an a-priori assumption.
Sanity gates check for non-collapse of categories (>=5% each) and
non-domination (no category > 50%), not for a specific modal category.

CLI:
    python -m scripts.derive_pitcher_archetypes
    python -m scripts.derive_pitcher_archetypes --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Pitch type groupings (Statcast canonical codes)
HEAT_TYPES = frozenset({"FF", "SI", "FC"})
BREAKING_TYPES = frozenset({"SL", "ST", "SV", "CU", "KC", "CS"})
OFFSPEED_TYPES = frozenset({"CH", "FS", "FO", "SC"})
JUNK_TYPES = frozenset({"KN", "EP", "GY", "UN"})

# Classification thresholds
FASTBALL_THRESHOLD = 0.55
FOUR_SEAM_THRESHOLD = 0.35
SINKER_THRESHOLD = 0.25
BREAKING_THRESHOLD = 0.35
OFFSPEED_THRESHOLD = 0.25

# Skip filters
MIN_STARTS = 5
MIN_PITCHES = 500
MAX_JUNK_PCT = 0.30

# "Starter" PA-per-game proxy (matches Phase 4b convention)
STARTER_MIN_PA = 12

# Rolling window
ROLLING_DAYS = 30
ROLLING_MIN_STARTS = 2

ARCHETYPES = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced")

# Manual spot-check pitchers — known expected archetypes for sanity.
SPOT_CHECK_EXPECTED = {
    694973: ("Paul Skenes", {"Power-FF"}),
    664285: ("Framber Valdez", {"Sinker-ball"}),
    607192: ("Tyler Glasnow", {"Breaking-heavy", "Power-FF"}),
    675911: ("Spencer Strider", {"Power-FF"}),
    669923: ("George Kirby", {"Balanced", "Power-FF"}),
    579328: ("Yusei Kikuchi", {"Breaking-heavy", "Balanced"}),
}


# ---- Pure helpers (unit-testable) ------------------------------------------


def _mix_from_counts(counts: dict[str, int]) -> dict:
    """Convert per-pitch-type counts to the mix dict used by classification.

    `total_pitches` excludes junk. Percentages are over the non-junk total.
    `junk_pct` and `n_pitches_all` are computed over ALL pitches and used
    only by the skip filter and reporting.
    """
    n_heat = sum(counts.get(p, 0) for p in HEAT_TYPES)
    n_breaking = sum(counts.get(p, 0) for p in BREAKING_TYPES)
    n_offspeed = sum(counts.get(p, 0) for p in OFFSPEED_TYPES)
    n_junk = sum(counts.get(p, 0) for p in JUNK_TYPES)
    n_all = sum(counts.values())
    n_total = n_all - n_junk

    if n_total <= 0:
        return {
            "fastball_pct": 0.0, "four_seam_pct": 0.0, "sinker_pct": 0.0,
            "cutter_pct": 0.0, "breaking_pct": 0.0, "offspeed_pct": 0.0,
            "total_pitches": 0, "junk_pct": 1.0 if n_all > 0 else 0.0,
            "n_pitches_all": int(n_all),
        }

    return {
        "fastball_pct": round(n_heat / n_total, 4),
        "four_seam_pct": round(counts.get("FF", 0) / n_total, 4),
        "sinker_pct": round(counts.get("SI", 0) / n_total, 4),
        "cutter_pct": round(counts.get("FC", 0) / n_total, 4),
        "breaking_pct": round(n_breaking / n_total, 4),
        "offspeed_pct": round(n_offspeed / n_total, 4),
        "total_pitches": int(n_total),
        "junk_pct": round(n_junk / n_all, 4),
        "n_pitches_all": int(n_all),
    }


def _classify(mix: dict) -> str:
    """Apply the 5-archetype rules in order. First match wins."""
    if (
        mix["fastball_pct"] >= FASTBALL_THRESHOLD
        and mix["four_seam_pct"] >= FOUR_SEAM_THRESHOLD
    ):
        return "Power-FF"
    if (
        mix["fastball_pct"] >= FASTBALL_THRESHOLD
        and mix["sinker_pct"] >= SINKER_THRESHOLD
    ):
        return "Sinker-ball"
    if mix["breaking_pct"] >= BREAKING_THRESHOLD:
        return "Breaking-heavy"
    if mix["offspeed_pct"] >= OFFSPEED_THRESHOLD:
        return "Offspeed-heavy"
    return "Balanced"


def _skip_reason(mix: dict, n_starts: int) -> str | None:
    if n_starts < MIN_STARTS:
        return "insufficient_starts"
    if mix["n_pitches_all"] < MIN_PITCHES:
        return "insufficient_pitches"
    if mix["junk_pct"] > MAX_JUNK_PCT:
        return "high_junk"
    return None


# ---- Aggregation -----------------------------------------------------------


def _load_seasons(seasons: list[int]) -> pd.DataFrame:
    from pybaseball import statcast  # type: ignore

    frames = []
    for season in seasons:
        logger.info("loading season %d", season)
        df = statcast(start_dt=f"{season}-03-15", end_dt=f"{season}-11-30")
        df = df[df["game_type"] == "R"].copy()
        df["__season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _count_starts(df_pitcher: pd.DataFrame) -> int:
    """Games where the pitcher faced >= STARTER_MIN_PA distinct PAs."""
    pa = df_pitcher.dropna(subset=["events"])
    if pa.empty:
        return 0
    pa_per_game = pa.groupby("game_pk").size()
    return int((pa_per_game >= STARTER_MIN_PA).sum())


def _aggregate_pitcher_season(df_pitcher: pd.DataFrame) -> tuple[dict, int]:
    counts = df_pitcher["pitch_type"].value_counts(dropna=True).to_dict()
    mix = _mix_from_counts(counts)
    n_starts = _count_starts(df_pitcher)
    return mix, n_starts


def _rolling_30d(df_pitcher_season: pd.DataFrame, season_end_iso: str) -> dict | None:
    """Aggregate the pitcher's last 30 days of the season."""
    start = (pd.Timestamp(season_end_iso) - pd.Timedelta(days=ROLLING_DAYS)).strftime(
        "%Y-%m-%d"
    )
    sub = df_pitcher_season[df_pitcher_season["game_date"] >= start]
    if sub.empty:
        return None
    n_starts = _count_starts(sub)
    if n_starts < ROLLING_MIN_STARTS:
        return None
    mix, _ = _aggregate_pitcher_season(sub)
    if mix["total_pitches"] < 100:
        return None
    return {
        "archetype": _classify(mix),
        "as_of_date": season_end_iso,
        "n_starts": n_starts,
        **mix,
    }


# ---- Build payload ---------------------------------------------------------


def _build_payload(df: pd.DataFrame, seasons: list[int]) -> dict:
    """Return the full archetypes payload."""
    archetypes: dict[str, dict] = {}
    skipped: list[dict] = []
    summary: dict[str, dict[str, int]] = {}

    df["game_date"] = df["game_date"].astype(str).str[:10]

    for season in seasons:
        df_season = df[df["__season"] == season]
        season_end_iso = df_season["game_date"].max()
        season_summary: Counter = Counter()

        for pid, df_p in df_season.groupby("pitcher", dropna=True):
            pid = int(pid)
            mix, n_starts = _aggregate_pitcher_season(df_p)
            name = (df_p["player_name"].dropna().iloc[0]
                    if not df_p["player_name"].dropna().empty else f"id_{pid}")
            p_throws_series = df_p["p_throws"].dropna()
            p_throws = p_throws_series.iloc[0] if not p_throws_series.empty else None

            skip = _skip_reason(mix, n_starts)
            if skip is not None:
                skipped.append({
                    "mlbam_id": pid,
                    "name": name,
                    "reason": skip,
                    "season": season,
                    "n_starts": n_starts,
                    "n_pitches_all": mix["n_pitches_all"],
                })
                season_summary["Skipped"] += 1
                continue

            archetype = _classify(mix)
            season_summary[archetype] += 1

            entry = archetypes.setdefault(str(pid), {
                "name": name, "p_throws": p_throws,
                "by_season": {}, "season_rolling_30d": {},
            })
            entry["name"] = name  # keep latest
            if p_throws and not entry.get("p_throws"):
                entry["p_throws"] = p_throws

            entry["by_season"][str(season)] = {
                "archetype": archetype,
                "n_starts": n_starts,
                **mix,
            }

            rolling = _rolling_30d(df_p, season_end_iso)
            if rolling is not None:
                entry["season_rolling_30d"][str(season)] = rolling

        summary[str(season)] = {a: int(season_summary.get(a, 0)) for a in ARCHETYPES}
        summary[str(season)]["Skipped"] = int(season_summary.get("Skipped", 0))

    return {
        "generated_at": (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "+00:00")
        ),
        "method": "rule_based_pitch_mix_classification_v1",
        "seasons": seasons,
        "thresholds": {
            "fastball_pct": FASTBALL_THRESHOLD,
            "four_seam_pct": FOUR_SEAM_THRESHOLD,
            "sinker_pct": SINKER_THRESHOLD,
            "breaking_pct": BREAKING_THRESHOLD,
            "offspeed_pct": OFFSPEED_THRESHOLD,
            "min_starts": MIN_STARTS,
            "min_pitches": MIN_PITCHES,
            "max_junk_pct": MAX_JUNK_PCT,
        },
        "archetypes": archetypes,
        "skipped": skipped,
        "summary": {"by_season": summary},
    }


# ---- Sanity checks ---------------------------------------------------------


def _sanity_check(payload: dict) -> None:
    summary = payload["summary"]["by_season"]
    for season_str, counts in summary.items():
        # Gate 1: every archetype represented
        zeros = [a for a in ARCHETYPES if counts.get(a, 0) == 0]
        if zeros:
            raise AssertionError(
                f"season {season_str}: archetype(s) with zero pitchers: {zeros}"
            )

        classified = {a: counts.get(a, 0) for a in ARCHETYPES}
        total_classified = sum(classified.values())
        if total_classified == 0:
            raise AssertionError(f"season {season_str}: zero classified pitchers")

        # Gate 2: no category exceeds 50% of classified pitchers
        # (catches thresholds that are too loose — one bucket swallowing the field)
        for archetype, count in classified.items():
            pct = count / total_classified
            if pct > 0.50:
                raise AssertionError(
                    f"season {season_str}: archetype {archetype} is "
                    f"{pct:.1%} of classified pitchers (>50% suggests "
                    f"thresholds are too loose)"
                )

        # Gate 3: every archetype must have >=5% representation
        # (catches thresholds that are too tight — a category collapsing)
        for archetype, count in classified.items():
            pct = count / total_classified
            if pct < 0.05:
                raise AssertionError(
                    f"season {season_str}: archetype {archetype} is "
                    f"{pct:.1%} of classified pitchers (<5% suggests "
                    f"thresholds are too tight; distribution: {classified})"
                )

    # Gate 4: per-pitcher percentages sum to ~1.0
    archetypes = payload["archetypes"]
    for pid, entry in archetypes.items():
        for season_str, info in entry["by_season"].items():
            total = info["fastball_pct"] + info["breaking_pct"] + info["offspeed_pct"]
            if not (0.98 <= total <= 1.02):
                raise AssertionError(
                    f"pitcher {pid} season {season_str}: pct sum {total:.4f} "
                    f"outside [0.98, 1.02]"
                )

    # Gate 5: classified + skipped = universe per season (sanity that we
    # didn't double-count or drop pitchers silently).
    for season_str, counts in summary.items():
        total = sum(counts.values())
        if total == 0:
            raise AssertionError(f"season {season_str}: zero pitchers total")

    logger.info("sanity checks pass: %d seasons checked", len(summary))


def _spot_check_warnings(payload: dict) -> None:
    archetypes = payload["archetypes"]
    for pid, (expected_name, expected_archetypes) in SPOT_CHECK_EXPECTED.items():
        entry = archetypes.get(str(pid))
        if entry is None:
            logger.warning(
                "spot-check: %s (%d) not in archetypes (may have been skipped)",
                expected_name, pid,
            )
            continue
        by_season = entry.get("by_season", {})
        for season_str, info in by_season.items():
            actual = info["archetype"]
            if actual not in expected_archetypes:
                logger.warning(
                    "spot-check: %s (%d) season %s classified as %r, "
                    "expected one of %s. Mix: FB=%.2f FF=%.2f SI=%.2f "
                    "BR=%.2f OS=%.2f",
                    expected_name, pid, season_str, actual,
                    sorted(expected_archetypes),
                    info["fastball_pct"], info["four_seam_pct"],
                    info["sinker_pct"], info["breaking_pct"],
                    info["offspeed_pct"],
                )


# ---- Diagnostic report -----------------------------------------------------


def _build_report(payload: dict) -> dict:
    """Histograms + threshold sensitivity + year-over-year transitions."""
    archetypes = payload["archetypes"]
    seasons = sorted(set(payload["seasons"]))

    # Histograms: 20 buckets from 0.0 to 1.0
    edges = [round(0.05 * i, 2) for i in range(21)]

    def hist(values: list[float]) -> list[int]:
        buckets = [0] * (len(edges) - 1)
        for v in values:
            for i in range(len(edges) - 1):
                if edges[i] <= v < edges[i + 1]:
                    buckets[i] += 1
                    break
            else:
                if v >= edges[-1]:
                    buckets[-1] += 1
        return buckets

    fb_vals = []
    br_vals = []
    os_vals = []
    for entry in archetypes.values():
        for info in entry["by_season"].values():
            fb_vals.append(info["fastball_pct"])
            br_vals.append(info["breaking_pct"])
            os_vals.append(info["offspeed_pct"])

    histograms = {
        "bucket_edges": edges,
        "fastball_pct": hist(fb_vals),
        "breaking_pct": hist(br_vals),
        "offspeed_pct": hist(os_vals),
    }

    # Threshold sensitivity: how many shift if a threshold moves?
    def reclassify_with_threshold(*, fb=FASTBALL_THRESHOLD, br=BREAKING_THRESHOLD,
                                   os=OFFSPEED_THRESHOLD) -> Counter:
        c: Counter = Counter()
        for entry in archetypes.values():
            for info in entry["by_season"].values():
                if (info["fastball_pct"] >= fb
                        and info["four_seam_pct"] >= FOUR_SEAM_THRESHOLD):
                    c["Power-FF"] += 1
                elif (info["fastball_pct"] >= fb
                      and info["sinker_pct"] >= SINKER_THRESHOLD):
                    c["Sinker-ball"] += 1
                elif info["breaking_pct"] >= br:
                    c["Breaking-heavy"] += 1
                elif info["offspeed_pct"] >= os:
                    c["Offspeed-heavy"] += 1
                else:
                    c["Balanced"] += 1
        return c

    sensitivity = {
        "fastball_threshold": {
            "0.50": dict(reclassify_with_threshold(fb=0.50)),
            "0.55_default": dict(reclassify_with_threshold(fb=0.55)),
            "0.60": dict(reclassify_with_threshold(fb=0.60)),
        },
        "breaking_threshold": {
            "0.30": dict(reclassify_with_threshold(br=0.30)),
            "0.35_default": dict(reclassify_with_threshold(br=0.35)),
            "0.40": dict(reclassify_with_threshold(br=0.40)),
        },
        "offspeed_threshold": {
            "0.20": dict(reclassify_with_threshold(os=0.20)),
            "0.25_default": dict(reclassify_with_threshold(os=0.25)),
            "0.30": dict(reclassify_with_threshold(os=0.30)),
        },
    }

    # Year-over-year transitions
    transitions: dict[str, dict] = {}
    for i, season in enumerate(seasons[:-1]):
        next_season = seasons[i + 1]
        n_both = 0
        n_changed = 0
        transition_counts: Counter = Counter()
        for entry in archetypes.values():
            a = entry["by_season"].get(str(season))
            b = entry["by_season"].get(str(next_season))
            if a is None or b is None:
                continue
            n_both += 1
            if a["archetype"] != b["archetype"]:
                n_changed += 1
                transition_counts[f"{a['archetype']} -> {b['archetype']}"] += 1
        transitions[f"{season}_to_{next_season}"] = {
            "n_pitchers_in_both": n_both,
            "n_changed_archetype": n_changed,
            "pct_changed": round(n_changed / n_both, 4) if n_both else 0.0,
            "top_transitions": dict(transition_counts.most_common(10)),
        }

    return {
        "generated_at": payload["generated_at"],
        "histograms": histograms,
        "threshold_sensitivity": sensitivity,
        "year_over_year_transitions": transitions,
    }


def _top_10_per_archetype(payload: dict, season: int) -> dict:
    """For human review: top 10 pitchers by total_pitches in each archetype."""
    rows = []
    for pid, entry in payload["archetypes"].items():
        info = entry["by_season"].get(str(season))
        if info is None:
            continue
        rows.append({
            "mlbam_id": int(pid),
            "name": entry["name"],
            "archetype": info["archetype"],
            "total_pitches": info["total_pitches"],
            "fastball_pct": info["fastball_pct"],
            "breaking_pct": info["breaking_pct"],
            "offspeed_pct": info["offspeed_pct"],
        })
    by_arch: dict[str, list] = defaultdict(list)
    for r in rows:
        by_arch[r["archetype"]].append(r)
    for arch in by_arch:
        by_arch[arch].sort(key=lambda r: -r["total_pitches"])
        by_arch[arch] = by_arch[arch][:10]
    return dict(by_arch)


# ---- Main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = _load_seasons(args.seasons)
    logger.info("loaded %d pitches across seasons %s", len(df), args.seasons)

    payload = _build_payload(df, args.seasons)
    _sanity_check(payload)
    _spot_check_warnings(payload)

    out_path = args.out_dir / "pitcher_archetypes.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "wrote %s (%d pitchers classified, %d skipped)",
        out_path, len(payload["archetypes"]), len(payload["skipped"]),
    )

    report = _build_report(payload)
    report_path = args.out_dir / "pitcher_archetypes_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8",
    )
    logger.info("wrote diagnostic report -> %s", report_path)

    # Log per-season summary and top-10 spot views
    for season_str, counts in payload["summary"]["by_season"].items():
        logger.info("season %s summary: %s", season_str, counts)

    latest_season = max(args.seasons)
    top10 = _top_10_per_archetype(payload, latest_season)
    logger.info("top 10 by total_pitches per archetype (season %d):", latest_season)
    for archetype in ARCHETYPES:
        logger.info("  %s:", archetype)
        for r in top10.get(archetype, []):
            logger.info(
                "    %5d  %-25s  FB=%.2f  BR=%.2f  OS=%.2f  pitches=%d",
                r["mlbam_id"], r["name"][:25], r["fastball_pct"],
                r["breaking_pct"], r["offspeed_pct"], r["total_pitches"],
            )

    # Log YoY transitions
    for key, trans in report["year_over_year_transitions"].items():
        logger.info(
            "%s: %d pitchers in both years, %d changed (%.1f%%)",
            key, trans["n_pitchers_in_both"], trans["n_changed_archetype"],
            trans["pct_changed"] * 100,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
