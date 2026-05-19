"""Phase 3-v2c-v: Integration smoke for the per-batter projector.

Picks 20 starter games from 2025 across 5 archetypes (4 each), builds
historical bundles, hydrates them with the v2c-i contract additions, runs
project() on each, and reports batch sanity checks.

Output: backend/data/phase3_v2c_v_smoke_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from scripts.derive_pa_distribution import _filter_sp_pas, _assign_tto
from scripts.derive_park_k_factors import TEAM_TO_VENUE_ID
from scripts.fit_feature_coefficients import _build_lineup_for_game
from scripts.historical_bundle import (
    GameRecord,
    build_batter_cache,
    build_bundle,
    build_pitcher_cache,
)
from src.projection.inputs import (
    PADistribution,
    ParkKFactorsByHand,
    PitcherArchetype,
    ProjectionBundle,
    ProjectionContext,
    TTOMultipliers,
)
from src.projection.projector import project

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
# Phase 4c-v2 Step 3 verification: write to a NEW file so the Phase 3-v2c-v
# reference output on disk is preserved for delta comparison.
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "phase4c_v2_step3_verification.json"
REFERENCE_FILE = Path(__file__).resolve().parents[1] / "data" / "phase3_v2c_v_smoke_results.json"

ARCHETYPES = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced")
N_PER_ARCHETYPE = 4
SEASON = 2025

# Batch sanity-check parameters
E_BF_GAME_RANGE = (16.0, 33.0)
E_BF_MEAN_RANGE = (22.0, 28.0)
E_K_GAME_RANGE = (2.0, 12.0)
E_K_MEAN_RANGE = (4.5, 7.0)
INDIVIDUAL_K_SOURCE_FLOOR = 7  # at least 7/9 batters using "individual" rate


# ---- Step 1: pitcher selection ---------------------------------------------


def _select_pitchers_per_archetype(
    archetypes_blob: dict, season: int, n_per: int,
) -> dict[str, list[tuple[int, str, str]]]:
    """For each archetype, pick the n_per pitchers with the most starts in
    the requested season. Returns {archetype: [(pitcher_id, name, hand)]}.
    """
    by_arch: dict[str, list[tuple[int, str, str, int]]] = {a: [] for a in ARCHETYPES}
    for pid_str, entry in archetypes_blob["archetypes"].items():
        by_season = entry.get("by_season", {})
        info = by_season.get(str(season))
        if not info:
            continue
        arch = info.get("archetype")
        if arch not in by_arch:
            continue
        by_arch[arch].append((
            int(pid_str),
            entry.get("name") or f"id_{pid_str}",
            entry.get("p_throws") or "R",
            int(info.get("n_starts") or 0),
        ))
    selected: dict[str, list[tuple[int, str, str]]] = {}
    for arch, candidates in by_arch.items():
        candidates.sort(key=lambda t: t[3], reverse=True)
        selected[arch] = [(pid, name, hand) for pid, name, hand, _ in candidates[:n_per]]
        if len(selected[arch]) < n_per:
            logger.warning(
                "archetype %s: only %d candidates available (wanted %d)",
                arch, len(selected[arch]), n_per,
            )
    return selected


# ---- Step 2: game selection ------------------------------------------------


def _starter_games_for_pitcher(
    pitches_all: pd.DataFrame, pitcher_id: int, min_pa: int = 18,
) -> list[dict]:
    """Find games in 2025 where pitcher_id was the starter (min_pa PAs faced).

    Returns a list of dicts with game_pk, game_date, home_team, away_team,
    observed_bf, observed_k.
    """
    p_rows = pitches_all[pitches_all["pitcher"] == pitcher_id]
    if p_rows.empty:
        return []
    p_pa = p_rows.dropna(subset=["events"])
    games = []
    for game_pk, g in p_pa.groupby("game_pk", dropna=True):
        if len(g) < min_pa:
            continue
        if int(g["inning"].min()) != 1:
            continue
        first_row = p_rows[p_rows["game_pk"] == game_pk].iloc[0]
        games.append({
            "game_pk": int(game_pk),
            "game_date": pd.to_datetime(
                str(first_row["game_date"])[:10]
            ).date(),
            "home_team": str(first_row.get("home_team")),
            "away_team": str(first_row.get("away_team")),
            "inning_topbot_first": str(first_row.get("inning_topbot") or "Top"),
            "observed_bf": int(len(g)),
            "observed_k": int(g["events"].isin(
                ("strikeout", "strikeout_double_play")
            ).sum()),
        })
    return games


# ---- Step 3: bundle hydration ----------------------------------------------


def _hydrate_bundle(
    bundle: ProjectionBundle,
    *,
    archetypes_blob: dict,
    tto_table: TTOMultipliers,
    pa_dist: PADistribution,
    park_k_path: Path,
) -> ProjectionBundle:
    """Attach the four Phase 3-v2c-i additive fields to a historical bundle.

    Uses :func:`dataclasses.replace` so we don't round-trip through JSON.
    """
    pa_lookup = PitcherArchetype.from_archetypes_lookup(
        archetypes_blob,
        bundle.metadata.pitcher_mlbam_id,
        bundle.metadata.game_date.year,
    )
    park_lookup = ParkKFactorsByHand.from_json_lookup(
        park_k_path, bundle.game_context.venue_id,
    )
    return replace(
        bundle,
        pitcher_archetype=pa_lookup,
        tto_multipliers=tto_table,
        park_k_factors_by_hand=park_lookup,
        pa_distribution=pa_dist,
    )


# ---- Step 4: smoke loop ----------------------------------------------------


def _run_smoke(seed: int = 42) -> dict:
    """Run the 20-game smoke and return the results dict."""
    logger.info("loading archetypes / static lookups")
    archetypes_blob = json.loads(
        (PROCESSED_DIR / "pitcher_archetypes.json").read_text(encoding="utf-8")
    )
    tto_table = TTOMultipliers.from_json(PROCESSED_DIR / "tto_multipliers.json")
    pa_dist = PADistribution.from_json(PROCESSED_DIR / "pa_distribution_by_bf.json")

    selected = _select_pitchers_per_archetype(archetypes_blob, SEASON, N_PER_ARCHETYPE)
    logger.info("selected pitchers per archetype:")
    for arch, plist in selected.items():
        logger.info("  %s: %s", arch, [name for _, name, _ in plist])

    logger.info("pulling cached %d Statcast (full season)", SEASON)
    from pybaseball import statcast
    df = statcast(start_dt=f"{SEASON}-03-15", end_dt=f"{SEASON}-11-30")
    df = df[df["game_type"] == "R"].copy()

    logger.info("building pitcher + batter caches (~30s)")
    pitcher_cache = build_pitcher_cache(df)
    batter_cache = build_batter_cache(df)

    ctx = ProjectionContext.from_default_paths()
    results: list[dict] = []
    failures: list[dict] = []

    pa_by_game = df.dropna(subset=["events", "pitcher", "batter"]).groupby("game_pk")

    for arch, pitcher_list in selected.items():
        for pid, name, hand in pitcher_list:
            games = _starter_games_for_pitcher(df, pid)
            if not games:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": "no starter games found"})
                continue
            # Pick the median game by date for stability
            games.sort(key=lambda g: g["game_date"])
            chosen = games[len(games) // 2]

            game_pk = chosen["game_pk"]
            try:
                pa_in_game = pa_by_game.get_group(game_pk)
            except KeyError:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": f"game_pk {game_pk} not in pa_by_game"})
                continue
            lineup = _build_lineup_for_game(pa_in_game, pid)
            if len(lineup) < 9:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": f"lineup only {len(lineup)} batters"})
                continue

            venue_id = TEAM_TO_VENUE_ID.get(chosen["home_team"])
            if venue_id is None:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": f"unknown home_team {chosen['home_team']}"})
                continue

            # Determine pitcher's team. If first inning was Top, away team
            # was batting -> pitcher is home team. If Bot, pitcher is away.
            is_home = chosen["inning_topbot_first"] == "Top"
            pitcher_team = chosen["home_team"] if is_home else chosen["away_team"]
            opposing_team = chosen["away_team"] if is_home else chosen["home_team"]

            record = GameRecord(
                season=SEASON,
                game_pk=game_pk,
                game_date=chosen["game_date"],
                pitcher_id=pid,
                pitcher_hand=hand,
                pitcher_team=pitcher_team,
                opposing_team=opposing_team,
                venue_id=venue_id,
                is_home=is_home,
                opposing_batters=lineup,
                observed_bf=chosen["observed_bf"],
                observed_k=chosen["observed_k"],
            )

            bundle = build_bundle(record, pitcher_cache, batter_cache)
            bundle = _hydrate_bundle(
                bundle,
                archetypes_blob=archetypes_blob,
                tto_table=tto_table,
                pa_dist=pa_dist,
                park_k_path=PROCESSED_DIR / "park_k_factors.json",
            )

            try:
                result = project(bundle, ctx)
            except Exception as exc:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": f"project() raised: {exc}"})
                continue

            if result.skipped:
                failures.append({"archetype": arch, "pitcher": name,
                                  "reason": f"projector skipped: {result.skip_reason}"})
                continue

            # Per-batter source breakdown (skip the Phase 4c-v2 Step 3 _meta key)
            n_individual = 0
            n_league = 0
            if result.per_batter_breakdown:
                for k, entry in result.per_batter_breakdown.items():
                    if k == "_meta":
                        continue
                    if entry["k_rate_source"] == "individual":
                        n_individual += 1
                    elif entry["k_rate_source"] == "league_fallback":
                        n_league += 1

            blend_meta = (result.per_batter_breakdown or {}).get("_meta") or {}

            results.append({
                "archetype_assigned": arch,
                "pitcher_id": pid,
                "pitcher_name": name,
                "game_pk": game_pk,
                "game_date": chosen["game_date"].isoformat(),
                "observed_bf": chosen["observed_bf"],
                "observed_k": chosen["observed_k"],
                "e_bf": round(result.e_bf, 3),
                "p_k_pa": round(result.p_k_pa, 4),
                "e_k": round(result.e_k, 3),
                "projection_method": result.projection_method,
                "archetype_used": result.archetype_used,
                "pa_distribution_bf_used": result.pa_distribution_bf_used,
                "n_batters_individual": n_individual,
                "n_batters_league_fallback": n_league,
                # Phase 4c-v2 Step 3 metadata: empty dict for legacy bundles
                # that didn't take the new path.
                "blend_meta": blend_meta,
            })

    return {"results": results, "failures": failures}


# ---- Step 5: batch sanity checks -------------------------------------------


def _sanity_check_batch(results: list[dict]) -> dict:
    """Run the batch sanity checks. Returns a dict of check_name -> pass/fail/details.

    Halts on systematic failure (2+ games failing the same individual-game
    range check, or any failure of the aggregate checks).
    """
    summary: dict = {}

    bf_values = [r["e_bf"] for r in results]
    ek_values = [r["e_k"] for r in results]

    # Per-game range checks (count failures, halt if >=2 same failure)
    bf_low_fails = [r for r in results if r["e_bf"] < E_BF_GAME_RANGE[0]]
    bf_high_fails = [r for r in results if r["e_bf"] > E_BF_GAME_RANGE[1]]
    ek_low_fails = [r for r in results if r["e_k"] < E_K_GAME_RANGE[0]]
    ek_high_fails = [r for r in results if r["e_k"] > E_K_GAME_RANGE[1]]

    summary["per_game_e_bf_range"] = {
        "range": list(E_BF_GAME_RANGE),
        "low_fails": [r["pitcher_name"] for r in bf_low_fails],
        "high_fails": [r["pitcher_name"] for r in bf_high_fails],
        "pass": len(bf_low_fails) + len(bf_high_fails) < 2,
    }
    summary["per_game_e_k_range"] = {
        "range": list(E_K_GAME_RANGE),
        "low_fails": [r["pitcher_name"] for r in ek_low_fails],
        "high_fails": [r["pitcher_name"] for r in ek_high_fails],
        "pass": len(ek_low_fails) + len(ek_high_fails) < 2,
    }

    # Mean checks
    bf_mean = mean(bf_values) if bf_values else 0.0
    ek_mean = mean(ek_values) if ek_values else 0.0
    summary["mean_e_bf"] = {
        "value": round(bf_mean, 3), "range": list(E_BF_MEAN_RANGE),
        "pass": E_BF_MEAN_RANGE[0] <= bf_mean <= E_BF_MEAN_RANGE[1],
    }
    summary["mean_e_k"] = {
        "value": round(ek_mean, 3), "range": list(E_K_MEAN_RANGE),
        "pass": E_K_MEAN_RANGE[0] <= ek_mean <= E_K_MEAN_RANGE[1],
    }

    # Per-archetype e_k means (reported only, no ordering gate)
    by_arch: dict[str, list[float]] = {a: [] for a in ARCHETYPES}
    for r in results:
        by_arch[r["archetype_assigned"]].append(r["e_k"])
    arch_means = {
        a: round(mean(vals), 3) if vals else None for a, vals in by_arch.items()
    }
    summary["mean_e_k_by_archetype"] = arch_means

    # NOTE: We do NOT check "Power-FF archetype mean e_k > Balanced mean e_k".
    # In a small batch (4 games per archetype), pitcher-level K-skill
    # variation dominates archetype-level TTO effects. The model is designed
    # so individual signal (CSW%, K%) outranks archetype as the primary
    # driver — archetype contributes only via TTO multipliers (~3-5% effect).
    # A 4-game sample can't isolate the archetype effect from individual
    # selection. Validated separately via TTO multiplier derivation
    # (Phase 3-v2b) which used 275k SP PAs.

    # No archetype has all games below league mean K rate (~5.0)
    LEAGUE_K_GAME_MEAN = 5.0
    archetype_all_below = []
    for a, vals in by_arch.items():
        if vals and all(v < LEAGUE_K_GAME_MEAN for v in vals):
            archetype_all_below.append(a)
    summary["no_archetype_all_below_league"] = {
        "league_mean": LEAGUE_K_GAME_MEAN,
        "archetypes_with_all_below": archetype_all_below,
        "pass": len(archetype_all_below) == 0,
    }

    # Individual-rate floor (avg across games)
    individual_floor_fails = [
        r for r in results
        if r["n_batters_individual"] < INDIVIDUAL_K_SOURCE_FLOOR
    ]
    summary["individual_k_source_floor"] = {
        "floor_per_game": INDIVIDUAL_K_SOURCE_FLOOR,
        "fails": [
            {"pitcher": r["pitcher_name"],
             "n_individual": r["n_batters_individual"],
             "n_league": r["n_batters_league_fallback"]}
            for r in individual_floor_fails
        ],
        "pass": len(individual_floor_fails) < 2,
    }

    return summary


# ---- Step 3 verification: comparison vs Phase 3-v2c-v reference ------------


def _compare_against_reference(new_results: list[dict]) -> dict:
    """If the Phase 3-v2c-v reference file exists, compute per-pitcher and
    per-archetype e_k deltas. Returns dict with deltas, summary, and any
    games flagged with |delta| > 1.0."""
    if not REFERENCE_FILE.exists():
        return {"status": "no_reference_file", "path": str(REFERENCE_FILE)}
    try:
        ref_blob = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "reference_unreadable", "error": str(exc)}
    ref_by_pid = {r["pitcher_id"]: r for r in ref_blob.get("results") or []}

    deltas: list[dict] = []
    by_arch_delta: dict[str, list[float]] = {a: [] for a in ARCHETYPES}
    big_shifts: list[dict] = []
    for r in new_results:
        ref = ref_by_pid.get(r["pitcher_id"])
        if ref is None:
            continue
        delta = r["e_k"] - ref["e_k"]
        rec = {
            "pitcher_name": r["pitcher_name"],
            "archetype": r["archetype_assigned"],
            "old_e_k": ref["e_k"],
            "new_e_k": r["e_k"],
            "delta": round(delta, 3),
            "blend_confidence": (r.get("blend_meta") or {}).get("blend_confidence"),
        }
        deltas.append(rec)
        by_arch_delta[r["archetype_assigned"]].append(delta)
        if abs(delta) > 1.0:
            big_shifts.append(rec)

    arch_summary = {}
    for arch, vals in by_arch_delta.items():
        if not vals:
            continue
        arch_summary[arch] = {
            "n_games": len(vals),
            "mean_delta": round(mean(vals), 3),
            "min_delta": round(min(vals), 3),
            "max_delta": round(max(vals), 3),
        }

    all_deltas = [r["delta"] for r in deltas]
    n_down = sum(1 for d in all_deltas if d < 0)
    n_up = sum(1 for d in all_deltas if d > 0)
    n_flat = sum(1 for d in all_deltas if d == 0)

    return {
        "status": "compared",
        "n_games_compared": len(deltas),
        "n_down_shift": n_down,
        "n_up_shift": n_up,
        "n_flat": n_flat,
        "overall_mean_delta": round(mean(all_deltas), 3) if all_deltas else 0.0,
        "overall_min_delta": round(min(all_deltas), 3) if all_deltas else 0.0,
        "overall_max_delta": round(max(all_deltas), 3) if all_deltas else 0.0,
        "by_archetype": arch_summary,
        "per_pitcher_deltas": sorted(deltas, key=lambda d: d["delta"]),
        "big_shifts_flagged": big_shifts,
    }


def _blend_confidence_dist(new_results: list[dict]) -> dict:
    """Tally blend_confidence labels across the batch."""
    counts: dict[str, int] = {}
    for r in new_results:
        meta = r.get("blend_meta") or {}
        label = meta.get("blend_confidence") or "missing"
        counts[label] = counts.get(label, 0) + 1
    return counts


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    payload = _run_smoke(seed=args.seed)
    payload["sanity_checks"] = _sanity_check_batch(payload["results"])

    # Per-archetype summary block
    by_arch_summary: dict = {}
    for arch in ARCHETYPES:
        vals = [r for r in payload["results"] if r["archetype_assigned"] == arch]
        if not vals:
            continue
        e_ks = [r["e_k"] for r in vals]
        e_bfs = [r["e_bf"] for r in vals]
        by_arch_summary[arch] = {
            "n_games": len(vals),
            "e_k_mean": round(mean(e_ks), 3),
            "e_k_std": round(stdev(e_ks), 3) if len(e_ks) > 1 else 0.0,
            "e_k_min": round(min(e_ks), 3),
            "e_k_max": round(max(e_ks), 3),
            "e_bf_mean": round(mean(e_bfs), 3),
        }
    payload["per_archetype_summary"] = by_arch_summary

    # Phase 4c-v2 Step 3 verification: compare against the Phase 3-v2c-v
    # reference results (if present) to confirm the CSW-blend wiring shifts
    # are consistent and not Rasmussen-specific.
    payload["step3_verification"] = _compare_against_reference(payload["results"])
    payload["blend_confidence_distribution"] = _blend_confidence_dist(payload["results"])

    OUT_FILE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", OUT_FILE)

    # Echo the report
    logger.info("=== Per-archetype summary ===")
    for arch, summ in by_arch_summary.items():
        logger.info(
            "  %s: n=%d, e_k=%.2f±%.2f [%.2f-%.2f], e_bf=%.2f",
            arch, summ["n_games"], summ["e_k_mean"], summ["e_k_std"],
            summ["e_k_min"], summ["e_k_max"], summ["e_bf_mean"],
        )

    logger.info("=== Sanity checks ===")
    all_pass = True
    for name, check in payload["sanity_checks"].items():
        if isinstance(check, dict) and "pass" in check:
            status = "PASS" if check["pass"] else "FAIL"
            if not check["pass"]:
                all_pass = False
            logger.info("  [%s] %s", status, name)
        else:
            logger.info("  %s: %s", name, check)

    if payload["failures"]:
        logger.warning("Failed bundles: %d", len(payload["failures"]))
        for f in payload["failures"]:
            logger.warning("  %s (%s): %s", f["pitcher"], f["archetype"], f["reason"])

    # ---- Step 3 verification report ----
    ver = payload.get("step3_verification") or {}
    if ver.get("status") == "compared":
        logger.info("=== Step 3 verification: e_k delta vs Phase 3-v2c-v ===")
        logger.info(
            "  n_games=%d  down=%d  up=%d  flat=%d  mean_delta=%+.3f  range=[%+.3f, %+.3f]",
            ver["n_games_compared"], ver["n_down_shift"], ver["n_up_shift"],
            ver["n_flat"], ver["overall_mean_delta"],
            ver["overall_min_delta"], ver["overall_max_delta"],
        )
        logger.info("  per-archetype mean delta:")
        for arch, summ in ver.get("by_archetype", {}).items():
            logger.info(
                "    %s: mean_delta=%+.3f range [%+.3f, %+.3f] (n=%d)",
                arch, summ["mean_delta"], summ["min_delta"],
                summ["max_delta"], summ["n_games"],
            )
        big = ver.get("big_shifts_flagged") or []
        if big:
            logger.warning("  big shifts (|delta| > 1.0):")
            for b in big:
                logger.warning(
                    "    %s (%s): %.2f -> %.2f (delta=%+.3f, conf=%s)",
                    b["pitcher_name"], b["archetype"],
                    b["old_e_k"], b["new_e_k"], b["delta"], b["blend_confidence"],
                )
    else:
        logger.info("=== Step 3 verification: no reference file (skipping) ===")

    conf_dist = payload.get("blend_confidence_distribution") or {}
    if conf_dist:
        logger.info("=== Blend confidence distribution ===")
        for label, n in sorted(conf_dist.items(), key=lambda kv: -kv[1]):
            logger.info("  %s: %d", label, n)

    return 0 if all_pass and len(payload["results"]) >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())
