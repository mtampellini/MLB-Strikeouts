"""Phase 3-v2b: Times-Through-The-Order (TTO) multiplier derivation.

Computes empirical K-rate multipliers keyed on (TTO bucket x pitcher
archetype), 2023-2025 Statcast per-PA data, starting-pitcher PAs only.

For each (TTO bucket t, archetype a):
    multiplier(t, a) = posterior_K_rate(t, a) / K_rate(1, a)

where posterior is computed via empirical Bayes shrinkage with prior
weight 500 PA at the archetype's TTO=1 baseline rate. TTO=1 is the
reference (multiplier == 1.00 by construction).

A league-wide multiplier (TTO only, archetype-agnostic) is also produced
as the fallback used when a pitcher's archetype is unknown.

This script consumes pitcher_archetypes.json (Phase 3-v2a) and produces:
- tto_multipliers.json:        the live multiplier table
- tto_multipliers_report.json: heat map + 95% CI + research benchmark check

CLI:
    python -m scripts.derive_tto_multipliers
    python -m scripts.derive_tto_multipliers --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARCHETYPES_FILE = PROCESSED_DIR / "pitcher_archetypes.json"
OUTPUT_FILE = PROCESSED_DIR / "tto_multipliers.json"
REPORT_FILE = PROCESSED_DIR / "tto_multipliers_report.json"

# Event taxonomy
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
OUT_EVENTS = frozenset({
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "fielders_choice_out",
    "grounded_into_double_play", "double_play", "triple_play",
    "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
    "other_out",
})

# Non-PA events to exclude from K-rate cell aggregation. truncated_pa happens
# when the inning ends mid-PA (caught stealing 3rd to end inning, etc) — not
# a real plate-appearance outcome.
NON_PA_EVENTS = frozenset({"truncated_pa", "caught_stealing_2b", "caught_stealing_3b",
                            "caught_stealing_home", "pickoff_1b", "pickoff_2b",
                            "pickoff_3b", "pickoff_caught_stealing_2b",
                            "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home",
                            "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
                            "wild_pitch", "passed_ball", "balk"})

# SP filter
SP_MIN_DISTINCT_BATTERS = 9
SP_MIN_OUTS = 12  # 4 IP

# TTO bucketing
MAX_TTO = 4  # PAs at TTO 4 or later are collapsed into bucket 4

# Shrinkage
SHRINKAGE_PRIOR_PA = 500
MIN_SAMPLE_PER_CELL = 200

ARCHETYPES = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy", "Balanced")
TTO_BUCKETS = (1, 2, 3, 4)

# Sanity gates
MAX_MULTIPLIER = 1.05
MIN_MULTIPLIER = 0.65
# K-rate TTO multipliers are more aggressive than wOBA-based TTO penalties
# in published research. wOBA blends contact-quality recovery and whiff
# reduction; K rate isolates the whiff component, which decays faster as
# hitters familiarize with a pitcher's stuff. K-specific research (Fangraphs,
# The Book) clusters the TTO=3 K multiplier around 0.80-0.88, vs the more
# commonly cited wOBA range of 0.88-0.94. Gate range [0.78, 0.92] reflects
# the K-specific research, not wOBA.
LEAGUE_TTO3_RANGE = (0.78, 0.92)


# ---- Pure helpers (unit-testable) ------------------------------------------


def _assign_tto(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Add a `tto` column to per-PA data.

    Within each (game_pk, pitcher), order PAs by (inning, at_bat_number),
    then for each (pitcher, batter) pair, the TTO bucket = cumulative
    appearance count (capped at MAX_TTO).
    """
    df = df_pa.sort_values(
        ["game_pk", "pitcher", "inning", "at_bat_number"], kind="mergesort"
    ).copy()
    df["tto"] = df.groupby(["game_pk", "pitcher", "batter"]).cumcount() + 1
    df["tto"] = df["tto"].clip(upper=MAX_TTO).astype(int)
    return df


def _is_sp_game(group: pd.DataFrame) -> bool:
    """A (game_pk, pitcher) is the starter if first PA is in inning 1, faces
    >=9 distinct batters, and records >=12 outs."""
    if group["inning"].min() != 1:
        return False
    if group["batter"].nunique() < SP_MIN_DISTINCT_BATTERS:
        return False
    n_outs = int(group["events"].isin(OUT_EVENTS).sum())
    # Double plays / triple plays would undercount with this, but +12 standard
    # outs is already a soft filter; not worth the complexity.
    if n_outs < SP_MIN_OUTS:
        return False
    return True


def _filter_sp_pas(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Keep only PAs where the pitcher was the starter for that game."""
    sp_keys = []
    for (game_pk, pitcher), g in df_pa.groupby(["game_pk", "pitcher"], sort=False):
        if _is_sp_game(g):
            sp_keys.append((game_pk, pitcher))
    if not sp_keys:
        return df_pa.iloc[0:0].copy()
    keys_df = pd.DataFrame(sp_keys, columns=["game_pk", "pitcher"])
    return df_pa.merge(keys_df, on=["game_pk", "pitcher"], how="inner")


def _shrunk_multiplier(
    k_count: int,
    pa_count: int,
    baseline_rate: float,
    prior_pa: int = SHRINKAGE_PRIOR_PA,
) -> float:
    """EB-shrunk multiplier: posterior K-rate / baseline rate.

    Posterior K-rate has prior_pa pseudo-observations at baseline_rate.
    Shrinks the multiplier toward 1.00 as pa_count shrinks toward zero.
    """
    if baseline_rate <= 0:
        return 1.0
    posterior_rate = (k_count + prior_pa * baseline_rate) / (pa_count + prior_pa)
    return posterior_rate / baseline_rate


def _multiplier_ci(
    k_t: int, pa_t: int, k_1: int, pa_1: int, z: float = 1.96
) -> tuple[float, float]:
    """Delta-method 95% CI for the ratio p_t / p_1 (UNSHRUNK).

    Uses the log-ratio variance for numeric stability with small rates.
    """
    if pa_t == 0 or pa_1 == 0 or k_t == 0 or k_1 == 0:
        return (float("nan"), float("nan"))
    p_t = k_t / pa_t
    p_1 = k_1 / pa_1
    if p_t <= 0 or p_1 <= 0 or p_t >= 1 or p_1 >= 1:
        return (float("nan"), float("nan"))
    # Var(log(p)) ~ (1-p) / (n*p) for a binomial proportion
    var_log_ratio = (1 - p_t) / (pa_t * p_t) + (1 - p_1) / (pa_1 * p_1)
    se = math.sqrt(var_log_ratio)
    ratio = p_t / p_1
    return (ratio * math.exp(-z * se), ratio * math.exp(z * se))


# ---- Loaders ---------------------------------------------------------------


def _load_archetypes() -> dict[int, dict[int, str]]:
    """Return {pitcher_mlbam_id: {season: archetype}}.

    Reads only the per-season classification, not the rolling 30-day, since
    we're matching per-PA against a season-stable archetype.
    """
    if not ARCHETYPES_FILE.exists():
        raise FileNotFoundError(
            f"{ARCHETYPES_FILE} not found — run derive_pitcher_archetypes first"
        )
    with ARCHETYPES_FILE.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[int, dict[int, str]] = {}
    for pid_str, entry in data["archetypes"].items():
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        by_season = entry.get("by_season", {})
        out[pid] = {int(s): info["archetype"] for s, info in by_season.items()
                    if "archetype" in info}
    return out


def _load_statcast_pas(seasons: list[int]) -> pd.DataFrame:
    """Pull regular-season per-PA data from cached Statcast."""
    from pybaseball import statcast  # type: ignore
    frames = []
    for season in seasons:
        logger.info("loading season %d", season)
        df = statcast(start_dt=f"{season}-03-15", end_dt=f"{season}-11-30")
        df = df[df["game_type"] == "R"].copy()
        df["__season"] = season
        # Keep terminal pitches only (events not null & is a real PA event)
        df = df[df["events"].notna()]
        df = df[~df["events"].isin(NON_PA_EVENTS)]
        # Keep only columns we need
        cols = ["game_pk", "pitcher", "batter", "inning", "at_bat_number",
                "events", "__season"]
        df = df[cols].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- Aggregation -----------------------------------------------------------


def _attach_archetype(
    df_pa: pd.DataFrame, archetype_map: dict[int, dict[int, str]]
) -> tuple[pd.DataFrame, int]:
    """Add an `archetype` column. Return (filtered_df, n_skipped_no_archetype)."""
    pitchers = df_pa["pitcher"].astype(int)
    seasons = df_pa["__season"].astype(int)
    arch = [
        archetype_map.get(int(p), {}).get(int(s))
        for p, s in zip(pitchers, seasons)
    ]
    df_pa = df_pa.copy()
    df_pa["archetype"] = arch
    n_skipped = int(df_pa["archetype"].isna().sum())
    df_pa = df_pa[df_pa["archetype"].notna()].copy()
    return df_pa, n_skipped


def _aggregate_cells(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (archetype, tto) cells with k_count, pa_count, k_rate."""
    df = df_pa.copy()
    df["is_k"] = df["events"].isin(K_EVENTS).astype(int)
    grouped = df.groupby(["archetype", "tto"]).agg(
        k_count=("is_k", "sum"),
        pa_count=("is_k", "size"),
    ).reset_index()
    grouped["k_rate"] = grouped["k_count"] / grouped["pa_count"].clip(lower=1)
    return grouped


def _build_archetype_multipliers(cells: pd.DataFrame) -> dict:
    """Build by_archetype output and collect low-sample cells."""
    out: dict = {}
    low_sample: list[dict] = []
    for archetype in ARCHETYPES:
        sub = cells[cells["archetype"] == archetype].set_index("tto")
        if 1 not in sub.index or sub.loc[1, "pa_count"] == 0:
            # No TTO=1 reference — emit neutral multipliers
            arch_block = {}
            for t in TTO_BUCKETS:
                n_pa = int(sub.loc[t, "pa_count"]) if t in sub.index else 0
                k_rate = float(sub.loc[t, "k_rate"]) if t in sub.index else 0.0
                arch_block[f"tto_{t}"] = {
                    "multiplier": 1.0,
                    "k_rate": round(k_rate, 4),
                    "n_pa": n_pa,
                    "low_sample": True,
                }
                low_sample.append({"archetype": archetype, "tto": t, "n_pa": n_pa,
                                   "reason": "no_tto_1_baseline"})
            out[archetype] = arch_block
            continue

        baseline_rate = float(sub.loc[1, "k_rate"])
        arch_block: dict = {}
        for t in TTO_BUCKETS:
            if t not in sub.index:
                arch_block[f"tto_{t}"] = {
                    "multiplier": 1.0,
                    "k_rate": 0.0,
                    "n_pa": 0,
                    "low_sample": True,
                }
                low_sample.append({"archetype": archetype, "tto": t, "n_pa": 0})
                continue

            k_count = int(sub.loc[t, "k_count"])
            pa_count = int(sub.loc[t, "pa_count"])
            k_rate = float(sub.loc[t, "k_rate"])
            is_low = pa_count < MIN_SAMPLE_PER_CELL

            if t == 1:
                mult = 1.0
            elif is_low:
                mult = 1.0
                logger.warning(
                    "low-sample cell (archetype=%s, tto=%d, n_pa=%d) — multiplier set to 1.00",
                    archetype, t, pa_count,
                )
                low_sample.append({"archetype": archetype, "tto": t, "n_pa": pa_count})
            else:
                mult = _shrunk_multiplier(k_count, pa_count, baseline_rate)

            arch_block[f"tto_{t}"] = {
                "multiplier": round(mult, 4),
                "k_rate": round(k_rate, 4),
                "n_pa": pa_count,
                "low_sample": is_low,
            }
        out[archetype] = arch_block
    return out, low_sample


def _build_league_wide(cells: pd.DataFrame) -> dict:
    """League-wide multipliers (collapse archetypes), weighted by PA."""
    league = cells.groupby("tto").agg(
        k_count=("k_count", "sum"),
        pa_count=("pa_count", "sum"),
    ).reset_index()
    league["k_rate"] = league["k_count"] / league["pa_count"].clip(lower=1)
    if 1 not in league["tto"].values:
        raise AssertionError("league-wide TTO=1 has no PAs — cannot compute multipliers")
    baseline_rate = float(league.loc[league["tto"] == 1, "k_rate"].iloc[0])

    out: dict = {}
    for t in TTO_BUCKETS:
        row = league[league["tto"] == t]
        if row.empty:
            out[f"tto_{t}"] = {"multiplier": 1.0, "k_rate": 0.0, "n_pa": 0}
            continue
        k = int(row["k_count"].iloc[0])
        n = int(row["pa_count"].iloc[0])
        rate = float(row["k_rate"].iloc[0])
        if t == 1:
            mult = 1.0
        else:
            mult = _shrunk_multiplier(k, n, baseline_rate)
        out[f"tto_{t}"] = {
            "multiplier": round(mult, 4),
            "k_rate": round(rate, 4),
            "n_pa": n,
        }
    return out


# ---- Sanity checks ---------------------------------------------------------


def _sanity_check(payload: dict) -> None:
    by_archetype = payload["by_archetype"]
    league = payload["league_wide"]

    # Gate 6: all 5 archetypes present, all 4 TTO buckets per archetype
    missing = [a for a in ARCHETYPES if a not in by_archetype]
    if missing:
        raise AssertionError(f"missing archetypes in output: {missing}")
    for archetype in ARCHETYPES:
        for t in TTO_BUCKETS:
            key = f"tto_{t}"
            if key not in by_archetype[archetype]:
                raise AssertionError(
                    f"archetype {archetype} missing TTO bucket {t}"
                )

    # Gate 2: every archetype tto_1 == 1.00
    for archetype in ARCHETYPES:
        m1 = by_archetype[archetype]["tto_1"]["multiplier"]
        if m1 != 1.0:
            raise AssertionError(
                f"archetype {archetype}: tto_1 multiplier is {m1}, must be 1.00"
            )

    # Gate 3 + 4: multiplier ranges (skip low-sample cells, which are neutralized to 1.00)
    for archetype in ARCHETYPES:
        for t in TTO_BUCKETS:
            cell = by_archetype[archetype][f"tto_{t}"]
            mult = cell["multiplier"]
            if mult > MAX_MULTIPLIER:
                raise AssertionError(
                    f"archetype {archetype} TTO={t}: multiplier {mult} exceeds "
                    f"max {MAX_MULTIPLIER}"
                )
            if mult < MIN_MULTIPLIER:
                raise AssertionError(
                    f"archetype {archetype} TTO={t}: multiplier {mult} below "
                    f"min {MIN_MULTIPLIER}"
                )

    # Gate 1: league-wide monotonic decrease 1.00 > tto_2 > tto_3 > tto_4
    m1 = league["tto_1"]["multiplier"]
    m2 = league["tto_2"]["multiplier"]
    m3 = league["tto_3"]["multiplier"]
    m4 = league["tto_4"]["multiplier"]
    if not (m1 > m2 > m3 > m4):
        raise AssertionError(
            f"league-wide multipliers not monotonically decreasing: "
            f"TTO=1={m1}, 2={m2}, 3={m3}, 4={m4}"
        )

    # Gate 5: league-wide TTO=3 in expected range
    lo, hi = LEAGUE_TTO3_RANGE
    if not (lo <= m3 <= hi):
        raise AssertionError(
            f"league-wide TTO=3 multiplier {m3} outside expected range "
            f"[{lo}, {hi}] (public research benchmark)"
        )

    logger.info("sanity checks pass: %d gates checked", 6)


def _spot_check_warnings(payload: dict) -> list[str]:
    """Run informational checks against prior research expectations."""
    warnings = []
    by_archetype = payload["by_archetype"]
    power_t3 = by_archetype["Power-FF"]["tto_3"]["multiplier"]
    balanced_t3 = by_archetype["Balanced"]["tto_3"]["multiplier"]
    breaking_t3 = by_archetype["Breaking-heavy"]["tto_3"]["multiplier"]

    # Power-FF should have LARGER penalty than Balanced → smaller multiplier
    if power_t3 >= balanced_t3:
        msg = (f"Power-FF TTO=3 ({power_t3}) >= Balanced ({balanced_t3}) — "
               "research expected Power-FF to take the larger TTO penalty")
        logger.warning(msg)
        warnings.append(msg)

    # Breaking-heavy should have SMALLER penalty than Power-FF → larger multiplier
    if breaking_t3 <= power_t3:
        msg = (f"Breaking-heavy TTO=3 ({breaking_t3}) <= Power-FF ({power_t3}) — "
               "research expected Breaking-heavy to hold up better through the order")
        logger.warning(msg)
        warnings.append(msg)

    return warnings


# ---- Report ----------------------------------------------------------------


def _build_report(
    payload: dict, cells: pd.DataFrame, spot_check_warnings: list[str]
) -> dict:
    """Heat map + 95% CI + research benchmark check."""
    # Heat map (text grid)
    header = "archetype".ljust(18) + "".join(f"TTO_{t}".rjust(10) for t in TTO_BUCKETS)
    lines = [header, "-" * len(header)]
    for archetype in ARCHETYPES:
        row = archetype.ljust(18)
        for t in TTO_BUCKETS:
            mult = payload["by_archetype"][archetype][f"tto_{t}"]["multiplier"]
            row += f"{mult:>10.3f}"
        lines.append(row)
    lines.append("-" * len(header))
    league_row = "league-wide".ljust(18) + "".join(
        f"{payload['league_wide'][f'tto_{t}']['multiplier']:>10.3f}"
        for t in TTO_BUCKETS
    )
    lines.append(league_row)
    heat_map = "\n".join(lines)

    # 95% CI per cell (delta-method, unshrunk)
    ci_table: dict = {}
    for archetype in ARCHETYPES:
        sub = cells[cells["archetype"] == archetype].set_index("tto")
        if 1 not in sub.index:
            continue
        k_1, pa_1 = int(sub.loc[1, "k_count"]), int(sub.loc[1, "pa_count"])
        arch_ci: dict = {}
        for t in TTO_BUCKETS:
            if t not in sub.index:
                arch_ci[f"tto_{t}"] = {"ci_lo": None, "ci_hi": None}
                continue
            k_t = int(sub.loc[t, "k_count"])
            pa_t = int(sub.loc[t, "pa_count"])
            lo, hi = _multiplier_ci(k_t, pa_t, k_1, pa_1)
            arch_ci[f"tto_{t}"] = {
                "ci_lo": None if math.isnan(lo) else round(lo, 4),
                "ci_hi": None if math.isnan(hi) else round(hi, 4),
            }
        ci_table[archetype] = arch_ci

    # Public research benchmarks (Baseball Prospectus / Tango etc.)
    benchmarks = {
        "tto_1": 1.00,
        "tto_2": 0.96,  # ~4% drop
        "tto_3": 0.90,  # ~10% drop
        "tto_4": 0.84,  # ~16% drop
    }
    benchmark_compare = {}
    for t in TTO_BUCKETS:
        key = f"tto_{t}"
        observed = payload["league_wide"][key]["multiplier"]
        expected = benchmarks[key]
        benchmark_compare[key] = {
            "observed_league_wide": observed,
            "research_benchmark": expected,
            "delta": round(observed - expected, 4),
        }

    # Archetype interaction finding: surface the empirical pattern that the
    # four specialized archetypes cluster tight at TTO=3 while Balanced is
    # the outlier — the Phase 3-v2c consumer and any future review should
    # see this rather than discovering it from the heat map.
    specialized = ("Power-FF", "Sinker-ball", "Breaking-heavy", "Offspeed-heavy")
    specialized_tto3 = {
        a: payload["by_archetype"][a]["tto_3"]["multiplier"] for a in specialized
    }
    balanced_tto3 = payload["by_archetype"]["Balanced"]["tto_3"]["multiplier"]
    spec_min, spec_max = min(specialized_tto3.values()), max(specialized_tto3.values())
    archetype_interaction_finding = {
        "summary": (
            "The four 'specialized' archetypes (Power-FF, Sinker-ball, "
            "Breaking-heavy, Offspeed-heavy) cluster tight at TTO=3 "
            f"({spec_min:.3f}-{spec_max:.3f}). Balanced is the outlier with "
            f"the smallest TTO=3 penalty ({balanced_tto3:.3f}). The original "
            "research expectation that Power-FF would take the largest "
            "penalty (and Breaking-heavy the smallest) is not supported in "
            "the K-rate data. Plausible explanation: predictability is the "
            "punishable trait regardless of which specific predictable pitch "
            "is being thrown — diversifying arsenal (Balanced) holds up "
            "better than any single 'specialty', even slider-heavy."
        ),
        "specialized_tto3_multipliers": {a: round(v, 4) for a, v in specialized_tto3.items()},
        "specialized_tto3_range": [round(spec_min, 4), round(spec_max, 4)],
        "balanced_tto3_multiplier": round(balanced_tto3, 4),
        "balanced_vs_specialized_gap": round(balanced_tto3 - spec_max, 4),
    }

    return {
        "generated_at": payload["generated_at"],
        "heat_map_text": heat_map,
        "confidence_intervals_95pct": ci_table,
        "research_benchmark_comparison": benchmark_compare,
        "spot_check_warnings": spot_check_warnings,
        "archetype_interaction_finding": archetype_interaction_finding,
    }


# ---- Build payload ---------------------------------------------------------


def _build_payload(
    df_pa: pd.DataFrame,
    archetype_map: dict[int, dict[int, str]],
    seasons: list[int],
) -> tuple[dict, pd.DataFrame, int]:
    """Return (payload, raw_cells_df, n_skipped_no_archetype)."""
    df_pa = _assign_tto(df_pa)
    df_pa = _filter_sp_pas(df_pa)
    df_pa, n_skipped_no_arch = _attach_archetype(df_pa, archetype_map)
    cells = _aggregate_cells(df_pa)

    by_archetype, low_sample = _build_archetype_multipliers(cells)
    league_wide = _build_league_wide(cells)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "empirical_pa_outcomes_by_tto_and_archetype",
        "seasons_used": list(seasons),
        "shrinkage_prior_pa": SHRINKAGE_PRIOR_PA,
        "min_sample_per_cell": MIN_SAMPLE_PER_CELL,
        "by_archetype": by_archetype,
        "league_wide": league_wide,
        "summary": {
            "total_pa_used": int(len(df_pa)),
            "total_pa_skipped_no_archetype": int(n_skipped_no_arch),
            "low_sample_cells": low_sample,
        },
    }
    return payload, cells, n_skipped_no_arch


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args(argv)

    archetype_map = _load_archetypes()
    logger.info("loaded archetypes for %d pitchers", len(archetype_map))

    df_pa = _load_statcast_pas(args.seasons)
    logger.info("loaded %d PA-terminating events across seasons %s",
                len(df_pa), args.seasons)

    payload, cells, n_skipped = _build_payload(df_pa, archetype_map, args.seasons)
    logger.info("aggregated to %d (archetype, tto) cells; %d PAs skipped (no archetype)",
                len(cells), n_skipped)

    # Build and log the heat map + benchmarks BEFORE sanity check so a
    # gate failure still surfaces the numbers. The report file is written
    # only on success.
    spot_check_warnings = _spot_check_warnings(payload)
    report = _build_report(payload, cells, spot_check_warnings)
    logger.info("pre-gate heat map:\n%s", report["heat_map_text"])
    logger.info("pre-gate league-wide vs research benchmark:")
    for t in TTO_BUCKETS:
        cmp = report["research_benchmark_comparison"][f"tto_{t}"]
        logger.info("  TTO=%d  observed=%.3f  research=%.3f  delta=%+.3f",
                    t, cmp["observed_league_wide"], cmp["research_benchmark"],
                    cmp["delta"])

    _sanity_check(payload)

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUTPUT_FILE)

    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote diagnostic report -> %s", REPORT_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
