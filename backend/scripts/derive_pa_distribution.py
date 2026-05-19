"""Phase 3-v2c-0: Empirical PA distribution by batting order slot and TTO.

For each integer total_BF in the observed range, captures the expected PA
count at each (batting_order_slot, tto_bucket) cell, aggregated across all
SP starter-games 2023-2025.

Why this exists: the projector's analytical model says "BF/9 PAs per slot,
equal across TTO buckets". But leadoff hitters get more 1st-time-through
opportunities than #9, and the asymmetry compounds with TTO. The
Phase 3-v2c-iv projector rewrite consumes this file to distribute a
projected E[BF] across (slot, TTO) cells.

Methodology
-----------
- Per game, the SP's first 9 distinct batters define slots 1-9 (the lineup
  as it appeared to the pitcher in PA order).
- TTO bucket is per-(pitcher, batter) cumulative-appearance count, capped
  at 4 (reuses derive_tto_multipliers logic).
- Games with pinch hitters during the SP's tenure are dropped (their PAs
  can't be cleanly mapped to slots 1-9). Modern MLB DH eliminates most of
  these cases anyway.
- For each integer total_BF, aggregate expected PAs per (slot, TTO) cell
  across all games at that BF. Smooth low-sample BF values via a 3-point
  rolling mean (weighted by game count).

CLI:
    python -m scripts.derive_pa_distribution
    python -m scripts.derive_pa_distribution --seasons 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.derive_tto_multipliers import (
    TTO_BUCKETS,
    _assign_tto,
    _filter_sp_pas,
    _load_statcast_pas,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "pa_distribution_by_bf.json"
REPORT_FILE = PROCESSED_DIR / "pa_distribution_report.json"

# BF range we emit. Outside this range is too rare to be useful.
BF_MIN = 12
BF_MAX = 35

# Smoothing
MIN_SMOOTHING_THRESHOLD = 50
SMOOTHING_WINDOW = 1  # 1 = 3-point rolling (BF-1, BF, BF+1)

SLOTS = tuple(range(1, 10))


# ---- Pure helpers (unit-testable) ------------------------------------------


def _attach_slot_and_drop_pinch_hits(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Add a `slot` column (1-9) per PA. Drop games where any batter cannot
    be mapped to the first-9 lineup (pinch hitter, mid-game replacement).

    Slot is derived per (game_pk, pitcher) from order of first appearance
    in PA-chronological order. This matches the batting order as observed
    by the SP.
    """
    df = df_pa.sort_values(
        ["game_pk", "pitcher", "inning", "at_bat_number"], kind="mergesort"
    ).copy()
    is_first = ~df.duplicated(["game_pk", "pitcher", "batter"], keep="first")
    df["__is_first_appearance"] = is_first
    df["__lineup_rank"] = df.groupby(["game_pk", "pitcher"])["__is_first_appearance"].cumsum()
    # Each batter's slot = the lineup_rank when they FIRST appeared
    slot_lookup = (
        df.loc[df["__is_first_appearance"],
               ["game_pk", "pitcher", "batter", "__lineup_rank"]]
        .rename(columns={"__lineup_rank": "slot"})
    )
    df = df.drop(columns=["__is_first_appearance", "__lineup_rank"]).merge(
        slot_lookup, on=["game_pk", "pitcher", "batter"], how="left"
    )
    # Identify games containing any batter outside slots 1-9 (pinch hit).
    bad_games = df.loc[df["slot"] > 9, ["game_pk", "pitcher"]].drop_duplicates()
    if not bad_games.empty:
        df = df.merge(bad_games.assign(__bad=True),
                      on=["game_pk", "pitcher"], how="left")
        df = df[df["__bad"].isna()].drop(columns=["__bad"])
    # Drop games that didn't reach 9 distinct batters (shouldn't happen
    # after _filter_sp_pas, but guard).
    n_slots = df.groupby(["game_pk", "pitcher"])["slot"].nunique().rename("n_slots")
    full_lineups = n_slots[n_slots == 9].index
    df = df.merge(pd.DataFrame(full_lineups.tolist(), columns=["game_pk", "pitcher"]),
                  on=["game_pk", "pitcher"], how="inner")
    df["slot"] = df["slot"].astype(int)
    return df


def _per_game_bf(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame indexed by (game_pk, pitcher) with `total_bf` col."""
    bf = df_pa.groupby(["game_pk", "pitcher"]).size().rename("total_bf").reset_index()
    bf["total_bf"] = bf["total_bf"].astype(int)
    return bf


def _aggregate_cell_pa(df_pa: pd.DataFrame, bf: pd.DataFrame) -> pd.DataFrame:
    """For each (total_bf, slot, tto), return (pa_count, n_games)."""
    df = df_pa.merge(bf, on=["game_pk", "pitcher"], how="inner")
    cells = (
        df.groupby(["total_bf", "slot", "tto"])
          .size()
          .rename("pa_count")
          .reset_index()
    )
    games = bf.groupby("total_bf").size().rename("n_games").reset_index()
    cells = cells.merge(games, on="total_bf", how="left")
    return cells


def _build_raw_distribution_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide form: row = (total_bf, slot), columns = tto_1..tto_4,
    values = expected PA per game. total_pa column added."""
    cells = cells.copy()
    cells["expected_pa"] = cells["pa_count"] / cells["n_games"].clip(lower=1)
    wide = cells.pivot_table(
        index=["total_bf", "slot", "n_games"],
        columns="tto",
        values="expected_pa",
        fill_value=0.0,
    ).reset_index()
    # Ensure all tto columns exist
    for t in TTO_BUCKETS:
        if t not in wide.columns:
            wide[t] = 0.0
    wide = wide.rename(columns={t: f"tto_{t}" for t in TTO_BUCKETS})
    wide["total_pa"] = sum(wide[f"tto_{t}"] for t in TTO_BUCKETS)
    return wide


def _smoothed_at_bf(
    raw_cells: pd.DataFrame, bf: int, window: int = SMOOTHING_WINDOW,
) -> tuple[dict, int, int, bool]:
    """Return ({slot -> {tto_* -> pa, total_pa}}, n_games_at_exact_bf,
    n_games_effective, was_smoothed).

    n_games_at_exact_bf is the raw count of games at this BF (may be 0).
    n_games_effective is what was actually used for averaging:
      - equal to n_games_at_exact_bf when smoothed=False
      - equal to the smoothing-window total when smoothed=True
    """
    raw_at_bf = raw_cells[raw_cells["total_bf"] == bf]
    n_games_at_bf = int(raw_at_bf["n_games"].iloc[0]) if not raw_at_bf.empty else 0

    if n_games_at_bf >= MIN_SMOOTHING_THRESHOLD:
        return _slot_dict_from_cells(raw_at_bf), n_games_at_bf, n_games_at_bf, False

    # Smooth: aggregate within window
    window_cells = raw_cells[
        (raw_cells["total_bf"] >= bf - window)
        & (raw_cells["total_bf"] <= bf + window)
    ]
    if window_cells.empty:
        # No data in window — try widening to ±2
        window_cells = raw_cells[
            (raw_cells["total_bf"] >= bf - 2)
            & (raw_cells["total_bf"] <= bf + 2)
        ]
    if window_cells.empty:
        # Still nothing — fall back to nearest available BF in raw_cells
        if raw_cells.empty:
            return _empty_slot_dict(), 0, 0, True
        all_bf = raw_cells["total_bf"].unique()
        nearest = min(all_bf, key=lambda x: abs(x - bf))
        nearest_cells = raw_cells[raw_cells["total_bf"] == nearest]
        n_effective = int(nearest_cells["n_games"].iloc[0])
        return _slot_dict_from_cells(nearest_cells), n_games_at_bf, n_effective, True

    # Weighted average across window. For each (slot, tto), sum PAs across
    # window games then divide by total games in window.
    total_games = int(window_cells.drop_duplicates("total_bf")["n_games"].sum())
    smoothed: dict = {}
    for slot in SLOTS:
        slot_cells = window_cells[window_cells["slot"] == slot]
        block: dict = {}
        slot_total = 0.0
        for t in TTO_BUCKETS:
            t_cells = slot_cells[slot_cells["tto"] == t]
            pa_sum = float(t_cells["pa_count"].sum()) if not t_cells.empty else 0.0
            value = pa_sum / total_games if total_games > 0 else 0.0
            block[f"tto_{t}"] = round(value, 4)
            slot_total += value
        block["total_pa"] = round(slot_total, 4)
        smoothed[str(slot)] = block
    return smoothed, n_games_at_bf, total_games, True


def _slot_dict_from_cells(cells: pd.DataFrame) -> dict:
    """Convert long-form raw cells (single BF) to {slot -> {tto_* -> pa}}."""
    out: dict = {}
    n_games = int(cells["n_games"].iloc[0]) if not cells.empty else 0
    for slot in SLOTS:
        slot_block: dict = {}
        slot_total = 0.0
        for t in TTO_BUCKETS:
            row = cells[(cells["slot"] == slot) & (cells["tto"] == t)]
            pa = float(row["pa_count"].iloc[0]) / n_games if not row.empty and n_games > 0 else 0.0
            slot_block[f"tto_{t}"] = round(pa, 4)
            slot_total += pa
        slot_block["total_pa"] = round(slot_total, 4)
        out[str(slot)] = slot_block
    return out


def _empty_slot_dict() -> dict:
    return {
        str(slot): {**{f"tto_{t}": 0.0 for t in TTO_BUCKETS}, "total_pa": 0.0}
        for slot in SLOTS
    }


def _build_distributions(raw_cells: pd.DataFrame) -> tuple[dict, list[int]]:
    """Build {bf_str: {n_games, smoothed, by_slot}} for all BF in [BF_MIN, BF_MAX]."""
    distributions: dict = {}
    smoothed_bf_values: list[int] = []
    for bf in range(BF_MIN, BF_MAX + 1):
        slot_dict, n_games_raw, n_effective, was_smoothed = _smoothed_at_bf(raw_cells, bf)
        distributions[str(bf)] = {
            "n_games": n_games_raw,
            "smoothed": was_smoothed,
            "smoothing_window_n_games": n_effective if was_smoothed else None,
            "by_slot": slot_dict,
        }
        if was_smoothed:
            smoothed_bf_values.append(bf)
    return distributions, smoothed_bf_values


def lookup_distribution(distributions: dict, projected_bf: float) -> dict:
    """Round projected_bf to nearest integer BF, clamp to range, return its entry."""
    bf_int = int(round(projected_bf))
    bf_int = max(BF_MIN, min(BF_MAX, bf_int))
    return distributions[str(bf_int)]


# ---- Analytical model (naive baseline for comparison) ----------------------


def _analytical_distribution(bf: int) -> dict:
    """The naive analytical model: BF/9 PAs per slot, distributed across
    TTO buckets in order (TTO=1 filled first, then TTO=2, etc.).

    For BF=27 → every slot gets 3.0 PAs (TTO_1=1, TTO_2=1, TTO_3=1).
    For BF=24 → every slot gets 2.67 PAs (TTO_1=1, TTO_2=1, TTO_3=0.67).
    For BF=18 → every slot gets 2.0 PAs (TTO_1=1, TTO_2=1, TTO_3=0).
    """
    pa_per_slot = bf / 9.0
    full = int(pa_per_slot)
    partial = pa_per_slot - full

    slot_block: dict = {}
    for t in TTO_BUCKETS:
        if t <= full:
            slot_block[f"tto_{t}"] = 1.0
        elif t == full + 1:
            slot_block[f"tto_{t}"] = round(partial, 4)
        else:
            slot_block[f"tto_{t}"] = 0.0
    slot_block["total_pa"] = round(pa_per_slot, 4)

    return {str(slot): dict(slot_block) for slot in SLOTS}


# ---- Sanity checks ---------------------------------------------------------


def _sanity_check(payload: dict) -> None:
    distributions = payload["distributions"]

    # Gate 6: every BF in [BF_MIN, BF_MAX] present
    missing = [str(b) for b in range(BF_MIN, BF_MAX + 1) if str(b) not in distributions]
    if missing:
        raise AssertionError(f"missing BF values in distributions: {missing}")

    for bf_str, entry in distributions.items():
        bf = int(bf_str)
        by_slot = entry["by_slot"]
        n_games = entry["n_games"]
        smoothed = entry["smoothed"]

        # Gate 1: sum across slots == BF (±0.5) for unsmoothed
        if not smoothed and n_games >= MIN_SMOOTHING_THRESHOLD:
            total_sum = sum(by_slot[str(s)]["total_pa"] for s in SLOTS)
            if abs(total_sum - bf) > 0.5:
                raise AssertionError(
                    f"BF={bf} (n_games={n_games}, raw): slot-sum {total_sum:.3f} "
                    f"does not equal BF (delta {total_sum - bf:+.3f})"
                )

        # Gate 5: no cell > 1.0
        for slot in SLOTS:
            for t in TTO_BUCKETS:
                v = by_slot[str(slot)][f"tto_{t}"]
                if v > 1.0:
                    raise AssertionError(
                        f"BF={bf} slot={slot} tto={t}: PA count {v} exceeds 1.0 "
                        f"(a batter cannot bat twice in the same TTO bucket)"
                    )

        # Gate 2: TTO=1 monotonically non-increasing across slots
        tto_1 = [by_slot[str(s)]["tto_1"] for s in SLOTS]
        for i in range(len(tto_1) - 1):
            if tto_1[i + 1] > tto_1[i] + 1e-6:
                raise AssertionError(
                    f"BF={bf}: TTO=1 not monotonic — slot {i+1}={tto_1[i]:.4f}, "
                    f"slot {i+2}={tto_1[i+1]:.4f}"
                )

        # Gate 3: BF >= 18 → every slot tto_2 > 0
        if bf >= 18 and n_games >= MIN_SMOOTHING_THRESHOLD and not smoothed:
            for slot in SLOTS:
                v = by_slot[str(slot)]["tto_2"]
                if v <= 0.0:
                    raise AssertionError(
                        f"BF={bf} slot={slot}: tto_2 PA is {v} "
                        f"(BF>=18 implies TTO=2 should be reached at every slot)"
                    )

        # Gate 4: BF < 27 → slot_9 tto_3 <= slot_1 tto_3
        if bf < 27:
            slot9_tto3 = by_slot["9"]["tto_3"]
            slot1_tto3 = by_slot["1"]["tto_3"]
            if slot9_tto3 > slot1_tto3 + 1e-6:
                raise AssertionError(
                    f"BF={bf}: slot_9 TTO=3 ({slot9_tto3:.4f}) > slot_1 "
                    f"TTO=3 ({slot1_tto3:.4f}) — order shouldn't fully turn over"
                )

    logger.info("sanity checks pass: 6 gates checked across %d BF values",
                len(distributions))


# ---- Report ----------------------------------------------------------------


def _build_report(payload: dict, raw_cells: pd.DataFrame) -> dict:
    """Diagnostic report: modal BF heatmap + top discrepancies vs analytical."""
    distributions = payload["distributions"]
    games_per_bf = (
        raw_cells.drop_duplicates("total_bf")
        .set_index("total_bf")["n_games"]
        .to_dict()
    )
    modal_bf = max(games_per_bf, key=games_per_bf.get)
    modal_n = int(games_per_bf[modal_bf])
    modal_entry = distributions[str(modal_bf)]

    # Heatmap for modal BF
    header = "slot".ljust(6) + "".join(f"TTO_{t}".rjust(10) for t in TTO_BUCKETS) + "total".rjust(10)
    lines = [f"Empirical distribution at modal BF={modal_bf} (n_games={modal_n}):",
             header, "-" * len(header)]
    for slot in SLOTS:
        row = str(slot).ljust(6)
        for t in TTO_BUCKETS:
            row += f"{modal_entry['by_slot'][str(slot)][f'tto_{t}']:>10.3f}"
        row += f"{modal_entry['by_slot'][str(slot)]['total_pa']:>10.3f}"
        lines.append(row)
    empirical_heatmap = "\n".join(lines)

    # Analytical at modal BF
    analytical = _analytical_distribution(modal_bf)
    lines = [f"Analytical distribution at BF={modal_bf} (naive BF/9 per slot):",
             header, "-" * len(header)]
    for slot in SLOTS:
        row = str(slot).ljust(6)
        for t in TTO_BUCKETS:
            row += f"{analytical[str(slot)][f'tto_{t}']:>10.3f}"
        row += f"{analytical[str(slot)]['total_pa']:>10.3f}"
        lines.append(row)
    analytical_heatmap = "\n".join(lines)

    # Compute cell-wise discrepancies across ALL BF in (BF_MIN..BF_MAX)
    discrepancies: list[dict] = []
    for bf_str, entry in distributions.items():
        bf = int(bf_str)
        if entry["smoothed"] or entry["n_games"] < MIN_SMOOTHING_THRESHOLD:
            continue  # only compare on solid data
        analytical_bf = _analytical_distribution(bf)
        for slot in SLOTS:
            for t in TTO_BUCKETS:
                emp = entry["by_slot"][str(slot)][f"tto_{t}"]
                ana = analytical_bf[str(slot)][f"tto_{t}"]
                delta = emp - ana
                if abs(delta) > 0.01:
                    discrepancies.append({
                        "bf": bf,
                        "slot": slot,
                        "tto": t,
                        "empirical": round(emp, 4),
                        "analytical": round(ana, 4),
                        "delta": round(delta, 4),
                    })

    discrepancies.sort(key=lambda d: abs(d["delta"]), reverse=True)
    top_discrepancies = discrepancies[:20]

    return {
        "generated_at": payload["generated_at"],
        "modal_bf": int(modal_bf),
        "modal_bf_n_games": modal_n,
        "empirical_heatmap_modal_bf": empirical_heatmap,
        "analytical_heatmap_modal_bf": analytical_heatmap,
        "top_empirical_vs_analytical_discrepancies": top_discrepancies,
        "n_total_discrepancies_over_0.01": len(discrepancies),
    }


# ---- Build payload ---------------------------------------------------------


def _build_payload(df_pa: pd.DataFrame, seasons: list[int]) -> tuple[dict, pd.DataFrame]:
    df_pa = _assign_tto(df_pa)
    df_pa = _filter_sp_pas(df_pa)
    logger.info("after SP filter: %d PAs", len(df_pa))

    df_pa = _attach_slot_and_drop_pinch_hits(df_pa)
    logger.info("after lineup-mapping + pinch-hit drop: %d PAs", len(df_pa))

    bf_df = _per_game_bf(df_pa)
    cells = _aggregate_cell_pa(df_pa, bf_df)

    # Trim raw cells to the BF range we care about
    cells_in_range = cells[
        (cells["total_bf"] >= BF_MIN) & (cells["total_bf"] <= BF_MAX)
    ]

    distributions, smoothed_bf_values = _build_distributions(cells_in_range)

    bf_observed = sorted(bf_df["total_bf"].unique().tolist())
    games_per_bf = bf_df.groupby("total_bf").size()
    modal_bf = int(games_per_bf.idxmax())

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "empirical_pa_distribution_by_batting_order_and_tto",
        "seasons_used": list(seasons),
        "n_starter_games": int(len(bf_df)),
        "min_smoothing_threshold": MIN_SMOOTHING_THRESHOLD,
        "distributions": distributions,
        "summary": {
            "bf_range_observed": [int(min(bf_observed)), int(max(bf_observed))],
            "modal_bf": modal_bf,
            "smoothed_bf_values": smoothed_bf_values,
        },
    }
    return payload, cells_in_range


# ---- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = parser.parse_args(argv)

    df_pa = _load_statcast_pas(args.seasons)
    logger.info("loaded %d PA-terminating events across seasons %s",
                len(df_pa), args.seasons)

    payload, raw_cells = _build_payload(df_pa, args.seasons)
    logger.info("built distributions across %d BF values (smoothed: %d)",
                len(payload["distributions"]),
                len(payload["summary"]["smoothed_bf_values"]))

    _sanity_check(payload)

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", OUTPUT_FILE)

    report = _build_report(payload, raw_cells)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote diagnostic report -> %s", REPORT_FILE)

    # Echo the key findings
    logger.info("n_starter_games used: %d", payload["n_starter_games"])
    logger.info("BF range observed: %s", payload["summary"]["bf_range_observed"])
    logger.info("modal BF: %d (n_games=%d)", report["modal_bf"], report["modal_bf_n_games"])
    logger.info("modal BF empirical:\n%s", report["empirical_heatmap_modal_bf"])
    logger.info("modal BF analytical:\n%s", report["analytical_heatmap_modal_bf"])
    logger.info("top 3 empirical-vs-analytical discrepancies:")
    for d in report["top_empirical_vs_analytical_discrepancies"][:3]:
        logger.info("  BF=%d slot=%d tto=%d empirical=%.3f analytical=%.3f delta=%+.3f",
                    d["bf"], d["slot"], d["tto"], d["empirical"], d["analytical"], d["delta"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
