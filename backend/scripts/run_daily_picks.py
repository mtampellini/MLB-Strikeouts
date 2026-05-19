"""Phase 7: hourly picks pipeline orchestrator.

Runs end-to-end: probables -> bundles -> projections -> odds -> picks ->
ledger update. Idempotent within a day: running multiple times appends
snapshots to the same picks rather than duplicating.

Picks ledger layout::

    data/picks/YYYY-MM-DD/
      picks.json              # primary tier (live)
      secondary_picks.json
      shadow_picks.json
      all_picks_debug.json    # primary + secondary + shadow + skipped + metadata
      snapshots/HHMM.json     # per-run snapshot of evaluated picks
      run_log.json            # append-only log of each run's metadata

Each pick in the ledger carries a ``snapshots`` array tracking how price
and edge evolved across the day's runs. ``clv_pct`` is updated each run
based on the latest snapshot vs the first.

CLI::

    python -m scripts.run_daily_picks
    python -m scripts.run_daily_picks --date 2026-05-19
    python -m scripts.run_daily_picks --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from src.picks.engine import PickResult, generate_picks
from src.picks.output import write_picks
from src.projection.inputs import ProjectionBundle, ProjectionContext

logger = logging.getLogger(__name__)

DATA_PICKS_DIR = Path(__file__).resolve().parents[1] / "data" / "picks"


# ---- Daily ledger I/O ------------------------------------------------------


def _ledger_paths(target_date: date) -> dict[str, Path]:
    day_dir = DATA_PICKS_DIR / target_date.isoformat()
    snapshot_dir = day_dir / "snapshots"
    return {
        "day_dir": day_dir,
        "snapshot_dir": snapshot_dir,
        "primary": day_dir / "picks.json",
        "secondary": day_dir / "secondary_picks.json",
        "shadow": day_dir / "shadow_picks.json",
        "debug": day_dir / "all_picks_debug.json",
        "run_log": day_dir / "run_log.json",
        "projections": day_dir / "projections.json",
    }


def _load_existing_ledger(debug_path: Path) -> dict | None:
    """Load the prior all_picks_debug.json if it exists."""
    if not debug_path.exists():
        return None
    try:
        return json.loads(debug_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ledger at %s unreadable (%s); starting fresh", debug_path, exc)
        return None


# ---- Snapshot capture ------------------------------------------------------


def _snapshot_dict_from_pick(pick: dict, run_time: datetime) -> dict:
    """Capture the time-varying part of a pick for the snapshot history."""
    return {
        "run_time": run_time.isoformat(timespec="seconds"),
        "american_odds": pick["american_odds"],
        "market_p": pick["market_p"],
        "edge_pct": pick["edge_pct"],
        "ev_pct": pick["ev_pct"],
        "model_p": pick["model_p"],
        "model_e_k": pick["model_e_k"],
        "devig_source": pick["devig_source"],
        "tier": pick["tier"],
        "rank_in_tier": pick["rank_in_tier"],
    }


# ---- CLV computation -------------------------------------------------------


def compute_clv_pct(
    first_seen_market_p: float, closing_market_p: float, side: str,
) -> float:
    """``clv_pct = first_seen_market_p - closing_market_p`` for Over;
    sign-flipped for Under.

    A positive CLV means the line moved against the book in our favor: the
    book became less confident the bet hits than when we first locked in.
    """
    raw = first_seen_market_p - closing_market_p
    return raw if side == "Over" else -raw


# ---- Merge logic (idempotency) ---------------------------------------------


def _merge_picks_with_snapshots(
    new_pick_list: list[dict],
    existing_ledger: dict | None,
    run_time: datetime,
) -> list[dict]:
    """Idempotent merge: keep existing pick records (and their snapshot
    histories), append the current run's snapshot, update closing_* fields.

    Picks newly seen this run get inserted with first_seen_* set to the
    current snapshot. Picks that vanished from the slate this run keep
    their existing history but are flagged active=False.

    Returns the merged list (preserves order: existing picks first by
    first_seen_at, then new picks).
    """
    by_id: dict[str, dict] = {}
    if existing_ledger:
        # Collect all picks from the existing ledger (primary + secondary
        # + shadow), keyed by pick_id.
        for tier_key in ("primary", "secondary", "shadow"):
            for p in existing_ledger.get(tier_key, []) or []:
                by_id[p["pick_id"]] = dict(p)  # copy

    # Mark all existing as "not seen this run" initially
    for p in by_id.values():
        p["active"] = False

    for new_pick in new_pick_list:
        pid = new_pick["pick_id"]
        snap = _snapshot_dict_from_pick(new_pick, run_time)
        if pid in by_id:
            existing = by_id[pid]
            # Don't double-record the snapshot if same HHMM bucket already
            # captured.
            same_minute = any(
                _same_minute_bucket(s.get("run_time"), snap["run_time"])
                for s in existing.get("snapshots", [])
            )
            if not same_minute:
                existing.setdefault("snapshots", []).append(snap)
            # Update tier + rank from the new run; first_seen_* never changes
            existing["tier"] = new_pick["tier"]
            existing["rank_in_tier"] = new_pick["rank_in_tier"]
            existing["american_odds"] = new_pick["american_odds"]
            existing["market_p"] = new_pick["market_p"]
            existing["edge_pct"] = new_pick["edge_pct"]
            existing["ev_pct"] = new_pick["ev_pct"]
            existing["model_p"] = new_pick["model_p"]
            existing["model_e_k"] = new_pick["model_e_k"]
            existing["model_e_bf"] = new_pick["model_e_bf"]
            existing["active"] = True
            # Closing snapshot = latest
            existing["closing_snapshot_run_time"] = snap["run_time"]
            existing["closing_american_odds"] = snap["american_odds"]
            existing["closing_market_p"] = snap["market_p"]
            existing["closing_edge_pct"] = snap["edge_pct"]
            # Recompute CLV
            existing["clv_pct"] = round(compute_clv_pct(
                existing["first_seen_market_p"],
                existing["closing_market_p"],
                existing["side"],
            ), 6)
        else:
            # New pick
            record = dict(new_pick)
            record["first_seen_at"] = snap["run_time"]
            record["first_seen_odds"] = snap["american_odds"]
            record["first_seen_market_p"] = snap["market_p"]
            record["first_seen_edge_pct"] = snap["edge_pct"]
            record["closing_snapshot_run_time"] = snap["run_time"]
            record["closing_american_odds"] = snap["american_odds"]
            record["closing_market_p"] = snap["market_p"]
            record["closing_edge_pct"] = snap["edge_pct"]
            record["clv_pct"] = 0.0  # first snapshot — no movement yet
            record["snapshots"] = [snap]
            record["active"] = True
            by_id[pid] = record

    return list(by_id.values())


def _same_minute_bucket(a: str | None, b: str | None) -> bool:
    """Two ISO timestamps fall in the same HH:MM bucket."""
    if not a or not b:
        return False
    return a[:16] == b[:16]


def _filter_by_tier(merged: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split merged picks into the three tier lists for output. Inactive
    picks (not seen this run) are excluded from the tier files but stay in
    the debug ledger."""
    primary = sorted(
        (p for p in merged if p.get("active") and p["tier"] == "primary"),
        key=lambda p: p.get("rank_in_tier", 0),
    )
    secondary = sorted(
        (p for p in merged if p.get("active") and p["tier"] == "secondary"),
        key=lambda p: p.get("rank_in_tier", 0),
    )
    shadow = sorted(
        (p for p in merged if p.get("active") and p["tier"] == "shadow"),
        key=lambda p: p.get("rank_in_tier", 0),
    )
    return primary, secondary, shadow


# ---- Probables + bundle construction ---------------------------------------


def _build_bundles_for_date(target_date: date) -> list[ProjectionBundle]:
    """Find today's probable starters, build a projection bundle for each.

    Uses scripts.build_sample_bundle.build_bundle which orchestrates
    Probables + StatsAPI + Statcast for a single (date, pitcher) tuple.
    """
    from src.data.probables_client import ProbablesClient
    from scripts.build_sample_bundle import build_bundle as _build_one_bundle

    bundles: list[ProjectionBundle] = []
    try:
        probables = ProbablesClient().fetch(cutoff_date=target_date)
    except Exception as exc:
        logger.warning("ProbablesClient failed (%s); slate empty", exc)
        return []

    for prob in probables:
        try:
            blob = _build_one_bundle(target_date, prob.pitcher_mlbam_id)
            bundle = ProjectionBundle.from_dict(blob)
            bundles.append(bundle)
        except Exception as exc:
            logger.warning(
                "bundle build failed for %s (%d): %s",
                prob.pitcher_name, prob.pitcher_mlbam_id, exc,
            )
            continue
    return bundles


# ---- Run log append --------------------------------------------------------


def _categorize_skips(skipped: list[dict]) -> dict[str, int]:
    """Group skip records into transient vs permanent buckets.

    Transient: ``lineup_not_posted`` — the pitcher will be re-evaluated
    next hourly run. Carried separately so the run log shows slate
    buildup across the day (early runs are lineup-light; later runs fill
    in as lineups post).

    Permanent: every other skip reason on this slate (career IP, season
    IP, archetype unavailable, no market data, projector failure). Once
    a pitcher is permanently skipped, no future run today will pick them
    up. Bucketed by the leading token of ``reason`` so the breakdown
    stays human-readable: e.g. ``projector_skipped: hard_filter: ...``
    collapses to ``permanent_projector_skipped``.
    """
    breakdown: dict[str, int] = {}
    for s in skipped:
        if s.get("is_transient"):
            key = "transient_" + s["reason"]
        else:
            reason = s.get("reason", "unknown")
            head = reason.split(":", 1)[0].strip()
            key = "permanent_" + head
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


def _append_run_log(run_log_path: Path, entry: dict) -> None:
    if run_log_path.exists():
        try:
            log = json.loads(run_log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = {"runs": []}
    else:
        log = {"runs": []}
    log["runs"].append(entry)
    run_log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")


# ---- Main ------------------------------------------------------------------


def run_pipeline(
    target_date: date,
    *,
    dry_run: bool = False,
    bundles: list[ProjectionBundle] | None = None,
) -> dict:
    """Single-run entry point — extracted for testability.

    If ``bundles`` is provided, skips the probables + build_bundle step
    (used by tests to inject synthetic bundles).
    """
    paths = _ledger_paths(target_date)
    if not dry_run:
        paths["day_dir"].mkdir(parents=True, exist_ok=True)
        paths["snapshot_dir"].mkdir(parents=True, exist_ok=True)

    run_time = datetime.now(timezone.utc)
    snapshot_filename = f"{run_time.strftime('%H%M')}.json"
    snapshot_path = paths["snapshot_dir"] / snapshot_filename

    if bundles is None:
        if dry_run:
            # Dry-run skips external API calls (Probables, StatsAPI, Statcast,
            # Odds) so plumbing can be verified in seconds, not minutes.
            logger.info("dry-run: skipping bundle construction (no API calls)")
            bundles = []
        else:
            bundles = _build_bundles_for_date(target_date)
    logger.info("built %d bundles for %s", len(bundles), target_date)

    if not bundles:
        result = PickResult(primary=[], secondary=[], shadow=[],
                             skipped=[], metadata={"n_bundles": 0})
    else:
        result = generate_picks(bundles)

    new_pick_list = [
        *result.primary, *result.secondary, *result.shadow,
    ]

    existing = _load_existing_ledger(paths["debug"])
    merged = _merge_picks_with_snapshots(new_pick_list, existing, run_time)
    primary, secondary, shadow = _filter_by_tier(merged)

    skip_breakdown = _categorize_skips(result.skipped)
    n_lineup_pending = skip_breakdown.get("transient_lineup_not_posted", 0)
    n_starters_evaluated = len(bundles) - n_lineup_pending

    summary = {
        "run_time": run_time.isoformat(timespec="seconds"),
        "target_date": target_date.isoformat(),
        "n_bundles": len(bundles),
        "n_starters_scheduled": len(bundles),
        "n_starters_evaluated": n_starters_evaluated,
        "n_lineup_pending": n_lineup_pending,
        "n_picks_generated": len(new_pick_list),
        "n_primary_active": len(primary),
        "n_secondary_active": len(secondary),
        "n_shadow_active": len(shadow),
        "n_total_merged": len(merged),
        "n_skipped_bundles": len(result.skipped),
        "skip_breakdown": skip_breakdown,
        "snapshot_file": snapshot_filename,
        "dry_run": dry_run,
    }

    if dry_run:
        logger.info("dry-run: would write ledger to %s", paths["day_dir"])
        logger.info("summary: %s", summary)
        return summary

    # Per-run snapshot — never overwrite if same HHMM bucket already on disk.
    if not snapshot_path.exists():
        snapshot_path.write_text(
            json.dumps({
                "run_time": run_time.isoformat(timespec="seconds"),
                "picks": [_snapshot_dict_from_pick(p, run_time)
                          for p in new_pick_list],
                "metadata": result.metadata,
            }, indent=2),
            encoding="utf-8",
        )

    # Merged ledger files
    paths["debug"].write_text(
        json.dumps({
            "metadata": {**result.metadata, "summary": summary},
            "primary": primary,
            "secondary": secondary,
            "shadow": shadow,
            "skipped": result.skipped,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["primary"].write_text(
        json.dumps({"metadata": summary, "picks": primary},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["secondary"].write_text(
        json.dumps({"metadata": summary, "picks": secondary},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["shadow"].write_text(
        json.dumps({"metadata": summary, "picks": shadow},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _append_run_log(paths["run_log"], summary)
    logger.info("wrote ledger to %s", paths["day_dir"])
    logger.info("summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=None,
                         help="Slate date (YYYY-MM-DD). Defaults to today (UTC).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Skip all writes; log what would happen.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = datetime.now(timezone.utc).date()

    try:
        run_pipeline(target_date, dry_run=args.dry_run)
    except Exception as exc:
        logger.error("run_daily_picks failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
