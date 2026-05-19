"""Phase 7: daily settlement orchestrator (6am ET cron).

Grades yesterday's picks against observed K counts and updates the
running performance ledger + calibration tracker.

Workflow:

1. Load the picks ledger from ``data/picks/YYYY-MM-DD/all_picks_debug.json``.
2. For each pick: pull observed K via MLB Stats API, grade win/loss/push,
   compute realized P/L at first-seen AND closing price.
3. Write results back into the daily ledger.
4. Append resolved picks to ``data/picks/performance_ledger.json``.
5. Update ``data/picks/calibration_tracker.json`` (per-line predicted vs
   observed across all settled picks).

CLI::

    python -m scripts.settle_results
    python -m scripts.settle_results --date 2026-05-18
    python -m scripts.settle_results --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.picks.devig import american_to_decimal

logger = logging.getLogger(__name__)

DATA_PICKS_DIR = Path(__file__).resolve().parents[1] / "data" / "picks"
PERFORMANCE_LEDGER_PATH = DATA_PICKS_DIR / "performance_ledger.json"
CALIBRATION_TRACKER_PATH = DATA_PICKS_DIR / "calibration_tracker.json"

ALT_LINES = (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)


# ---- Grading logic ---------------------------------------------------------


def grade_pick(side: str, line: float, observed_k: int | None) -> str:
    """Return ``"win"`` | ``"loss"`` | ``"push"`` | ``"pitcher_did_not_start"``.

    For half-integer lines (5.5), there's no push outcome: Over wins on
    K >= ceil(line), Under wins on K <= floor(line).
    For integer lines (5.0), K == line is a push.
    """
    if observed_k is None:
        return "pitcher_did_not_start"
    line_is_integer = float(line).is_integer()
    if line_is_integer and observed_k == int(line):
        return "push"
    if side == "Over":
        threshold = int(line) + 1 if line_is_integer else int(math.ceil(line))
        return "win" if observed_k >= threshold else "loss"
    if side == "Under":
        threshold = int(line) - 1 if line_is_integer else int(math.floor(line))
        return "win" if observed_k <= threshold else "loss"
    raise ValueError(f"side must be 'Over' or 'Under', got {side!r}")


def profit_loss_units(american_odds: int, result: str) -> float:
    """Realized P/L per 1 unit risked.

    Win at +150: +1.50 units. Win at -110: +100/110 ≈ +0.909 units.
    Loss: -1.000 units. Push: 0.0 units.
    """
    if result == "push" or result == "pitcher_did_not_start" or result == "no_game":
        return 0.0
    if result == "loss":
        return -1.0
    if result == "win":
        return american_to_decimal(american_odds) - 1.0
    raise ValueError(f"unknown result: {result!r}")


# ---- Observed K lookup -----------------------------------------------------


def _fetch_observed_k(pitcher_mlbam_id: int, game_pk: int) -> int | None:
    """Pull observed K count for a pitcher in a specific game from
    MLB Stats API. Returns None if the pitcher didn't start (no boxscore
    entry) or the API is unreachable.
    """
    import requests
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        blob = resp.json()
    except Exception as exc:
        logger.warning("MLB Stats boxscore fetch failed for %d: %s", game_pk, exc)
        return None

    for side in ("home", "away"):
        side_blob = (blob.get("teams") or {}).get(side) or {}
        players = side_blob.get("players") or {}
        key = f"ID{pitcher_mlbam_id}"
        if key not in players:
            continue
        pstat = (players[key].get("stats") or {}).get("pitching") or {}
        if "strikeOuts" in pstat:
            try:
                return int(pstat["strikeOuts"])
            except (TypeError, ValueError):
                return None
    return None


# ---- Settlement ------------------------------------------------------------


def settle_pick(
    pick: dict, observed_k: int | None,
) -> dict:
    """Compute result + P/L for one pick. Returns the updated pick dict."""
    result = grade_pick(pick["side"], pick["line"], observed_k)
    first_seen_odds = pick.get("first_seen_odds", pick.get("american_odds"))
    closing_odds = pick.get("closing_american_odds", first_seen_odds)
    updated = {
        **pick,
        "result": result,
        "observed_k": observed_k,
        "profit_loss_first_seen": round(
            profit_loss_units(int(first_seen_odds), result), 4,
        ),
        "profit_loss_closing": round(
            profit_loss_units(int(closing_odds), result), 4,
        ),
        "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return updated


# ---- Performance ledger ----------------------------------------------------


def _empty_aggregate() -> dict:
    return {
        "n": 0, "wins": 0, "losses": 0, "pushes": 0, "no_results": 0,
        "win_rate": 0.0, "roi_pct": 0.0, "mean_clv_pct": 0.0,
    }


def _accumulate_into_aggregate(agg: dict, pick: dict) -> None:
    result = pick.get("result")
    if result in (None, "pitcher_did_not_start", "no_game"):
        agg["no_results"] += 1
        return
    agg["n"] += 1
    if result == "win":
        agg["wins"] += 1
    elif result == "loss":
        agg["losses"] += 1
    elif result == "push":
        agg["pushes"] += 1


def _finalize_aggregate(agg: dict, pl_sum: float, clv_sum: float) -> None:
    n = agg["n"]
    if n == 0:
        return
    agg["win_rate"] = round((agg["wins"] / n) * 100, 3)
    # ROI in units: total P/L / units risked (1 unit per pick)
    agg["roi_pct"] = round((pl_sum / n) * 100, 3)
    agg["mean_clv_pct"] = round((clv_sum / n) * 100, 3)


def update_performance_ledger(
    resolved_picks: list[dict], existing_blob: dict | None,
) -> dict:
    """Append resolved picks to the running performance ledger and
    recompute by-tier / by-book / by-line-bucket aggregates."""
    existing = existing_blob or {"all_picks": []}
    seen_ids = {p["pick_id"] for p in existing.get("all_picks") or []}

    for pick in resolved_picks:
        if pick["pick_id"] in seen_ids:
            continue
        existing.setdefault("all_picks", []).append({
            "pick_id": pick["pick_id"],
            "date": pick.get("game_date"),
            "pitcher_name": pick.get("pitcher_name"),
            "pitcher_mlbam_id": pick.get("pitcher_mlbam_id"),
            "game_pk": pick.get("game_pk"),
            "line": pick["line"],
            "side": pick["side"],
            "book": pick["book"],
            "tier": pick.get("tier"),
            "first_seen_odds": pick.get("first_seen_odds"),
            "first_seen_edge_pct": pick.get("first_seen_edge_pct"),
            "closing_odds": pick.get("closing_american_odds"),
            "closing_edge_pct": pick.get("closing_edge_pct"),
            "clv_pct": pick.get("clv_pct"),
            "result": pick.get("result"),
            "observed_k": pick.get("observed_k"),
            "profit_loss_units": pick.get("profit_loss_first_seen"),
            "profit_loss_closing_units": pick.get("profit_loss_closing"),
            "model_p": pick.get("model_p"),
            "model_e_k": pick.get("model_e_k"),
            "settled_at": pick.get("settled_at"),
        })

    # Recompute aggregates
    by_tier: dict[str, dict] = {}
    by_book: dict[str, dict] = {}
    by_line_bucket: dict[str, dict] = {}
    tier_pl_sums: dict[str, float] = {}
    book_pl_sums: dict[str, float] = {}
    bucket_pl_sums: dict[str, float] = {}
    tier_clv_sums: dict[str, float] = {}
    book_clv_sums: dict[str, float] = {}
    bucket_clv_sums: dict[str, float] = {}

    for pick in existing["all_picks"]:
        tier = pick.get("tier") or "unknown"
        book = pick.get("book") or "unknown"
        bucket = _line_bucket_label(pick.get("line") or 0.0)
        for agg, store in (
            (by_tier.setdefault(tier, _empty_aggregate()), tier_pl_sums),
            (by_book.setdefault(book, _empty_aggregate()), book_pl_sums),
            (by_line_bucket.setdefault(bucket, _empty_aggregate()), bucket_pl_sums),
        ):
            _accumulate_into_aggregate(agg, pick)
        pl = pick.get("profit_loss_units")
        clv = pick.get("clv_pct")
        if pick.get("result") in ("win", "loss", "push") and pl is not None:
            tier_pl_sums[tier] = tier_pl_sums.get(tier, 0.0) + pl
            book_pl_sums[book] = book_pl_sums.get(book, 0.0) + pl
            bucket_pl_sums[bucket] = bucket_pl_sums.get(bucket, 0.0) + pl
            if clv is not None:
                tier_clv_sums[tier] = tier_clv_sums.get(tier, 0.0) + clv
                book_clv_sums[book] = book_clv_sums.get(book, 0.0) + clv
                bucket_clv_sums[bucket] = bucket_clv_sums.get(bucket, 0.0) + clv

    for tier, agg in by_tier.items():
        _finalize_aggregate(agg, tier_pl_sums.get(tier, 0.0),
                             tier_clv_sums.get(tier, 0.0))
    for book, agg in by_book.items():
        _finalize_aggregate(agg, book_pl_sums.get(book, 0.0),
                             book_clv_sums.get(book, 0.0))
    for bucket, agg in by_line_bucket.items():
        _finalize_aggregate(agg, bucket_pl_sums.get(bucket, 0.0),
                             bucket_clv_sums.get(bucket, 0.0))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_picks_resolved": sum(a["n"] for a in by_tier.values()),
        "by_tier": by_tier,
        "by_book": by_book,
        "by_line_bucket": by_line_bucket,
        "all_picks": existing["all_picks"],
    }


def _line_bucket_label(line: float) -> str:
    """Coarser bucket for line-level performance tracking."""
    if line < 5.5:
        return "3.5-5.5"
    if line < 7.5:
        return "5.5-7.5"
    return "7.5+"


# ---- Calibration tracker ---------------------------------------------------


def update_calibration_tracker(
    resolved_picks: list[dict], existing_blob: dict | None,
) -> dict:
    """Accumulate per-line predicted vs observed P(K >= line) across all
    settled picks. This is the live counterpart to Phase 4d's per-line
    calibration table.
    """
    existing = existing_blob or {"by_line": {}, "tracked_pick_ids": []}
    tracked = set(existing.get("tracked_pick_ids") or [])
    by_line: dict[str, dict] = dict(existing.get("by_line") or {})

    new_picks_added = 0
    for pick in resolved_picks:
        if pick["pick_id"] in tracked:
            continue
        if pick.get("result") not in ("win", "loss", "push"):
            continue
        if pick.get("side") != "Over":
            # Per-line calibration is anchored to P(K >= line); for Under
            # picks we'd track P(K <= line) separately. Skip Under for
            # now — keeps the table interpretable.
            tracked.add(pick["pick_id"])
            continue

        line_str = str(pick["line"])
        entry = by_line.setdefault(line_str, {
            "n_picks_over": 0,
            "sum_predicted_p_over": 0.0,
            "n_wins": 0,  # observed Over hits
            "n_pushes": 0,
        })
        entry["n_picks_over"] += 1
        entry["sum_predicted_p_over"] += float(pick.get("model_p") or 0.0)
        if pick.get("result") == "win":
            entry["n_wins"] += 1
        elif pick.get("result") == "push":
            entry["n_pushes"] += 1
        tracked.add(pick["pick_id"])
        new_picks_added += 1

    # Compute derived fields
    out_by_line: dict[str, dict] = {}
    for line_str, entry in by_line.items():
        n = entry["n_picks_over"]
        if n == 0:
            continue
        observed_p = entry["n_wins"] / n
        predicted_p = entry["sum_predicted_p_over"] / n
        out_by_line[line_str] = {
            "n_picks_over": n,
            "predicted_p_over_mean": round(predicted_p, 4),
            "observed_p_over_mean": round(observed_p, 4),
            "deviation_pp": round((observed_p - predicted_p) * 100, 2),
            # Internal accumulators (kept so we can update incrementally)
            "sum_predicted_p_over": entry["sum_predicted_p_over"],
            "n_wins": entry["n_wins"],
            "n_pushes": entry["n_pushes"],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_picks_tracked": len(tracked),
        "by_line": out_by_line,
        "tracked_pick_ids": sorted(tracked),
    }


# ---- Main ------------------------------------------------------------------


def settle_day(target_date: date, *, dry_run: bool = False) -> dict:
    """Settle one day's picks. Returns a summary dict."""
    day_dir = DATA_PICKS_DIR / target_date.isoformat()
    debug_path = day_dir / "all_picks_debug.json"
    if not debug_path.exists():
        logger.info("no picks ledger for %s (%s missing)", target_date, debug_path)
        return {"date": target_date.isoformat(), "n_picks_settled": 0,
                 "reason": "no_ledger"}

    debug_blob = json.loads(debug_path.read_text(encoding="utf-8"))
    all_picks: list[dict] = []
    for tier_key in ("primary", "secondary", "shadow"):
        all_picks.extend(debug_blob.get(tier_key) or [])

    # Group by (pitcher, game) so we fetch observed_k once per game
    obs_cache: dict[tuple[int, int], int | None] = {}
    for pick in all_picks:
        key = (int(pick["pitcher_mlbam_id"]), int(pick["game_pk"]))
        if key not in obs_cache:
            obs_cache[key] = _fetch_observed_k(*key)

    resolved: list[dict] = []
    for pick in all_picks:
        key = (int(pick["pitcher_mlbam_id"]), int(pick["game_pk"]))
        observed_k = obs_cache.get(key)
        updated = settle_pick(pick, observed_k)
        resolved.append(updated)

    # Per-tier resolved lists
    by_tier: dict[str, list[dict]] = {"primary": [], "secondary": [], "shadow": []}
    for p in resolved:
        if p.get("tier") in by_tier:
            by_tier[p["tier"]].append(p)

    summary = {
        "date": target_date.isoformat(),
        "n_picks_settled": len(resolved),
        "n_wins": sum(1 for p in resolved if p["result"] == "win"),
        "n_losses": sum(1 for p in resolved if p["result"] == "loss"),
        "n_pushes": sum(1 for p in resolved if p["result"] == "push"),
        "n_no_start": sum(1 for p in resolved if p["result"] == "pitcher_did_not_start"),
        "dry_run": dry_run,
    }
    pl_first_seen = sum(p.get("profit_loss_first_seen") or 0.0 for p in resolved)
    summary["profit_loss_first_seen_units_total"] = round(pl_first_seen, 4)
    summary["profit_loss_closing_units_total"] = round(
        sum(p.get("profit_loss_closing") or 0.0 for p in resolved), 4,
    )

    logger.info("settled %d picks for %s: %s", len(resolved), target_date, summary)

    if dry_run:
        return summary

    # Write resolved picks back into the daily ledger
    debug_blob["primary"] = by_tier["primary"]
    debug_blob["secondary"] = by_tier["secondary"]
    debug_blob["shadow"] = by_tier["shadow"]
    debug_blob.setdefault("metadata", {})["settlement_summary"] = summary
    debug_path.write_text(json.dumps(debug_blob, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    # Mirror into the tier files
    for tier_key, items in by_tier.items():
        (day_dir / f"{'picks' if tier_key == 'primary' else tier_key + '_picks'}.json").write_text(
            json.dumps({"metadata": summary, "picks": items}, indent=2,
                        ensure_ascii=False),
            encoding="utf-8",
        )

    # Update performance ledger
    existing_pl = (
        json.loads(PERFORMANCE_LEDGER_PATH.read_text(encoding="utf-8"))
        if PERFORMANCE_LEDGER_PATH.exists() else None
    )
    updated_pl = update_performance_ledger(resolved, existing_pl)
    PERFORMANCE_LEDGER_PATH.write_text(
        json.dumps(updated_pl, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Update calibration tracker
    existing_cal = (
        json.loads(CALIBRATION_TRACKER_PATH.read_text(encoding="utf-8"))
        if CALIBRATION_TRACKER_PATH.exists() else None
    )
    updated_cal = update_calibration_tracker(resolved, existing_cal)
    CALIBRATION_TRACKER_PATH.write_text(
        json.dumps(updated_cal, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=None,
                         help="Date to settle (YYYY-MM-DD). Defaults to yesterday UTC.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Skip writes; log what would happen.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    try:
        settle_day(target_date, dry_run=args.dry_run)
    except Exception as exc:
        logger.error("settle_results failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
