"""Daily orchestration.

Built in Phase 7:
- run_daily.py        — idempotent merge-mode daily pick generation.
                        Writes picks.json, secondary_picks.json,
                        shadow_picks.json + data/archive/YYYY-MM-DD/.
- run_settlement.py   — next-morning W/L marking + tracker update + CLV log.
"""
