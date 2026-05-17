"""Walk-forward backtest 2024-04-01 to 2025-10-01.

Implemented in Phase 6. Outline:
- Re-run the full projection pipeline as-of every game date in the window.
- Strict AsOfContext for every feature. No end-of-season aggregates.
- No historical odds available, so no ROI claims. Projection accuracy only.
- Output: backtest_report.html with:
    * Projection MAE by pitcher quartile.
    * Bias by line bucket (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5+).
    * Calibration: NB-predicted P(K >= line) vs observed frequency.
    * Cohort bias by park, lineup K% quartile, days_rest.

Gate (in main()): no cohort bias > 0.3 K, no calibration deviation > 5pp.
Pipeline must not proceed to Phase 7 if any gate fails.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Phase 6 — not yet implemented.")


if __name__ == "__main__":
    main()
