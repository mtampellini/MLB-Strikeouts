"""Fit the Negative Binomial dispersion parameter for pitcher K counts.

Implemented in Phase 4. Outline:
- Pull every starter game from 2024-01-01 to 2025-12-31 via Statcast.
- For each game, compute projected mean K via simple proxy (season K% x actual BF).
- Fit NB dispersion alpha via MLE on (observed K, projected mean K) pairs.
- Write data/processed/nb_dispersion.json with the fitted alpha. LOCKED.
- Produce a calibration plot: observed vs predicted CDF across quintiles of
  projected mean.

Re-run ONLY when a new season is added. Never per pick run.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Phase 4 — not yet implemented.")


if __name__ == "__main__":
    main()
