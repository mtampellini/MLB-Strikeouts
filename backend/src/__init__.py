"""MLB Strikeouts backend.

Module layout:
- src.data       External clients + AsOfContext.
- src.features   E[BF] and P(K|PA) feature builders.
- src.projection E[K], NB CDF, P(K >= line). Book-agnostic.
- src.picks      Edge / EV / three-tier selection. Book-specific.
- src.pipeline   Daily orchestration (run_daily, run_settlement).
- src.results    Settlement, tracker, CLV logging.
- src.backtest   AsOf walk-forward + calibration diagnostics.

The projection layer must remain book-agnostic; the scaffold test
`tests/test_scaffolding.py::test_projection_does_not_import_picks` enforces it.
"""

__version__ = "0.1.0"
