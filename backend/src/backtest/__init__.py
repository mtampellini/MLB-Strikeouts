"""Walk-forward backtest with strict AsOfContext.

Built in Phase 6. Output: backtest_report.html with projection MAE by pitcher
quartile, bias by line bucket, NB calibration plot, cohort bias diagnostics.

Gate: no cohort bias > 0.3 K, no calibration deviation > 5pp in any line bucket.
"""
