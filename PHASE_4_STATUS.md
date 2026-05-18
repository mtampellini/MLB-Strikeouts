# Phase 4 status — HALTED at 4c

**Date:** 2026-05-17
**State:** Sample fit ATTEMPTED, gates FAILED, calibration not deployed.

## What's done

### Phase 4a (league averages) — ✅ COMPLETE on full data
- `backend/scripts/derive_league_averages.py`
- Full 2024 + 2025 Statcast pulls, regular season only
- Output: `data/processed/league_averages_{2023,2024,2025}.json`
- Sanity checks all pass: league K% in [0.21, 0.24], RR > LR, sample sizes
  above thresholds
- 2025 K% = 22.18%, 2024 K% = 22.54%, 2023 K% = 22.69% (declining trend)

### Phase 4b (park K factors) — ✅ COMPLETE on full data
- `backend/scripts/derive_park_k_factors.py`
- 3-season pull (2023-2025), shrunk observed/expected per venue
- Output: `data/processed/park_k_factors.json`
- All 30 current MLB venues present (Statcast normalizes Athletics-2023/24 to
  ATH→Sutter Health rather than OAK→Coliseum; both physical venues blend
  into 2529)
- Highest K factor: T-Mobile (1.16); lowest: Nationals (0.97)
- Spec's prior assumption that Coors should be < 1.0 was wrong for the
  current ball/rule environment — Coors is 1.04 (slightly K-promoting).
  Sanity check updated from "Coors < 1.0" to range-based distribution
  check.

### Phase 4c (feature coefficients) — ❌ ATTEMPTED, GATES FAILED
- `backend/scripts/fit_feature_coefficients.py` runs cleanly.
- 1000-game stratified sample (24 strata: 2 seasons × 4 K% quartiles ×
  3 park types). Actual size 944 (some strata short).
- Walk-forward: 473 train (2024) / 471 test (2025).
- Output JSONs `feature_coefficients_{bf,kpa}.json` written with
  `sample_fit: true` and full leakage-shuffle diagnostics.

**Gate results:**

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| E[BF] out-of-sample R² | ≥ 0.12 | 0.10 | ❌ FAIL |
| P(K|PA) out-of-sample R²_logit | ≥ 0.25 | **−1.53** | ❌ FAIL |
| pitcher_k_pct_season coef > 0.5 | yes | 10.28 (but unstable) | ⚠️ unstable |
| No coefficient sign flips from expected | yes | `pitcher_pa_per_start_season` came in negative | ❌ FAIL |
| Leakage shuffle: real R² > shuffled | yes | BF shuffled higher than real | ❌ FAIL |

**Coefficient instability — multicollinearity diagnosis:**

The P(K|PA) fit produced absurd coefficients:
- `intercept_logit = +112.5878`
- `league_k_pct_vs_hand = −525.2249`
- `pitcher_k_pct_season = +10.2842`

These offset each other to produce reasonable predictions on the training
set but blow up out-of-sample. Root cause: `league_k_pct_vs_hand` is
constant within each (season, hand) — essentially a 4-valued
categorical — and is highly correlated with `pitcher_k_pct_season`.
The OLS solver has too few degrees of freedom and finds an unstable
solution.

**Generalization failure between 2024 and 2025:**

League K% dropped 0.36pp YoY (22.54% → 22.18%). The fit was trained on
2024 levels and applied to 2025 games — projections systematically
overestimate the 2025 K rate. The leakage-shuffle test confirms this:
shuffled R² (0.05) is better than real R² (−1.53), meaning the temporal
split actively HURTS the model rather than helping.

**Scope reduction that contributed to the failure:**

This session's fit dropped many features compared to the full spec
(documented in the script's `missing_features` JSON list):
- 30-day rolling features (ip_per_start_30d, pitches_per_pa_30d,
  pitcher_k_pct_30d_blended)
- All pitch-level features (CSW, chase-whiff, velocity trend, putaway)
- Per-batter lineup aggregation (used league-vs-hand as a constant proxy)
- Weather, umpire, days-rest features

With a richer feature set, the multicollinearity between
`pitcher_k_pct_season` and `league_k_pct_vs_hand` would be diluted.

### Phase 4d (NB dispersion) — ⏸️ NOT RUN
Depends on a calibrated mean from 4c. Cannot run until 4c passes gates.

## What's NOT done (and won't be in this session)

- Coefficients are NOT wired into `features_bf.py` / `features_kpa.py`.
  Production projector still uses Phase 3 placeholder coefficients.
- The "SAMPLE FIT — NOT PRODUCTION" startup warning is NOT added (would
  only be wired after a passing fit).
- The Phase 2c sample bundle is NOT re-run with new coefficients.
- Phase 4d (NB dispersion fit) is NOT executed.

## Required next steps before Phase 5

### Option A: full overnight rerun with richer features
Run `fit_feature_coefficients.py --full` after extending it to include the
30-day rolling and pitch-level features. The vectorization work for
rolling 30d windows over 15k starter-games is straightforward but was
out of scope this session (the naive pandas-loop implementation went
quadratic).

### Option B: re-parameterize to dodge multicollinearity
Replace `league_k_pct_vs_hand` (a near-constant) with `pitcher_k_pct_season
- league_k_pct_vs_hand` (the centered "delta from league"). This makes
the design matrix well-conditioned. Re-run the sample fit; if gates pass,
proceed to Phase 4d.

### Option C: simpler additive model
Drop the log-odds composition for P(K|PA) and fit a direct additive model
in K-rate space. Loses the (0, 1) guarantee but may be more stable on
small samples. Less principled, but pragmatic if multicollinearity keeps
breaking the logit fit.

## Files committed in this Phase 4 attempt

### Full-data, FINAL
- `backend/scripts/derive_league_averages.py` + 3 output JSONs
- `backend/scripts/derive_park_k_factors.py` + 1 output JSON
- `backend/tests/test_derive_league_averages.py`
- `backend/tests/test_derive_park_k_factors.py`

### Sample fit, DIAGNOSTIC ONLY
- `backend/scripts/fit_feature_coefficients.py`
- `backend/data/processed/feature_coefficients_{bf,kpa}.json` (gate-failing
  fit; preserved for diagnostic purposes, NOT used by the projector)

### NOT created
- `backend/scripts/fit_nb_dispersion.py` (deferred to post-4c-pass)
- Updated `features_bf.py` / `features_kpa.py` reading from the coefficient
  JSONs (deferred to post-4c-pass)
- `coefficient_fit_report.html` (deferred)

## Bottom line

**Phase 4 is not complete.** League averages and park K factors are real
and final. Feature coefficients are NOT calibrated; the projector still
uses Phase 3 hand-coded coefficients. Phase 5 (picks layer) MUST NOT
deploy real money against an uncalibrated model.

Recommended next action: re-parameterize the K|PA features to fix the
multicollinearity (Option B above), run the sample fit again, and if it
passes the gates, proceed to wiring + Phase 4d.
