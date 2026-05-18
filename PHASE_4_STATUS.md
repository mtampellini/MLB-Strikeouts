# Phase 4 status

**Last update:** 2026-05-17 (re-parameterization + full-feature wiring landed)

## State summary

| Sub-phase | Description | State |
|-----------|-------------|-------|
| 4a | League averages (full data) | ✅ DONE, final |
| 4b | Park K factors (full data) | ✅ DONE, final |
| 4c | Feature coefficients | ⏳ Re-parameterized + full-feature wiring committed; awaiting overnight `--full` rerun |
| 4d | NB dispersion | ⏸ Blocked on 4c passing gates |

The projector currently uses Phase 3 placeholder coefficients. **DO NOT
deploy real money against the current model.** The overnight rerun produces
the calibrated coefficients that gate Phase 5.

## Phase 4c first attempt — diagnostic record

Failed gates on 944-game stratified sample:

| Gate | Threshold | Observed | Status |
|------|-----------|----------|--------|
| E[BF] out-of-sample R² | ≥ 0.12 | 0.10 | ❌ FAIL |
| P(K|PA) out-of-sample R²_logit | ≥ 0.25 | **−1.53** | ❌ FAIL |
| `pitcher_pa_per_start_season` sign | positive | negative | ❌ FAIL |
| Leakage shuffle: real ≥ shuffled | yes | shuffled higher | ❌ FAIL |

**Root cause: structural collinearity.** `pitcher_k_pct_season` (per-pitcher
continuous) and `league_k_pct_vs_hand` (4-valued per season×hand) both
expressed league-level information. OLS landed in an identifiable-but-
unstable basin (`intercept_logit=+112.6`, `league_coef=−525.2`,
`pitcher_coef=+10.3`) that fit 2024 training but blew up at 2025 test
time — compounded by the 0.36pp YoY league K% drop.

The collinearity is structural, not feature-count-dependent. Adding more
features dilutes the symptom but doesn't fix the cause.

## Re-parameterization (this commit)

The K|PA design matrix now uses **delta-from-baseline** parameterization:

- `pitcher_k_pct_delta = pitcher_k_pct_season_shrunk - league_k_pct_vs_hand`
- `league_k_pct_vs_hand` kept as the explicit baseline
- 30d K% expressed as `delta-from-pitcher-season` (form vs own baseline)
- CSW%, chase-whiff%, putaway: centered on league anchors
- Velocity trend: already a Z-score, kept as-is
- Lineup K% / zone-contact / chase: centered on league-vs-hand anchors
- `park_k_factor` and `umpire_k_factor`: log-transformed (additive in log-odds)

The E[BF] design matrix uses analogous deltas:

- `pitcher_pitches_per_pa_30d_delta = 30d - season` (form vs own baseline)
- Park factors log-transformed

Each coefficient now has a clean identification: pitcher quality is the
signed deviation from league; league level is its own explicit feature.

## Full-feature wiring

The fit script now calls the **production feature builders**
(`compute_e_bf`, `compute_p_k_pa`) on historical bundles constructed from
cached Statcast — eliminates fit-time/projection-time divergence. The
full Phase 3 feature set participates in the regression: 30-day rolling,
pitch-level (CSW / chase-whiff / velocity / putaway), real per-batter
lineup aggregation, park K factor, umpire K factor.

`scripts/historical_bundle.py` is the new module that:

1. Builds per-pitcher and per-batter Statcast caches once from the bulk
   2023-2025 pull (already cached by pybaseball)
2. For each starter-game, slices the cached data by date and constructs a
   `ProjectionBundle`
3. The fit script then calls the same production builders that run in
   live projection

## Pre-fit sanity checks (`scripts/design_matrix.py`)

`check_design_matrix` halts the script BEFORE OLS runs if any of:

- NaN columns
- Zero-variance columns
- Condition number on the standardized matrix > 100
- Delta columns not centered (mean above tolerance)

These are regression-tested in `tests/test_design_matrix.py` (14 tests). A
future change that reintroduces a collinear pair will fail those tests
without burning a full overnight cycle.

The condition-number check alone wouldn't have caught the Phase 4c first
attempt (the matrix wasn't strictly singular — it was nearly redundant in
a way that produced unstable but technically-valid OLS solutions). The
real fix is the re-parameterization; the condition check is defensive
sanity to catch unrelated future regressions.

## In-session validation

This session did **NOT** run a real fit. The deliverable is code readiness
for the overnight `--full` rerun on user's machine. The end-to-end pipeline
is smoke-testable via:

```
python -m scripts.fit_feature_coefficients --sample 50 --smoke
```

`--smoke` builds features + runs sanity checks but skips OLS. Confirms the
bundle-builder + production-feature path works end-to-end without
committing to a real fit's compute.

## Overnight rerun procedure

On user's machine:

```powershell
cd C:\Users\mtamp\Documents\MLB-Strikeouts\backend
python -m scripts.fit_feature_coefficients --full
```

Expected behavior:

1. Statcast cache load (~2 min if seasons already cached, ~30 min cold)
2. Pitcher + batter cache build (~1 min)
3. Bundle construction + feature extraction for ~5000 starter games
   (~30-90 min depending on per-game feature compute)
4. Re-parameterization (instant)
5. **Pre-fit sanity checks** — halts here if design matrix unhealthy
6. OLS for E[BF] + weighted-OLS in logit space for P(K|PA)
7. Walk-forward 2024-fit / 2025-holdout
8. Output to `data/processed/feature_coefficients_{bf,kpa}.json`

If gates pass (E[BF] R² ≥ 0.15, P(K|PA) R²_logit ≥ 0.30, no sign flips,
leakage check confirms real > shuffled), proceed to:

1. Wire coefficients into `features_bf.py` / `features_kpa.py`
2. Add "SAMPLE FIT" warning logger (or remove if `sample_fit: false`)
3. Run projector on Phase 2c sample bundle, document delta
4. Execute Phase 4d (`fit_nb_dispersion.py`)
5. Update this status doc to "Phase 4 complete"

If gates fail: halt, surface to chat, do not proceed to Phase 4d.

## Files committed in Phase 4 (cumulative)

### Real, final
- `backend/scripts/derive_league_averages.py` + 3 output JSONs
  (2023, 2024, 2025)
- `backend/scripts/derive_park_k_factors.py` + output JSON
- `backend/tests/test_derive_league_averages.py` (11 tests)
- `backend/tests/test_derive_park_k_factors.py` (8 tests)

### Ready for overnight, not yet validated
- `backend/scripts/fit_feature_coefficients.py` (re-parameterized,
  full-feature-wired, calls production builders)
- `backend/scripts/historical_bundle.py` (new — bundle builder for
  historical games)
- `backend/scripts/design_matrix.py` (new — re-parameterization +
  sanity checks)
- `backend/tests/test_design_matrix.py` (14 tests)

### Failed-fit diagnostic (preserved, NOT used in production)
- `backend/data/processed/feature_coefficients_{bf,kpa}.json` from the
  first attempt — marked `sample_fit: true`. The overnight rerun
  overwrites these.

## What's still TODO (post-overnight)

- Phase 4d: `backend/scripts/fit_nb_dispersion.py` (cannot write until 4c
  passes — depends on the calibrated mean)
- Wire coefficients into `features_bf.py` / `features_kpa.py`
- Remove Phase 3 placeholder annotations
- Run projector on Phase 2c sample bundle, document delta
- HTML diagnostic report (deferred from original spec; gate metrics in
  the JSON outputs are sufficient for now)
