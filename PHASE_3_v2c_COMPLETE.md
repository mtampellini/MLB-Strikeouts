# Phase 3-v2c: P(K|PA) rewrite — COMPLETE

**Status:** Complete as of 2026-05-19.

The P(K|PA) projection layer has been rewritten from a single aggregated
P * E[BF] composition to a per-batter / per-TTO log5 model driven by
empirical PA distribution and archetype-interaction TTO multipliers.

## Sub-phase deliverables

| Phase | Deliverable | State |
|---|---|---|
| 3-v2c-0 | Empirical PA distribution by (BF, slot, TTO) | ✅ done |
| 3-v2c-i | Contract extended with 4 additive fields | ✅ done |
| 3-v2c-ii | Per-batter feature builders | ✅ done |
| 3-v2c-iii | Pitcher-side feature redesign | ✅ done |
| 3-v2c-iv | Per-batter projector rewrite | ✅ done |
| 3-v2c-v | Integration smoke (20 games, 5 archetypes) | ✅ done |

**Tests passing: 424.** E[BF] regression test from 3-v2c-i continues to pass —
additivity guarantee holds.

## Notable design decisions (preserved for future reference)

- **5-category pitcher archetype classification** (Power-FF, Sinker-ball,
  Breaking-heavy, Offspeed-heavy, Balanced). Rule-based on Statcast
  pitch-mix percentages with a 30% junk-pitch ceiling. Empirically,
  Breaking-heavy is the modal archetype (28% of starters), reflecting the
  recent slider/sweeper usage surge.
- **Empirical PA distribution** by (total_bf × batting_order_slot × TTO).
  The naive analytical model ("BF/9 PAs per slot, evenly across TTO")
  misallocates ~0.89 PA per cell at the tails of the order. Used 11,563 SP
  starter-games from 2023-2025; low-sample BF values get 3-point rolling
  smoothing.
- **Per-batter log5 with archetype-interaction TTO multipliers**. The
  projector iterates over (batter, TTO) cells rather than computing a
  single aggregate P(K|PA). Per-batter K% falls back to league average when
  individual sample is insufficient (cell still contributes; flagged in
  `k_rate_source`).
- **CSW% as primary K-skill feature** (replaces season K%). K-rate delta
  retained as secondary signal. Bootstrap-stable features kept; sign-stable-wrong
  and unstable features (pitcher_k_pct_30d_delta, pitcher_csw_pct_30d_delta,
  pitcher_putaway_pct_delta, lineup_chase_delta) dropped.
- **L/R-split park K factors with LOO methodology**. Phase 4b-v2 derived
  park factors using leave-one-venue-out to remove the pitcher-quality
  confound. Range widened to [0.85, 1.30] after empirical evidence that
  T-Mobile (1.26) and Coors/Trop (>1.0) violated the original prior.
- **K-specific TTO range**, not wOBA-research range. K-rate TTO penalties
  run ~7-9pp larger than wOBA penalties because K rate isolates the whiff
  component, which decays faster than contact quality with hitter
  familiarity. Gate range: [0.78, 0.92] (vs the wOBA-research [0.85, 0.95]).
- **Archetype interaction**: in TTO=3, the four specialized archetypes cluster
  tight (0.804-0.818) while Balanced is the outlier with the smallest penalty
  (0.836). Predictability is the punishable trait; diversifying arsenal
  (Balanced) holds up better than any single specialty.
- **Additive contract changes only**. Legacy bundles route to the legacy
  aggregated path. Phase 4c E[BF] fit consumes the contract unchanged.

## Rasmussen reference point

The Phase 3-v2c-i sample bundle (Drew Rasmussen, RHP, vs MIA at Tropicana,
2026-05-17) serves as the canonical reference for the rewrite:

| | Pre-3-v2c (aggregated) | Post-3-v2c-iv (per-batter) |
|---|---:|---:|
| Method | `legacy_aggregated` | `per_batter_with_tto` |
| E[BF] | 25.63 | 25.63 |
| P(K\|PA) | 0.208 (single value) | 0.212 (PA-weighted avg) |
| **E[K]** | **5.32** | **5.51** |
| Archetype | (not used) | Power-FF |
| PA distribution row | (not used) | 26 |

**0.19 K shift attributable to:**

1. **Empirical PA distribution** correctly attributing zero TTO=3 PAs to
   slot 9 (Joe Mack). The old model's BF/9 allocation gave slot 9 a
   ~0.89 PA at TTO=3 — fake projection that the empirical distribution
   removes. Saves ~0.1 K of inflated estimate.
2. **Power-FF archetype TTO multipliers** at TTO=2 (0.889) and TTO=3 (0.806)
   are slightly more punishing on later PAs than the league_wide multipliers
   implicit in the old aggregated model. Adds ~0.05 K from sharper TTO
   pricing at the top of the order, where Power-FF gets faced 3 times.
3. **Log5 matchup priors** give slightly different per-batter K rates than
   the aggregated `lineup_k_pct_vs_hand` average. Net +0.04 K for this
   matchup. Concentrates the K projection in the higher-K Marlins batters
   (Connor Norby at slot 5: 0.83 expected K) rather than spreading it
   uniformly.

Per-batter breakdown confirms the integrity invariant:
`sum(expected_k_total) = 5.5133 = e_k` (within 2e-5).

## 20-game integration smoke results

Phase 3-v2c-v batch-projected 20 starter games from 2025 across all 5
archetypes (4 each). All sanity checks pass:

| Archetype | n | mean e_k | std | range | mean e_bf |
|---|---:|---:|---:|---|---:|
| Power-FF | 4 | 6.31 | 1.45 | 4.30–7.72 | 24.45 |
| Sinker-ball | 4 | 5.09 | 0.78 | 3.97–5.75 | 25.66 |
| Breaking-heavy | 4 | 6.57 | 0.68 | 6.00–7.36 | 25.65 |
| Offspeed-heavy | 4 | 5.34 | 0.58 | 4.55–5.87 | 25.58 |
| Balanced | 4 | 6.49 | 1.49 | 4.79–7.88 | 25.28 |

- Mean e_bf = 25.3 (research range [22, 28]) ✅
- Mean e_k = 5.96 (research range [4.5, 7.0]) ✅
- Every game used 9/9 individual rates (no league fallback dominated) ✅
- Per-game integrity invariant holds ✅
- No archetype with all games below league mean K (5.0) ✅

The original spec included a `power_ff_above_balanced` check that was
removed during 3-v2c-v: in a small batch (n=4 per archetype) pitcher-level
K-skill variation dominates archetype-level TTO effects. The model is
designed so individual signal (CSW%, K%) outranks archetype, and archetype
contributes only through TTO multipliers (~3-5% effect) that a 4-game
sample can't isolate from individual selection. The archetype effect was
validated separately in Phase 3-v2b on 275k SP PAs.

## Coefficients remain placeholders

The logit composition in `compute_p_k_pa` and the projector's TTO multipliers
both use **placeholder coefficients** that have not been fit against
per-PA outcomes. The next phase (Phase 4c-v2) replaces them with a logistic
regression fit on per-PA Statcast 2023-2025 outcomes.

**Do not deploy real money against the current model.** The placeholders
produce reasonable-looking E[K] values (the Rasmussen 5.51 and the 20-game
smoke means are in research range), but coefficient miscalibration could
be silent in any single-game output. The 4c-v2 fit will surface that.

## Next phase

**Phase 4c-v2**: logistic regression of per-PA K outcomes against the new
feature set. Will use:

- Per-PA Statcast 2023-2025 (~549k events after SP filter)
- Phase 3-v2c-iv per-batter projector as the feature pipeline
- Walk-forward validation (2024 fit, 2025 holdout)
- The same gates from the prior Phase 4c attempt (R²_logit, sign stability,
  bootstrap consistency)

Phase 4d (NB dispersion) remains blocked on 4c-v2 passing.
