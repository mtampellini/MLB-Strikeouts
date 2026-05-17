# MLB Strikeouts — backend

MLB pitcher strikeout prop betting model. Decomposes expected strikeouts as
`E[K] = E[BF] × P(K|PA)`, prices alt lines via a Negative Binomial distribution
(empirically-fit dispersion — Poisson under-fits the tails), and surfaces
three tiers of picks against FanDuel / DraftKings.

Architecturally mirrors the HR-Picks repo. Same patterns, different target
variable and feature set.

---

## The architectural principle: decouple projection from picks

The **projection** layer is a pure function of pitcher + game state. It produces
`E[K]`, the NB CDF, and `P(K ≥ line)` for every line. It is book-agnostic.

The **picks** layer consumes projections and applies book-specific selection
logic (FD/DK devig, edge_pct, EV%, three-tier ranking).

They live in separate modules — `src/projection/` and `src/picks/` — so a second
picks layer (different book, different selection criteria) can wire onto the
same projections later without refactoring. A scaffold test enforces that
`projection/` has zero imports from `picks/`.

---

## Build phases

| Phase | What ships | Status |
|-------|------------|--------|
| 1 | Repo scaffolding | ✅ |
| 2 | Data layer: ProbablesClient (FG + StatsAPI chain), StatsAPI, Statcast, OddsAPI, AsOfContext | ⬜ |
| 3 | Feature engineering (E[BF] additive set + P(K\|PA) multiplicative set) + hard filters | ⬜ |
| 4 | NB dispersion fit (`scripts/fit_nb_dispersion.py`) — locked once fit | ⬜ |
| 5 | Pricing engine: NB CDF → P(K ≥ line), devig, edge, EV, three-tier selection | ⬜ |
| 6 | Backtest (`scripts/backtest_2024_2025.py`) — MUST PASS GATES before launch | ⬜ |
| 7 | Daily pipeline: `run_daily.py`, `run_settlement.py`, archives, tracker, CLV | ⬜ |
| 8 | Front-end (mirrors HR's `src/pages/`) | ⬜ |
| 9 | GitHub Actions cron workflows | ⬜ |

Strict order. Do not skip ahead. Do not proceed past Phase 6 if backtest gates fail.

---

## Model decomposition

```
E[K] = E[BF] × P(K|PA)
```

Distribution: **Negative Binomial** with empirically-fit dispersion (`scripts/
fit_nb_dispersion.py`). **Not Poisson.** Pitcher-game K counts are overdispersed
and Poisson mis-calibrates the tails — this is the most common silent failure
mode for strikeout prop models.

The dispersion parameter is fit once on 2024-2025 starter games and **locked**.
Stored at `data/processed/nb_dispersion.json`. Re-fit only when a new season is
added — never per pick run.

---

## "Starter" definition (opener handling)

A "starter" here is **the pitcher expected to face the most batters in the
game**, not necessarily the listed probable. On opener days, this is the bulk
reliever, not the opener.

Detection chain (`src/data/probables.py`):
1. **Primary:** FanGraphs RosterResource probables (scrape, cache 1hr).
2. **Fallback:** MLB StatsAPI `probablePitcher` field.
3. **Cross-check:** if FanGraphs disagrees with StatsAPI, log it and trust FanGraphs.
4. **Opener override triggers** (skip listed probable, look for bulk pitcher):
   - Listed probable has <30 IP this season AND avg IP/outing < 2.
   - Team used an opener at this rotation slot in the last 7 days.
   - Listed probable has no posted K prop on FD/DK but another team pitcher does
     (market signal — trust the market).
5. **Universe filter:** pitcher must have a posted `pitcher_strikeouts_alternate`
   prop on FD or DK. This catches most opener edge cases for free.
6. **Hard filter:** `projected_BF < 12` catches residual book-mispriced openers.

---

## Feature set

### E[BF] (additive)
- `pitcher_ip_per_start_30d` (Bayesian shrinkage → season → prior year)
- `pitcher_pitches_per_pa_season`
- `pitcher_pitches_per_pa_30d`
- `team_bullpen_short_hook_indicator` (manager-specific, persistent)
- `lineup_obp_vs_hand`
- `park_run_environment_factor`
- `weather_run_environment` (wind-out + temp + humidity composite)
- `days_rest` (categorical: `<4`, `4`, `5`, `6+`)

### P(K|PA) (multiplicative / log-odds)
- `pitcher_k_pct_season` (empirical-Bayes shrink to prior year)
- `pitcher_k_pct_30d` (Bayesian blend with season)
- `pitcher_csw_pct_30d`
- `pitcher_chase_whiff_pct_30d`
- `pitcher_velocity_trend_3starts` (Z-score vs career)
- `pitcher_putaway_pitch_concentration`
- `lineup_k_pct_vs_hand` (weighted by batting-order spot)
- `lineup_zone_contact_pct` ← the key book-underweighted feature
- `lineup_chase_rate`
- `park_k_factor` (precomputed → `data/processed/park_k_factors.parquet`)
- `umpire_k_zone_factor` (rolling 162-game)
- `catcher_framing_runs` (CSAA, latest)

### Combination (`src/projection/`)
- `E[BF] = baseline_bf + Σ(additive adjustments)`, clipped to `[12, 32]`
- `P(K|PA) = pitcher_k_rate × Π(ratio adjustments)`, clipped to `[0.10, 0.45]`
- `E[K] = E[BF] × P(K|PA)`

### Hard filters (skip pitcher entirely if ANY fail)
- Career IP < 50
- (Season IP < 20) AND (Prior-year IP < 80)
- **Any feature missing.** No median fill. Ever.
- Projected BF < 12 (opener / bullpen game)
- Not in the database (no synthetic generation)
- Opener detected and no bulk pitcher identified
- No posted K prop on FD or DK for this pitcher

---

## Three-tier picks (`src/picks/`)

| Tier | Criteria | Ranking |
|------|----------|---------|
| Primary | `edge_pct ≥ 20%` AND `book_price ≥ -180` | by `edge_pct` (NOT EV%) |
| Secondary | `edge_pct ≥ 20%` AND (`book_price < -180` OR rank 11+ in primary) | by `edge_pct` |
| Shadow | `10% ≤ edge_pct < 20%` | by `edge_pct` |

Edge: `(model_p - market_p) / market_p`, where `market_p` is the devigged
probability (multiplicative devig, same method as HR repo).

Primary ranks by `edge_pct`, not `EV%` — ranking by EV creates a long-shot
leverage bias that has bitten this kind of model before.

---

## Hard rules (carried from HR — do not violate)

1. **No median fill.** Skip the pitcher if any feature is missing.
2. **No synthetic odds**, no reverse-engineering odds from K rates.
3. **No end-of-season aggregates in training.** Every feature is as-of the game
   date via `AsOfContext`. Tests enforce this.
4. **ML infrastructure ships dormant.** Only baseline empirical-Bayes model active
   until 60+ days of logged odds outcomes.
5. **CLV tracking from Day 1.** Log closing line at game start for every pick.
6. **Hard filters in Phase 3 are absolute** — no overrides.
7. **Rank primary picks by `edge_pct`**, never by EV%.
8. **NB dispersion is locked** at the value from `fit_nb_dispersion.py`. Not
   re-fit per pick run.
9. **Projection layer must remain book-agnostic.** Picks layer is the only place
   book-specific logic lives. Enforced by scaffold test.

---

## Backtest gates (must pass before Phase 7)

`scripts/backtest_2024_2025.py` re-runs the full projection pipeline as-of every
game date from 2024-04-01 to 2025-10-01. Outputs `backtest_report.html`. **All
gates must pass:**

- No cohort bias > 0.3 K in any quartile (pitcher quartile, lineup K% quartile,
  park, days_rest, line bucket).
- Calibration deviation ≤ 5pp in any line bucket (3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5+).

Projection accuracy only — no historical odds, no ROI claims.

---

## Validation gates (before real money)

1. 60+ days of logged paper picks.
2. 150+ primary picks logged.
3. Calibration plot within bands across all line buckets.
4. Positive CLV trend over the window.
5. Hit rate within 3pp of implied probability across primary picks.

---

## Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env       # then fill in ODDS_API_KEY
pytest tests/
```

## Running (once Phase 7 ships)

```powershell
python -m src.pipeline.run_daily         # generate picks.json for today
python -m src.pipeline.run_settlement    # next morning, settle yesterday's picks
```
