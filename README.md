# MLB Strikeouts

MLB strikeout prop betting model + front-end. Single repo, single deploy. Mirrors
the architecture of HR-Picks (https://github.com/mtampellini/MLB-Home-Runs) but
targets pitcher strikeouts via The Odds API market key
`pitcher_strikeouts_alternate` on FanDuel and DraftKings.

```
/                       <- Next.js root, deployed to Vercel
├── src/pages/          <- /strikeouts, /strikeouts-tracker, /calibration, /personal-bets
├── picks.json          <- written by the daily cron, picked up at build time
├── backend/            <- Python pipeline (data, projection, picks, results)
│   ├── src/
│   │   ├── data/           <- ProbablesClient, StatsAPI, Statcast, OddsAPI, AsOfContext
│   │   ├── features/       <- E[BF] features + P(K|PA) features
│   │   ├── projection/     <- E[K], NB CDF, P(K ≥ line)  (BOOK-AGNOSTIC)
│   │   ├── picks/          <- edge/EV/three-tier selection  (FD/DK-specific)
│   │   ├── pipeline/       <- run_daily, run_settlement
│   │   ├── results/        <- settlement, tracker, CLV logging
│   │   └── backtest/       <- as-of walk-forward
│   ├── tests/
│   ├── scripts/            <- fit_nb_dispersion.py, backtest_2024_2025.py
│   └── data/
│       ├── odds/                       <- committed (this IS our dataset)
│       ├── archive/YYYY-MM-DD/         <- daily snapshots
│       ├── processed/
│       │   ├── park_k_factors.parquet  <- committed (precomputed once)
│       │   ├── nb_dispersion.json      <- committed (locked after fit)
│       │   └── tracker.json            <- committed (updated nightly)
│       └── raw/                        <- gitignored Statcast caches
└── .github/workflows/
    ├── daily_picks.yml      <- 11am ET cron, runs run_daily.py, commits picks.json
    └── settle_results.yml   <- next-morning cron, settles + updates tracker
```

Full architecture, build phases, hard rules, and validation gates are in
[`backend/README.md`](backend/README.md).

## Local development

```powershell
# Backend (Python 3.11+):
cd backend
pip install -e ".[dev]"
pytest tests/
python -m src.pipeline.run_daily  # generates picks.json + dated artifacts

# Front-end (Node 18+):
cd ..
npm install
npm run dev                       # http://localhost:3000
```

## Secrets

GitHub Actions secrets required:
- `ODDS_API_KEY` — from https://the-odds-api.com (500/month on free tier; shared
  budget with HR repo)
