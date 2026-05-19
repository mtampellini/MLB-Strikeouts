# Phase 8 — HR-Picks tracker audit

Source: `C:\Users\mtamp\Documents\Fantasy Baseball\HR Picks` (the live
deployment repo behind `mlb-home-runs.vercel.app`).

## Stack

- **Next.js 14.2.5**, **React 18.3.1**, **react-dom 18.3.1**. No other
  runtime dependencies.
- **Pages Router** (`src/pages/*.js`), not the App Router.
- **Plain JavaScript** (no TypeScript).
- **Static export**: `next.config.js` sets `output: 'export'`,
  `images.unoptimized: true`, `trailingSlash: true`. Site builds to
  `out/` and is served as static HTML by Vercel.
- **No CSS framework**. No Tailwind. No CSS modules. No styled-components.
  Just inline styles via a shared theme object `T`.
- **No chart library**. No recharts, no chart.js, no plotly. Charts
  (e.g. cumulative units chart) are hand-rolled SVG or simple bar tables.

## File layout

```
HR Picks/
├── package.json              # next/react only
├── next.config.js            # output: 'export'
├── picks.json                # Primary picks; daily_picks Action writes here
├── secondary_picks.json
├── shadow_picks.json
├── public/                   # static assets
└── src/
    ├── pages/
    │   ├── _app.js           # 4-line wrapper, imports globals.css
    │   ├── globals.css       # 8 lines: html/body reset + bg color
    │   ├── index.js          # 388 lines: picks display
    │   └── tracker.js        # 862 lines: performance tracker
    └── components/           # empty in the live repo; everything inline
```

Plus `backend/` Python pipeline that produces the JSON files. Front-end
reads them at build time via static `import` (index.js) or
`getStaticProps` + `fs.readFileSync` (tracker.js).

## Theme (mirror exactly)

```js
const T = {
  bg: '#ffffff',
  border: '#e5e5e5',
  borderStrong: '#d4d4d4',
  text: '#0a0a0a',
  textMedium: '#525252',
  textLight: '#a3a3a3',
  bgSubtle: '#f5f5f5',   // tracker.js only
  accent: '#2563eb',
  positive: '#16a34a',
  negative: '#dc2626',
}
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif'
const TABULAR = { fontVariantNumeric: 'tabular-nums' }
```

White background, near-black text. Quiet gray secondary text. Subtle
gray borders (no shadows, no gradients). Green for positive numbers,
red for negative. Blue accent reserved for links/hover.

Number formatting always tabular (`fontVariantNumeric: 'tabular-nums'`).

## Page structure

### `/` (index.js)

- `<Head>`: `<title>HR Picks</title>` + viewport meta
- Inline `<style jsx global>` block: 2 lines (bg + a:hover)
- Container: `maxWidth: 1080`, `margin: '0 auto'`, `padding: '40px 24px'`
- **Header**: site title (22px / 700 weight / letterSpacing -0.4) +
  Tracker link (14px / textMedium). Below: model version · As of · EV
  threshold · Generated-at timestamp (12px / textLight).
- **Stat row**: 4 `<StatCard>` components in flex row, gap 14. Each card:
  bordered, 24-26px padding, 32px / 600 weight number + 12px label +
  optional 11px sub.
- **Methodology accordion**: collapsible section explaining the model.
  ▾ chevron rotates on open. Inside: prose paragraphs + columns table.
- **PicksTable**: `<table>` with 9 columns (Batter / Park / vs Pitcher /
  Line / Model / Market / Odds / EV / Why). Sticky header. Borders
  between rows.
- **Footer**: 11px textLight legal/disclaimer line at the bottom of the
  container (no separate footer container).

### `/tracker` (tracker.js)

- Same header pattern as index but with "Tracker" title + "← Picks" link
- `getStaticProps` reads:
  - `backend/data/daily_archives/YYYY-MM-DD.json` files
  - `backend/data/processed/tracker.json`
- Tier selector (radio buttons): All / Primary / Secondary / Shadow
- **Stat cards row**: total picks, hit rate, units profit, ROI, CLV
- **Cumulative units chart**: hand-rolled SVG bar chart of daily units
  profit accumulated over time
- **Per-day archive table**: one row per day, columns for date / picks /
  W-L-V / units / cumulative
- **Per-archive expandable detail**: click a date to expand its picks
  with their settlement results

### No `/calibration` or `/personal-bets` in HR

The Phase 8 spec for Strikeouts asks for both. HR doesn't have them, so
they're additions specific to the K-prop project. Build them in the
SAME visual style as the existing pages.

- `/calibration` will visualize Phase 4d's per-line predicted vs
  observed P(K >= line) plus the live calibration_tracker.json that
  accumulates during paper trade. Hand-rolled SVG bar chart (no chart
  library, per HR convention).
- `/personal-bets` is a user-curated list of bets the user actually
  placed (model picks vs personal bets separation). Same table style as
  the picks page.

## Quirks to preserve

- Trailing slash on URLs (`/tracker/` not `/tracker`). Set globally via
  `trailingSlash: true`. Internal links should NOT include the trailing
  slash in `href=` — Next.js handles it.
- Inline styles only, no className. Each component owns its style.
- No `useEffect` for data fetching — everything is build-time static.
- `<Head>` per page, not a shared layout component.
- `formatOdds`: positive prepends `+`, negative keeps as-is. Null → `—`.
- Game time formatting: `toLocaleTimeString` with `hour: 'numeric',
  minute: '2-digit'` and manual am/pm lowercase.
- 22px / 700 / -0.4 letter-spacing for page titles. 14px / 400 for
  navigation links.

## Data integration plan for Strikeouts

Mirror HR's static-import-at-repo-root pattern. The Phase 7 orchestrator
writes to `backend/data/picks/YYYY-MM-DD/*.json`; the front-end needs
these at the repo root for static export to pick them up at build time.

Two paths:
1. **GitHub Actions step copies latest day to repo root** before
   building the front-end. This mirrors HR exactly.
2. **`getStaticProps` reads from `backend/data/picks/`** with logic to
   find the latest date directory. Used in tracker pages for HR.

For the picks page (`/`), use path 1 (static import of `picks.json`).
For the tracker / calibration / personal-bets pages, use path 2
(`getStaticProps` reads the picks history under `backend/data/picks/`).

## Tier handling

HR's `picks.json` carries a single `tier` field at the top level
("primary"). `secondary_picks.json` and `shadow_picks.json` carry the
other two. Strikeouts will follow the same convention. The Phase 5
schema already produces these three files.

Calibration_note (Phase 5 schema addition) shows up as quiet gray text
under each pick row, similar to HR's pitcher-name secondary line.
