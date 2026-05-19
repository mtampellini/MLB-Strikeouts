import Head from 'next/head'
import Link from 'next/link'
import fs from 'fs'
import path from 'path'

const T = {
  bg: '#ffffff',
  border: '#e5e5e5',
  borderStrong: '#d4d4d4',
  text: '#0a0a0a',
  textMedium: '#525252',
  textLight: '#a3a3a3',
  bgSubtle: '#f5f5f5',
  accent: '#2563eb',
  positive: '#16a34a',
  negative: '#dc2626',
}
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif'
const TABULAR = { fontVariantNumeric: 'tabular-nums' }

const CALIB_FILE = path.join(process.cwd(), 'backend/data/calibration_tracker.json')
const BACKTEST_FILE = path.join(process.cwd(), 'backend/data/processed/backtest_2024_2025_report.json')
const NB_CAL_FILE = path.join(process.cwd(), 'backend/data/processed/nb_dispersion_calibration.json')
const PICKS_DIR = path.join(process.cwd(), 'backend/data/picks')

function readJSONSafe(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) }
  catch { return null }
}

function latestCalibrationNote() {
  try {
    const dates = fs.readdirSync(PICKS_DIR)
      .filter(d => /^\d{4}-\d{2}-\d{2}$/.test(d))
      .sort((a, b) => b.localeCompare(a))
    for (const d of dates) {
      const picks = readJSONSafe(path.join(PICKS_DIR, d, 'picks.json'))
      if (picks?.picks?.[0]?.calibration_note) return picks.picks[0].calibration_note
    }
  } catch { /* directory not present */ }
  return null
}

export async function getStaticProps() {
  return {
    props: {
      liveCal: readJSONSafe(CALIB_FILE),
      backtest: readJSONSafe(BACKTEST_FILE),
      nbCal: readJSONSafe(NB_CAL_FILE),
      calibrationNote: latestCalibrationNote(),
    },
  }
}

function fmtPct(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}
function fmtDeviation(pp) {
  if (pp == null || Number.isNaN(pp)) return '—'
  const sign = pp > 0 ? '+' : ''
  return `${sign}${pp.toFixed(1)}pp`
}

function CalibrationTable({ rows, emptyLabel }) {
  if (!rows || rows.length === 0) {
    return (
      <div style={{
        padding: '32px 18px', textAlign: 'center', color: T.textLight, fontSize: 13,
        border: `1px solid ${T.border}`, borderRadius: 6,
      }}>{emptyLabel}</div>
    )
  }
  const maxN = Math.max(...rows.map(r => r.n || 0), 1)
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
      <thead>
        <tr style={{ borderBottom: `1px solid ${T.border}` }}>
          {['Line', 'n', 'Predicted', 'Observed', 'Deviation', 'Volume'].map((h, i) => (
            <th key={h} style={{
              padding: '10px 8px', fontSize: 11, fontWeight: 500,
              color: T.textMedium, letterSpacing: 0.4,
              textAlign: i === 0 ? 'left' : (i === 5 ? 'left' : 'right'),
              width: i === 5 ? '30%' : undefined,
            }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(r => {
          const dev = r.deviation_pp
          const driftBad = dev != null && Math.abs(dev) > 2.0
          return (
            <tr key={r.line} style={{ borderBottom: `1px solid ${T.border}` }}>
              <td style={{
                padding: '14px 8px 14px 0', color: T.text, fontSize: 12, ...TABULAR,
              }}>{Number(r.line).toFixed(1)}</td>
              <td style={{ padding: '14px 8px', textAlign: 'right', fontSize: 12, color: T.textMedium, ...TABULAR }}>
                {r.n || '—'}
              </td>
              <td style={{ padding: '14px 8px', textAlign: 'right', fontSize: 12, color: T.textMedium, ...TABULAR }}>
                {fmtPct(r.predicted)}
              </td>
              <td style={{ padding: '14px 8px', textAlign: 'right', fontSize: 12, color: T.text, fontWeight: 600, ...TABULAR }}>
                {fmtPct(r.observed)}
              </td>
              <td style={{
                padding: '14px 8px', textAlign: 'right', fontSize: 12,
                color: driftBad ? T.negative : T.textMedium,
                fontWeight: driftBad ? 600 : 400, ...TABULAR,
              }}>{fmtDeviation(dev)}</td>
              <td style={{ padding: '14px 0 14px 8px' }}>
                {r.n > 0 && (
                  <div style={{
                    height: 6, width: `${(r.n / maxN) * 100}%`,
                    background: T.bgSubtle, borderRadius: 2,
                    borderRight: `2px solid ${T.borderStrong}`,
                  }} />
                )}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function liveCalibrationRows(liveCal) {
  if (!liveCal?.by_line) return []
  return Object.entries(liveCal.by_line)
    .map(([line, e]) => ({
      line: parseFloat(line),
      n: e.n_picks_over || 0,
      predicted: e.predicted_p_over_mean,
      observed: e.observed_p_over_mean,
      deviation_pp: e.deviation_pp,
    }))
    .sort((a, b) => a.line - b.line)
}

function backtestCalibrationRows(backtest) {
  if (!backtest?.per_line) return []
  return Object.entries(backtest.per_line)
    .map(([line, e]) => ({
      line: parseFloat(line),
      n: e.n,
      predicted: e.predicted_p_geq_line,
      observed: e.observed_p_geq_line,
      deviation_pp: e.predicted_p_geq_line != null && e.observed_p_geq_line != null
        ? (e.observed_p_geq_line - e.predicted_p_geq_line) * 100
        : null,
    }))
    .sort((a, b) => a.line - b.line)
}

export default function Calibration({ liveCal, backtest, nbCal, calibrationNote }) {
  const liveRows = liveCalibrationRows(liveCal)
  const backtestRows = backtestCalibrationRows(backtest)
  const nTracked = liveCal?.n_picks_tracked || 0

  return (
    <>
      <Head>
        <title>Strikeouts — Calibration</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <style jsx global>{`
        html, body { margin: 0; padding: 0; background: ${T.bg}; }
        a:hover { text-decoration: underline; }
      `}</style>

      <div style={{
        minHeight: '100vh', background: T.bg, color: T.text,
        fontFamily: FONT, padding: '40px 24px',
        maxWidth: 1080, margin: '0 auto',
      }}>
        <div style={{ marginBottom: 36 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 24, flexWrap: 'wrap' }}>
            <Link href="/" style={{
              fontSize: 22, fontWeight: 700, color: T.text,
              letterSpacing: -0.4, textDecoration: 'none',
            }}>Strikeouts</Link>
            <span style={{ fontSize: 14, color: T.text, fontWeight: 500 }}>Calibration</span>
            <Link href="/strikeouts-tracker" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Tracker</Link>
            <Link href="/personal-bets" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Personal bets</Link>
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginTop: 8 }}>
            Per-line predicted vs observed P(K ≥ line). Live tracker accumulates
            from Over picks resolved during paper trade.
          </div>
        </div>

        {calibrationNote && (
          <div style={{
            border: `1px solid ${T.border}`, borderRadius: 6,
            padding: '14px 18px', marginBottom: 28,
            fontSize: 12, color: T.textMedium, lineHeight: 1.6, background: T.bgSubtle,
          }}>
            <strong style={{ color: T.text, fontWeight: 600 }}>Note from the model.</strong>{' '}
            {calibrationNote}
          </div>
        )}

        <div style={{ marginBottom: 36 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: T.text, marginBottom: 4 }}>
            Live calibration
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginBottom: 18 }}>
            {nTracked} Over picks tracked since paper-trade start · deviation = observed − predicted (percentage points)
          </div>
          <CalibrationTable rows={liveRows} emptyLabel="No live picks resolved yet." />
        </div>

        <div style={{ marginBottom: 36 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: T.text, marginBottom: 4 }}>
            Phase 6 backtest (2024–2025)
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginBottom: 18 }}>
            Held-out predictions from the pre-paper-trade backtest. Drives the
            calibration note shown above — the model over-predicts at high
            lines and we publish raw probabilities without auto-correction.
          </div>
          <CalibrationTable rows={backtestRows}
            emptyLabel="Backtest report not present." />
        </div>

        {nbCal && (
          <div style={{
            border: `1px solid ${T.border}`, borderRadius: 6,
            padding: '20px 26px', marginBottom: 36,
          }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: T.text, marginBottom: 6 }}>
              NB dispersion fit
            </div>
            <div style={{ fontSize: 12, color: T.textLight, marginBottom: 14 }}>
              Phase 4d: fit α on per-game K outcomes. α ≈ 0 means effectively Poisson.
            </div>
            <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', fontSize: 13, ...TABULAR }}>
              <div>
                <div style={{ color: T.textMedium, fontSize: 11 }}>α (fit)</div>
                <div style={{ color: T.text, fontWeight: 600, fontSize: 18, marginTop: 4 }}>
                  {nbCal.alpha_fit != null ? nbCal.alpha_fit.toFixed(4) : '—'}
                </div>
              </div>
              <div>
                <div style={{ color: T.textMedium, fontSize: 11 }}>n training games</div>
                <div style={{ color: T.text, fontWeight: 600, fontSize: 18, marginTop: 4 }}>
                  {nbCal.n_train ?? '—'}
                </div>
              </div>
              <div>
                <div style={{ color: T.textMedium, fontSize: 11 }}>MAE</div>
                <div style={{ color: T.text, fontWeight: 600, fontSize: 18, marginTop: 4 }}>
                  {nbCal.mae != null ? nbCal.mae.toFixed(2) : '—'}
                </div>
              </div>
            </div>
          </div>
        )}

        <div style={{
          marginTop: 40, paddingTop: 20, borderTop: `1px solid ${T.border}`,
          fontSize: 11, color: T.textLight, lineHeight: 1.7,
        }}>
          Deviation flagged red when |observed − predicted| &gt; 2pp. Live
          calibration only counts Over picks (the side anchored to P(K ≥ line));
          Under picks contribute to ROI and CLV elsewhere.
        </div>
      </div>
    </>
  )
}
