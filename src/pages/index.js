import Head from 'next/head'
import Link from 'next/link'
import fs from 'fs'
import path from 'path'
import { useState } from 'react'

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

const PICKS_DIR = path.join(process.cwd(), 'backend/data/picks')

function readJSONSafe(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) }
  catch { return null }
}

function latestPicksDate() {
  try {
    const entries = fs.readdirSync(PICKS_DIR).filter(d => /^\d{4}-\d{2}-\d{2}$/.test(d))
    entries.sort((a, b) => b.localeCompare(a))
    return entries[0] || null
  } catch { return null }
}

export async function getStaticProps() {
  const date = latestPicksDate()
  if (!date) {
    return { props: { date: null, primary: [], secondary: [], shadow: [],
                       metadata: null, skipped: [] } }
  }
  const dir = path.join(PICKS_DIR, date)
  const primary = readJSONSafe(path.join(dir, 'picks.json'))
  const secondary = readJSONSafe(path.join(dir, 'secondary_picks.json'))
  const shadow = readJSONSafe(path.join(dir, 'shadow_picks.json'))
  const debug = readJSONSafe(path.join(dir, 'all_picks_debug.json'))
  return {
    props: {
      date,
      primary: primary?.picks || [],
      secondary: secondary?.picks || [],
      shadow: shadow?.picks || [],
      metadata: primary?.metadata || secondary?.metadata || shadow?.metadata || null,
      skipped: debug?.skipped || [],
    },
  }
}

function fmtOdds(o) {
  if (o == null) return '—'
  return o > 0 ? `+${o}` : `${o}`
}

function StatCard({ label, value, sub }) {
  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 6,
      padding: '24px 26px', minWidth: 140, flex: '1 1 140px', background: T.bg,
    }}>
      <div style={{
        fontSize: 32, fontWeight: 600, color: T.text,
        letterSpacing: -0.5, lineHeight: 1.1, ...TABULAR,
      }}>{value}</div>
      <div style={{ fontSize: 12, color: T.textMedium, marginTop: 10 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

function PickRow({ pick }) {
  const evPositive = pick.ev_pct >= 0
  const edgePositive = pick.edge_pct >= 0
  const bookLabel = pick.book === 'draftkings' ? 'DK' : 'FD'
  const parkFactor = pick.park_k_factor_by_hand
  const parkPct = parkFactor != null ? (parkFactor - 1) * 100 : null

  const metaParts = []
  if (pick.pitcher_archetype) metaParts.push(pick.pitcher_archetype)
  if (parkPct != null) {
    const sign = parkPct > 0 ? '+' : ''
    metaParts.push(`park ${sign}${parkPct.toFixed(0)}%`)
  }
  if (pick.devig_source && pick.devig_source !== 'two_sided') {
    metaParts.push(`devig: ${pick.devig_source}`)
  }
  if (pick.tier === 'secondary') metaParts.push('secondary')
  if (pick.tier === 'shadow') metaParts.push('shadow')

  return (
    <tr style={{ borderBottom: `1px solid ${T.border}` }}>
      <td style={{ padding: '18px 10px', verticalAlign: 'top', whiteSpace: 'nowrap' }}>
        <div style={{ fontWeight: 600, color: T.text, fontSize: 14 }}>{pick.pitcher_name}</div>
        <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>
          {metaParts.join(' · ')}
        </div>
      </td>

      <td style={{
        padding: '18px 10px', verticalAlign: 'top', whiteSpace: 'nowrap',
        fontSize: 13, color: T.textMedium,
      }}>
        <div>{pick.venue_name || '—'}</div>
        <div style={{ color: T.textLight, fontSize: 11, marginTop: 4 }}>{pick.game_date}</div>
      </td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 13, color: T.text, fontWeight: 500, ...TABULAR,
      }}>
        {pick.side} {pick.line.toFixed(1)}
      </td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 14, fontWeight: 600, color: T.text, ...TABULAR,
      }}>{(pick.model_p * 100).toFixed(1)}%</td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 13, color: T.textMedium, ...TABULAR,
      }}>{(pick.market_p * 100).toFixed(1)}%</td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 13, fontWeight: 600,
        color: edgePositive ? T.positive : T.text, ...TABULAR,
      }}>{edgePositive ? '+' : ''}{(pick.edge_pct * 100).toFixed(1)}pp</td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        fontWeight: 700, fontSize: 14,
        color: evPositive ? T.positive : T.text, ...TABULAR,
      }}>{evPositive ? '+' : ''}{(pick.ev_pct * 100).toFixed(1)}%</td>

      <td style={{
        padding: '18px 10px', textAlign: 'right', verticalAlign: 'top',
        whiteSpace: 'nowrap', ...TABULAR,
      }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: T.text }}>
          {fmtOdds(pick.american_odds)}{' '}
          <span style={{ fontSize: 10, color: T.textLight, fontWeight: 500 }}>{bookLabel}</span>
        </div>
        <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>
          E[K] {pick.model_e_k?.toFixed(2)}
        </div>
      </td>
    </tr>
  )
}

function PicksTable({ picks, emptyLabel }) {
  if (!picks || picks.length === 0) {
    return (
      <div style={{
        padding: '48px 18px', textAlign: 'center',
        color: T.textLight, fontSize: 14,
        border: `1px solid ${T.border}`, borderRadius: 6,
      }}>
        {emptyLabel}
      </div>
    )
  }
  const cols = [
    { label: 'Pitcher', align: 'left' },
    { label: 'Park', align: 'left' },
    { label: 'Bet', align: 'right' },
    { label: 'Model', align: 'right' },
    { label: 'Market', align: 'right' },
    { label: 'Edge', align: 'right' },
    { label: 'EV', align: 'right' },
    { label: 'Odds', align: 'right' },
  ]
  return (
    <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 920 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${T.borderStrong}` }}>
            {cols.map(h => (
              <th key={h.label} style={{
                padding: '10px 10px', textAlign: h.align,
                color: T.textMedium, fontWeight: 500, fontSize: 11, letterSpacing: 0.4,
              }}>{h.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {picks.map(p => <PickRow key={p.pick_id} pick={p} />)}
        </tbody>
      </table>
    </div>
  )
}

function Methodology() {
  const [open, setOpen] = useState(false)
  return (
    <div style={{
      marginBottom: 32, border: `1px solid ${T.border}`,
      borderRadius: 6, overflow: 'hidden',
    }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%', padding: '14px 18px', background: 'none', border: 'none',
          color: T.text, cursor: 'pointer', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between', fontFamily: 'inherit', fontSize: 13, fontWeight: 500,
        }}
      >
        <span>How this works</span>
        <span style={{
          fontSize: 12, color: T.textLight,
          transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s',
        }}>▾</span>
      </button>
      {open && (
        <div style={{ padding: '0 18px 22px', borderTop: `1px solid ${T.border}` }}>
          <div style={{ marginTop: 18, marginBottom: 22, fontSize: 13, color: T.textMedium, lineHeight: 1.7 }}>
            Per-batter P(K|PA) model with log5 matchup priors using a
            CSW%-to-K% blended pitcher rate as the offset (resolves the
            collinearity between pitcher K% and CSW%). Aggregated to E[K] via
            empirical PA distribution by batting order spot and TTO multipliers,
            then priced through NB(α=0.001) — effectively Poisson at this
            dispersion — for P(K ≥ line).
          </div>
          <div style={{ marginBottom: 22 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 10 }}>Columns</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <tbody>
                {[
                  ['Pitcher', 'Starter, with pitcher archetype and handedness-specific park K factor.'],
                  ['Park', 'Venue. Game date below.'],
                  ['Bet', "Side and line we'd take (Over or Under K)."],
                  ['Model', 'P(K ≥ line) under the projector + NB pricing.'],
                  ['Market', 'De-vigged P(K ≥ line) from FD/DK. One-sided lines impute the missing side via nearest paired line.'],
                  ['Edge', 'Model − market, in percentage points.'],
                  ['EV', 'Expected return per $1 stake at the listed odds. Higher is better.'],
                  ['Odds', 'American price at the book the pick was sourced from. E[K] underneath.'],
                ].map(([term, desc], i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${T.border}` }}>
                    <td style={{
                      padding: '12px 14px 12px 0', fontWeight: 600, color: T.text,
                      fontSize: 12, whiteSpace: 'nowrap', verticalAlign: 'top', width: 90,
                    }}>{term}</td>
                    <td style={{ padding: '12px 0', color: T.textMedium, lineHeight: 1.6 }}>{desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginBottom: 22, fontSize: 12, color: T.textMedium, lineHeight: 1.7 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 8 }}>Tiers</div>
            <strong style={{ color: T.text, fontWeight: 600 }}>Primary</strong>: edge ≥ 20pp, American
            price ≥ −180, top 10 by edge.{' '}
            <strong style={{ color: T.text, fontWeight: 600 }}>Secondary</strong>: edge ≥ 20pp but
            outside the primary cap (price &lt; −180 or rank &gt; 10).{' '}
            <strong style={{ color: T.text, fontWeight: 600 }}>Shadow</strong>: 10–20pp edge — tracked
            for calibration but not bet.
          </div>
          <div style={{ fontSize: 12, color: T.textMedium, lineHeight: 1.7 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: T.text, marginBottom: 8 }}>Data hygiene</div>
            No median-fill, no synthetic odds. Pitchers without sufficient
            track record are skipped and logged, not imputed. AsOfContext
            strictly enforced — every feature is computed from data available
            before first pitch.
          </div>
        </div>
      )}
    </div>
  )
}

function CalibrationBanner({ note }) {
  if (!note) return null
  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 6,
      padding: '14px 18px', marginBottom: 28,
      fontSize: 12, color: T.textMedium, lineHeight: 1.6,
      background: T.bgSubtle,
    }}>
      <strong style={{ color: T.text, fontWeight: 600 }}>Calibration note.</strong>{' '}
      {note}
    </div>
  )
}

function LineupsPendingBanner({ pending }) {
  if (!pending || pending.length === 0) return null
  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 6,
      padding: '14px 18px', marginBottom: 28,
      fontSize: 12, color: T.textMedium, lineHeight: 1.6,
      background: T.bgSubtle,
    }}>
      <strong style={{ color: T.text, fontWeight: 600 }}>
        Lineups Pending ({pending.length}).
      </strong>{' '}
      We only generate picks for games where the opposing lineup has been
      officially posted. The hourly run picks these up automatically — more
      picks may land as lineups post:{' '}
      <span style={{ color: T.textLight }}>
        {pending.map(p => p.pitcher_name).filter(Boolean).join(', ')}
      </span>
    </div>
  )
}

export default function Home({ date, primary, secondary, shadow, metadata, skipped }) {
  const [tab, setTab] = useState('primary')
  const picks = tab === 'primary' ? primary : tab === 'secondary' ? secondary : shadow

  const generatedAt = metadata?.generated_at
  const nbAlpha = metadata?.nb_alpha
  let generatedLocal = ''
  if (generatedAt) {
    try { generatedLocal = new Date(generatedAt).toLocaleString() } catch { /* swallow */ }
  }

  const calibrationNote = picks[0]?.calibration_note || primary[0]?.calibration_note
                        || secondary[0]?.calibration_note || shadow[0]?.calibration_note

  const lineupsPending = skipped.filter(s => s.is_transient && s.reason === 'lineup_not_posted')
  const permanentSkipped = skipped.filter(s => !s.is_transient)

  return (
    <>
      <Head>
        <title>Strikeouts Picks</title>
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
            <span style={{ fontSize: 22, fontWeight: 700, color: T.text, letterSpacing: -0.4 }}>
              Strikeouts
            </span>
            <Link href="/strikeouts-tracker" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Tracker</Link>
            <Link href="/calibration" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Calibration</Link>
            <Link href="/personal-bets" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Personal bets</Link>
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginTop: 8 }}>
            {date ? `As of ${date}` : 'No picks generated yet'}
            {nbAlpha != null && ` · NB α=${nbAlpha}`}
            {generatedLocal && ` · Generated ${generatedLocal}`}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 14, marginBottom: 28, flexWrap: 'wrap' }}>
          <StatCard label="Primary"   value={primary.length}   sub="edge ≥ 20pp, ≥ −180, top 10" />
          <StatCard label="Secondary" value={secondary.length} sub="edge ≥ 20pp outside cap" />
          <StatCard label="Shadow"    value={shadow.length}    sub="10–20pp edge, tracked" />
          <StatCard
            label="Lineups pending"
            value={lineupsPending.length}
            sub={lineupsPending.length > 0 ? 'transient — re-checked hourly' : 'all lineups posted'}
          />
          <StatCard
            label="Skipped"
            value={permanentSkipped.length}
            sub={permanentSkipped.length > 0 ? 'permanent for this slate' : 'no permanent skips'}
          />
        </div>

        <LineupsPendingBanner pending={lineupsPending} />

        <CalibrationBanner note={calibrationNote} />

        <Methodology />

        <div style={{
          display: 'flex', gap: 24, marginBottom: 18,
          borderBottom: `1px solid ${T.border}`, paddingBottom: 12,
        }}>
          {[
            ['primary',   `Primary (${primary.length})`],
            ['secondary', `Secondary (${secondary.length})`],
            ['shadow',    `Shadow (${shadow.length})`],
          ].map(([k, lbl]) => (
            <button key={k} onClick={() => setTab(k)} style={{
              background: 'transparent', border: 'none', padding: '4px 0',
              color: tab === k ? T.text : T.textLight,
              fontWeight: tab === k ? 700 : 400, fontSize: 13,
              cursor: 'pointer', fontFamily: 'inherit',
              textDecoration: tab === k ? 'underline' : 'none',
              textUnderlineOffset: 6, textDecorationThickness: 2,
            }}>{lbl}</button>
          ))}
        </div>

        <PicksTable picks={picks} emptyLabel={
          tab === 'primary' ? 'No primary picks for this slate.'
          : tab === 'secondary' ? 'No secondary picks for this slate.'
          : 'No shadow picks for this slate.'
        } />

        <div style={{
          marginTop: 48, paddingTop: 20, borderTop: `1px solid ${T.border}`,
          fontSize: 11, color: T.textLight, lineHeight: 1.7,
        }}>
          Odds: FanDuel + DraftKings via The Odds API. De-vigged when both
          books quote both sides; nearest-paired-line imputation otherwise.
          Paper-trade gate: ≥ 60 days. Always verify the current price on the
          listed book before placing a bet — lines move.
        </div>
      </div>
    </>
  )
}
