import Head from 'next/head'
import Link from 'next/link'
import fs from 'fs'
import path from 'path'
import { useMemo, useState } from 'react'

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
const LEDGER_FILE = path.join(process.cwd(), 'backend/data/performance_ledger.json')

function readJSONSafe(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) }
  catch { return null }
}

export async function getStaticProps() {
  let archives = []
  try {
    const entries = fs.readdirSync(PICKS_DIR).filter(d => /^\d{4}-\d{2}-\d{2}$/.test(d))
    archives = entries.map(date => {
      const dir = path.join(PICKS_DIR, date)
      const debug = readJSONSafe(path.join(dir, 'all_picks_debug.json'))
      const settled = readJSONSafe(path.join(dir, 'settled_picks.json'))
      if (!debug && !settled) return null
      return {
        date,
        metadata: debug?.metadata || null,
        primary: debug?.primary || [],
        secondary: debug?.secondary || [],
        shadow: debug?.shadow || [],
        settled: settled?.picks || [],
        is_settled: settled != null,
      }
    }).filter(Boolean)
    archives.sort((a, b) => b.date.localeCompare(a.date))
  } catch { /* picks dir not present yet */ }

  const ledger = readJSONSafe(LEDGER_FILE)
  return { props: { archives, ledger } }
}

function fmtPct(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return '—'
  return `${v.toFixed(digits)}%`
}
function fmtUnits(v) {
  if (v == null || Number.isNaN(v)) return '—'
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(2)}u`
}
function fmtSigned(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return '—'
  const sign = v > 0 ? '+' : ''
  return `${sign}${v.toFixed(digits)}`
}
function fmtOdds(o) {
  if (o == null) return '—'
  return o > 0 ? `+${o}` : `${o}`
}

function settledIndex(settled) {
  const m = {}
  for (const p of settled || []) m[p.pick_id] = p
  return m
}

function summarizeArchive(arc, tierFilter) {
  const tiers = tierFilter === 'all'
    ? ['primary', 'secondary', 'shadow']
    : [tierFilter]
  const picks = tiers.flatMap(t => arc[t] || [])
  const settledById = settledIndex(arc.settled)

  let n_wins = 0, n_losses = 0, n_voids = 0, profit = 0
  for (const p of picks) {
    const s = settledById[p.pick_id]
    if (!s) continue
    if (s.result === 'win') { n_wins++; profit += s.profit_loss_first_seen ?? 0 }
    else if (s.result === 'loss') { n_losses++; profit += s.profit_loss_first_seen ?? 0 }
    else { n_voids++ }
  }
  const settled = n_wins + n_losses
  const hit_rate = settled > 0 ? (n_wins / settled) * 100 : null
  const roi_pct = settled > 0 ? (profit / settled) * 100 : null
  return {
    count: picks.length,
    primary_count: arc.primary?.length || 0,
    secondary_count: arc.secondary?.length || 0,
    shadow_count: arc.shadow?.length || 0,
    n_wins, n_losses, n_voids, profit, hit_rate, roi_pct,
    is_settled: arc.is_settled && picks.some(p => settledById[p.pick_id]),
  }
}

function computeTopSummary(archives, tierFilter, dateFilter) {
  let arcs = archives
  if (dateFilter !== 'all') {
    const today = new Date(); today.setHours(0, 0, 0, 0)
    let cutoff = null
    if (dateFilter === '7d') { cutoff = new Date(today); cutoff.setDate(cutoff.getDate() - 7) }
    else if (dateFilter === '30d') { cutoff = new Date(today); cutoff.setDate(cutoff.getDate() - 30) }
    else if (dateFilter === 'yesterday') {
      const y = new Date(today); y.setDate(y.getDate() - 1)
      const ys = y.toISOString().slice(0, 10)
      arcs = arcs.filter(a => a.date === ys)
    }
    if (cutoff) arcs = arcs.filter(a => new Date(a.date) >= cutoff)
  }
  let wins = 0, losses = 0, voids = 0, profit = 0, clvSum = 0, clvCount = 0
  const tierKeys = tierFilter === 'all'
    ? ['primary', 'secondary', 'shadow']
    : [tierFilter]
  for (const arc of arcs) {
    const by = settledIndex(arc.settled)
    for (const t of tierKeys) {
      for (const p of arc[t] || []) {
        const s = by[p.pick_id]
        if (!s) continue
        if (s.result === 'win') { wins++; profit += s.profit_loss_first_seen ?? 0 }
        else if (s.result === 'loss') { losses++; profit += s.profit_loss_first_seen ?? 0 }
        else { voids++ }
        if (p.clv_pct != null) { clvSum += p.clv_pct; clvCount++ }
      }
    }
  }
  const settled = wins + losses
  return {
    wins, losses, voids, settled, profit,
    hit_rate: settled > 0 ? (wins / settled) * 100 : null,
    roi_pct: settled > 0 ? (profit / settled) * 100 : null,
    avg_clv_pct: clvCount > 0 ? (clvSum / clvCount) * 100 : null,
    n_picks_with_clv: clvCount,
  }
}

function StatCard({ label, value, sub, tone = 'default' }) {
  const valueColor =
    tone === 'positive' ? T.positive
    : tone === 'negative' ? T.negative
    : tone === 'muted' ? T.textLight
    : T.text
  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 6,
      padding: '24px 26px', minWidth: 140, flex: '1 1 140px', background: T.bg,
    }}>
      <div style={{
        fontSize: 32, fontWeight: 600, color: valueColor,
        letterSpacing: -0.5, lineHeight: 1.1, ...TABULAR,
      }}>{value}</div>
      <div style={{ fontSize: 12, color: T.textMedium, marginTop: 10 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

function PickRow({ pick, settled }) {
  const result = settled?.result
  const evPositive = pick.ev_pct >= 0
  const edgePositive = pick.edge_pct >= 0
  const bookLabel = pick.book === 'draftkings' ? 'DK' : 'FD'

  const metaParts = []
  if (pick.pitcher_archetype) metaParts.push(pick.pitcher_archetype)
  if (pick.tier) metaParts.push(pick.tier)

  const renderResult = () => {
    if (result === 'win') return (
      <span style={{ color: T.positive, fontWeight: 600 }}>
        W{' '}<span style={{ color: T.textLight, fontWeight: 500, marginLeft: 4 }}>
          {fmtSigned(settled.profit_loss_first_seen, 2)}u
        </span>
      </span>
    )
    if (result === 'loss') return <span style={{ color: T.negative, fontWeight: 600 }}>L −1u</span>
    if (result === 'push') return <span style={{ color: T.textLight, fontWeight: 500 }}>PUSH</span>
    if (result === 'pitcher_did_not_start')
      return <span style={{ color: T.textLight, fontWeight: 500 }}>DNS</span>
    return <span style={{ color: T.textLight }}>pending</span>
  }
  const observedK = settled?.observed_k
  return (
    <tr style={{ borderBottom: `1px solid ${T.border}` }}>
      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 11, color: T.textLight, ...TABULAR,
      }}>{pick.rank_in_tier || '—'}</td>

      <td style={{ padding: '14px 8px', verticalAlign: 'top', whiteSpace: 'nowrap' }}>
        <div style={{ fontWeight: 600, color: T.text, fontSize: 13 }}>{pick.pitcher_name}</div>
        <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>{metaParts.join(' · ')}</div>
      </td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, color: T.text, fontWeight: 500, ...TABULAR,
      }}>{pick.side} {pick.line.toFixed(1)}</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 13, fontWeight: 600, color: T.text, ...TABULAR,
      }}>{(pick.model_p * 100).toFixed(1)}%</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, color: T.textMedium, ...TABULAR,
      }}>{(pick.market_p * 100).toFixed(1)}%</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, fontWeight: 600,
        color: edgePositive ? T.positive : T.text, ...TABULAR,
      }}>{edgePositive ? '+' : ''}{(pick.edge_pct * 100).toFixed(1)}pp</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 13, fontWeight: 700,
        color: evPositive ? T.positive : T.text, ...TABULAR,
      }}>{evPositive ? '+' : ''}{(pick.ev_pct * 100).toFixed(0)}%</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top', whiteSpace: 'nowrap',
        ...TABULAR,
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.text }}>
          {fmtOdds(pick.american_odds)}{' '}
          <span style={{ fontSize: 10, color: T.textLight, fontWeight: 500 }}>{bookLabel}</span>
        </div>
        {pick.closing_american_odds != null && pick.closing_american_odds !== pick.american_odds && (
          <div style={{ fontSize: 10, color: T.textLight, marginTop: 4 }}>
            close {fmtOdds(pick.closing_american_odds)}
          </div>
        )}
      </td>

      <td style={{
        padding: '14px 8px', verticalAlign: 'top', textAlign: 'left', fontSize: 12,
      }}>
        {renderResult()}
        {observedK != null && (
          <div style={{ fontSize: 10, color: T.textLight, marginTop: 4 }}>K: {observedK}</div>
        )}
      </td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 11, ...TABULAR,
        color: pick.clv_pct == null ? T.textLight
              : pick.clv_pct > 0 ? T.positive
              : pick.clv_pct < 0 ? T.negative
              : T.textMedium,
      }}>
        {pick.clv_pct == null ? '—' : `${pick.clv_pct >= 0 ? '+' : ''}${(pick.clv_pct * 100).toFixed(1)}%`}
      </td>
    </tr>
  )
}

function DayBlock({ arc, expanded, onToggle, tierFilter }) {
  const summary = summarizeArchive(arc, tierFilter)
  const settledById = useMemo(() => settledIndex(arc.settled), [arc])
  const picks = useMemo(() => {
    if (tierFilter === 'all') return [...(arc.primary || []), ...(arc.secondary || []), ...(arc.shadow || [])]
    return arc[tierFilter] || []
  }, [arc, tierFilter])

  const profitTone =
    !summary.is_settled ? T.textLight
    : summary.profit > 0 ? T.positive
    : summary.profit < 0 ? T.negative
    : T.textMedium

  return (
    <div style={{ borderTop: `1px solid ${T.border}` }}>
      <button onClick={onToggle} style={{
        width: '100%', padding: '18px 4px', background: 'none', border: 'none',
        color: T.text, cursor: 'pointer', fontFamily: 'inherit',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        flexWrap: 'wrap', gap: 12, textAlign: 'left',
      }}>
        <div style={{ display: 'flex', gap: 18, alignItems: 'baseline', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: T.text }}>{arc.date}</span>
          <span style={{ fontSize: 12, color: T.textLight }}>
            {summary.primary_count} primary
            {summary.secondary_count > 0 && ` · ${summary.secondary_count} secondary`}
            {summary.shadow_count > 0 && ` · ${summary.shadow_count} shadow`}
          </span>
          {summary.is_settled ? (
            <>
              <span style={{ fontSize: 12, color: T.textMedium, ...TABULAR }}>
                {summary.n_wins}W–{summary.n_losses}L{summary.n_voids > 0 && `–${summary.n_voids}V`}
              </span>
              <span style={{ fontSize: 12, color: profitTone, fontWeight: 600, ...TABULAR }}>
                {fmtUnits(summary.profit)}
                {summary.roi_pct != null && ` (${summary.roi_pct >= 0 ? '+' : ''}${summary.roi_pct.toFixed(1)}%)`}
              </span>
            </>
          ) : (
            <span style={{ fontSize: 12, color: T.textLight }}>unsettled</span>
          )}
        </div>
        <span style={{
          fontSize: 12, color: T.textLight,
          transform: expanded ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s',
        }}>▾</span>
      </button>
      {expanded && (
        <div style={{ padding: '0 0 24px', overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 980 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${T.borderStrong}` }}>
                {[
                  ['#', 'right'], ['Pitcher', 'left'], ['Bet', 'right'],
                  ['Model', 'right'], ['Market', 'right'], ['Edge', 'right'],
                  ['EV', 'right'], ['Odds', 'right'],
                  ['Result', 'left'], ['CLV', 'right'],
                ].map(([h, align]) => (
                  <th key={h} style={{
                    padding: '12px 8px', fontSize: 11, fontWeight: 500,
                    color: T.textMedium, letterSpacing: 0.4, textAlign: align,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {picks.map(p => (
                <PickRow key={p.pick_id} pick={p} settled={settledById[p.pick_id]} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function FilterButton({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      background: 'transparent', border: 'none', padding: '4px 0',
      color: active ? T.text : T.textLight,
      fontWeight: active ? 700 : 400,
      fontSize: 13, cursor: 'pointer', fontFamily: 'inherit',
      textDecoration: active ? 'underline' : 'none',
      textUnderlineOffset: 6, textDecorationThickness: 2,
    }}>{children}</button>
  )
}
function FilterRow({ label, value, options, onChange }) {
  return (
    <div style={{ display: 'flex', gap: 18, alignItems: 'center', flexWrap: 'wrap' }}>
      <span style={{
        fontSize: 11, color: T.textLight, minWidth: 50,
        textTransform: 'uppercase', letterSpacing: 0.6,
      }}>{label}</span>
      {options.map(([k, lbl]) => (
        <FilterButton key={k} active={value === k} onClick={() => onChange(k)}>{lbl}</FilterButton>
      ))}
    </div>
  )
}

export default function Tracker({ archives, ledger }) {
  const [tierFilter, setTierFilter] = useState('primary')
  const [dateFilter, setDateFilter] = useState('all')
  const [expanded, setExpanded] = useState(() => {
    const newest = archives[0]?.date
    return newest ? { [newest]: true } : {}
  })

  const sum = useMemo(
    () => computeTopSummary(archives, tierFilter, dateFilter),
    [archives, tierFilter, dateFilter],
  )

  const filteredArchives = useMemo(() => {
    let list = archives
    if (dateFilter !== 'all') {
      const today = new Date(); today.setHours(0, 0, 0, 0)
      if (dateFilter === 'yesterday') {
        const y = new Date(today); y.setDate(y.getDate() - 1)
        const ys = y.toISOString().slice(0, 10)
        list = list.filter(a => a.date === ys)
      } else if (dateFilter === '7d') {
        const cutoff = new Date(today); cutoff.setDate(cutoff.getDate() - 7)
        list = list.filter(a => new Date(a.date) >= cutoff)
      } else if (dateFilter === '30d') {
        const cutoff = new Date(today); cutoff.setDate(cutoff.getDate() - 30)
        list = list.filter(a => new Date(a.date) >= cutoff)
      }
    }
    return list
  }, [archives, dateFilter])

  const settled = sum.settled
  const daysSinceDeploy = archives.length === 0 ? 0
    : Math.max(0, Math.floor(
        (new Date() - new Date(archives[archives.length - 1].date)) / (1000 * 60 * 60 * 24)
      ) + 1)

  const calibrationNote = archives[0]?.primary?.[0]?.calibration_note

  return (
    <>
      <Head>
        <title>Strikeouts — Tracker</title>
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
            <span style={{ fontSize: 14, color: T.text, fontWeight: 500 }}>Tracker</span>
            <Link href="/calibration" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Calibration</Link>
            <Link href="/personal-bets" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Personal bets</Link>
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginTop: 8 }}>
            {archives.length} archived day{archives.length === 1 ? '' : 's'} · day {daysSinceDeploy} since deploy
            {ledger?.total_picks_resolved != null && ` · ${ledger.total_picks_resolved} picks resolved`}
          </div>
        </div>

        {settled === 0 && (
          <div style={{
            border: `1px solid ${T.border}`, borderRadius: 6,
            padding: '16px 20px', marginBottom: 28,
            fontSize: 13, color: T.textMedium, lineHeight: 1.5,
          }}>
            <strong style={{ color: T.text, fontWeight: 600 }}>Day {daysSinceDeploy} — no settled picks yet.</strong>{' '}
            Today's picks settle the morning after their game. ROI, hit rate, and CLV populate after the first settlement run.
          </div>
        )}

        {calibrationNote && (
          <div style={{
            border: `1px solid ${T.border}`, borderRadius: 6,
            padding: '14px 18px', marginBottom: 28,
            fontSize: 12, color: T.textMedium, lineHeight: 1.6, background: T.bgSubtle,
          }}>
            <strong style={{ color: T.text, fontWeight: 600 }}>Calibration note.</strong>{' '}
            {calibrationNote}
          </div>
        )}

        <div style={{ display: 'flex', gap: 14, marginBottom: 28, flexWrap: 'wrap' }}>
          <StatCard
            label="Picks settled"
            value={settled}
            sub={`${sum.wins}W–${sum.losses}L · ${sum.voids} void`}
            tone={settled > 0 ? 'default' : 'muted'}
          />
          <StatCard
            label="Hit rate"
            value={settled > 0 ? fmtPct(sum.hit_rate) : '—'}
            tone={settled > 0 ? 'default' : 'muted'}
          />
          <StatCard
            label="ROI"
            value={settled > 0 ? `${sum.roi_pct >= 0 ? '+' : ''}${sum.roi_pct.toFixed(1)}%` : '—'}
            sub={settled > 0 ? `${fmtUnits(sum.profit)} on ${settled}u` : null}
            tone={settled > 0 ? (sum.roi_pct >= 0 ? 'positive' : 'negative') : 'muted'}
          />
          <StatCard
            label="Net @ $1/bet"
            value={settled > 0 ? `${sum.profit >= 0 ? '+' : '−'}$${Math.abs(sum.profit).toFixed(2)}` : '—'}
            sub={settled > 0 ? `on ${settled} bets` : null}
            tone={settled > 0 ? (sum.profit >= 0 ? 'positive' : 'negative') : 'muted'}
          />
          <StatCard
            label="Avg CLV"
            value={sum.avg_clv_pct != null
              ? `${sum.avg_clv_pct >= 0 ? '+' : ''}${sum.avg_clv_pct.toFixed(1)}%`
              : '—'}
            sub={sum.n_picks_with_clv ? `${sum.n_picks_with_clv} picks` : 'awaiting closing snaps'}
            tone={sum.avg_clv_pct != null
              ? (sum.avg_clv_pct >= 0 ? 'positive' : 'negative')
              : 'muted'}
          />
        </div>

        <div style={{
          padding: '20px 0', marginBottom: 8,
          borderTop: `1px solid ${T.border}`, borderBottom: `1px solid ${T.border}`,
          display: 'flex', flexDirection: 'column', gap: 14,
        }}>
          <FilterRow label="Tier" value={tierFilter} onChange={setTierFilter}
            options={[
              ['primary',   'Primary'],
              ['secondary', 'Secondary'],
              ['shadow',    'Shadow'],
              ['all',       'All'],
            ]} />
          <FilterRow label="Range" value={dateFilter} onChange={setDateFilter}
            options={[
              ['all',       'All time'],
              ['30d',       '30 days'],
              ['7d',        '7 days'],
              ['yesterday', 'Yesterday'],
            ]} />
        </div>

        {filteredArchives.length === 0 ? (
          <div style={{
            padding: '48px 18px', textAlign: 'center', color: T.textLight, fontSize: 13,
            border: `1px solid ${T.border}`, borderRadius: 6, marginTop: 28,
          }}>
            No archived days match the current filter.
          </div>
        ) : (
          <div style={{ marginTop: 8, marginBottom: 36 }}>
            {filteredArchives.map(a => (
              <DayBlock key={a.date} arc={a}
                        expanded={!!expanded[a.date]}
                        onToggle={() => setExpanded(p => ({ ...p, [a.date]: !p[a.date] }))}
                        tierFilter={tierFilter} />
            ))}
            <div style={{ borderTop: `1px solid ${T.border}` }} />
          </div>
        )}

        <div style={{
          marginTop: 40, paddingTop: 20, borderTop: `1px solid ${T.border}`,
          fontSize: 11, color: T.textLight, lineHeight: 1.7,
        }}>
          P&amp;L assumes a flat 1u ($1) stake on every pick of the selected
          tier, priced at first-seen odds. CLV is sign-flipped for Under
          picks so positive always means the line moved in our favor.
        </div>
      </div>
    </>
  )
}
