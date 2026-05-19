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

const PERSONAL_FILE = path.join(process.cwd(), 'backend/data/personal_bets.json')

function readJSONSafe(file) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')) }
  catch { return null }
}

export async function getStaticProps() {
  const blob = readJSONSafe(PERSONAL_FILE)
  return { props: { bets: blob?.bets || [], updatedAt: blob?.updated_at || null } }
}

function fmtOdds(o) {
  if (o == null) return '—'
  return o > 0 ? `+${o}` : `${o}`
}
function fmtStake(v) {
  if (v == null) return '—'
  return `$${v.toFixed(2)}`
}
function fmtPL(v) {
  if (v == null) return '—'
  const sign = v > 0 ? '+' : v < 0 ? '−' : ''
  return `${sign}$${Math.abs(v).toFixed(2)}`
}

function summarize(bets) {
  let wins = 0, losses = 0, voids = 0, staked = 0, profit = 0
  for (const b of bets) {
    const r = (b.result || '').toLowerCase()
    if (r === 'win') wins++
    else if (r === 'loss') losses++
    else if (r === 'push' || r === 'void' || r === 'pitcher_did_not_start') voids++
    staked += b.stake || 0
    if (b.profit_loss != null) profit += b.profit_loss
  }
  const settled = wins + losses
  return {
    n: bets.length, wins, losses, voids, staked, profit,
    hit_rate: settled > 0 ? (wins / settled) * 100 : null,
    roi_pct: staked > 0 ? (profit / staked) * 100 : null,
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

function BetRow({ bet }) {
  const r = (bet.result || '').toLowerCase()
  const result = r === 'win' ? <span style={{ color: T.positive, fontWeight: 600 }}>W</span>
              : r === 'loss' ? <span style={{ color: T.negative, fontWeight: 600 }}>L</span>
              : r === 'push' || r === 'void' ? <span style={{ color: T.textLight }}>{r.toUpperCase()}</span>
              : r === 'pitcher_did_not_start' ? <span style={{ color: T.textLight }}>DNS</span>
              : <span style={{ color: T.textLight }}>pending</span>
  const pl = bet.profit_loss
  return (
    <tr style={{ borderBottom: `1px solid ${T.border}` }}>
      <td style={{
        padding: '14px 8px', verticalAlign: 'top', whiteSpace: 'nowrap',
        fontSize: 12, color: T.textMedium, ...TABULAR,
      }}>{bet.game_date || '—'}</td>

      <td style={{ padding: '14px 8px', verticalAlign: 'top', whiteSpace: 'nowrap' }}>
        <div style={{ fontWeight: 600, color: T.text, fontSize: 13 }}>{bet.pitcher_name}</div>
        {bet.book && (
          <div style={{ fontSize: 11, color: T.textLight, marginTop: 4 }}>
            {bet.book === 'draftkings' ? 'DK' : bet.book === 'fanduel' ? 'FD' : bet.book}
          </div>
        )}
      </td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, color: T.text, fontWeight: 500, ...TABULAR,
      }}>{bet.side} {bet.line?.toFixed(1)}</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, color: T.text, fontWeight: 600, ...TABULAR,
      }}>{fmtOdds(bet.american_odds)}</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, color: T.textMedium, ...TABULAR,
      }}>{fmtStake(bet.stake)}</td>

      <td style={{ padding: '14px 8px', verticalAlign: 'top', fontSize: 12 }}>{result}</td>

      <td style={{
        padding: '14px 8px', textAlign: 'right', verticalAlign: 'top',
        fontSize: 12, fontWeight: 600, ...TABULAR,
        color: pl == null ? T.textLight : pl > 0 ? T.positive : pl < 0 ? T.negative : T.textMedium,
      }}>{fmtPL(pl)}</td>

      <td style={{
        padding: '14px 8px', verticalAlign: 'top',
        fontSize: 11, color: T.textLight, lineHeight: 1.6,
      }}>{bet.note || ''}</td>
    </tr>
  )
}

export default function PersonalBets({ bets, updatedAt }) {
  const sum = summarize(bets)
  let updatedLocal = ''
  if (updatedAt) {
    try { updatedLocal = new Date(updatedAt).toLocaleString() } catch { /* swallow */ }
  }

  return (
    <>
      <Head>
        <title>Strikeouts — Personal bets</title>
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
            <span style={{ fontSize: 14, color: T.text, fontWeight: 500 }}>Personal bets</span>
            <Link href="/strikeouts-tracker" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Tracker</Link>
            <Link href="/calibration" style={{ fontSize: 14, color: T.textMedium, textDecoration: 'none' }}>Calibration</Link>
          </div>
          <div style={{ fontSize: 12, color: T.textLight, marginTop: 8 }}>
            Bets I actually placed (vs. the model's paper picks). Edit{' '}
            <code style={{ background: T.bgSubtle, padding: '0 4px', borderRadius: 3 }}>
              backend/data/personal_bets.json
            </code>{' '}to add or settle entries.
            {updatedLocal && ` · Updated ${updatedLocal}`}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 14, marginBottom: 28, flexWrap: 'wrap' }}>
          <StatCard
            label="Bets logged"
            value={sum.n}
            sub={`${sum.wins}W–${sum.losses}L · ${sum.voids} void`}
            tone={sum.n > 0 ? 'default' : 'muted'}
          />
          <StatCard
            label="Hit rate"
            value={sum.hit_rate != null ? `${sum.hit_rate.toFixed(1)}%` : '—'}
            tone={sum.hit_rate != null ? 'default' : 'muted'}
          />
          <StatCard
            label="ROI"
            value={sum.roi_pct != null ? `${sum.roi_pct >= 0 ? '+' : ''}${sum.roi_pct.toFixed(1)}%` : '—'}
            sub={sum.staked > 0 ? `${fmtPL(sum.profit)} on ${fmtStake(sum.staked)}` : null}
            tone={sum.roi_pct != null ? (sum.roi_pct >= 0 ? 'positive' : 'negative') : 'muted'}
          />
          <StatCard
            label="Net"
            value={sum.staked > 0 ? fmtPL(sum.profit) : '—'}
            tone={sum.staked > 0 ? (sum.profit >= 0 ? 'positive' : 'negative') : 'muted'}
          />
        </div>

        {bets.length === 0 ? (
          <div style={{
            padding: '48px 18px', textAlign: 'center', color: T.textLight, fontSize: 14,
            border: `1px solid ${T.border}`, borderRadius: 6,
          }}>
            No personal bets logged yet. Add entries to{' '}
            <code style={{ background: T.bgSubtle, padding: '0 4px', borderRadius: 3 }}>
              backend/data/personal_bets.json
            </code>{' '}with shape:{' '}
            <code style={{ fontSize: 11 }}>
              {`{"bets": [{"game_date":"YYYY-MM-DD","pitcher_name":"...","side":"Over","line":5.5,"american_odds":-110,"stake":1,"book":"fanduel","result":"win|loss|push","profit_loss":0.91,"note":""}]}`}
            </code>
          </div>
        ) : (
          <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, minWidth: 880 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${T.borderStrong}` }}>
                  {[
                    ['Date', 'left'], ['Pitcher', 'left'], ['Bet', 'right'],
                    ['Odds', 'right'], ['Stake', 'right'], ['Result', 'left'],
                    ['P/L', 'right'], ['Note', 'left'],
                  ].map(([h, align]) => (
                    <th key={h} style={{
                      padding: '12px 8px', fontSize: 11, fontWeight: 500,
                      color: T.textMedium, letterSpacing: 0.4, textAlign: align,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {bets.map((b, i) => <BetRow key={b.id || i} bet={b} />)}
              </tbody>
            </table>
          </div>
        )}

        <div style={{
          marginTop: 40, paddingTop: 20, borderTop: `1px solid ${T.border}`,
          fontSize: 11, color: T.textLight, lineHeight: 1.7,
        }}>
          Personal bets are tracked separately from the model's paper picks so
          discretionary choices don't pollute model calibration.
        </div>
      </div>
    </>
  )
}
