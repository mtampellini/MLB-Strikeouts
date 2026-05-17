export default function Home() {
  return (
    <main style={{ maxWidth: 720, margin: '40px auto', padding: '0 16px' }}>
      <h1>MLB Strikeouts</h1>
      <p style={{ color: 'var(--muted)' }}>
        Pitcher strikeout prop model. Three-tier picks (primary / secondary / shadow)
        ship once Phase 5 lands.
      </p>
      <ul>
        <li><a href="/strikeouts-tracker/">Tracker</a></li>
        <li><a href="/calibration/">Calibration</a></li>
        <li><a href="/personal-bets/">Personal bets</a></li>
      </ul>
      <p style={{ color: 'var(--muted)', fontSize: 12, marginTop: 40 }}>
        Phase 1 scaffold. See backend/README.md for build status.
      </p>
    </main>
  )
}
