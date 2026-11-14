import React, { useEffect, useState } from 'react'
import {
  dataset,
  franchiseAliases,
  leadingVenues,
  seasons,
  teams,
  tossStats,
  venues,
} from './data.js'

const navigation = [
  ['Overview', 'overview'],
  ['How It Works', 'how-it-works'],
  ['Match Predictor', 'predictor'],
  ['Findings', 'findings'],
  ['Model Performance', 'performance'],
  ['About', 'about'],
]

const steps = [
  { number: '01', glyph: '▦', title: 'Data Preparation', text: 'Inspect raw match records, resolve missing outcomes, and document team and venue names.' },
  { number: '02', glyph: '◇', title: 'Feature Engineering', text: 'Build historical team, venue, form, and toss features from matches before the prediction cutoff.' },
  { number: '03', glyph: '⚙', title: 'Model Training', text: 'Compare baseline, Random Forest, and XGBoost candidates on the same training plan.' },
  { number: '04', glyph: '◎', title: 'Prediction', text: 'Send validated pre-match inputs to the future FastAPI service and show both teams’ probabilities.' },
  { number: '05', glyph: '✓', title: 'Evaluation', text: 'Report held-out metrics, calibration, and limitations for the released model version.' },
]

const emptyForm = {
  team1: '',
  team2: '',
  venue: '',
  season: '',
  toss_winner: '',
  toss_decision: '',
  match_date: '',
}

function Window({ id, title, tone = 'blue', className = '', children }) {
  return (
    <section id={id} className={`window ${className}`} aria-labelledby={`${id}-title`}>
      <div className={`window-titlebar titlebar-${tone}`}>
        <span id={`${id}-title`} className="window-title">{title}</span>
        <span className="window-controls" aria-hidden="true"><i>_</i><i>□</i><i>×</i></span>
      </div>
      <div className="window-body">{children}</div>
    </section>
  )
}

function Badge({ children, variant = 'neutral' }) {
  return <span className={`badge badge-${variant}`}><span className="badge-dot" />{children}</span>
}

function StatCard({ number, label, note, tone }) {
  return (
    <div className={`stat-card stat-${tone}`}>
      <strong>{number}</strong>
      <span>{label}</span>
      <small>{note}</small>
      <div className="stat-blocks" aria-hidden="true"><i /><i /><i /></div>
    </div>
  )
}

function SectionIntro({ eyebrow, title, children }) {
  return (
    <div className="section-intro">
      <div className="eyebrow">{eyebrow}</div>
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  )
}

function getErrors(form) {
  const errors = {}
  if (!form.team1) errors.team1 = 'Choose Team A.'
  if (!form.team2) errors.team2 = 'Choose Team B.'
  if (form.team1 && form.team1 === form.team2) errors.team2 = 'Choose a different franchise.'
  if (!form.venue) errors.venue = 'Choose a venue.'
  if (!form.season) errors.season = 'Choose a season.'
  if (!form.toss_winner) errors.toss_winner = 'Choose the toss winner.'
  if (form.toss_winner && ![form.team1, form.team2].includes(form.toss_winner)) {
    errors.toss_winner = 'The toss winner must be playing this match.'
  }
  if (!form.toss_decision) errors.toss_decision = 'Choose the toss decision.'
  if (form.match_date && form.season && !form.match_date.startsWith(`${form.season}-`)) {
    errors.match_date = `The match date must be in the ${form.season} season, or leave it blank.`
  }
  return errors
}

function SelectField({ id, label, value, onChange, placeholder, options, error, disabled = false }) {
  return (
    <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <select id={id} name={id} value={value} onChange={onChange} disabled={disabled} aria-invalid={Boolean(error)} aria-describedby={error ? `${id}-error` : undefined}>
        <option value="">{placeholder}</option>
        {options.map((option) => {
          const item = typeof option === 'string' ? { value: option, label: option } : option
          return <option key={item.value} value={item.value} disabled={item.disabled}>{item.label}</option>
        })}
      </select>
      {error && <small id={`${id}-error`} className="field-error">{error}</small>}
    </div>
  )
}

function DateField({ id, label, value, onChange, season, error, hint }) {
  const describedBy = [error && `${id}-error`, `${id}-hint`].filter(Boolean).join(' ')
  return (
    <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <input type="date" id={id} name={id} value={value} onChange={onChange} min={season ? `${season}-01-01` : undefined} max={season ? `${season}-12-31` : undefined} aria-invalid={Boolean(error)} aria-describedby={describedBy} />
      {error ? <small id={`${id}-error`} className="field-error">{error}</small> : null}
      <small id={`${id}-hint`} className="field-hint">{hint}</small>
    </div>
  )
}

function HistoricalLine({ label, value, max, color = 'blue' }) {
  return (
    <div className="historical-line">
      <div className="historical-line-top"><span>{label}</span><strong>{value}</strong></div>
      <div className="meter" aria-hidden="true"><span className={`meter-fill meter-${color}`} style={{ width: `${(value / max) * 100}%` }} /></div>
    </div>
  )
}

function Predictor() {
  const [form, setForm] = useState(emptyForm)
  const [attempted, setAttempted] = useState(false)
  const [submitted, setSubmitted] = useState(null)
  const errors = attempted ? getErrors(form) : {}
  const teamOptions = teams.map((team) => team.name)

  function change(event) {
    const { name, value } = event.target
    setForm((previous) => {
      const next = { ...previous, [name]: value }
      if ((name === 'team1' || name === 'team2') && ![next.team1, next.team2].includes(next.toss_winner)) {
        next.toss_winner = ''
      }
      return next
    })
  }

  function submit(event) {
    event.preventDefault()
    setAttempted(true)
    const nextErrors = getErrors(form)
    if (Object.keys(nextErrors).length > 0) {
      const firstField = document.getElementById(Object.keys(nextErrors)[0])
      firstField?.focus()
      return
    }
    // This object matches the planned POST /api/v1/predict request body.
    // No network call is made until the real FastAPI route and model exist.
    setSubmitted({ ...form })
  }

  const teamA = teams.find((team) => team.name === submitted?.team1)
  const teamB = teams.find((team) => team.name === submitted?.team2)

  return (
    <Window id="predictor" title="predict_match.exe — pre-match configuration" className="predictor-window">
      <div className="predictor-heading">
        <SectionIntro eyebrow="MATCH PREDICTOR / INPUT CONSOLE" title="Set up the match.">
          Configure a scenario after the toss and before the first ball. Add a match date to use only results before that day. The interface validates your inputs now; model inference will arrive with the FastAPI backend.
        </SectionIntro>
        <Badge variant="red">ML OFFLINE</Badge>
      </div>
      <div className="predictor-layout">
        <form className="console-form inset-panel" onSubmit={submit} noValidate>
          <div className="panel-caption">MATCH_SETUP.INI <span>07 FIELDS / 1 OPTIONAL</span></div>
          <div className="form-grid">
            <SelectField id="team1" label="TEAM A" value={form.team1} onChange={change} placeholder="Select franchise" options={teamOptions} error={errors.team1} />
            <SelectField id="team2" label="TEAM B" value={form.team2} onChange={change} placeholder="Select franchise" options={teamOptions.map((name) => ({ value: name, label: name, disabled: name === form.team1 }))} error={errors.team2} />
            <SelectField id="venue" label="VENUE" value={form.venue} onChange={change} placeholder="Select ground" options={venues} error={errors.venue} />
            <SelectField id="season" label="SEASON" value={form.season} onChange={change} placeholder="Select year" options={seasons.map(String)} error={errors.season} />
            <SelectField id="toss_winner" label="TOSS WINNER" value={form.toss_winner} onChange={change} placeholder="Select match team" options={[form.team1, form.team2].filter(Boolean).filter((name, index, all) => all.indexOf(name) === index)} error={errors.toss_winner} disabled={!form.team1 || !form.team2} />
            <SelectField id="toss_decision" label="TOSS DECISION" value={form.toss_decision} onChange={change} placeholder="Select decision" options={[{ value: 'bat', label: 'Bat first' }, { value: 'field', label: 'Field first' }]} error={errors.toss_decision} />
            <DateField id="match_date" label="MATCH DATE (OPTIONAL)" value={form.match_date} onChange={change} season={form.season} error={errors.match_date} hint="Blank: uses history before the season starts." />
          </div>
          <div className="form-actions">
            <button type="submit" className="button button-primary">Check configuration <span aria-hidden="true">→</span></button>
            <span className="form-hint">Match date is optional. All other fields are required.</span>
          </div>
        </form>

        <div className="result-stack" aria-live="polite">
          <div className="result-panel inset-panel">
            <div className="panel-caption">PREDICTION_RESULT.LOG <span>{submitted ? 'CONFIGURED' : 'IDLE'}</span></div>
            {submitted ? (
              <>
                <div className="result-alert"><span className="alert-icon" aria-hidden="true">!</span><div><strong>ML model not connected</strong><p>Your valid match setup is ready. Connect the trained model and FastAPI endpoint to return a prediction.</p></div></div>
                <div className="result-matchup"><strong>{submitted.team1}</strong><span>VS</span><strong>{submitted.team2}</strong></div>
              </>
            ) : (
              <div className="result-empty"><span className="empty-glyph" aria-hidden="true">◎</span><strong>Awaiting match setup</strong><p>Complete the form to check the configuration.</p></div>
            )}
            <div className="result-readouts">
              <div><span>PROJECTED WINNER</span><strong>—</strong></div>
              <div><span>TEAM PROBABILITIES</span><strong>Unavailable</strong></div>
              <div><span>CONFIDENCE</span><strong>Unavailable</strong></div>
            </div>
            {submitted && <p className="result-note">No probability, winner, or confidence label has been generated.</p>}
          </div>
          <div className="terminal" aria-label="Historical context">
            <div className="terminal-header"><span>historical_context.txt</span><span>● READ ONLY</span></div>
            {submitted ? (
              <div className="terminal-content">
                <p><span className="prompt">&gt;</span> {submitted.match_date || `${submitted.season} season`} · {submitted.venue}</p>
                <p><span className="prompt">&gt;</span> {teamA.name}: {teamA.wins} wins / {teamA.played} decided matches</p>
                <p><span className="prompt">&gt;</span> {teamB.name}: {teamB.wins} wins / {teamB.played} decided matches</p>
                <p className="terminal-muted">Historical totals are descriptive, not prediction inputs or probabilities.</p>
              </div>
            ) : (
              <div className="terminal-content"><p><span className="prompt">&gt;</span> Choose teams to inspect historical totals.</p><p className="terminal-muted">Dataset: 2008–2024 / {dataset.matches.toLocaleString()} match rows</p></div>
            )}
          </div>
        </div>
      </div>
    </Window>
  )
}

function Findings() {
  const [selectedTeam, setSelectedTeam] = useState('Mumbai Indians')
  const team = teams.find((item) => item.name === selectedTeam)
  const formerNames = Object.entries(franchiseAliases).filter(([, canonical]) => canonical === selectedTeam).map(([former]) => former)
  const winRate = Math.round((team.wins / team.played) * 1000) / 10
  const fieldRate = Math.round((tossStats.field.tossWinnerWon / tossStats.field.matches) * 1000) / 10
  const batRate = Math.round((tossStats.bat.tossWinnerWon / tossStats.bat.matches) * 1000) / 10

  return (
    <Window id="findings" title="findings\ — observed match history" tone="slate">
      <SectionIntro eyebrow="DATASET EXPLORER / OBSERVED DATA" title="What the scorecards say.">
        Browse match-level summaries from the cleaned 2008–2024 data. These are descriptive counts; they do not indicate what a future match will do.
      </SectionIntro>
      <div className="findings-grid">
        <article className="finding-card">
          <div className="finding-top"><span>01 / TEAM PERFORMANCE</span><span aria-hidden="true">◉</span></div>
          <label htmlFor="finding-team">FRANCHISE</label>
          <select id="finding-team" value={selectedTeam} onChange={(event) => setSelectedTeam(event.target.value)}>{teams.map((item) => <option key={item.name}>{item.name}</option>)}</select>
          <div className="finding-big"><strong>{winRate}%</strong><span>observed win rate</span></div>
          {formerNames.length > 0 && <div className="former-name">Also recorded as: {formerNames.join(', ')}</div>}
          <p>{team.wins} wins in {team.played} decided matches. Historical rename variants are counted under this franchise.</p>
          <div className="mini-status">{dataset.franchises} canonical franchise identities</div>
        </article>
        <article className="finding-card">
          <div className="finding-top"><span>02 / VENUE TRENDS</span><span aria-hidden="true">⌖</span></div>
          <h3>Most used grounds</h3>
          <div className="venue-list">{leadingVenues.map((venue) => <HistoricalLine key={venue.name} label={venue.name} value={venue.matches} max={leadingVenues[0].matches} />)}</div>
          <p>Decided matches per ground. The {dataset.rawVenueLabels} raw venue labels, including renamed grounds such as Feroz Shah Kotla, are merged into {dataset.grounds} grounds.</p>
        </article>
        <article className="finding-card">
          <div className="finding-top"><span>03 / TOSS ANALYSIS</span><span aria-hidden="true">◉</span></div>
          <h3>Did the toss winner win?</h3>
          <div className="toss-total"><strong>{tossStats.tossWinnerWon}</strong><span>of {tossStats.decided} decided matches</span></div>
          <div className="toss-row"><span>Chose to field</span><strong>{fieldRate}%</strong></div>
          <div className="meter"><span className="meter-fill meter-green" style={{ width: `${fieldRate}%` }} /></div>
          <div className="toss-row"><span>Chose to bat</span><strong>{batRate}%</strong></div>
          <div className="meter"><span className="meter-fill meter-red" style={{ width: `${batRate}%` }} /></div>
          <p>Rates describe the toss-winning side’s wins within each toss-decision group.</p>
        </article>
      </div>
      <div className="source-strip"><span>DATA SOURCE</span> Cleaned <code>data/processed/matches_clean.csv</code> · 5 no-result matches excluded from win-rate calculations · Ties with a recorded winner included.</div>
    </Window>
  )
}

function Performance() {
  const metrics = ['Accuracy', 'F1-score', 'ROC-AUC']
  return (
    <Window id="performance" title="model_performance.log — evaluation status">
      <div className="performance-heading"><SectionIntro eyebrow="MODEL STATUS / AWAITING TRAINING" title="No score without a model.">Evaluation figures will appear after candidates are trained and tested on a documented later-period holdout set.</SectionIntro><Badge variant="red">NOT TRAINED</Badge></div>
      <div className="performance-grid">
        <div className="metric-grid">{metrics.map((metric) => <div className="metric-card" key={metric}><span>{metric}</span><strong>—</strong><small>Unavailable</small></div>)}</div>
        <div className="confusion-panel inset-panel"><div className="panel-caption">CONFUSION_MATRIX.CSV</div><div className="matrix"><span>Actual A / Pred A</span><span>Actual A / Pred B</span><span>Actual B / Pred A</span><span>Actual B / Pred B</span></div><p>No held-out predictions exist yet.</p></div>
        <div className="importance-panel terminal"><div className="terminal-header"><span>feature_importance.log</span><span>● PENDING</span></div><div className="terminal-content"><p><span className="prompt">&gt;</span> Waiting for trained model artifacts...</p><p className="terminal-muted">Feature rankings and model version will appear here after evaluation.</p></div></div>
      </div>
    </Window>
  )
}

function App() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('ipl-theme') || 'light' } catch { return 'light' }
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('ipl-theme', theme) } catch { /* Storage may be disabled. */ }
  }, [theme])

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#overview"><img className="brand-symbol" src="/infinity-symbol.png" alt="" /><span>IPL ANALYTICS<span className="brand-exe">.EXE</span></span></a>
          <nav aria-label="Main navigation">{navigation.map(([label, id]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav>
          <button className="button theme-toggle" type="button" onClick={() => setTheme((current) => current === 'light' ? 'dark' : 'light')} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}><span aria-hidden="true">{theme === 'light' ? '☾' : '☀'}</span> {theme === 'light' ? 'Night' : 'Day'}</button>
        </div>
      </header>

      <main className="desktop">
        <Window id="overview" title="ipl_analytics.exe — match intelligence console" className="hero-window">
          <div className="hero-grid">
            <div className="hero-copy">
              <div className="eyebrow eyebrow-green">● IPL DATA / 2008—2024</div>
              <h1>Read the match.<br /><span>Respect the uncertainty.</span></h1>
              <p>Explore the history behind IPL matchups and prepare a winner-prediction scenario in a desktop-style cricket analytics console.</p>
              <div className="hero-actions"><a className="button button-primary" href="#predictor">Predict a Match <span aria-hidden="true">→</span></a><a className="button" href="#findings">Explore Analytics</a></div>
              <div className="hero-stats"><StatCard number={dataset.matches.toLocaleString()} label="historical matches" note="raw match rows" tone="red" /><StatCard number={`${dataset.firstSeason}–${dataset.lastSeason}`} label="seasons covered" note={`${dataset.seasons} calendar seasons`} tone="green" /><StatCard number="03" label="analytics views" note="team / venue / toss" tone="blue" /></div>
            </div>
            <div className="hero-visual" aria-label="Console preview showing data available and model offline">
              <div className="floating-window">
                <div className="mini-titlebar"><span>match_status.exe</span><span aria-hidden="true">×</span></div>
                <div className="floating-body">
                  <img className="hero-symbol" src="/infinity-symbol.png" alt="Blue-to-turquoise infinity symbol" />
                  <div><div className="floating-kicker">SYSTEM CHECK</div><h2>History loaded.<br />Model pending.</h2><p>Match records are ready to explore. Predictions will activate when the trained model is connected.</p></div>
                  <div className="floating-status"><Badge variant="green">DATA READY</Badge><Badge variant="red">MODEL OFFLINE</Badge></div>
                  <div className="floating-terminal"><span>&gt; records_loaded: {dataset.matches}</span><span>&gt; franchise_identities: {dataset.franchises}</span><span>&gt; inference_service: awaiting connection_</span></div>
                </div>
              </div>
            </div>
          </div>
          <div className="window-statusbar"><span>● LOCAL DATASET LOADED</span><span>PRE-MATCH ANALYTICS / EDUCATIONAL USE</span></div>
        </Window>

        <Window id="how-it-works" title="how_it_works.txt — the prediction pipeline" tone="slate">
          <SectionIntro eyebrow="THE PIPELINE / FIVE STEPS" title="From scorecards to a tested model.">A match prediction should come from a documented sequence of data preparation, leakage-safe features, training, inference, and evaluation.</SectionIntro>
          <div className="steps-grid">{steps.map((step) => <article className="step-card" key={step.number}><div className="step-number">STEP {step.number}</div><div className="step-glyph" aria-hidden="true">{step.glyph}</div><h3>{step.title}</h3><p>{step.text}</p></article>)}</div>
        </Window>

        <Predictor />
        <Findings />
        <Performance />

        <Window id="about" title="about_this_project.txt — project notes" tone="slate">
          <div className="about-grid"><div><SectionIntro eyebrow="ABOUT / COLLEGE PROJECT" title="Built to make predictions explainable.">IPL Cricket Match Winner Prediction is a student project by Hrituraj Roy and Amul Rai. This frontend presents verified historical summaries and prepares a clear path for later FastAPI and machine-learning integration.</SectionIntro><p className="about-disclaimer"><strong>Educational-use notice:</strong> Historical patterns and future model estimates cannot guarantee a match outcome. This project is not intended for betting.</p></div><div className="about-checklist"><div><span>✓</span> Historical franchise names preserved</div><div><span>✓</span> Input checks before inference</div><div><span>○</span> FastAPI connection pending</div><div><span>○</span> Trained model and metrics pending</div></div></div>
        </Window>
      </main>

      <footer className="taskbar"><div className="taskbar-inner"><a className="taskbar-start" href="#overview"><img className="taskbar-symbol" src="/infinity-symbol.png" alt="" /> IPL ANALYTICS</a><span className="taskbar-note">IPL Cricket Match Winner Prediction · Student project · Educational use only</span><span className="taskbar-state">● DATA READY / ML OFFLINE</span></div></footer>
    </>
  )
}

export default App
