import React, { useCallback, useEffect, useState } from 'react'
import { downloadText, getHealth, getOptions, predict, predictBatch, predictionLogUrl, predictLive, toCsv } from './api.js'
import { getErrors, isWhole } from './validation.js'
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
  ['Batch', 'batch'],
  ['Findings', 'findings'],
  ['About', 'about'],
]

const steps = [
  { number: '01', glyph: '▦', title: 'Data Preparation', text: 'Inspect raw match records, resolve missing outcomes, and document team and venue names.' },
  { number: '02', glyph: '◇', title: 'Feature Engineering', text: 'Build historical team, venue, form, and toss features from matches before the prediction cutoff, plus the live match state from ball-by-ball data.' },
  { number: '03', glyph: '⚙', title: 'Model Training', text: 'Compare baselines, logistic regression, Random Forest, and XGBoost on seasons the models never trained on.' },
  { number: '04', glyph: '◎', title: 'Prediction', text: 'Send validated inputs to the FastAPI service and show both teams’ probabilities, before the match or from the live score.' },
  { number: '05', glyph: '✓', title: 'Evaluation', text: 'Test the selected models once on 2023–2024 and report accuracy for each prediction moment.' },
]

const emptyForm = {
  mode: 'prematch',
  team1: '',
  team2: '',
  venue: '',
  season: '',
  toss_winner: '',
  toss_decision: '',
  match_date: '',
  innings: '2',
  overs_done: '',
  runs: '',
  wickets: '',
  first_innings_total: '',
}

const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`

// Input columns of the batch CSV template (same as the API's batch input columns).
const INPUT_COLUMNS = ['team1', 'team2', 'venue', 'season', 'toss_winner', 'toss_decision', 'match_date', 'is_playoff', 'innings', 'overs_done', 'runs', 'wickets', 'first_innings_total']

const BATCH_TEMPLATE = toCsv(INPUT_COLUMNS, [
  { team1: 'Mumbai Indians', team2: 'Chennai Super Kings', venue: 'Wankhede Stadium, Mumbai', season: 2025, toss_winner: 'Chennai Super Kings', toss_decision: 'field' },
  { team1: 'Kolkata Knight Riders', team2: 'Sunrisers Hyderabad', venue: 'Eden Gardens, Kolkata', season: 2025, toss_winner: 'Sunrisers Hyderabad', toss_decision: 'bat', innings: 2, overs_done: 10, runs: 85, wickets: 3, first_innings_total: 190 },
])

function formatInput(value, format) {
  if (value === null || value === undefined) return '—'
  if (format === 'percent') return percent(value)
  if (format === 'yesno') return value ? 'yes' : 'no'
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
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

function ServiceBadge({ service }) {
  if (service.status === 'online') return <Badge variant="green">MODEL ONLINE</Badge>
  if (service.status === 'checking') return <Badge>CHECKING…</Badge>
  return <Badge variant="red">{service.status === 'degraded' ? 'MODEL NOT LOADED' : 'API OFFLINE'}</Badge>
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

function requestBody(form) {
  const body = {
    team1: form.team1,
    team2: form.team2,
    venue: form.venue,
    season: Number(form.season),
    toss_winner: form.toss_winner,
    toss_decision: form.toss_decision,
    ...(form.match_date ? { match_date: form.match_date } : {}),
  }
  if (form.mode !== 'live') return body
  return {
    ...body,
    innings: Number(form.innings),
    overs_done: Number(form.overs_done),
    runs: Number(form.runs),
    wickets: Number(form.wickets),
    ...(form.innings === '2' ? { first_innings_total: Number(form.first_innings_total) } : {}),
  }
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

function NumberField({ id, label, value, onChange, min, max, error, hint }) {
  const describedBy = [error && `${id}-error`, hint && `${id}-hint`].filter(Boolean).join(' ') || undefined
  return (
    <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <input type="number" inputMode="numeric" id={id} name={id} value={value} onChange={onChange} min={min} max={max} step="1" aria-invalid={Boolean(error)} aria-describedby={describedBy} />
      {error && <small id={`${id}-error`} className="field-error">{error}</small>}
      {hint && <small id={`${id}-hint`} className="field-hint">{hint}</small>}
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

function HistoricalLine({ label, value, max, color = 'blue', display }) {
  return (
    <div className="historical-line">
      <div className="historical-line-top"><span>{label}</span><strong>{display ?? value}</strong></div>
      <div className="meter" aria-hidden="true"><span className={`meter-fill meter-${color}`} style={{ width: `${(value / max) * 100}%` }} /></div>
    </div>
  )
}

function ModeSwitch({ value, onChange }) {
  const choices = [
    ['prematch', 'Before the first ball', 'Toss done, no balls bowled'],
    ['live', 'During the match', 'Score at the end of an over'],
  ]
  return (
    <fieldset className="mode-switch">
      <legend>PREDICTION MOMENT</legend>
      {choices.map(([key, title, note]) => (
        <label key={key} className={value === key ? 'is-selected' : ''}>
          <input type="radio" name="mode" value={key} checked={value === key} onChange={onChange} />
          <span><strong>{title}</strong><small>{note}</small></span>
        </label>
      ))}
    </fieldset>
  )
}

function ProbabilityBar({ result }) {
  const p1 = result.team1_probability
  return (
    <div className="probability">
      <div className="probability-labels">
        <span><strong>{result.team1}</strong> {percent(p1)}</span>
        <span>{percent(1 - p1)} <strong>{result.team2}</strong></span>
      </div>
      <div className="probability-bar" role="img" aria-label={`${result.team1} ${percent(p1)}, ${result.team2} ${percent(1 - p1)}`}>
        <span className="probability-team1" style={{ width: `${p1 * 100}%` }} />
        <span className="probability-team2" style={{ width: `${(1 - p1) * 100}%` }} />
      </div>
    </div>
  )
}

function stateSummary(result) {
  const s = result.state
  if (!s) return null
  const score = `${s.runs}/${s.wickets}`
  if (s.innings === 1) return `${result.batting_first} ${score} after ${s.overs_done} overs (1st innings)`
  if (s.overs_done === 0) return `Innings break: ${result.batting_first === result.team1 ? result.team2 : result.team1} need ${s.target} to win`
  return `Chase: ${score} after ${s.overs_done} overs, target ${s.target} (${s.target - s.runs} needed off ${120 - 6 * s.overs_done} balls)`
}

function ModelInputs({ result }) {
  return (
    <div className="terminal" aria-label="Model inputs">
      <div className="terminal-header"><span>model_inputs.txt</span><span>● {result.mode === 'live' ? 'LIVE STATE' : 'PRE-MATCH'}</span></div>
      <div className="terminal-content">
        {result.inputs.map((item) => (
          <p key={item.label}>
            <span className="prompt">&gt;</span> {item.label}:{' '}
            {'value' in item && item.value !== null
              ? formatInput(item.value, item.format)
              : `${result.team1} ${formatInput(item.team1, item.format)} · ${result.team2} ${formatInput(item.team2, item.format)}`}
            {item.note && <span className="terminal-muted"> ({item.note})</span>}
          </p>
        ))}
        <p className="terminal-muted">Historical inputs use matches before {result.history_before}; live inputs use the score you entered. The model itself was trained through {result.model_trained_through}. Win rates are pulled toward 50% when there are few matches. These are inputs, not reasons.</p>
      </div>
    </div>
  )
}

function ResultView({ result, stale }) {
  return (
    <>
      {result.retrospective && (
        <div className="result-alert result-alert-close"><span className="alert-icon" aria-hidden="true">!</span><div><strong>Retrospective simulation</strong><p>The model was trained on matches through {result.model_trained_through}, including matches on or after this scenario’s cutoff. This does not show what it could have predicted at the time.</p></div></div>
      )}
      {result.close_call ? (
        <div className="result-alert result-alert-close"><span className="alert-icon" aria-hidden="true">≈</span><div><strong>Too close to call</strong><p>The two probabilities are within 10 points. Treat this match as roughly 50/50.</p></div></div>
      ) : null}
      <div className="result-matchup"><strong>{result.team1}</strong><span>VS</span><strong>{result.team2}</strong></div>
      {stateSummary(result) && <p className="result-state">{stateSummary(result)}</p>}
      <ProbabilityBar result={result} />
      <div className="result-readouts">
        <div><span>PROJECTED WINNER</span><strong>{result.predicted_winner}</strong></div>
        <div><span>WIN PROBABILITY</span><strong>{percent(Math.max(result.team1_probability, result.team2_probability))}</strong></div>
        <div><span>CONFIDENCE</span><strong>{result.confidence}</strong></div>
      </div>
      <p className="result-note">
        {result.mode === 'live' ? 'Live model' : 'Pre-match model'}: {result.model.name}, release {result.model.release}. Historical inputs use matches before {result.history_before}. Model trained through {result.model_trained_through}.
        {result.mode === 'pre-match' && ' Pre-match predictions were no better than a coin flip on the 2023–2024 test seasons, so read them as close to 50/50.'}
        {' '}This is an estimate, not a guarantee.
      </p>
      {stale && <p className="result-stale">Inputs changed since this prediction. Predict again to update it.</p>}
    </>
  )
}

function BatchPredictor({ service, options }) {
  const [file, setFile] = useState(null)
  const [running, setRunning] = useState(false)
  const [outcome, setOutcome] = useState(null)
  const maxRows = options?.batch?.max_rows ?? 200

  async function run(event) {
    event.preventDefault()
    if (!file) {
      setOutcome({ error: { message: 'Choose a CSV file first.' } })
      return
    }
    setRunning(true)
    try {
      const text = await file.text()
      setOutcome({ data: await predictBatch(text), name: file.name })
    } catch (error) {
      setOutcome({ error })
    } finally {
      setRunning(false)
    }
  }

  const data = outcome?.data
  const resultName = outcome?.name ? outcome.name.replace(/\.csv$/i, '') + '_predictions.csv' : 'predictions.csv'

  return (
    <Window id="batch" title="batch_predict.exe — many matches from a CSV" tone="slate">
      <div className="predictor-heading">
        <SectionIntro eyebrow="BATCH / CSV IN, CSV OUT" title="Predict a list of matches.">
          Upload up to {maxRows} matches as a CSV with the same fields as the form. Rows with innings, overs, runs and wickets get a live prediction. Bad rows are reported, not dropped.
        </SectionIntro>
        <ServiceBadge service={service} />
      </div>
      <div className="batch-layout">
        <form className="console-form inset-panel" onSubmit={run}>
          <div className="panel-caption">BATCH_INPUT.CSV <span>MAX {maxRows} ROWS</span></div>
          <ol className="batch-steps">
            <li><button type="button" className="link-button" onClick={() => downloadText('batch_template.csv', BATCH_TEMPLATE)}>Download the template</button> (one pre-match and one live example).</li>
            <li>Fill in one match per row. Leave the live columns empty for a pre-match prediction.</li>
            <li>Choose the file and run the batch.</li>
          </ol>
          <div className="form-field">
            <label htmlFor="batch-file">CSV FILE</label>
            <input id="batch-file" type="file" accept=".csv,text/csv" onChange={(event) => { setFile(event.target.files?.[0] ?? null); setOutcome(null) }} />
          </div>
          <div className="form-actions">
            <button type="submit" className="button button-primary" disabled={running}>{running ? 'Running…' : 'Run batch'} <span aria-hidden="true">→</span></button>
            <a className="form-hint" href={predictionLogUrl} download>Download the prediction log (CSV)</a>
          </div>
        </form>
        <div className="result-panel inset-panel batch-results" aria-live="polite" aria-busy={running}>
          <div className="panel-caption">BATCH_RESULTS.LOG <span>{running ? 'RUNNING' : data ? 'COMPLETE' : outcome?.error ? 'ERROR' : 'IDLE'}</span></div>
          {outcome?.error && (
            <div className="result-alert" role="alert"><span className="alert-icon" aria-hidden="true">!</span><div><strong>Batch not run</strong><p>{outcome.error.message}</p>{outcome.error.requestId && <p className="request-id">Request id: {outcome.error.requestId}</p>}</div></div>
          )}
          {data && (
            <>
              <p className="batch-summary"><strong>{data.rows}</strong> matches: <strong>{data.succeeded}</strong> predicted, <strong>{data.failed}</strong> with errors.</p>
              {data.results.some((row) => row.retrospective) && <p className="result-stale">Rows marked “Retrospective simulation” use models trained on matches on or after the scenario cutoff. They do not show what could have been predicted at the time. Training cutoffs are included in the download.</p>}
              <div className="table-scroll">
                <table className="data-table">
                  <caption className="sr-only">Batch prediction results</caption>
                  <thead><tr><th scope="col">#</th><th scope="col">Match</th><th scope="col">Mode</th><th scope="col">Predicted winner</th><th scope="col">Chance</th><th scope="col">Confidence / error</th></tr></thead>
                  <tbody>
                    {data.results.slice(0, 50).map((row, index) => (
                      <tr key={index} className={row.error ? 'row-error' : ''}>
                        <td>{index + 1}</td>
                        <td>{row.team1 || '?'} v {row.team2 || '?'}</td>
                        <td>{row.mode || '—'}{row.retrospective && <><br /><strong>Retrospective simulation</strong></>}</td>
                        <td>{row.predicted_winner || '—'}</td>
                        <td>{row.error ? '—' : percent(Math.max(row.team1_probability, row.team2_probability))}</td>
                        <td>{row.error || row.confidence}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {data.results.length > 50 && <p className="result-note">Showing the first 50 rows. The download has all {data.rows}.</p>}
              <div className="result-actions"><button type="button" className="button button-small" onClick={() => downloadText(resultName, data.csv)}>Download results CSV</button></div>
            </>
          )}
          {!data && !outcome?.error && <div className="result-empty"><span className="empty-glyph" aria-hidden="true">▤</span><strong>No batch yet</strong><p>Results appear here with a download link.</p></div>}
        </div>
      </div>
    </Window>
  )
}

function Predictor({ service, options, onRetry }) {
  const [form, setForm] = useState(emptyForm)
  const [attempted, setAttempted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [outcome, setOutcome] = useState(null) // { result } or { error }
  const [stale, setStale] = useState(false)
  const errors = attempted ? getErrors(form, options) : {}
  const allowed = options?.teams_by_season?.[form.season]
  const teamOptions = teams.map((team) => ({ value: team.name, label: team.name, disabled: Boolean(allowed && !allowed.includes(team.name)) }))
  const live = form.mode === 'live'
  const futureSeason = form.season && Number(form.season) > dataset.lastSeason

  function change(event) {
    const { name, value } = event.target
    setStale(true)
    setForm((previous) => {
      const next = { ...previous, [name]: value }
      if ((name === 'team1' || name === 'team2') && ![next.team1, next.team2].includes(next.toss_winner)) {
        next.toss_winner = ''
      }
      return next
    })
  }

  async function submit(event) {
    event.preventDefault()
    setAttempted(true)
    const nextErrors = getErrors(form, options)
    if (Object.keys(nextErrors).length > 0) {
      document.getElementById(Object.keys(nextErrors)[0])?.focus()
      return
    }
    const body = requestBody(form)
    setLoading(true)
    try {
      const result = live ? await predictLive(body) : await predict(body)
      setOutcome({ result })
      setStale(false)
    } catch (error) {
      setOutcome({ error })
      if (error.code === 'OFFLINE') onRetry()
    } finally {
      setLoading(false)
    }
  }

  const status = loading ? 'RUNNING' : outcome?.error ? 'ERROR' : outcome?.result ? 'COMPLETE' : 'IDLE'

  return (
    <Window id="predictor" title="predict_match.exe — match configuration" className="predictor-window">
      <div className="predictor-heading">
        <SectionIntro eyebrow="MATCH PREDICTOR / INPUT CONSOLE" title="Set up the match.">
          Predict after the toss, or update the prediction during the match from the score at the end of an over. A match date limits historical inputs to earlier matches. Dates within the model’s training period produce a retrospective simulation.
        </SectionIntro>
        <ServiceBadge service={service} />
      </div>
      <div className="predictor-layout">
        <form className="console-form inset-panel" onSubmit={submit} noValidate>
          <div className="panel-caption">MATCH_SETUP.INI <span>{live ? 'LIVE SCORE' : 'PRE-MATCH'}</span></div>
          <ModeSwitch value={form.mode} onChange={change} />
          <div className="form-grid">
            <SelectField id="season" label="SEASON" value={form.season} onChange={change} placeholder="Select year" options={seasons.map((season) => ({ value: String(season), label: season > dataset.lastSeason ? `${season} (after the data)` : String(season) }))} error={errors.season} />
            <SelectField id="venue" label="VENUE" value={form.venue} onChange={change} placeholder="Select ground" options={venues} error={errors.venue} />
            <SelectField id="team1" label="TEAM A" value={form.team1} onChange={change} placeholder="Select franchise" options={teamOptions} error={errors.team1} />
            <SelectField id="team2" label="TEAM B" value={form.team2} onChange={change} placeholder="Select franchise" options={teamOptions.map((option) => ({ ...option, disabled: option.disabled || option.value === form.team1 }))} error={errors.team2} />
            <SelectField id="toss_winner" label="TOSS WINNER" value={form.toss_winner} onChange={change} placeholder="Select match team" options={[form.team1, form.team2].filter(Boolean).filter((name, index, all) => all.indexOf(name) === index)} error={errors.toss_winner} disabled={!form.team1 || !form.team2} />
            <SelectField id="toss_decision" label="TOSS DECISION" value={form.toss_decision} onChange={change} placeholder="Select decision" options={[{ value: 'bat', label: 'Bat first' }, { value: 'field', label: 'Field first' }]} error={errors.toss_decision} />
            <DateField id="match_date" label="MATCH DATE (OPTIONAL)" value={form.match_date} onChange={change} season={form.season} error={errors.match_date} hint={futureSeason ? 'Seasons after 2024 use all history up to May 2024.' : 'Blank: uses history before the season starts.'} />
          </div>
          {live && (
            <div className="live-fields">
              <div className="panel-caption">LIVE_SCORE.DAT <span>END OF AN OVER</span></div>
              <div className="form-grid">
                <SelectField id="innings" label="INNINGS" value={form.innings} onChange={change} placeholder="Select innings" options={[{ value: '1', label: '1st innings' }, { value: '2', label: 'Chase (2nd innings)' }]} error={errors.innings} />
                {form.innings === '2'
                  ? <NumberField id="first_innings_total" label="FIRST-INNINGS TOTAL" value={form.first_innings_total} onChange={change} min={0} max={400} error={errors.first_innings_total} hint={isWhole(form.first_innings_total, 0, 400) ? `Target: ${Number(form.first_innings_total) + 1}` : 'Runs scored by the side batting first.'} />
                  : <div className="field-spacer" aria-hidden="true" />}
                <NumberField id="overs_done" label="OVERS COMPLETED" value={form.overs_done} onChange={change} min={form.innings === '2' ? 0 : 1} max={19} error={errors.overs_done} hint={form.innings === '2' ? '0 = innings break; runs and wickets must both be 0.' : 'Whole overs only.'} />
                <NumberField id="runs" label={form.innings === '2' ? 'RUNS (CHASING SIDE)' : 'RUNS'} value={form.runs} onChange={change} min={0} max={400} error={errors.runs} />
                <NumberField id="wickets" label="WICKETS LOST" value={form.wickets} onChange={change} min={0} max={9} error={errors.wickets} />
              </div>
            </div>
          )}
          <div className="form-actions">
            <button type="submit" className="button button-primary" disabled={loading}>{loading ? 'Predicting…' : 'Predict'} <span aria-hidden="true">→</span></button>
            <span className="form-hint">Match date is optional. All other fields are required.</span>
          </div>
        </form>

        <div className="result-stack">
          <div className="result-panel inset-panel" aria-live="polite" aria-busy={loading}>
            <div className="panel-caption">PREDICTION_RESULT.LOG <span>{status}</span></div>
            {outcome?.result ? (
              <ResultView result={outcome.result} stale={stale} />
            ) : outcome?.error ? (
              <div className="result-alert" role="alert"><span className="alert-icon" aria-hidden="true">!</span><div><strong>{outcome.error.code === 'OFFLINE' ? 'Prediction service offline' : 'Prediction not made'}</strong><p>{outcome.error.message}</p>{outcome.error.requestId && <p className="request-id">Request id: {outcome.error.requestId}</p>}</div></div>
            ) : (
              <div className="result-empty"><span className="empty-glyph" aria-hidden="true">◎</span><strong>Awaiting match setup</strong><p>Complete the form and press Predict.</p></div>
            )}
          </div>
          {outcome?.result?.inputs?.length ? (
            <ModelInputs result={outcome.result} />
          ) : (
            <div className="terminal" aria-label="Model inputs">
              <div className="terminal-header"><span>model_inputs.txt</span><span>● WAITING</span></div>
              <div className="terminal-content"><p><span className="prompt">&gt;</span> Predict a match to see what the model looked at.</p><p className="terminal-muted">Dataset: 2008–2024 / {dataset.matches.toLocaleString()} match rows</p></div>
            </div>
          )}
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

function useService() {
  const [service, setService] = useState({ status: 'checking' })
  const check = useCallback(async () => {
    try {
      const health = await getHealth()
      setService(health.model_loaded
        ? { status: 'online', release: health.release, cutoff: health.history_cutoff }
        : { status: 'degraded', detail: health.detail })
    } catch {
      setService({ status: 'offline' })
    }
  }, [])
  useEffect(() => { check() }, [check])
  return [service, check]
}

function App() {
  const [service, recheck] = useService()
  const [options, setOptions] = useState(null)

  useEffect(() => {
    if (service.status !== 'online') return
    getOptions().then(setOptions).catch(() => setOptions(null))
  }, [service.status])

  const online = service.status === 'online'

  return (
    <>
      <div className="bg-logo" aria-hidden="true"><img src="/ipl-logo-bg.png" alt="" /></div>
      <div className="bg-blobs" aria-hidden="true">
        <span className="blob blob-left blob-1" />
        <span className="blob blob-left blob-2" />
        <span className="blob blob-left blob-3" />
        <span className="blob blob-right blob-4" />
        <span className="blob blob-right blob-5" />
        <span className="blob blob-right blob-6" />
      </div>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#overview"><img className="brand-symbol" src="/infinity-symbol.png" alt="" /><span>IPL ANALYTICS<span className="brand-exe">.EXE</span></span></a>
          <nav aria-label="Main navigation">{navigation.map(([label, id]) => <a key={id} href={`#${id}`}>{label}</a>)}</nav>        </div>
      </header>

      <main className="desktop">
        <Window id="overview" title="ipl_analytics.exe — match intelligence console" className="hero-window">
          <div className="hero-grid">
            <div className="hero-copy">
              <div className="eyebrow eyebrow-green">● IPL DATA / 2008—2024</div>
              <h1>Read the match.<br /><span>Respect the uncertainty.</span></h1>
              <p>Explore the history behind IPL matchups and estimate each team’s chance of winning, before the first ball or live from the score.</p>
              <div className="hero-actions"><a className="button button-primary" href="#predictor">Predict a Match <span aria-hidden="true">→</span></a><a className="button" href="#findings">Explore Analytics</a></div>
              <div className="hero-stats"><StatCard number={dataset.matches.toLocaleString()} label="historical matches" note="raw match rows" tone="red" /><StatCard number={`${dataset.firstSeason}–${dataset.lastSeason}`} label="seasons covered" note={`${dataset.seasons} calendar seasons`} tone="green" /><StatCard number="02" label="prediction modes" note="pre-match / live score" tone="blue" /></div>
            </div>
            <div className="hero-visual" aria-label={`Console preview: data loaded, model ${online ? 'online' : 'offline'}`}>
              <div className="floating-window">
                <div className="mini-titlebar"><span>match_status.exe</span><span aria-hidden="true">×</span></div>
                <div className="floating-body">
                  <img className="hero-symbol" src="/infinity-symbol.png" alt="Blue-to-turquoise infinity symbol" />
                  <div><div className="floating-kicker">SYSTEM CHECK</div><h2>History loaded.<br />{online ? 'Model online.' : 'Model offline.'}</h2><p>{online ? `Release ${service.release} is ready. Predictions use matches up to ${service.cutoff}.` : 'Match records are ready to explore. Start the API to make predictions.'}</p></div>
                  <div className="floating-status"><Badge variant="green">DATA READY</Badge><ServiceBadge service={service} /></div>
                  <div className="floating-terminal"><span>&gt; records_loaded: {dataset.matches}</span><span>&gt; franchise_identities: {dataset.franchises}</span><span>&gt; inference_service: {online ? `connected (release ${service.release})` : service.status === 'checking' ? 'checking…' : 'not reachable_'}</span></div>
                </div>
              </div>
            </div>
          </div>
          <div className="window-statusbar"><span>● LOCAL DATASET LOADED</span><span>PRE-MATCH + LIVE ANALYTICS / EDUCATIONAL USE</span></div>
        </Window>

        <Window id="how-it-works" title="how_it_works.txt — the prediction pipeline" tone="slate">
          <SectionIntro eyebrow="THE PIPELINE / FIVE STEPS" title="From scorecards to a tested model.">A match prediction should come from a documented sequence of data preparation, leakage-safe features, training, inference, and evaluation.</SectionIntro>
          <div className="steps-grid">{steps.map((step) => <article className="step-card" key={step.number}><div className="step-number">STEP {step.number}</div><div className="step-glyph" aria-hidden="true">{step.glyph}</div><h3>{step.title}</h3><p>{step.text}</p></article>)}</div>
        </Window>

        <Predictor service={service} options={options} onRetry={recheck} />
        <BatchPredictor service={service} options={options} />
        <Findings />

        <Window id="about" title="about_this_project.txt — project notes" tone="slate">
          <div className="about-grid"><div><SectionIntro eyebrow="ABOUT / COLLEGE PROJECT" title="Built to make predictions explainable.">IPL Cricket Match Winner Prediction is a student project by Hrituraj Roy and Amul Rai. It pairs verified historical summaries with tested models served by a FastAPI backend.</SectionIntro><p className="about-disclaimer"><strong>Educational-use notice:</strong> Historical patterns and model estimates cannot guarantee a match outcome. This project is not intended for betting.</p></div><div className="about-checklist"><div><span>✓</span> Historical franchise names preserved</div><div><span>✓</span> Input checks before inference</div><div><span>✓</span> Models tested on unseen seasons</div><div><span>{online ? '✓' : '○'}</span> FastAPI service {online ? 'connected' : 'not connected'}</div></div></div>
        </Window>
      </main>

      <footer className="taskbar"><div className="taskbar-inner"><a className="taskbar-start" href="#overview"><img className="taskbar-symbol" src="/infinity-symbol.png" alt="" /> IPL ANALYTICS</a><span className="taskbar-note">IPL Cricket Match Winner Prediction · Student project · Educational use only</span><span className={`taskbar-state ${online ? '' : 'taskbar-offline'}`}>● DATA READY / {online ? 'MODEL ONLINE' : 'MODEL OFFLINE'}</span></div></footer>
    </>
  )
}

export default App
