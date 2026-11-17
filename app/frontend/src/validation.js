export function isWhole(value, min, max) {
  if (value === '' || !/^\d+$/.test(String(value))) return false
  const number = Number(value)
  return number >= min && number <= max
}

export function getErrors(form, options) {
  const errors = {}
  if (!form.season) errors.season = 'Choose a season.'
  if (!form.venue) errors.venue = 'Choose a venue.'
  if (!form.team1) errors.team1 = 'Choose Team A.'
  if (!form.team2) errors.team2 = 'Choose Team B.'
  if (form.team1 && form.team1 === form.team2) errors.team2 = 'Choose a different franchise.'
  const allowed = options?.teams_by_season?.[form.season]
  if (allowed) {
    const future = Number(form.season) > options.last_data_season
    for (const key of ['team1', 'team2']) {
      if (form[key] && !errors[key] && !allowed.includes(form[key])) {
        errors[key] = future
          ? `${form[key]} was not active in ${options.last_data_season}, the latest season in the data.`
          : `${form[key]} did not play in ${form.season}.`
      }
    }
  }
  if (!form.toss_winner) errors.toss_winner = 'Choose the toss winner.'
  if (form.toss_winner && ![form.team1, form.team2].includes(form.toss_winner)) {
    errors.toss_winner = 'The toss winner must be playing this match.'
  }
  if (!form.toss_decision) errors.toss_decision = 'Choose the toss decision.'
  if (form.match_date && form.season && !form.match_date.startsWith(`${form.season}-`)) {
    errors.match_date = `The match date must be in the ${form.season} season, or leave it blank.`
  }
  if (form.mode === 'live') {
    if (!['1', '2'].includes(form.innings)) errors.innings = 'Choose the innings.'
    const chase = form.innings === '2'
    const minOvers = chase ? 0 : 1
    if (!isWhole(form.overs_done, minOvers, 19)) {
      errors.overs_done = chase ? 'Enter completed overs from 0 (innings break) to 19.' : 'Enter completed overs from 1 to 19.'
    }
    if (!isWhole(form.runs, 0, 400)) errors.runs = 'Enter runs as a whole number from 0 to 400.'
    if (!isWhole(form.wickets, 0, 9)) errors.wickets = 'Enter wickets from 0 to 9 (10 means the innings is over).'
    if (chase) {
      if (!isWhole(form.first_innings_total, 0, 400)) {
        errors.first_innings_total = 'Enter the first-innings total (0 to 400).'
      } else if (isWhole(form.runs, 0, 400) && Number(form.runs) > Number(form.first_innings_total)) {
        errors.runs = 'The chase is already won at this score.'
      }
      if (isWhole(form.overs_done, 0, 0)) {
        if (isWhole(form.runs, 1, 400)) errors.runs = 'At the innings break (0 overs), runs must be 0.'
        if (isWhole(form.wickets, 1, 9)) errors.wickets = 'At the innings break (0 overs), wickets must be 0.'
      }
    }
  }
  return errors
}
