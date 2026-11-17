import assert from 'node:assert/strict'
import test from 'node:test'
import { getErrors } from '../src/validation.js'

const form = {
  mode: 'live', season: '2025', venue: 'Wankhede Stadium, Mumbai',
  team1: 'Mumbai Indians', team2: 'Chennai Super Kings', toss_winner: 'Mumbai Indians',
  toss_decision: 'bat', innings: '2', overs_done: '0', runs: '0', wickets: '0', first_innings_total: '180',
}

test('the innings break accepts a zero score and requires both values', () => {
  assert.deepEqual(getErrors(form), {})
  assert.ok(getErrors({ ...form, runs: '' }).runs)
  assert.ok(getErrors({ ...form, wickets: '' }).wickets)
})

test('nonzero runs and wickets at the innings break have field errors', () => {
  assert.match(getErrors({ ...form, runs: '100' }).runs, /innings break/)
  assert.match(getErrors({ ...form, wickets: '5' }).wickets, /innings break/)
  const errors = getErrors({ ...form, overs_done: '00', runs: '100', wickets: '5' })
  assert.match(errors.runs, /innings break/)
  assert.match(errors.wickets, /innings break/)
})

test('scores after completed overs are still accepted', () => {
  assert.deepEqual(getErrors({ ...form, overs_done: '10', runs: '100', wickets: '5' }), {})
  assert.deepEqual(getErrors({ ...form, innings: '1', overs_done: '10', runs: '100', wickets: '5' }), {})
})

test('existing form validation still rejects finished and inconsistent matches', () => {
  assert.ok(getErrors({ ...form, overs_done: '10', runs: '181' }).runs)
  assert.ok(getErrors({ ...form, overs_done: '10', wickets: '10' }).wickets)
  assert.ok(getErrors({ ...form, team2: form.team1 }).team2)
  assert.ok(getErrors({ ...form, innings: '1' }).overs_done)
})
