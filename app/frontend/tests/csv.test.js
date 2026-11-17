import assert from 'node:assert/strict'
import test from 'node:test'
import { toCsv } from '../src/csv.js'

test('formula-like strings are exported as text', () => {
  for (const value of ['=1+1', '+1+1', '-1+1', '@SUM(1)', '  =1+1', '\t=1+1', '\ufeff =1+1', '\x00=1+1']) {
    assert.equal(toCsv(['input'], [{ input: value }]), `input\r\n'${value}\r\n`)
  }
})

test('CSV quoting contains commas, quotes, CR and LF inside a single cell', () => {
  assert.equal(toCsv(['input'], [{ input: '=HYPERLINK("example","label")' }]), 'input\r\n"\'=HYPERLINK(""example"",""label"")"\r\n')
  for (const value of ['\r=1+1', '\n=1+1']) {
    assert.equal(toCsv(['input'], [{ input: value }]), `input\r\n"'${value}"\r\n`)
  }
  assert.equal(toCsv(['venue'], [{ venue: 'Wankhede Stadium, Mumbai' }]), 'venue\r\n"Wankhede Stadium, Mumbai"\r\n')
})

test('numbers, booleans and empty values retain their original representation', () => {
  assert.equal(toCsv(['number', 'flag', 'empty'], [{ number: -1, flag: false, empty: null }]), 'number,flag,empty\r\n-1,false,\r\n')
})
