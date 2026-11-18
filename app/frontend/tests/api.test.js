import assert from 'node:assert/strict'
import test from 'node:test'
import { ApiError, getHealth } from '../src/api.js'

function respond(status, body, headers = {}) {
  globalThis.fetch = async () => new Response(body === undefined ? null : JSON.stringify(body), {
    status, headers: new Headers(headers),
  })
}

async function failure() {
  try {
    await getHealth()
  } catch (error) {
    assert.ok(error instanceof ApiError)
    return error
  }
  assert.fail('expected an ApiError')
}

test('a JSON body is returned on success', async () => {
  respond(200, { status: 'ok' })
  assert.deepEqual(await getHealth(), { status: 'ok' })
})

test('the API error shape keeps its code, message, request id and status', async () => {
  respond(400, { error: { code: 'UNKNOWN_NAME', message: 'Unknown team', request_id: 'r1' } })
  const error = await failure()
  assert.deepEqual([error.code, error.message, error.requestId, error.status], ['UNKNOWN_NAME', 'Unknown team', 'r1', 400])
})

test('a network failure or a 5xx without an error body means the service is offline', async () => {
  globalThis.fetch = async () => { throw new TypeError('fetch failed') }
  assert.equal((await failure()).code, 'OFFLINE')
  respond(502)
  const error = await failure()
  assert.equal(error.code, 'OFFLINE')
  assert.equal(error.status, 502)
})

test('other statuses without an error body are reported with the status', async () => {
  respond(404, undefined, { 'X-Request-ID': 'r2' })
  const error = await failure()
  assert.deepEqual([error.code, error.status, error.requestId], ['HTTP_ERROR', 404, 'r2'])
})
