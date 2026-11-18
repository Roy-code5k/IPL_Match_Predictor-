// Small client for the FastAPI service (app/backend/main.py).
// In development Vite forwards /api to http://127.0.0.1:8000 (see vite.config.js).
const BASE = `${import.meta.env?.VITE_API_BASE || ''}/api/v1` // `?.` lets node tests import this file
const OFFLINE_MESSAGE = 'The prediction service is not reachable. Start it from the project root with: uvicorn app.backend.main:app'

export class ApiError extends Error {
  constructor(code, message, requestId = null, status = 0) {
    super(message)
    this.code = code
    this.requestId = requestId
    this.status = status
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  } catch {
    throw new ApiError('OFFLINE', OFFLINE_MESSAGE)
  }
  let payload = null
  try { payload = await response.json() } catch { /* empty or non-JSON body */ }
  if (response.ok) return payload
  const error = payload?.error
  if (error) throw new ApiError(error.code, error.message, error.request_id, response.status)
  // No JSON error body: usually the Vite proxy reporting that the API is down.
  if (response.status >= 500) throw new ApiError('OFFLINE', OFFLINE_MESSAGE, null, response.status)
  throw new ApiError('HTTP_ERROR', `Request failed with status ${response.status}.`, response.headers.get('X-Request-ID'), response.status)
}

export const getHealth = () => request('/health')
export const getOptions = () => request('/options')
export const getMetrics = () => request('/model-metrics')
export const predict = (body) => request('/predict', { method: 'POST', body: JSON.stringify(body) })
export const predictLive = (body) => request('/predict-live', { method: 'POST', body: JSON.stringify(body) })
export const predictBatch = (csv) => request('/predict-batch', { method: 'POST', body: JSON.stringify({ csv }) })
export const predictionLogUrl = `${BASE}/predictions/log.csv`

export { toCsv } from './csv.js'

export function downloadText(filename, text, type = 'text/csv') {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
