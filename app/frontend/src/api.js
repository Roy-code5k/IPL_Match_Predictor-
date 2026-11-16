// Small client for the FastAPI service (app/backend/main.py).
// In development Vite forwards /api to http://127.0.0.1:8000 (see vite.config.js).
const BASE = `${import.meta.env.VITE_API_BASE || ''}/api/v1`
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
