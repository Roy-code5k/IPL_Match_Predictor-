import { defineConfig } from 'vite'

// During development the React app calls /api/... on the Vite server, which forwards
// to the FastAPI service started with: uvicorn app.backend.main:app (port 8000).
const api = { '/api': 'http://127.0.0.1:8000' }

export default defineConfig({
  server: { proxy: api },
  preview: { proxy: api },
})
