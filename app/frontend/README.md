# IPL Analytics frontend

A single-page React and Vite interface for the **IPL Cricket Match Winner Prediction** college project. It uses a Windows 95/98 inspired design and historical summaries from the raw and cleaned match data (`data/raw/matches.csv`, `data/processed/matches_clean.csv`).

## Run locally

The predictor needs the FastAPI service. From the project root, after `python -m src.models.release`:

```powershell
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --port 8000
```

Then, from `app/frontend` in a second terminal:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the Local URL printed by Vite, normally `http://localhost:5173/`. Vite forwards `/api` to `http://127.0.0.1:8000` (`vite.config.js`), so no CORS setup is needed. On Windows PowerShell, `npm.cmd` works even when the execution policy blocks `npm.ps1`. Without the API the page still loads; the predictor and model-performance sections say the service is offline.

For a production check: `npm.cmd run build`, then `npm.cmd run preview` (also proxies `/api`).

Run `npm.cmd test` for the frontend regression tests (Node's built-in test runner; no extra packages). They cover innings-break validation and spreadsheet-safe CSV downloads.

Historical scenarios whose feature cutoff falls within the model's training period are explicitly labelled **Retrospective simulation**. The date limits historical input features; the model itself remains trained through the displayed cutoff. The API, batch table and all prediction CSV downloads carry `model_trained_through` and `retrospective` alongside `history_before`.

In live mode, 0 completed chase overs means the innings break, so runs and wickets must both be 0. The form and API reject nonzero values.

Batch CSVs require unique, non-empty headers and the same number of cells in every nonblank row. Quote values containing commas (including venue names) and retain empty cells for unused columns. Structural errors or oversized fields return `INVALID_CSV` before any predictions are made; valid CSV rows with invalid match inputs still receive individual errors. Exported strings beginning with spreadsheet formula characters are prefixed with an apostrophe; JSON responses and stored raw inputs retain their original values.

## What each file does

- `src/App.jsx` contains the window, badge, field, and chart components plus the page sections. The `Predictor` component has two modes: **before the first ball** (`POST /api/v1/predict`) and **during the match** (`POST /api/v1/predict-live`, with innings, completed overs, runs, wickets, and the first-innings total in the chase). It validates input first, using `GET /api/v1/options` to disable teams that did not play in the chosen season. The result shows both probabilities as text and a bar, the confidence label in words, the model release, and the history cutoff; API errors show their message and request id.
- The `ModelInputs` panel lists the key inputs behind a result (from `inputs` in the API response). The single-result "Download CSV" button was removed on 2026-11-17, together with its helper code.
- `BatchPredictor` uploads a CSV (template downloadable from the page) to `POST /api/v1/predict-batch`, shows the first 50 results, and downloads the full results CSV. It also links to the prediction log export, `GET /api/v1/predictions/log.csv`.
- `src/api.js` is the small fetch client and download helper, re-exporting `toCsv` from `src/csv.js`. Every API error becomes an `ApiError` with the server's `code`, `message`, and `request_id`; an unreachable server becomes `OFFLINE`.
- `src/csv.js` formats spreadsheet-safe CSV downloads. `src/validation.js` validates the match form, including the innings-break constraint.
- `src/data.js` contains verified descriptive totals, franchise name mappings, venue choices, and the selectable seasons (2008–2027; seasons after 2024 use all history up to May 2024).
- `src/styles.css` defines the desktop visual system, and responsive layouts. The app has a single light theme (the night-mode toggle was removed on 2026-11-17).
- `src/main.jsx` mounts the React app. `index.html` is Vite's page entry.
- `public/infinity-symbol.png` is the transparent symbol supplied for the project branding.

The model-performance section reads `GET /api/v1/model-metrics`, so its figures always come from the release that is making predictions.

## Dataset notes

The current raw file has 1,095 match rows dated 2008–2024, 17 calendar seasons, and 58 distinct raw venue labels. Five no-result matches are excluded from win-rate calculations. Four spelling/name changes are grouped under one franchise identity:

| Raw historical name | Display name |
|---|---|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Rising Pune Supergiants | Rising Pune Supergiant |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |

Deccan Chargers and Sunrisers Hyderabad, Pune Warriors and Rising Pune Supergiant, and Gujarat Lions and Gujarat Titans remain distinct. The 58 raw venue labels are merged into 36 grounds named "Ground, City", following `src/data/mappings.py`. The values in `src/data.js` are a snapshot of the cleaned data; `python -m pytest` (see `tests/test_frontend_data.py`) fails if the team list, venue list, or counts drift from the Python mappings and cleaned data.

Historical summaries describe previous matches; they are not model predictions. This project is for education, not betting.
