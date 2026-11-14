# IPL Analytics frontend

A single-page React and Vite interface for the **IPL Cricket Match Winner Prediction** college project. It uses a Windows 95/98 inspired design and historical summaries from the raw and cleaned match data (`data/raw/matches.csv`, `data/processed/matches_clean.csv`).

## Run locally

From `app/frontend`:

```powershell
npm.cmd install
npm.cmd run dev
```

Open the Local URL printed by Vite, normally `http://localhost:5173/`. On Windows PowerShell, `npm.cmd` works even when the execution policy blocks `npm.ps1`.

For a production check:

```powershell
npm.cmd run build
npm.cmd run preview
```

## What each file does

- `src/App.jsx` contains the reusable window, badge, statistic, field, and chart components, plus the page sections and interactions.
- `src/data.js` contains verified descriptive totals, franchise name mappings, and venue choices derived from the raw match CSV.
- `src/styles.css` defines the desktop visual system, responsive layouts, and light/dark themes.
- `src/main.jsx` mounts the React app. `index.html` is Vite's page entry.
- `public/infinity-symbol.png` is the transparent symbol supplied for the project branding.

The match form has six required inputs and an optional match date. Team A and Team B must be different, the toss winner must be one of them, and a match date must fall in the chosen season. A valid submission displays **“ML model not connected”** and the submitted setup. It does not call an API or generate a prediction.

## Dataset notes

The current raw file has 1,095 match rows dated 2008–2024, 17 calendar seasons, and 58 distinct raw venue labels. Five no-result matches are excluded from win-rate calculations. Four spelling/name changes are grouped under one franchise identity:

| Raw historical name | Display name |
|---|---|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Rising Pune Supergiants | Rising Pune Supergiant |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |

Deccan Chargers and Sunrisers Hyderabad, Pune Warriors and Rising Pune Supergiant, and Gujarat Lions and Gujarat Titans remain distinct. The 58 raw venue labels are merged into 36 grounds named "Ground, City", following `src/data/mappings.py`. The values in `src/data.js` are a snapshot of the cleaned data; `python -m pytest` (see `tests/test_frontend_data.py`) fails if the team list, venue list, or counts drift from the Python mappings and cleaned data.

## Future FastAPI connection

The `Predictor` component's validated `form` object is ready to send to the planned `POST /api/v1/predict` endpoint. Its keys are `team1`, `team2`, `venue`, `season`, `toss_winner`, `toss_decision`, and the optional `match_date` (`YYYY-MM-DD`, empty when not given; it must fall in the chosen season). Replace the marked `setSubmitted` path with a request only after the backend and trained model exist. Then populate winner, both probabilities, confidence, and model version from the actual response. The model-performance section should likewise use real held-out evaluation data from `GET /api/v1/model-metrics`.

Historical summaries describe previous matches; they are not model predictions. This project is for education, not betting.
