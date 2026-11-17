# Local setup

Tested on Windows 11 with Python 3.13.14 and Node.js (for the frontend). Run every command from the project root unless a step says otherwise.

## 1. Python environment

PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. Alternatively, skip activation and call `.venv\Scripts\python.exe` directly.

Git Bash / macOS / Linux: use `python3 -m venv .venv` and `source .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).

## 2. Raw data

Download `matches.csv` and `deliveries.csv` as described in [data_source.md](data_source.md) and place them in `data/raw/`.

## 3. Verify the data

```powershell
python -m src.data.inspect_raw   # checks hashes and schema, rewrites reports/data_quality_report.md
python -m pytest                  # runs all tests
```

Both must report success before you move on. A hash mismatch means you have a different dataset version.

## 4. Build the processed data

Processed CSVs are not committed. Rebuild them from the raw files:

```powershell
python -m src.data.cleaning       # writes data/processed/matches_clean.csv and manifest.json
python -m src.features.builder    # writes data/processed/features.csv and features_manifest.json
git status                        # both manifests should show no changes
```

Both scripts are deterministic. If `git status` shows a changed manifest, compare the `sha256` values: your output differs from the committed version, often because of a different pandas version (see `requirements.txt`).

## 5. Compare the models (Phase 4)

```powershell
python -m src.models.prematch     # writes reports/prematch_models.md (about 10 seconds)
python -m src.models.live         # writes reports/live_models.md and reports/figures/08_*.png (about 1 minute)
git status                        # the reports should show no changes
```

Both scripts use fixed seeds, so the reports are reproducible. Neither script reads the 2023–2024 test seasons.

## 6. Build the model release (Phase 5)

```powershell
python -m src.models.calibration  # reports/calibration_review.md (pre-test data only)
python -m src.models.release      # test evaluation, reports/model_evaluation.md, models/release-1.0.0/
git status                        # models/release-1.0.0/metadata.json should show no changes
```

The release folder holds the saved models and the history snapshot they need. Only `metadata.json` is committed. The other files are rebuilt byte-identically, and `Predictor.load()` refuses to load them if their hashes do not match it. See [model_release.md](model_release.md).

## 7. Run the app (Phase 6)

Terminal 1, from the project root (needs the release from step 6):

```powershell
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --port 8000
```

Check http://127.0.0.1:8000/api/v1/health: it should report `"model_loaded": true`. Interactive API docs are at http://127.0.0.1:8000/api/docs.

Terminal 2:

```powershell
cd app\frontend
npm.cmd install
npm.cmd run dev
```

Open http://localhost:5173/. Vite forwards `/api` to port 8000. See `app/frontend/README.md`.

Every prediction is appended to `logs/predictions.jsonl` (not committed). Set `IPL_LOG_DIR` to use another folder, or `IPL_LOG=off` to turn logging off. Download the log as CSV from the Batch section or http://127.0.0.1:8000/api/v1/predictions/log.csv.

## 8. Verify everything (Phase 8)

To check a fresh copy end to end, run steps 3–6 in order, then the tests and a frontend build:

```powershell
python -m pytest                  # 174 tests, about 20 seconds
git status                        # must show no changes after steps 4-6
cd app\frontend; npm.cmd test     # 7 frontend tests (CSV export, form validation)
npm.cmd run build
```

Steps 3–6 take about 1 minute 45 seconds in total. If `python -m pytest` says "No module named pytest", the virtual environment is not active: activate it, or call `.venv\Scripts\python.exe -m pytest`. Results of the last full check, including prediction timings and the browser checks, are in [reports/verification.md](../reports/verification.md). [demo_script.md](demo_script.md) has a run-through of the app with the expected numbers.

## 9. Optional pre-match accuracy experiments

After preparing the original raw and processed data, run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_performance_features.py tests/test_prematch_experiment.py
.venv\Scripts\python.exe -m src.models.prematch_experiment
```

This compares recent batting/bowling features and training recency with the existing pre-match baseline. It saves `reports/prematch_v2_protocol.json` before fitting, writes the comparison to `reports/prematch_v2_experiment.md` and `.json`, and regenerates `data/processed/prematch_v2_features.csv` and `prematch_v2_predictions.csv`. It uses existing dependencies and does not rebuild or replace the serving release. It takes about 50 seconds, and a rerun writes byte-identical reports. The protocol records hashes of the experiment code, so editing that code changes `prematch_v2_protocol.json`.

The 2023–2024 seasons have already been inspected. The report labels their results exploratory; they are not a new untouched test. Candidate selection uses only 2015–2022 development predictions, and all 16 configurations are reported. See [the experiment report](../reports/prematch_v2_experiment.md) for the selection rule, feature definitions, uncertainty and results.
