# Final verification (Phase 8)

Run on 2026-11-17 on Windows 11, Python 3.13.14 (project `.venv`), Node.js 25.2.1, and Microsoft Edge (headless, driven by Playwright). Everything was run from the documented setup in [docs/setup.md](../docs/setup.md).

The first check ran on the Phase 8 code. Later the same day, review corrections changed the code:

- retrospective labelling;
- the rule that a chase at 0 overs must have 0 runs and 0 wickets;
- stricter batch CSV parsing and spreadsheet-safe exports.

A follow-up pre-match experiment was also added. The checks were then run again on the changed code. Unless a section says otherwise, this report gives the results of that second run.

**Result: every check passed.**

- 174 Python tests and 7 frontend tests pass.
- A full rebuild from the raw CSVs reproduces every committed file byte for byte, and the experiment reruns byte for byte.
- A prediction takes 33–80 ms depending on machine load, against a one-second target. The changes did not slow it down.
- The app works end to end in the browser, including the new retrospective notice, with no unexpected console errors.

The only open item is the team's demo rehearsal (section 7).

## 1. Automated tests

`python -m pytest`: **174 passed** in 15–60 s depending on machine load (105 test functions; some run several cases). The first check had 109, and the second had 168, before the empty-CSV tests in section 9 were added.

| File | Functions | Cases | Covers |
|---|---|---|---|
| `tests/test_raw_data.py` | 4 | 5 | Raw file hashes, schema, row counts |
| `tests/test_cleaning.py` | 12 | 17 | Cleaning rules, team and venue mappings, no-result removal |
| `tests/test_features.py` | 15 | 15 | Hand-checked feature values; leakage (a changed result alters nothing on or before its date); same-day independence; row-order independence; training rows equal inference rows; team-swap symmetry |
| `tests/test_frontend_data.py` | 3 | 3 | The app's team, venue and count data match the Python mappings |
| `tests/test_live.py` | 10 | 18 | Simulated ball-by-ball states, retired hurt, rain exclusion, par-score leakage, training rows equal inference rows; a 0-over chase with runs or wickets is rejected |
| `tests/test_models.py` | 11 | 19 | Season split, metrics, symmetric predictions, deterministic training, confidence labels |
| `tests/test_predictor.py` | 8 | 13 | Release loading, refusal of a tampered or missing release, order-invariant predictions, the history cache gives identical results; the retrospective flag one day before, on and after the training cutoff |
| `tests/test_api.py` | 26 | 63 | Every route; validation and error codes with request ids; missing release (503 and degraded health); batch good and bad rows; malformed CSVs (blank or duplicate headers, wrong cell counts, broken quoting); empty or blank-only files; a header after blank lines; row limit; logging and CSV export with formula-like text escaped |
| `tests/test_performance_features.py` | 7 | 7 | Batting and bowling form features: legal balls, wickets and phase boundaries; same-day and future deliveries cannot change earlier features; ineligible matches excluded; finite rates without history; training rows equal inference rows in both team orders |
| `tests/test_prematch_experiment.py` | 9 | 14 | Training windows exclude evaluation and future seasons; the base candidate reproduces the released model's settings; team symmetry and weights; selection by development accuracy, then log loss; paired comparison; stable JSON output |

`npm test` in `app/frontend`: **7 passed** (CSV export escaping and quoting; form validation, including the 0-over chase rule). The production build (`npm run build`) succeeds: 265 kB of JavaScript (82 kB gzipped).

## 2. Reproducibility

Each pipeline step was rerun in order from `data/raw/`. Every step exited cleanly, and `git status` showed no changes afterwards. The SHA-256 of every file that git does not track (processed data, saved models, history snapshots) was the same before and after.

| Step | Time |
|---|---|
| `src.data.inspect_raw` | 3 s |
| `src.data.cleaning` | 2 s |
| `src.features.builder` | 4 s |
| `src.models.prematch` | 16 s |
| `src.models.live` | 30 s |
| `src.models.calibration` | 33 s |
| `src.models.release` | 19 s |
| **Total** | **about 1 minute 50 seconds** |
| `src.models.prematch_experiment` (optional) | 48 s |

So every report, figure, manifest and `models/release-1.0.0/metadata.json` came out identical to the committed versions. The review corrections changed nothing the pipeline produces.

The experiment's Markdown report, protocol, features and predictions were also identical on the rerun. Its `prematch_v2_experiment.json` was not: a few metrics differed in the 16th significant digit, because the parallel Random Forest adds up its trees in a varying order. The JSON writer now rounds floats to 12 significant digits, and two further reruns gave identical bytes. That fix is in the experiment code, so the code hash recorded in `prematch_v2_protocol.json` changed. No protocol setting changed.

## 3. Training time

These times come from `src.models.prematch` and `src.models.live`. Each covers the whole walk-forward comparison: 8 refits, one per season from 2015 to 2022.

| Model | Pre-match | Live, 1st innings | Live, chase |
|---|---|---|---|
| Coin flip / Elo only | 0.1 s | 0.1 s | 0.0 s |
| Logistic regression | 0.2 s | 0.2 s | 0.1 s |
| Random Forest | 4.7 s | 4.0 s | 3.9 s |
| XGBoost | 3.4 s | 3.9 s | 1.6 s |

The selected models are therefore cheap to retrain. Logistic regression, used for live predictions, trains in under a quarter of a second.

## 4. Prediction latency (target: under one second)

These requests went over HTTP to the running API, with logging on. The server loads the release once at startup, which takes about 6.5 s including Python imports and the hash checks.

| Request | n | Median | 95th percentile | Max |
|---|---|---|---|---|
| Pre-match, a new match date each time (cache miss) | 25 | 32.3 ms | 34.4 ms | 54.9 ms |
| Pre-match, same request repeated | 30 | 32.7 ms | 34.6 ms | 35.4 ms |
| Pre-match, future season 2026 | 30 | 33.5 ms | 35.2 ms | 37.3 ms |
| Live, both innings, overs 1–19 | 38 | 3.9 ms | 4.6 ms | 5.1 ms |
| Batch, 200 mixed rows in one request | 1 | 3,415 ms in total (about 17 ms per row) | | |

**The target is met:** the slowest single prediction took 55 ms. All 200 batch rows succeeded.

**Rerun after the review corrections.** The machine was busier during the rerun: a repeated pre-match request took a median of 80 ms (max 180 ms), a live request 9.5 ms, and a 200-row batch 8.7 s. To check whether the code changes caused this, the last committed code ran on port 8001 next to the changed code on port 8000. Both used the same release, and the requests alternated between them (40 each):

| Request | Changed code (median) | Committed code (median) |
|---|---|---|
| Pre-match, 2026 | 80.1 ms | 78.8 ms |
| Live chase, 95/2 after 10 overs | 10.1 ms | 9.9 ms |

The changes add no measurable time; the slowdown came from the machine. Predictions dated inside the data seasons each build a new history snapshot the first time a date is used. Those took about 0.6 s on the busier machine, still under the target.

## 5. Browser checks (Edge, headless)

| Check | Result |
|---|---|
| Pre-match prediction (KKR v SRH, 2024 final date): both probabilities, bar, confidence label, model-inputs panel, single-result CSV download (button removed from the app afterwards, see PROJECT.md) | Pass |
| Live prediction (chase, 95/2 after 10 overs, target 186): 77.5% to the chasing side, "Clear edge", target, runs needed and required rate | Pass |
| Validation: teams outside the season are disabled; the same team twice shows "Choose a different franchise." on Team B | Pass |
| Empty submit marks the 6 required fields as invalid (`aria-invalid`, message linked by `aria-describedby`) | Pass |
| Keyboard only: every select filled by typing, Tab reaches Predict (focus outline visible), Enter submits and the result appears | Pass |
| Batch: template download; an upload with 2 good rows (one live) and 2 bad rows (a team from the wrong season, the same team twice) gives results with an error column; results CSV download | Pass |
| Prediction log CSV export through the app | Pass |
| Model Performance: metric cards, confusion matrix, accuracy by moment, comparison tables, all read from the release | Pass |
| Dark theme at phone width (390 px): readable text, no sideways scrolling (the dark theme was removed afterwards; a recheck with no toggle showed no sideways scrolling at 390 px) | Pass |
| API stopped: badges switch to offline, and Predict shows "Prediction service offline" with the start command; no crash | Pass |
| Browser console errors across all runs | None |

Rerun on the changed code (19 checks, all passed):

| Check | Result |
|---|---|
| Demo pre-match scenario (2026, MI v CSK): CSK 52.8%, "Too close to call", no retrospective notice; the note shows "Model trained through 2024-05-26" | Pass |
| Pre-match on the 2024 final date (KKR v SRH): "Retrospective simulation" notice | Pass |
| Single-result CSV download: 24 columns, `model_trained_through` = 2024-05-26 and `retrospective` = true (button removed from the app afterwards) | Pass |
| Chase at 0 overs with 5 runs: blocked in the form ("At the innings break (0 overs), runs must be 0.", `aria-invalid`), no request sent | Pass |
| Demo live chase states: 56.3% (0/0), 77.5% (95/2 after 10), 84.9% to MI (120/6 after 15), 50.3% (160/5 after 18), none retrospective | Pass |
| Batch with a 2023 row, a 2025 row and a row whose team is `=1+1`: 2 predicted, 1 error; the 2023 row is marked "Retrospective simulation" and a notice explains it | Pass |
| Batch results CSV: `=1+1` written as `'=1+1`; `retrospective` true and false rows | Pass |
| Batch with an unquoted comma in a venue: "Batch not run", "has 7 cells; the header has 6" | Pass |
| Dark theme at phone width (390 px) with the retrospective notice: no sideways scrolling (dark theme removed afterwards) | Pass |
| Browser console errors | Only the browser's network log of the deliberate 400 from the malformed batch |

Figure 10 (`reports/figures/10_app_live_prediction.png`) was retaken with the same scenario, because the result note's wording changed. Figures 11 and 12 show screens whose text did not change.

The browser driver itself reports a `TargetClosedError` when it closes after downloads. This comes from Playwright, not from the app.

## 6. Definition of done (from PROJECT.md)

| Criterion | Status | Evidence |
|---|---|---|
| Every Must-have feature works from the documented local setup | ✅ | Section 8 below; sections 1, 2 and 5 |
| Training and inference share the same leakage-safe feature logic | ✅ | `match_features` and `state_features`; "training rows equal inference rows" tests |
| The displayed metrics match the saved evaluation report and version | ✅ | The app reads `/model-metrics` from `metadata.json`; the release rebuild matches the committed file and `reports/model_evaluation.md` |
| Baseline, Random Forest and XGBoost compared honestly on the same held-out period | ✅ | `reports/model_evaluation.md`: every candidate on 2023–2024, with the selection frozen beforehand (commit `4c34e4b`) |
| Invalid input and model-load failures give clear responses, not crashes | ✅ | API tests for every error code and a missing release; offline check in the browser |
| Documentation covers dataset coverage, feature timing, model limitations and reproduction | ✅ | `docs/data_source.md`, `docs/features.md`, `docs/model_limitations.md`, `docs/setup.md`, `docs/architecture.md` |
| Both students can explain the pipeline and demonstrate a prediction end to end | ⏳ | `docs/demo_script.md`; needs a rehearsal by the team |

## 7. Open items

- **Demo rehearsal** by both students (section 6).
- **MongoDB: resolved 2026-11-17.** The team decided the file log (`logs/predictions.jsonl`, behind a storage interface) is enough for now.

## 8. Must-have features

| Feature | Where to see it |
|---|---|
| Load and validate the CSVs, with source and version recorded | `src.data.inspect_raw`, `docs/data_source.md` |
| Clean and normalise records | `src.data.cleaning`, `docs/cleaning_rules.md` |
| Team performance notebook | `notebooks/01_eda.ipynb`, `reports/eda_findings.md`; the app's Findings section |
| Historical feature pipeline | `src.features.builder`, `docs/features.md` |
| Baseline, Random Forest and XGBoost | `reports/prematch_models.md`, `reports/model_evaluation.md` |
| Model dashboard: accuracy, F1, ROC-AUC, labelled confusion matrix, comparison | The app's Model Performance section |
| Match form and result: validation, both probabilities, predicted winner | The app's Match Predictor section |
| Uncertainty shown in text as well as visually, with documented labels | Confidence label and close-call notice; `docs/model_release.md` |

## 9. Pre-push check (2026-11-17, evening)

This check ran after the night-mode toggle and the single-result "Download CSV" button were removed, before the code was pushed.

| Check | Result |
|---|---|
| `python -m pytest` | 174 passed |
| `npm test` and `npm run build` | 7 passed; build succeeds |
| Rebuild of all 8 steps from the raw CSVs, including the experiment (2 min 19 s) | All 36 output files byte-identical; no tracked report, figure, manifest or `metadata.json` changed |
| EDA notebook executed from scratch | 8 code cells, 0 errors; figures unchanged |
| Installed packages against `requirements.txt` | All pinned versions match; `pip check` clean |
| HTTP checks against the running API | 48 of 48: every route, every error code with its request id, the demo numbers, the 0-over chase rule, batch edge cases, formula escaping in exports, log export, a 200-row batch (3.3 s), and latency (median 32 ms pre-match and 4 ms live; slowest 36 ms) |
| Browser walkthrough in Edge | 49 of 49: every section and nav link, required-field messages, all demo-script numbers (pre-match and the 7 live states), the retrospective notice, keyboard-only use, batch template, upload, errors and downloads, model performance figures, phone width, and the API offline |
| Repository hygiene | No whitespace errors or conflict markers in the diff; no secrets, personal paths or emails; no tool-attribution text; every Markdown link resolves |

**One problem found and fixed.** An empty batch file was rejected with a technical message ("csv: String should have at least 1 character"). A file containing only blank lines got a misleading one ("Every CSV column needs a non-empty header."). The CSV reader now takes the first non-blank row as the header, so both cases return `INVALID_CSV` with "The CSV is empty. It needs a header row.", and a header after blank lines is accepted. Six API tests cover this (`test_batch_empty_file_says_empty`, `test_batch_header_after_blank_lines_is_accepted`).
