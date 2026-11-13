# IPL Cricket Match Winner Prediction

> **Project source of truth** — Update this file when scope, decisions, milestones, or implementation status change. Keep it aligned with what the project actually does.

| Item | Current position |
|---|---|
| Students | Hrituraj Roy (PST-25-0047) and Amul Rai (PST-25-0122) |
| Course stage | Semester 3, sections A and B |
| Project type | Data science and machine learning, delivered as a web application |
| Status | Phases 1–3 complete (2026-11-13). React frontend shell built early (Phase 6 UI, not connected to a model). |
| Next milestone | Phase 4: baselines (coin flip, Elo, logistic regression), Random Forest and XGBoost on the time-based split |
| Planning horizon | 12 weeks across 3 months, from 2026-09-27 |
| Schedule | Week 7 of 12 on 2026-11-13. Data/ML work is about 2 weeks behind plan (Phase 4 was due in week 6); see the catch-up note in section 6. |
| Repository | https://github.com/Roy-code5k/IPL_Match_Predictor- (push after every phase) |

## 1. Product summary

The product estimates the probability that either of two IPL teams will win a match. A user selects the teams, venue, season, toss winner, and toss decision. The application returns the predicted winner, a probability for each team, an uncertainty indication, and supporting historical context. It also lets users inspect model performance and team trends.

The prediction is made **after the toss and before the first ball**. It uses only information available at that point and historical matches that occurred before the prediction cutoff. It is a decision-support and learning tool, not a guarantee or a betting product.

### Intended users

- **ML and data science students:** build, compare, and explain reproducible models.
- **Sports data analysts:** inspect historical team, venue, and toss patterns.
- **Coaches, managers, and other domain stakeholders:** review a match-up and compare scenarios.

### Success criteria

- The same valid input and model version always produce the same result.
- Historical features never use the current match result or a future result.
- Model metrics come from a documented, held-out evaluation and are tied to the displayed model version.
- A user can complete the prediction flow locally without a crash.
- Probabilities and uncertainty are visible and explained in plain language.
- Another student can reproduce data preparation, training, evaluation, and local application startup from the documentation.

## 2. Scope and priorities

### Must have

| Area | Feature | Expected behavior |
|---|---|---|
| Data | Load IPL match CSV data | Validate schema and record the dataset source/version. `matches.csv` is required; `deliveries.csv` is optional for exploration. |
| Data | Clean and normalize records | Handle missing/invalid records, remove no-result matches, and standardize team and venue names with documented mappings. |
| Analysis | Team performance notebook | Show season, team, toss, venue, missing-data, and class-balance findings. |
| Features | Historical feature pipeline | Calculate team win rates, head-to-head, venue record, recent form, and toss features using only earlier matches. Use the same definitions for training and inference. |
| Modeling | Baseline, Random Forest, and XGBoost | Train reproducibly, compare on the same evaluation data, and serialize the selected model with its preprocessing and metadata. |
| Evaluation | Model dashboard | Show accuracy, F1, ROC-AUC, a labeled confusion matrix, and model comparison information. |
| Prediction | Match configuration and result | Validate teams, venue, season, toss winner, and decision; show both team probabilities and predicted winner. |
| Trust | Uncertainty indication | Flag close predictions in text as well as visually; document how confidence labels are assigned. |

### Should have

- Version and configuration controls for trained models.
- Bulk scenario predictions from a custom CSV.
- Basic model explanation, such as feature importance or SHAP where appropriate.
- CSV export for single and bulk predictions, including inputs and model version.

### Could have

- PDF evaluation report.
- Multiple languages or theme customization.

### Out of scope for this release

- Live ball-by-ball prediction or live score feeds.
- Player-level prediction features.
- Authentication or multi-user account management.
- Betting or gambling workflows.

## 3. User experience and product behavior

The planned navigation is **Overview → Match configuration → Prediction result → Model evaluation → Team performance analysis → Exports**.

1. The match form offers Team A, Team B, venue, season, toss winner, and toss decision. It rejects duplicate teams and a toss winner outside the selected match-up. Predict is enabled only for complete, valid input.
2. The result shows the predicted winner, both win probabilities, a probability bar, the confidence label, the model version, and a short explanation grounded in the input features.
3. The evaluation view shows the active model's metrics, confusion matrix, and comparison with the baseline and other trained candidates. Metric labels specify the evaluation period.
4. The team analysis view shows descriptive historical trends. It must distinguish observed statistics from model predictions.
5. Forms are keyboard navigable. Errors identify the invalid field and explain how to correct it. Confidence is never communicated by color alone.

The intended application states are `CONFIGURED → FEATURES_READY → PREDICTION_COMPLETE`, followed by `REVIEW_REQUIRED` for a close prediction or `ERROR` if a request fails.

## 4. Data and modeling rules

### Data source and target

The source is the Kaggle **IPL Complete Dataset (2008–2024)** by `patrickb1912` (https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020), built from Cricsheet data. The URL slug still says 2008–2020; the supplied files run to the 2024 final. The version is pinned by SHA-256 hash rather than download date. Details are in [docs/data_source.md](docs/data_source.md), and every column is described in [docs/data_dictionary.md](docs/data_dictionary.md). Raw CSVs are not committed to git. The dataset owner is known to the team, so licensing is not a concern; the final report credits the publisher and Cricsheet.

The cleaned dataset is `data/processed/matches_clean.csv` (1,090 decided matches), produced by `python -m src.data.cleaning` under the rules in [docs/cleaning_rules.md](docs/cleaning_rules.md). EDA results are summarised in [reports/eda_findings.md](reports/eda_findings.md). `matches.csv` is the primary match-level file; `deliveries.csv` is optional for exploration. The binary target is whether Team A wins (`1`) or Team B wins (`0`). Matches without a valid winner are excluded from supervised training and evaluation.

### Raw dataset inventory (initial inspection, 2026-09-28)

| File | Location | Initial findings |
|---|---|---|
| `matches.csv` | `data/raw/matches.csv` | 1,095 matches dated 2008-04-18 through 2024-05-26; 20 columns; no duplicate match IDs; 19 distinct raw team names and 58 distinct raw venue names. |
| `deliveries.csv` | `data/raw/deliveries.csv` | 260,920 delivery rows covering all 1,095 match IDs; no delivery references a missing match. |

In `matches.csv`, five matches have `result = no result` and no winner; 14 ties have a recorded winner. `city` is blank or `NA` in 51 matches. The `season` field mixes calendar years with labels such as `2007/08`, `2009/10`, and `2020/21`, so cleaning must define a canonical season value without changing the raw files. These are inspection findings, not final cleaning rules.

Phase 1 inspection added these findings (full output in [reports/data_quality_report.md](reports/data_quality_report.md), regenerated by `python -m src.data.inspect_raw`):

- Each raw season label covers exactly one calendar year (`2007/08` → 2008, `2009/10` → 2010, `2020/21` → 2020), so the year of `date` is a safe canonical season.
- 272 dates host two matches (544 matches). There is no start time, and match IDs are not in date order, so same-day ordering cannot be recovered from the file.
- `team1` is not consistently the side batting first (795 of 1,095), and it wins 50.9% of decided matches, so the target classes are balanced.
- Toss winner and winner are always one of the two teams, and the toss decision agrees with the actual batting order in all 1,095 matches.
- All 14 ties were decided by super overs and have a winner. 21 results used D/L. All 51 missing `city` values are Dubai or Sharjah venues.
- `target_runs`, `target_overs`, `method`, `result`, `result_margin`, `super_over`, and `player_of_match` are set during or after the match and are excluded as model inputs.

Raw file SHA-256 hashes: `matches.csv` — `8A0394246D2A76B44526565F2245D8F4FEFF91BABE57AB16676A2C3D67AA0846`; `deliveries.csv` — `142236EA52950795DAB266D4AA392C7645BD1A0DB4704F0B8058AFD96E24E2C2`.

Candidate inputs are match date/season, teams, venue/city, toss winner, toss decision, and historical statistics available before the match. Match outcomes and post-match fields—including winning margin and player of the match—must never be model inputs. `deliveries.csv` is exploratory only for this release.

### Leakage prevention

- Parse and order matches chronologically before calculating historical statistics.
- For each match, use only results from strictly earlier matches. Define a safe policy for matches on the same date when time of day is unavailable.
- Fit encoders, imputers, and other learned preprocessing only on training data within each evaluation split.
- Reuse the same feature definitions and name mappings for training and served predictions.
- Record the historical data cutoff in each model artifact and expose it to the application.
- Add focused tests with small, known match histories that would fail if a future result enters a feature.

### Training and evaluation

1. Establish a simple baseline (for example, logistic regression and/or a toss-winner heuristic).
2. Train Random Forest and XGBoost with a fixed seed and recorded package versions.
3. **Time-based split (decided 2026-11-13; no random split).** Proposed periods: train 2008–2021 (872 matches), validate 2022 (74), test 2023–2024 (144). Use validation data for tuning; keep the final test untouched until selection is complete. Confirm the periods at the start of Phase 4.
4. Report accuracy, F1, ROC-AUC, confusion matrix, training time, and performance differences from baseline. Inspect errors by team and season.
5. Review probability calibration before attaching confidence language. A gap below 10 percentage points between the two probabilities is the proposed low-confidence flag; other thresholds need to be defined and documented.
6. Save the chosen estimator, fitted preprocessing, feature specification, historical snapshot/cutoff, metrics, and version as one compatible release set. Retain previous releases for rollback.

## 5. Technical design

| Layer | Responsibility | Planned technology |
|---|---|---|
| Data | Load, validate, clean, and document match data | Python, pandas, NumPy |
| Features | Create leakage-safe historical and categorical features | Python, scikit-learn preprocessing |
| Models | Train, evaluate, version, and load estimators | scikit-learn, XGBoost |
| API | Validate requests and serve saved-model predictions and metrics | FastAPI |
| UI | Collect match input and show results, evaluation, and analysis | React 19 + Vite (`app/frontend/`, built) |
| Persistence | Store model-run metadata and prediction logs after the core flow works | MongoDB |

Planned repository areas are `data/`, `notebooks/`, `src/data/`, `src/features/`, `src/models/`, `app/`, `models/`, `reports/`, `tests/`, and `docs/`. The model is trained offline; the API loads a saved artifact rather than training on a prediction request. Local operation is the default deployment target.

The Python environment uses Python 3.13 with pinned packages in `requirements.txt`. Setup and verification commands are in [docs/setup.md](docs/setup.md).

### Initial API contract

- `POST /api/v1/predict`: accepts `team1`, `team2`, `venue`, `season`, `toss_winner`, `toss_decision`, and optional `match_date`; returns predicted winner, probabilities for both teams, confidence, and model version.
- `GET /api/v1/model-metrics`: returns the active model name/version and its evaluation metrics.
- `GET /api/v1/health`: reports application and model-load status.
- Errors use a stable code, clear message, and request ID. Invalid teams or inconsistent toss input are rejected before inference.

MongoDB is in the proposed stack for `ModelRun` and `PredictionLog` records. The approved context also allows flat files for the first working version. Prediction serving must use its versioned local model artifact; the exact database requirement for the final demonstration is an open decision below.

## 6. Implementation plan

| Phase | Time | Steps | Exit criteria / deliverables |
|---|---|---|---|
| **1. Setup and data inspection** | Week 1 | Set up reproducible environments and commands; obtain and version the approved CSV; inspect schema, date range, outcomes, missing data, and team/venue variants; draft the data dictionary. | Raw source recorded; initial quality report and data dictionary drafted. |
| **2. Cleaning and EDA** | Weeks 2–4 | Implement documented cleaning and name normalization; handle invalid/no-result records; analyze season counts, team wins, toss effects, venue trends, missingness, and class balance; write findings and limitations. | Cleaned dataset, reproducible cleaning process, EDA notebook, and findings summary. **Month 1 checkpoint.** |
| **3. Historical features** | Week 5 | Define chronological ordering and same-day policy; build overall, head-to-head, venue, recent-form, and toss features; align training and inference behavior; test for leakage and input-order issues. | Reusable feature builder and passing chronology tests. |
| **4. Model training** | Week 6 | Establish baseline; train Random Forest and XGBoost using recorded seeds/configuration; separate train, validation, and future-period test data. | Reproducible candidate model runs. |
| **5. Evaluation and release artifact** | Weeks 7–8 | Tune on validation data; evaluate once on held-out data; review errors and calibration; compare candidates; select and serialize the model, pipeline, historical snapshot, and metadata. | Model comparison report and loadable versioned artifact. **Month 2 checkpoint.** |
| **6. API and core UI** | Weeks 9–10 | Add FastAPI routes, input validation, saved-model loading, React match form, result view, evaluation view, and team analysis view. | End-to-end deterministic prediction with model metrics visible. |
| **7. Persistence and polish** | Week 11 | Add MongoDB metadata/logging as agreed; implement CSV export and remaining Must-have loading behavior; improve errors, accessibility, and close-prediction display. Attempt Should-have work only after the core flow passes. | Must-have flows integrated and checked. |
| **8. Final verification and handoff** | Week 12 | Run unit, integration, model, and UI checks; verify prediction latency target of under one second locally; finish setup/architecture/model-limitations documentation and final report; rehearse the demo. | Reproducible, demo-ready application and report. **Month 3 checkpoint.** |

### Progress by phase

| Phase | Status | Evidence |
|---|---|---|
| 1. Setup and data inspection | ✅ Complete 2026-11-13 | `requirements.txt`, `docs/setup.md`, `docs/data_source.md`, `docs/data_dictionary.md`, `src/data/inspect_raw.py`, `reports/data_quality_report.md`, `tests/test_raw_data.py` (5 passing) |
| 2. Cleaning and EDA | ✅ Complete 2026-11-13 | `src/data/mappings.py`, `src/data/cleaning.py`, `docs/cleaning_rules.md`, `reports/cleaning_report.md`, `data/processed/manifest.json`, `notebooks/01_eda.ipynb`, `reports/figures/`, `reports/eda_findings.md`, `tests/test_cleaning.py` (22 tests passing in total) |
| 3. Historical features | ✅ Complete 2026-11-13 | `src/features/builder.py`, `docs/features.md`, `data/processed/features_manifest.json`, `HOME_CITIES` in `src/data/mappings.py`, `tests/test_features.py` (36 tests passing in total) |
| 4–5. Training, evaluation | Not started | — |
| 6. API and core UI | UI shell done; API not started | `app/frontend/` validates the match form and shows "ML model not connected"; `app/backend/` is empty |
| 7–8. Persistence, polish, handoff | Not started | — |

**Catch-up note:** with 6 weeks left, prioritise the model path (Phases 4–5) and train a first baseline model next. The frontend should not grow further until a model artifact exists. Single-feature AUCs are only 0.52–0.55 (see `docs/features.md`), so expect modest accuracy. Compare every model against the coin-flip and Elo baselines, and consider team-swap augmentation.

### Final definition of done

- Every Must-have feature above works from the documented local setup.
- Training and inference share the same leakage-safe feature logic.
- The active model's displayed metrics match its saved evaluation report and version.
- Baseline, Random Forest, and XGBoost are compared honestly on the same held-out period.
- Invalid input and model-load failures produce clear responses rather than crashes.
- Project documentation explains dataset coverage, feature timing, model limitations, and how to reproduce results.
- Both students can explain the pipeline and demonstrate a prediction end to end.

## 7. Open decisions and risks

| Decision or risk | Why it matters | Resolve by |
|---|---|---|
| ~~**Dataset provenance.**~~ **Resolved 2026-11-13:** source and hash-pinned version in `docs/data_source.md`; owner known to the team, licence not a concern. | Reproducibility and attribution require an identifiable source. | Done |
| **README vs PROJECT.md: UI stack.** The split is resolved (time-based; README updated). **Still open:** the README describes a Streamlit app, while this file specifies React + FastAPI and the React frontend is already built. Agree on one, then update whichever document is wrong. | Assessors read the README first. | Before Phase 6 |
| ~~**Prediction date and historical cutoff.**~~ **Resolved in Phase 3:** the API accepts an optional `match_date`. With a date, history stops before it; without one, a known season uses the pre-season view (history before its first match) and a future season uses all data. See `resolve_as_of` in `docs/features.md`. | Season alone is insufficient to build strictly prior-match features for an in-season historical scenario. | Done |
| ~~**Season normalization.**~~ **Resolved in Phase 2:** `season` = year of `date`; raw label kept as `season_raw`. | The UI and chronological analysis need consistent seasons. | Done |
| ~~**Same-day match ordering.**~~ **Resolved in Phase 3:** features use only matches on strictly earlier dates, so neither same-day match sees the other (272 dates affected). Covered by tests. | An arbitrary order could leak a same-day result. | Done |
| ~~**Team and venue mappings.**~~ **Resolved in Phase 2:** 19 raw team names → 15 franchises (renames only; matches the frontend) and 58 venue labels → 36 grounds (renamed grounds merged), all in `src/data/mappings.py`. The feature builder and API must import these mappings. | Incorrect merging changes historical rates. | Done |
| **Supported seasons and venues.** Define accepted prediction inputs and behavior for categories absent from training. | Prevents invalid or misleading scenarios. | Before Phase 6 |
| **MongoDB requirement.** Confirm whether final assessment requires database-backed logs or only the documented architecture. | Affects Week 11 scope and local setup. | Before Phase 7 |
| **Confidence wording and thresholds.** Set labels after reviewing calibration and state that probability is an estimate. | Prevents overconfident presentation. | Phase 5 |

## 8. Status and change log

- **2026-09-27 — Planning:** Consolidated the approved project context and implementation phases into this source-of-truth file. No implementation has started. Update the status table, open decisions, and milestones as work progresses.
- **2026-09-28 — Data received:** Moved the supplied CSVs into `data/raw/` without altering their contents. Initial inspection found coverage through 2024 and recorded the inventory and data-quality questions above. No cleaning or application code has started.
- **2026-09-30 — Cleaning stub:** Added `src/data/cleaning.py`, which loads `matches.csv` and prints its shape and columns. No cleaning logic yet.
- **2026-10-13 — Frontend shell:** Built the React + Vite frontend in `app/frontend/` (desktop-style UI, hand-calculated historical summaries, validated match form). It deliberately shows "ML model not connected" and calls no API.
- **2026-11-13 — Phase 1 complete:** Initialised git on top of the existing GitHub history. Repaired the virtual environment and pinned packages in `requirements.txt`. Recorded the Kaggle source and hash-pinned version. Wrote the data dictionary and setup guide. Added a reproducible inspection script, report, and tests (5 passing). Recorded recommendations for season and same-day policies, and flagged the README conflict (Streamlit vs React + FastAPI).
- **2026-11-13 — Decisions:** Evaluation uses a time-based split (README updated to match). Dataset licence is not a concern because the owner is known to the team.
- **2026-11-13 — Phase 2 complete:** Added shared team and venue mappings (`src/data/mappings.py`) and a deterministic cleaning pipeline that writes `data/processed/matches_clean.csv` (1,090 matches), a hash manifest, and a cleaning report. Wrote `docs/cleaning_rules.md`. Added the executed EDA notebook (`notebooks/01_eda.ipynb`), 7 charts in `reports/figures/`, and `reports/eda_findings.md`. Key results: the toss winner wins 50.8% (coin-flip baseline); fielding first wins 53.9%; team strength swings strongly between seasons. Proposed split: train 2008–2021, validate 2022, test 2023–2024. 22 tests passing.
- **2026-11-13 — Phase 3 complete:** Added the leakage-safe feature builder (`src/features/builder.py`). It produces 38 features: Elo with season regression, overall and recent form, season-to-date, head-to-head, venue record, home ground, toss and batting order, and venue/league chase rates, all smoothed toward 0.5 for small samples. Training and inference share `match_features`. Features use only strictly earlier dates (resolves same-day ordering), and `resolve_as_of` defines the prediction cutoff (resolves the cutoff decision; the API gains an optional `match_date`). Added `HOME_CITIES` to the mappings, `docs/features.md`, a hash manifest, and 14 leakage/chronology/symmetry tests (36 tests passing). Single-feature AUCs are 0.52–0.55, so pre-match signal is weak.
