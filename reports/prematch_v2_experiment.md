# Pre-match v2 experiment: performance features and recent training

Reproduce with `python -m src.models.prematch_experiment`. Protocol: `reports/prematch_v2_protocol.json`.
Detailed metrics: `reports/prematch_v2_experiment.json`. Per-match probabilities: `data/processed/prematch_v2_predictions.csv` (regenerated, not tracked).

## Result

- Selected on **2015-2022 development predictions only**: `random_forest/base/last_5`.
- Development accuracy: incumbent **54.2%**, selected **55.0%**.
- Exploratory 2023-2024 accuracy: incumbent **48.6%**, selected **51.4%**, on the same 144 matches.
- Exploratory change: **+2.78 percentage points**. Paired match-bootstrap 95% interval: **-5.56 to +11.81 points** (22 newly correct, 18 newly wrong).
- Exploratory log loss: incumbent **0.6987**, selected **0.6940**; lower is better.
- **No production promotion.** The existing release, API, dashboard metrics and original evaluation report remain unchanged. A newer untouched season is needed before claiming a validated improvement.

## Interpretation and protocol

2023-2024 were already inspected before this experiment was designed. Their results are an exploratory comparison, **not a fresh unbiased test**. The bootstrap interval is descriptive and does not account for selecting among candidates or season-level dependence. Development scores also contain selection optimism. Do not quote the best 2023-2024 row as a separately validated improvement.

The 16 settings were fixed before the run: Random Forest and logistic regression with their existing hyperparameters; original 38 features or 64 features including performance; all earlier seasons, last 3, last 5, or exponential weighting with a 2-year half-life. Accuracy on pooled development predictions selects the experiment candidate; lower log loss breaks ties. This accuracy-first experiment rule does not change the original release's log-loss selection rule.

Development predicts each season 2015-2022 using a fit on earlier seasons. Exploratory evaluation fits once before 2023 and predicts both 2023 and 2024, matching the incumbent's evaluation protocol. Historical features may incorporate completed earlier dates within evaluation seasons, but model weights are not refitted there. Same-day outcomes and deliveries are always excluded from features.

## New information

The additional 26 columns describe the last 10 eligible matches per team: overall batting run rate and bowling runs conceded per over, wickets lost/taken per 120 legal balls, powerplay (overs 1-6) and death-over (16-20) run rates. Each of eight metrics has Team A, Team B and difference columns; two columns record how much recent history is available.

Rates are pooled by legal-ball exposure and smoothed with earlier league form (last 60 team innings). Wides/no-balls add runs but no legal balls. Retired hurt is not a wicket; run-outs are team wickets. Bowling rates include all team runs conceded, rather than individual bowler economy. Super overs and rain-shortened/D/L matches do not enter the performance history. They can still be target matches for pre-match prediction. Missing death-over exposure uses the past-only prior.

Training windows restrict fitted examples, not the historical information available to create a feature. Recency weights are normalised to mean one, duplicated with team-swap augmentation, and applied to both the logistic scaler and classifier. All predictions average the two team orders.

## All candidates

| Candidate / features / training policy | Dev accuracy | Dev log loss | 2023-24 accuracy | 2023-24 log loss | 2023-24 AUC |
|---|---|---|---|---|---|
| random_forest/base/all | 54.2% | 0.6900 | 48.6% | 0.6987 | 0.525 |
| random_forest/base/last_3 | 54.4% | 0.6916 | 55.6% | 0.6970 | 0.527 |
| random_forest/base/last_5 **selected** | 55.0% | 0.6923 | 51.4% | 0.6940 | 0.544 |
| random_forest/base/decay_2 | 53.8% | 0.6932 | 47.9% | 0.6959 | 0.521 |
| random_forest/performance/all | 53.8% | 0.6921 | 54.2% | 0.6906 | 0.555 |
| random_forest/performance/last_3 | 53.4% | 0.6914 | 51.4% | 0.6956 | 0.526 |
| random_forest/performance/last_5 | 53.0% | 0.6915 | 58.3% | 0.6884 | 0.568 |
| random_forest/performance/decay_2 | 53.8% | 0.6901 | 47.2% | 0.6949 | 0.503 |
| logistic_regression/base/all | 52.6% | 0.6976 | 51.4% | 0.7143 | 0.541 |
| logistic_regression/base/last_3 | 50.7% | 0.8729 | 52.1% | 0.8239 | 0.528 |
| logistic_regression/base/last_5 | 53.0% | 0.7120 | 51.4% | 0.7381 | 0.514 |
| logistic_regression/base/decay_2 | 53.0% | 0.7075 | 50.7% | 0.7276 | 0.521 |
| logistic_regression/performance/all | 53.8% | 0.7095 | 49.3% | 0.7150 | 0.547 |
| logistic_regression/performance/last_3 | 50.3% | 0.8276 | 49.3% | 0.8252 | 0.513 |
| logistic_regression/performance/last_5 | 51.5% | 0.7249 | 51.4% | 0.7882 | 0.488 |
| logistic_regression/performance/decay_2 | 53.8% | 0.7173 | 50.0% | 0.7509 | 0.498 |

## Reference baselines

| Period | Baseline | Accuracy | Log loss |
|---|---|---|---|
| development | coin_flip | 47.4% | 0.6931 |
| development | elo | 51.1% | 0.7017 |
| exploratory | coin_flip | 52.1% | 0.6931 |
| exploratory | elo | 47.9% | 0.7050 |

The 50/50 baseline always picks Team A at an exact tie; its accuracy is Team A's win share.

## Selected candidate by season

| Season | Use | Matches | Incumbent accuracy | Selected accuracy |
|---|---|---|---|---|
| 2015 | development | 57 | 52.6% | 56.1% |
| 2016 | development | 60 | 51.7% | 56.7% |
| 2017 | development | 59 | 57.6% | 54.2% |
| 2018 | development | 60 | 60.0% | 61.7% |
| 2019 | development | 59 | 54.2% | 57.6% |
| 2020 | development | 60 | 46.7% | 45.0% |
| 2021 | development | 60 | 60.0% | 60.0% |
| 2022 | development | 74 | 51.4% | 50.0% |
| 2023 | exploratory | 73 | 46.6% | 49.3% |
| 2024 | exploratory | 71 | 50.7% | 53.5% |

The model selection was not changed after inspecting this table. No newer data, playing-XI inputs, hyperparameter search, or serving changes are included in this experiment.
