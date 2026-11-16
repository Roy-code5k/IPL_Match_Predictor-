# Live (in-match) win probability (v1.0.0)

Added 2026-11-16 as a second prediction mode next to the pre-match model.
Code: `src/features/live.py` (states and features) and `src/models/live.py` (training and evaluation).
Results: [reports/live_models.md](../reports/live_models.md).

## Why a second mode

Before the first ball, IPL results are close to a coin flip: the best pre-match model reaches about 54% walk-forward accuracy (ROC-AUC 0.55, see [reports/prematch_models.md](../reports/prematch_models.md)).
Once the match is under way, the score, wickets and target carry far more signal.
The live mode shows how the win probability moves through a match. Its accuracy is always reported **per prediction moment**, never as one headline number, so it is not confused with the pre-match accuracy.

## Prediction moments and target

- A prediction is made at the end of a completed over, or at the innings break.
- Output: P(the side batting first wins). The chasing side's probability is 1 minus this.
- In-progress states only:
  - innings 1 after 1–19 completed overs (before that, use the pre-match model);
  - innings 2 at the break (0 overs, target known) and after 1–19 completed overs, while runs are still needed and wickets remain.

## Data rules

- Source: `deliveries.csv` (innings 1 and 2; super overs ignored) joined to `matches_clean.csv`. Team names go through `src/data/mappings.py`.
- Batting order is taken from the toss and checked against the ball-by-ball data; a mismatch raises an error.
- **Rain-affected matches are left out** (D/L, or a target set over fewer than 20 overs: 29 of the 1,090 decided matches). Their target is not "first-innings total + 1 in 20 overs".
- A `retired hurt` dismissal does not use up a wicket.
- 1,061 full-length decided matches remain (920 before the test seasons).

## Features

Context in every state (all from strictly earlier dates):

| Column | Definition |
|---|---|
| `par_score` | Smoothed mean first-innings total of the previous 60 full-length matches: `(sum + 10 × 160) / (n + 10)`. Tracks league scoring, which rose from about 165 in 2022 to 184 in 2024. |
| `elo_prob_bf` | Pre-match Elo probability of the batting-first side (from `src.features.builder`) |
| `venue_chase_rate` | Smoothed share of matches won by the chasing side at the ground (from `src.features.builder`) |

Innings 1: `overs_done`, `balls_left`, `runs`, `wickets`, `run_rate`, `projected_minus_par` (run rate × 20 − par).

Innings 2: `overs_done`, `balls_left`, `target`, `runs_needed`, `wickets_left`, `required_rate`, `target_minus_par`.

Training and inference share `state_features`. Training (`build_live_states`) walks matches date by date and adds a date's first-innings totals to the par history only after that date. Inference (`live_scenario`) gets the pre-match context from `scenario_features` and the par score from totals strictly before the same cutoff.

## Models and evaluation

- One model per innings. Candidates: pre-match Elo only (ignores the match state), logistic regression, Random Forest, XGBoost. Hyperparameters were fixed before results were seen.
- Same views and selection rule as the pre-match models (`src/models/evaluation.py`): walk-forward 2015–2022 and validation 2022; the lowest walk-forward log loss over all in-progress states wins.
- Checkpoint accuracy counts only matches still in progress at that checkpoint.
- The 2023–2024 test seasons have not been used.

Selected: logistic regression for both innings. Walk-forward 2015–2022 accuracy:

| Moment | Accuracy | ROC-AUC |
|---|---|---|
| Pre-match (selected pre-match model, same matches) | 54.5% | 0.53 |
| 1st innings, after 10 overs | 60.4% | 0.65 |
| Innings break | 63.9% | 0.73 |
| Chase, after 6 overs | 71.7% | 0.82 |
| Chase, after 10 overs | 77.3% | 0.85 |
| Chase, after 15 overs | 82.6% | 0.92 |

## Tests (`tests/test_live.py`)

Synthetic matches with simulated ball-by-ball data:
- hand-checked feature values, and rejection of finished or invalid states;
- per-over runs and wickets match a manual count, and `retired hurt` is not a wicket;
- every state is in progress and labelled with the real winner;
- excluded (rain-affected) matches produce no states;
- leakage: changing one match's first-innings total changes no other state on that date or earlier;
- inference rows equal training rows for the same match and cutoff.
