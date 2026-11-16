# Model evaluation and release (Phase 5)

Code: `src/models/calibration.py`, `src/models/confidence.py`, `src/models/release.py`, `src/models/predictor.py`.
Results: [reports/calibration_review.md](../reports/calibration_review.md) and [reports/model_evaluation.md](../reports/model_evaluation.md).

## Protocol

1. **Freeze first.** The model selection (Phase 4 rule: lowest walk-forward log loss) and the confidence labels were fixed using only 2015–2022 walk-forward predictions. They were committed (`4c34e4b`) before the test seasons were read.
2. **Test once.** `python -m src.models.release` trains every candidate on 2008–2022 and scores it on 2023–2024. The selected models are the result; the other candidates are shown for transparency, not to re-select. Nothing was tuned after this run.
3. **Release.** The selected models are refitted on 2008–2024 with the same settings and saved as release `1.0.0`. The test metrics describe the 2008–2022 fit; the release adds the two newest seasons for future predictions. `metadata.json` records both.

## Confidence labels

The label depends only on the favourite's probability, max(p, 1 − p):

| Label | Favourite probability | Meaning in the app |
|---|---|---|
| Too close to call | 50–55% | Low-confidence flag (gap below 10 points); shown in text, not colour alone |
| Slight edge | 55–65% | |
| Clear edge | 65–80% | |
| Strong edge | 80% and above | Only the live mode reaches this |

No recalibration was applied. The pre-match model is close to calibrated. The live models are a few points overconfident between about 60% and 75%, but on the test seasons each label's favourite won about as often as predicted (chase: strong edge 89.0% vs 92.1% predicted; clear edge 68.5% vs 73.0%).

## Test results (2023–2024)

| Moment | Walk-forward 2015–2022 | Test 2023–2024 |
|---|---|---|
| Pre-match, after the toss | 54.5% | 48.9% ± 8.3% |
| 1st innings, after 10 overs | 60.4% | 67.4% ± 7.7% |
| Innings break | 63.9% | 64.5% ± 7.9% |
| Chase, after 6 overs | 71.7% | 74.5% ± 7.2% |
| Chase, after 10 overs | 77.3% | 75.5% ± 7.1% |
| Chase, after 15 overs | 82.6% | 84.2% ± 6.2% |

- **Live mode holds up.** Every checkpoint is within its interval of the walk-forward estimate. Chase ROC-AUC on test is 0.83–0.93.
- **Pre-match did not beat a coin flip on the test seasons.** The selected Random Forest scored 48.6% on all 144 test matches (ROC-AUC 0.53, log loss 0.699 vs 0.693 for a coin flip). No pre-match candidate was reliably better than 50%.
- **Why: the chasing advantage reversed.** From 2008 to 2022 the side batting first won 44.9% of matches; in 2023–2024 it won 52.1%, in line with higher first-innings totals since the 2023 rule changes (the par score rose from 165 to 184). The pre-match models had learned to favour the chasing side. In these two seasons `team1` is always the side batting first, so the model predicted Team B in 107 of 144 matches. The released model includes 2023–2024 in training, which partly corrects this, but the pre-match probabilities should be presented as close to 50/50.

## Release contents (`models/release-1.0.0/`)

| File | Contents | In git |
|---|---|---|
| `metadata.json` | Version, training ranges, test metrics, features, estimator settings, confidence bands, input hashes, package versions, file hashes | Yes |
| `prematch.joblib` | Random Forest (trained on both team orders) | No (rebuilt) |
| `live_innings1.joblib`, `live_innings2.joblib` | Logistic regression per innings | No (rebuilt) |
| `history.csv` | The 1,090 matches used for pre-match features (history cutoff 2024-05-26) | No (rebuilt) |
| `first_innings_totals.csv` | Full-length first-innings totals for the par score | No (rebuilt) |

The build is deterministic: rebuilding gives byte-identical files, so `git status` shows no change to `metadata.json`.

## Using a release

```python
from src.models.predictor import Predictor

p = Predictor.load()  # checks every file against metadata.json
p.predict(team1="Mumbai Indians", team2="Chennai Super Kings",
          venue="Wankhede Stadium, Mumbai", season=2025,
          toss_winner="Chennai Super Kings", toss_decision="field")
p.predict_live(team1="Mumbai Indians", team2="Chennai Super Kings",
               venue="Wankhede Stadium, Mumbai", season=2025,
               toss_winner="Chennai Super Kings", toss_decision="field",
               innings=2, overs_done=10, runs=95, wickets=2, first_innings_total=185)
```

Both return the two teams' probabilities, the predicted winner, the confidence label, a `close_call` flag, the history cutoff and the model release. The live result also names the batting-first side and echoes the state. Unknown names raise `UnknownNameError`, an impossible state raises `ValueError`, and a missing or altered release raises `ReleaseError`. A prediction takes about 0.35 seconds locally.
