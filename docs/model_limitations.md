# Model limitations

What the models can and cannot tell you, with the evidence for each point. Every number comes from [reports/model_evaluation.md](../reports/model_evaluation.md) (test seasons 2023–2024) or [reports/prematch_models.md](../reports/prematch_models.md) and [reports/live_models.md](../reports/live_models.md) (walk-forward 2015–2022).

## 1. Pre-match predictions are close to a coin flip

- **Walk-forward 2015–2022:** the best pre-match model (Random Forest) was right 54.2% of the time (ROC-AUC 0.55).
- **Test seasons 2023–2024:** it was right 48.6% of the time (ROC-AUC 0.525). Its log loss was 0.699, against 0.693 for always saying 50%.
- **No candidate won reliably.** Every pre-match model, including logistic regression and XGBoost, scored within the ±8-point uncertainty of 50%.
- **What this means:** before the first ball, this data cannot pick the winner much better than chance. The app shows pre-match probabilities with a note to read them as close to 50/50. On the test seasons no pre-match prediction went above "Slight edge".

## 2. The game changed in 2023, and history did not warn the model

- **The chasing side used to win more often.** From 2008 to 2022 it won 55.1% of matches, and the models learned to favour it.
- **In 2023–2024 the side batting first won 52.1%.**
  - First-innings scores rose sharply: the par score went from about 165 to 184.
  - This coincides with the Impact Player rule, which started in 2023.
- **In those two seasons, Team A in the data always batted first.** The pre-match model therefore picked Team B in 107 of 144 matches.
- **The released model has partly adapted.** It was refitted on 2008–2024, so it has seen the new pattern. But nothing has tested it on a season after 2024.
- **Any future rule or format change can break learned patterns the same way.**

## 3. The samples are small

- **The test set has 144 matches.** A 95% interval on accuracy is about ±8 points, so differences of a few points between models are noise.
- **The 2022 validation season has 74 matches.** That is why models were selected on walk-forward results over 2015–2022 (about 480 matches) rather than on 2022 alone.
- **New franchises have little history.** Gujarat Titans and Lucknow Super Giants joined in 2022.
- **Some grounds were rarely used.** Their rates are smoothed toward 50%, so they carry little information.

## 4. Much of what decides a match is not in the data

The dataset has teams, grounds, the toss and the score. It has nothing on:

- players, playing XIs, injuries or auction changes;
- pitch reports, weather or dew;
- player form.

This is the main reason pre-match accuracy is low. The features describe the franchise, not the eleven players on the day.

## 5. Live-mode limits

- **Over ends only.** Predictions are made at the end of a completed over or at the innings break, not ball by ball. In the first innings at least one over must be complete; before that, use the pre-match prediction.
- **The innings break means 0/0.** In the chase, `overs_done = 0` requires zero runs and zero wickets. Scores partway through the first over are not supported.
- **The user types the score.** There is no live feed, so a mistyped score gives a wrong prediction. The app rejects states that are impossible or already decided, such as a chase that has passed the target or 10 wickets down, but not merely unlikely ones.
- **Rain-affected matches are not modelled.** 29 D/L or shortened matches were left out of training. A revised target cannot be entered, so do not use the live mode for a shortened match.
- **Every wicket counts the same.** The model cannot tell a set opener from a tail-ender.
- **Probabilities are slightly overconfident in the middle range.** On the test seasons, "Clear edge" favourites in the chase won 68.5% of the time against 73.0% predicted, and "Strong edge" favourites won 89.0% against 92.1%. They are good guides, not certainties.
- **Accuracy depends on the moment.** On the test seasons it was 64.5% at the innings break, 74.5–75.5% after 6–10 chase overs and 84.2% after 15. Always quote the moment with the number.

## 6. Predictions for future seasons

- **The data ends with the 2024 final.** For 2025–2027 the app uses every result up to 2024-05-26 and nothing later.
- **Squads, ratings and ground conditions have changed since then.** The model doesn't know about the 2025 auction, for example.
- **Only franchises active in 2024 are accepted** for these seasons.

## 7. What the explanations do and don't say

- **Historical scenarios are retrospective simulations.** A date restricts the input history, while the fitted model still uses training through 2024-05-26. Scenarios on or before that cutoff are labelled in the app, API and CSV exports; they do not reproduce what could have been predicted at the time. Held-out evaluation uses separate fits trained only on earlier seasons.
- **The model-inputs panel lists what the model looked at.** That includes Elo ratings, form, head-to-head and record at the ground, or, in the live mode, runs needed, balls left and required rate. These are inputs, not proven reasons.
- **Coefficients describe associations, not cause and effect.** The logistic-regression coefficients in `reports/prematch_models.md` show which inputs move the prediction.

## 8. Engineering limits

- **Local only.** The app runs on a single machine, with no authentication. Anyone who can reach the API can download every logged prediction from `GET /api/v1/predictions/log.csv`, so do not expose the server beyond `localhost` without adding access control.
- **The log is a plain file.** Predictions go to `logs/predictions.jsonl`, which keeps growing. It is written under a lock within one server process; several server processes writing to the same file are not supported. A database backend can be added behind the same interface.
- **Batch speed.** A batch of 200 rows takes 3–9 seconds, depending on machine load. A single prediction takes 33–80 ms (live: 4–10 ms). The first prediction for a new date inside the data seasons builds a history snapshot and can take about 0.6 s.

## Responsible use

This is an educational project. The probabilities are estimates from historical patterns and cannot guarantee a result. Do not use them for betting, gambling or any financial decision.
