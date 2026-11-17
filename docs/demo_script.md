# Demo script

A demonstration of about 12 minutes, for Hrituraj and Amul to rehearse together. The speaker split is a suggestion; swap parts as you like, but both of you should be able to give every part.

Every number below was produced by release 1.0.0. If the app shows something different, the wrong release or the wrong data is loaded (see "If something goes wrong").

## Before the demo (10 minutes before)

1. Terminal 1, project root:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.backend.main:app --port 8000
   ```
   Wait for `Application startup complete` (about 7 seconds).
2. Terminal 2:
   ```powershell
   cd app\frontend
   npm.cmd run dev
   ```
3. Open http://localhost:5173/. The badge at the top right must say **MODEL ONLINE**.
4. Optional backup: open http://127.0.0.1:8000/api/docs in a second tab.
5. Have `reports/final_report.md` and `reports/figures/09_accuracy_by_moment.png` open in case the projector fails.
6. Make one prediction to warm up, then reload the page so the form is empty.

## 1. The problem (Hrituraj, 1 minute) — Overview section

- "We estimate each team's chance of winning an IPL match: before the first ball, and live from the score."
- "We used 1,090 IPL matches from 2008 to 2024 and every ball bowled in them: 260,920 deliveries."
- "It's an educational tool, not for betting. Every number is an estimate with its uncertainty shown."

## 2. How it works (Hrituraj, 2 minutes) — How It Works section

- **Cleaning:** renamed franchises merged (Delhi Daredevils → Delhi Capitals), 58 ground labels mapped to 36, no-result matches removed.
- **No peeking at the future:** "Every feature for a match uses only matches on earlier dates. Two matches were played on the same day 272 times and we have no start times, so same-day matches never see each other. Our tests fail if any future result leaks in."
- **38 features:** Elo rating, recent form, head-to-head, record at the ground, home ground, toss and batting order.
- **Honest testing:** "We trained on 2008–2022 and chose our models on 2015–2022. We locked that choice in git, then tested once on 2023–2024, seasons the models had never seen."

## 3. Pre-match prediction (Amul, 2 minutes) — Match Predictor

Enter:

| Field | Value |
|---|---|
| Prediction moment | Before the first ball |
| Season | 2026 |
| Venue | Wankhede Stadium, Mumbai |
| Team A | Mumbai Indians |
| Team B | Chennai Super Kings |
| Toss winner | Chennai Super Kings |
| Toss decision | Field first |

Press **Predict**. Expected: **Chennai Super Kings 52.8%, Mumbai Indians 47.2%, "Too close to call"**.

Points to make:

- "Both probabilities, a bar, and the confidence in words. The close-call notice appears because the favourite is under 55%."
- Point to the **model inputs** panel: "These are the inputs the model used: Elo ratings, form, head-to-head. They are inputs, not proof of cause."
- "Pre-match, this is close to a coin flip, and the app says so." Read out the note under the result.
- **Order doesn't matter:** set Team A to Chennai Super Kings, then Team B to Mumbai Indians (the toss stays with CSK), and press Predict. It still shows CSK 52.8%. "The model is trained on both team orders, so listing a team first can't change the result."
- **Validation:** set Team B to Mumbai Indians as well. It shows "Choose a different franchise." under Team B. Open the Team A list: Deccan Chargers is greyed out, because it did not exist in 2024. Put Team B back to Chennai Super Kings.

## 4. Live prediction (Amul, 3 minutes) — the same match, during play

Switch to **During the match** and keep the same teams, ground and toss. CSK fielded first, so **Mumbai bats first** and CSK chases. Run through the match:

| Enter | Expected result | What to say |
|---|---|---|
| Innings 1st, 6 overs, 52 runs, 1 wicket | CSK 53.6%, Too close to call | "A normal start: still even." |
| Innings 1st, 15 overs, 150 runs, 3 wickets | MI 67.4%, Clear edge | "150 after 15 is well above par, so Mumbai lead." |
| Innings Chase, first-innings total 185, 0 overs, 0 runs, 0 wickets | MI 56.3%, Slight edge | "Target 186. Defending sides have had the edge since 2023." |
| Chase, 10 overs, 95 runs, 2 wickets | CSK 77.5%, Clear edge | "91 needed off 60 with 8 wickets in hand: CSK are clear favourites." |
| Chase, 15 overs, 140 runs, 3 wickets | CSK 86.0%, Strong edge | |
| Chase, 15 overs, 120 runs, 6 wickets | **MI 84.9%**, Strong edge | "Same over, 20 fewer runs and 3 more wickets down: the prediction flips." |
| Chase, 18 overs, 160 runs, 5 wickets | MI 50.3%, Too close to call | "26 off 12 with 5 wickets left: a true toss-up." |

Point to the model-inputs panel: target, runs needed, balls left, required rate and par score.

## 5. Model performance (Hrituraj, 2 minutes) — Model Performance section

- "These numbers come from the saved model release, so they always describe the model actually loaded."
- **Pre-match: 48.6% on 2023–2024, no better than a coin flip.** Point to the confusion matrix: "It picked Team B 107 times out of 144."
- **Why:** "From 2008 to 2022 the chasing side won 55% of matches, and the model learned that. In 2023 the Impact Player rule arrived, scores jumped, and the side batting first won 52%. A random split would have hidden this; testing on later seasons exposed it."
- **Live:** point to the accuracy-by-moment bars. "64.5% at the innings break, 75.5% after 10 overs of the chase, 84.2% after 15. That is how we reach 70–80%: honestly, by predicting during the match. We always quote the moment with the number."
- **Comparison tables:** "Every model against the coin flip and Elo baselines. The selected model was chosen before the test, so we didn't pick the winner after seeing the results."

## 6. Batch and exports (Amul, 1 minute) — Batch section

1. Click **Download the template** and open it: one pre-match row and one live row.
2. Upload it and press **Run batch**. Both rows are predicted.
3. Click **Download results CSV**. "Every result carries its inputs and the model version."
4. Mention the **prediction log**: "Every prediction is logged with its inputs, result, model version and response time, and can be downloaded as CSV."

## 7. Close (Hrituraj, 30 seconds)

- "174 automated Python tests and 7 for the web app, a full rebuild from raw data in under two minutes with identical results, and predictions in well under a tenth of a second."
- "The main lesson: before the toss, team history barely predicts the IPL; once the match starts, the score does."

## Likely questions

| Question | Answer |
|---|---|
| Why is pre-match accuracy so low? | The data has no players, playing XIs, injuries, pitch or weather, only team-level history. Even the toss winner wins just 50.8%. On top of that, the chase advantage reversed in 2023. |
| Why not a random 80/20 split? | It would train on matches played after the ones being predicted, which is leakage, and it would hide changes over time, such as the 2023 reversal. |
| How do you know there is no leakage? | Features use strictly earlier dates. Tests change one match's result and check that nothing on or before that date changes. Training and serving use the same feature function, and a test checks that they give identical rows. |
| Why logistic regression for live and not XGBoost? | It had the lowest walk-forward log loss. On the test seasons all three were within 0.3 points of each other, so the simplest model is the sensible choice. |
| Why select by log loss instead of accuracy? | Log loss rewards honest probabilities. A model that says 90% and is wrong is punished more than one that says 55%. Our app shows probabilities, so their quality matters. |
| What do the confidence labels mean? | They depend on the favourite's probability: below 55% "Too close to call", 55–65% "Slight edge", 65–80% "Clear edge", 80%+ "Strong edge". They were fixed before testing. In the test chases, "Strong edge" favourites won 89% of the time. |
| Is the 2024 final in the training data? | The released model was refitted on 2008–2024, so yes. The test numbers come from a separate fit on 2008–2022. That is why the demo uses a 2026 scenario. If you pick a date on or before 2024-05-26, the app shows a **Retrospective simulation** notice: the date limits the history used as input, but the model has already seen those matches. |
| Did you try to improve the pre-match model? | Yes. We added team batting and bowling form and tried training on recent seasons only, with the 16 settings fixed in advance. The selected one got 55.0% on 2015–2022 and 51.4% on 2023–2024: 4 more correct out of 144, which is within noise. We had already seen 2023–2024, so we did not change the released model. It needs a fresh season to prove anything (`reports/prematch_v2_experiment.md`). |
| What about rain-affected matches? | The 29 D/L or shortened matches are left out of the live model, because their target isn't "first-innings total + 1 in 20 overs". The live mode shouldn't be used for them. |
| Where is the database? | We decided a file is enough for now: every prediction goes to a JSON-lines log (`logs/predictions.jsonl`) that can be downloaded as CSV. The log sits behind a storage interface, so a database such as MongoDB can be added later without changing the API. Model metadata lives in the release's `metadata.json`. |
| How long does a prediction take? | About 30–80 ms before the match and 4–10 ms live, depending on how busy the laptop is; the target was under a second. A batch of 200 takes a few seconds. |

## If something goes wrong

| Problem | Fix |
|---|---|
| Badge says MODEL OFFLINE, or "Prediction service offline" | Terminal 1 isn't running, or has stopped. Start it again (step 1), wait for `Application startup complete`, and press Predict again; the app checks the connection again. |
| `GET /` shows a JSON message | You opened port 8000. The app is at port **5173**. |
| Health says the model failed to load | The release files are missing or were changed. Rebuild them with `python -m src.models.release` (about 20 seconds). |
| Numbers differ from this script | Check that the season is 2026, the venue is Wankhede, and CSK won the toss and chose to field. |
| Projector or laptop fails | Present from `reports/final_report.md` and the figures in `reports/figures/`. |
