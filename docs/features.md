# Feature specification (v1.1.0)

Produced by `python -m src.features.builder` from `data/processed/matches_clean.csv`.
Output: `data/processed/features.csv` (1,090 rows, 38 features, not committed) and
`data/processed/features_manifest.json` (input/output hashes, column order, constants).

## Timing rules

- A prediction is made **after the toss and before the first ball**.
- **Same-day policy:** features for a match use only matches on **strictly earlier dates**.
  The data has no start times and 272 dates host two matches, so neither same-day match sees the other.
- Training and inference share one function, `match_features(history, ...)`.
  Training walks the matches date by date: it computes features for every match on a date
  from the current `History`, then adds that date's results.
  Inference (`scenario_features`) builds a `History` from matches before the cutoff and calls the same function.
- Team and venue names pass through `src/data/mappings.py` in both paths.

## Prediction cutoff (`resolve_as_of`)

The API receives a season and can optionally receive a match date.

| Input | Cutoff (history strictly before) |
|---|---|
| Match date given | That date |
| Season present in the data, no date | First match date of that season (pre-season view: no result from that season is used) |
| Season after the data ends, no date | Day after the last match in the data |

## Features (Team A = `team1`)

| Group | Columns | Definition |
|---|---|---|
| Context | `is_playoff`, `is_neutral_country` | Playoff stage; venue outside India (2009 South Africa; UAE in 2014, 2020 and part of 2021) |
| Home | `home_a`, `home_b` | Venue city is the team's home city (`HOME_CITIES` in mappings; fixed domain knowledge, not learned). Always 0 in `NO_HOME_SEASONS` (2020–2022, when no team played at home). |
| Toss | `toss_won_a`, `toss_field`, `a_bats_first` | Toss winner, toss decision, and the batting order they imply |
| Elo | `elo_a`, `elo_b`, `elo_diff`, `elo_prob_a` | Start 1500, K = 20, zero-sum. At a new season each rating keeps 75% of its distance from 1500. All matches on a date use pre-day ratings. |
| Overall record | `win_rate_a/b`, `win_rate_diff`, `log_played_a/b` | All-time smoothed win rate and log(1 + matches played) |
| Recent form | `form5_*`, `form10_*` | Smoothed win rate over the last 5 / 10 matches, across seasons |
| Season | `season_win_rate_*`, `season_played_*`, `prev_season_win_rate_*` | Season-to-date record and the previous season's record |
| Head to head | `h2h_win_rate_a`, `h2h_played` | Smoothed record of A against B |
| Venue | `venue_win_rate_*`, `venue_played_*` | Each team's smoothed record at this ground |
| Batting order | `venue_chase_rate`, `batting_order_rate_a`, `recent_chase_rate` | Share of matches won by the chasing side at this venue (shrunk toward the league rate), the rate for Team A's batting order, and the league-wide chase rate over the last 60 matches |

**Smoothing:** `rate = (wins + k * prior) / (played + k)`, with prior 0.5 and k = 5 (k = 10 for venue and chase rates).
A team with no history gets exactly the prior, so new franchises (for example GT and LSG in 2022) need no special-case defaults.

All constants were fixed before any model results were seen. They are recorded in the manifest.

**Versions:** 1.0.0 first release. 1.1.0 sets the home flags to 0 in 2020–2022; before this, 8 Mumbai Indians matches in the 2022 Mumbai bubble counted as home matches.

## Team-swap symmetry

`swap_teams(df)` mirrors a feature table so Team B becomes Team A: paired columns swap,
`*_diff` columns change sign, probability-style columns become `1 - x`, and the target flips.
A test checks that this matches rebuilding the features with team1 and team2 exchanged.
Phase 4 can use it to augment training data or to average both orientations at prediction time.

## Tests (`tests/test_features.py`)

- Hand-checked values on a three-match history (rates, Elo, H2H, chase rate).
- Season rollover: Elo regression and season-stat reset.
- Leakage: changing one result alters no feature on that date or earlier; same-day matches are independent;
  adding future matches never changes earlier rows; input row order does not matter.
- Inference rows equal training rows for the same match and cutoff.
- Swap symmetry, Elo zero-sum, input validation, cutoff resolution, and value ranges on the real data.

## Signal check (single features, ROC-AUC for `team1_win`)

| Feature | 2008–2021 (train) | 2022 (validate) | 2023–2024 (test) |
|---|---|---|---|
| `elo_prob_a` | 0.547 | 0.431 | 0.488 |
| `win_rate_diff` | 0.551 | 0.461 | 0.553 |
| `form10_diff` | 0.533 | 0.497 | 0.483 |
| `home_a` | 0.544 | 0.500 | 0.494 |
| `toss_won_a` | 0.521 | 0.486 | 0.461 |

Pre-match signal in the IPL is weak, and no single feature is reliable in the recent seasons.
That fits the EDA (toss winner wins 50.8%, team strength swings between seasons).
High single-feature AUCs would point to leakage, and none appear.
Phase 4 should therefore expect accuracy only a few points above 50% and must compare every model against the coin-flip and Elo baselines.
The 2022 validation season is small (74 matches), so its metrics will be noisy.
