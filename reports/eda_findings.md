# EDA findings summary (Phase 2)

Source: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb), run on `data/processed/matches_clean.csv` (cleaning v1.0.0: 1,090 decided matches, 2008-04-18 to 2024-05-26, 15 franchises, 36 grounds). Charts are in [`figures/`](figures/). All figures are descriptive over the whole dataset. None of them may be used directly as model features.

## Key numbers

| Measure | Value |
|---|---|
| Decided matches | 1,090 (5 no-results removed) |
| `team1` wins (class balance) | 50.9% (555 of 1,090) |
| Toss winner wins | 50.8% |
| Toss winner wins after choosing to field / to bat | 53.9% (700 matches) / 45.4% (390 matches) |
| Chasing team wins | 54.1% |
| Franchise all-time win rates | 27% (Pune Warriors) to 62% (Gujarat Titans) |
| Chasing win rate at grounds with 25+ matches | 42% (MA Chidambaram, Chennai) to 65% (Sawai Mansingh, Jaipur) |
| Matches at the five busiest grounds | 44% |
| Matches outside India | 168 (South Africa 57, UAE 111) |

## Findings and what they mean for the model

1. **Classes are balanced, but team order is arbitrary.** `team1` wins 50.9% overall, yet anywhere from 38% to 71% in a single season. Accuracy is a fair headline metric. In Phase 3, each training match should also be added with the teams swapped, so the model can't learn a team-order effect. Predictions for (A, B) and (B, A) should then sum to 100%.
2. **The toss alone is close to a coin flip.** "Toss winner wins" scores 50.8%. That is the floor any model must clearly beat on the held-out seasons.
3. **The toss decision matters, and its effect drifts.** Fielding first wins more often. Field choice rose from under half of tosses (2009–2013) to about three-quarters (2016 onward). This drift is why the evaluation must use a time-based split.
4. **Form beats all-time record.** Franchise win rates swing widely from season to season (see the heatmap). Recent-form and season-to-date features should matter more than all-time win rates.
5. **Grounds differ in chasing advantage.** A venue chasing win rate, combined with the toss decision, is a candidate feature. It must use only earlier matches, smoothed toward the overall rate where a ground has few matches.
6. **Some seasons were played away from usual grounds.** 2009 (South Africa), part of 2014, all of 2020, and part of 2021 (UAE). Home-ground effects don't apply there.
7. **New franchises have little history.** Gujarat Titans and Lucknow Super Giants have 3 seasons each. Features need a default (league average) when a team has little or no history.

## Agreed evaluation design (time-based split)

The team decided on a time-based split, not a random 80/20. The proposed periods, to be confirmed in Phase 4:

| Set | Seasons | Decided matches | Use |
|---|---|---|---|
| Train | 2008–2021 | 872 | Fit models |
| Validation | 2022 | 74 | Tune and select |
| Test | 2023–2024 | 144 | Final, one-time evaluation |

## Limitations

- **Missing context.** No player, squad, injury, pitch, or weather data. A large part of every T20 result is outside this dataset, so expect modest accuracy (roughly 55–60%) and present probabilities as estimates.
- **Small samples.** 1,090 matches in total and 144 in the test set. Differences of a few accuracy points between models may be noise; Phase 5 should report confidence intervals.
- **Changing conditions.** Team ownership, squads (auctions every 3 years), the Impact Player rule (from 2023), and renovated grounds such as Motera change what history means.
- **Descriptive bias.** These EDA rates use all seasons, including later matches. Using them directly as features would leak the future. Phase 3 recomputes every statistic from earlier matches only.
