# Data dictionary

Covers the raw files in `data/raw/`. Counts come from `reports/data_quality_report.md` (regenerate with `python -m src.data.inspect_raw`). Both files mark missing values with the literal text `NA`. In `deliveries.csv`, `extras_type` is blank instead.

**Model role** says whether a column may feed the model:

- **Input:** known after the toss and before the first ball. May be used directly or to build historical features.
- **Ordering:** used to sort matches so features only use earlier results; not a direct input.
- **Target:** source of the label.
- **Excluded:** known only during or after the match (leakage), or out of scope.

## matches.csv (one row per match, 1,095 rows)

| Column | Type | Description | Missing | Model role |
|---|---|---|---|---|
| `id` | integer | Unique match ID. Not in date order. | 0 | Identifier only |
| `season` | text | Season label. Mixes years (`2009`) with labels (`2007/08`, `2009/10`, `2020/21`). Each label covers exactly one calendar year. | 0 | Input after normalising to the year of `date` (Phase 2) |
| `city` | text | Host city. Spelling varies (e.g. `Bangalore` / `Bengaluru`). | 51 (Dubai and Sharjah matches) | Input candidate; mostly redundant with `venue` |
| `date` | date `YYYY-MM-DD` | Match date, 2008-04-18 to 2024-05-26. No start time. 272 dates have two matches. | 0 | Ordering |
| `match_type` | text | `League`, `Qualifier 1`, `Qualifier 2`, `Eliminator`, `Final`, and older playoff names (`Semi Final`, `Elimination Final`, `3rd Place Play-Off`). | 0 | Input candidate (known before the match) |
| `player_of_match` | text | Player of the match award. | 5 (no results) | Excluded: post-match |
| `venue` | text | Stadium name. 58 raw labels with duplicates such as `Eden Gardens` / `Eden Gardens, Kolkata`. | 0 | Input after mapping (Phase 2) |
| `team1` | text | One side in the match-up. **Not** consistently the side batting first (batted first in 795 of 1,095 matches). | 0 | Input ("Team A") |
| `team2` | text | The other side. | 0 | Input ("Team B") |
| `toss_winner` | text | Team that won the toss. Always `team1` or `team2`. | 0 | Input |
| `toss_decision` | `bat` / `field` | Toss winner's choice. Agrees with the inning-1 batting team in all 1,095 matches. | 0 | Input |
| `winner` | text | Winning team. Always `team1` or `team2` when present. Includes super-over winners of tied matches. | 5 (no results) | Target: `1` if `winner == team1`, else `0` |
| `result` | `runs` / `wickets` / `tie` / `no result` | How the match was decided. | 0 | Excluded: post-match. `no result` rows are dropped from training |
| `result_margin` | number | Winning margin in runs or wickets. | 19 (5 no results + 14 ties) | Excluded: post-match |
| `target_runs` | integer | Runs needed by the chasing side. | 3 (abandoned before a target) | Excluded: set during the match |
| `target_overs` | number | Overs available to the chasing side; below 20 when rain shortened the match. | 3 | Excluded: set during the match |
| `super_over` | `Y` / `N` | Whether a super over decided a tie (14 matches). | 0 | Excluded: post-match |
| `method` | `D/L` / `NA` | `D/L` when Duckworth-Lewis decided the result (21 matches). | 1,074 (`NA` means not applied) | Excluded: set during the match |
| `umpire1`, `umpire2` | text | On-field umpires. | 0 | Excluded: out of scope for this release |

### Team names

There are 19 raw team names. Four are the same franchise under an old and new name. The frontend already uses this grouping, and Phase 2 must adopt it in the cleaning code:

| Raw name | Canonical franchise |
|---|---|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Rising Pune Supergiants | Rising Pune Supergiant |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |

Deccan Chargers / Sunrisers Hyderabad, Pune Warriors / Rising Pune Supergiant, and Gujarat Lions / Gujarat Titans remain separate franchises. This leaves 15 franchises. The mapping is proposed, not final, until Phase 2 records it in PROJECT.md.

## deliveries.csv (one row per ball, 260,920 rows)

Exploration only. Nothing from this file is a model input in this release, because every column describes play during the match.

| Column | Type | Description |
|---|---|---|
| `match_id` | integer | Links to `matches.id`. Every match has deliveries. |
| `inning` | integer | 1 and 2 are regular innings; 3–6 are super-over innings. |
| `batting_team`, `bowling_team` | text | Raw team names (same 19 as `matches.csv`). |
| `over` | integer | Over number, starting at 0. |
| `ball` | integer | Ball within the over; can exceed 6 when wides or no-balls are re-bowled. |
| `batter`, `bowler`, `non_striker` | text | Player names in scorecard form (e.g. `SC Ganguly`). |
| `batsman_runs` | integer | Runs off the bat. |
| `extra_runs` | integer | Extras on this ball. |
| `total_runs` | integer | `batsman_runs + extra_runs`. |
| `extras_type` | text | `wides`, `legbyes`, `byes`, `noballs`, `penalty`; blank when no extra. |
| `is_wicket` | `0` / `1` | Whether a wicket fell. |
| `player_dismissed`, `dismissal_kind`, `fielder` | text | Wicket details; `NA` when there was no wicket or no fielder involved. |
