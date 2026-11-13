# Cleaning rules (version 1.0.0)

`python -m src.data.cleaning` turns `data/raw/matches.csv` into `data/processed/matches_clean.csv`. Name mappings live in `src/data/mappings.py`, which is the only copy. The feature builder and the prediction API must import it rather than re-typing names. Every run writes `reports/cleaning_report.md` (what changed) and `data/processed/manifest.json` (input and output hashes). The same raw file always gives a byte-identical output.

## Rules

| # | Rule | Reason |
|---|---|---|
| 1 | Raw files are read-only. The pipeline refuses to run if `matches.csv` differs from the recorded hash. | Results stay tied to one known dataset version. |
| 2 | `season` = calendar year of `date`. The original label is kept as `season_raw`. | Phase 1 verified that every raw label covers exactly one year (`2007/08` → 2008, `2009/10` → 2010, `2020/21` → 2020). |
| 3 | Team names map to 15 franchises. Only renames of the same franchise are merged: Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings, Royal Challengers Bangalore → Royal Challengers Bengaluru, Rising Pune Supergiants → Rising Pune Supergiant. Original names are kept in `team1_raw` and `team2_raw`. | Merging different franchises would mix unrelated histories. Deccan Chargers / Sunrisers Hyderabad, Pune Warriors / Rising Pune Supergiant, and Gujarat Lions / Gujarat Titans stay separate. |
| 4 | Venue labels map to 36 physical grounds named "Ground, City". Renamed grounds are merged: Feroz Shah Kotla → Arun Jaitley Stadium, Sardar Patel Stadium (Motera) → Narendra Modi Stadium, Subrata Roy Sahara Stadium → MCA Stadium Pune, Sheikh Zayed Stadium → Zayed Cricket Stadium. Mullanpur's new stadium (2024) stays separate from Mohali. The original label is kept in `venue_raw`. | Venue features should describe the playing ground, not the spelling. Motera was rebuilt in 2020–21, so its pre-2021 history is a weaker guide; EDA notes this. |
| 5 | `city` and `country` come from the venue table, not the raw `city` column. | Fills the 51 missing cities (Dubai, Sharjah) and fixes spelling variants such as Bangalore / Bengaluru and Chandigarh / Mohali. |
| 6 | `is_playoff` is true for Qualifier 1, Qualifier 2, Eliminator, Final, and the older Semi Final, Elimination Final, and 3rd Place Play-Off. | Known before the match; a candidate input. |
| 7 | Matches with `result = no result` (5) are dropped. | No winner means no label. |
| 8 | Tied matches keep their super-over winner. | The recorded winner is the official result. |
| 9 | Target `team1_win` = 1 if `winner == team1`, else 0. | Binary target defined in PROJECT.md. |
| 10 | Rows are sorted by `date`, then `match_id`. Order within a date has no meaning. | Phase 3 must use strictly earlier dates for features. |
| 11 | Unknown team names, venues, match types, toss decisions, or invalid dates stop the pipeline with an error. | New data must be mapped deliberately, never guessed. |
| 12 | Validation checks the cleaned rows: teams and winner are canonical, toss winner and winner belong to the match-up, no team plays itself, key columns have no missing values, and match IDs are unique. | Catches mapping mistakes before features are built. |

## Columns in `matches_clean.csv`

Inputs for later phases: `date`, `season`, `match_type`, `is_playoff`, `venue`, `city`, `country`, `team1`, `team2`, `toss_winner`, `toss_decision`.

Label: `winner`, `team1_win`.

Kept for EDA only and **never** used as model inputs: `result`, `result_margin`, `super_over`, `method`.

Kept for traceability: `match_id`, `season_raw`, `venue_raw`, `team1_raw`, `team2_raw`.

Dropped from the raw file: `player_of_match`, `target_runs`, `target_overs`, `umpire1`, and `umpire2`. They are known only after the match starts or are out of scope.
