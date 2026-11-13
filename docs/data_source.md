# Data source and provenance

## Source

| Item | Value |
|---|---|
| Dataset | IPL Complete Dataset (2008–2024) |
| Publisher | Kaggle user `patrickb1912` |
| URL | https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020 |
| Upstream data | Ball-by-ball records from [Cricsheet](https://cricsheet.org/), as stated by public mirrors of this dataset |
| Files used | `matches.csv` (required), `deliveries.csv` (exploration only) |
| Received into project | 2026-09-28 |

The URL slug still says `20082020` because the dataset was first published for 2008–2020. Later versions extended it. The copy used here runs to 2024-05-26 (the 2024 final). Its row counts match public descriptions of this dataset: 1,095 matches and 260,920 deliveries.

## Version pinning

Kaggle can replace files under the same URL, so the project pins the exact files by hash rather than by download date.

| File | Rows | SHA-256 |
|---|---|---|
| `matches.csv` | 1,095 | `8A0394246D2A76B44526565F2245D8F4FEFF91BABE57AB16676A2C3D67AA0846` |
| `deliveries.csv` | 260,920 | `142236EA52950795DAB266D4AA392C7645BD1A0DB4704F0B8058AFD96E24E2C2` |

`python -m src.data.inspect_raw` and `pytest` both fail if a file differs from these hashes. If you download a newer version, results will differ. Record the new hashes and rerun the whole pipeline instead of mixing versions.

## Getting the files

The raw CSVs are not committed to this repository. Download them from the Kaggle URL above (a free Kaggle account is required), then place `matches.csv` and `deliveries.csv` in `data/raw/` without editing them. Some downloads name the files `matches_2008-2024.csv` and `deliveries_2008-2024.csv`; rename them to `matches.csv` and `deliveries.csv`.

## Licence and attribution

The licence shown on the Kaggle page has not yet been recorded by the team. **To do:** check the Kaggle page's licence field and record it here. Cricsheet data is released under the Open Data Commons Attribution License (ODC-By). Credit both Cricsheet and the Kaggle publisher in the final report.

## Raw-data rules

- Files in `data/raw/` are read-only. All cleaning writes new files to `data/processed/`.
- The project is educational and not a betting tool. See the limitations in `PROJECT.md`.
