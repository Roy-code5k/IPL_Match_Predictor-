"""Phase 2 cleaning pipeline for matches.csv.

Reads the raw file (never modifies it), applies the documented rules in
docs/cleaning_rules.md, validates the result, and writes:

- data/processed/matches_clean.csv   one row per decided match
- data/processed/manifest.json       input/output hashes and row counts
- reports/cleaning_report.md         what each step did

Output is deterministic: the same raw file always gives byte-identical files.

Run from the project root:
    python -m src.data.cleaning
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import mappings
from src.data.inspect_raw import EXPECTED_SHA256, MATCHES_COLUMNS, md_table, sha256, tidy_markdown

ROOT = Path(__file__).resolve().parents[2]
RAW_MATCHES = ROOT / "data" / "raw" / "matches.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "matches_clean.csv"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
REPORT_PATH = ROOT / "reports" / "cleaning_report.md"

CLEANING_VERSION = "1.0.0"

CLEAN_COLUMNS = [
    "match_id", "date", "season", "season_raw", "match_type", "is_playoff",
    "venue", "city", "country", "venue_raw",
    "team1", "team2", "toss_winner", "toss_decision",
    "winner", "team1_win",
    # Outcome details: for EDA only, never model inputs.
    "result", "result_margin", "super_over", "method",
    "team1_raw", "team2_raw",
]


class CleaningError(ValueError):
    """Raised when the data breaks a cleaning rule."""


@dataclass
class CleaningLog:
    steps: list[tuple[str, int, str]] = field(default_factory=list)

    def add(self, step: str, rows: int, note: str = "") -> None:
        self.steps.append((step, rows, note))


def _is_missing(series: pd.Series) -> pd.Series:
    return series.str.strip().isin({"", "NA"})


def clean_matches(raw: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Apply the cleaning rules to a raw matches frame (all columns as strings)."""
    log = log or CleaningLog()
    missing_cols = [c for c in MATCHES_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise CleaningError(f"Raw matches file is missing columns: {missing_cols}")

    df = raw[MATCHES_COLUMNS].copy()
    df = df.apply(lambda s: s.str.strip())
    log.add("Load raw matches", len(df))

    if df["id"].duplicated().any():
        raise CleaningError("Duplicate match ids in raw data")

    # Dates and canonical season (year of the match date; raw label kept).
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any():
        raise CleaningError(f"{int(df['date'].isna().sum())} rows have an invalid date")
    df["season_raw"] = df["season"]
    df["season"] = df["date"].dt.year.astype(int)
    log.add("Parse dates; season = year of date", len(df),
            f"{df['season_raw'].nunique()} raw labels -> {df['season'].nunique()} seasons")

    # Teams.
    df["team1_raw"], df["team2_raw"] = df["team1"], df["team2"]
    for col in ("team1", "team2", "toss_winner"):
        df[col] = df[col].map(mappings.normalize_team)
    has_winner = ~_is_missing(df["winner"])
    df["winner"] = df["winner"].where(has_winner, None)
    df.loc[has_winner, "winner"] = df.loc[has_winner, "winner"].map(mappings.normalize_team)
    renamed = int((df["team1"] != df["team1_raw"]).sum() + (df["team2"] != df["team2_raw"]).sum())
    log.add("Map team names to franchises", len(df),
            f"{renamed} team slots renamed; "
            f"{pd.concat([df['team1_raw'], df['team2_raw']]).nunique()} raw names -> "
            f"{pd.concat([df['team1'], df['team2']]).nunique()} franchises")

    # Venues, with city and country taken from the venue table.
    df["venue_raw"] = df["venue"]
    city_missing = int(_is_missing(df["city"]).sum())
    df["venue"] = df["venue"].map(mappings.normalize_venue)
    df["city"] = df["venue"].map(mappings.venue_city)
    df["country"] = df["venue"].map(mappings.venue_country)
    log.add("Map venues; derive city and country", len(df),
            f"{df['venue_raw'].nunique()} raw labels -> {df['venue'].nunique()} grounds; "
            f"fills {city_missing} missing cities")

    # Match stage.
    unknown_types = set(df["match_type"]) - mappings.PLAYOFF_TYPES - mappings.LEAGUE_TYPES
    if unknown_types:
        raise CleaningError(f"Unknown match types: {sorted(unknown_types)}")
    df["is_playoff"] = df["match_type"].isin(mappings.PLAYOFF_TYPES)

    if not df["toss_decision"].isin({"bat", "field"}).all():
        raise CleaningError("toss_decision must be 'bat' or 'field'")

    # Drop matches without a result: no winner means no training label.
    no_result = df["result"].eq("no result") | ~has_winner
    df = df[~no_result].copy()
    log.add("Drop no-result matches", len(df), f"{int(no_result.sum())} removed")

    # Other outcome columns, typed for EDA.
    df["result_margin"] = pd.to_numeric(df["result_margin"].replace("NA", None), errors="coerce").astype("Int64")
    df["super_over"] = df["super_over"].eq("Y")
    df["method"] = df["method"].where(df["method"] != "NA", None)
    df["team1_win"] = (df["winner"] == df["team1"]).astype(int)
    df["match_id"] = df["id"].astype(int)

    validate_clean(df)
    log.add("Validate cleaned rows", len(df), "all invariants hold")

    # Stable order: by date, then id. Order within a date carries no meaning.
    df = df.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    return df[CLEAN_COLUMNS]


def validate_clean(df: pd.DataFrame) -> None:
    problems = []
    teams = mappings.CANONICAL_TEAMS
    for col in ("team1", "team2", "toss_winner", "winner"):
        if not df[col].isin(teams).all():
            problems.append(f"{col} has a non-canonical team")
    if (df["team1"] == df["team2"]).any():
        problems.append("a team plays itself")
    if not (df["toss_winner"].eq(df["team1"]) | df["toss_winner"].eq(df["team2"])).all():
        problems.append("toss winner outside the match-up")
    if not (df["winner"].eq(df["team1"]) | df["winner"].eq(df["team2"])).all():
        problems.append("winner outside the match-up")
    if not df["venue"].isin(mappings.VENUES).all():
        problems.append("non-canonical venue")
    key_cols = ["match_id", "date", "season", "venue", "city", "country",
                "team1", "team2", "toss_winner", "toss_decision", "winner"]
    if df[key_cols].isna().any().any():
        problems.append("missing value in a key column")
    if df["match_id"].duplicated().any():
        problems.append("duplicate match ids")
    if problems:
        raise CleaningError("; ".join(problems))


def _file_sha256(path: Path) -> str:
    return sha256(path)


def _frame_to_csv(df: pd.DataFrame) -> str:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False, lineterminator="\n")


def build_report(raw: pd.DataFrame, clean: pd.DataFrame, log: CleaningLog, raw_hash: str) -> str:
    parts = [
        "# Cleaning report\n",
        "Generated by `python -m src.data.cleaning`. Do not edit by hand. "
        "Rules are explained in `docs/cleaning_rules.md`.\n",
        md_table(["Item", "Value"], [
            ["Cleaning version", CLEANING_VERSION],
            ["Raw `matches.csv` SHA-256", f"`{raw_hash}`"],
            ["Output", "`data/processed/matches_clean.csv`"],
            ["Rows in / out", f"{len(raw):,} / {len(clean):,}"],
            ["Seasons", f"{clean['season'].min()}-{clean['season'].max()} ({clean['season'].nunique()})"],
        ]),
        "\n## Steps\n",
        md_table(["Step", "Rows after", "Note"], [[s, f"{n:,}", note] for s, n, note in log.steps]),
        "\n## Team mapping applied\n",
    ]

    raw_names = pd.concat([raw["team1"], raw["team2"]]).str.strip()
    team_rows = []
    for raw_name in sorted(raw_names.unique()):
        canonical = mappings.normalize_team(raw_name)
        team_rows.append([raw_name, canonical, int((raw_names == raw_name).sum()),
                          "renamed" if canonical != raw_name else ""])
    parts.append(md_table(["Raw name", "Franchise", "Appearances (raw)", ""], team_rows))

    parts.append("\n## Venue mapping applied\n")
    venue_rows = []
    counts = raw["venue"].str.strip().value_counts()
    for canonical, (city, country, _) in sorted(mappings.VENUES.items()):
        labels = [lab for lab in counts.index if mappings.normalize_venue(lab) == canonical]
        if not labels:
            continue
        venue_rows.append([
            canonical, city, country,
            "<br>".join(f"{lab} ({int(counts[lab])})" for lab in sorted(labels)),
        ])
    parts.append(md_table(["Ground", "City", "Country", "Raw labels (matches)"], venue_rows))

    parts.append("\n## Season labels\n")
    seasons = clean.groupby(["season_raw", "season"]).size().reset_index(name="n")
    parts.append(md_table(["Raw label", "Season", "Decided matches"],
                          seasons.sort_values("season").values.tolist()))
    return tidy_markdown("\n".join(parts))


def main() -> int:
    raw_hash = _file_sha256(RAW_MATCHES)
    if raw_hash != EXPECTED_SHA256["matches.csv"]:
        print("matches.csv does not match the recorded hash. Update the hash in "
              "src/data/inspect_raw.py only if you intend to use a new dataset version.")
        return 1

    raw = pd.read_csv(RAW_MATCHES, dtype=str, keep_default_na=False)
    log = CleaningLog()
    clean = clean_matches(raw, log)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_text = _frame_to_csv(clean)
    CLEAN_PATH.write_text(csv_text, encoding="utf-8", newline="\n")

    manifest = {
        "cleaning_version": CLEANING_VERSION,
        "input": {"file": "data/raw/matches.csv", "sha256": raw_hash, "rows": len(raw)},
        "output": {
            "file": "data/processed/matches_clean.csv",
            "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest().upper(),
            "rows": len(clean),
            "columns": CLEAN_COLUMNS,
            "first_date": clean["date"].min().strftime("%Y-%m-%d"),
            "last_date": clean["date"].max().strftime("%Y-%m-%d"),
        },
        "pandas_version": pd.__version__,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(raw, clean, log, raw_hash), encoding="utf-8")

    print(f"Wrote {CLEAN_PATH.relative_to(ROOT)} ({len(clean):,} rows)")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
