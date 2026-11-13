"""Phase 1 raw-data inspection.

Verifies the raw CSVs against their recorded SHA-256 hashes and expected
schemas, then writes reports/data_quality_report.md. Raw files are only read,
never modified.

Run from the project root:
    python -m src.data.inspect_raw
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
REPORT_PATH = ROOT / "reports" / "data_quality_report.md"

EXPECTED_SHA256 = {
    "matches.csv": "8A0394246D2A76B44526565F2245D8F4FEFF91BABE57AB16676A2C3D67AA0846",
    "deliveries.csv": "142236EA52950795DAB266D4AA392C7645BD1A0DB4704F0B8058AFD96E24E2C2",
}

MATCHES_COLUMNS = [
    "id", "season", "city", "date", "match_type", "player_of_match", "venue",
    "team1", "team2", "toss_winner", "toss_decision", "winner", "result",
    "result_margin", "target_runs", "target_overs", "super_over", "method",
    "umpire1", "umpire2",
]

DELIVERIES_COLUMNS = [
    "match_id", "inning", "batting_team", "bowling_team", "over", "ball",
    "batter", "bowler", "non_striker", "batsman_runs", "extra_runs",
    "total_runs", "extras_type", "is_wicket", "player_dismissed",
    "dismissal_kind", "fielder",
]

MISSING_TOKENS = {"", "NA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_raw(name: str) -> pd.DataFrame:
    """Read a raw CSV as strings so 'NA' and blanks stay visible."""
    return pd.read_csv(RAW_DIR / name, dtype=str, keep_default_na=False)


def missing_count(series: pd.Series) -> int:
    return int(series.str.strip().isin(MISSING_TOKENS).sum())


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def inspect_matches(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (report sections, failed checks)."""
    failures: list[str] = []
    out: list[str] = []

    if list(df.columns) != MATCHES_COLUMNS:
        failures.append("matches.csv columns differ from the expected schema")

    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        failures.append(f"{int(dates.isna().sum())} unparseable match dates")
    if df["id"].duplicated().any():
        failures.append("duplicate match ids")
    if (df["team1"] == df["team2"]).any():
        failures.append("a team plays itself")
    toss_ok = (df["toss_winner"] == df["team1"]) | (df["toss_winner"] == df["team2"])
    if not toss_ok.all():
        failures.append("toss winner outside the match-up")
    decided = df[~df["winner"].isin(MISSING_TOKENS)]
    winner_ok = (decided["winner"] == decided["team1"]) | (decided["winner"] == decided["team2"])
    if not winner_ok.all():
        failures.append("winner outside the match-up")

    teams = sorted(set(df["team1"]) | set(df["team2"]))
    out.append("## matches.csv overview\n")
    out.append(md_table(["Check", "Value"], [
        ["Rows / columns", f"{len(df):,} / {df.shape[1]}"],
        ["Date range", f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"],
        ["Distinct match ids", f"{df['id'].nunique():,}"],
        ["Distinct raw team names", len(teams)],
        ["Distinct raw venue names", df["venue"].nunique()],
        ["Distinct raw city names (excluding missing)", df.loc[~df["city"].isin(MISSING_TOKENS), "city"].nunique()],
        ["Matches with a recorded winner", f"{len(decided):,}"],
        ["Team 1 win share among decided matches", f"{(decided['winner'] == decided['team1']).mean():.1%}"],
        ["Toss winner always in match-up", bool(toss_ok.all())],
        ["Winner always in match-up", bool(winner_ok.all())],
    ]))

    out.append("\n\n## Missing values in matches.csv\n")
    out.append("Blank strings and the literal `NA` are both treated as missing.\n")
    out.append(md_table(["Column", "Missing", "Explanation"], [
        [f"`{c}`", missing_count(df[c]), MISSING_NOTES.get(c, "")]
        for c in MATCHES_COLUMNS if missing_count(df[c])
    ]))

    out.append("\n\n## Outcomes\n")
    super_over = pd.crosstab(df["result"], df["super_over"])
    out.append(md_table(
        ["result", "super_over = N", "super_over = Y"],
        [[r, int(super_over.loc[r].get("N", 0)), int(super_over.loc[r].get("Y", 0))] for r in super_over.index],
    ))
    out.append(
        "\n\nEvery tie was decided by a super over and has a recorded winner. "
        f"{int((df['method'] == 'D/L').sum())} results used the D/L method."
    )

    out.append("\n\n## Season labels versus match dates\n")
    years = df.assign(year=dates.dt.year).groupby("season")["year"].agg(["min", "max", "count"])
    out.append(md_table(
        ["Raw season", "Calendar year of matches", "Matches"],
        [[s, r["min"] if r["min"] == r["max"] else f"{r['min']}-{r['max']}", r["count"]] for s, r in years.iterrows()],
    ))
    if (years["min"] != years["max"]).any():
        failures.append("a season label spans more than one calendar year")
    out.append(
        "\n\nEach raw season label covers exactly one calendar year, so the year of "
        "`date` is a safe canonical season (Phase 2 decision)."
    )

    per_day = dates.value_counts()
    shared = per_day[per_day > 1]
    out.append("\n\n## Same-day matches\n")
    out.append(
        f"{len(shared)} dates host more than one match ({int(shared.sum())} matches in total; "
        f"at most {int(per_day.max())} per day). The file has no start time, and match ids are "
        "not in date order, so neither can order same-day matches."
    )
    if df.sort_values("date")["id"].astype(int).is_monotonic_increasing:
        out.append(" (Note: ids do happen to follow date order in this file.)")

    out.append("\n\n## Raw team names\n")
    counts = pd.concat([df["team1"], df["team2"]]).value_counts()
    out.append(md_table(["Raw team name", "Appearances"], [[t, int(counts[t])] for t in teams]))

    out.append("\n\n## Match types and toss decisions\n")
    out.append(md_table(["match_type", "Matches"], [[k, int(v)] for k, v in df["match_type"].value_counts().items()]))
    out.append("\n\n")
    out.append(md_table(["toss_decision", "Matches"], [[k, int(v)] for k, v in df["toss_decision"].value_counts().items()]))

    city_missing = df[df["city"].isin(MISSING_TOKENS)]["venue"].value_counts()
    out.append("\n\n## Venues with a missing city\n")
    out.append(md_table(["Venue", "Matches"], [[k, int(v)] for k, v in city_missing.items()]))
    return out, failures


MISSING_NOTES = {
    "city": "UAE venues (Dubai, Sharjah); city is recoverable from the venue name.",
    "player_of_match": "No-result matches; post-match field, never a model input.",
    "winner": "No-result matches; excluded from training.",
    "result_margin": "No-result matches and ties (a tie has no margin).",
    "target_runs": "No-result matches abandoned before a target was set.",
    "target_overs": "Same rows as `target_runs`.",
    "method": "Expected: only D/L-affected matches have a value.",
}


def inspect_deliveries(deliveries: pd.DataFrame, matches: pd.DataFrame) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    if list(deliveries.columns) != DELIVERIES_COLUMNS:
        failures.append("deliveries.csv columns differ from the expected schema")
    orphan = set(deliveries["match_id"]) - set(matches["id"])
    if orphan:
        failures.append(f"{len(orphan)} delivery match ids are missing from matches.csv")
    covered = matches["id"].isin(set(deliveries["match_id"])).sum()
    innings = deliveries["inning"].value_counts().sort_index()

    # The team batting in inning 1 must agree with the recorded toss outcome.
    bat_first = (
        deliveries[deliveries["inning"] == "1"]
        .drop_duplicates("match_id")
        .set_index("match_id")["batting_team"]
    )
    m = matches.assign(bat_first=matches["id"].map(bat_first)).dropna(subset=["bat_first"])
    toss_consistent = ((m["toss_winner"] == m["bat_first"]) == (m["toss_decision"] == "bat"))
    if not toss_consistent.all():
        failures.append(f"{int((~toss_consistent).sum())} matches where the toss decision contradicts the batting order")
    team1_bats_first = int((m["team1"] == m["bat_first"]).sum())
    out = ["\n\n## deliveries.csv overview\n", md_table(["Check", "Value"], [
        ["Rows / columns", f"{len(deliveries):,} / {deliveries.shape[1]}"],
        ["Matches covered", f"{covered:,} of {len(matches):,}"],
        ["Deliveries referencing an unknown match", len(orphan)],
        ["Rows per inning", ", ".join(f"{k}: {v:,}" for k, v in innings.items())],
        ["Toss decision agrees with inning-1 batting team", f"{int(toss_consistent.sum()):,} of {len(m):,}"],
        ["`team1` batted first", f"{team1_bats_first:,} of {len(m):,}"],
    ])]
    out.append(
        "\n\n`team1` is not consistently the side batting first, so the team1/team2 "
        "order carries no fixed cricket meaning."
    )
    out.append(
        "\n\nInnings 3-6 are super-over innings. `deliveries.csv` is for exploration only "
        "in this release and is not a model input."
    )
    return out, failures


def build_report() -> tuple[str, list[str]]:
    failures: list[str] = []
    hash_rows = []
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256(RAW_DIR / name)
        hash_rows.append([f"`{name}`", "match" if actual == expected else "**MISMATCH**", f"`{actual}`"])
        if actual != expected:
            failures.append(f"{name} hash differs from the recorded hash")

    matches = load_raw("matches.csv")
    deliveries = load_raw("deliveries.csv")
    match_sections, match_failures = inspect_matches(matches)
    delivery_sections, delivery_failures = inspect_deliveries(deliveries, matches)
    failures += match_failures + delivery_failures

    parts = [
        "# Raw data quality report\n",
        "Generated by `python -m src.data.inspect_raw`. Do not edit by hand; rerun the script.\n",
        "These are inspection findings, not cleaning rules. Cleaning decisions belong in Phase 2.\n",
        "## File integrity\n",
        md_table(["File", "SHA-256 check", "Actual hash"], hash_rows),
        "\n\n**Automated checks:** " + ("all passed." if not failures else "FAILED: " + "; ".join(failures)),
        "\n\n",
        *match_sections,
        *delivery_sections,
        "\n",
    ]
    return "".join(parts), failures


def tidy_markdown(text: str) -> str:
    """Put exactly one blank line between blocks (headings, paragraphs, tables)."""
    blocks: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if blocks and blocks[-1]:
                blocks.append([])
            continue
        is_table = stripped.startswith("|")
        current = blocks[-1] if blocks else None
        same_table = bool(current) and is_table and current[-1].startswith("|")
        if current is None or (current and not same_table):
            blocks.append([])
        blocks[-1].append(stripped)
    return "\n\n".join("\n".join(b) for b in blocks if b) + "\n"


def main() -> int:
    report, failures = build_report()
    report = tidy_markdown(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    if failures:
        print("Checks failed:\n- " + "\n- ".join(failures))
        return 1
    print("All raw-data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
