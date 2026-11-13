"""Leakage-safe historical features (Phase 3).

One definition serves training and inference:

    snapshot = History(...)             # results strictly before the as-of date
    row = match_features(snapshot, ...)  # features for one match-up

Training walks the cleaned matches date by date. For each date it computes
features for every match on that date from the snapshot, and only then adds
that date's results to the snapshot. Same-day matches never see each other.
Inference builds a snapshot from all matches before its as-of date and calls
the same `match_features`.

Features are written from Team A's (team1's) point of view. `swap_teams`
mirrors a feature table to Team B's point of view for augmentation and
symmetry checks.

Run from the project root to write data/processed/features.csv:
    python -m src.features.builder
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data import mappings

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "processed" / "matches_clean.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
FEATURES_MANIFEST = ROOT / "data" / "processed" / "features_manifest.json"

FEATURE_VERSION = "1.0.0"

# Tunable constants, fixed before looking at any results.
ELO_START = 1500.0
ELO_K = 20.0
ELO_SEASON_CARRY = 0.75   # share of a rating's distance from 1500 kept into a new season
PRIOR_STRENGTH = 5.0      # pseudo-matches pulling small-sample rates toward the prior
VENUE_PRIOR_STRENGTH = 10.0
RECENT_CHASE_WINDOW = 60  # roughly one season of league-wide matches

KEY_COLUMNS = ["match_id", "date", "season", "team1", "team2", "venue", "toss_winner", "toss_decision"]
TARGET = "team1_win"

# Model inputs. Order is part of the feature specification.
FEATURE_COLUMNS = [
    # match context (known after the toss)
    "is_playoff", "is_neutral_country", "home_a", "home_b",
    "toss_won_a", "toss_field", "a_bats_first",
    # team strength
    "elo_a", "elo_b", "elo_diff", "elo_prob_a",
    "win_rate_a", "win_rate_b", "win_rate_diff",
    "log_played_a", "log_played_b",
    "form5_a", "form5_b", "form5_diff",
    "form10_a", "form10_b", "form10_diff",
    "season_win_rate_a", "season_win_rate_b", "season_win_rate_diff",
    "season_played_a", "season_played_b",
    "prev_season_win_rate_a", "prev_season_win_rate_b",
    # head to head
    "h2h_win_rate_a", "h2h_played",
    # venue and batting order
    "venue_win_rate_a", "venue_win_rate_b", "venue_played_a", "venue_played_b",
    "venue_chase_rate", "batting_order_rate_a", "recent_chase_rate",
]

# How each column changes when Team A and Team B are swapped.
_PAIRS = [
    ("home_a", "home_b"), ("elo_a", "elo_b"), ("win_rate_a", "win_rate_b"),
    ("log_played_a", "log_played_b"), ("form5_a", "form5_b"), ("form10_a", "form10_b"),
    ("season_win_rate_a", "season_win_rate_b"), ("season_played_a", "season_played_b"),
    ("prev_season_win_rate_a", "prev_season_win_rate_b"),
    ("venue_win_rate_a", "venue_win_rate_b"), ("venue_played_a", "venue_played_b"),
]
_NEGATE = ["elo_diff", "win_rate_diff", "form5_diff", "form10_diff", "season_win_rate_diff"]
_COMPLEMENT = ["toss_won_a", "a_bats_first", "elo_prob_a", "h2h_win_rate_a", "batting_order_rate_a"]
_UNCHANGED = ["is_playoff", "is_neutral_country", "toss_field", "h2h_played",
              "venue_chase_rate", "recent_chase_rate"]


def smoothed_rate(wins: float, played: float, prior: float = 0.5, strength: float = PRIOR_STRENGTH) -> float:
    """Win rate pulled toward `prior`; equals `prior` when there is no history."""
    return (wins + strength * prior) / (played + strength)


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def bat_first_team(team1: str, team2: str, toss_winner: str, toss_decision: str) -> str:
    if toss_decision == "bat":
        return toss_winner
    return team2 if toss_winner == team1 else team1


@dataclass(frozen=True)
class Result:
    """One finished match, as stored in the history snapshot."""
    date: pd.Timestamp
    season: int
    team1: str
    team2: str
    venue: str
    winner: str
    bat_first: str


@dataclass
class History:
    """Everything known from finished matches before an as-of date."""

    results: list[Result] = field(default_factory=list)
    ratings: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: ELO_START))
    rating_season: int | None = None
    by_team: dict[str, list[Result]] = field(default_factory=lambda: defaultdict(list))

    @property
    def last_date(self) -> pd.Timestamp | None:
        return self.results[-1].date if self.results else None

    @classmethod
    def from_matches(cls, matches: pd.DataFrame, before: pd.Timestamp | None = None) -> "History":
        """Snapshot from cleaned matches, optionally only those strictly before `before`."""
        df = matches if before is None else matches[matches["date"] < before]
        history = cls()
        for _, day in _ordered(df).groupby("date", sort=True):
            history.add_day(day)
        return history

    def rating(self, team: str, season: int) -> float:
        """Rating for a match in `season`, regressed toward 1500 if a new season has begun."""
        r = self.ratings[team]
        if self.rating_season is not None and season > self.rating_season:
            r = ELO_START + ELO_SEASON_CARRY * (r - ELO_START)
        return r

    def add_day(self, day: pd.DataFrame) -> None:
        """Add all results from one date. Ratings use pre-day values for every match that day."""
        if day.empty:
            return
        date = day["date"].iloc[0]
        if self.last_date is not None and date <= self.last_date:
            raise ValueError("History must be extended in strictly increasing date order")
        season = int(day["season"].iloc[0])
        if self.rating_season is None or season > self.rating_season:
            if self.rating_season is not None:
                for team in list(self.ratings):
                    self.ratings[team] = self.rating(team, season)
            self.rating_season = season

        start = dict(self.ratings)
        for row in day.itertuples(index=False):
            a, b = row.team1, row.team2
            ra, rb = start.get(a, ELO_START), start.get(b, ELO_START)
            score_a = 1.0 if row.winner == a else 0.0
            delta = ELO_K * (score_a - elo_expected(ra, rb))
            self.ratings[a] += delta
            self.ratings[b] -= delta
            result = Result(date, season, a, b, row.venue, row.winner,
                            bat_first_team(a, b, row.toss_winner, row.toss_decision))
            self.results.append(result)
            self.by_team[a].append(result)
            self.by_team[b].append(result)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "match_id"], kind="mergesort")


def _team_stats(history: History, team: str, season: int, venue: str) -> dict[str, float]:
    games = history.by_team.get(team, [])
    wins = [g.winner == team for g in games]
    n = len(games)
    last5, last10 = wins[-5:], wins[-10:]
    this_season = [w for g, w in zip(games, wins) if g.season == season]
    prev_season = [w for g, w in zip(games, wins) if g.season == season - 1]
    at_venue = [w for g, w in zip(games, wins) if g.venue == venue]
    return {
        "win_rate": smoothed_rate(sum(wins), n),
        "log_played": math.log1p(n),
        "form5": smoothed_rate(sum(last5), len(last5)),
        "form10": smoothed_rate(sum(last10), len(last10)),
        "season_win_rate": smoothed_rate(sum(this_season), len(this_season)),
        "season_played": float(len(this_season)),
        "prev_season_win_rate": smoothed_rate(sum(prev_season), len(prev_season)),
        "venue_win_rate": smoothed_rate(sum(at_venue), len(at_venue)),
        "venue_played": float(len(at_venue)),
    }


def match_features(
    history: History,
    *,
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
    is_playoff: bool = False,
) -> dict[str, float]:
    """Features for one match-up, using only what `history` contains.

    Names must already be canonical (see src.data.mappings).
    """
    if team1 == team2:
        raise ValueError("team1 and team2 must differ")
    if toss_winner not in (team1, team2):
        raise ValueError("toss_winner must be team1 or team2")
    if toss_decision not in ("bat", "field"):
        raise ValueError("toss_decision must be 'bat' or 'field'")

    a = _team_stats(history, team1, season, venue)
    b = _team_stats(history, team2, season, venue)
    elo_a, elo_b = history.rating(team1, season), history.rating(team2, season)

    h2h = [r for r in history.by_team.get(team1, []) if team2 in (r.team1, r.team2)]
    h2h_wins = sum(r.winner == team1 for r in h2h)

    def chase_won(r: Result) -> bool:
        return r.winner != r.bat_first

    all_chases = [chase_won(r) for r in history.results]
    global_chase = smoothed_rate(sum(all_chases), len(all_chases), strength=VENUE_PRIOR_STRENGTH)
    venue_chases = [chase_won(r) for r in history.results if r.venue == venue]
    venue_chase = smoothed_rate(sum(venue_chases), len(venue_chases), prior=global_chase,
                                strength=VENUE_PRIOR_STRENGTH)
    recent = all_chases[-RECENT_CHASE_WINDOW:]
    recent_chase = smoothed_rate(sum(recent), len(recent), strength=VENUE_PRIOR_STRENGTH)

    a_bats_first = bat_first_team(team1, team2, toss_winner, toss_decision) == team1
    city = mappings.venue_city(venue)
    home = mappings.HOME_CITIES

    return {
        "is_playoff": float(bool(is_playoff)),
        "is_neutral_country": float(mappings.venue_country(venue) != "India"),
        "home_a": float(city in home.get(team1, ())),
        "home_b": float(city in home.get(team2, ())),
        "toss_won_a": float(toss_winner == team1),
        "toss_field": float(toss_decision == "field"),
        "a_bats_first": float(a_bats_first),
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "elo_prob_a": elo_expected(elo_a, elo_b),
        "win_rate_a": a["win_rate"], "win_rate_b": b["win_rate"],
        "win_rate_diff": a["win_rate"] - b["win_rate"],
        "log_played_a": a["log_played"], "log_played_b": b["log_played"],
        "form5_a": a["form5"], "form5_b": b["form5"], "form5_diff": a["form5"] - b["form5"],
        "form10_a": a["form10"], "form10_b": b["form10"], "form10_diff": a["form10"] - b["form10"],
        "season_win_rate_a": a["season_win_rate"], "season_win_rate_b": b["season_win_rate"],
        "season_win_rate_diff": a["season_win_rate"] - b["season_win_rate"],
        "season_played_a": a["season_played"], "season_played_b": b["season_played"],
        "prev_season_win_rate_a": a["prev_season_win_rate"],
        "prev_season_win_rate_b": b["prev_season_win_rate"],
        "h2h_win_rate_a": smoothed_rate(h2h_wins, len(h2h)),
        "h2h_played": float(len(h2h)),
        "venue_win_rate_a": a["venue_win_rate"], "venue_win_rate_b": b["venue_win_rate"],
        "venue_played_a": a["venue_played"], "venue_played_b": b["venue_played"],
        "venue_chase_rate": venue_chase,
        "batting_order_rate_a": (1 - venue_chase) if a_bats_first else venue_chase,
        "recent_chase_rate": recent_chase,
    }


def build_training_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Features for every cleaned match, each from strictly earlier dates only."""
    history = History()
    rows = []
    for _, day in _ordered(matches).groupby("date", sort=True):
        for m in day.itertuples(index=False):
            feats = match_features(
                history, team1=m.team1, team2=m.team2, venue=m.venue,
                toss_winner=m.toss_winner, toss_decision=m.toss_decision,
                season=int(m.season), is_playoff=bool(m.is_playoff),
            )
            rows.append({**{k: getattr(m, k) for k in KEY_COLUMNS}, **feats, TARGET: int(m.team1_win)})
        history.add_day(day)
    return pd.DataFrame(rows, columns=KEY_COLUMNS + FEATURE_COLUMNS + [TARGET])


def swap_teams(features: pd.DataFrame) -> pd.DataFrame:
    """Mirror a feature table so Team B becomes Team A (target flips too)."""
    out = features.copy()
    for left, right in _PAIRS:
        out[left], out[right] = features[right], features[left]
    for col in _NEGATE:
        out[col] = -features[col]
    for col in _COMPLEMENT:
        out[col] = 1 - features[col]
    if {"team1", "team2"} <= set(features.columns):
        out["team1"], out["team2"] = features["team2"], features["team1"]
    if TARGET in features.columns:
        out[TARGET] = 1 - features[TARGET]
    return out


def resolve_as_of(matches: pd.DataFrame, season: int, match_date: str | pd.Timestamp | None = None
                  ) -> pd.Timestamp:
    """The cutoff for a prediction scenario. History used is strictly before this date.

    - With a match date: that date.
    - Season already in the data, no date: the season's first match date
      (a pre-season view, so no result from that season is used).
    - Later season, no date: the day after the last match in the data.
    """
    if match_date is not None:
        return pd.Timestamp(match_date).normalize()
    in_season = matches.loc[matches["season"] == season, "date"]
    if not in_season.empty:
        return in_season.min()
    last = matches["date"].max()
    if season < last.year:
        # A season with no matches in the data (none exist today) -> pre-season of that year.
        return pd.Timestamp(year=season, month=1, day=1)
    return last + pd.Timedelta(days=1)


def scenario_features(
    matches: pd.DataFrame,
    *,
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    season: int,
    match_date: str | pd.Timestamp | None = None,
    is_playoff: bool = False,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Inference entry point: normalise names, pick the cutoff, return one feature row.

    Returns (features, as_of). Only matches strictly before `as_of` are used.
    """
    team1, team2 = mappings.normalize_team(team1), mappings.normalize_team(team2)
    toss_winner = mappings.normalize_team(toss_winner)
    venue = mappings.normalize_venue(venue)
    as_of = resolve_as_of(matches, season, match_date)
    history = History.from_matches(matches, before=as_of)
    feats = match_features(history, team1=team1, team2=team2, venue=venue,
                           toss_winner=toss_winner, toss_decision=toss_decision,
                           season=season, is_playoff=is_playoff)
    return pd.DataFrame([feats], columns=FEATURE_COLUMNS), as_of


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def main() -> int:
    matches = load_clean()
    features = build_training_features(matches)
    out = features.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    text = out.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    FEATURES_PATH.write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "feature_version": FEATURE_VERSION,
        "input": {"file": "data/processed/matches_clean.csv",
                  "sha256": hashlib.sha256(CLEAN_PATH.read_bytes()).hexdigest().upper()},
        "output": {"file": "data/processed/features.csv",
                   "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest().upper(),
                   "rows": len(features)},
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "constants": {"elo_start": ELO_START, "elo_k": ELO_K, "elo_season_carry": ELO_SEASON_CARRY,
                      "prior_strength": PRIOR_STRENGTH, "venue_prior_strength": VENUE_PRIOR_STRENGTH,
                      "recent_chase_window": RECENT_CHASE_WINDOW},
        "same_day_policy": "features use only matches on strictly earlier dates",
    }
    FEATURES_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {FEATURES_PATH.relative_to(ROOT)} ({len(features):,} rows, "
          f"{len(FEATURE_COLUMNS)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
