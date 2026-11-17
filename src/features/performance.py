"""Experimental pre-match batting/bowling form, from strictly earlier matches.

The serving release still uses builder.py. This module adds 26 features without
changing its feature specification. Both training and scenarios use FormHistory.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import mappings
from src.features import builder as fb
from src.features import live as lf

VERSION = "prematch-performance-1.0.0"
TEAM_WINDOW = 10
LEAGUE_WINDOW = 60  # earlier team innings, to adapt the smoothing prior to the era
RAW_COLUMNS = ["match_id", "inning", "batting_team", "bowling_team", "over",
               "total_runs", "extras_type", "is_wicket", "dismissal_kind"]
COUNTS = ["runs", "balls", "wickets", "powerplay_runs", "powerplay_balls", "death_runs", "death_balls"]
# name -> numerator, denominator, scale, prior legal balls, cold-start value
RATES = {
    "run_rate": ("runs", "balls", 6.0, 240.0, 8.0),
    "wickets_per_120": ("wickets", "balls", 120.0, 240.0, 6.0),
    "powerplay_rate": ("powerplay_runs", "powerplay_balls", 6.0, 72.0, 7.5),
    "death_rate": ("death_runs", "death_balls", 6.0, 60.0, 10.0),
}
METRICS = [f"{side}_{name}" for side in ("bat", "bowl") for name in RATES]
EXTRA_FEATURES = [f"{name}_{suffix}" for name in METRICS for suffix in ("a", "b", "diff")]
EXTRA_FEATURES += ["performance_matches_a", "performance_matches_b"]
FEATURES = fb.FEATURE_COLUMNS + EXTRA_FEATURES


def load_deliveries(path: Path = lf.RAW_DELIVERIES) -> pd.DataFrame:
    return pd.read_csv(path, usecols=RAW_COLUMNS)


def aggregate_performance(matches: pd.DataFrame, deliveries: pd.DataFrame,
                          eligible_ids: set[int]) -> pd.DataFrame:
    """Two rows per eligible match: each team's batting and bowling counts.

    Use full-length, decided matches only. Wides/no-balls contribute runs but not
    legal balls. Team wickets include run-outs, but not retired hurt. Super overs
    are excluded. Bowling run rate is *team runs conceded*, including byes.
    """
    selected = matches[matches.match_id.isin(eligible_ids)].copy()
    d = deliveries[deliveries.match_id.isin(selected.match_id) & deliveries.inning.isin([1, 2])].copy()
    for column in ("batting_team", "bowling_team"):
        d[column] = d[column].map(mappings.normalize_team)
    d["balls"] = (~d.extras_type.isin(["wides", "noballs"])).astype(int)
    d["wickets"] = ((d.is_wicket == 1) & ~d.dismissal_kind.isin(lf.NOT_A_WICKET)).astype(int)
    d["runs"] = d.total_runs
    for name, mask in (("powerplay", d.over.between(0, 5)), ("death", d.over.between(15, 19))):
        for count in ("runs", "balls"):
            d[f"{name}_{count}"] = np.where(mask, d[count], 0)
    innings = d.groupby(["match_id", "inning", "batting_team", "bowling_team"], as_index=False)[COUNTS].sum()
    by_match = {mid: group for mid, group in innings.groupby("match_id", sort=False)}
    rows = []
    for m in fb._ordered(selected).itertuples(index=False):
        group = by_match.get(m.match_id)
        if group is None or len(group) != 2 or set(group.inning) != {1, 2}:
            raise ValueError(f"Match {m.match_id} needs both regulation innings for performance history")
        records = {team: {"match_id": m.match_id, "date": m.date, "team": team} for team in (m.team1, m.team2)}
        if set(group.batting_team) != set(records) or set(group.bowling_team) != set(records):
            raise ValueError(f"Match {m.match_id}: deliveries and match teams disagree")
        for innings_row in group.itertuples(index=False):
            if innings_row.batting_team == innings_row.bowling_team:
                raise ValueError(f"Match {m.match_id}: batting and bowling teams must differ")
            for side, team in (("bat", innings_row.batting_team), ("bowl", innings_row.bowling_team)):
                records[team].update({f"{side}_{key}": int(getattr(innings_row, key)) for key in COUNTS})
        rows.extend(records.values())
    columns = ["match_id", "date", "team"] + [f"{side}_{key}" for side in ("bat", "bowl") for key in COUNTS]
    return pd.DataFrame(rows, columns=columns).sort_values(["date", "match_id", "team"]).reset_index(drop=True)


@dataclass
class FormHistory:
    teams: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=TEAM_WINDOW)))
    league: deque = field(default_factory=lambda: deque(maxlen=LEAGUE_WINDOW))

    def add(self, records: pd.DataFrame) -> None:
        for row in records.to_dict("records"):
            self.teams[row["team"]].append(row)
            self.league.append(row)

    @classmethod
    def before(cls, records: pd.DataFrame, date: pd.Timestamp) -> "FormHistory":
        history = cls()
        history.add(records.loc[records.date < date].sort_values(["date", "match_id", "team"]))
        return history

    def team_features(self, team: str) -> dict[str, float]:
        recent = self.teams.get(team, [])
        out = {"performance_matches": float(len(recent))}
        for side in ("bat", "bowl"):
            for name, (numerator, denominator, scale, strength, fallback) in RATES.items():
                num, den = f"{side}_{numerator}", f"{side}_{denominator}"
                prior_balls = sum(row[den] for row in self.league)
                prior = sum(row[num] for row in self.league) / prior_balls if prior_balls else fallback / scale
                out[f"{side}_{name}"] = scale * (sum(row[num] for row in recent) + strength * prior) / (
                    sum(row[den] for row in recent) + strength)
        return out

    def matchup(self, team1: str, team2: str) -> dict[str, float]:
        a, b = self.team_features(team1), self.team_features(team2)
        out = {}
        for name in METRICS:
            out.update({f"{name}_a": a[name], f"{name}_b": b[name], f"{name}_diff": a[name] - b[name]})
        out.update(performance_matches_a=a["performance_matches"], performance_matches_b=b["performance_matches"])
        return out


def enrich_features(base: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Attach form before each feature date, updating whole days only afterwards."""
    base = base.copy()
    base["date"] = pd.to_datetime(base.date)
    history = FormHistory()
    days = iter(records.sort_values(["date", "match_id", "team"]).groupby("date", sort=True))
    upcoming = next(days, None)
    rows = []
    for date, day in fb._ordered(base).groupby("date", sort=True):
        while upcoming is not None and upcoming[0] < date:
            history.add(upcoming[1])
            upcoming = next(days, None)
        for row in day.to_dict("records"):
            rows.append({**row, **history.matchup(row["team1"], row["team2"])})
    return pd.DataFrame(rows, columns=fb.KEY_COLUMNS + FEATURES + [fb.TARGET])


def scenario_features(matches: pd.DataFrame, records: pd.DataFrame, **setup) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Experimental inference uses exactly the same past-only performance definitions."""
    base, cutoff = fb.scenario_features(matches, **setup)
    extra = FormHistory.before(records, cutoff).matchup(
        mappings.normalize_team(setup["team1"]), mappings.normalize_team(setup["team2"]))
    return pd.DataFrame([{**base.iloc[0].to_dict(), **extra}], columns=FEATURES), cutoff


def swap_teams(features: pd.DataFrame) -> pd.DataFrame:
    out = fb.swap_teams(features)
    for name in METRICS + ["performance_matches"]:
        if f"{name}_a" in features:
            out[f"{name}_a"], out[f"{name}_b"] = features[f"{name}_b"], features[f"{name}_a"]
    for name in METRICS:
        if f"{name}_diff" in features:
            out[f"{name}_diff"] = -features[f"{name}_diff"]
    return out
