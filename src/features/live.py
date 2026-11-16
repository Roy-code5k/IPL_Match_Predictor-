"""In-match (live) states from deliveries.csv.

A live prediction is made at the end of an over, or at the innings break, and gives
P(the side batting first wins). The match state comes from the ball-by-ball data;
the pre-match context (Elo, venue chase rate) comes from `src.features.builder`,
so it uses only matches on strictly earlier dates.

States (all "in progress": the innings continues after the snapshot):
- innings 1 after 1-19 completed overs;
- innings 2 at the break (0 overs) and after 1-19 completed overs.

Rain-affected matches (D/L, or fewer than 20 overs a side) are left out: their
target is not "first-innings total + 1 in 20 overs". Super overs are ignored.

One definition serves training and inference:
    state_features(...)   features for one state
    build_live_states(...) every training state, walking matches in date order
    live_scenario(...)     one inference row, using only history before the cutoff
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import mappings
from src.data.inspect_raw import load_raw
from src.features import builder as fb

ROOT = Path(__file__).resolve().parents[2]
RAW_DELIVERIES = ROOT / "data" / "raw" / "deliveries.csv"

LIVE_VERSION = "1.0.0"
OVERS = 20
BALLS = 6 * OVERS
NOT_A_WICKET = {"retired hurt"}  # the batter may return, so no wicket is used up

# Par score: smoothed mean first-innings total of recent full-length matches (strictly earlier dates).
# Tracks league-wide scoring changes, such as the jump in 2023-2024.
PAR_WINDOW = 60
PAR_PRIOR = 160.0
PAR_PRIOR_STRENGTH = 10.0

TARGET = "bat_first_win"
KEY_COLUMNS = ["match_id", "date", "season", "bat_first", "chaser", "innings", "overs_done"]
CONTEXT = ["par_score", "elo_prob_bf", "venue_chase_rate"]
INNINGS1_FEATURES = ["overs_done", "balls_left", "runs", "wickets", "run_rate",
                     "projected_minus_par"] + CONTEXT
INNINGS2_FEATURES = ["overs_done", "balls_left", "target", "runs_needed", "wickets_left",
                     "required_rate", "target_minus_par"] + CONTEXT
FEATURES_BY_INNINGS = {1: INNINGS1_FEATURES, 2: INNINGS2_FEATURES}


def par_score(totals: list[float]) -> float:
    """Smoothed mean of the last PAR_WINDOW first-innings totals (oldest first)."""
    recent = totals[-PAR_WINDOW:]
    return (sum(recent) + PAR_PRIOR_STRENGTH * PAR_PRIOR) / (len(recent) + PAR_PRIOR_STRENGTH)


def state_features(*, innings: int, overs_done: int, runs: int, wickets: int,
                   target: int | None = None, par: float, elo_prob_bf: float,
                   venue_chase_rate: float) -> dict[str, float]:
    """Features for one in-progress state, from the batting-first side's point of view."""
    if innings == 1:
        if not 1 <= overs_done < OVERS:
            raise ValueError("innings 1 states need 1-19 completed overs (use the pre-match model before that)")
    elif innings == 2:
        if not 0 <= overs_done < OVERS:
            raise ValueError("innings 2 states need 0-19 completed overs")
        if target is None or not 0 <= runs < target:
            raise ValueError("innings 2 needs a target above the runs scored so far")
    else:
        raise ValueError("innings must be 1 or 2")
    if not 0 <= wickets < 10:
        raise ValueError("wickets must be 0-9 for an innings in progress")

    balls_left = BALLS - 6 * overs_done
    context = {"par_score": par, "elo_prob_bf": elo_prob_bf, "venue_chase_rate": venue_chase_rate}
    if innings == 1:
        run_rate = runs * 6 / (6 * overs_done)
        return {"overs_done": float(overs_done), "balls_left": float(balls_left), "runs": float(runs),
                "wickets": float(wickets), "run_rate": run_rate,
                "projected_minus_par": run_rate * OVERS - par, **context}
    runs_needed = target - runs
    return {"overs_done": float(overs_done), "balls_left": float(balls_left), "target": float(target),
            "runs_needed": float(runs_needed), "wickets_left": float(10 - wickets),
            "required_rate": runs_needed * 6 / balls_left, "target_minus_par": target - par, **context}


# --- data loading ---------------------------------------------------------------

def load_deliveries(path: Path = RAW_DELIVERIES) -> pd.DataFrame:
    """Regulation innings only, canonical team names, wickets that use up a batter."""
    d = pd.read_csv(path, usecols=["match_id", "inning", "batting_team", "over", "total_runs",
                                   "is_wicket", "dismissal_kind"])
    d = d[d["inning"].isin([1, 2])].copy()
    d["batting_team"] = d["batting_team"].map(mappings.normalize_team)
    d["wicket"] = ((d["is_wicket"] == 1) & ~d["dismissal_kind"].isin(NOT_A_WICKET)).astype(int)
    return d


def full_length_ids(raw_matches: pd.DataFrame | None = None) -> set[int]:
    """Match ids with no D/L adjustment and a 20-over target."""
    raw = load_raw("matches.csv") if raw_matches is None else raw_matches
    overs = pd.to_numeric(raw["target_overs"], errors="coerce")
    ok = (raw["method"] != "D/L") & (overs == OVERS)
    return set(raw.loc[ok, "id"].astype(int))


def over_progress(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Score at the end of every over, and whether the innings went on after it."""
    per_over = (deliveries.groupby(["match_id", "inning", "over"], as_index=False)
                .agg(runs=("total_runs", "sum"), wickets=("wicket", "sum")))
    per_over = per_over.sort_values(["match_id", "inning", "over"])
    grouped = per_over.groupby(["match_id", "inning"])
    per_over["runs"] = grouped["runs"].cumsum()
    per_over["wickets"] = grouped["wickets"].cumsum()
    per_over["in_progress"] = per_over["over"] < grouped["over"].transform("max")
    per_over["overs_done"] = per_over["over"] + 1
    return per_over.rename(columns={"inning": "innings"})


def first_innings_totals(deliveries: pd.DataFrame) -> pd.Series:
    return deliveries[deliveries["inning"] == 1].groupby("match_id")["total_runs"].sum()


def context_from_features(row: pd.Series | dict) -> tuple[float, float]:
    """(Elo probability of the batting-first side, venue chase rate) from a pre-match feature row."""
    elo = row["elo_prob_a"] if row["a_bats_first"] == 1 else 1 - row["elo_prob_a"]
    return float(elo), float(row["venue_chase_rate"])


# --- training states --------------------------------------------------------------

def build_live_states(matches: pd.DataFrame, features: pd.DataFrame, deliveries: pd.DataFrame,
                      full_length: set[int]) -> pd.DataFrame:
    """Every in-progress state of every full-length match, with the final result as target."""
    table = totals_table(matches, deliveries, full_length)
    m = fb._ordered(matches[matches["match_id"].isin(set(table["match_id"]))])
    feats = features.set_index("match_id")
    totals = table.set_index("match_id")["total"]
    batting = deliveries[deliveries["inning"] == 1].groupby("match_id")["batting_team"].first()
    progress = over_progress(deliveries)
    progress = progress[progress["in_progress"]].set_index("match_id")

    history: list[float] = []  # first-innings totals of earlier dates
    rows = []
    for _, day in m.groupby("date", sort=True):
        par = par_score(history)
        for match in day.itertuples(index=False):
            bat_first = fb.bat_first_team(match.team1, match.team2, match.toss_winner, match.toss_decision)
            if batting[match.match_id] != bat_first:
                raise ValueError(f"Match {match.match_id}: toss and ball-by-ball batting order disagree")
            chaser = match.team2 if bat_first == match.team1 else match.team1
            elo_bf, chase_rate = context_from_features(feats.loc[match.match_id])
            target = int(totals[match.match_id]) + 1
            key = {"match_id": match.match_id, "date": match.date, "season": int(match.season),
                   "bat_first": bat_first, "chaser": chaser}
            label = {TARGET: int(match.winner == bat_first)}
            context = {"par": par, "elo_prob_bf": elo_bf, "venue_chase_rate": chase_rate}

            # Innings break, then every completed over that the innings outlasted.
            states = [(2, 0, 0, 0)]
            if match.match_id in progress.index:
                snap = progress.loc[[match.match_id]]
                states += [(int(s.innings), int(s.overs_done), int(s.runs), int(s.wickets))
                           for s in snap.itertuples(index=False)]
            for innings, overs_done, runs, wickets in sorted(states):
                f = state_features(innings=innings, overs_done=overs_done, runs=runs, wickets=wickets,
                                   target=target if innings == 2 else None, **context)
                rows.append({**key, "innings": innings, "overs_done": overs_done, **f, **label})
        history += [float(totals[i]) for i in day["match_id"]]

    columns = list(dict.fromkeys(KEY_COLUMNS + INNINGS1_FEATURES + INNINGS2_FEATURES)) + [TARGET]
    return pd.DataFrame(rows).reindex(columns=columns)


# --- inference --------------------------------------------------------------------

def live_scenario(matches: pd.DataFrame, totals_by_date: pd.DataFrame, *, team1: str, team2: str,
                  venue: str, toss_winner: str, toss_decision: str, season: int,
                  innings: int, overs_done: int, runs: int, wickets: int,
                  first_innings_total: int | None = None,
                  match_date: str | pd.Timestamp | None = None,
                  is_playoff: bool = False,
                  history_for=None) -> tuple[pd.DataFrame, str]:
    """One live feature row (columns for that innings) and the name of the batting-first side.

    `totals_by_date` has columns `date`, `match_id`, `total` for full-length matches
    (see `totals_table`). Only rows strictly before the cutoff are used.
    `history_for` is passed to `scenario_features` (an optional history cache).
    """
    pre, as_of = fb.scenario_features(matches, team1=team1, team2=team2, venue=venue,
                                      toss_winner=toss_winner, toss_decision=toss_decision,
                                      season=season, match_date=match_date, is_playoff=is_playoff,
                                      history_for=history_for)
    elo_bf, chase_rate = context_from_features(pre.iloc[0])
    earlier = totals_by_date[totals_by_date["date"] < as_of].sort_values(["date", "match_id"])
    par = par_score(earlier["total"].astype(float).tolist())
    target = None if innings == 1 else first_innings_total + 1
    f = state_features(innings=innings, overs_done=overs_done, runs=runs, wickets=wickets,
                       target=target, par=par, elo_prob_bf=elo_bf, venue_chase_rate=chase_rate)
    t1, t2 = mappings.normalize_team(team1), mappings.normalize_team(team2)
    bat_first = fb.bat_first_team(t1, t2, mappings.normalize_team(toss_winner), toss_decision)
    return pd.DataFrame([f], columns=FEATURES_BY_INNINGS[innings]), bat_first


def totals_table(matches: pd.DataFrame, deliveries: pd.DataFrame, full_length: set[int]) -> pd.DataFrame:
    """The matches live models use: decided, full-length, with a second innings.

    Columns `match_id`, `date`, `total` (first-innings runs), ordered by date then id.
    It is the training sample and the par-score history for both training and inference.
    """
    chased = set(deliveries.loc[deliveries["inning"] == 2, "match_id"])
    totals = first_innings_totals(deliveries).rename("total").reset_index()
    out = totals.merge(matches[["match_id", "date"]], on="match_id")
    out = out[out["match_id"].isin(full_length) & out["match_id"].isin(chased)]
    return out.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
