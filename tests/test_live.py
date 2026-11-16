"""Tests for live (in-match) states. Synthetic matches and deliveries only."""

import random

import numpy as np
import pandas as pd
import pytest

from src.features import builder as fb
from src.features import live as lf
from tests.test_features import synthetic_matches

RAW_COLUMNS = ["match_id", "inning", "batting_team", "over", "total_runs", "is_wicket", "dismissal_kind"]


def simulate(matches: pd.DataFrame, seed=3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ball-by-ball data consistent with each match; the winner is set from the simulation."""
    rng = random.Random(seed)
    rows, matches = [], matches.copy()
    for i, m in matches.iterrows():
        bat_first = fb.bat_first_team(m.team1, m.team2, m.toss_winner, m.toss_decision)
        chaser = m.team2 if bat_first == m.team1 else m.team1
        totals = {}
        for inning, team in ((1, bat_first), (2, chaser)):
            runs = wkts = 0
            for over in range(20):
                for _ in range(6):
                    r = rng.choice([0, 0, 1, 1, 2, 4, 6])
                    out = rng.random() < 0.05
                    kind = "retired hurt" if out and rng.random() < 0.1 else ("caught" if out else "NA")
                    rows.append([m.match_id, inning, team, over, r, int(out), kind])
                    runs += r
                    wkts += int(out and kind != "retired hurt")
                    if wkts == 10 or (inning == 2 and runs > totals[1]):
                        break
                if wkts == 10 or (inning == 2 and runs > totals[1]):
                    break
            totals[inning] = runs
        winner = chaser if totals[2] > totals[1] else bat_first
        matches.loc[i, "winner"] = winner
        matches.loc[i, "team1_win"] = int(winner == m.team1)
    return matches, pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    matches, raw = simulate(synthetic_matches(n=80))
    path = tmp_path_factory.mktemp("live") / "deliveries.csv"
    raw.to_csv(path, index=False)
    deliveries = lf.load_deliveries(path)
    features = fb.build_training_features(matches)
    full = set(matches["match_id"])
    states = lf.build_live_states(matches, features, deliveries, full)
    return matches, deliveries, features, full, states


# --- state features -----------------------------------------------------------

def test_state_features_known_values():
    ctx = {"par": 160.0, "elo_prob_bf": 0.6, "venue_chase_rate": 0.55}
    f1 = lf.state_features(innings=1, overs_done=10, runs=90, wickets=2, **ctx)
    assert f1["balls_left"] == 60 and f1["run_rate"] == 9.0
    assert f1["projected_minus_par"] == pytest.approx(180 - 160)
    f2 = lf.state_features(innings=2, overs_done=15, runs=150, wickets=4, target=181, **ctx)
    assert f2["runs_needed"] == 31 and f2["wickets_left"] == 6 and f2["balls_left"] == 30
    assert f2["required_rate"] == pytest.approx(6.2)
    assert f2["target_minus_par"] == pytest.approx(21)
    assert list(f1) == lf.INNINGS1_FEATURES and list(f2) == lf.INNINGS2_FEATURES


@pytest.mark.parametrize("kwargs", [
    {"innings": 1, "overs_done": 0, "runs": 0, "wickets": 0},           # before the first over: pre-match model
    {"innings": 1, "overs_done": 20, "runs": 150, "wickets": 5},        # innings over: that is the break
    {"innings": 2, "overs_done": 5, "runs": 50, "wickets": 10},         # all out: chase over
    {"innings": 2, "overs_done": 5, "runs": 181, "wickets": 2, "target": 181},  # target reached
    {"innings": 2, "overs_done": 5, "runs": 50, "wickets": 2},          # no target
    {"innings": 3, "overs_done": 1, "runs": 5, "wickets": 0},
])
def test_state_features_rejects_finished_or_invalid_states(kwargs):
    with pytest.raises(ValueError):
        lf.state_features(par=160, elo_prob_bf=0.5, venue_chase_rate=0.5, **kwargs)


def test_par_score():
    assert lf.par_score([]) == lf.PAR_PRIOR
    old = [100.0] * 500
    recent = [200.0] * lf.PAR_WINDOW
    expected = (200 * lf.PAR_WINDOW + lf.PAR_PRIOR_STRENGTH * lf.PAR_PRIOR) / (lf.PAR_WINDOW + lf.PAR_PRIOR_STRENGTH)
    assert lf.par_score(old + recent) == pytest.approx(expected)


# --- loading and progress -------------------------------------------------------

def test_retired_hurt_is_not_a_wicket(world):
    _, deliveries, *_ = world
    retired = deliveries["dismissal_kind"] == "retired hurt"
    assert retired.any()
    assert (deliveries.loc[retired, "wicket"] == 0).all()


def test_over_progress_matches_a_manual_count(world):
    _, deliveries, *_ = world
    progress = lf.over_progress(deliveries).set_index(["match_id", "innings", "overs_done"])
    mid = deliveries["match_id"].iloc[0]
    one = deliveries[(deliveries["match_id"] == mid) & (deliveries["inning"] == 1)]
    for overs in (1, 6, 10):
        upto = one[one["over"] < overs]
        assert progress.loc[(mid, 1, overs), "runs"] == upto["total_runs"].sum()
        assert progress.loc[(mid, 1, overs), "wickets"] == upto["wicket"].sum()
    last = one["over"].max() + 1
    assert not progress.loc[(mid, 1, last), "in_progress"]


def test_full_length_ids():
    raw = pd.DataFrame({"id": ["1", "2", "3", "4"], "method": ["NA", "D/L", "NA", "NA"],
                        "target_overs": ["20", "20", "12", "NA"]})
    assert lf.full_length_ids(raw) == {1}


# --- training states ------------------------------------------------------------

def test_states_are_in_progress_and_labelled(world):
    matches, *_, states = world
    breaks = states[(states["innings"] == 2) & (states["overs_done"] == 0)]
    assert len(breaks) == len(matches)
    two = states[states["innings"] == 2]
    assert (two["runs_needed"] > 0).all() and (two["wickets_left"] > 0).all()
    one = states[states["innings"] == 1]
    assert one["overs_done"].between(1, 19).all() and (one["wickets"] < 10).all()
    winners = matches.set_index("match_id")["winner"]
    assert (states[lf.TARGET] == (states["match_id"].map(winners) == states["bat_first"]).astype(int)).all()


def test_excluded_matches_produce_no_states(world):
    matches, deliveries, features, full, _ = world
    dropped = sorted(full)[:5]
    states = lf.build_live_states(matches, features, deliveries, full - set(dropped))
    assert not states["match_id"].isin(dropped).any()


def test_par_uses_only_earlier_dates(world):
    matches, deliveries, features, full, states = world
    dates = matches.set_index("match_id")["date"]
    day = sorted(matches["date"].unique())[len(matches["date"].unique()) // 2]
    victim = matches.loc[matches["date"] == day, "match_id"].iloc[0]

    changed = deliveries.copy()
    first = changed.index[(changed["match_id"] == victim) & (changed["inning"] == 1)][0]
    changed.loc[first, "total_runs"] += 50  # the first-innings total of one match changes
    after = lf.build_live_states(matches, features, changed, full)

    others = states["match_id"] != victim
    same_or_earlier = others & (states["match_id"].map(dates) <= day)
    later = states["match_id"].map(dates) > day
    pd.testing.assert_frame_equal(states[same_or_earlier], after[same_or_earlier])
    assert not np.allclose(states.loc[later, "par_score"], after.loc[later, "par_score"])


# --- inference --------------------------------------------------------------------

def test_inference_rows_equal_training_rows(world):
    matches, deliveries, _, full, states = world
    totals = lf.totals_table(matches, deliveries, full)
    late = matches.sort_values("date").iloc[-3]
    mine = states[states["match_id"] == late.match_id]
    common = dict(team1=late.team1, team2=late.team2, venue=late.venue, toss_winner=late.toss_winner,
                  toss_decision=late.toss_decision, season=int(late.season), match_date=late.date,
                  is_playoff=bool(late.is_playoff))

    row1 = mine[(mine["innings"] == 1) & (mine["overs_done"] == 6)].iloc[0]
    got, bat_first = lf.live_scenario(matches, totals, innings=1, overs_done=6, runs=int(row1.runs),
                                      wickets=int(row1.wickets), **common)
    assert bat_first == row1.bat_first
    np.testing.assert_allclose(got.iloc[0].to_numpy(dtype=float),
                               row1[lf.INNINGS1_FEATURES].to_numpy(dtype=float))

    row2 = mine[(mine["innings"] == 2) & (mine["overs_done"] == 0)].iloc[0]
    got, _ = lf.live_scenario(matches, totals, innings=2, overs_done=0, runs=0, wickets=0,
                              first_innings_total=int(row2.target) - 1, **common)
    np.testing.assert_allclose(got.iloc[0].to_numpy(dtype=float),
                               row2[lf.INNINGS2_FEATURES].to_numpy(dtype=float))
