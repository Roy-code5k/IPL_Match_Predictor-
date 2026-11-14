"""Leakage, chronology, and consistency tests for the feature builder.

Most tests use small synthetic histories so they run without the Kaggle data.
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from src.features import builder as fb
from src.features.builder import FEATURE_COLUMNS, History

TEAMS = ["Mumbai Indians", "Chennai Super Kings", "Kolkata Knight Riders", "Rajasthan Royals"]
VENUES = ["Wankhede Stadium, Mumbai", "MA Chidambaram Stadium, Chennai", "Eden Gardens, Kolkata"]


def match(match_id, date, team1, team2, winner, venue=VENUES[0], toss_winner=None,
          toss_decision="field", season=None, is_playoff=False):
    date = pd.Timestamp(date)
    return {
        "match_id": match_id, "date": date, "season": season or date.year,
        "team1": team1, "team2": team2, "venue": venue,
        "toss_winner": toss_winner or team1, "toss_decision": toss_decision,
        "winner": winner, "team1_win": int(winner == team1), "is_playoff": is_playoff,
    }


def synthetic_matches(n=120, seed=7):
    """Random league over three seasons with plenty of same-day double headers."""
    rng = random.Random(seed)
    rows, day = [], pd.Timestamp("2010-04-01")
    for i in range(n):
        if i % 40 == 0 and i:
            day = pd.Timestamp(f"{day.year + 1}-04-01")
        elif rng.random() < 0.6:
            day += pd.Timedelta(days=1)
        a, b = rng.sample(TEAMS, 2)
        rows.append(match(1000 + i, day, a, b, rng.choice([a, b]), venue=rng.choice(VENUES),
                          toss_winner=rng.choice([a, b]), toss_decision=rng.choice(["bat", "field"]),
                          is_playoff=(i % 40) >= 37))
    return pd.DataFrame(rows)


def feats(df):
    return fb.build_training_features(df).set_index("match_id")[FEATURE_COLUMNS]


# --- hand-checked values ------------------------------------------------------

def test_known_small_history():
    mi, csk = TEAMS[0], TEAMS[1]
    df = pd.DataFrame([
        match(1, "2010-04-01", mi, csk, mi, toss_decision="bat"),     # MI bat first and win
        match(2, "2010-04-02", csk, mi, csk, toss_decision="field"),  # CSK chase and win
        match(3, "2010-04-03", mi, csk, mi),
    ])
    f = feats(df)

    first = f.loc[1]
    assert first["win_rate_a"] == 0.5 and first["elo_diff"] == 0 and first["h2h_played"] == 0

    third = f.loc[3]                     # MI vs CSK after one win each
    assert third["win_rate_a"] == pytest.approx(0.5)
    assert third["h2h_played"] == 2 and third["h2h_win_rate_a"] == pytest.approx(0.5)
    assert third["form5_a"] == pytest.approx((1 + 2.5) / (2 + 5))
    assert third["season_played_a"] == 2
    # chases so far: match 1 lost the chase, match 2 won it -> 1 of 2
    assert third["venue_chase_rate"] == pytest.approx((1 + 10 * ((1 + 5) / (2 + 10))) / (2 + 10))

    # Elo after match 1: MI +10, CSK -10. Match 2: CSK (1490) beats MI (1510).
    exp = fb.elo_expected(1490, 1510)
    gain = 20 * (1 - exp)
    assert third["elo_a"] == pytest.approx(1510 - gain)
    assert third["elo_b"] == pytest.approx(1490 + gain)


def test_new_season_regresses_elo_and_resets_season_stats():
    mi, csk = TEAMS[0], TEAMS[1]
    df = pd.DataFrame([
        match(1, "2010-04-01", mi, csk, mi),
        match(2, "2011-04-01", mi, csk, mi),
    ])
    second = feats(df).loc[2]
    assert second["elo_a"] == pytest.approx(1500 + 0.75 * 10)
    assert second["season_played_a"] == 0 and second["season_win_rate_a"] == 0.5
    assert second["prev_season_win_rate_a"] == pytest.approx((1 + 2.5) / (1 + 5))


def test_home_flags_respect_no_home_seasons():
    mi, csk = TEAMS[0], TEAMS[1]
    df = pd.DataFrame([
        match(1, "2019-04-01", mi, csk, mi, venue="Wankhede Stadium, Mumbai"),
        match(2, "2022-04-01", mi, csk, mi, venue="Wankhede Stadium, Mumbai"),
        match(3, "2023-04-01", csk, mi, mi, venue="Wankhede Stadium, Mumbai"),
    ])
    f = feats(df)
    assert (f.loc[1, "home_a"], f.loc[1, "home_b"]) == (1.0, 0.0)
    assert (f.loc[2, "home_a"], f.loc[2, "home_b"]) == (0.0, 0.0)   # 2022 bubble season
    assert (f.loc[3, "home_a"], f.loc[3, "home_b"]) == (0.0, 1.0)


# --- leakage -------------------------------------------------------------------

def test_same_day_matches_do_not_see_each_other():
    mi, csk, kkr = TEAMS[:3]
    base = [match(1, "2010-04-01", mi, csk, mi), match(2, "2010-04-02", mi, kkr, mi),
            match(3, "2010-04-02", csk, mi, csk)]
    flipped = [dict(r) for r in base]
    flipped[1].update(winner=kkr, team1_win=0)       # change match 2's result
    a, b = feats(pd.DataFrame(base)), feats(pd.DataFrame(flipped))
    pd.testing.assert_series_equal(a.loc[3], b.loc[3])  # same-day match 3 is unaffected


def test_changing_a_result_only_affects_later_dates():
    df = synthetic_matches()
    original = feats(df)
    for idx in [5, 40, 77, 110]:
        changed = df.copy()
        row = changed.loc[idx]
        loser = row["team2"] if row["winner"] == row["team1"] else row["team1"]
        changed.loc[idx, ["winner", "team1_win"]] = [loser, 1 - row["team1_win"]]
        new = feats(changed)
        cutoff = row["date"]
        same_or_before = df.loc[df["date"] <= cutoff, "match_id"]
        pd.testing.assert_frame_equal(original.loc[same_or_before], new.loc[same_or_before])
        later = df.loc[df["date"] > cutoff, "match_id"]
        if len(later):
            assert not original.loc[later].equals(new.loc[later]), "later matches should react"


def test_future_matches_never_change_earlier_features():
    df = synthetic_matches()
    full = feats(df)
    cut = df["date"].iloc[60]
    partial = feats(df[df["date"] <= cut])
    pd.testing.assert_frame_equal(full.loc[partial.index], partial)


def test_input_row_order_does_not_matter():
    df = synthetic_matches()
    shuffled = df.sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(feats(df), feats(shuffled).loc[feats(df).index])


def test_history_rejects_out_of_order_days():
    df = synthetic_matches()
    history = History.from_matches(df)
    with pytest.raises(ValueError):
        history.add_day(df[df["date"] == df["date"].min()])


# --- training and inference agree ---------------------------------------------

def test_inference_matches_training_features():
    df = synthetic_matches()
    train = feats(df)
    for idx in [0, 1, 39, 40, 41, 80, 119]:
        m = df.loc[idx]
        row, as_of = fb.scenario_features(
            df, team1=m.team1, team2=m.team2, venue=m.venue, toss_winner=m.toss_winner,
            toss_decision=m.toss_decision, season=int(m.season), match_date=m.date,
            is_playoff=bool(m.is_playoff),
        )
        assert as_of == m.date
        np.testing.assert_allclose(row.iloc[0].to_numpy(dtype=float),
                                   train.loc[m.match_id].to_numpy(dtype=float), rtol=0, atol=1e-12)


def test_scenario_accepts_raw_names_and_validates():
    df = synthetic_matches()
    row, _ = fb.scenario_features(df, team1="Mumbai Indians", team2="Chennai Super Kings",
                                  venue="Wankhede Stadium", toss_winner="Mumbai Indians",
                                  toss_decision="bat", season=2012)
    assert list(row.columns) == FEATURE_COLUMNS and row.notna().all().all()
    with pytest.raises(ValueError):
        fb.scenario_features(df, team1="Mumbai Indians", team2="Mumbai Indians",
                             venue="Wankhede Stadium", toss_winner="Mumbai Indians",
                             toss_decision="bat", season=2012)
    with pytest.raises(ValueError):
        fb.scenario_features(df, team1="Mumbai Indians", team2="Chennai Super Kings",
                             venue="Wankhede Stadium", toss_winner="Rajasthan Royals",
                             toss_decision="bat", season=2012)


def test_resolve_as_of():
    df = synthetic_matches()
    assert fb.resolve_as_of(df, 2011, "2011-05-03") == pd.Timestamp("2011-05-03")
    assert fb.resolve_as_of(df, 2011) == df.loc[df["season"] == 2011, "date"].min()
    assert fb.resolve_as_of(df, 2030) == df["date"].max() + pd.Timedelta(days=1)


# --- symmetry ------------------------------------------------------------------

def test_swap_classification_covers_every_feature_once():
    groups = ([c for pair in fb._PAIRS for c in pair] + fb._NEGATE + fb._COMPLEMENT + fb._UNCHANGED)
    assert sorted(groups) == sorted(FEATURE_COLUMNS)


def test_swapping_teams_mirrors_features():
    df = synthetic_matches()
    swapped_input = df.rename(columns={"team1": "team2", "team2": "team1"})
    swapped_input["team1_win"] = 1 - df["team1_win"]
    direct = fb.build_training_features(swapped_input).set_index("match_id")
    mirrored = fb.swap_teams(fb.build_training_features(df)).set_index("match_id")
    cols = FEATURE_COLUMNS + ["team1", "team2", "team1_win"]
    pd.testing.assert_frame_equal(direct[cols], mirrored[cols], check_exact=False, atol=1e-12)


def test_elo_is_zero_sum():
    df = synthetic_matches()
    history = History.from_matches(df)
    total = sum(history.ratings[t] for t in TEAMS)
    assert total == pytest.approx(1500 * len(TEAMS))


# --- real data -----------------------------------------------------------------

@pytest.mark.skipif(not fb.CLEAN_PATH.exists(), reason="run python -m src.data.cleaning first")
def test_real_features_shape_and_ranges():
    f = fb.build_training_features(fb.load_clean())
    assert len(f) == 1090 and f[FEATURE_COLUMNS].notna().all().all()
    rates = [c for c in FEATURE_COLUMNS
             if not c.endswith("_diff") and ("rate" in c or c.startswith("form") or c == "elo_prob_a")]
    assert len(rates) == 17
    assert ((f[rates] >= 0) & (f[rates] <= 1)).all().all()
    diffs = [c for c in FEATURE_COLUMNS if c.endswith("_diff") and c != "elo_diff"]
    assert ((f[diffs] >= -1) & (f[diffs] <= 1)).all().all()
    assert not math.isnan(f["elo_diff"].abs().max())
