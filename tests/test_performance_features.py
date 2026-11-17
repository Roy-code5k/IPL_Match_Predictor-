"""Numeric, chronology and symmetry checks for experimental pre-match form."""

import numpy as np
import pandas as pd
import pytest

from src.features import builder as fb
from src.features import performance as pf
from tests.test_features import TEAMS, match


def deliveries_for(matches):
    rows = []
    for m in matches.itertuples(index=False):
        for innings, bat, bowl in [(1, m.team1, m.team2), (2, m.team2, m.team1)]:
            for over, runs, extras, wicket, kind in [(0, 2, None, 0, None), (0, 2, "wides", 0, None),
                                                     (5, 0, None, 1, "retired hurt"),
                                                     (15, 6, "noballs", 0, None),
                                                     (19, 1, None, 1, "run out")]:
                rows.append([m.match_id, innings, bat, bowl, over, runs, extras, wicket, kind])
        rows.append([m.match_id, 3, m.team1, m.team2, 0, 100, None, 1, "bowled"])
    return pd.DataFrame(rows, columns=pf.RAW_COLUMNS)


@pytest.fixture
def history():
    matches = pd.DataFrame([match(i, date, TEAMS[0], TEAMS[1], TEAMS[i % 2])
                            for i, date in enumerate(["2010-04-01", "2010-04-02", "2010-04-02", "2010-04-03"], 1)])
    deliveries = deliveries_for(matches)
    records = pf.aggregate_performance(matches, deliveries, set(matches.match_id))
    return matches, deliveries, records


def test_legal_balls_wickets_and_phase_boundaries(history):
    matches, _, records = history
    assert len(records) == 2 * len(matches)
    row = records.iloc[0]
    for side in ("bat", "bowl"):
        assert row[f"{side}_runs"] == 11  # wides and no-balls count; super overs do not
        assert row[f"{side}_balls"] == 3
        assert row[f"{side}_wickets"] == 1  # run-out counts; retired hurt does not
        assert row[f"{side}_powerplay_runs"] == 4
        assert row[f"{side}_powerplay_balls"] == 2
        assert row[f"{side}_death_runs"] == 7
        assert row[f"{side}_death_balls"] == 1


def test_missing_or_inconsistent_innings_fail_explicitly(history):
    matches, deliveries, _ = history
    with pytest.raises(ValueError, match="both regulation innings"):
        pf.aggregate_performance(matches, deliveries[deliveries.inning != 2], set(matches.match_id))
    bad = deliveries.copy()
    bad.loc[bad.inning == 2, "bowling_team"] = TEAMS[2]
    with pytest.raises(ValueError, match="teams disagree"):
        pf.aggregate_performance(matches, bad, set(matches.match_id))


def test_ineligible_matches_do_not_enter_form(history):
    matches, deliveries, _ = history
    records = pf.aggregate_performance(matches, deliveries, {1, 4})
    assert set(records.match_id) == {1, 4}


def test_rates_are_finite_without_history_or_death_overs(history):
    matches, deliveries, _ = history
    cold = pf.FormHistory().matchup(TEAMS[0], TEAMS[1])
    assert cold["bat_run_rate_a"] == 8
    assert cold["bat_powerplay_rate_a"] == 7.5
    assert cold["bat_death_rate_a"] == 10
    assert cold["performance_matches_a"] == 0
    records = pf.aggregate_performance(matches, deliveries[deliveries.over < 15], set(matches.match_id))
    form = pf.FormHistory.before(records, pd.Timestamp("2011-01-01"))
    features = form.matchup(TEAMS[0], TEAMS[2])
    assert np.isfinite(list(features.values())).all()
    assert features["bat_death_rate_a"] == 10
    assert features["bat_run_rate_a"] == pytest.approx(12.0)
    assert features["performance_matches_b"] == 0


def test_same_day_and_future_deliveries_cannot_change_earlier_features(history):
    matches, deliveries, records = history
    base = fb.build_training_features(matches)
    original = pf.enrich_features(base, records)
    changed = deliveries.copy()
    changed.loc[changed.match_id == 2, "total_runs"] += 100
    new = pf.enrich_features(base, pf.aggregate_performance(matches, changed, set(matches.match_id)))
    before = original.date <= pd.Timestamp("2010-04-02")
    pd.testing.assert_frame_equal(original[before], new[before])
    assert not original.loc[~before, pf.EXTRA_FEATURES].equals(new.loc[~before, pf.EXTRA_FEATURES])
    # Shuffling raw inputs and same-day matches must not alter the snapshot.
    shuffled = pf.aggregate_performance(matches.sample(frac=1, random_state=1),
                                         deliveries.sample(frac=1, random_state=2), set(matches.match_id))
    pd.testing.assert_frame_equal(original, pf.enrich_features(base.sample(frac=1, random_state=3), shuffled))


def test_scenario_and_training_features_match_in_both_team_orders(history):
    matches, _, records = history
    enriched = pf.enrich_features(fb.build_training_features(matches), records).set_index("match_id")
    for m in matches.itertuples(index=False):
        setup = dict(team1=m.team1, team2=m.team2, venue=m.venue, toss_winner=m.toss_winner,
                     toss_decision=m.toss_decision, season=m.season, match_date=m.date, is_playoff=m.is_playoff)
        row, cutoff = pf.scenario_features(matches, records, **setup)
        assert cutoff == m.date
        np.testing.assert_allclose(row.iloc[0], enriched.loc[m.match_id, pf.FEATURES].to_numpy(dtype=float))
        reverse, _ = pf.scenario_features(matches, records, **{**setup, "team1": m.team2, "team2": m.team1})
        pd.testing.assert_frame_equal(reverse, pf.swap_teams(row))
        pd.testing.assert_frame_equal(row, pf.swap_teams(pf.swap_teams(row)))


def test_recent_history_is_bounded(history):
    _, _, records = history
    pairs = records[records.match_id == 1]
    extended = pd.concat([pairs.assign(date=pd.Timestamp("2010-01-01") + pd.Timedelta(days=i), match_id=i)
                          for i in range(31)], ignore_index=True)
    form = pf.FormHistory.before(extended, pd.Timestamp("2011-01-01"))
    assert len(form.league) == pf.LEAGUE_WINDOW
    assert form.team_features(TEAMS[0])["performance_matches"] == pf.TEAM_WINDOW
    changed = extended.copy()
    changed.loc[changed.match_id == 0, "bat_runs"] = 10000
    assert form.matchup(*TEAMS[:2]) == pf.FormHistory.before(changed, pd.Timestamp("2011-01-01")).matchup(*TEAMS[:2])
