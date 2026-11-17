"""Validate experiment chronology, weighting, incumbent parity and selection."""

import numpy as np
import pandas as pd
import pytest

from src.features import builder as fb
from src.features import performance as pf
from src.models import prematch as pm
from src.models import prematch_experiment as ex
from tests.test_features import synthetic_matches


@pytest.fixture(scope="module")
def features():
    frame = fb.build_training_features(synthetic_matches())
    for name in pf.METRICS:
        frame[f"{name}_a"] = np.linspace(0.1, 1, len(frame))
        frame[f"{name}_b"] = np.linspace(1, 0.1, len(frame))
        frame[f"{name}_diff"] = frame[f"{name}_a"] - frame[f"{name}_b"]
    frame["performance_matches_a"] = frame["performance_matches_b"] = 10.0
    return frame


@pytest.mark.parametrize("policy,first", [("all", 2008), ("last_3", 2020), ("last_5", 2018), ("decay_2", 2008)])
def test_training_policies_exclude_evaluation_and_future_seasons(policy, first):
    frame = pd.DataFrame({"season": range(2008, 2025),
                          "date": pd.to_datetime([f"{year}-04-01" for year in range(2008, 2025)])})
    train, weights = ex.training_data(frame, 2023, policy)
    assert list(train.season) == list(range(first, 2023))
    if policy == "decay_2":
        assert weights.mean() == pytest.approx(1)
        assert (np.diff(weights) > 0).all()
        assert weights[-1] / weights[-3] == pytest.approx(2, rel=0.002)
    else:
        assert weights is None


def test_no_future_date_slips_through_an_earlier_season_label():
    frame = pd.DataFrame({"season": [2021, 2022], "date": pd.to_datetime(["2021-05-01", "2023-04-01"])})
    train, _ = ex.training_data(frame, 2023, "all")
    assert list(train.index) == [0]
    with pytest.raises(ValueError, match="Unknown training policy"):
        ex.training_data(frame, 2023, "invalid")


@pytest.mark.parametrize("name", ["random_forest", "logistic_regression"])
def test_base_candidate_reproduces_original_model(features, name):
    train, _ = ex.training_data(features, 2012, "all")
    evaluate = features[features.season == 2012]
    old = pm.Symmetric(getattr(pm, name)).fit(train[fb.FEATURE_COLUMNS], train[fb.TARGET].to_numpy())
    new = ex.ExperimentModel(ex.Candidate(name, "base", "all")).fit(train)
    np.testing.assert_allclose(new.predict(evaluate), old.predict(evaluate[fb.FEATURE_COLUMNS]), atol=1e-12)


@pytest.mark.parametrize("name", ["random_forest", "logistic_regression"])
def test_weighted_performance_candidate_preserves_team_symmetry(features, name):
    train, weights = ex.training_data(features, 2012, "decay_2")
    model = ex.ExperimentModel(ex.Candidate(name, "performance", "decay_2")).fit(train, weights)
    evaluate = features[features.season == 2012]
    pred = model.predict(evaluate)
    np.testing.assert_allclose(pred + model.predict(pf.swap_teams(evaluate)), 1, atol=1e-12)
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()


def test_weights_are_duplicated_for_the_same_swapped_matches(features, monkeypatch):
    class Recorder:
        def fit(self, x, y, **kwargs):
            self.x, self.y, self.kwargs = x, y, kwargs
    recorder = Recorder()
    monkeypatch.setattr(pm, "logistic_regression", lambda: recorder)
    train, weights = ex.training_data(features, 2012, "decay_2")
    ex.ExperimentModel(ex.Candidate("logistic_regression", "performance", "decay_2")).fit(train, weights)
    for value in recorder.kwargs.values():
        np.testing.assert_allclose(value, np.r_[weights, weights])
    n = len(train)
    np.testing.assert_array_equal(recorder.y[n:], 1 - recorder.y[:n])
    pd.testing.assert_frame_equal(recorder.x.iloc[n:].reset_index(drop=True),
                                  pf.swap_teams(recorder.x.iloc[:n]).reset_index(drop=True))


def test_selection_uses_development_accuracy_then_log_loss():
    scores = {"a": {"accuracy": 0.55, "log_loss": 0.68, "exploratory_accuracy": 1},
              "b": {"accuracy": 0.56, "log_loss": 0.70, "exploratory_accuracy": 0},
              "c": {"accuracy": 0.56, "log_loss": 0.69, "exploratory_accuracy": 0}}
    assert ex.select(scores) == "c"
    scores["a"]["exploratory_accuracy"] = 0
    assert ex.select(scores) == "c"


def test_evaluation_cannot_precede_training_cutoff(features):
    with pytest.raises(ValueError, match="Evaluation must start"):
        ex.predict_period(features, ex.CANDIDATES[0], (2011,), 2012)


def test_paired_comparison_aligns_matches_and_counts_disagreements():
    base = pd.DataFrame({"match_id": [1, 2, 3], fb.TARGET: [1, 0, 1], "prob": [0.6, 0.6, 0.6]})
    same = ex.paired_accuracy_delta(base, base.iloc[::-1])
    assert same["accuracy_difference"] == 0 and same["bootstrap_95_interval"] == [0, 0]
    better = base.copy()
    better.loc[1, "prob"] = 0.4
    result = ex.paired_accuracy_delta(base, better)
    assert result["newly_correct"] == 1 and result["newly_wrong"] == 0
    assert result["accuracy_difference"] == pytest.approx(1 / 3)


def test_json_output_ignores_last_bit_float_noise(tmp_path):
    """Parallel forests can change a metric's last bit; the saved JSON must not change."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ex.write_json(a, {"log_loss": 0.6916189382612133, "interval": [-0.05555555555555555], "n": 3, "ok": True})
    ex.write_json(b, {"log_loss": 0.6916189382612132, "interval": [-0.05555555555555556], "n": 3, "ok": True})
    assert a.read_bytes() == b.read_bytes()
    assert '"n": 3' in a.read_text() and '"ok": true' in a.read_text()
