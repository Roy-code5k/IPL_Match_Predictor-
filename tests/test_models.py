"""Tests for the evaluation split and the pre-match model wrappers (synthetic data only)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features import builder as fb
from src.models import confidence as cf
from src.models import evaluation as ev
from src.models import prematch as pm


def random_features(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(fb.FEATURE_COLUMNS))), columns=fb.FEATURE_COLUMNS)
    for col in ("toss_won_a", "toss_field", "a_bats_first", "home_a", "home_b"):
        X[col] = (X[col] > 0.5).astype(float)
    y = (X["elo_prob_a"] + 0.3 * rng.standard_normal(n) > 0.5).astype(int).to_numpy()
    return X, y


def seasons_frame(seasons=range(2008, 2025), per_season=10):
    rows = [{"season": s, "x": i, "y": i % 2} for s in seasons for i in range(per_season)]
    return pd.DataFrame(rows)


# --- split discipline ---------------------------------------------------------

def test_walk_forward_trains_only_on_earlier_seasons_and_never_sees_test():
    seen = []

    def fit_predict(train, evaluate):
        assert evaluate["season"].nunique() == 1
        season = evaluate["season"].iloc[0]
        assert train["season"].max() < season
        seen.append((train["season"].max(), season))
        return np.full(len(evaluate), 0.5)

    out = ev.walk_forward(seasons_frame(), fit_predict)
    assert sorted(out["season"].unique()) == list(ev.WALK_FORWARD_SEASONS)
    assert all(season not in ev.TEST_SEASONS for _, season in seen)
    assert out["season"].max() < min(ev.TEST_SEASONS)


def test_validation_is_2022_trained_on_2008_to_2021():
    def fit_predict(train, evaluate):
        assert train["season"].min() == 2008 and train["season"].max() == 2021
        assert set(evaluate["season"]) == {2022}
        return np.zeros(len(evaluate))

    out = ev.validation(seasons_frame(), fit_predict)
    assert len(out) == 10


# --- metrics ------------------------------------------------------------------

def test_score_known_values():
    y = np.array([0, 1, 1, 0])
    perfect = ev.score(y, np.array([0.1, 0.9, 0.8, 0.2]))
    assert perfect.accuracy == 1.0 and perfect.roc_auc == 1.0
    assert perfect.confusion == ((2, 0), (0, 2))

    coin = ev.score(y, np.full(4, 0.5))
    assert coin.log_loss == pytest.approx(math.log(2))
    assert coin.brier == pytest.approx(0.25)
    assert coin.accuracy == 0.5  # 0.5 counts as predicting class 1


def test_calibration_table_counts_every_row():
    prob = np.array([0.05, 0.3, 0.5, 0.55, 0.99, 1.0])
    rows = ev.calibration_table(np.array([0, 0, 1, 1, 1, 1]), prob)
    assert sum(r[1] for r in rows) == len(prob)


# --- model wrappers -----------------------------------------------------------

@pytest.mark.parametrize("factory", [pm.logistic_regression, pm.xgboost])
def test_symmetric_model_is_order_invariant(factory):
    X, y = random_features()
    model = pm.Symmetric(factory).fit(X, y)
    p = model.predict(X)
    p_swapped = model.predict(fb.swap_teams(X))
    np.testing.assert_allclose(p + p_swapped, 1.0, atol=1e-6)
    assert ((p > 0) & (p < 1)).all()


def test_training_is_deterministic():
    X, y = random_features()
    a = pm.Symmetric(pm.random_forest).fit(X, y).predict(X)
    b = pm.Symmetric(pm.random_forest).fit(X, y).predict(X)
    # Same trees; parallel averaging can differ only in the last floating-point bits.
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_baselines():
    X, y = random_features(n=20)
    np.testing.assert_array_equal(pm.Constant().fit(X, y).predict(X), np.full(20, 0.5))
    np.testing.assert_array_equal(pm.EloBaseline().fit(X, y).predict(X), X["elo_prob_a"].to_numpy())


def test_selection_picks_lowest_walk_forward_log_loss():
    def fake(log_loss):
        return {"wf_metrics": ev.Metrics(1, 0.5, 0, 0, 0.5, log_loss, 0.25, ((0, 0), (0, 0)))}

    assert pm.select({"a": fake(0.70), "b": fake(0.69), "c": fake(0.71)}) == "b"


# --- confidence labels ----------------------------------------------------------


@pytest.mark.parametrize("prob, expected", [
    (0.50, cf.CLOSE), (0.549, cf.CLOSE), (0.55, "Slight edge"), (0.649, "Slight edge"),
    (0.65, "Clear edge"), (0.799, "Clear edge"), (0.80, "Strong edge"), (1.0, "Strong edge"),
])
def test_confidence_label_boundaries(prob, expected):
    assert cf.label(prob) == expected
    assert cf.label(1 - prob) == expected  # the same for either team


def test_confidence_label_rejects_invalid_probability():
    with pytest.raises(ValueError):
        cf.label(1.2)


def test_band_table_counts_every_prediction():
    prob = np.array([0.5, 0.52, 0.4, 0.3, 0.9, 0.05, 0.7])
    y = np.array([1, 0, 0, 0, 1, 0, 1])
    rows = cf.band_table(y, prob)
    assert [r[0] for r in rows] == [name for _, name in reversed(cf.BANDS)]
    assert sum(r[2] for r in rows) == len(prob)
    strong = next(r for r in rows if r[0] == "Strong edge")
    assert strong[2] == 2 and strong[5] == "100.0%"  # 0.9 -> side won; 0.05 -> other side won
