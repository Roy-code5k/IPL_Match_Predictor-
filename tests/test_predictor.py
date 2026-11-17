"""Tests for loading a release and predicting from it.

Most tests build a small release from synthetic data in a temporary folder.
The last test checks the real release when it has been built.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.inspect_raw import sha256
from src.data.mappings import UnknownNameError
from src.features import builder as fb
from src.features import live as lf
from src.models import confidence as cf
from src.models import live as lm
from src.models import prematch as pm
from src.models import release as rl
from src.models.predictor import Predictor, ReleaseError
from tests.test_features import synthetic_matches
from tests.test_live import simulate


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    folder = tmp_path_factory.mktemp("release")
    matches, states = build_test_release(folder)
    return folder, matches, states


def build_test_release(folder):
    """Write a small but complete release from synthetic data into `folder`."""
    matches, raw = simulate(synthetic_matches(n=120))
    raw.to_csv(folder / "raw_deliveries.csv", index=False)
    deliveries = lf.load_deliveries(folder / "raw_deliveries.csv")
    features = fb.build_training_features(matches)
    full = set(matches["match_id"])
    states = lf.build_live_states(matches, features, deliveries, full)

    files = {}
    pre_model = pm.Symmetric(pm.logistic_regression).fit(features[pm.FEATURES], features[pm.TARGET].to_numpy())
    joblib.dump(pre_model, folder / "prematch.joblib")
    for i in (1, 2):
        data = states[states["innings"] == i]
        model = lm.logistic_regression().fit(data[lf.FEATURES_BY_INNINGS[i]], data[lf.TARGET].to_numpy())
        joblib.dump(model, folder / f"live_innings{i}.joblib")
    for name in ("prematch.joblib", "live_innings1.joblib", "live_innings2.joblib"):
        files[name] = sha256(folder / name)
    files["history.csv"] = rl._write_csv(matches[rl.HISTORY_COLUMNS], folder / "history.csv")
    files["first_innings_totals.csv"] = rl._write_csv(lf.totals_table(matches, deliveries, full),
                                                      folder / "first_innings_totals.csv")
    metadata = {"release": "test", "packages": {},
                "history_cutoff": f"{matches['date'].max():%Y-%m-%d}",
                "training_seasons": [int(matches["season"].min()), int(matches["season"].max())],
                "evaluation": {"test_seasons": []},
                "prematch": {"model": "Logistic regression", "file": "prematch.joblib", "test_metrics": {}},
                "live": {**{f"innings{i}": {"model": "Logistic regression", "file": f"live_innings{i}.joblib"}
                            for i in (1, 2)}, "test_accuracy_by_moment": {}},
                "confidence_bands": [{"from": lo, "label": name} for lo, name in cf.BANDS],
                "close_call_label": cf.CLOSE,
                "files": files}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return matches, states


def scenario(m):
    return dict(team1=m.team1, team2=m.team2, venue=m.venue, toss_winner=m.toss_winner,
                toss_decision=m.toss_decision, season=int(m.season), match_date=m.date,
                is_playoff=bool(m.is_playoff))


def test_prediction_is_order_invariant_and_labelled(release):
    folder, matches, _ = release
    p = Predictor.load(folder)
    m = matches.iloc[-1]
    a = p.predict(**scenario(m))
    b = p.predict(**{**scenario(m), "team1": m.team2, "team2": m.team1})
    assert a["probabilities"][m.team1] == pytest.approx(b["probabilities"][m.team1])
    assert sum(a["probabilities"].values()) == pytest.approx(1)
    assert a["confidence"] == cf.label(a["probabilities"][m.team1])
    assert a["close_call"] == (a["confidence"] == cf.CLOSE)
    assert a["history_before"] == f"{m.date:%Y-%m-%d}"


def test_live_prediction_matches_the_model_on_the_training_state(release):
    folder, matches, states = release
    p = Predictor.load(folder)
    m = matches.iloc[-5]
    row = states[(states["match_id"] == m.match_id) & (states["innings"] == 2) & (states["overs_done"] == 6)]
    if row.empty:
        pytest.skip("chase finished before 6 overs in this synthetic match")
    row = row.iloc[0]
    target = int(row.target)
    out = p.predict_live(**scenario(m), innings=2, overs_done=6, runs=target - int(row.runs_needed),
                         wickets=10 - int(row.wickets_left), first_innings_total=target - 1)
    expected_bf = p.live_models[2].predict_proba(row[lf.INNINGS2_FEATURES].to_frame().T.astype(float))[0, 1]
    assert out["batting_first"] == row.bat_first
    assert out["probabilities"][row.bat_first] == pytest.approx(expected_bf)
    assert out["state"]["target"] == target


def test_invalid_input_is_rejected(release):
    folder, matches, _ = release
    p = Predictor.load(folder)
    m = matches.iloc[-1]
    with pytest.raises(UnknownNameError):
        p.predict(**{**scenario(m), "team1": "Not A Team"})
    with pytest.raises(ValueError):  # chase already won
        p.predict_live(**scenario(m), innings=2, overs_done=5, runs=200, wickets=1, first_innings_total=150)


def test_tampered_or_missing_release_is_refused(release, tmp_path):
    folder, *_ = release
    for f in folder.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    history = pd.read_csv(tmp_path / "history.csv")
    history.loc[0, "winner"] = history.loc[0, "team2"] if history.loc[0, "winner"] == history.loc[0, "team1"] \
        else history.loc[0, "team1"]
    history.to_csv(tmp_path / "history.csv", index=False)
    with pytest.raises(ReleaseError):
        Predictor.load(tmp_path)
    with pytest.raises(ReleaseError):
        Predictor.load(tmp_path / "nowhere")


@pytest.mark.skipif(not (rl.RELEASE_DIR / "prematch.joblib").exists(),
                    reason="release not built (python -m src.models.release)")
def test_real_release_loads_and_predicts():
    p = Predictor.load()
    assert p.metadata["release"] == rl.RELEASE_VERSION
    assert p.metadata["evaluation"]["test_seasons"] == [2023, 2024]
    out = p.predict(team1="Mumbai Indians", team2="Chennai Super Kings", venue="Wankhede Stadium, Mumbai",
                    season=2025, toss_winner="Mumbai Indians", toss_decision="bat")
    assert 0 < out["probabilities"]["Mumbai Indians"] < 1
    assert out["history_before"] > p.metadata["history_cutoff"]
    assert np.isfinite(list(out["probabilities"].values())).all()


def test_history_cache_gives_the_same_predictions(release):
    folder, matches, _ = release
    p = Predictor.load(folder)
    for m in matches.iloc[[-1, -20, -40]].itertuples(index=False):
        cached = p.predict(**scenario(m))["probabilities"][m.team1]
        uncached_row, _ = fb.scenario_features(p.history, **scenario(m))
        assert cached == pytest.approx(float(p.prematch_model.predict(uncached_row)[0]), abs=1e-12)
        assert p.predict(**scenario(m))["probabilities"][m.team1] == cached  # second call hits the cache
    assert len(p._histories) == 3


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("offset,retrospective", [(-1, True), (0, True), (1, False)])
def test_retrospective_flag_uses_model_training_cutoff(release, live, offset, retrospective):
    folder, matches, _ = release
    p = Predictor.load(folder)
    date = pd.Timestamp(p.metadata["history_cutoff"]) + pd.Timedelta(days=offset)
    setup = {**scenario(matches.iloc[-1]), "match_date": date, "season": date.year}
    out = (p.predict_live(**setup, innings=2, overs_done=0, runs=0, wickets=0, first_innings_total=180)
           if live else p.predict(**setup))
    assert out["retrospective"] is retrospective
    assert out["model_trained_through"] == p.metadata["history_cutoff"]
    assert out["history_before"] == f"{date:%Y-%m-%d}"


def test_retrospective_flag_without_explicit_match_date(release):
    folder, matches, _ = release
    p = Predictor.load(folder)
    setup = scenario(matches.iloc[-1])
    setup.pop("match_date")
    assert p.predict(**setup)["retrospective"] is True
    assert p.predict(**{**setup, "season": setup["season"] + 1})["retrospective"] is False
