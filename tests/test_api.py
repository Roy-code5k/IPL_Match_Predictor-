"""API tests against a small synthetic release (no Kaggle data needed)."""

import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

from app.backend.main import create_app
from tests.test_predictor import build_test_release


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("api_release")
    matches, states = build_test_release(folder)
    with TestClient(create_app(folder)) as client:
        yield client, matches, states


def body_for(m, **changes):
    body = {"team1": m.team1, "team2": m.team2, "venue": m.venue, "season": int(m.season),
            "toss_winner": m.toss_winner, "toss_decision": m.toss_decision,
            "match_date": f"{m.date:%Y-%m-%d}"}
    return {**body, **changes}


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert error["message"]


def test_health_reports_the_loaded_release(setup):
    client, *_ = setup
    r = client.get("/api/v1/health")
    assert r.json()["model_loaded"] is True and r.json()["release"] == "test"
    assert r.headers["X-Request-ID"]


def test_request_id_is_echoed(setup):
    client, *_ = setup
    assert client.get("/api/v1/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"


def test_predict_returns_both_probabilities_and_is_deterministic(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    first = client.post("/api/v1/predict", json=body_for(m)).json()
    second = client.post("/api/v1/predict", json=body_for(m)).json()
    assert first == second
    assert first["mode"] == "pre-match"
    assert first["team1_probability"] + first["team2_probability"] == pytest.approx(1, abs=1e-4)
    assert first["predicted_winner"] in (m.team1, m.team2)
    assert first["model"] == {"release": "test", "name": "Logistic regression"}
    assert first["history_before"] == f"{m.date:%Y-%m-%d}"


def test_predict_live_in_the_chase(setup):
    client, matches, states = setup
    row = states[(states["innings"] == 2) & (states["overs_done"] == 0)].iloc[-1]
    m = matches.set_index("match_id").loc[row.match_id]
    m = m.to_frame().T.assign(match_id=row.match_id).iloc[0]
    r = client.post("/api/v1/predict-live", json=body_for(
        m, innings=2, overs_done=0, runs=0, wickets=0, first_innings_total=int(row.target) - 1))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["mode"] == "live" and out["batting_first"] == row.bat_first
    assert out["state"]["target"] == int(row.target)


@pytest.mark.parametrize("changes", [
    {"team2": "SAME"},                           # same team twice
    {"toss_winner": "Gujarat Titans"},            # not one of the two teams
    {"toss_decision": "bowl"},
    {"season": 2007},
    {"season": 2028},
    {"match_date": "1999-01-01"},
    {"unexpected": 1},
])
def test_invalid_setup_is_rejected_before_inference(setup, changes):
    client, matches, _ = setup
    m = matches.iloc[-1]
    if changes.get("team2") == "SAME":
        changes = {"team2": m.team1}
    assert_error(client.post("/api/v1/predict", json=body_for(m, **changes)), 422, "INVALID_REQUEST")


def test_unknown_names_and_teams_outside_the_season(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    assert_error(client.post("/api/v1/predict", json=body_for(m, venue="Nowhere Oval")), 400, "UNKNOWN_NAME")
    outsider = "Gujarat Titans"  # a real franchise that never plays in the synthetic seasons
    r = client.post("/api/v1/predict", json=body_for(m, team1=outsider, toss_winner=outsider))
    assert_error(r, 400, "TEAM_NOT_IN_SEASON")


@pytest.mark.parametrize("state, status, code", [
    ({"innings": 1, "overs_done": 0, "runs": 0, "wickets": 0}, 422, "INVALID_REQUEST"),
    ({"innings": 2, "overs_done": 5, "runs": 40, "wickets": 1}, 422, "INVALID_REQUEST"),         # no target
    ({"innings": 2, "overs_done": 5, "runs": 200, "wickets": 1, "first_innings_total": 150}, 422,
     "INVALID_REQUEST"),                                                                          # chase won
    ({"innings": 1, "overs_done": 5, "runs": 40, "wickets": 10}, 422, "INVALID_REQUEST"),
    ({"innings": 1, "overs_done": 20, "runs": 160, "wickets": 3}, 422, "INVALID_REQUEST"),
])
def test_invalid_live_states(setup, state, status, code):
    client, matches, _ = setup
    assert_error(client.post("/api/v1/predict-live", json=body_for(matches.iloc[-1], **state)), status, code)


def test_options_and_metrics(setup):
    client, matches, _ = setup
    options = client.get("/api/v1/options").json()
    last = str(int(matches["season"].max()))
    assert options["teams_by_season"]["2027"] == options["teams_by_season"][last]
    assert len(options["venues"]) == 36
    metrics = client.get("/api/v1/model-metrics").json()
    assert metrics["release"] == "test" and metrics["close_call_label"] == "Too close to call"


def test_missing_release_degrades_cleanly(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as client:
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        m = {"team1": "Mumbai Indians", "team2": "Chennai Super Kings", "venue": "Wankhede Stadium, Mumbai",
             "season": 2024, "toss_winner": "Mumbai Indians", "toss_decision": "bat"}
        assert_error(client.post("/api/v1/predict", json=m), 503, "MODEL_UNAVAILABLE")


def test_root_points_to_the_app_and_docs(setup):
    client, *_ = setup
    body = client.get("/").json()
    assert "localhost:5173" in body["app"] and body["docs"] == "/api/docs"
