"""API tests against a small synthetic release (no Kaggle data needed)."""

import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

from app.backend.main import create_app
from app.backend.store import JsonlLog
from tests.test_predictor import build_test_release


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("api_release")
    matches, states = build_test_release(folder)
    log = JsonlLog(tmp_path_factory.mktemp("api_logs"))
    with TestClient(create_app(folder, prediction_log=log)) as client:
        client.log = log
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
    with TestClient(create_app(tmp_path / "missing", prediction_log=JsonlLog(tmp_path))) as client:
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        m = {"team1": "Mumbai Indians", "team2": "Chennai Super Kings", "venue": "Wankhede Stadium, Mumbai",
             "season": 2024, "toss_winner": "Mumbai Indians", "toss_decision": "bat"}
        assert_error(client.post("/api/v1/predict", json=m), 503, "MODEL_UNAVAILABLE")


def test_root_points_to_the_app_and_docs(setup):
    client, *_ = setup
    body = client.get("/").json()
    assert "localhost:5173" in body["app"] and body["docs"] == "/api/docs"


# --- Phase 7: inputs, logging, batch and export ------------------------------------

def test_prediction_includes_readable_inputs(setup):
    client, matches, _ = setup
    out = client.post("/api/v1/predict", json=body_for(matches.iloc[-1])).json()
    labels = [item["label"] for item in out["inputs"]]
    assert "Elo rating" in labels and "Head-to-head win rate" in labels
    elo = out["inputs"][0]
    assert isinstance(elo["team1"], float) and elo["format"] == "number"
    home = next(i for i in out["inputs"] if i["label"] == "Home ground")
    assert isinstance(home["team1"], bool)


def test_every_prediction_is_logged_with_its_request_id(setup):
    client, matches, _ = setup
    before = len(client.log.read_all())
    ok = client.post("/api/v1/predict", json=body_for(matches.iloc[-1]), headers={"X-Request-ID": "log-ok"})
    bad = client.post("/api/v1/predict", json=body_for(matches.iloc[-1], venue="Nowhere Oval"),
                      headers={"X-Request-ID": "log-bad"})
    records = client.log.read_all()[before:]
    assert [r["request_id"] for r in records] == ["log-ok", "log-bad"]
    assert records[0]["status"] == "ok" and records[0]["result"]["team1_probability"] == ok.json()["team1_probability"]
    assert records[0]["result"]["model_release"] == "test" and records[0]["latency_ms"] >= 0
    assert records[1]["status"] == "error" and records[1]["error_code"] == "UNKNOWN_NAME"
    assert bad.status_code == 400


def batch_csv(matches, extra_rows=""):
    rows = []
    for m in matches.tail(3).itertuples(index=False):
        rows.append(f"{m.team1},{m.team2},\"{m.venue}\",{m.season},{m.toss_winner},{m.toss_decision},{m.date:%Y-%m-%d},,,,")
    header = "team1,team2,venue,season,toss_winner,toss_decision,match_date,innings,overs_done,runs,wickets"
    return "\n".join([header, *rows]) + extra_rows


def test_batch_predicts_good_rows_and_reports_bad_ones(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    bad_rows = (f"\n{m.team1},{m.team1},\"{m.venue}\",{m.season},{m.team1},bat,,,,,"
                f"\n{m.team1},{m.team2},Nowhere Oval,{m.season},{m.team1},bat,,,,,"
                f"\n{m.team1},{m.team2},\"{m.venue}\",{m.season},{m.team1},bat,,2,0,0,0")
    r = client.post("/api/v1/predict-batch", json={"csv": batch_csv(matches, bad_rows)})
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["rows"], out["succeeded"], out["failed"]) == (6, 3, 3)
    errors = [row["error"] for row in out["results"]]
    assert errors[3].startswith("INVALID_REQUEST") and errors[4].startswith("UNKNOWN_NAME")
    assert errors[5].startswith("INVALID_REQUEST")  # chase without a first-innings total
    single = client.post("/api/v1/predict", json=body_for(m)).json()
    assert out["results"][2]["team1_probability"] == single["team1_probability"]
    lines = out["csv"].strip().splitlines()
    assert len(lines) == 7 and lines[0].endswith("model_name,error")


@pytest.mark.parametrize("text, code", [
    ("team1,team2\nA,B", "INVALID_CSV"),                                   # missing columns
    ("team1,team2,venue,season,toss_winner,toss_decision,colour\n", "INVALID_CSV"),  # unknown column
    ("team1,team2,venue,season,toss_winner,toss_decision\n", "INVALID_CSV"),  # no rows
])
def test_batch_rejects_malformed_csv(setup, text, code):
    client, *_ = setup
    assert_error(client.post("/api/v1/predict-batch", json={"csv": text}), 400, code)


def test_batch_row_limit(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    row = f"{m.team1},{m.team2},\"{m.venue}\",{m.season},{m.toss_winner},{m.toss_decision}"
    text = "team1,team2,venue,season,toss_winner,toss_decision\n" + "\n".join([row] * 201)
    assert_error(client.post("/api/v1/predict-batch", json={"csv": text}), 400, "TOO_MANY_ROWS")


def test_log_export_is_csv_with_inputs_and_model(setup):
    client, matches, _ = setup
    client.post("/api/v1/predict", json=body_for(matches.iloc[-1]))
    r = client.get("/api/v1/predictions/log.csv")
    assert r.headers["content-type"].startswith("text/csv")
    header, *rows = r.text.strip().splitlines()
    assert header.startswith("logged_at,request_id,endpoint,status") and "model_release" in header
    assert rows and "test" in rows[-1]
