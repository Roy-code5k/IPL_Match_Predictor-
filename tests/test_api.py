"""API tests against a small synthetic release (no Kaggle data needed)."""

import csv
import io
import warnings

import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

from app.backend.main import create_app, parse_batch, to_csv
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


def test_options(setup):
    client, matches, _ = setup
    options = client.get("/api/v1/options").json()
    last = str(int(matches["season"].max()))
    assert options["teams_by_season"]["2027"] == options["teams_by_season"][last]
    assert len(options["venues"]) == 36


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


@pytest.mark.parametrize("text", ["", "﻿", "   \n  \n", "\r\n\r\n", ",,,\n"])
def test_batch_empty_file_says_empty(setup, text):
    client, *_ = setup
    response = client.post("/api/v1/predict-batch", json={"csv": text})
    assert_error(response, 400, "INVALID_CSV")
    assert "empty" in response.json()["error"]["message"]


def test_batch_header_after_blank_lines_is_accepted(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    text = ("\n\nteam1,team2,venue,season,toss_winner,toss_decision\n"
            f'{m.team1},{m.team2},"{m.venue}",{m.season},{m.toss_winner},{m.toss_decision}\n')
    out = client.post("/api/v1/predict-batch", json={"csv": text}).json()
    assert out["rows"] == 1


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


def test_batch_live_rows_and_csv_booleans(setup):
    client, matches, states = setup
    row = states[(states["innings"] == 2) & (states["overs_done"] == 0)].iloc[-1]
    m = matches.set_index("match_id").loc[row.match_id]
    text = ("team1,team2,venue,season,toss_winner,toss_decision,match_date,innings,overs_done,runs,wickets,"
            "first_innings_total\n"
            f'{m.team1},{m.team2},"{m.venue}",{m.season},{m.toss_winner},{m.toss_decision},{m.date:%Y-%m-%d},'
            f"2,0,0,0,{int(row.target) - 1}")
    out = client.post("/api/v1/predict-batch", json={"csv": text}).json()
    assert out["succeeded"] == 1, out["results"][0]["error"]
    assert out["results"][0]["mode"] == "live"
    assert ",true," in out["csv"] or ",false," in out["csv"]
    assert "True" not in out["csv"] and "False" not in out["csv"]
    assert out["results"][0]["retrospective"] is True
    exported = next(csv.DictReader(io.StringIO(out["csv"])))
    assert exported["retrospective"] == "true"
    assert exported["model_trained_through"] == f"{matches.date.max():%Y-%m-%d}"


@pytest.mark.parametrize("live", [False, True])
def test_historical_label_reaches_api_and_log_export(setup, live):
    client, matches, _ = setup
    m = matches.iloc[-1]
    state = dict(innings=2, overs_done=0, runs=0, wickets=0, first_innings_total=180) if live else {}
    endpoint = "predict-live" if live else "predict"
    request_id = f"retrospective-{endpoint}"
    r = client.post(f"/api/v1/{endpoint}", json=body_for(m, **state),
                    headers={"X-Request-ID": request_id})
    assert r.status_code == 200
    assert r.json()["retrospective"] is True
    assert r.json()["model_trained_through"] == f"{matches.date.max():%Y-%m-%d}"
    exported = list(csv.DictReader(io.StringIO(client.get("/api/v1/predictions/log.csv").text)))
    row = next(row for row in exported if row["request_id"] == request_id)
    assert row["retrospective"] == "true"
    assert row["model_trained_through"] == r.json()["model_trained_through"]


@pytest.mark.parametrize("runs,wickets", [(1, 0), (0, 1), (100, 5)])
def test_innings_break_rejects_nonzero_score(setup, runs, wickets):
    client, matches, _ = setup
    r = client.post("/api/v1/predict-live", json=body_for(
        matches.iloc[-1], innings=2, overs_done=0, runs=runs, wickets=wickets, first_innings_total=180))
    assert_error(r, 422, "INVALID_REQUEST")
    assert "innings break" in r.json()["error"]["message"]


def csv_text(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerows([fields, *rows])
    return out.getvalue()


@pytest.mark.parametrize("malformation", ["duplicate", "duplicate_spaces", "empty_header", "extra_cell",
                                         "missing_cell", "unterminated_quote", "bad_quote", "large_field",
                                         "large_header"])
def test_csv_structure_errors_never_make_predictions(setup, malformation):
    client, matches, _ = setup
    fields = ["team1", "team2", "venue", "season", "toss_winner", "toss_decision"]
    body = body_for(matches.iloc[-1])
    row = [body[k] for k in fields]
    if malformation in ("duplicate", "duplicate_spaces"):
        text = csv_text(fields + ["season" if malformation == "duplicate" else " season "], [row + [2026]])
    elif malformation == "empty_header":
        text = csv_text(fields + [""], [row + ["discarded"]])
    elif malformation == "extra_cell":
        text = csv_text(fields, [row, row + ["discarded"]])
    elif malformation == "missing_cell":
        text = csv_text(fields, [row, row[:-1]])
    elif malformation == "unterminated_quote":
        text = csv_text(fields, [row]) + '"not closed'
    elif malformation == "bad_quote":
        text = csv_text(fields, [row]) + '"closed"trailing,other\n'
    elif malformation == "large_field":
        text = csv_text(fields, [row, ["x" * 140000, *row[1:]]])
    else:
        text = csv_text(["x" * 140000, *fields[1:]], [row])
    before = len(client.log.read_all())
    assert_error(client.post("/api/v1/predict-batch", json={"csv": text}), 400, "INVALID_CSV")
    assert len(client.log.read_all()) == before


def test_csv_parser_preserves_quoted_values_and_empty_optional_cells():
    fields = ["team1", "team2", "venue", "season", "toss_winner", "toss_decision", "match_date"]
    text = "\ufeff" + csv_text(fields, [['Team "A"\nname', 'Team B', 'Ground, City', 2025, 'Team B', 'bat', '']])
    row, = parse_batch(text + "\r\n")
    assert row["team1"] == 'Team "A"\nname'
    assert row["venue"] == "Ground, City"
    assert row["match_date"] == ""


@pytest.mark.parametrize("value", ["=1+1", "+1+1", "-1+1", "@SUM(1)", "  =1+1", "\t=1+1",
                                  "\r=1+1", "\n=1+1", "\ufeff =1+1", "\x00=1+1",
                                  '=HYPERLINK("example","label")'])
def test_csv_exports_protect_formula_strings(value):
    text = to_csv([{"input": value, "numeric": -1, "flag": False}], ["input", "numeric", "flag"])
    row, = list(csv.DictReader(io.StringIO(text, newline="")))
    assert row == {"input": "'" + value, "numeric": "-1", "flag": "false"}


def test_batch_and_log_exports_escape_formulas_without_changing_raw_inputs(setup):
    client, matches, _ = setup
    body = body_for(matches.iloc[-1], venue="=1+1")
    text = csv_text(list(body), [list(body.values())])
    response = client.post("/api/v1/predict-batch", json={"csv": text}, headers={"X-Request-ID": "=2+2"})
    assert response.status_code == 200
    out = response.json()
    assert out["failed"] == 1 and out["results"][0]["venue"] == "=1+1"
    row, = list(csv.DictReader(io.StringIO(out["csv"])))
    assert row["venue"] == "'=1+1"
    assert client.log.read_all()[-1]["inputs"]["venue"] == "=1+1"
    log_rows = list(csv.DictReader(io.StringIO(client.get("/api/v1/predictions/log.csv").text)))
    assert log_rows[-1]["venue"] == "'=1+1" and log_rows[-1]["request_id"] == "'=2+2"


def test_batch_innings_break_reports_bad_row_and_keeps_valid_row(setup):
    client, matches, _ = setup
    good = body_for(matches.iloc[-1], innings=2, overs_done=0, runs=0, wickets=0, first_innings_total=180)
    bad = {**good, "runs": 100, "wickets": 5}
    r = client.post("/api/v1/predict-batch", json={"csv": csv_text(list(good), [list(bad.values()), list(good.values())])})
    assert r.status_code == 200
    out = r.json()
    assert (out["failed"], out["succeeded"]) == (1, 1)
    assert "innings break" in out["results"][0]["error"]


# --- review fixes (2026-11-18) ---------------------------------------------------

def test_setup_compares_franchises_not_spellings():
    from app.backend.schemas import MatchSetup
    base = {"team1": "Delhi Capitals", "team2": "Mumbai Indians", "venue": "Wankhede Stadium, Mumbai",
            "season": 2018, "toss_decision": "bat"}
    assert MatchSetup(**base, toss_winner="Delhi Daredevils").toss_winner == "Delhi Daredevils"
    with pytest.raises(ValueError, match="different franchises"):
        MatchSetup(**{**base, "team2": "Delhi Daredevils"}, toss_winner="Delhi Capitals")
    with pytest.raises(ValueError, match="toss_winner"):
        MatchSetup(**base, toss_winner="Chennai Super Kings")


def test_unknown_team_is_still_reported_as_unknown_name(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    r = client.post("/api/v1/predict", json=body_for(m, team1="Nowhere XI", toss_winner="Nowhere XI"))
    assert_error(r, 400, "UNKNOWN_NAME")


def test_batch_score_without_innings_says_innings_is_missing(setup):
    client, matches, _ = setup
    m = matches.iloc[-1]
    row = f"\n{m.team1},{m.team2},\"{m.venue}\",{m.season},{m.team1},bat,,,5,40,1"
    out = client.post("/api/v1/predict-batch", json={"csv": batch_csv(matches, row)}).json()
    error = out["results"][-1]["error"]
    assert error.startswith("INVALID_REQUEST") and "innings" in error and "extra" not in error.lower()


def test_null_log_and_open_log_settings(tmp_path, monkeypatch):
    from app.backend.store import NullLog, open_log
    monkeypatch.setenv("IPL_LOG", "OFF")
    assert isinstance(open_log(), NullLog)
    NullLog().write({"a": 1})
    assert NullLog().read_all() == []
    monkeypatch.delenv("IPL_LOG")
    monkeypatch.setenv("IPL_LOG_DIR", str(tmp_path))
    log = open_log()
    assert isinstance(log, JsonlLog) and log.path == tmp_path / "predictions.jsonl"


def test_jsonl_log_skips_bad_lines_with_a_warning(tmp_path, caplog):
    log = JsonlLog(tmp_path)
    assert log.read_all() == []  # no file yet
    log.write({"n": 1})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"n": 2, "trunc\n\n')
    log.write({"n": 3})
    with caplog.at_level("WARNING", logger="ipl.api"):
        assert [r["n"] for r in log.read_all()] == [1, 3]
    assert "Skipped 1 unreadable line" in caplog.text
