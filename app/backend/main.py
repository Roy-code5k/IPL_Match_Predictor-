"""FastAPI service for IPL match predictions (Phases 6-7).

Run from the project root, after `python -m src.models.release`:
    uvicorn app.backend.main:app --reload

Routes (all under /api/v1):
    GET  /health              service and model-load status
    GET  /options             seasons, teams per season and venues the API accepts
    GET  /model-metrics       the loaded release's test metrics, model comparison and confidence bands
    POST /predict             pre-match prediction (after the toss)
    POST /predict-live        in-match prediction from the score at the end of an over
    POST /predict-batch       up to 200 matches from CSV text; returns a results CSV
    GET  /predictions/log.csv every logged prediction (inputs, outputs, model release)

Errors always have the shape {"error": {"code", "message", "request_id"}}; the same
request id is sent in the X-Request-ID response header. Every prediction is logged
(see app/backend/store.py).
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.backend import schemas
from app.backend.store import PredictionLog, open_log
from src.data import mappings
from src.models.predictor import Predictor, ReleaseError
from src.models.release import RELEASE_DIR

log = logging.getLogger("ipl.api")

RESULT_COLUMNS = ["mode", "team1_probability", "team2_probability", "predicted_winner", "confidence",
                  "close_call", "history_before", "model_trained_through", "retrospective",
                  "model_release", "model_name"]
LIVE_COLUMNS = ("innings", "overs_done", "runs", "wickets", "first_innings_total")
LOG_COLUMNS = (["logged_at", "request_id", "endpoint", "status", "error_code", "latency_ms"]
               + schemas.BATCH_INPUT_COLUMNS + RESULT_COLUMNS)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _error(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(status_code=status, headers={"X-Request-ID": request_id},
                        content={"error": {"code": code, "message": message, "request_id": request_id}})


def teams_by_season(history) -> dict[int, list[str]]:
    teams: dict[int, set[str]] = {}
    for row in history[["season", "team1", "team2"]].itertuples(index=False):
        teams.setdefault(int(row.season), set()).update((row.team1, row.team2))
    return {season: sorted(names) for season, names in sorted(teams.items())}


def validation_message(exc: ValidationError | RequestValidationError) -> str:
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = first.get("msg", "Invalid request").removeprefix("Value error, ")
    return f"{field}: {message}" if field else message


def result_fields(prediction: schemas.Prediction | None) -> dict:
    if prediction is None:
        return {}
    return {"mode": prediction.mode, "team1_probability": prediction.team1_probability,
            "team2_probability": prediction.team2_probability, "predicted_winner": prediction.predicted_winner,
            "confidence": prediction.confidence, "close_call": prediction.close_call,
            "history_before": prediction.history_before,
            "model_trained_through": prediction.model_trained_through,
            "retrospective": prediction.retrospective, "model_release": prediction.model.release,
            "model_name": prediction.model.name}


def csv_value(value):
    """Keep untrusted strings as spreadsheet text, while preserving numeric types."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and re.match(r"^[\s\x00-\x1f\ufeff]*[=+\-@]|^[\t\r\n]", value):
        return "'" + value
    return value


def to_csv(rows: list[dict], columns: list[str]) -> str:
    """Spreadsheet-safe CSV; booleans written as true/false, numeric values unchanged."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows({k: csv_value(v) for k, v in row.items()} for row in rows)
    return out.getvalue()


def create_app(release_dir: Path | None = None, prediction_log: PredictionLog | None = None) -> FastAPI:
    release_dir = Path(release_dir or os.environ.get("IPL_RELEASE_DIR", RELEASE_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.log = prediction_log or open_log()
        try:
            app.state.predictor = Predictor.load(release_dir)
            app.state.season_teams = teams_by_season(app.state.predictor.history)
            app.state.load_error = None
        except (ReleaseError, OSError, KeyError) as exc:
            app.state.predictor, app.state.season_teams = None, {}
            app.state.load_error = str(exc)
            log.error("Model release not loaded: %s", exc)
        yield

    app = FastAPI(title="IPL Match Winner Prediction API", version="1.1.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")

    cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    if cors_origins == ["*"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def normalize_path(request: Request, call_next):
        path = request.scope.get("path", "")
        if "//" in path:
            request.scope["path"] = re.sub(r"/+", "/", path)
        return await call_next(request)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return _error(request, 422, "INVALID_REQUEST", validation_message(exc))

    @app.exception_handler(mappings.UnknownNameError)
    async def unknown_name(request: Request, exc: mappings.UnknownNameError):
        return _error(request, 400, "UNKNOWN_NAME", str(exc))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("Unhandled error (request %s)", getattr(request.state, "request_id", "-"))
        return _error(request, 500, "INTERNAL_ERROR", "Something went wrong. Quote the request id if you report it.")

    # --- shared prediction logic ---------------------------------------------------

    def predictor(request: Request) -> Predictor:
        p = request.app.state.predictor
        if p is None:
            raise ApiError(503, "MODEL_UNAVAILABLE",
                           "The model release is not loaded. Build it with: python -m src.models.release")
        return p

    def check_setup(request: Request, body: schemas.MatchSetup) -> None:
        """Teams must exist in the chosen season (or be active in the latest season for future ones)."""
        season_teams = request.app.state.season_teams
        last = max(season_teams)
        allowed = season_teams.get(body.season, season_teams[last] if body.season > last else [])
        for team in (body.team1, body.team2):
            name = mappings.normalize_team(team)
            if name not in allowed:
                when = f"the {body.season} season" if body.season <= last else f"{last}, the latest season in the data"
                raise ApiError(400, "TEAM_NOT_IN_SEASON", f"{name} did not play in {when}.")
        mappings.normalize_venue(body.venue)

    def to_response(result: dict) -> schemas.Prediction:
        t1, t2 = result["team1"], result["team2"]
        return schemas.Prediction(
            mode=result["mode"], team1=t1, team2=t2,
            team1_probability=round(result["probabilities"][t1], 4),
            team2_probability=round(result["probabilities"][t2], 4),
            predicted_winner=result["predicted_winner"], confidence=result["confidence"],
            close_call=result["close_call"], history_before=result["history_before"],
            model_trained_through=result["model_trained_through"], retrospective=result["retrospective"],
            model=schemas.ModelInfo(**result["model"]),
            batting_first=result.get("batting_first"),
            state=schemas.LiveState(**result["state"]) if "state" in result else None,
            inputs=[schemas.InputItem(**item) for item in result.get("inputs", [])],
        )

    def run(request: Request, body: schemas.MatchSetup) -> schemas.Prediction:
        """One prediction. Raises ApiError or UnknownNameError for input the model cannot use."""
        p = predictor(request)
        check_setup(request, body)
        setup = {"team1": body.team1, "team2": body.team2, "venue": body.venue, "season": body.season,
                 "toss_winner": body.toss_winner, "toss_decision": body.toss_decision,
                 "match_date": body.match_date.isoformat() if body.match_date else None,
                 "is_playoff": body.is_playoff}
        live = isinstance(body, schemas.LiveRequest)
        try:
            if live:
                result = p.predict_live(**setup, innings=body.innings, overs_done=body.overs_done, runs=body.runs,
                                        wickets=body.wickets, first_innings_total=body.first_innings_total)
            else:
                result = p.predict(**setup)
        except mappings.UnknownNameError:
            raise
        except ValueError as exc:
            raise ApiError(400, "INVALID_MATCH_STATE" if live else "INVALID_INPUT", str(exc)) from None
        return to_response(result)

    def record(request: Request, endpoint: str, inputs: dict, started: float,
               prediction: schemas.Prediction | None = None, error_code: str | None = None,
               error_message: str | None = None) -> None:
        entry = {
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request_id": request.state.request_id,
            "endpoint": endpoint,
            "status": "ok" if prediction else "error",
            "error_code": error_code,
            "error_message": error_message,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "inputs": inputs,
            "result": result_fields(prediction),
        }
        try:
            request.app.state.log.write(entry)
        except OSError as exc:  # logging must never break a prediction
            log.warning("Prediction log not written: %s", exc)

    def run_logged(request: Request, endpoint: str, body: schemas.MatchSetup) -> schemas.Prediction:
        started = time.perf_counter()
        inputs = body.model_dump(mode="json", exclude_none=True)
        try:
            prediction = run(request, body)
        except ApiError as exc:
            record(request, endpoint, inputs, started, error_code=exc.code, error_message=exc.message)
            raise
        except mappings.UnknownNameError as exc:
            record(request, endpoint, inputs, started, error_code="UNKNOWN_NAME", error_message=str(exc))
            raise
        record(request, endpoint, inputs, started, prediction=prediction)
        return prediction

    # --- routes ---------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "IPL Match Winner Prediction API",
                "app": "This is the API only. Open the web app at http://localhost:5173 "
                       "(start it with: cd app\\frontend; npm.cmd run dev).",
                "docs": "/api/docs", "health": "/api/v1/health"}

    @app.get("/api/v1/health", response_model=schemas.Health)
    async def health(request: Request):
        p = request.app.state.predictor
        if p is None:
            return schemas.Health(status="degraded", model_loaded=False, detail=request.app.state.load_error)
        return schemas.Health(status="ok", model_loaded=True, release=p.release,
                              history_cutoff=p.metadata["history_cutoff"])

    @app.get("/api/v1/options")
    async def options(request: Request):
        predictor(request)
        season_teams = request.app.state.season_teams
        last = max(season_teams)
        future = {s: season_teams[last] for s in range(last + 1, schemas.LAST_SUPPORTED_SEASON + 1)}
        return {"first_season": schemas.FIRST_SEASON, "last_data_season": last,
                "last_supported_season": schemas.LAST_SUPPORTED_SEASON,
                "teams_by_season": {str(s): t for s, t in {**season_teams, **future}.items()},
                "venues": sorted(mappings.VENUES),
                "batch": {"max_rows": schemas.MAX_BATCH_ROWS, "columns": schemas.BATCH_INPUT_COLUMNS,
                          "required": schemas.BATCH_REQUIRED_COLUMNS}}

    @app.get("/api/v1/model-metrics")
    async def model_metrics(request: Request):
        meta = predictor(request).metadata
        return {
            "release": meta["release"],
            "history_cutoff": meta["history_cutoff"],
            "training_seasons": meta["training_seasons"],
            "evaluation": meta["evaluation"],
            "prematch": {"model": meta["prematch"]["model"], "test_metrics": meta["prematch"]["test_metrics"]},
            "live": {"innings1_model": meta["live"]["innings1"]["model"],
                     "innings2_model": meta["live"]["innings2"]["model"],
                     "test_accuracy_by_moment": meta["live"]["test_accuracy_by_moment"]},
            "comparison": meta.get("comparison", {}),
            "confidence_bands": meta["confidence_bands"],
            "close_call_label": meta["close_call_label"],
        }

    @app.post("/api/v1/predict", response_model=schemas.Prediction)
    def predict(body: schemas.MatchSetup, request: Request):
        return run_logged(request, "predict", body)

    @app.post("/api/v1/predict-live", response_model=schemas.Prediction)
    def predict_live(body: schemas.LiveRequest, request: Request):
        return run_logged(request, "predict-live", body)

    @app.post("/api/v1/predict-batch", response_model=schemas.BatchResponse)
    def predict_batch(body: schemas.BatchRequest, request: Request):
        predictor(request)
        rows = parse_batch(body.csv)
        out = []
        for row in rows:
            values = {k: v for k, v in row.items() if v != ""}
            started = time.perf_counter()
            prediction, error = None, ""
            try:
                # Any score cell makes it a live row, so a missing innings is reported as such.
                live = any(values.get(c) for c in LIVE_COLUMNS)
                model = schemas.LiveRequest if live else schemas.MatchSetup
                setup = model.model_validate(values)
            except ValidationError as exc:
                error = f"INVALID_REQUEST: {validation_message(exc)}"
                record(request, "predict-batch", values, started, error_code="INVALID_REQUEST",
                       error_message=validation_message(exc))
            else:
                try:
                    prediction = run_logged(request, "predict-batch", setup)
                except ApiError as exc:
                    error = f"{exc.code}: {exc.message}"
                except mappings.UnknownNameError as exc:
                    error = f"UNKNOWN_NAME: {exc}"
            out.append({**row, **result_fields(prediction), "error": error})
        succeeded = sum(1 for r in out if not r["error"])
        return schemas.BatchResponse(rows=len(out), succeeded=succeeded, failed=len(out) - succeeded,
                                     csv=to_csv(out, schemas.BATCH_INPUT_COLUMNS + RESULT_COLUMNS + ["error"]),
                                     results=out)

    @app.get("/api/v1/predictions/log.csv", include_in_schema=True)
    def prediction_log_csv(request: Request):
        rows = [{**{k: r.get(k) for k in ("logged_at", "request_id", "endpoint", "status", "error_code",
                                          "latency_ms")},
                 **(r.get("inputs") or {}), **(r.get("result") or {})}
                for r in request.app.state.log.read_all()]
        return Response(to_csv(rows, LOG_COLUMNS), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="prediction_log.csv"'})

    return app


def parse_batch(text: str) -> list[dict[str, str]]:
    """Rows of a batch CSV as stripped strings. Raises ApiError for a malformed file."""
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff"), newline=""), strict=True)
    try:
        return _batch_rows(reader)
    except csv.Error as exc:
        raise ApiError(400, "INVALID_CSV", f"Cannot read CSV near line {reader.line_num}: {exc}.") from None


def _batch_rows(reader) -> list[dict[str, str]]:
    # The header is the first row with any text, so an empty or blank-lines-only file reads as empty.
    header = next((row for row in reader if any(cell.strip() for cell in row)), None)
    if header is None:
        raise ApiError(400, "INVALID_CSV", "The CSV is empty. It needs a header row.")
    fields = [name.strip() for name in header]
    if any(not name for name in fields):
        raise ApiError(400, "INVALID_CSV", "Every CSV column needs a non-empty header.")
    if len(set(fields)) != len(fields):
        raise ApiError(400, "INVALID_CSV", "CSV column headers must be unique (including after trimming spaces).")
    missing = [c for c in schemas.BATCH_REQUIRED_COLUMNS if c not in fields]
    unknown = [c for c in fields if c and c not in schemas.BATCH_INPUT_COLUMNS]
    if missing or unknown:
        parts = ([f"missing columns: {', '.join(missing)}"] if missing else []) + \
                ([f"unknown columns: {', '.join(unknown)}"] if unknown else [])
        raise ApiError(400, "INVALID_CSV", "; ".join(parts).capitalize() +
                       f". Allowed columns: {', '.join(schemas.BATCH_INPUT_COLUMNS)}.")
    rows = []
    for raw in reader:
        if not any(cell.strip() for cell in raw):
            continue
        if len(raw) != len(fields):
            raise ApiError(400, "INVALID_CSV", f"CSV line {reader.line_num} has {len(raw)} cells; "
                                              f"the header has {len(fields)}. Quote values containing commas "
                                              "and include empty cells for unused columns.")
        row = dict(zip(fields, (cell.strip() for cell in raw)))
        rows.append({c: row.get(c, "") for c in schemas.BATCH_INPUT_COLUMNS})
        if len(rows) > schemas.MAX_BATCH_ROWS:
            raise ApiError(400, "TOO_MANY_ROWS", f"At most {schemas.MAX_BATCH_ROWS} matches per batch.")
    if not rows:
        raise ApiError(400, "INVALID_CSV", "The CSV has a header but no matches.")
    return rows


app = create_app()
