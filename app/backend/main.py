"""FastAPI service for IPL match predictions (Phase 6).

Run from the project root, after `python -m src.models.release`:
    uvicorn app.backend.main:app --reload

Routes (all under /api/v1):
    GET  /health          service and model-load status
    GET  /options         seasons, teams per season and venues the API accepts
    GET  /model-metrics   the loaded release's test metrics and confidence bands
    POST /predict         pre-match prediction (after the toss)
    POST /predict-live    in-match prediction from the score at the end of an over

Errors always have the shape {"error": {"code", "message", "request_id"}}; the same
request id is sent in the X-Request-ID response header.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.backend import schemas
from src.data import mappings
from src.models.predictor import Predictor, ReleaseError
from src.models.release import RELEASE_DIR

log = logging.getLogger("ipl.api")


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


def create_app(release_dir: Path | None = None) -> FastAPI:
    release_dir = Path(release_dir or os.environ.get("IPL_RELEASE_DIR", RELEASE_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = Predictor.load(release_dir)
            app.state.season_teams = teams_by_season(app.state.predictor.history)
            app.state.load_error = None
        except (ReleaseError, OSError, KeyError) as exc:
            app.state.predictor, app.state.season_teams = None, {}
            app.state.load_error = str(exc)
            log.error("Model release not loaded: %s", exc)
        yield

    app = FastAPI(title="IPL Match Winner Prediction API", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")

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
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "Invalid request").removeprefix("Value error, ")
        return _error(request, 422, "INVALID_REQUEST", f"{field}: {message}" if field else message)

    @app.exception_handler(mappings.UnknownNameError)
    async def unknown_name(request: Request, exc: mappings.UnknownNameError):
        return _error(request, 400, "UNKNOWN_NAME", str(exc))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("Unhandled error (request %s)", getattr(request.state, "request_id", "-"))
        return _error(request, 500, "INTERNAL_ERROR", "Something went wrong. Quote the request id if you report it.")

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
                "venues": sorted(mappings.VENUES)}

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
            "confidence_bands": meta["confidence_bands"],
            "close_call_label": meta["close_call_label"],
        }

    def to_response(result: dict) -> schemas.Prediction:
        t1, t2 = result["team1"], result["team2"]
        return schemas.Prediction(
            mode=result["mode"], team1=t1, team2=t2,
            team1_probability=round(result["probabilities"][t1], 4),
            team2_probability=round(result["probabilities"][t2], 4),
            predicted_winner=result["predicted_winner"], confidence=result["confidence"],
            close_call=result["close_call"], history_before=result["history_before"],
            model=schemas.ModelInfo(**result["model"]),
            batting_first=result.get("batting_first"),
            state=schemas.LiveState(**result["state"]) if "state" in result else None,
        )

    def setup_kwargs(body: schemas.MatchSetup) -> dict:
        return {"team1": body.team1, "team2": body.team2, "venue": body.venue, "season": body.season,
                "toss_winner": body.toss_winner, "toss_decision": body.toss_decision,
                "match_date": body.match_date.isoformat() if body.match_date else None,
                "is_playoff": body.is_playoff}

    @app.post("/api/v1/predict", response_model=schemas.Prediction)
    def predict(body: schemas.MatchSetup, request: Request):
        p = predictor(request)
        check_setup(request, body)
        try:
            return to_response(p.predict(**setup_kwargs(body)))
        except mappings.UnknownNameError:
            raise
        except ValueError as exc:
            raise ApiError(400, "INVALID_INPUT", str(exc)) from None

    @app.post("/api/v1/predict-live", response_model=schemas.Prediction)
    def predict_live(body: schemas.LiveRequest, request: Request):
        p = predictor(request)
        check_setup(request, body)
        try:
            return to_response(p.predict_live(**setup_kwargs(body), innings=body.innings,
                                              overs_done=body.overs_done, runs=body.runs,
                                              wickets=body.wickets, first_innings_total=body.first_innings_total))
        except mappings.UnknownNameError:
            raise
        except ValueError as exc:
            raise ApiError(400, "INVALID_MATCH_STATE", str(exc)) from None

    return app


app = create_app()
