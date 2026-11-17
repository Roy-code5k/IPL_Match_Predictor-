"""Request and response bodies for the prediction API."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIRST_SEASON = 2008
LAST_SUPPORTED_SEASON = 2027  # data seasons (2008-2024) plus three future seasons
MAX_RUNS = 400


class MatchSetup(BaseModel):
    """What is known after the toss."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    team1: str = Field(min_length=1, description="Team A")
    team2: str = Field(min_length=1, description="Team B")
    venue: str = Field(min_length=1)
    season: int = Field(ge=FIRST_SEASON, le=LAST_SUPPORTED_SEASON)
    toss_winner: str = Field(min_length=1)
    toss_decision: Literal["bat", "field"]
    match_date: dt.date | None = Field(default=None, description="Optional; history stops before this day")
    is_playoff: bool = False

    @model_validator(mode="after")
    def check_consistency(self) -> "MatchSetup":
        if self.team1 == self.team2:
            raise ValueError("team1 and team2 must be different franchises")
        if self.toss_winner not in (self.team1, self.team2):
            raise ValueError("toss_winner must be team1 or team2")
        if self.match_date is not None and self.match_date.year != self.season:
            raise ValueError(f"match_date must fall in the {self.season} season")
        return self


class LiveRequest(MatchSetup):
    """Match setup plus the score at the end of an over (or the innings break)."""

    innings: int = Field(ge=1, le=2, description="1 = first innings, 2 = chase")  # int, so CSV text "2" works
    overs_done: int = Field(ge=0, le=19, description="Completed overs in the current innings")
    runs: int = Field(ge=0, le=MAX_RUNS)
    wickets: int = Field(ge=0, le=9)
    first_innings_total: int | None = Field(default=None, ge=0, le=MAX_RUNS)

    @model_validator(mode="after")
    def check_state(self) -> "LiveRequest":
        if self.innings == 1:
            if self.overs_done < 1:
                raise ValueError("first-innings predictions need at least 1 completed over; "
                                 "use the pre-match prediction before that")
            if self.first_innings_total is not None:
                raise ValueError("first_innings_total is only used in the second innings")
        else:
            if self.first_innings_total is None:
                raise ValueError("first_innings_total is required in the second innings")
            if self.overs_done == 0 and (self.runs != 0 or self.wickets != 0):
                raise ValueError("0 overs means the innings break: runs and wickets must both be 0")
            if self.runs > self.first_innings_total:
                raise ValueError("the chase is already won; no prediction is needed")
        return self


class LiveState(BaseModel):
    innings: int
    overs_done: int
    runs: int
    wickets: int
    target: int | None = None


class ModelInfo(BaseModel):
    release: str
    name: str


class InputItem(BaseModel):
    """One model input shown to the user: a pair (team1, team2) or a single value."""

    label: str
    format: Literal["number", "percent", "yesno"]
    team1: bool | float | None = None
    team2: bool | float | None = None
    value: float | None = None
    note: str | None = None


class Prediction(BaseModel):
    mode: Literal["pre-match", "live"]
    team1: str
    team2: str
    team1_probability: float
    team2_probability: float
    predicted_winner: str
    confidence: str
    close_call: bool
    history_before: str = Field(description="Historical input features use only matches before this date")
    model_trained_through: str = Field(description="Last match date used to train the fitted model")
    retrospective: bool = Field(description="True when the feature cutoff is within model training; "
                                           "this is a retrospective simulation, not an unseen forecast")
    model: ModelInfo
    batting_first: str | None = None
    state: LiveState | None = None
    inputs: list[InputItem] = Field(default_factory=list,
                                    description="Key model inputs (from matches before history_before)")


MAX_BATCH_ROWS = 200
BATCH_INPUT_COLUMNS = ["team1", "team2", "venue", "season", "toss_winner", "toss_decision", "match_date",
                       "is_playoff", "innings", "overs_done", "runs", "wickets", "first_innings_total"]
BATCH_REQUIRED_COLUMNS = ["team1", "team2", "venue", "season", "toss_winner", "toss_decision"]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    csv: str = Field(max_length=200_000,
                     description=f"CSV text with a header row; at most {MAX_BATCH_ROWS} matches")


class BatchResponse(BaseModel):
    rows: int
    succeeded: int
    failed: int
    csv: str
    results: list[dict]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    release: str | None = None
    history_cutoff: str | None = None
    detail: str | None = None
