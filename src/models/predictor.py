"""Load a model release and make predictions (used by the Phase 6 API).

    predictor = Predictor.load()                     # models/release-<version>/
    predictor.predict(team1=..., team2=..., venue=..., season=2025,
                      toss_winner=..., toss_decision="field")
    predictor.predict_live(..., innings=2, overs_done=10, runs=85, wickets=3,
                           first_innings_total=180)

Loading checks every file against the hashes in metadata.json, so a model can never
be paired with a different history snapshot. Unknown team or venue names raise
`src.data.mappings.UnknownNameError`; an impossible match state raises ValueError.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.data import mappings
from src.data.inspect_raw import sha256
from src.features import builder as fb
from src.features import live as lf
from src.models import confidence as cf
from src.models import release as rl


class ReleaseError(RuntimeError):
    """Raised when a release is missing or its files do not match metadata.json."""


HISTORY_CACHE_SIZE = 64


def _plain(x):
    """numpy scalars -> plain Python values, so results serialise cleanly."""
    return bool(x) if isinstance(x, bool) else (x.item() if hasattr(x, "item") else x)


def _pair(label: str, a: float, b: float, fmt: str, note: str | None = None) -> dict:
    item = {"label": label, "team1": _plain(a), "team2": _plain(b), "format": fmt}
    return {**item, "note": note} if note else item


def _single(label: str, value: float, fmt: str) -> dict:
    return {"label": label, "value": _plain(value), "format": fmt}


def prematch_inputs(f: pd.Series) -> list[dict]:
    """The pre-match features a person can read, from Team A's (team1's) point of view."""
    return [
        _pair("Elo rating", round(f.elo_a, 1), round(f.elo_b, 1), "number"),
        _pair("Win rate, all matches", f.win_rate_a, f.win_rate_b, "percent"),
        _pair("Form, last 10 matches", f.form10_a, f.form10_b, "percent"),
        _pair("Win rate this season", f.season_win_rate_a, f.season_win_rate_b, "percent",
              f"{int(f.season_played_a)} and {int(f.season_played_b)} matches played this season"),
        _pair("Head-to-head win rate", f.h2h_win_rate_a, 1 - f.h2h_win_rate_a, "percent",
              f"{int(f.h2h_played)} earlier meetings"),
        _pair("Win rate at this ground", f.venue_win_rate_a, f.venue_win_rate_b, "percent",
              f"{int(f.venue_played_a)} and {int(f.venue_played_b)} matches here"),
        _pair("Home ground", bool(f.home_a), bool(f.home_b), "yesno"),
        _pair("Bats first", bool(f.a_bats_first), not bool(f.a_bats_first), "yesno"),
        _single("Chasing side's win rate at this ground", f.venue_chase_rate, "percent"),
    ]


def live_inputs(f: pd.Series, innings: int) -> list[dict]:
    """The live features a person can read, from the batting-first side's point of view."""
    items = [_single("Par first-innings score (recent matches)", round(f.par_score, 1), "number"),
             _single("Batting-first side's pre-match Elo chance", f.elo_prob_bf, "percent"),
             _single("Chasing side's win rate at this ground", f.venue_chase_rate, "percent")]
    if innings == 1:
        return [_single("Run rate", round(f.run_rate, 2), "number"),
                _single("Projected total at this rate", round(f.run_rate * lf.OVERS), "number"),
                _single("Wickets lost", int(f.wickets), "number"), *items]
    return [_single("Target", int(f.target), "number"),
            _single("Runs needed", int(f.runs_needed), "number"),
            _single("Balls left", int(f.balls_left), "number"),
            _single("Required run rate", round(f.required_rate, 2), "number"),
            _single("Wickets left", int(f.wickets_left), "number"), *items]


@dataclass
class Predictor:
    metadata: dict
    prematch_model: object
    live_models: dict[int, object]
    history: pd.DataFrame
    totals: pd.DataFrame
    _histories: dict = field(default_factory=dict, repr=False)

    def history_for(self, as_of: pd.Timestamp) -> fb.History:
        """History snapshot before `as_of`, cached because building it dominates prediction time."""
        cached = self._histories.get(as_of)
        if cached is None:
            cached = fb.History.from_matches(self.history, before=as_of)
            if len(self._histories) >= HISTORY_CACHE_SIZE:
                self._histories.pop(next(iter(self._histories)))
            self._histories[as_of] = cached
        return cached

    @classmethod
    def load(cls, release_dir: Path = rl.RELEASE_DIR) -> "Predictor":
        meta_path = release_dir / "metadata.json"
        if not meta_path.exists():
            raise ReleaseError(f"No release at {release_dir}. Build it with: python -m src.models.release")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for name, expected in metadata["files"].items():
            path = release_dir / name
            if not path.exists():
                raise ReleaseError(f"{name} is missing. Rebuild with: python -m src.models.release")
            if sha256(path) != expected:
                raise ReleaseError(f"{name} does not match metadata.json. Rebuild with: python -m src.models.release")
        for package, wanted in metadata["packages"].items():
            if version(package) != wanted:
                warnings.warn(f"{package} {version(package)} differs from the release ({wanted}); "
                              "install requirements.txt", stacklevel=2)
        history = pd.read_csv(release_dir / "history.csv", parse_dates=["date"])
        totals = pd.read_csv(release_dir / "first_innings_totals.csv", parse_dates=["date"])
        prematch_model = joblib.load(release_dir / metadata["prematch"]["file"])
        # One match at a time: thread start-up costs more than it saves (same probabilities).
        inner = getattr(prematch_model, "model", prematch_model)
        if hasattr(inner, "n_jobs"):
            inner.n_jobs = 1
        return cls(
            metadata=metadata,
            prematch_model=prematch_model,
            live_models={i: joblib.load(release_dir / metadata["live"][f"innings{i}"]["file"]) for i in (1, 2)},
            history=history,
            totals=totals,
        )

    @property
    def release(self) -> str:
        return self.metadata["release"]

    def _result(self, mode: str, team1: str, team2: str, prob_team1: float, as_of: pd.Timestamp,
                model_name: str, **extra) -> dict:
        team1, team2 = mappings.normalize_team(team1), mappings.normalize_team(team2)
        label = cf.label(prob_team1)
        trained_through = pd.Timestamp(self.metadata["history_cutoff"])
        return {
            "mode": mode,
            "team1": team1,
            "team2": team2,
            "probabilities": {team1: prob_team1, team2: 1 - prob_team1},
            "predicted_winner": team1 if prob_team1 >= 0.5 else team2,
            "confidence": label,
            "close_call": label == cf.CLOSE,
            "history_before": f"{as_of:%Y-%m-%d}",
            "model_trained_through": f"{trained_through:%Y-%m-%d}",
            "retrospective": bool(as_of <= trained_through),
            "model": {"release": self.release, "name": model_name},
            **extra,
        }

    def predict(self, *, team1: str, team2: str, venue: str, season: int, toss_winner: str,
                toss_decision: str, match_date: str | None = None, is_playoff: bool = False) -> dict:
        """Pre-match estimate; dates within model training are labelled retrospective.

        The date restricts feature history, not the fitted model's training data.
        """
        feats, as_of = fb.scenario_features(self.history, team1=team1, team2=team2, venue=venue,
                                            toss_winner=toss_winner, toss_decision=toss_decision,
                                            season=season, match_date=match_date, is_playoff=is_playoff,
                                            history_for=self.history_for)
        prob = float(self.prematch_model.predict(feats)[0])
        return self._result("pre-match", team1, team2, prob, as_of, self.metadata["prematch"]["model"],
                            inputs=prematch_inputs(feats.iloc[0]))

    def predict_live(self, *, team1: str, team2: str, venue: str, season: int, toss_winner: str,
                     toss_decision: str, innings: int, overs_done: int, runs: int, wickets: int,
                     first_innings_total: int | None = None, match_date: str | None = None,
                     is_playoff: bool = False) -> dict:
        """Live prediction at the end of an over (or the innings break: innings=2, overs_done=0)."""
        row, bat_first = lf.live_scenario(self.history, self.totals, team1=team1, team2=team2, venue=venue,
                                          toss_winner=toss_winner, toss_decision=toss_decision,
                                          season=season, match_date=match_date, is_playoff=is_playoff,
                                          innings=innings, overs_done=overs_done, runs=runs,
                                          wickets=wickets, first_innings_total=first_innings_total,
                                          history_for=self.history_for)
        prob_bf = float(self.live_models[innings].predict_proba(row)[0, 1])
        prob_team1 = prob_bf if bat_first == mappings.normalize_team(team1) else 1 - prob_bf
        as_of = fb.resolve_as_of(self.history, season, match_date)
        state = {"innings": innings, "overs_done": overs_done, "runs": runs, "wickets": wickets}
        if innings == 2:
            state["target"] = first_innings_total + 1
        return self._result("live", team1, team2, prob_team1, as_of,
                            self.metadata["live"][f"innings{innings}"]["model"],
                            batting_first=bat_first, state=state, inputs=live_inputs(row.iloc[0], innings))
