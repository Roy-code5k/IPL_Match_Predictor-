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
from dataclasses import dataclass
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


@dataclass
class Predictor:
    metadata: dict
    prematch_model: object
    live_models: dict[int, object]
    history: pd.DataFrame
    totals: pd.DataFrame

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
        return cls(
            metadata=metadata,
            prematch_model=joblib.load(release_dir / metadata["prematch"]["file"]),
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
        return {
            "mode": mode,
            "team1": team1,
            "team2": team2,
            "probabilities": {team1: prob_team1, team2: 1 - prob_team1},
            "predicted_winner": team1 if prob_team1 >= 0.5 else team2,
            "confidence": label,
            "close_call": label == cf.CLOSE,
            "history_before": f"{as_of:%Y-%m-%d}",
            "model": {"release": self.release, "name": model_name},
            **extra,
        }

    def predict(self, *, team1: str, team2: str, venue: str, season: int, toss_winner: str,
                toss_decision: str, match_date: str | None = None, is_playoff: bool = False) -> dict:
        """Pre-match prediction (after the toss, before the first ball)."""
        feats, as_of = fb.scenario_features(self.history, team1=team1, team2=team2, venue=venue,
                                            toss_winner=toss_winner, toss_decision=toss_decision,
                                            season=season, match_date=match_date, is_playoff=is_playoff)
        prob = float(self.prematch_model.predict(feats)[0])
        return self._result("pre-match", team1, team2, prob, as_of, self.metadata["prematch"]["model"])

    def predict_live(self, *, team1: str, team2: str, venue: str, season: int, toss_winner: str,
                     toss_decision: str, innings: int, overs_done: int, runs: int, wickets: int,
                     first_innings_total: int | None = None, match_date: str | None = None,
                     is_playoff: bool = False) -> dict:
        """Live prediction at the end of an over (or the innings break: innings=2, overs_done=0)."""
        row, bat_first = lf.live_scenario(self.history, self.totals, team1=team1, team2=team2, venue=venue,
                                          toss_winner=toss_winner, toss_decision=toss_decision,
                                          season=season, match_date=match_date, is_playoff=is_playoff,
                                          innings=innings, overs_done=overs_done, runs=runs,
                                          wickets=wickets, first_innings_total=first_innings_total)
        prob_bf = float(self.live_models[innings].predict_proba(row)[0, 1])
        prob_team1 = prob_bf if bat_first == mappings.normalize_team(team1) else 1 - prob_bf
        as_of = fb.resolve_as_of(self.history, season, match_date)
        state = {"innings": innings, "overs_done": overs_done, "runs": runs, "wickets": wickets}
        if innings == 2:
            state["target"] = first_innings_total + 1
        return self._result("live", team1, team2, prob_team1, as_of,
                            self.metadata["live"][f"innings{innings}"]["model"],
                            batting_first=bat_first, state=state)
