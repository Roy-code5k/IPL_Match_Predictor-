"""Reproducible experiments for recent performance features and training recency.

Run: python -m src.models.prematch_experiment
Writes separate experiment reports; never changes the serving model or v1 metrics.
2023-2024 have already been inspected, so their results are exploratory, not a new
untouched test. Candidate settings and selection rules are recorded before fitting.
"""

from __future__ import annotations

import dataclasses
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.inspect_raw import md_table, sha256
from src.features import builder as fb
from src.features import live as lf
from src.features import performance as pf
from src.models import evaluation as ev
from src.models import prematch as pm

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "prematch_v2_experiment.md"
RESULTS = REPORT.with_suffix(".json")
PROTOCOL = ROOT / "reports" / "prematch_v2_protocol.json"
FEATURE_PATH = ROOT / "data" / "processed" / "prematch_v2_features.csv"
PREDICTION_PATH = ROOT / "data" / "processed" / "prematch_v2_predictions.csv"
POLICIES = ("all", "last_3", "last_5", "decay_2")
HALF_LIFE_YEARS = 2.0
DEVELOPMENT_SEASONS = tuple(range(2015, 2023))
EXPLORATORY_SEASONS = (2023, 2024)


@dataclasses.dataclass(frozen=True)
class Candidate:
    model: str
    features: str
    policy: str

    @property
    def key(self) -> str:
        return f"{self.model}/{self.features}/{self.policy}"

    @property
    def columns(self) -> list[str]:
        return fb.FEATURE_COLUMNS if self.features == "base" else pf.FEATURES


CANDIDATES = tuple(Candidate(model, features, policy)
                   for model in ("random_forest", "logistic_regression")
                   for features in ("base", "performance") for policy in POLICIES)
INCUMBENT = Candidate("random_forest", "base", "all").key


def training_data(df: pd.DataFrame, cutoff_season: int, policy: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Never train on the evaluated season; recency only changes earlier samples."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown training policy: {policy}")
    cutoff = pd.Timestamp(year=cutoff_season, month=1, day=1)
    dates = pd.to_datetime(df.date)
    mask = (df.season < cutoff_season) & (dates < cutoff)
    if policy.startswith("last_"):
        mask &= df.season >= cutoff_season - int(policy.removeprefix("last_"))
    train = df.loc[mask].copy()
    if train.empty:
        raise ValueError(f"No training matches before season {cutoff_season}")
    weights = None
    if policy == "decay_2":
        age = (cutoff - pd.to_datetime(train.date)).dt.days.to_numpy() / 365.25
        weights = np.exp2(-age / HALF_LIFE_YEARS)
        weights /= weights.mean()  # retain regularisation scale across policies
    return train, weights


class ExperimentModel:
    def __init__(self, candidate: Candidate):
        self.candidate = candidate

    def fit(self, train: pd.DataFrame, weights: np.ndarray | None = None) -> "ExperimentModel":
        x = train[self.candidate.columns]
        y = train[fb.TARGET].to_numpy()
        x = pd.concat([x, pf.swap_teams(x)], ignore_index=True)
        y = np.concatenate([y, 1 - y])
        self.model = getattr(pm, self.candidate.model)()
        fit_args = {}
        if weights is not None:
            weights = np.concatenate([weights, weights])
            if self.candidate.model == "logistic_regression":
                fit_args = {"standardscaler__sample_weight": weights,
                            "logisticregression__sample_weight": weights}
            else:
                fit_args = {"sample_weight": weights}
        self.model.fit(x, y, **fit_args)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.candidate.columns]
        p = self.model.predict_proba(x)[:, 1]
        q = self.model.predict_proba(pf.swap_teams(x))[:, 1]
        return (p + 1 - q) / 2


def predict_period(df: pd.DataFrame, candidate: Candidate, seasons: tuple[int, ...],
                   cutoff_season: int) -> pd.DataFrame:
    if min(seasons) < cutoff_season:
        raise ValueError("Evaluation must start on or after the training cutoff")
    train, weights = training_data(df, cutoff_season, candidate.policy)
    evaluate = df[df.season.isin(seasons)].copy()
    if evaluate.empty:
        raise ValueError(f"No evaluation matches for {seasons}")
    model = ExperimentModel(candidate).fit(train, weights)
    out = evaluate[fb.KEY_COLUMNS + [fb.TARGET]].copy()
    out["prob"] = model.predict(evaluate)
    out["training_matches"] = len(train)
    return out


def development_predictions(df: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    return pd.concat([predict_period(df, candidate, (season,), season)
                      for season in DEVELOPMENT_SEASONS], ignore_index=True)


def metrics(pred: pd.DataFrame) -> dict:
    return dataclasses.asdict(ev.score(pred[fb.TARGET], pred.prob))


def select(development_metrics: dict[str, dict]) -> str:
    """Accuracy is the experiment objective; log loss breaks ties. No test input."""
    return min(development_metrics, key=lambda key: (-development_metrics[key]["accuracy"],
                                                    development_metrics[key]["log_loss"], key))


def paired_accuracy_delta(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    left, right = baseline.set_index("match_id"), candidate.set_index("match_id")
    if set(left.index) != set(right.index):
        raise ValueError("Paired comparisons need the same matches")
    right = right.loc[left.index]
    if not left[fb.TARGET].equals(right[fb.TARGET]):
        raise ValueError("Paired comparisons need identical outcomes")
    baseline_correct = (left.prob >= 0.5).to_numpy() == left[fb.TARGET].to_numpy()
    candidate_correct = (right.prob >= 0.5).to_numpy() == right[fb.TARGET].to_numpy()
    diff = candidate_correct.astype(float) - baseline_correct.astype(float)
    rng = np.random.default_rng(ev.SEED)
    sampled = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    return {"accuracy_difference": float(diff.mean()),
            "bootstrap_95_interval": [float(x) for x in np.quantile(sampled, [0.025, 0.975])],
            "newly_correct": int((diff == 1).sum()), "newly_wrong": int((diff == -1).sum())}


def _stable_floats(value):
    """Round floats to 12 significant digits so reruns write identical bytes.

    Parallel Random Forest averaging can change the last bit of a metric between runs.
    """
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {k: _stable_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_floats(v) for v in value]
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(_stable_floats(value), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def report(result: dict, predictions: dict) -> str:
    chosen = result["selected_on_development"]
    rows = []
    for key, item in result["candidates"].items():
        w, t = item["development"], item["exploratory"]
        rows.append([key + (" **selected**" if key == chosen else ""), f"{w['accuracy']:.1%}",
                     f"{w['log_loss']:.4f}", f"{t['accuracy']:.1%}", f"{t['log_loss']:.4f}", f"{t['roc_auc']:.3f}"])
    current = result["candidates"][INCUMBENT]
    selected = result["candidates"][chosen]
    difference = result["paired_exploratory_comparison"]
    lo, hi = difference["bootstrap_95_interval"]
    season_rows = []
    for period, seasons in (("development", DEVELOPMENT_SEASONS), ("exploratory", EXPLORATORY_SEASONS)):
        base, new = predictions[INCUMBENT][period], predictions[chosen][period]
        for season in seasons:
            old = metrics(base[base.season == season])
            fresh = metrics(new[new.season == season])
            season_rows.append([season, period, old["n"], f"{old['accuracy']:.1%}", f"{fresh['accuracy']:.1%}"])
    baseline_rows = [[period, key, f"{m['accuracy']:.1%}", f"{m['log_loss']:.4f}"]
                     for period, group in result["baselines"].items() for key, m in group.items()]
    return f"""# Pre-match v2 experiment: performance features and recent training

Reproduce with `python -m src.models.prematch_experiment`. Protocol: `reports/prematch_v2_protocol.json`.
Detailed metrics: `reports/prematch_v2_experiment.json`. Per-match probabilities: `data/processed/prematch_v2_predictions.csv` (regenerated, not tracked).

## Result

- Selected on **2015-2022 development predictions only**: `{chosen}`.
- Development accuracy: incumbent **{current['development']['accuracy']:.1%}**, selected **{selected['development']['accuracy']:.1%}**.
- Exploratory 2023-2024 accuracy: incumbent **{current['exploratory']['accuracy']:.1%}**, selected **{selected['exploratory']['accuracy']:.1%}**, on the same {selected['exploratory']['n']} matches.
- Exploratory change: **{difference['accuracy_difference'] * 100:+.2f} percentage points**. Paired match-bootstrap 95% interval: **{lo * 100:+.2f} to {hi * 100:+.2f} points** ({difference['newly_correct']} newly correct, {difference['newly_wrong']} newly wrong).
- Exploratory log loss: incumbent **{current['exploratory']['log_loss']:.4f}**, selected **{selected['exploratory']['log_loss']:.4f}**; lower is better.
- **No production promotion.** The existing release, API, dashboard metrics and original evaluation report remain unchanged. A newer untouched season is needed before claiming a validated improvement.

## Interpretation and protocol

2023-2024 were already inspected before this experiment was designed. Their results are an exploratory comparison, **not a fresh unbiased test**. The bootstrap interval is descriptive and does not account for selecting among candidates or season-level dependence. Development scores also contain selection optimism. Do not quote the best 2023-2024 row as a separately validated improvement.

The 16 settings were fixed before the run: Random Forest and logistic regression with their existing hyperparameters; original 38 features or 64 features including performance; all earlier seasons, last 3, last 5, or exponential weighting with a 2-year half-life. Accuracy on pooled development predictions selects the experiment candidate; lower log loss breaks ties. This accuracy-first experiment rule does not change the original release's log-loss selection rule.

Development predicts each season 2015-2022 using a fit on earlier seasons. Exploratory evaluation fits once before 2023 and predicts both 2023 and 2024, matching the incumbent's evaluation protocol. Historical features may incorporate completed earlier dates within evaluation seasons, but model weights are not refitted there. Same-day outcomes and deliveries are always excluded from features.

## New information

The additional 26 columns describe the last 10 eligible matches per team: overall batting run rate and bowling runs conceded per over, wickets lost/taken per 120 legal balls, powerplay (overs 1-6) and death-over (16-20) run rates. Each of eight metrics has Team A, Team B and difference columns; two columns record how much recent history is available.

Rates are pooled by legal-ball exposure and smoothed with earlier league form (last 60 team innings). Wides/no-balls add runs but no legal balls. Retired hurt is not a wicket; run-outs are team wickets. Bowling rates include all team runs conceded, rather than individual bowler economy. Super overs and rain-shortened/D/L matches do not enter the performance history. They can still be target matches for pre-match prediction. Missing death-over exposure uses the past-only prior.

Training windows restrict fitted examples, not the historical information available to create a feature. Recency weights are normalised to mean one, duplicated with team-swap augmentation, and applied to both the logistic scaler and classifier. All predictions average the two team orders.

## All candidates

{md_table(['Candidate / features / training policy', 'Dev accuracy', 'Dev log loss', '2023-24 accuracy', '2023-24 log loss', '2023-24 AUC'], rows)}

## Reference baselines

{md_table(['Period', 'Baseline', 'Accuracy', 'Log loss'], baseline_rows)}

The 50/50 baseline always picks Team A at an exact tie; its accuracy is Team A's win share.

## Selected candidate by season

{md_table(['Season', 'Use', 'Matches', 'Incumbent accuracy', 'Selected accuracy'], season_rows)}

The model selection was not changed after inspecting this table. No newer data, playing-XI inputs, hyperparameter search, or serving changes are included in this experiment.
"""


def main() -> int:
    matches = fb.load_clean()
    base = pm.load_features()
    base["date"] = pd.to_datetime(base.date)
    if set(base.match_id) != set(matches.match_id):
        raise ValueError("Rebuild features.csv: match IDs do not match the cleaned history")
    protocol = {
        "experiment": "prematch-v2-performance-recency-1",
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "exploratory_seasons": list(EXPLORATORY_SEASONS),
        "selection": "highest pooled development accuracy, then lowest log loss, then candidate key",
        "warning": "2023-2024 were already inspected; no fresh holdout or automatic promotion",
        "candidates": [dataclasses.asdict(c) for c in CANDIDATES],
        "feature_version": pf.VERSION, "features": pf.FEATURES,
        "team_window": pf.TEAM_WINDOW, "league_window": pf.LEAGUE_WINDOW,
        "rate_definitions": pf.RATES, "half_life_years": HALF_LIFE_YEARS,
        "estimators": {name: repr(getattr(pm, name)()) for name in ("random_forest", "logistic_regression")},
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in
                         (fb.CLEAN_PATH, fb.FEATURES_PATH, lf.RAW_DELIVERIES, ROOT / 'data/raw/matches.csv')},
        "code_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in
                        (Path(__file__), Path(pf.__file__), Path(pm.__file__), Path(fb.__file__))},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL, protocol)  # persisted before any estimator is fitted
    print("Protocol saved. Building past-only batting/bowling features...", flush=True)
    records = pf.aggregate_performance(matches, pf.load_deliveries(), lf.full_length_ids())
    df = pf.enrich_features(base, records)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_PATH, index=False, float_format="%.6f", lineterminator="\n")
    predictions, scores = {}, {}
    for candidate in CANDIDATES:
        wf = development_predictions(df, candidate)
        scores[candidate.key] = metrics(wf)
        predictions[candidate.key] = {"development": wf}
        print(f"Development {candidate.key:48s} accuracy={scores[candidate.key]['accuracy']:.1%} "
              f"log_loss={scores[candidate.key]['log_loss']:.4f}", flush=True)
    chosen = select(scores)
    print(f"Selection frozen before exploratory comparison: {chosen}", flush=True)
    result = {"protocol_sha256": sha256(PROTOCOL), "selected_on_development": chosen,
              "performance_history_matches": int(records.match_id.nunique()), "candidates": {}, "baselines": {}}
    for candidate in CANDIDATES:
        pred = predict_period(df, candidate, EXPLORATORY_SEASONS, min(EXPLORATORY_SEASONS))
        predictions[candidate.key]["exploratory"] = pred
        result["candidates"][candidate.key] = {"development": scores[candidate.key], "exploratory": metrics(pred)}
        print(f"Exploratory {candidate.key:48s} accuracy={metrics(pred)['accuracy']:.1%}", flush=True)
    for period, seasons in (("development", DEVELOPMENT_SEASONS), ("exploratory", EXPLORATORY_SEASONS)):
        subset = df[df.season.isin(seasons)]
        result["baselines"][period] = {"coin_flip": metrics(subset.assign(prob=0.5)),
                                       "elo": metrics(subset.assign(prob=subset.elo_prob_a))}
    result["paired_exploratory_comparison"] = paired_accuracy_delta(
        predictions[INCUMBENT]["exploratory"], predictions[chosen]["exploratory"])
    pd.concat([pred.assign(candidate=key, period=period) for key, parts in predictions.items()
               for period, pred in parts.items()], ignore_index=True).to_csv(
                   PREDICTION_PATH, index=False, float_format="%.10f", lineterminator="\n")
    result["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in (FEATURE_PATH, PREDICTION_PATH)}
    write_json(RESULTS, result)
    REPORT.write_text(report(result, predictions), encoding="utf-8", newline="\n")
    print(f"Wrote {REPORT.relative_to(ROOT)}. Serving release unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
