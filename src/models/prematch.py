"""Phase 4: pre-match model candidates (after the toss, before the first ball).

Compares two baselines and three learned models on the 38 leakage-safe features
from `src.features.builder`, using the walk-forward and 2022 validation views in
`src.models.evaluation`. The 2023-2024 test seasons are not touched.

Learned models are trained on both team orders (`swap_teams` augmentation) and
predict by averaging both orders, so P(A beats B) = 1 - P(B beats A) exactly and
the listing order of the teams cannot change a prediction.

All hyperparameters below were fixed before any results were seen.

Run from the project root (needs data/processed/features.csv):
    python -m src.models.prematch
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.inspect_raw import md_table, tidy_markdown
from src.features import builder as fb
from src.models import evaluation as ev

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "prematch_models.md"
FEATURES = fb.FEATURE_COLUMNS
TARGET = fb.TARGET


class Constant:
    """Coin flip: every match is 50/50."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Constant":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)


class EloBaseline:
    """Elo expected score for Team A, used as-is (nothing is fitted)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EloBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["elo_prob_a"].to_numpy(dtype=float)


class Symmetric:
    """Fit on both team orders; predict the average of both orders."""

    def __init__(self, make: Callable[[], object]):
        self.make = make

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Symmetric":
        y = np.asarray(y)
        self.model = self.make()
        self.model.fit(pd.concat([X, fb.swap_teams(X)], ignore_index=True), np.concatenate([y, 1 - y]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        q = self.model.predict_proba(fb.swap_teams(X))[:, 1]
        return (p + 1 - q) / 2


def logistic_regression():
    # Strong L2 penalty: the signal is weak and many features are correlated.
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def random_forest():
    return RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=20,
                                  max_features="sqrt", random_state=ev.SEED, n_jobs=-1)


def xgboost():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                         eval_metric="logloss", random_state=ev.SEED, n_jobs=-1)


# key -> (display name, model factory)
CANDIDATES: dict[str, tuple[str, Callable[[], object]]] = {
    "coin_flip": ("Coin flip (always 50%)", Constant),
    "elo": ("Elo baseline", EloBaseline),
    "logistic_regression": ("Logistic regression", lambda: Symmetric(logistic_regression)),
    "random_forest": ("Random Forest", lambda: Symmetric(random_forest)),
    "xgboost": ("XGBoost", lambda: Symmetric(xgboost)),
}


def fit_predict_for(make: Callable[[], object]) -> ev.FitPredict:
    def fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        model = make().fit(train[FEATURES], train[TARGET].to_numpy())
        return model.predict(evaluate[FEATURES])
    return fit_predict


def load_features(path: Path = fb.FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(int)
    return df


def evaluate_candidates(df: pd.DataFrame) -> dict[str, dict]:
    results = {}
    for key, (name, make) in CANDIDATES.items():
        start = time.perf_counter()
        fp = fit_predict_for(make)
        wf, val = ev.walk_forward(df, fp), ev.validation(df, fp)
        results[key] = {
            "name": name, "walk_forward": wf, "validation": val,
            "wf_metrics": ev.score(wf[TARGET], wf["prob"]),
            "val_metrics": ev.score(val[TARGET], val["prob"]),
            "seconds": time.perf_counter() - start,
        }
    return results


def select(results: dict[str, dict]) -> str:
    """The selection rule from src.models.evaluation: lowest walk-forward log loss."""
    return min(results, key=lambda k: results[k]["wf_metrics"].log_loss)


def logistic_coefficients(df: pd.DataFrame, top: int = 10) -> list[list[object]]:
    """Standardised coefficients of the logistic regression trained on 2008-2021."""
    train = ev.pre_test(df)
    train = train[train["season"] < ev.VALID_SEASON]
    model = Symmetric(logistic_regression).fit(train[FEATURES], train[TARGET].to_numpy())
    coefs = model.model[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))[:top]
    return [[f"`{FEATURES[i]}`", f"{coefs[i]:+.3f}"] for i in order]


def build_report(df: pd.DataFrame, results: dict[str, dict], best: str) -> str:
    wf_rows = [ev.metrics_row(r["name"], r["wf_metrics"]) for r in results.values()]
    val_rows = [ev.metrics_row(r["name"], r["val_metrics"]) for r in results.values()]

    seasons = list(ev.WALK_FORWARD_SEASONS)
    season_rows = []
    for season in seasons:
        row: list[object] = [season]
        parts = [r["walk_forward"][r["walk_forward"]["season"] == season] for r in results.values()]
        for part in parts:
            row.append(f"{ev.score(part[TARGET], part['prob']).accuracy:.1%}")
        season_rows.append(row + [len(parts[0])])  # every model is scored on the same matches

    chosen = results[best]
    (tn, fp_), (fn, tp) = chosen["val_metrics"].confusion
    wf = chosen["walk_forward"]
    pre = ev.pre_test(df)

    return tidy_markdown(f"""
# Pre-match models (Phase 4)

Generated by `python -m src.models.prematch` from `data/processed/features.csv` (feature version {fb.FEATURE_VERSION}). Prediction moment: after the toss, before the first ball. The target is `team1_win` (Team A wins). The 2023–2024 test seasons were not used.

## Setup

- Data before the test seasons: {len(pre):,} matches (2008–2022).
- **Walk-forward 2015–2022:** each season is predicted by a model trained on all earlier seasons ({chosen['wf_metrics'].n} matches in total).
- **Validation 2022:** trained on 2008–2021 ({(pre['season'] < ev.VALID_SEASON).sum()} matches), evaluated on 2022 ({chosen['val_metrics'].n} matches).
- Learned models are trained on both team orders and average both orders when predicting.
- **Selection rule (fixed in advance):** lowest walk-forward log loss. A probability of exactly 50% counts as a Team A prediction, so the coin flip's accuracy is Team A's win share.

Lower log loss and Brier score are better. Log loss for an honest 50/50 guess is 0.6931.

## Walk-forward 2015–2022

{md_table(ev.METRIC_HEADERS, wf_rows)}

## Validation 2022

{md_table(ev.METRIC_HEADERS, val_rows)}

## Walk-forward accuracy by season

{md_table(["Season"] + [r["name"] for r in results.values()] + ["n"], season_rows)}

## Selected model: {chosen['name']}

Validation 2022 confusion matrix (rows = actual, columns = predicted):

{md_table(["", "Predicted Team B wins", "Predicted Team A wins"],
          [["Actual Team B wins", tn, fp_], ["Actual Team A wins", fn, tp]])}

Calibration over the walk-forward predictions (predicted vs observed Team A win rate):

{md_table(["Predicted range", "Matches", "Mean predicted", "Observed"],
          ev.calibration_table(wf[TARGET], wf["prob"]))}

## Logistic regression: largest standardised coefficients (trained on 2008–2021)

A positive coefficient raises Team A's win probability.

{md_table(["Feature", "Coefficient"], logistic_coefficients(df))}
""")


def main() -> int:
    df = load_features()
    results = evaluate_candidates(df)
    best = select(results)
    REPORT_PATH.write_text(build_report(df, results, best), encoding="utf-8", newline="\n")
    print(f"{'Model':26s} {'WF acc':>7s} {'WF AUC':>7s} {'WF logloss':>10s} {'2022 acc':>9s} {'time':>6s}")
    for r in results.values():
        w, v = r["wf_metrics"], r["val_metrics"]
        print(f"{r['name']:26s} {w.accuracy:7.1%} {w.roc_auc:7.3f} {w.log_loss:10.4f} "
              f"{v.accuracy:9.1%} {r['seconds']:5.1f}s")
    print(f"Selected: {results[best]['name']}. Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
