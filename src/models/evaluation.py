"""Shared evaluation tools: the time-based split, walk-forward folds, and metrics.

Split (decided 2026-11-13, confirmed at the start of Phase 4 on 2026-11-16):
    train 2008-2021, validate 2022, test 2023-2024.

Only `final_test` reads the test seasons; Phase 5 calls it once the selection and
confidence labels are frozen (src/models/confidence.py). Phase 4 compares
candidates on two views of the pre-test data:

- **Walk-forward 2015-2022:** for each season, train on all earlier seasons and
  predict that season. This pools about 490 matches, so differences between
  models are far less noisy than on 2022 alone.
- **Validation 2022:** train on 2008-2021, predict 2022 (74 matches).

Model selection rule (fixed before any results were seen): the candidate with the
lowest walk-forward log loss is selected. Log loss rewards honest probabilities,
which is what the app shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)

TRAIN_SEASONS = range(2008, 2022)
VALID_SEASON = 2022
TEST_SEASONS = (2023, 2024)
WALK_FORWARD_SEASONS = range(2015, 2023)
SEED = 42

# fit_predict(train, evaluate) -> probability of the positive class for each row of `evaluate`
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def pre_test(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from before the test seasons. Everything in Phase 4 goes through this."""
    return df[df["season"] < min(TEST_SEASONS)]


def walk_forward(df: pd.DataFrame, fit_predict: FitPredict,
                 seasons=WALK_FORWARD_SEASONS) -> pd.DataFrame:
    """Out-of-sample probabilities for every row in `seasons`.

    Each season is predicted by a model trained only on strictly earlier seasons.
    Returns `df`'s rows for those seasons with an added `prob` column.
    """
    df = pre_test(df)
    parts = []
    for season in seasons:
        train, evaluate = df[df["season"] < season], df[df["season"] == season]
        if evaluate.empty:
            continue
        out = evaluate.copy()
        out["prob"] = fit_predict(train, evaluate)
        parts.append(out)
    return pd.concat(parts)


def validation(df: pd.DataFrame, fit_predict: FitPredict) -> pd.DataFrame:
    """Train on 2008-2021, predict 2022."""
    df = pre_test(df)
    train, evaluate = df[df["season"] < VALID_SEASON], df[df["season"] == VALID_SEASON]
    out = evaluate.copy()
    out["prob"] = fit_predict(train, evaluate)
    return out


def final_test(df: pd.DataFrame, fit_predict: FitPredict) -> pd.DataFrame:
    """Phase 5 only, after selection is frozen: train on 2008-2022, predict 2023-2024."""
    train = df[df["season"] < min(TEST_SEASONS)]
    out = df[df["season"].isin(TEST_SEASONS)].copy()
    out["prob"] = fit_predict(train, out)
    return out


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    accuracy_ci: float  # half-width of a normal-approximation 95% interval
    f1: float
    roc_auc: float
    log_loss: float
    brier: float
    confusion: tuple[tuple[int, int], tuple[int, int]]  # rows: actual 0/1, columns: predicted 0/1


def score(y: np.ndarray, prob: np.ndarray) -> Metrics:
    """Metrics for binary outcomes. A probability of exactly 0.5 predicts class 1."""
    y = np.asarray(y, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, prob) if len(set(y)) == 2 else float("nan")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return Metrics(
        n=len(y),
        accuracy=acc,
        accuracy_ci=1.96 * math.sqrt(acc * (1 - acc) / len(y)),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=auc,
        log_loss=log_loss(y, prob, labels=[0, 1]),
        brier=brier_score_loss(y, prob),
        confusion=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
    )


def metrics_row(name: str, m: Metrics) -> list[object]:
    return [name, m.n, f"{m.accuracy:.1%} ± {m.accuracy_ci:.1%}", f"{m.f1:.3f}",
            f"{m.roc_auc:.3f}", f"{m.log_loss:.4f}", f"{m.brier:.4f}"]


METRIC_HEADERS = ["Model", "n", "Accuracy (95% CI)", "F1", "ROC-AUC", "Log loss", "Brier"]


def calibration_table(y: np.ndarray, prob: np.ndarray, bins: int = 5) -> list[list[object]]:
    """Predicted vs observed rate in equal-width probability bins (empty bins skipped)."""
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & ((prob < hi) | (hi == 1.0))
        if mask.any():
            rows.append([f"{lo:.0%}–{hi:.0%}", int(mask.sum()),
                         f"{prob[mask].mean():.1%}", f"{y[mask].mean():.1%}"])
    return rows
