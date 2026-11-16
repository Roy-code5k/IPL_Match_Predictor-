"""Confidence labels and the frozen model selection (Phase 5).

Both were fixed on 2026-11-16 from pre-test data only (walk-forward 2015-2022),
before the 2023-2024 test seasons were evaluated. See reports/calibration_review.md.

A label depends only on the favourite's probability, max(p, 1 - p). The lowest band
is the low-confidence flag proposed in PROJECT.md: a gap below 10 percentage points
between the two teams' probabilities.
"""

from __future__ import annotations

import numpy as np

# Selected in Phase 4 by the rule in src.models.evaluation (lowest walk-forward log loss).
SELECTED_PREMATCH = "random_forest"
SELECTED_LIVE = {1: "logistic_regression", 2: "logistic_regression"}

CLOSE = "Too close to call"
# (lowest favourite probability, label), strongest first.
BANDS = [
    (0.80, "Strong edge"),
    (0.65, "Clear edge"),
    (0.55, "Slight edge"),
    (0.50, CLOSE),
]


def favourite_probability(prob: float) -> float:
    return max(prob, 1 - prob)


def label(prob: float) -> str:
    """Confidence label for a probability of either team winning."""
    if not 0 <= prob <= 1:
        raise ValueError("probability must be between 0 and 1")
    fav = favourite_probability(prob)
    return next(name for lowest, name in BANDS if fav >= lowest)


def band_table(y: np.ndarray, prob: np.ndarray) -> list[list[object]]:
    """Per label: count, share, mean favourite probability, and how often the favourite won."""
    y, prob = np.asarray(y, dtype=int), np.asarray(prob, dtype=float)
    fav = np.maximum(prob, 1 - prob)
    fav_won = np.where(prob >= 0.5, y, 1 - y)
    labels = np.array([label(p) for p in prob])
    rows = []
    for lowest, name in reversed(BANDS):
        mask = labels == name
        upper = min((lo for lo, _ in BANDS if lo > lowest), default=1.0)
        span = f"{lowest:.0%}–{upper:.0%}"
        if mask.any():
            rows.append([name, span, int(mask.sum()), f"{mask.mean():.1%}",
                         f"{fav[mask].mean():.1%}", f"{fav_won[mask].mean():.1%}"])
        else:
            rows.append([name, span, 0, "0.0%", "–", "–"])
    return rows


BAND_HEADERS = ["Label", "Favourite probability", "n", "Share", "Mean predicted", "Favourite won"]
