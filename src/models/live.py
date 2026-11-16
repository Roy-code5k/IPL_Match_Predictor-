"""Phase 4 (live mode): in-match win-probability models.

One model per innings predicts P(batting-first side wins) from the match state at
the end of an over (see `src.features.live`). Candidates are compared on the same
walk-forward 2015-2022 and 2022 validation views as the pre-match models, with the
same selection rule (lowest walk-forward log loss, over all in-progress states of
that innings). The 2023-2024 test seasons are not touched.

Accuracy is reported at fixed checkpoints, counting only matches still in progress
there, so a chase that already finished never inflates a later checkpoint.

All hyperparameters below were fixed before any results were seen.

Run from the project root (needs data/raw/*.csv and data/processed/*.csv):
    python -m src.models.live
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import viz
from src.data.inspect_raw import md_table, tidy_markdown
from src.features import builder as fb
from src.features import live as lf
from src.models import evaluation as ev
from src.models import prematch as pm

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "live_models.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "08_live_win_probability_2022_final.png"
TARGET = lf.TARGET

# (innings, completed overs) -> label. Innings 2 at 0 overs is the innings break.
CHECKPOINTS = {
    (1, 6): "1st innings, after 6 overs",
    (1, 10): "1st innings, after 10 overs",
    (1, 15): "1st innings, after 15 overs",
    (2, 0): "Innings break (target known)",
    (2, 6): "Chase, after 6 overs",
    (2, 10): "Chase, after 10 overs",
    (2, 15): "Chase, after 15 overs",
}


class PrematchElo:
    """Ignores the match state: the batting-first side's pre-match Elo probability."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PrematchElo":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["elo_prob_bf"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def logistic_regression():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def random_forest():
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                  max_features="sqrt", random_state=ev.SEED, n_jobs=-1)


def xgboost():
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
                         eval_metric="logloss", random_state=ev.SEED, n_jobs=-1)


CANDIDATES: dict[str, tuple[str, Callable[[], object]]] = {
    "prematch_elo": ("Pre-match Elo only", PrematchElo),
    "logistic_regression": ("Logistic regression", logistic_regression),
    "random_forest": ("Random Forest", random_forest),
    "xgboost": ("XGBoost", xgboost),
}


def fit_predict_for(make: Callable[[], object], innings: int) -> ev.FitPredict:
    cols = lf.FEATURES_BY_INNINGS[innings]

    def fit_predict(train: pd.DataFrame, evaluate: pd.DataFrame) -> np.ndarray:
        model = make().fit(train[cols], train[TARGET].to_numpy())
        return model.predict_proba(evaluate[cols])[:, 1]
    return fit_predict


def load_states() -> pd.DataFrame:
    matches = fb.load_clean()
    deliveries = lf.load_deliveries()
    return lf.build_live_states(matches, pm.load_features(), deliveries, lf.full_length_ids())


def evaluate_innings(states: pd.DataFrame, innings: int) -> dict[str, dict]:
    data = states[states["innings"] == innings]
    results = {}
    for key, (name, make) in CANDIDATES.items():
        start = time.perf_counter()
        fp = fit_predict_for(make, innings)
        wf, val = ev.walk_forward(data, fp), ev.validation(data, fp)
        results[key] = {"name": name, "walk_forward": wf, "validation": val,
                        "wf_metrics": ev.score(wf[TARGET], wf["prob"]),
                        "val_metrics": ev.score(val[TARGET], val["prob"]),
                        "seconds": time.perf_counter() - start}
    return results


def select(results: dict[str, dict]) -> str:
    return min(results, key=lambda k: results[k]["wf_metrics"].log_loss)


def at_checkpoint(pred: pd.DataFrame, innings: int, overs: int) -> ev.Metrics:
    part = pred[(pred["innings"] == innings) & (pred["overs_done"] == overs)]
    return ev.score(part[TARGET], part["prob"])


def prematch_on_live_sample(states: pd.DataFrame) -> tuple[ev.Metrics, ev.Metrics, str]:
    """The selected pre-match model, re-scored on the live sample from the batting-first view."""
    features = pm.load_features()
    results = pm.evaluate_candidates(features)
    best = results[pm.select(results)]
    sample = states.loc[(states["innings"] == 2) & (states["overs_done"] == 0), ["match_id", TARGET]]

    def rescore(pred: pd.DataFrame) -> ev.Metrics:
        joined = pred.merge(sample, on="match_id")
        prob_bf = np.where(joined["a_bats_first"] == 1, joined["prob"], 1 - joined["prob"])
        return ev.score(joined[TARGET], prob_bf)

    return rescore(best["walk_forward"]), rescore(best["validation"]), best["name"]


def win_probability_curve(states: pd.DataFrame, chosen: dict[int, str]) -> None:
    """Chart of the 2022 final using models trained on 2008-2021 (the validation models)."""
    pre = ev.pre_test(states)
    train = pre[pre["season"] < ev.VALID_SEASON]
    final_id = pre.loc[pre["season"] == ev.VALID_SEASON].sort_values(["date", "match_id"])["match_id"].iloc[-1]
    match = pre[pre["match_id"] == final_id].sort_values(["innings", "overs_done"])

    xs, ps = [0.0], [float(match["elo_prob_bf"].iloc[0])]  # x = overs bowled in the match
    for innings in (1, 2):
        cols = lf.FEATURES_BY_INNINGS[innings]
        part = match[match["innings"] == innings]
        tr = train[train["innings"] == innings]
        model = CANDIDATES[chosen[innings]][1]().fit(tr[cols], tr[TARGET].to_numpy())
        xs += list(part["overs_done"] + (20 if innings == 2 else 0))
        ps += list(model.predict_proba(part[cols])[:, 1])

    bat_first, chaser = match["bat_first"].iloc[0], match["chaser"].iloc[0]
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0.5, color=viz.BASELINE, linewidth=0.8)
    ax.axvline(20, color=viz.BASELINE, linewidth=0.8, linestyle="--")
    ax.plot(xs, ps, color=viz.BLUE, linewidth=2, marker="o", markersize=3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 40)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([0, 6, 10, 15, 20, 26, 30, 35, 40])
    ax.set_xticklabels(["toss", "6", "10", "15", "break", "6", "10", "15", "20"])
    ax.set_xlabel("1st innings overs  |  chase overs")
    ax.set_ylabel(f"P({bat_first} win)")
    ax.set_title(f"Live win probability, {match['date'].iloc[0]:%Y-%m-%d}: "
                 f"{bat_first} (batting first) v {chaser}")
    winner = bat_first if match[TARGET].iloc[0] == 1 else chaser
    ax.text(0.99, 0.02, f"Winner: {winner}. Models trained on 2008–2021 only.",
            transform=ax.transAxes, ha="right", va="bottom", color=viz.INK_SECONDARY, fontsize=8)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, metadata={"Software": None})
    plt.close(fig)


def build_report(states: pd.DataFrame, by_innings: dict[int, dict], chosen: dict[int, str],
                 prematch: tuple[ev.Metrics, ev.Metrics, str]) -> str:
    pre = ev.pre_test(states)
    n_matches = pre["match_id"].nunique()

    pm_wf, pm_val, pm_name = prematch
    summary = [["Pre-match, after the toss", pm_name, pm_wf.n, f"{pm_wf.accuracy:.1%} ± {pm_wf.accuracy_ci:.1%}",
                f"{pm_wf.roc_auc:.3f}", f"{pm_wf.log_loss:.4f}", f"{pm_val.accuracy:.1%} ({pm_val.n})"]]
    for (innings, overs), label in CHECKPOINTS.items():
        r = by_innings[innings][chosen[innings]]
        wf, val = at_checkpoint(r["walk_forward"], innings, overs), at_checkpoint(r["validation"], innings, overs)
        summary.append([label, r["name"], wf.n, f"{wf.accuracy:.1%} ± {wf.accuracy_ci:.1%}",
                        f"{wf.roc_auc:.3f}", f"{wf.log_loss:.4f}", f"{val.accuracy:.1%} ({val.n})"])

    sections = []
    for innings, title in ((1, "First innings"), (2, "Chase (innings break onwards)")):
        results = by_innings[innings]
        pooled = [ev.metrics_row(r["name"], r["wf_metrics"]) for r in results.values()]
        val = [ev.metrics_row(r["name"], r["val_metrics"]) for r in results.values()]
        cp_rows = []
        for (inn, overs), label in CHECKPOINTS.items():
            if inn == innings:
                cp_rows.append([label] + [f"{at_checkpoint(r['walk_forward'], inn, overs).accuracy:.1%}"
                                          for r in results.values()])
        chosen_wf = results[chosen[innings]]["walk_forward"]
        sections.append(f"""
## {title}: model comparison

Features: {", ".join(f"`{c}`" for c in lf.FEATURES_BY_INNINGS[innings])}.

Walk-forward 2015–2022, every in-progress over-end state (n counts states, about 20 per match):

{md_table(ev.METRIC_HEADERS, pooled)}

Validation 2022, every in-progress state:

{md_table(ev.METRIC_HEADERS, val)}

Walk-forward accuracy at each checkpoint:

{md_table(["Checkpoint"] + [r["name"] for r in results.values()], cp_rows)}

Selected: **{results[chosen[innings]]['name']}**. Calibration over its walk-forward states (predicted vs observed batting-first win rate):

{md_table(["Predicted range", "States", "Mean predicted", "Observed"],
          ev.calibration_table(chosen_wf[TARGET], chosen_wf["prob"]))}
""")

    return tidy_markdown(f"""
# Live (in-match) models (Phase 4)

Generated by `python -m src.models.live` (live features version {lf.LIVE_VERSION}). A live prediction is made at the end of an over or at the innings break and gives P(batting-first side wins). The 2023–2024 test seasons were not used.

## Setup

- Sample: {n_matches:,} full-length decided matches from 2008–2022 ({len(pre):,} in-progress states). Matches with D/L or fewer than 20 overs a side are left out.
- One model per innings. Same views as the pre-match models: walk-forward 2015–2022 and validation 2022 (trained on 2008–2021).
- **Selection rule (fixed in advance):** lowest walk-forward log loss over all in-progress states of that innings.
- Checkpoint rows count only matches still in progress at that point; a chase that has already ended is not scored at later checkpoints.
- Pre-match context in every state (Elo, venue chase rate) comes from the leakage-safe pre-match features. The par score is the smoothed mean first-innings total of the previous {lf.PAR_WINDOW} full-length matches on earlier dates.

## Accuracy by prediction moment (selected models)

Walk-forward 2015–2022, with the 2022 validation accuracy alongside. The pre-match row is the selected pre-match model (see `reports/prematch_models.md`), re-scored on this sample.

{md_table(["Moment", "Model", "Matches", "Accuracy (95% CI)", "ROC-AUC", "Log loss", "2022 accuracy (n)"], summary)}

{''.join(sections)}

## Example

![Live win probability for the 2022 final](figures/{FIGURE_PATH.name})

The chart replays the 2022 final with the selected models trained on 2008–2021 only. The first point is the pre-match Elo probability.
""")


def main() -> int:
    states = load_states()
    by_innings = {i: evaluate_innings(states, i) for i in (1, 2)}
    chosen = {i: select(by_innings[i]) for i in (1, 2)}
    prematch = prematch_on_live_sample(states)
    win_probability_curve(states, chosen)
    REPORT_PATH.write_text(build_report(states, by_innings, chosen, prematch), encoding="utf-8", newline="\n")

    for i in (1, 2):
        print(f"Innings {i}:")
        for key, r in by_innings[i].items():
            w = r["wf_metrics"]
            mark = "*" if key == chosen[i] else " "
            print(f"  {mark} {r['name']:22s} acc {w.accuracy:6.1%}  auc {w.roc_auc:.3f}  "
                  f"logloss {w.log_loss:.4f}  {r['seconds']:5.1f}s")
    for (innings, overs), label in CHECKPOINTS.items():
        m = at_checkpoint(by_innings[innings][chosen[innings]]["walk_forward"], innings, overs)
        print(f"  {label:32s} {m.accuracy:6.1%}  (n={m.n})")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} and {FIGURE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
