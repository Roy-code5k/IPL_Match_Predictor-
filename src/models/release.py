"""Phase 5: the one evaluation on 2023-2024, then the versioned release set.

1. **Test evaluation.** The model selection and confidence labels were frozen first
   (src/models/confidence.py, committed before this ran). Every candidate is trained on
   2008-2022 and evaluated on the 2023-2024 test seasons. The selected model's result is
   the headline; the others are shown for transparency, not to re-select.
   Writes reports/model_evaluation.md and reports/figures/09_accuracy_by_moment.png.
2. **Release.** The selected models are refitted with the same hyperparameters on all
   seasons (2008-2024), so predictions for future seasons use every known result. They
   are saved with the history snapshot they need in models/release-<version>/.
   metadata.json records both training ranges, the test metrics, file hashes, and package
   versions. Only metadata.json is committed; the other files are rebuilt by this script.

Run from the project root (needs data/raw/*.csv and data/processed/*.csv):
    python -m src.models.release
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import viz
from src.data.inspect_raw import md_table, sha256, tidy_markdown
from src.features import builder as fb
from src.features import live as lf
from src.models import confidence as cf
from src.models import evaluation as ev
from src.models import live as lm
from src.models import prematch as pm

ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION = "1.0.0"
MODELS_DIR = ROOT / "models"
RELEASE_DIR = MODELS_DIR / f"release-{RELEASE_VERSION}"
REPORT_PATH = ROOT / "reports" / "model_evaluation.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "09_accuracy_by_moment.png"

HISTORY_COLUMNS = ["match_id", "date", "season", "team1", "team2", "venue",
                   "toss_winner", "toss_decision", "winner", "is_playoff"]
PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]
PREMATCH_LABEL = "Pre-match, after the toss"


# --- test evaluation ---------------------------------------------------------------

def _scored(pred: pd.DataFrame, target: str) -> dict:
    return {"pred": pred, "metrics": ev.score(pred[target], pred["prob"])}


def prematch_test(features: pd.DataFrame) -> dict[str, dict]:
    return {key: {"name": name, **_scored(ev.final_test(features, pm.fit_predict_for(make)), pm.TARGET)}
            for key, (name, make) in pm.CANDIDATES.items()}


def live_test(states: pd.DataFrame) -> dict[int, dict[str, dict]]:
    out = {}
    for innings in (1, 2):
        data = states[states["innings"] == innings]
        out[innings] = {key: {"name": name, **_scored(ev.final_test(data, lm.fit_predict_for(make, innings)),
                                                      lf.TARGET)}
                        for key, (name, make) in lm.CANDIDATES.items()}
    return out


def walk_forward_selected(features: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Phase 4 walk-forward predictions of the selected models, for side-by-side comparison."""
    pre = ev.walk_forward(features, pm.fit_predict_for(pm.CANDIDATES[cf.SELECTED_PREMATCH][1]))
    live = {i: ev.walk_forward(states[states["innings"] == i],
                               lm.fit_predict_for(lm.CANDIDATES[cf.SELECTED_LIVE[i]][1], i)) for i in (1, 2)}
    return pre, live


def prematch_as_bat_first(pred: pd.DataFrame, states: pd.DataFrame) -> ev.Metrics:
    """Re-score pre-match predictions on the live sample, from the batting-first side's view."""
    sample = states.loc[(states["innings"] == 2) & (states["overs_done"] == 0), ["match_id", lf.TARGET]]
    joined = pred.merge(sample, on="match_id")
    prob_bf = np.where(joined["a_bats_first"] == 1, joined["prob"], 1 - joined["prob"])
    return ev.score(joined[lf.TARGET], prob_bf)


def moments(pre_pred: pd.DataFrame, live_pred: dict[int, pd.DataFrame],
            states: pd.DataFrame) -> list[tuple[str, ev.Metrics]]:
    out = [(PREMATCH_LABEL, prematch_as_bat_first(pre_pred, states))]
    for (innings, overs), label in lm.CHECKPOINTS.items():
        out.append((label, lm.at_checkpoint(live_pred[innings], innings, overs)))
    return out


def accuracy_figure(wf: list[tuple[str, ev.Metrics]], test: list[tuple[str, ev.Metrics]]) -> None:
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = [label for label, _ in wf]
    y = np.arange(len(labels))[::-1]
    for offset, series, color, name in ((0.15, wf, viz.BLUE, "Walk-forward 2015–2022"),
                                        (-0.15, test, viz.ORANGE, "Test 2023–2024")):
        acc = [m.accuracy for _, m in series]
        ci = [m.accuracy_ci for _, m in series]
        ax.errorbar(acc, y + offset, xerr=ci, fmt="o", color=color, markersize=6,
                    elinewidth=1.5, capsize=0, label=name)
    ax.axvline(0.5, color=viz.BASELINE, linewidth=1)
    ax.text(0.505, y[-1] - 0.55, "coin flip", color=viz.INK_MUTED, fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(y[-1] - 0.8, y[0] + 0.6)
    ax.set_xlim(0.35, 1.0)
    ticks = np.arange(0.4, 1.01, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy (dot) with 95% interval (line)")
    ax.set_title("Accuracy by prediction moment", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              fontsize=9, handletextpad=0.3, borderaxespad=0.2)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, metadata={"Software": None})
    plt.close(fig)


def _test_rows(results: dict[str, dict], selected: str) -> list[list[object]]:
    return [ev.metrics_row(r["name"] + (" (selected)" if k == selected else ""), r["metrics"])
            for k, r in results.items()]


def build_report(pre: dict, live: dict, wf_moments, test_moments) -> str:
    summary = [[label, f"{w.accuracy:.1%} ({w.n})", f"{t.accuracy:.1%} ± {t.accuracy_ci:.1%}", t.n,
                f"{t.roc_auc:.3f}", f"{t.log_loss:.4f}"]
               for (label, w), (_, t) in zip(wf_moments, test_moments)]

    sel = pre[cf.SELECTED_PREMATCH]
    sel_pred = sel["pred"]
    season_rows = []
    for season in ev.TEST_SEASONS:
        row: list[object] = [season]
        for key in ("coin_flip", "elo", cf.SELECTED_PREMATCH):
            part = pre[key]["pred"][pre[key]["pred"]["season"] == season]
            row.append(f"{ev.score(part[pm.TARGET], part['prob']).accuracy:.1%}")
        season_rows.append(row + [int((sel_pred["season"] == season).sum())])

    team_rows = []
    correct = (sel_pred["prob"] >= 0.5).astype(int) == sel_pred[pm.TARGET]
    for team in sorted(set(sel_pred["team1"]) | set(sel_pred["team2"])):
        involved = (sel_pred["team1"] == team) | (sel_pred["team2"] == team)
        team_rows.append([team, int(involved.sum()), f"{correct[involved].mean():.1%}"])

    (tn, fp_), (fn, tp) = sel["metrics"].confusion

    live_sections = []
    for innings, title in ((1, "First innings"), (2, "Chase")):
        res = live[innings]
        chosen = res[cf.SELECTED_LIVE[innings]]["pred"]
        cp_rows = []
        for (inn, overs), label in lm.CHECKPOINTS.items():
            if inn != innings:
                continue
            row: list[object] = [label]
            for season in ev.TEST_SEASONS:
                m = lm.at_checkpoint(chosen[chosen["season"] == season], inn, overs)
                row.append(f"{m.accuracy:.1%} ({m.n})")
            cp_rows.append(row)
        live_sections.append(f"""
### {title}

All in-progress over-end states in 2023–2024 (n counts states):

{md_table(ev.METRIC_HEADERS, _test_rows(res, cf.SELECTED_LIVE[innings]))}

Selected model by season and checkpoint (accuracy, matches in progress):

{md_table(["Checkpoint"] + [str(s) for s in ev.TEST_SEASONS], cp_rows)}

Confidence labels on the test states:

{md_table(cf.BAND_HEADERS, cf.band_table(chosen[lf.TARGET], chosen["prob"]))}
""")

    return tidy_markdown(f"""
# Model evaluation on the test seasons (Phase 5)

Generated by `python -m src.models.release`. Every model was trained on 2008–2022 and evaluated once on the held-out 2023–2024 seasons. The selected models and the confidence labels were fixed before this evaluation (`src/models/confidence.py`, `reports/calibration_review.md`); the other candidates are shown for transparency only.

The released models (release {RELEASE_VERSION}) are then refitted on 2008–2024 with the same settings, so they can use the two most recent seasons when predicting future matches. The metrics below describe the models trained on 2008–2022.

## Accuracy by prediction moment (selected models)

Live rows count only matches still in progress at that point; the pre-match row is scored on the same full-length matches from the batting-first side's view. The walk-forward column repeats the Phase 4 result for comparison.

{md_table(["Moment", "Walk-forward 2015–2022 (n)", "Test accuracy (95% CI)", "Test n", "Test ROC-AUC", "Test log loss"], summary)}

![Accuracy by prediction moment](figures/{FIGURE_PATH.name})

## Pre-match models on 2023–2024

{md_table(ev.METRIC_HEADERS, _test_rows(pre, cf.SELECTED_PREMATCH))}

By season:

{md_table(["Season", "Coin flip", "Elo baseline", sel["name"] + " (selected)", "n"], season_rows)}

Selected model by team (matches involving the team):

{md_table(["Team", "Matches", "Accuracy"], team_rows)}

Confusion matrix of the selected model (rows = actual, columns = predicted):

{md_table(["", "Predicted Team B wins", "Predicted Team A wins"],
          [["Actual Team B wins", tn, fp_], ["Actual Team A wins", fn, tp]])}

Confidence labels on the test matches:

{md_table(cf.BAND_HEADERS, cf.band_table(sel_pred[pm.TARGET], sel_pred["prob"]))}

## Live models on 2023–2024

{''.join(live_sections)}
""")


# --- release ----------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    text = out.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    path.write_text(text, encoding="utf-8", newline="\n")
    return _hash_text(text)


def _metrics_dict(m: ev.Metrics) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dataclasses.asdict(m).items()}


def build_release(matches: pd.DataFrame, features: pd.DataFrame, states: pd.DataFrame,
                  deliveries: pd.DataFrame, pre: dict, live: dict, test_moments) -> dict:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    files = {}

    pre_model = pm.CANDIDATES[cf.SELECTED_PREMATCH][1]().fit(features[pm.FEATURES],
                                                                features[pm.TARGET].to_numpy())
    live_models = {}
    for innings in (1, 2):
        data = states[states["innings"] == innings]
        cols = lf.FEATURES_BY_INNINGS[innings]
        live_models[innings] = lm.CANDIDATES[cf.SELECTED_LIVE[innings]][1]().fit(
            data[cols], data[lf.TARGET].to_numpy())

    for name, model in [("prematch.joblib", pre_model), ("live_innings1.joblib", live_models[1]),
                        ("live_innings2.joblib", live_models[2])]:
        joblib.dump(model, RELEASE_DIR / name)
        files[name] = sha256(RELEASE_DIR / name)
    files["history.csv"] = _write_csv(matches[HISTORY_COLUMNS], RELEASE_DIR / "history.csv")
    totals = lf.totals_table(matches, deliveries, lf.full_length_ids())
    files["first_innings_totals.csv"] = _write_csv(totals, RELEASE_DIR / "first_innings_totals.csv")

    inner = pre_model.model if isinstance(pre_model, pm.Symmetric) else pre_model
    metadata = {
        "release": RELEASE_VERSION,
        "built_by": "python -m src.models.release",
        "history_cutoff": f"{matches['date'].max():%Y-%m-%d}",
        "history_rule": "features use only matches on dates strictly before the prediction cutoff",
        "training_seasons": [int(matches["season"].min()), int(matches["season"].max())],
        "evaluation": {
            "trained_on_seasons": [int(matches["season"].min()), max(ev.TRAIN_SEASONS) + 1],
            "test_seasons": list(ev.TEST_SEASONS),
            "note": "Test metrics come from the same configuration trained on 2008-2022; "
                    "the released models were refitted on all seasons afterwards.",
        },
        "prematch": {
            "model": pm.CANDIDATES[cf.SELECTED_PREMATCH][0],
            "file": "prematch.joblib",
            "estimator": repr(inner),
            "symmetric": isinstance(pre_model, pm.Symmetric),
            "output": "probability that team1 wins",
            "feature_version": fb.FEATURE_VERSION,
            "features": pm.FEATURES,
            "training_rows": len(features),
            "test_metrics": _metrics_dict(pre[cf.SELECTED_PREMATCH]["metrics"]),
        },
        "live": {
            "feature_version": lf.LIVE_VERSION,
            "output": "probability that the batting-first side wins",
            "par_score": {"window": lf.PAR_WINDOW, "prior": lf.PAR_PRIOR, "strength": lf.PAR_PRIOR_STRENGTH},
            "excluded": "D/L matches and matches with fewer than 20 overs a side",
            **{f"innings{i}": {"model": lm.CANDIDATES[cf.SELECTED_LIVE[i]][0],
                               "file": f"live_innings{i}.joblib",
                               "estimator": repr(live_models[i]),
                               "features": lf.FEATURES_BY_INNINGS[i],
                               "training_rows": int((states["innings"] == i).sum())} for i in (1, 2)},
            "test_accuracy_by_moment": {label: _metrics_dict(m) for label, m in test_moments
                                        if label != PREMATCH_LABEL},
        },
        "comparison": {
            "note": "All candidates on the 2023-2024 test seasons, trained on 2008-2022. "
                    "Selection was fixed beforehand on 2015-2022 data.",
            "prematch": [{"model": r["name"], "selected": k == cf.SELECTED_PREMATCH,
                          "metrics": _metrics_dict(r["metrics"])} for k, r in pre.items()],
            **{f"live_innings{i}": [{"model": r["name"], "selected": k == cf.SELECTED_LIVE[i],
                                     "metrics": _metrics_dict(r["metrics"])} for k, r in live[i].items()]
               for i in (1, 2)},
        },
        "confidence_bands": [{"from": lowest, "label": name} for lowest, name in cf.BANDS],
        "close_call_label": cf.CLOSE,
        "inputs": {"data/processed/matches_clean.csv": sha256(fb.CLEAN_PATH),
                   "data/processed/features.csv": sha256(fb.FEATURES_PATH),
                   "data/raw/deliveries.csv": sha256(lf.RAW_DELIVERIES)},
        "packages": {p: version(p) for p in PACKAGES},
        "files": files,
    }
    (RELEASE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                encoding="utf-8", newline="\n")
    return metadata


def main() -> int:
    matches = fb.load_clean()
    features = pm.load_features()
    deliveries = lf.load_deliveries()
    states = lf.build_live_states(matches, features, deliveries, lf.full_length_ids())

    pre = prematch_test(features)
    live = live_test(states)
    wf_pre, wf_live = walk_forward_selected(features, states)
    test_live = {i: live[i][cf.SELECTED_LIVE[i]]["pred"] for i in (1, 2)}
    wf_moments = moments(wf_pre, wf_live, states)
    test_moments = moments(pre[cf.SELECTED_PREMATCH]["pred"], test_live, states)

    accuracy_figure(wf_moments, test_moments)
    REPORT_PATH.write_text(build_report(pre, live, wf_moments, test_moments), encoding="utf-8", newline="\n")
    build_release(matches, features, states, deliveries, pre, live, test_moments)

    for label, m in test_moments:
        print(f"{label:32s} test {m.accuracy:6.1%} ± {m.accuracy_ci:4.1%}  auc {m.roc_auc:.3f}  (n={m.n})")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}, {FIGURE_PATH.relative_to(ROOT)}, "
          f"{RELEASE_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
