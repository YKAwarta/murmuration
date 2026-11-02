"""Train the LightGBM classifier, calibrate it on a held-out split and evaluate on a test split.

    python -m src.model.train --koi data/raw/koi_min.csv --toi data/raw/toi_min.csv

Splits (stratified by class and mission, seed 42):
    train        60%  hyperparameter search (5-fold CV) and the final model fit
    calibration  20%  temperature fit, calibration-method comparison, decision threshold
    test         20%  every number reported in metrics.json; never used for fitting
"""
import argparse
import json
import os
import platform
from itertools import product

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.harmonize import load_and_merge
from src.model.calibration import (apply_temperature, calibration_report, decision_stats,
                                   fit_temperature, threshold_for_precision)
from src.model.predictor import CALIBRATION_FILE, FEATURES_FILE, METRICS_FILE, MODEL_FILE

SEED = 42
TARGET_PRECISION = 0.90
N_BINS = 10

GRID = {
    "n_estimators": [300, 600],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
}
BASE_PARAMS = dict(
    objective="multiclass",
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    class_weight="balanced",
    n_jobs=-1,
    deterministic=True,
    force_col_wise=True,
    random_state=SEED,
    verbose=-1,
)


def split_indices(df: pd.DataFrame, seed: int = SEED):
    """60/20/20 train/calibration/test split, stratified on class x mission."""
    strat = df["label"] + "|" + df["mission"]
    idx = np.arange(len(df))
    train_idx, rest = train_test_split(idx, test_size=0.4, stratify=strat, random_state=seed)
    cal_idx, test_idx = train_test_split(rest, test_size=0.5, stratify=strat.iloc[rest],
                                         random_state=seed)
    return np.sort(train_idx), np.sort(cal_idx), np.sort(test_idx)


def search_params(X: pd.DataFrame, y: np.ndarray, n_classes: int):
    """Small grid search on the training split, scored by 5-fold CV macro-F1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best = None
    for n_est, lr, nl in product(GRID["n_estimators"], GRID["learning_rate"], GRID["num_leaves"]):
        params = BASE_PARAMS | {"num_class": n_classes, "n_estimators": n_est,
                                "learning_rate": lr, "num_leaves": nl}
        f1s = []
        for tr, va in skf.split(X, y):
            clf = LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
            f1s.append(f1_score(y[va], clf.predict(X.iloc[va]), average="macro"))
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
        if best is None or mean_f1 > best["cv_macro_f1_mean"]:
            best = {"params": params, "cv_macro_f1_mean": mean_f1, "cv_macro_f1_std": std_f1}
    return best


def compare_calibrators(clf, X_cal: pd.DataFrame, y_cal: np.ndarray, n_classes: int):
    """5-fold CV on the calibration split only.

    Scores the uncalibrated model and three post-hoc methods on the held-out folds, and
    returns out-of-fold temperature-scaled probabilities for choosing the decision threshold.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logits = clf.predict_proba(X_cal, raw_score=True)
    oof_temperature = np.zeros((len(y_cal), n_classes))
    scores = {m: [] for m in ["uncalibrated", "temperature", "sigmoid", "isotonic"]}
    for fit_idx, eval_idx in skf.split(X_cal, y_cal):
        y_eval = y_cal[eval_idx]
        candidates = {"uncalibrated": apply_temperature(logits[eval_idx], 1.0)}
        t = fit_temperature(logits[fit_idx], y_cal[fit_idx])
        candidates["temperature"] = apply_temperature(logits[eval_idx], t)
        oof_temperature[eval_idx] = candidates["temperature"]
        for method in ["sigmoid", "isotonic"]:
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
            cal.fit(X_cal.iloc[fit_idx], y_cal[fit_idx])
            candidates[method] = cal.predict_proba(X_cal.iloc[eval_idx])
        for name, prob in candidates.items():
            rep = calibration_report(y_eval, prob, N_BINS)
            scores[name].append([rep["log_loss"], rep["brier"], rep["ece_top_label"]])
    summary = {
        name: dict(zip(["log_loss", "brier", "ece_top_label"],
                       np.mean(np.array(vals), axis=0).astype(float).tolist()))
        for name, vals in scores.items()
    }
    return summary, oof_temperature


def downsample_curve(x: np.ndarray, y: np.ndarray, n: int = 200):
    if len(x) <= n:
        return x.tolist(), y.tolist()
    keep = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
    return x[keep].tolist(), y[keep].tolist()


def curves_ovr(y: np.ndarray, prob: np.ndarray, labels: list):
    onehot = np.eye(len(labels))[y]
    roc, pr, auc = {}, {}, {}
    for k, lab in enumerate(labels):
        fpr, tpr, _ = roc_curve(onehot[:, k], prob[:, k])
        prec, rec, _ = precision_recall_curve(onehot[:, k], prob[:, k])
        fpr_s, tpr_s = downsample_curve(fpr, tpr)
        rec_s, prec_s = downsample_curve(rec[::-1], prec[::-1])
        roc[lab] = {"fpr": fpr_s, "tpr": tpr_s}
        pr[lab] = {"recall": rec_s, "precision": prec_s,
                   "ap": float(average_precision_score(onehot[:, k], prob[:, k]))}
        auc[lab] = float(roc_auc_score(onehot[:, k], prob[:, k]))
    return roc, pr, auc


def classification_summary(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


def plot_reliability(y: np.ndarray, before: dict, after: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.6, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", linewidth=1, label="Perfect calibration")
    for rep, name, color in [(before, "Uncalibrated", "#ef4444"), (after, "Temperature-scaled", "#2563eb")]:
        b = rep["reliability_bins"]
        ax.plot(b["conf"], b["acc"], marker="o", color=color,
                label=f"{name} (ECE {rep['ece_top_label']:.3f})")
    ax.set_ylabel("Accuracy of top-class prediction")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(f"Reliability diagram, held-out test split (n = {len(y)})", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    b = after["reliability_bins"]
    axh.bar(b["bin_mid"], b["count"], width=0.09, color="#2563eb", alpha=0.7)
    axh.set_xlabel("Top-class probability (confidence)")
    axh.set_ylabel("Rows")
    axh.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    df, F, L = load_and_merge(args.koi, args.toi)
    y_all = df["label"].map({lab: i for i, lab in enumerate(L)}).to_numpy()
    X_all = df[F]
    tr, ca, te = split_indices(df)
    X_tr, X_cal, X_te = X_all.iloc[tr], X_all.iloc[ca], X_all.iloc[te]
    y_tr, y_cal, y_te = y_all[tr], y_all[ca], y_all[te]
    print(f"rows: {len(df)} (train {len(tr)}, calibration {len(ca)}, test {len(te)})")

    best = search_params(X_tr, y_tr, len(L))
    print("best params:", {k: best["params"][k] for k in GRID},
          f"CV macro-F1 {best['cv_macro_f1_mean']:.3f} ± {best['cv_macro_f1_std']:.3f}")
    clf = LGBMClassifier(**best["params"]).fit(X_tr, y_tr)

    # Calibration: compare methods with CV inside the calibration split, then fit the
    # temperature on the whole calibration split.
    comparison, oof_prob = compare_calibrators(clf, X_cal, y_cal, len(L))
    temperature = fit_temperature(clf.predict_proba(X_cal, raw_score=True), y_cal)
    threshold = threshold_for_precision(y_cal, oof_prob, TARGET_PRECISION)
    print(f"temperature T = {temperature:.4f}; threshold = {threshold:.4f}")

    # Test-split evaluation
    logits_te = clf.predict_proba(X_te, raw_score=True)
    prob_uncal = apply_temperature(logits_te, 1.0)
    prob_cal = apply_temperature(logits_te, temperature)
    rep_before = calibration_report(y_te, prob_uncal, N_BINS)
    rep_after = calibration_report(y_te, prob_cal, N_BINS)
    pred = prob_cal.argmax(axis=1)
    cm = confusion_matrix(y_te, pred, labels=list(range(len(L))))
    confusions = sorted(((L[i], L[j], int(cm[i, j])) for i in range(len(L)) for j in range(len(L)) if i != j),
                        key=lambda t: t[2], reverse=True)[:5]
    roc_pts, pr_pts, auc_per_class = curves_ovr(y_te, prob_cal, L)
    grid = np.round(np.arange(0.34, 0.995, 0.01), 2)
    threshold_curve = [decision_stats(y_te, prob_cal, t) for t in grid]

    missions = df["mission"].to_numpy()[te]
    by_mission = {}
    for m in sorted(set(missions)):
        mask = missions == m
        rep = calibration_report(y_te[mask], prob_cal[mask], N_BINS)
        by_mission[m] = classification_summary(y_te[mask], prob_cal[mask]) | {
            "brier": rep["brier"], "ece_top_label": rep["ece_top_label"]}

    # Global importance: mean |TreeSHAP| over a sample of training rows, all classes
    sample = X_tr.sample(n=min(2000, len(X_tr)), random_state=SEED)
    contrib = clf.booster_.predict(sample, pred_contrib=True).reshape(len(sample), len(L), len(F) + 1)
    mean_abs = np.abs(contrib[:, :, :len(F)]).mean(axis=(0, 1))
    gain = clf.booster_.feature_importance(importance_type="gain")

    snapshot = {}
    snapshot_path = os.path.join(os.path.dirname(args.koi), "snapshot.json")
    if os.path.exists(snapshot_path):
        with open(snapshot_path) as f:
            snap = json.load(f)
        snapshot = {"retrieved_utc": snap.get("retrieved_utc"),
                    "koi_rows": snap["tables"]["cumulative"]["rows"],
                    "toi_rows": snap["tables"]["toi"]["rows"]}

    os.makedirs(args.outdir, exist_ok=True)
    clf.booster_.save_model(os.path.join(args.outdir, MODEL_FILE))
    with open(os.path.join(args.outdir, FEATURES_FILE), "w") as f:
        json.dump(F, f)
    with open(os.path.join(args.outdir, CALIBRATION_FILE), "w") as f:
        json.dump({"method": "temperature", "temperature": temperature,
                   "fitted_on": "calibration split", "n_rows": int(len(ca))}, f, indent=2)

    metrics = {
        "labels": L,
        "features": F,
        "data": snapshot | {
            "labeled_rows": int(len(df)),
            "class_counts": {k: int(v) for k, v in df["label"].value_counts().reindex(L).items()},
            "class_counts_by_mission": {
                m: {k: int(v) for k, v in g["label"].value_counts().reindex(L, fill_value=0).items()}
                for m, g in df.groupby("mission")},
        },
        "splits": {"seed": SEED, "stratified_on": "label x mission",
                   "train": int(len(tr)), "calibration": int(len(ca)), "test": int(len(te))},
        "search": {"grid": GRID, "best_params": {k: best["params"][k] for k in GRID},
                   "cv_macro_f1_mean": best["cv_macro_f1_mean"], "cv_macro_f1_std": best["cv_macro_f1_std"]},
        "test": classification_summary(y_te, prob_cal) | {
            "roc_auc_ovr_macro": float(roc_auc_score(y_te, prob_cal, multi_class="ovr")),
            "auc_per_class": auc_per_class,
            "confusion_matrix": cm.tolist(),
            "top_confusions": [{"true": a, "pred": b, "count": n} for a, b, n in confusions],
            "roc": roc_pts,
            "pr": pr_pts,
            "by_mission": by_mission,
        },
        "calibration": {
            "method": "temperature",
            "temperature": temperature,
            "n_bins": N_BINS,
            "method_comparison_cv_on_calibration_split": comparison,
            "test_before": rep_before,
            "test_after": rep_after,
        },
        "decision": {
            "rule": "accept when the top calibrated probability >= threshold",
            "target_precision": TARGET_PRECISION,
            "threshold_chosen_on": "out-of-fold temperature-scaled probabilities on the calibration split",
            "recommended_threshold": threshold,
            "test": decision_stats(y_te, prob_cal, threshold),
            "threshold_curve_test": threshold_curve,
        },
        "feature_importance": {
            "mean_abs_shap": dict(zip(F, mean_abs.astype(float).tolist())),
            "gain": dict(zip(F, gain.astype(float).tolist())),
        },
        "environment": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                        "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__},
    }
    with open(os.path.join(args.outdir, METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=2)

    # Test rows the UI can load with "Use Sample": the first one of each mission x class
    sample_rows = X_te.copy()
    sample_rows["true_label"] = [L[i] for i in y_te]
    sample_rows["mission"] = missions
    sample_rows = sample_rows.groupby(["mission", "true_label"], sort=True).head(1)
    sample_rows.to_csv(os.path.join(args.outdir, "sample_inputs.csv"), index=False)

    plot_reliability(y_te, rep_before, rep_after, os.path.join(args.figdir, "reliability_diagram.png"))

    print("Saved artifacts to", args.outdir)
    t = metrics["test"]
    print(f"test accuracy {t['accuracy']:.3f}, macro-F1 {t['macro_f1']:.3f}, "
          f"OvR AUC {t['roc_auc_ovr_macro']:.3f}")
    print(f"test Brier {rep_before['brier']:.4f} -> {rep_after['brier']:.4f}, "
          f"ECE {rep_before['ece_top_label']:.4f} -> {rep_after['ece_top_label']:.4f}")
    d = metrics["decision"]["test"]
    print(f"threshold {threshold:.3f}: test coverage {d['coverage']:.3f}, "
          f"precision on accepted {d['precision_on_accepted']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train, calibrate and evaluate the classifier.")
    p.add_argument("--koi", default="data/raw/koi_min.csv")
    p.add_argument("--toi", default="data/raw/toi_min.csv")
    p.add_argument("--outdir", default="artifacts")
    p.add_argument("--figdir", default="docs")
    main(p.parse_args())
