import argparse, json, os, joblib, numpy as np, pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_auc_score, roc_curve, confusion_matrix
from lightgbm import LGBMClassifier
from lightgbm.callback import log_evaluation
from src.data.harmonize import load_and_merge, FEATURES, LABELS

def ece_multiclass(y_true, y_prob, n_bins=10):
    # top-class calibration: place max prob into bins and see accuracy per bin
    top_prob = y_prob.max(axis=1)
    top_pred = y_prob.argmax(axis=1)
    correct = (top_pred == y_true).astype(int)
    bins = np.linspace(0,1,n_bins+1)
    mids, accs, confs, counts = [], [], [], []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (top_prob >= lo) & (top_prob < hi if i < n_bins-1 else top_prob <= hi)
        if mask.sum()==0: continue
        mids.append((lo+hi)/2)
        conf = float(top_prob[mask].mean())
        acc = float(correct[mask].mean())
        confs.append(conf); accs.append(acc); counts.append(int(mask.sum()))
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece), {"bin_mid": mids, "acc": accs, "conf": confs, "count": counts}

def curves_ovr(y_true, y_prob, labels):
    # one-vs-rest ROC & PR points for plotting
    roc = {}; pr = {}; auc = {}
    y_true_oh = np.eye(len(labels))[y_true]
    for k, lab in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_oh[:,k], y_prob[:,k])
        prec, rec, _ = precision_recall_curve(y_true_oh[:,k], y_prob[:,k])
        try:
            auc_k = roc_auc_score(y_true_oh[:,k], y_prob[:,k])
        except Exception:
            auc_k = float("nan")
        roc[lab] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pr[lab]  = {"precision": prec.tolist(), "recall": rec.tolist(),
                    "ap": float(average_precision_score(y_true_oh[:,k], y_prob[:,k]))}
        auc[lab] = float(auc_k)
    return roc, pr, auc

def threshold_for_precision(y_true, y_prob, target_prec=0.90):
    # choose a global abstain threshold on top-class prob to achieve ~target precision
    order = np.argsort(-y_prob.max(axis=1))
    probs = y_prob.max(axis=1)[order]
    preds = y_prob.argmax(axis=1)[order]
    correct = (preds == y_true[order]).astype(int)
    cum_tp = np.cumsum(correct)
    cum_pp = np.arange(1, len(correct)+1)
    precision = cum_tp / np.maximum(cum_pp,1)
    idx = np.where(precision >= target_prec)[0]
    if len(idx)==0:  # fallback: 0.5
        return 0.5
    return float(probs[idx[-1]])

def train_eval_once(X, y, params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for tr, va in skf.split(X, y):
        clf = LGBMClassifier(**params)
        clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="multi_logloss", callbacks=[log_evaluation(period=0)])
        pred = clf.predict(X.iloc[va])
        f1s.append(f1_score(y[va], pred, average="macro"))
    return float(np.mean(f1s)), float(np.std(f1s))

def main(args):
    df, F, L = load_and_merge(args.koi, args.toi)
    y_map = {l:i for i,l in enumerate(L)}
    y = df["label"].map(y_map).values
    X = df[F].copy()

    # ----- tiny hyperparam sweep (fast, robust) -----
    grid = {
        "n_estimators": [300, 600],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    }
    base = dict(
        objective="multiclass",
        num_class=len(L),
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    best = None
    for n_est, lr, nl in product(grid["n_estimators"], grid["learning_rate"], grid["num_leaves"]):
        params = base | {"n_estimators": n_est, "learning_rate": lr, "num_leaves": nl}
        mean_f1, std_f1 = train_eval_once(X, y, params)
        if (best is None) or (mean_f1 > best["mean_f1"]):
            best = {"params": params, "mean_f1": mean_f1, "std_f1": std_f1}
    best_params = best["params"]

    # ----- fit final model / hold-out for curves & ECE (fast split) -----
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = LGBMClassifier(**best_params)
    clf.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="multi_logloss", callbacks=[log_evaluation(period=0)])


    prob = clf.predict_proba(Xte)  # (n, K)
    pred = prob.argmax(axis=1)

    # metrics
    macro_f1 = f1_score(yte, pred, average="macro")
    try:
        auc_ovr = roc_auc_score(yte, prob, multi_class="ovr")
    except Exception:
        auc_ovr = float("nan")
    cm = confusion_matrix(yte, pred).tolist()
    
    cm_np = np.array(cm)
    pairs = []
    for i, li in enumerate(L):
        for j, lj in enumerate(L):
            if i != j:
                pairs.append((li, lj, int(cm_np[i, j])))
    pairs_sorted = sorted(pairs, key=lambda t: t[2], reverse=True)[:5]

    # curves + calibration
    roc_pts, pr_pts, auc_per_class = curves_ovr(yte, prob, L)
    ece, cal_bins = ece_multiclass(yte, prob, n_bins=10)
    rec_thresh = threshold_for_precision(yte, prob, target_prec=0.90)

    top_prob = prob.max(axis=1)
    top_pred = prob.argmax(axis=1)
    accept = top_prob >= rec_thresh
    
    coverage = float(np.mean(accept))                      # fraction accepted
    accepted_correct = float(np.mean((top_pred[accept] == yte[accept])) if accept.any() else np.nan)
    overall_precision_at_thresh = accepted_correct         # same as precision on accepted set
    
    # per-class acceptance rates
    per_class_accept = {
        L[k]: float(np.mean(accept[yte == k])) if np.any(yte == k) else 0.0
        for k in range(len(L))
    }

    # global importances (gain) + SHAP global (mean |shap|)
    importances = dict(zip(F, clf.feature_importances_.astype(float).tolist()))

    # sample for SHAP summary
    try:
        import shap
        bg_idx = np.random.RandomState(42).choice(len(Xtr), size=min(200, len(Xtr)), replace=False)
        expl = shap.TreeExplainer(clf)
        shap_vals = expl.shap_values(Xtr.iloc[bg_idx])
        # shap_values is list per class; take mean abs over classes then over rows
        if isinstance(shap_vals, list):
            abs_means = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
        else:
            abs_means = np.mean(np.abs(shap_vals), axis=0)
        shap_global = dict(zip(F, abs_means.astype(float).tolist()))
    except Exception:
        shap_global = {}

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.outdir, "model.pkl"))
    with open(os.path.join(args.outdir, "feature_list.json"), "w") as f:
        json.dump(F, f)

    metrics = {
    "labels": L,
    "n_total": int(len(X)),
    "n_train": int(len(Xtr)),
    "n_test": int(len(Xte)),
    "search": {"best_params": best_params, "cv_macro_f1_mean": best["mean_f1"], "cv_macro_f1_std": best["std_f1"]},
    "macro_f1": float(macro_f1),
    "roc_auc_ovr": float(auc_ovr),
    "confusion_matrix": cm,
    "top_confusions": [{"true": a, "pred": b, "count": n} for a, b, n in pairs_sorted],
    "roc": roc_pts,
    "pr": pr_pts,
    "auc_per_class": auc_per_class,
    "ece": float(ece),
    "calibration_bins": cal_bins,
    "recommended_threshold": rec_thresh,
    "decision": {"recommended_threshold": rec_thresh, "coverage_at_threshold": coverage, "precision_on_accepted": overall_precision_at_thresh, "per_class_accept_rate": per_class_accept},
    "feature_importances_gain": importances,
    "feature_importances_shap": shap_global
    }

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # sample inputs for UI/API dev (JSON-safe: drop NaNs)
    sample = Xte.head(5).copy()
    sample["true_label"] = [L[i] for i in yte[:5]]
    sample.to_csv(os.path.join(args.outdir, "sample_inputs.csv"), index=False)

    print("Saved artifacts to", args.outdir)
    print("CV macro-F1 (mean±std):", round(best["mean_f1"],3), "±", round(best["std_f1"],3))
    print("Holdout macro-F1:", round(macro_f1,3), "OvR AUC:", round(auc_ovr,3), "ECE:", round(ece,3))
    print("Recommended decision threshold:", round(rec_thresh,2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--koi", required=True)
    p.add_argument("--toi", required=True)
    p.add_argument("--outdir", default="artifacts")
    main(p.parse_args())
