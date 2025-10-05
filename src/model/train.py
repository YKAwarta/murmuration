import argparse, json, os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
from src.data.harmonize import load_and_merge, FEATURES, LABELS

def main(args):
    df, FEATURES_, LABELS_ = load_and_merge(args.koi, args.toi)

    y_map = {l:i for i,l in enumerate(LABELS_)}
    y = df["label"].map(y_map).values
    X = df[FEATURES_].copy()

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LGBM handles NaN; keep it simple and fast
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(Xtrain, ytrain)

    probs = clf.predict_proba(Xtest)
    preds = probs.argmax(axis=1)

    metrics = {}
    metrics["labels"] = LABELS_
    metrics["macro_f1"] = f1 = f1_score(ytest, preds, average="macro")
    try:
        metrics["roc_auc_ovr"] = roc_auc_score(ytest, probs, multi_class="ovr")
    except Exception:
        metrics["roc_auc_ovr"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(ytest, preds).tolist()
    metrics["class_report"] = classification_report(ytest, preds, target_names=LABELS_, output_dict=True)
    metrics["n_train"] = int(len(Xtrain))
    metrics["n_test"] = int(len(Xtest))
    metrics["feature_importances"] = dict(zip(FEATURES_, clf.feature_importances_.astype(float).tolist()))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.outdir, "model.pkl"))
    with open(os.path.join(args.outdir, "feature_list.json"), "w") as f:
        json.dump(FEATURES_, f)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # sample rows for quick API tests
    sample = Xtest.head(5).copy()
    sample["true_label"] = [LABELS_[i] for i in ytest[:5]]
    sample.to_csv(os.path.join(args.outdir, "sample_inputs.csv"), index=False)

    print("Saved artifacts to", args.outdir)
    print("macro_f1:", round(f1, 3), "AUC(ovr):", metrics["roc_auc_ovr"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--koi", required=True)
    p.add_argument("--toi", required=True)
    p.add_argument("--outdir", default="artifacts")
    main(p.parse_args())
