import os, json, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Dict, Any, List
from api.schemas import PredictPayload

FEATURE_INFO = {
    "period":  {"unit": "days",  "hint": "Orbital period"},
    "duration":{"unit": "hours", "hint": "Transit duration"},
    "depth":   {"unit": "ppm",   "hint": "Transit depth"},
    "impact":  {"unit": "",      "hint": "Impact parameter (0–1)"},
    "prad":    {"unit": "R⊕",    "hint": "Planet radius"},
    "insol":   {"unit": "F⊕",    "hint": "Insolation vs Earth"},
    "teq":     {"unit": "K",     "hint": "Equilibrium temperature"},
    "steff":   {"unit": "K",     "hint": "Stellar effective temp"},
    "slogg":   {"unit": "cgs",   "hint": "Stellar surface gravity (log g)"},
    "srad":    {"unit": "R☉",    "hint": "Stellar radius"},
    "smass":   {"unit": "M☉",    "hint": "Stellar mass"},
    "smet":    {"unit": "dex",   "hint": "Stellar metallicity"},
    "star_mag":{"unit": "mag",   "hint": "Kepler/TESS magnitude"},
    "snr":     {"unit": "",      "hint": "Transit SNR"},
    "ntrans":  {"unit": "",      "hint": "Number of observed transits"}
}

ART_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "model.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_list.json")
METRICS_PATH = os.path.join(ART_DIR, "metrics.json")
SAMPLE_PATH = os.path.join(ART_DIR, "sample_inputs.csv")

app = FastAPI(title="Starling API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

model = joblib.load(MODEL_PATH)
FEATURES = json.load(open(FEATURES_PATH))
METRICS = json.load(open(METRICS_PATH))
LABELS = METRICS["labels"]

# Optional SHAP explainer
explainer = None
try:
    import shap
    explainer = shap.TreeExplainer(model)
except Exception:
    explainer = None

def to_vector(d: Dict[str, Any]) -> np.ndarray:
    vals = []
    for f in FEATURES:
        v = d.get(f, None)
        vals.append(np.nan if v is None or v == "" else float(v))
    return np.array(vals, dtype=float).reshape(1, -1)

@app.get("/metadata")
def metadata():
    slim = {
        "macro_f1": METRICS.get("macro_f1"),
        "roc_auc_ovr": METRICS.get("roc_auc_ovr"),
        "n_train": METRICS.get("n_train"),
        "n_test": METRICS.get("n_test"),
        "recommended_threshold": METRICS.get("recommended_threshold")
    }
    return {"features": FEATURES, "labels": LABELS, "metrics": slim, "version": "0.2.0"}

@app.get("/metrics_full")
def metrics_full():
    # Full metrics for charts (ROC/PR/calibration, confusion matrix, importances)
    return JSONResponse(content=jsonable_encoder(METRICS))

@app.get("/echo-sample")
def echo_sample():
    if not os.path.exists(SAMPLE_PATH):
        raise HTTPException(404, "sample_inputs.csv not found")
    df = pd.read_csv(SAMPLE_PATH)
    row = df.iloc[0]
    features = {f: (None if pd.isna(row.get(f)) else float(row.get(f))) for f in FEATURES}
    payload = {"features": features}
    if "true_label" in df.columns and not pd.isna(row["true_label"]):
        payload["true_label"] = str(row["true_label"])
    return JSONResponse(content=jsonable_encoder(payload))

@app.post("/predict")
def predict(payload: PredictPayload):
    x = to_vector(payload.features)
    try:
        proba_vec = model.predict_proba(pd.DataFrame(x, columns=FEATURES))[0]
    except Exception as e:
        raise HTTPException(400, str(e))

    # Top class + confidence stats
    order = np.argsort(-proba_vec)
    top_idx, second_idx = int(order[0]), int(order[1])
    top_lab, second_lab = LABELS[top_idx], LABELS[second_idx]
    top_p, second_p = float(proba_vec[top_idx]), float(proba_vec[second_idx])
    margin = float(top_p - second_p)

    # Use recommended threshold from training (fallback 0.5)
    th = float(METRICS.get("recommended_threshold", 0.5))
    accepted = bool(top_p >= th and margin >= 0.05)  # margin rule is cheap & effective

    # SHAP per-sample (best-effort)
    top_contrib = []
    try:
        if explainer is not None:
            sv = explainer.shap_values(pd.DataFrame(x, columns=FEATURES))
            contrib = sv[top_idx][0] if isinstance(sv, list) else sv[0]
            pairs = sorted(zip(FEATURES, contrib), key=lambda t: abs(t[1]), reverse=True)[:5]
            top_contrib = [{"feature": f, "shap": float(v), "value": payload.features.get(f, None)} for f,v in pairs]
    except Exception:
        fi = METRICS.get("feature_importances_gain", {})
        pairs = sorted([(f, float(fi.get(f, 0.0)), payload.features.get(f, None)) for f in FEATURES], key=lambda t: t[1], reverse=True)[:5]
        top_contrib = [{"feature": f, "importance": imp, "value": val} for f,imp,val in pairs]

    out = {
        "label": top_lab,
        "probs": {LABELS[i]: float(proba_vec[i]) for i in range(len(LABELS))},
        "decision": {
            "accepted": accepted,
            "reason": "above_threshold_and_margin" if accepted else "low_confidence",
            "threshold": th,
            "confidence": top_p,
            "margin": margin,
            "second_best": {"label": second_lab, "prob": second_p}
        },
        "top_factors": top_contrib
    }

    # (Optional) lightweight inference log for your demo
    try:
        log_row = {
            **{f: payload.features.get(f, None) for f in FEATURES},
            "pred_label": top_lab, "top_p": top_p, "margin": margin, "accepted": accepted
        }
        log_path = os.path.join(ART_DIR, "infer_log.csv")
        pd.DataFrame([log_row]).to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))
    except Exception:
        pass

    return out


@app.post("/batch_predict")
def batch_predict(rows: List[Dict[str, Any]]):
    if not isinstance(rows, list) or len(rows) == 0:
        raise HTTPException(400, "Provide a non-empty JSON array of {features:{...}} or raw feature dicts.")
    # accept either [{features:{...}}, {...}] formats; normalize
    cleaned = []
    for r in rows:
        feats = r.get("features", r)
        arr = []
        for f in FEATURES:
            v = feats.get(f, None)
            arr.append(np.nan if v is None or v == "" else float(v))
        cleaned.append(arr)
    X = pd.DataFrame(np.array(cleaned), columns=FEATURES)
    probs = model.predict_proba(X)
    preds = probs.argmax(axis=1)
    out = []
    for i in range(len(rows)):
        out.append({
            "label": LABELS[int(preds[i])],
            "probs": {LABELS[j]: float(probs[i,j]) for j in range(len(LABELS))}
        })
    return out

@app.get("/health")
def health():
    return {"ok": True, "version": "0.2.0"}

@app.get("/feature_info")
def feature_info():
    return FEATURE_INFO