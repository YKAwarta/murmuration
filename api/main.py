import os, json, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from api.schemas import PredictPayload

ART_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ART_DIR, "model.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_list.json")
METRICS_PATH = os.path.join(ART_DIR, "metrics.json")
SAMPLE_PATH = os.path.join(ART_DIR, "sample_inputs.csv")

app = FastAPI(title="Starling API", version="0.1.0")

# CORS for local dev & Vercel later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Load artifacts once
model = joblib.load(MODEL_PATH)
FEATURES = json.load(open(FEATURES_PATH))
METRICS = json.load(open(METRICS_PATH))
LABELS = METRICS["labels"]

def to_vector(d: Dict[str, Any]) -> np.ndarray:
    vals = []
    for f in FEATURES:
        v = d.get(f, None)
        vals.append(np.nan if v is None or v == "" else float(v))
    return np.array(vals, dtype=float).reshape(1, -1)

@app.get("/metadata")
def metadata():
    return {
        "features": FEATURES,
        "labels": LABELS,
        "metrics": {k: METRICS[k] for k in ["macro_f1","roc_auc_ovr","n_train","n_test"]},
        "version": "0.1.0"
    }

@app.get("/echo-sample")
def echo_sample():
    if not os.path.exists(SAMPLE_PATH):
        raise HTTPException(404, "sample_inputs.csv not found")
    row = pd.read_csv(SAMPLE_PATH).iloc[0].to_dict()
    return row

@app.post("/predict")
def predict(payload: PredictPayload):
    x = to_vector(payload.features)
    try:
        probs = model.predict_proba(pd.DataFrame(x, columns=FEATURES))[0]
    except Exception as e:
        raise HTTPException(400, str(e))
    pred_idx = int(np.argmax(probs))
    label = LABELS[pred_idx]

    # Lightweight "explanation": top 5 features by global importance that have non-null inputs
    fi = METRICS.get("feature_importances", {})
    pairs = [(f, float(fi.get(f, 0.0)), payload.features.get(f, None)) for f in FEATURES]
    top = sorted([p for p in pairs if p[2] is not None], key=lambda t: t[1], reverse=True)[:5]

    return {
        "label": label,
        "probs": {LABELS[i]: float(p) for i,p in enumerate(probs)},
        "top_factors": [{"feature": f, "importance": imp, "value": val} for f,imp,val in top]
    }
