import json
import logging
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictPayload
from src.model.predictor import METRICS_FILE, Predictor

VERSION = "0.3.0"

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
SAMPLE_PATH = os.path.join(ART_DIR, "sample_inputs.csv")
# Set to a CSV path to append every /predict request and its result (off by default)
INFER_LOG = os.getenv("MURMURATION_INFER_LOG")

log = logging.getLogger("murmuration")

predictor = Predictor.load(ART_DIR)
with open(os.path.join(ART_DIR, METRICS_FILE)) as f:
    METRICS = json.load(f)
FEATURES = predictor.features
LABELS = predictor.labels
DEFAULT_THRESHOLD = float(METRICS["decision"]["recommended_threshold"])
SAMPLES = pd.read_csv(SAMPLE_PATH) if os.path.exists(SAMPLE_PATH) else None

app = FastAPI(title="Murmuration API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def clean_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Validate one feature dict; missing, null or empty values become NaN (LightGBM handles them)."""
    if not isinstance(raw, dict):
        raise HTTPException(422, "features must be an object of {feature: number}")
    unknown = sorted(set(raw) - set(FEATURES))
    if unknown:
        raise HTTPException(422, f"unknown features: {unknown}; expected a subset of {FEATURES}")
    out = {}
    for f in FEATURES:
        v = raw.get(f)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            out[f] = math.nan
            continue
        if isinstance(v, bool):
            raise HTTPException(422, f"feature '{f}' must be a number, got a boolean")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise HTTPException(422, f"feature '{f}' must be a number, got {v!r}")
        if math.isinf(x):
            raise HTTPException(422, f"feature '{f}' must be finite")
        out[f] = x
    return out


def json_value(x: float):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def append_log(features: Dict[str, float], result: dict) -> None:
    row = {**{f: json_value(features[f]) for f in FEATURES},
           "pred_label": result["label"], "confidence": result["decision"]["confidence"],
           "threshold": result["decision"]["threshold"], "accepted": result["decision"]["accepted"]}
    try:
        pd.DataFrame([row]).to_csv(INFER_LOG, mode="a", index=False, header=not os.path.exists(INFER_LOG))
    except OSError as e:
        log.warning("could not write inference log %s: %s", INFER_LOG, e)


@app.get("/health")
def health():
    return {"ok": True, "version": VERSION}


@app.get("/metadata")
def metadata():
    cal = METRICS["calibration"]
    test = METRICS["test"]
    slim = {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "roc_auc_ovr": test["roc_auc_ovr_macro"],
        "n_train": METRICS["splits"]["train"],
        "n_calibration": METRICS["splits"]["calibration"],
        "n_test": METRICS["splits"]["test"],
        "ece_uncalibrated": cal["test_before"]["ece_top_label"],
        "ece_calibrated": cal["test_after"]["ece_top_label"],
        "brier_uncalibrated": cal["test_before"]["brier"],
        "brier_calibrated": cal["test_after"]["brier"],
        "recommended_threshold": DEFAULT_THRESHOLD,
        "target_precision": METRICS["decision"]["target_precision"],
    }
    return {
        "features": FEATURES,
        "labels": LABELS,
        "metrics": slim,
        "calibration": {"method": cal["method"], "temperature": predictor.temperature},
        "data": {"retrieved_utc": METRICS["data"].get("retrieved_utc"),
                 "labeled_rows": METRICS["data"]["labeled_rows"]},
        "version": VERSION,
    }


@app.get("/metrics_full")
def metrics_full():
    # Full evaluation record for charts (calibration, ROC/PR, confusion matrix, importances)
    return METRICS


@app.get("/feature_info")
def feature_info():
    return FEATURE_INFO


@app.get("/echo-sample")
def echo_sample(index: int = Query(0, ge=0)):
    if SAMPLES is None or len(SAMPLES) == 0:
        raise HTTPException(404, "sample_inputs.csv not found")
    i = index % len(SAMPLES)
    row = SAMPLES.iloc[i]
    payload = {"features": {f: (None if pd.isna(row.get(f)) else float(row.get(f))) for f in FEATURES},
               "index": i, "n_samples": int(len(SAMPLES))}
    for key in ("true_label", "mission"):
        if key in SAMPLES.columns and not pd.isna(row[key]):
            payload[key] = str(row[key])
    return payload


@app.post("/predict")
def predict(payload: PredictPayload):
    features = clean_features(payload.features)
    X = predictor.frame([features])
    prob = predictor.predict_proba(X)[0]
    prob_uncal = predictor.predict_proba(X, calibrated=False)[0]

    order = np.argsort(-prob)
    top_idx, second_idx = int(order[0]), int(order[1])
    top_p, second_p = float(prob[top_idx]), float(prob[second_idx])
    threshold = DEFAULT_THRESHOLD if payload.threshold is None else float(payload.threshold)
    accepted = top_p >= threshold

    # TreeSHAP contributions to the predicted class's calibrated logit
    contrib, base = predictor.contributions(X)
    shap_row = contrib[0, top_idx]
    contributions = sorted(
        ({"feature": f, "value": json_value(features[f]), "shap": float(s)} for f, s in zip(FEATURES, shap_row)),
        key=lambda c: abs(c["shap"]), reverse=True)

    out = {
        "label": LABELS[top_idx],
        "probs": {LABELS[i]: float(prob[i]) for i in range(len(LABELS))},
        "probs_uncalibrated": {LABELS[i]: float(prob_uncal[i]) for i in range(len(LABELS))},
        "decision": {
            "accepted": bool(accepted),
            "reason": "confidence_at_or_above_threshold" if accepted else "below_threshold",
            "threshold": threshold,
            "threshold_source": "default" if payload.threshold is None else "request",
            "confidence": top_p,
            "margin": top_p - second_p,
            "second_best": {"label": LABELS[second_idx], "prob": second_p},
        },
        "attribution": {
            "method": "TreeSHAP (LightGBM pred_contrib)",
            "explains": "calibrated logit of the predicted class",
            "class": LABELS[top_idx],
            "base_value": float(base[0, top_idx]),
            "logit": float(base[0, top_idx] + shap_row.sum()),
            "contributions": contributions,
        },
        "top_factors": contributions[:5],
    }
    if INFER_LOG:
        append_log(features, out)
    return out


@app.post("/batch_predict")
def batch_predict(rows: List[Dict[str, Any]]):
    if len(rows) == 0:
        raise HTTPException(422, "Provide a non-empty JSON array of {features:{...}} or raw feature dicts.")
    # accept either [{features:{...}}, {...}] formats; normalize
    cleaned = [clean_features(r.get("features", r)) for r in rows]
    prob = predictor.predict_proba(predictor.frame(cleaned))
    out = []
    for p in prob:
        k = int(p.argmax())
        out.append({
            "label": LABELS[k],
            "probs": {LABELS[j]: float(p[j]) for j in range(len(LABELS))},
            "confidence": float(p[k]),
            "accepted": bool(p[k] >= DEFAULT_THRESHOLD),
        })
    return out
