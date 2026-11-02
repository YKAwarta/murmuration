"""Load the trained model artifacts and produce calibrated predictions with attributions.

Artifacts (written by src/model/train.py):
    lightgbm_model.txt   LightGBM booster in its native text format
    calibration.json     fitted temperature and how it was obtained
    feature_list.json    feature order expected by the booster
    metrics.json         evaluation results, labels and the default threshold
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model.calibration import apply_temperature

MODEL_FILE = "lightgbm_model.txt"
CALIBRATION_FILE = "calibration.json"
FEATURES_FILE = "feature_list.json"
METRICS_FILE = "metrics.json"


class Predictor:
    def __init__(self, booster: lgb.Booster, temperature: float, features: list, labels: list):
        if booster.num_model_per_iteration() != len(labels):
            raise ValueError("booster class count does not match labels")
        self.booster = booster
        self.temperature = float(temperature)
        self.features = list(features)
        self.labels = list(labels)

    @classmethod
    def load(cls, art_dir: str) -> "Predictor":
        booster = lgb.Booster(model_file=os.path.join(art_dir, MODEL_FILE))
        with open(os.path.join(art_dir, CALIBRATION_FILE)) as f:
            calibration = json.load(f)
        with open(os.path.join(art_dir, FEATURES_FILE)) as f:
            features = json.load(f)
        with open(os.path.join(art_dir, METRICS_FILE)) as f:
            labels = json.load(f)["labels"]
        return cls(booster, calibration["temperature"], features, labels)

    def frame(self, rows) -> pd.DataFrame:
        """Build a float frame in booster feature order; missing features become NaN."""
        if isinstance(rows, pd.DataFrame):
            return rows.reindex(columns=self.features).astype("float64")
        return pd.DataFrame([[r.get(f, np.nan) for f in self.features] for r in rows],
                            columns=self.features, dtype="float64")

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        """Raw LightGBM margins, one column per class (inputs to the softmax)."""
        return np.asarray(self.booster.predict(X, raw_score=True)).reshape(len(X), -1)

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        return apply_temperature(self.logits(X), self.temperature if calibrated else 1.0)

    def contributions(self, X: pd.DataFrame):
        """TreeSHAP contributions to the calibrated logits.

        Returns (contrib, base) with contrib of shape (n, K, F) and base of shape (n, K) such
        that base + contrib.sum(axis=2) equals logits / T for every row and class.
        """
        raw = np.asarray(self.booster.predict(X, pred_contrib=True))
        n, k, f = len(X), len(self.labels), len(self.features)
        raw = raw.reshape(n, k, f + 1) / self.temperature
        return raw[:, :, :f], raw[:, :, f]
