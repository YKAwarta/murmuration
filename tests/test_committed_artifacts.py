"""Checks on the model shipped in artifacts/ (skipped if it has not been trained)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.predictor import Predictor

ART = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "lightgbm_model.txt").exists(), reason="no trained model")


def test_committed_model_loads_and_is_consistent():
    p = Predictor.load(str(ART))
    samples = pd.read_csv(ART / "sample_inputs.csv")
    X = p.frame(samples[p.features])
    prob = p.predict_proba(X)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)
    contrib, base = p.contributions(X)
    np.testing.assert_allclose(base + contrib.sum(axis=2), p.logits(X) / p.temperature, atol=1e-6)


def test_committed_metrics_record_the_calibration_gain():
    m = json.loads((ART / "metrics.json").read_text())
    cal = m["calibration"]
    assert cal["method"] == "temperature"
    assert cal["temperature"] == pytest.approx(json.loads((ART / "calibration.json").read_text())["temperature"])
    assert cal["test_after"]["ece_top_label"] < cal["test_before"]["ece_top_label"]
    assert cal["test_after"]["brier"] < cal["test_before"]["brier"]
