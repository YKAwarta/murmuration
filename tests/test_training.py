import json

import lightgbm as lgb
import numpy as np
import pytest

from src.data.harmonize import load_and_merge
from src.model.calibration import fit_temperature
from src.model.predictor import Predictor
from src.model.train import split_indices


def test_splits_are_disjoint_and_cover_every_row(synthetic_csvs):
    df, _, _ = load_and_merge(*map(str, synthetic_csvs))
    tr, ca, te = split_indices(df)
    assert not set(tr) & set(ca) and not set(tr) & set(te) and not set(ca) & set(te)
    assert len(tr) + len(ca) + len(te) == len(df)
    assert abs(len(ca) - 0.2 * len(df)) <= 2 and abs(len(te) - 0.2 * len(df)) <= 2


def test_training_writes_artifacts_and_metrics(trained_artifacts):
    out, figs = trained_artifacts
    for name in ["lightgbm_model.txt", "calibration.json", "feature_list.json",
                 "metrics.json", "sample_inputs.csv"]:
        assert (out / name).exists(), name
    assert (figs / "reliability_diagram.png").stat().st_size > 0
    m = json.loads((out / "metrics.json").read_text())
    cal = m["calibration"]
    for key in ["test_before", "test_after"]:
        for metric in ["brier", "ece_top_label", "ece_classwise", "log_loss"]:
            assert 0 <= cal[key][metric]
    assert set(cal["method_comparison_cv_on_calibration_split"]) == {
        "uncalibrated", "temperature", "sigmoid", "isotonic"}
    assert 0 < m["decision"]["recommended_threshold"] <= 1
    s = m["splits"]
    assert s["train"] + s["calibration"] + s["test"] == m["data"]["labeled_rows"]


def test_temperature_is_fitted_on_calibration_rows_only(trained_artifacts, synthetic_csvs):
    out, _ = trained_artifacts
    stored = json.loads((out / "calibration.json").read_text())["temperature"]
    df, F, L = load_and_merge(*map(str, synthetic_csvs))
    y = df["label"].map({lab: i for i, lab in enumerate(L)}).to_numpy()
    _, ca, te = split_indices(df)
    booster = lgb.Booster(model_file=str(out / "lightgbm_model.txt"))
    logits = booster.predict(df[F], raw_score=True)
    assert stored == pytest.approx(fit_temperature(logits[ca], y[ca]), rel=1e-6)
    assert stored != pytest.approx(fit_temperature(logits[te], y[te]), rel=1e-3)


def test_predictor_contributions_add_up_to_calibrated_logits(trained_artifacts, synthetic_csvs):
    out, _ = trained_artifacts
    p = Predictor.load(str(out))
    df, F, _ = load_and_merge(*map(str, synthetic_csvs))
    X = p.frame(df[F].head(50))
    contrib, base = p.contributions(X)
    np.testing.assert_allclose(base + contrib.sum(axis=2), p.logits(X) / p.temperature, atol=1e-6)
    np.testing.assert_allclose(p.predict_proba(X).sum(axis=1), 1.0)
