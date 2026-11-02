import math

import numpy as np
import pytest

FEATURES = ["period", "duration", "depth", "impact", "prad", "insol", "teq", "steff",
            "slogg", "srad", "smass", "smet", "star_mag", "snr", "ntrans"]


@pytest.fixture()
def sample_features(client):
    return client.get("/echo-sample").json()["features"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_metadata(client, api_module):
    m = client.get("/metadata").json()
    assert m["features"] == FEATURES
    assert m["labels"] == ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
    assert m["metrics"]["recommended_threshold"] == pytest.approx(api_module.DEFAULT_THRESHOLD)
    assert m["calibration"]["method"] == "temperature"
    assert m["calibration"]["temperature"] > 0


def test_metrics_full(client):
    m = client.get("/metrics_full").json()
    assert {"calibration", "decision", "test", "feature_importance"} <= set(m)
    assert len(m["calibration"]["test_after"]["reliability_bins"]["conf"]) > 0


def test_feature_info(client):
    assert set(client.get("/feature_info").json()) == set(FEATURES)


def test_echo_sample_cycles(client):
    first = client.get("/echo-sample").json()
    n = first["n_samples"]
    assert n >= 1 and set(first["features"]) == set(FEATURES)
    again = client.get(f"/echo-sample?index={n}").json()
    assert again["index"] == 0 and again["features"] == first["features"]
    assert client.get("/echo-sample?index=-1").status_code == 422


def test_predict_returns_calibrated_probabilities(client, api_module, sample_features):
    r = client.post("/predict", json={"features": sample_features})
    assert r.status_code == 200
    out = r.json()
    probs = out["probs"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert out["label"] == max(probs, key=probs.get)
    # the served probabilities are the temperature-scaled ones
    p = api_module.predictor
    X = p.frame([{f: (math.nan if v is None else v) for f, v in sample_features.items()}])
    expected = p.predict_proba(X)[0]
    assert [probs[lab] for lab in p.labels] == pytest.approx(expected.tolist())
    raw = p.predict_proba(X, calibrated=False)[0]
    assert [out["probs_uncalibrated"][lab] for lab in p.labels] == pytest.approx(raw.tolist())


def test_predict_attributions_explain_the_prediction(client, api_module, sample_features):
    out = client.post("/predict", json={"features": sample_features}).json()
    att = out["attribution"]
    assert att["class"] == out["label"]
    assert [c["feature"] for c in att["contributions"]] != []
    assert {c["feature"] for c in att["contributions"]} == set(FEATURES)
    shap_sum = sum(c["shap"] for c in att["contributions"])
    assert att["base_value"] + shap_sum == pytest.approx(att["logit"], abs=1e-6)
    # the explained logit is the calibrated logit of the predicted class
    p = api_module.predictor
    X = p.frame([{f: (math.nan if v is None else v) for f, v in sample_features.items()}])
    k = p.labels.index(out["label"])
    assert att["logit"] == pytest.approx(p.logits(X)[0, k] / p.temperature, abs=1e-6)
    # sorted by magnitude, top_factors are the first five
    mags = [abs(c["shap"]) for c in att["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert out["top_factors"] == att["contributions"][:5]


def test_attributions_are_present_for_every_sample(client):
    n = client.get("/echo-sample").json()["n_samples"]
    for i in range(n):
        feats = client.get(f"/echo-sample?index={i}").json()["features"]
        out = client.post("/predict", json={"features": feats}).json()
        assert len(out["attribution"]["contributions"]) == len(FEATURES)
        assert any(c["shap"] != 0 for c in out["attribution"]["contributions"])


def test_default_threshold_and_override(client, api_module, sample_features):
    out = client.post("/predict", json={"features": sample_features}).json()
    d = out["decision"]
    assert d["threshold"] == pytest.approx(api_module.DEFAULT_THRESHOLD)
    assert d["threshold_source"] == "default"
    assert d["accepted"] == (d["confidence"] >= d["threshold"])
    low = client.post("/predict", json={"features": sample_features, "threshold": 0.0}).json()
    assert low["decision"]["accepted"] is True and low["decision"]["threshold_source"] == "request"
    high = client.post("/predict", json={"features": sample_features, "threshold": 1.0}).json()
    assert high["decision"]["accepted"] is (high["decision"]["confidence"] >= 1.0)
    assert client.post("/predict", json={"features": sample_features, "threshold": 1.5}).status_code == 422


def test_predict_handles_missing_values(client):
    out = client.post("/predict", json={"features": {"period": 10.5, "depth": "", "impact": None}}).json()
    assert sum(out["probs"].values()) == pytest.approx(1.0)
    values = {c["feature"]: c["value"] for c in out["attribution"]["contributions"]}
    assert values["period"] == 10.5 and values["depth"] is None and values["snr"] is None


@pytest.mark.parametrize("features", [
    {"not_a_feature": 1.0},
    {"period": "abc"},
    {"period": True},
    {"period": [1, 2]},
])
def test_predict_rejects_bad_input(client, features):
    assert client.post("/predict", json={"features": features}).status_code == 422


def test_batch_predict_matches_single_predictions(client, sample_features):
    single = client.post("/predict", json={"features": sample_features}).json()
    r = client.post("/batch_predict", json=[{"features": sample_features}, sample_features])
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    for row in rows:
        assert row["label"] == single["label"]
        assert row["probs"] == pytest.approx(single["probs"])
        assert row["accepted"] == single["decision"]["accepted"]
    assert client.post("/batch_predict", json=[]).status_code == 422
    assert client.post("/batch_predict", json=[{"period": "x"}]).status_code == 422


def test_no_inference_log_by_default(client, trained_artifacts, sample_features):
    out, _ = trained_artifacts
    client.post("/predict", json={"features": sample_features})
    assert not (out / "infer_log.csv").exists()
