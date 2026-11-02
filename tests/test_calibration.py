import numpy as np
import pytest
from scipy.special import softmax

from src.model.calibration import (apply_temperature, brier_multiclass, calibration_report,
                                   classwise_ece, decision_stats, fit_temperature,
                                   log_loss_multiclass, threshold_for_precision, top_label_ece)


def sample_from(logits, rng):
    p = softmax(logits, axis=1)
    return np.array([rng.choice(p.shape[1], p=row) for row in p])


def test_fit_temperature_recovers_known_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(20000, 3))
    # labels drawn from softmax(z / 2): the logits are over-confident by a factor of 2
    y = sample_from(z / 2.0, rng)
    assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.05)


def test_temperature_keeps_rows_normalised_and_argmax():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(500, 3))
    for t in [0.5, 1.0, 1.4, 3.0]:
        p = apply_temperature(z, t)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        np.testing.assert_array_equal(p.argmax(axis=1), z.argmax(axis=1))
    np.testing.assert_allclose(apply_temperature(z, 1.0), softmax(z, axis=1))


def test_temperature_scaling_improves_calibration_of_overconfident_logits():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 3, size=(6000, 3))
    y = sample_from(z / 1.8, rng)
    fit, held = slice(0, 3000), slice(3000, None)
    t = fit_temperature(z[fit], y[fit])
    before = calibration_report(y[held], apply_temperature(z[held], 1.0))
    after = calibration_report(y[held], apply_temperature(z[held], t))
    assert after["ece_top_label"] < before["ece_top_label"]
    assert after["brier"] < before["brier"]
    assert after["log_loss"] < before["log_loss"]


def test_matches_scikit_learn_temperature_scaling():
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 4))
    y = (X[:, 0] + rng.normal(0, 1, 1500) > 0).astype(int) + (X[:, 1] > 0.5).astype(int)
    clf = LGBMClassifier(n_estimators=80, verbose=-1, random_state=0).fit(X[:900], y[:900])
    t = fit_temperature(clf.predict_proba(X[900:1200], raw_score=True), y[900:1200])
    ours = apply_temperature(clf.predict_proba(X[1200:], raw_score=True), t)
    sk = CalibratedClassifierCV(FrozenEstimator(clf), method="temperature").fit(X[900:1200], y[900:1200])
    np.testing.assert_allclose(ours, sk.predict_proba(X[1200:]), atol=2e-3)


def test_brier_score_by_hand():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert brier_multiclass(y, perfect) == 0.0
    uniform = np.full((3, 3), 1 / 3)
    assert brier_multiclass(y, uniform) == pytest.approx(2 / 3)
    p = np.array([[0.7, 0.2, 0.1]])
    assert brier_multiclass(np.array([0]), p) == pytest.approx(0.3**2 + 0.2**2 + 0.1**2)


def test_log_loss_by_hand():
    p = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    assert log_loss_multiclass(np.array([0, 1]), p) == pytest.approx(-(np.log(0.5) + np.log(0.8)) / 2)


def test_top_label_ece_by_hand():
    # two rows at confidence 0.9 (one right, one wrong) and two at 0.6 (both right)
    p = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.6, 0.3, 0.1]])
    y = np.array([0, 1, 0, 0])
    ece, bins = top_label_ece(y, p, n_bins=10)
    # bin (0.8, 0.9]: |0.5 - 0.9| * 2/4, bin (0.5, 0.6]: |1.0 - 0.6| * 2/4
    assert ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)
    assert bins["count"] == [2, 2]
    assert bins["acc"] == [1.0, 0.5]


def test_ece_is_small_for_calibrated_probabilities():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 2, size=(30000, 3))
    p = softmax(z, axis=1)
    y = sample_from(z, rng)
    assert top_label_ece(y, p)[0] < 0.01
    assert classwise_ece(y, p) < 0.01


def test_threshold_for_precision():
    # sorted by confidence: correct, correct, correct, wrong, correct, wrong
    conf = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6])
    p = np.stack([conf, 1 - conf, np.zeros_like(conf)], axis=1)
    y = np.array([0, 0, 0, 1, 0, 1])
    # running precision: 1, 1, 1, .75, .8, .667 -> last index with >= 0.8 is conf 0.7
    assert threshold_for_precision(y, p, 0.8) == pytest.approx(0.7)
    assert threshold_for_precision(y, p, 1.0) == pytest.approx(0.85)
    assert threshold_for_precision(np.ones(6, dtype=int), p, 0.9) == 0.5  # unreachable target


def test_decision_stats():
    conf = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6])
    p = np.stack([conf, 1 - conf, np.zeros_like(conf)], axis=1)
    y = np.array([0, 0, 0, 1, 0, 1])
    s = decision_stats(y, p, 0.8)
    assert s["coverage"] == pytest.approx(4 / 6)
    assert s["precision_on_accepted"] == pytest.approx(3 / 4)
    assert decision_stats(y, p, 0.99)["precision_on_accepted"] is None
