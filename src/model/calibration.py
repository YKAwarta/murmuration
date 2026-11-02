"""Probability calibration and calibration metrics for the multiclass model.

The classifier is a LightGBM softmax model, so its raw output is one logit per
class. Temperature scaling divides those logits by a single scalar T fitted by
minimising log loss on a held-out calibration split:

    p_calibrated = softmax(z / T)

Because the transform is a positive rescaling of the logits it never changes
the predicted class, and per-feature contributions to z (TreeSHAP) map exactly
to contributions to z / T.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Return the temperature T > 0 that minimises the log loss of softmax(logits / T)."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)
    rows = np.arange(len(y))

    def nll(log_t):
        return -np.mean(log_softmax(logits / np.exp(log_t), axis=1)[rows, y])

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded",
                          options={"xatol": 1e-6})
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def brier_multiclass(y: np.ndarray, prob: np.ndarray) -> float:
    """Multiclass Brier score: mean over rows of sum_k (p_k - 1[y = k])^2. Ranges 0 to 2."""
    onehot = np.eye(prob.shape[1])[np.asarray(y, dtype=int)]
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def log_loss_multiclass(y: np.ndarray, prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(prob[np.arange(len(y)), np.asarray(y, dtype=int)], eps, 1.0)
    return float(-np.mean(np.log(p)))


def top_label_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 10):
    """Expected calibration error of the top-class probability with equal-width bins.

    Returns (ece, bins) where bins lists, per non-empty bin, its midpoint, the mean
    confidence, the accuracy and the number of rows.
    """
    y = np.asarray(y, dtype=int)
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin i covers (edges[i], edges[i+1]]; the first bin also includes 0
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    bins = {"bin_mid": [], "conf": [], "acc": [], "count": []}
    ece = 0.0
    for i in range(n_bins):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        c, a = float(conf[mask].mean()), float(correct[mask].mean())
        ece += n / len(y) * abs(a - c)
        bins["bin_mid"].append(float((edges[i] + edges[i + 1]) / 2))
        bins["conf"].append(c)
        bins["acc"].append(a)
        bins["count"].append(n)
    return float(ece), bins


def classwise_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean over classes of the ECE of each class's one-vs-rest probability."""
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_class = []
    for k in range(prob.shape[1]):
        p = prob[:, k]
        hit = (y == k).astype(float)
        idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, n_bins - 1)
        e = 0.0
        for i in range(n_bins):
            mask = idx == i
            if mask.any():
                e += mask.mean() * abs(hit[mask].mean() - p[mask].mean())
        per_class.append(e)
    return float(np.mean(per_class))


def calibration_report(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    ece, bins = top_label_ece(y, prob, n_bins)
    return {
        "log_loss": log_loss_multiclass(y, prob),
        "brier": brier_multiclass(y, prob),
        "ece_top_label": ece,
        "ece_classwise": classwise_ece(y, prob, n_bins),
        "reliability_bins": bins,
    }


def threshold_for_precision(y: np.ndarray, prob: np.ndarray, target_prec: float = 0.90) -> float:
    """Lowest top-class probability threshold whose accepted set still reaches target precision.

    Rows are sorted by confidence; the threshold is the confidence of the last row at which
    the running precision of the accepted set is >= target_prec. Falls back to 0.5.
    """
    y = np.asarray(y, dtype=int)
    conf = prob.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    correct = (prob.argmax(axis=1)[order] == y[order]).astype(int)
    precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    idx = np.where(precision >= target_prec)[0]
    if len(idx) == 0:
        return 0.5
    return float(conf[order][idx[-1]])


def decision_stats(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    """Coverage and precision of the predictions whose top-class probability is >= threshold."""
    y = np.asarray(y, dtype=int)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    accept = conf >= threshold
    return {
        "threshold": float(threshold),
        "coverage": float(accept.mean()),
        "precision_on_accepted": float((pred[accept] == y[accept]).mean()) if accept.any() else None,
        "n_accepted": int(accept.sum()),
    }
