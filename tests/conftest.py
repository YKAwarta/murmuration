import argparse
import importlib
import os

import numpy as np
import pandas as pd
import pytest

from src.data.harmonize import FEATURES

KOI_LABELS = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
TOI_CODES = {"FALSE POSITIVE": ["FP", "FA"], "CANDIDATE": ["PC", "APC"], "CONFIRMED": ["CP", "KP"]}
TOI_MISSING = ["impact", "smass", "smet", "snr", "ntrans"]


def synthetic_tables(n_koi=480, n_toi=360, seed=0):
    """KOI/TOI-shaped tables whose features carry a learnable but noisy class signal."""
    rng = np.random.default_rng(seed)

    def features(labels):
        k = np.array([KOI_LABELS.index(lab) for lab in labels], dtype=float)
        cols = {}
        for j, f in enumerate(FEATURES):
            cols[f] = (j % 4 + 1) * k + rng.normal(0, 2.0, len(labels))
        return pd.DataFrame(cols)

    koi_labels = rng.choice(KOI_LABELS, size=n_koi, p=[0.5, 0.2, 0.3])
    koi = features(koi_labels)
    koi.insert(0, "label", koi_labels)

    toi_labels = rng.choice(KOI_LABELS, size=n_toi, p=[0.2, 0.6, 0.2])
    toi = features(toi_labels)
    toi[TOI_MISSING] = np.nan
    toi.insert(0, "raw_label", [rng.choice(TOI_CODES[lab]) for lab in toi_labels])
    return koi, toi


@pytest.fixture(scope="session")
def synthetic_csvs(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    koi, toi = synthetic_tables()
    koi.to_csv(d / "koi_min.csv", index=False)
    toi.to_csv(d / "toi_min.csv", index=False)
    return d / "koi_min.csv", d / "toi_min.csv"


@pytest.fixture(scope="session")
def trained_artifacts(synthetic_csvs, tmp_path_factory):
    """Run the real training pipeline on the synthetic tables with a one-point grid."""
    from src.model import train

    koi, toi = synthetic_csvs
    out = tmp_path_factory.mktemp("artifacts")
    figs = tmp_path_factory.mktemp("figs")
    grid = train.GRID
    train.GRID = {"n_estimators": [60], "learning_rate": [0.1], "num_leaves": [15]}
    try:
        train.main(argparse.Namespace(koi=str(koi), toi=str(toi), outdir=str(out), figdir=str(figs)))
    finally:
        train.GRID = grid
    return out, figs


@pytest.fixture(scope="session")
def api_module(trained_artifacts):
    """Import api.main against the synthetic artifacts."""
    out, _ = trained_artifacts
    old = os.environ.get("ARTIFACTS_DIR")
    os.environ["ARTIFACTS_DIR"] = str(out)
    os.environ.pop("MURMURATION_INFER_LOG", None)
    try:
        import api.main
        module = importlib.reload(api.main)
    finally:
        if old is None:
            os.environ.pop("ARTIFACTS_DIR", None)
        else:
            os.environ["ARTIFACTS_DIR"] = old
    return module


@pytest.fixture(scope="session")
def client(api_module):
    from fastapi.testclient import TestClient
    return TestClient(api_module.app)
