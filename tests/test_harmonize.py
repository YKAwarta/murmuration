import numpy as np
import pandas as pd

from src.data.harmonize import FEATURES, LABELS, TOI_DISPOSITION_MAP, harmonize, load_and_merge


def tables():
    koi = pd.DataFrame({
        "label": ["CONFIRMED", " candidate ", "FALSE POSITIVE", "NOT DISPOSITIONED"],
        "period": [1.0, 2.0, 3.0, 4.0],
        "impact": [0.1, 0.2, 0.3, 0.4],
    })
    toi = pd.DataFrame({
        "raw_label": ["CP", "KP", "PC", "APC", "FP", "FA", "XX", None],
        "period": ["5.5", 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "impact": [np.nan] * 8,
    })
    return koi, toi


def test_toi_dispositions_map_to_koi_classes():
    _, toi = tables()
    df = harmonize(tables()[0], toi)
    tess = df[df["mission"] == "TESS"]
    assert tess["label"].tolist() == ["CONFIRMED", "CONFIRMED", "CANDIDATE", "CANDIDATE",
                                      "FALSE POSITIVE", "FALSE POSITIVE"]
    # unknown and blank codes are dropped
    assert len(tess) == 6


def test_mapping_covers_every_tfopwg_code_and_only_known_labels():
    for code in ["CP", "KP", "PC", "APC", "FP", "FA"]:
        assert code in TOI_DISPOSITION_MAP
    assert set(TOI_DISPOSITION_MAP.values()) == set(LABELS)


def test_koi_labels_are_normalised_and_undispositioned_rows_dropped():
    koi, toi = tables()
    df = harmonize(koi, toi)
    kepler = df[df["mission"] == "KEPLER"]
    assert kepler["label"].tolist() == ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]


def test_every_feature_exists_as_float64():
    koi, toi = tables()
    df = harmonize(koi, toi)
    for f in FEATURES:
        assert f in df.columns
        assert df[f].dtype == np.float64
    # columns absent from both inputs are all-NaN, strings are parsed as numbers
    assert df["smass"].isna().all()
    assert df.loc[df["mission"] == "TESS", "period"].iloc[0] == 5.5


def test_inputs_are_not_modified():
    koi, toi = tables()
    before = (koi.copy(), toi.copy())
    harmonize(koi, toi)
    pd.testing.assert_frame_equal(koi, before[0])
    pd.testing.assert_frame_equal(toi, before[1])


def test_load_and_merge_reads_archive_csvs(tmp_path):
    koi, toi = tables()
    koi_path, toi_path = tmp_path / "koi.csv", tmp_path / "toi.csv"
    # archive CSVs can carry '#' comment lines
    koi_path.write_text("# comment\n" + koi.to_csv(index=False))
    toi.to_csv(toi_path, index=False)
    df, features, labels = load_and_merge(str(koi_path), str(toi_path))
    assert features == FEATURES and labels == LABELS
    assert len(df) == 9
    assert set(df["label"]) <= set(LABELS)
