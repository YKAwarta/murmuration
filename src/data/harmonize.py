import pandas as pd
import numpy as np

FEATURES = [
    "period","duration","depth","impact","prad","insol","teq",
    "steff","slogg","srad","smass","smet","star_mag","snr","ntrans"
]
LABELS = ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]

def load_and_merge(koi_path: str, toi_path: str):
    # Kepler KOI (already has 'label')
    koi = pd.read_csv(koi_path, comment="#", encoding="utf-8-sig")
    koi["label"] = koi["label"].astype(str).str.upper().str.strip()
    koi["mission"] = "KEPLER"

    # TESS TOI (has 'raw_label' → map to label)
    toi = pd.read_csv(toi_path, comment="#", encoding="utf-8-sig")
    map_disp = {
        "CP":"CONFIRMED","KP":"CONFIRMED","KNOWN PLANET":"CONFIRMED","CONFIRMED":"CONFIRMED",
        "PC":"CANDIDATE","APC":"CANDIDATE","CANDIDATE":"CANDIDATE",
        "FP":"FALSE POSITIVE","FA":"FALSE POSITIVE","FALSE POSITIVE":"FALSE POSITIVE"
    }
    toi["label"] = toi["raw_label"].astype(str).str.upper().str.strip().map(map_disp)
    if "raw_label" in toi.columns:
        toi.drop(columns=["raw_label"], inplace=True)
    toi["mission"] = "TESS"

    df = pd.concat([koi, toi], ignore_index=True)

    # Ensure all expected feature columns exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep only rows with a valid label
    df = df[df["label"].isin(LABELS)].copy()

    # Cast numeric
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, FEATURES, LABELS
