import pandas as pd

FEATURES = [
    "period","duration","depth","impact","prad","insol","teq",
    "steff","slogg","srad","smass","smet","star_mag","snr","ntrans"
]
LABELS = ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]

# TESS TFOPWG dispositions -> the three KOI classes.
#   CP  confirmed planet, KP known planet          -> CONFIRMED
#   PC  planet candidate, APC ambiguous candidate  -> CANDIDATE
#   FP  false positive, FA false alarm             -> FALSE POSITIVE
# Anything else (blank, unknown codes) is dropped.
TOI_DISPOSITION_MAP = {
    "CP":"CONFIRMED","KP":"CONFIRMED","KNOWN PLANET":"CONFIRMED","CONFIRMED":"CONFIRMED",
    "PC":"CANDIDATE","APC":"CANDIDATE","CANDIDATE":"CANDIDATE",
    "FP":"FALSE POSITIVE","FA":"FALSE POSITIVE","FALSE POSITIVE":"FALSE POSITIVE"
}


def harmonize(koi: pd.DataFrame, toi: pd.DataFrame):
    """Merge KOI and TOI rows into one table with a shared label set and feature columns."""
    koi = koi.copy()
    # Kepler KOI already carries the archive disposition in 'label'
    koi["label"] = koi["label"].astype(str).str.upper().str.strip()
    koi["mission"] = "KEPLER"

    # TESS TOI has the TFOPWG code in 'raw_label'
    toi = toi.copy()
    toi["label"] = toi["raw_label"].astype(str).str.upper().str.strip().map(TOI_DISPOSITION_MAP)
    toi = toi.drop(columns=["raw_label"])
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
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    return df.reset_index(drop=True)


def load_and_merge(koi_path: str, toi_path: str):
    koi = pd.read_csv(koi_path, comment="#", encoding="utf-8-sig")
    toi = pd.read_csv(toi_path, comment="#", encoding="utf-8-sig")
    return harmonize(koi, toi), FEATURES, LABELS
