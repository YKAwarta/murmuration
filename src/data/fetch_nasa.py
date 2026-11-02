"""Download the Kepler KOI and TESS TOI tables from the NASA Exoplanet Archive.

Writes two CSVs plus a snapshot.json that records when the tables were pulled,
the exact ADQL queries and the row counts, so a training run can be traced back
to the archive state it used.

    python -m src.data.fetch_nasa --outdir data/raw
"""
import argparse
import io
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Kepler Objects of Interest, cumulative table. koi_disposition is the archive
# disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE).
KOI_QUERY = """
SELECT
  koi_disposition AS label,
  koi_period      AS period,
  koi_duration    AS duration,
  koi_depth       AS depth,
  koi_impact      AS impact,
  koi_prad        AS prad,
  koi_insol       AS insol,
  koi_teq         AS teq,
  koi_steff       AS steff,
  koi_slogg       AS slogg,
  koi_srad        AS srad,
  koi_smass       AS smass,
  koi_smet        AS smet,
  koi_kepmag      AS star_mag,
  koi_model_snr   AS snr,
  koi_num_transits AS ntrans
FROM cumulative
"""

# TESS Objects of Interest. tfopwg_disp is the TFOP working group disposition
# (CP, KP, PC, APC, FP, FA); it is mapped to the KOI classes in harmonize.py.
# The TOI table has no impact parameter, stellar mass, metallicity, model SNR
# or transit count, so those columns are NULL for every TESS row.
TOI_QUERY = """
SELECT
  tfopwg_disp       AS raw_label,
  pl_orbper         AS period,
  pl_trandurh       AS duration,
  pl_trandep        AS depth,
  NULL              AS impact,
  pl_rade           AS prad,
  pl_insol          AS insol,
  pl_eqt            AS teq,
  st_teff           AS steff,
  st_logg           AS slogg,
  st_rad            AS srad,
  NULL              AS smass,
  NULL              AS smet,
  st_tmag           AS star_mag,
  NULL              AS snr,
  NULL              AS ntrans
FROM toi
"""


def fetch(query: str, out_csv: str, retries: int = 4) -> int:
    """Run an ADQL query against the archive TAP service and save the result as CSV."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(TAP, params={"query": query, "format": "csv"}, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            last_error = e
            print(f"TAP request failed (attempt {attempt}/{retries}): {e}")
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"NASA Exoplanet Archive query failed after {retries} attempts") from last_error

    df = pd.read_csv(io.StringIO(r.text), comment="#", encoding="utf-8-sig")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} rows: {len(df)}")
    return len(df)


def main(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    koi_rows = fetch(KOI_QUERY, os.path.join(outdir, "koi_min.csv"))
    toi_rows = fetch(TOI_QUERY, os.path.join(outdir, "toi_min.csv"))
    snapshot = {
        "source": TAP,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tables": {
            "cumulative": {"file": "koi_min.csv", "rows": koi_rows, "query": KOI_QUERY.strip()},
            "toi": {"file": "toi_min.csv", "rows": toi_rows, "query": TOI_QUERY.strip()},
        },
    }
    with open(os.path.join(outdir, "snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)
    print("Wrote", os.path.join(outdir, "snapshot.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", default="data/raw")
    main(p.parse_args().outdir)
