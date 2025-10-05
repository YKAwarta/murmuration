import os, io, requests, pandas as pd

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
OUTDIR = "artifacts/data"
os.makedirs(OUTDIR, exist_ok=True)

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

def fetch(query: str, out_csv: str):
    r = requests.get(TAP, params={"query": query, "format": "csv"}, timeout=120)
    try:
        r.raise_for_status()
    except Exception:
        print("TAP returned an error:\n", r.text[:1000])
        raise
    df = pd.read_csv(io.StringIO(r.text), comment="#", encoding="utf-8-sig")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} rows: {len(df)}")

if __name__ == "__main__":
    fetch(KOI_QUERY, os.path.join(OUTDIR, "koi_min.csv"))
    fetch(TOI_QUERY, os.path.join(OUTDIR, "toi_min.csv"))
