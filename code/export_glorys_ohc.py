"""Export native-grid GLORYS OHC to CSV for the R leakage-curve analysis.

Computes OHC = trapz(z, T*cp*rho) (0-700 m and 0-2000 m; see ``ohc.py``) on the
full ``data/glorys/`` domain, then writes one tidy CSV per year to
``data/glorys_ohc/`` -- columns ``lon, lat, date, ohc_700, ohc_2000`` in
GJ/m^2 -- so ``code/leakage_curve.R`` can read it with ``data.table::fread``
without an R NetCDF package. Reuses the cached ``data/ohc_truth/truth_ohc_*.nc``
fields when present (same domain/computation) rather than re-reading the 41
depth levels of every monthly file.

Run from the repo root: ``python code/export_glorys_ohc.py``.
"""

import os
import glob
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ohc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLORYS_DIR = os.path.join(REPO_ROOT, "data", "glorys")
TRUTH_DIR = os.path.join(REPO_ROOT, "data", "ohc_truth")
OUT_DIR = os.path.join(REPO_ROOT, "data", "glorys_ohc")
YEARS = (2020, 2021, 2022)
VALUE_COLS = [f"ohc_{z}" for z in ohc.DEPTHS]


def truth_field_for_year(year):
    cache = os.path.join(TRUTH_DIR, f"truth_ohc_{year}.nc")
    if os.path.exists(cache):
        return xr.open_dataset(cache)
    files = sorted(glob.glob(os.path.join(GLORYS_DIR, f"velocity_{year}_*.nc")))
    if not files:
        raise FileNotFoundError(f"no GLORYS files for {year} in {GLORYS_DIR}")
    return ohc.truth_ohc_field(xr.open_mfdataset(files)).compute()


def export_year(year):
    ds = truth_field_for_year(year)
    df = (
        ds[VALUE_COLS]
        .to_dataframe()
        .reset_index()
        .rename(columns={"latitude": "lat", "longitude": "lon", "time": "date"})
        .dropna(subset=VALUE_COLS, how="all")
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in VALUE_COLS:
        df[c] = (df[c] * ohc.J_TO_GJ).round(6)
    out = os.path.join(OUT_DIR, f"glorys_ohc_{year}.csv")
    df[["lon", "lat", "date"] + VALUE_COLS].to_csv(out, index=False)
    print(f"{year}: {len(df):,} rows -> {out}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for y in YEARS:
        export_year(y)
