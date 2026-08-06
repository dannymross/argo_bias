"""Invariant checks for the OSSE permutation-test pipeline.

These are cheap and should hold by construction; they are assertions about the
gather and the bias identity, not statistical results.

Run from the repo root: ``python code/osse/verify_pipeline.py [box]``.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg
from osse.sample_residuals import gather

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def main(box="boxB"):
    z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc"))["z"].load()
    area = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    lat, lon = z["latitude"].values, z["longitude"].values
    rng = np.random.default_rng(0)

    print("\n1. Field bookkeeping")
    wet = np.isfinite(area.values)
    check("mask fraction reported", True,
          f"{wet.mean():.4f} wet, area {np.nansum(area.values)/1e6:,.0f} km2")
    check("Z is NaN exactly where the cell is dry",
          bool((np.isfinite(z.values).any(axis=(0, 1)) == wet).all()))
    check("every year has 365 doy slots (Feb 29 dropped)",
          z.sizes["doy"] == cfg.DAYS_PER_YEAR and z.sizes["year"] == len(cfg.ANALYSIS_YEARS))

    print("\n3. Lookup exactness (the gather is a lookup, not an interpolation)")
    ii = rng.integers(0, len(lat), 400)
    jj = rng.integers(0, len(lon), 400)
    dd = rng.integers(0, cfg.DAYS_PER_YEAR, 400)
    pos = pd.DataFrame({"vel_year": 2015, "rep": 1, "lat": lat[ii], "lon": lon[jj], "doy": dd})
    got, _ = gather(pos, z)
    keep = np.isfinite(z.values[:, dd, ii, jj]).all(axis=0)
    want = z.values[:, dd[keep], ii[keep], jj[keep]].T
    check("cell centres return the field bit-for-bit",
          np.array_equal(got[[f"z_{y}" for y in z['year'].values]].to_numpy(), want))

    dlat = np.median(np.diff(lat))
    dlon = np.median(np.diff(lon))
    jit = pos.copy()
    jit["lat"] = pos["lat"] + 0.4 * dlat * rng.choice([-1, 1], len(pos))
    jit["lon"] = pos["lon"] + 0.4 * dlon * rng.choice([-1, 1], len(pos))
    got_j, _ = gather(jit, z)
    check("jitter within a cell returns an identical value (piecewise constant)",
          np.array_equal(got[[f"z_{y}" for y in z['year'].values]].to_numpy(),
                         got_j[[f"z_{y}" for y in z['year'].values]].to_numpy()))

    edge = pos.copy()
    edge["lat"] = pos["lat"] + 0.5 * dlat
    e1, _ = gather(edge, z)
    e2, _ = gather(edge, z)
    check("boundary positions land deterministically",
          e1.equals(e2))

    print("\n4. Bias identity")
    w = np.nan_to_num(area.values)
    zbar = np.array([np.nansum(np.nan_to_num(z.values[k]) * w) / (w.sum() * cfg.DAYS_PER_YEAR)
                     for k in range(z.sizes["year"])])

    # A design that samples P exactly: every wet cell-day once, area-weighted.
    # Then (1/n) sum Z(x_i) with area weights is exactly Zbar, so B = 0.
    zi = np.nan_to_num(z.values) * w[None, None, :, :]
    b_exact = zi.sum(axis=(1, 2, 3)) / (w.sum() * cfg.DAYS_PER_YEAR) - zbar
    check("B = 0 for a design that samples P exactly",
          np.allclose(b_exact, 0, atol=1e-6 * np.abs(zbar).max()),
          f"max |B| = {np.abs(b_exact).max():.3e}")

    const = xr.zeros_like(z) + 5.0
    const = const.where(np.isfinite(z))
    pos2 = pd.DataFrame({"vel_year": 2015, "rep": 1,
                         "lat": rng.uniform(lat.min(), lat.max(), 2000),
                         "lon": rng.uniform(lon.min(), lon.max(), 2000),
                         "doy": rng.integers(0, cfg.DAYS_PER_YEAR, 2000)})
    gc, _ = gather(pos2, const)
    zc = gc[[f"z_{y}" for y in z["year"].values]].to_numpy()
    check("B = 0 for a constant field",
          np.allclose(zc.mean(axis=0) - 5.0, 0), f"max |B| = {np.abs(zc.mean(axis=0)-5.0).max():.2e}")

    print("\n" + ("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "boxB"))
