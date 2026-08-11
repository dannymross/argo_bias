"""Solve for the deployment schedule that holds the in-box standing stock flat.

A constant deployment rate cannot reach equilibrium here: the empirical Argo
lifespan pool has a median of 1339 d and the velocity buffer is large, so floats
keep re-entering the box for years and the standing stock is still climbing
after 900 d of spin-up (cv 0.11, 1.44x across the analysis year). An array that
ramps puts a late-loaded day-of-year marginal into the design, which is the
dominant term in the sampling bias -- so the ramp has to be removed by design,
not corrected afterwards.

Let g(u) be the probability a float is inside the box at age u, estimated from a
constant-rate pilot run. For a release density f over [-T, run_days) the
standing stock is (f * g)(t), so a flat stock is a non-negative deconvolution:

    min_f  || (f * g) - 1 ||^2 + lam || D^2 f ||^2   subject to f >= 0

The roughness penalty is load-bearing: the unpenalised solution is a sparse
vertex of the feasible set, and releasing floats in a few bursts would rebuild
the cohort structure in age that this arm exists to remove.

    python code/osse/calibrate_release.py <pilot.zarr> [--box boxB]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg
from osse.design_invariance import CYCLE_DAYS


def inbox_response(zarr_path, box="boxB", year=None, run_days=360, min_n=30):
    """g(u): P(float is in the box at age u), from a constant-rate pilot.

    Floats released late in the run are truncated before reaching large u, so
    each age is averaged only over floats that had that much run remaining --
    otherwise g decays for a bookkeeping reason rather than a physical one.
    """
    ds = xr.open_zarr(zarr_path)
    lat, lon, t = ds["lat"].values, ds["lon"].values, ds["time"].values
    # The run ENDS inside the analysis year; it starts spin_days before it, so
    # the last record is what identifies the year, not the first.
    year = year or int(str(t[np.isfinite(lat)].max())[:4])
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
    inside = (np.isfinite(lat) & (lat >= la0) & (lat <= la1)
              & (lon >= lo0) & (lon <= lo1))
    rel = np.nanmin((t - np.datetime64(f"{year}-01-01", "ns")) / np.timedelta64(1, "D"),
                    axis=1)
    ages = np.arange(inside.shape[1]) * CYCLE_DAYS
    g = np.full(len(ages), np.nan)
    for j, u in enumerate(ages):
        ok = rel + u <= run_days
        if ok.sum() >= min_n:
            g[j] = inside[ok, j].mean()
    keep = np.isfinite(g)
    return ages[keep], g[keep]


def solve_schedule(ages, g, spin_days=540, run_days=360, step=CYCLE_DAYS, lam=20.0):
    """Smooth non-negative release density giving a flat in-box standing stock."""
    s = np.arange(-spin_days, run_days, step)
    t = np.arange(0, run_days, step)
    lag = np.subtract.outer(t, s)
    G = np.interp(lag, ages, g, left=0.0, right=0.0)
    G[lag < 0] = 0.0

    n = len(s)
    D2 = np.zeros((n - 2, n))
    rows = np.arange(n - 2)
    D2[rows, rows], D2[rows, rows + 1], D2[rows, rows + 2] = 1.0, -2.0, 1.0
    f, _ = nnls(np.vstack([G, lam * D2]),
                np.concatenate([np.ones(len(t)), np.zeros(n - 2)]))
    if f.sum() <= 0:
        raise RuntimeError("degenerate schedule: check g and lam")
    return s, f / f.sum(), G @ f


def main(zarr_path, box="boxB", spin_days=540, run_days=360, lam=20.0, out=None,
         year=None):
    ages, g = inbox_response(zarr_path, box, year=year, run_days=run_days)
    s, f, stock = solve_schedule(ages, g, spin_days, run_days, lam=lam)
    const = np.interp(np.subtract.outer(np.arange(0, run_days, CYCLE_DAYS), s),
                      ages, g, left=0.0, right=0.0)
    const[np.subtract.outer(np.arange(0, run_days, CYCLE_DAYS), s) < 0] = 0.0
    c = const.sum(axis=1)

    out = out or os.path.join(cfg.OUT_DIR, f"release_schedule_{box}.csv")
    pd.DataFrame({"release_day": s, "weight": f}).to_csv(out, index=False)
    print(f"g(u) over ages {ages[0]}-{ages[-1]} d, g range {g.min():.3f}-{g.max():.3f}")
    print(f"tapered : stock cv {stock.std()/stock.mean():.4f}, "
          f"last/first {stock[-1]/stock[0]:.3f}, {f[s < 0].sum():.0%} released in spin-up")
    print(f"constant: stock cv {c.std()/c.mean():.4f}, last/first {c[-1]/c[0]:.3f}")
    print(f"wrote {out}")
    return s, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--spin-days", type=int, default=540)
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--lam", type=float, default=20.0)
    a = ap.parse_args()
    main(a.zarr, a.box, a.spin_days, a.days, a.lam, year=a.year)
