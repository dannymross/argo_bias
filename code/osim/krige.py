"""Global space-time covariance fit for the simple-kriging OHC estimator.

Fits one stationary anisotropic exponential covariance

    k(x, y) = phi * exp(-sqrt( (dlon/th_lon)^2 + (dlat/th_lat)^2 + (dt/th_t)^2 ))

plus a nugget, by Vecchia MLE (GpGp::exponential_scaledim) on a subsample of the
GLORYS residual field over the conditioning buffer. Fitting the *truth* field
rather than the sampled profiles makes the covariance an oracle quantity: it is
fixed across deployment realizations, so the kriging weights depend only on
design geometry and the bias splits into alignment + coverage with no
parameter-estimation term.

Sampling design. A uniform random subsample is nearly useless here: the domain is
~31 x 41 x 183 correlation volumes, so random points land about one range apart
and short lags -- which is what identifies the ranges against the nugget -- go
unsampled. Instead use a station design: clustered sites (companions at
log-spaced offsets resolve short spatial lags) observed on a common set of days
(repeat visits resolve temporal lags at fixed location).

    python code/osim/krige.py [--tag buffer] [--years 2015,...,2019]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg

REPO_ROOT = cfg.REPO_ROOT
EPOCH = np.datetime64("2010-01-01")


def grid_sample(tag="buffer", years=cfg.ANALYSIS_YEARS, coarse=(12, 12, 20),
                n_blocks=3, block=(18, 18, 90, 3)):
    """Nested regular lattice over the residual field -> (lon, lat, day, z).

    Two deterministic grids, because one spacing cannot do both jobs. The coarse
    lattice (`coarse` = lat/lon cell strides and a day stride) spans the whole
    buffer and carries the long lags. The fine blocks (`n_blocks` squared, laid
    out regularly, each `block` = nlat cells x nlon cells x ndays at a day
    stride, at NATIVE spatial resolution) carry lags below the coarse spacing.

    The fine part is not optional: the nugget here is not measurement error --
    the field is read exactly -- so it absorbs whatever variance falls below the
    fitting grid (10% of total at 0.5 deg, 28% at 1 deg). An over-large nugget
    shrinks the kriging weights, lowering the mass m and inflating the coverage
    term, which is the quantity the whole analysis reports.
    """
    path = os.path.join(cfg.OUT_DIR, f"residual_field_{tag}.nc")
    with xr.open_dataset(path) as ds:
        z = ds["z"]
        t = pd.DatetimeIndex(z["time"].values)
        idx = np.flatnonzero(np.isin(t.year, list(years)))
        t0, t1 = int(idx[0]), int(idx[-1]) + 1
        kla, klo, kt = coarse

        parts = [z.isel(time=slice(t0, t1, kt), latitude=slice(None, None, kla),
                        longitude=slice(None, None, klo))]

        nla, nlo, nd, sd = block
        # Block origins on a regular lattice inset by half a block; each block
        # takes a different time window so the record is covered evenly.
        la0 = np.linspace(0, z.sizes["latitude"] - nla, n_blocks).astype(int)
        lo0 = np.linspace(0, z.sizes["longitude"] - nlo, n_blocks).astype(int)
        starts = np.linspace(t0, t1 - nd, n_blocks * n_blocks).astype(int)
        for k, (a, b) in enumerate((a, b) for a in la0 for b in lo0):
            parts.append(z.isel(time=slice(int(starts[k]), int(starts[k]) + nd, sd),
                                latitude=slice(a, a + nla), longitude=slice(b, b + nlo)))

    out = []
    for p in parts:
        v = p.values
        day = (p["time"].values - EPOCH) / np.timedelta64(1, "D")
        LO, LA = np.meshgrid(p["longitude"].values, p["latitude"].values)
        out.append(pd.DataFrame({
            "lon": np.tile(LO.ravel(), len(day)),
            "lat": np.tile(LA.ravel(), len(day)),
            "day": np.repeat(day, LO.size),
            "z": v.ravel(),
        }))
    d = pd.concat(out, ignore_index=True).dropna()
    # Blocks can overlap the coarse lattice; duplicated locations would be a
    # zero-distance pair the exponential model cannot accommodate.
    return d.drop_duplicates(["lon", "lat", "day"]).reset_index(drop=True)


def run_fit(sample, out_csv, cache_rds, m_seq="10,30", covfun="exponential_scaledim"):
    """Rscript krige_fit.R on a written sample; cwd=REPO_ROOT for renv."""
    src = os.path.join(cfg.OUT_DIR, "krige_sample.csv")
    sample.to_csv(src, index=False)
    cmd = ["Rscript", "code/osim/krige_fit.R", os.path.abspath(src),
           os.path.abspath(out_csv), os.path.abspath(cache_rds), m_seq, covfun]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return pd.read_csv(out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="buffer")
    ap.add_argument("--years", default=None, help="comma-separated; default cfg.ANALYSIS_YEARS")
    ap.add_argument("--coarse", default="12,12,20", help="lat,lon cell strides and day stride")
    ap.add_argument("--n-blocks", type=int, default=3, help="n_blocks^2 native-resolution blocks")
    ap.add_argument("--m-seq", default="10,30")
    ap.add_argument("--covfun", default="exponential_scaledim",
                    choices=("exponential_scaledim", "matern_scaledim"))
    a = ap.parse_args()
    yrs = [int(y) for y in a.years.split(",")] if a.years else cfg.ANALYSIS_YEARS
    s = grid_sample(a.tag, yrs, coarse=tuple(int(x) for x in a.coarse.split(",")),
                    n_blocks=a.n_blocks)
    print(f"sample: {len(s):,} obs, {s[['lon','lat']].drop_duplicates().shape[0]} distinct sites, "
          f"{s['day'].nunique()} distinct days, sd(z) = {s['z'].std()*1e-9:.3f} GJ/m2", flush=True)
    sfx = "" if a.covfun.startswith("exponential") else "_matern"
    fit = run_fit(s, os.path.join(cfg.OUT_DIR, f"krige_fit_{a.tag}{sfx}.csv"),
                  os.path.join(cfg.OUT_DIR, f"krige_fit_{a.tag}{sfx}.rds"), a.m_seq, a.covfun)
    print(fit.to_string(index=False))
