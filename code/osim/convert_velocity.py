"""Park-depth velocity as a 2D float32 field, one chunk per day, for faster advection.

The source files hold uo/vo on the two GLORYS levels bracketing PARK_DEPTH, in small
compressed 4-day chunks. Parcels zeroes NaNs on load and interpolates linearly in
depth, and floats never leave PARK_DEPTH, so zero-filling then interpolating to
PARK_DEPTH here reproduces the velocity Parcels samples, up to float32 rounding.
File names are unchanged, so only the directory differs.

    python code/osim/convert_velocity.py [--years 2015] [--complevel 1]
    python code/osim/convert_velocity.py --verify [--years 2015]
"""

import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import deployment as di


def to_park_depth(ds, depth=di.PARK_DEPTH):
    z = ds["depth"].values.astype(float)
    k = int(np.clip(np.searchsorted(z, depth) - 1, 0, len(z) - 2))
    w = (depth - z[k]) / (z[k + 1] - z[k])
    out = xr.Dataset({v: ((1 - w) * ds[v].isel(depth=k).fillna(0.0)
                          + w * ds[v].isel(depth=k + 1).fillna(0.0)).astype(np.float32)
                      for v in ("uo", "vo")})
    for v in ("uo", "vo"):
        out[v].attrs = {a: ds[v].attrs[a] for a in ("units", "standard_name", "long_name")
                        if a in ds[v].attrs}
    out.attrs.update(park_depth_m=float(depth), source_levels_m=f"{z[k]:.3f},{z[k + 1]:.3f}",
                     note="NaN set to 0 before linear depth interpolation (Parcels load convention)")
    return out


def convert(src, dst, complevel=1, overwrite=False, digits=None):
    """digits: decimal places kept (netCDF least_significant_digit); float32 either way."""
    if os.path.exists(dst) and not overwrite:
        return False
    with xr.open_dataset(src, chunks={"time": 4, "depth": -1, "latitude": -1, "longitude": -1}) as ds:
        out = to_park_depth(ds)
        shape = (1,) + out["uo"].shape[1:]
        enc = {v: {"chunksizes": shape, "zlib": complevel > 0, "complevel": complevel,
                   "shuffle": complevel > 0,
                   **({"least_significant_digit": digits} if digits else {})} for v in ("uo", "vo")}
        tmp = dst + ".tmp"
        out.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, dst)
    return True


def verify(src, dst, n=20000, seed=0):
    """Max abs difference at random grid points between the converted field and the
    zero-filled, depth-interpolated source computed in float64."""
    r = np.random.default_rng(seed)
    with xr.open_dataset(src) as a, xr.open_dataset(dst) as b:
        nt, _, ny, nx = a["uo"].shape
        it, iy, ix = r.integers(0, nt, n), r.integers(0, ny, n), r.integers(0, nx, n)
        z = a["depth"].values.astype(float)
        k = int(np.clip(np.searchsorted(z, di.PARK_DEPTH) - 1, 0, len(z) - 2))
        w = (di.PARK_DEPTH - z[k]) / (z[k + 1] - z[k])
        err = {}
        for v in ("uo", "vo"):
            A = np.nan_to_num(a[v].values)
            ref = (1 - w) * A[it, k, iy, ix] + w * A[it, k + 1, iy, ix]
            err[v] = float(np.abs(b[v].values[it, iy, ix] - ref).max())
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="comma-separated; default all in VEL_DIR")
    ap.add_argument("--complevel", type=int, default=1, help="0 = uncompressed")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out-dir", default=di.VEL1000_DIR)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc")))
    if a.years:
        yrs = a.years.split(",")
        srcs = [f for f in srcs if os.path.basename(f).split("_")[-2] in yrs]
    for f in srcs:
        g = os.path.join(a.out_dir, os.path.basename(f))
        if a.verify:
            print(os.path.basename(f), verify(f, g), flush=True)
        else:
            done = convert(f, g, a.complevel, a.overwrite)
            print(f"{os.path.basename(f)}: {'written' if done else 'exists'} "
                  f"{os.path.getsize(g) / 1e6:.0f} MB (src {os.path.getsize(f) / 1e6:.0f} MB)",
                  flush=True)
