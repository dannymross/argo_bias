"""Benchmark advection on the source vs converted park velocity field.

Identical seeds (uniform deployment_plan, all released at --start) are advected
through one velocity directory; wall time and peak RSS go to a JSON next to the
trajectory zarr. --compare reports the great-circle separation between two runs at
each 10-day output, which for a correct conversion starts at ~0 and grows only
through chaotic amplification of float32 rounding.

    python code/osim/bench_velocity.py run --label orig --vel-dir data/glorys_osim/velocity_daily
    python code/osim/bench_velocity.py run --label c1000        # default: deployment.SIM_VEL_DIR
    python code/osim/bench_velocity.py compare orig c1000
"""

import argparse
import json
import os
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import continuous_sim as cs
from osim import deployment as di
from osim import osim_config as cfg

di_default = di.SIM_VEL_DIR
BENCH_DIR = os.path.join(cfg.OUT_DIR, "_bench")


def run(label, vel_dir=None, n=1190, start="2015-01-01", days=365, spread=False, zarr_dir=None):
    os.makedirs(BENCH_DIR, exist_ok=True)
    end = str((pd.Timestamp(start) + pd.Timedelta(days=days)).date())
    lat, lon, rel, life = cs.deployment_plan(n, 0, start, end)
    if not spread:
        rel = np.full(n, np.datetime64(start, "s"))
    zdir = zarr_dir or BENCH_DIR
    os.makedirs(zdir, exist_ok=True)
    zpath = os.path.join(zdir, f"{label}.zarr")
    shutil.rmtree(zpath, ignore_errors=True)
    t0 = time.perf_counter()
    cs.run_continuous(0, zpath, n, start, end, seeds=(lat, lon, rel, life), vel_dir=vel_dir)
    t1 = time.perf_counter()
    prof = cs.profiles_from_zarr(zpath, 0)
    t2 = time.perf_counter()
    nfiles = sum(len(f) for _, _, f in os.walk(zpath))
    if zarr_dir:
        shutil.copytree(zpath, os.path.join(BENCH_DIR, f"{label}.zarr"), dirs_exist_ok=True)
        shutil.rmtree(zpath, ignore_errors=True)
    out = {"label": label, "vel_dir": vel_dir or di_default, "n_floats": n, "start": start,
           "days": days, "release": "spread" if spread else "start", "zarr_dir": zdir,
           "seconds": t2 - t0, "advect_s": t1 - t0, "profiles_s": t2 - t1,
           "copyback_s": time.perf_counter() - t2, "n_profiles": len(prof), "zarr_files": nfiles,
           "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           "node": os.uname().nodename}
    with open(os.path.join(BENCH_DIR, f"{label}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out)
    return out


def compare(a, b):
    pa, pb = (xr.open_zarr(os.path.join(BENCH_DIR, f"{x}.zarr")) for x in (a, b))
    la, lb = np.deg2rad(pa["lat"].values), np.deg2rad(pb["lat"].values)
    oa, ob = np.deg2rad(pa["lon"].values), np.deg2rad(pb["lon"].values)
    h = np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin((ob - oa) / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    d = pd.DataFrame({"day": np.tile(np.arange(km.shape[1]) * cs.CYCLE_DAYS, km.shape[0]),
                      "km": km.ravel()}).dropna()
    s = d.groupby("day")["km"].describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
    print(s.iloc[[1, 3, 9, 18, -1]].round(4).to_string() if len(s) > 18 else s.round(4).to_string())
    for x in (a, b):
        p = os.path.join(BENCH_DIR, f"{x}.json")
        if os.path.exists(p):
            print(json.load(open(p)))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--vel-dir", default=None)
    r.add_argument("--n-floats", type=int, default=1190)
    r.add_argument("--start", default="2015-01-01")
    r.add_argument("--days", type=int, default=365)
    r.add_argument("--spread", action="store_true", help="release uniformly over the window")
    r.add_argument("--zarr-dir", default=None, help="write trajectories here (e.g. node-local)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.label, a.vel_dir, a.n_floats, a.start, a.days, a.spread, a.zarr_dir)
    else:
        compare(a.a, a.b)
