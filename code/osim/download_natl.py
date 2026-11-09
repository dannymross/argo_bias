"""GLORYS for the deep North Atlantic, month by month, resumable.

Raw thetao and uo/vo stay on scratch (cfg.NATL_RAW); OHC layers and the 1000 m
velocity go to cfg.NATL_DIR. A month whose output exists is skipped, and a raw file
is renamed into place only once complete, so a re-submit fills gaps. The monthly
cadence writes layers only, for the mean-field fit.

Run the pilot first: it writes the static grid the array tasks read.

    python code/osim/download_natl.py --start 2015-01 --end 2015-01 [--check]
    python code/osim/download_natl.py --cadence monthly --start 1993-01 --end 2025-12 [--check]
"""

import argparse
import calendar
import glob
import os
import resource
import shutil
import sys
import time

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ohc
from osim import convert_velocity as cv
from osim import download_glorys as dg
from osim import download_ohc_layers as dl
from osim import osim_config as cfg

LAT, LON = cfg.NATL_BOX["lat_bounds"], cfg.NATL_BOX["lon_bounds"]
LAYER_DIR = os.path.join(cfg.GLORYS_DIR, cfg.LAYER_DIR)
OLD_LAYER_DIR = os.path.join(cfg.GLORYS_DIR, "ohc_layers")  # buffer box, four layers, 1993-2019 monthly
VEL_DIR = os.path.join(cfg.NATL_DIR, "velocity_1000")
STATIC_DIR = os.path.join(cfg.NATL_DIR, "static")
STATIC_PARTS = {"bathy": ["deptho", "deptho_lev", "mask"], "coords": ["e1t", "e2t", "e3t"]}
LAYER_ENC = dict(dtype="int32", scale_factor=1e5, add_offset=0.0, _FillValue=-2147483647,
                 zlib=True, complevel=4, shuffle=True)  # 1e5 J/m2 = 1e-4 GJ/m2
VEL_DIGITS = 4


def log(stage, t0):
    print(f"{time.strftime('%H:%M:%S')} {stage}: {time.perf_counter() - t0:.0f} s, "
          f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)


def seafloor():
    b = xr.open_dataset(os.path.join(STATIC_DIR, "natl_bathy.nc")).deptho.squeeze(drop=True).load()
    return b.sortby("latitude")


def fetch_raw(name, y, m, variables, depth_bounds, level, dataset_id=dg.DAILY_ID):
    """Complete raw file for one month, fetched under a temporary name if missing."""
    path = os.path.join(cfg.NATL_RAW, name)
    if os.path.exists(path):
        return path
    part = f"part_{name}"
    for f in glob.glob(os.path.join(cfg.NATL_RAW, part + "*")):
        os.remove(f)
    t0 = time.perf_counter()
    got = dg.fetch_subset(out_dir=cfg.NATL_RAW, out_file=part, dataset_id=dataset_id,
                          variables=variables, lon_bounds=LON, lat_bounds=LAT,
                          depth_bounds=depth_bounds, start_date=f"{y:04d}-{m:02d}-01",
                          end_date=f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}",
                          compression_level=level)
    os.replace(got + ".json", path + ".json")
    os.replace(got, path)
    log(f"fetch {name} {os.path.getsize(path) / 1e9:.2f} GB", t0)
    return path


def month(y, m, floor, cadence="daily"):
    name = f"{y:04d}_{m:02d}.nc"
    layers = os.path.join(LAYER_DIR, f"ohc_layers_{cadence}_{name}")
    vel = os.path.join(VEL_DIR, f"velocity_daily_park_{name}")
    if not os.path.exists(layers):
        raw = fetch_raw(f"thetao_{name}" if cadence == "daily" else f"thetao_{cadence}_{name}", y, m,
                        ["thetao"], (0.0, dl.MAX_DEPTH), 1, dl.CADENCE[cadence][0])
        t0 = time.perf_counter()
        with xr.open_dataset(raw) as r:
            f = floor.reindex(latitude=r.latitude, longitude=r.longitude, method="nearest", tolerance=1e-3)
        dl.collapse(raw, layers, layers=ohc.OHC_LAYERS_700, floor=f, encoding=LAYER_ENC)
        shutil.copy(raw + ".json", layers + ".json")
        log(f"layers {os.path.basename(layers)} {os.path.getsize(layers) / 1e6:.0f} MB", t0)
    if cadence == "daily" and not os.path.exists(vel):
        raw = fetch_raw(f"uv_{name}", y, m, ["uo", "vo"], (900.0, 1100.0), 4)
        t0 = time.perf_counter()
        cv.convert(raw, vel, complevel=1, digits=VEL_DIGITS)
        shutil.copy(raw + ".json", vel + ".json")
        log(f"velocity {os.path.basename(vel)} {os.path.getsize(vel) / 1e6:.0f} MB", t0)


def check(y, m, cadence="daily"):
    """Max abs difference from the old stores over their extent and valid cells."""
    name = f"{y:04d}_{m:02d}.nc"
    pa = os.path.join(OLD_LAYER_DIR, f"ohc_layers_{cadence}_{name}")
    pv = os.path.join(cfg.GLORYS_DIR, "velocity_daily_1000", f"velocity_daily_park_{name}")
    if not os.path.exists(pa):
        print(f"{y}-{m:02d}: no old {cadence} layers to compare", flush=True)
        return
    with xr.open_dataset(pa) as a, xr.open_dataset(os.path.join(LAYER_DIR, f"ohc_layers_{cadence}_{name}")) as n:
        n = n.sel(latitude=a.latitude, longitude=a.longitude, method="nearest", tolerance=1e-3)
        n = n.assign_coords(latitude=a.latitude, longitude=a.longitude)
        pairs = {"ohc_0_15": n.ohc_0_15, "ohc_15_975": n.ohc_15_700 + n.ohc_700_975,
                 "ohc_975_1850": n.ohc_975_1850, "ohc_1850_2000": n.ohc_1850_2000}
        for v, b in pairs.items():
            d = abs(b - a[v]).where(a[v].notnull()) / 1e9
            print(f"{y}-{m:02d} {v}: max {float(d.max()):.2e} GJ/m2, old valid and new NaN "
                  f"{int((a[v].notnull() & b.isnull()).sum())}, new cells gained "
                  f"{int((a[v].isnull() & b.notnull()).sum() / a.sizes['time'])}", flush=True)
    if cadence != "daily" or not os.path.exists(pv):
        return
    with xr.open_dataset(pv) as a, xr.open_dataset(os.path.join(VEL_DIR, f"velocity_daily_park_{name}")) as n:
        n = n.sel(latitude=a.latitude, longitude=a.longitude, method="nearest", tolerance=1e-3)
        for v in ("uo", "vo"):
            print(f"{y}-{m:02d} {v}: max {float(np.abs(n[v].values - a[v].values).max()):.2e} m/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--cadence", choices=sorted(dl.CADENCE), default="daily")
    ap.add_argument("--check", action="store_true", help="also compare with the old stores")
    a = ap.parse_args()
    dask.config.set(scheduler="threads", num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    for d in (cfg.NATL_RAW, LAYER_DIR, VEL_DIR):
        os.makedirs(d, exist_ok=True)
    dg.fetch_static(STATIC_DIR, LON, LAT, STATIC_PARTS, depth_bounds=(0.0, dl.MAX_DEPTH), file_prefix="natl")
    floor = seafloor()
    for y, m in dg._month_range(a.start, a.end):
        month(y, m, floor, a.cadence)
        if a.check:
            check(y, m, a.cadence)
    print("done", flush=True)
