"""Stream GLORYS thetao month by month, collapse to layer OHC, discard the raw file.

Raw thetao over the analysis buffer is ~2.5 GB/month and ~300 GB over the record,
against ~15 GB for the four depth-layer integrals that are all any downstream step
uses. Each month is therefore fetched, integrated and deleted in turn. Resumable:
a month whose output already exists is skipped, and the download sidecar is kept
beside the output so provenance survives the raw file.

The depth request runs past ohc.OHC_LAYERS[-1] because PCHIP is local: integrating
to 2000 m needs one level beyond the pair bracketing it (2533.336 m).

    python code/osim/download_ohc_layers.py daily   --start 2010-01 --end 2019-12
    python code/osim/download_ohc_layers.py monthly --start 1993-01 --end 2019-12
"""

import argparse
import calendar
import os
import shutil
import sys
import time

import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ohc
from osim import download_glorys as dg
from osim import osim_config as cfg

LAT_BOUNDS = (14.0, 55.0)
LON_BOUNDS = (-72.0, -32.0)
MAX_DEPTH = 2560.0
OUT_DIR = os.path.join(cfg.GLORYS_DIR, "ohc_layers")
RAW_DIR = os.path.join(cfg.GLORYS_DIR, "_raw_tmp")
CADENCE = {"daily": (dg.DAILY_ID, "P1D"), "monthly": (dg.MONTHLY_ID, "P1M")}


def collapse(raw_path, out_path, layers=ohc.OHC_LAYERS, chunk=4, tile=256, floor=None, encoding=None):
    """Layer OHC (J/m2) from one raw thetao file, streamed to out_path chunk by chunk.

    Default storage is float32; an integer ``encoding`` needs the float64 values.
    """
    with xr.open_dataset(raw_path, chunks={"time": chunk, "depth": -1, "latitude": tile,
                                           "longitude": tile}) as ds:
        out = ohc.truth_ohc_field(ds, depths=(), layers=layers, floor=floor)
        if encoding is None:
            out, encoding = out.astype("float32"), {"zlib": True, "complevel": 4}
        out.attrs.update(source=os.path.basename(raw_path), layers=repr(tuple(layers)),
                         max_depth=MAX_DEPTH, cp=ohc.CP, rho=ohc.RHO)
        size = (min(chunk, ds.sizes["time"]), min(tile, ds.sizes["latitude"]),
                min(tile, ds.sizes["longitude"]))
        tmp = out_path + ".tmp"
        out.to_netcdf(tmp, encoding={v: {**encoding, "chunksizes": size} for v in out.data_vars})
    os.replace(tmp, out_path)
    return out_path


def run(cadence, start, end, lat_bounds=LAT_BOUNDS, lon_bounds=LON_BOUNDS,
        out_dir=OUT_DIR, raw_dir=RAW_DIR, keep_raw=False):
    dataset_id, _ = CADENCE[cadence]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    months = list(dg._month_range(start, end))
    print(f"{cadence}: {len(months)} months {start}..{end}, lat {lat_bounds}, lon {lon_bounds}, "
          f"depth 0-{MAX_DEPTH} m -> {out_dir}", flush=True)

    for i, (y, m) in enumerate(months, 1):
        out_path = os.path.join(out_dir, f"ohc_layers_{cadence}_{y:04d}_{m:02d}.nc")
        if os.path.exists(out_path):
            print(f"[{i}/{len(months)}] skip (exists) {os.path.basename(out_path)}", flush=True)
            continue
        raw_name = f"thetao_{cadence}_{y:04d}_{m:02d}.nc"
        # copernicusmarine appends "_(1)" rather than overwriting, so a raw file
        # left by a killed run would be collapsed instead of the fresh download.
        for stale in (os.path.join(raw_dir, raw_name), os.path.join(raw_dir, raw_name + ".json")):
            if os.path.exists(stale):
                os.remove(stale)
        t0 = time.perf_counter()
        raw = dg.fetch_subset(
            out_dir=raw_dir, out_file=raw_name, dataset_id=dataset_id,
            variables=["thetao"], lon_bounds=lon_bounds, lat_bounds=lat_bounds,
            depth_bounds=(0.0, MAX_DEPTH),
            start_date=f"{y:04d}-{m:02d}-01",
            end_date=f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}",
        )
        t1 = time.perf_counter()
        collapse(raw, out_path)
        t2 = time.perf_counter()

        if os.path.exists(raw + ".json"):
            shutil.move(raw + ".json", out_path + ".json")
        if not keep_raw:
            os.remove(raw)
        print(f"[{i}/{len(months)}] {y}-{m:02d}: fetch {t1-t0:6.1f}s  collapse {t2-t1:5.1f}s  "
              f"-> {os.path.getsize(out_path)/1e6:5.1f} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cadence", choices=sorted(CADENCE))
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--raw-dir", default=RAW_DIR, help="scratch for the discarded thetao files")
    a = ap.parse_args()
    run(a.cadence, a.start, a.end, raw_dir=a.raw_dir, keep_raw=a.keep_raw)
