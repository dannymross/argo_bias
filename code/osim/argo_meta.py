"""Argo float launch metadata (GDAC <WMO>_meta.nc): true deployment position and time.

Only floats present in data/argo_ohc.csv are fetched: a float with no profile there
contributes nothing to the profile intensity, so it cannot affect q-hat.

    python code/osim/argo_meta.py [--pad 3] [--start 2009-10-01] [--end 2020-01-31]
"""

import argparse
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import deployment as di
from osim import osim_config as cfg

GDAC = "https://data-argo.ifremer.fr"
META_DIR = os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_meta")
LAUNCH_CSV = os.path.join(META_DIR, "launch.csv")


def read_argo(path=None):
    """Raw profile record: every row of argo_ohc.csv with a position and date."""
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date", "max_depth"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    return d.dropna(subset=["lat", "lon", "date"]).sort_values(["float_id", "date"])


def candidates(d, pad=3.0, start="2009-10-01", end="2020-01-31"):
    """Floats whose first profile lies within the velocity domain padded by `pad` deg
    and inside [start, end]; the launch point is within ~1 cycle of the first profile."""
    (la0, la1), (lo0, lo1) = di.velocity_extent()
    f = d.groupby("float_id").first()
    ok = (f["lat"].between(la0 - pad, la1 + pad) & f["lon"].between(lo0 - pad, lo1 + pad)
          & f["date"].between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")))
    return f.index[ok].to_numpy()


def meta_index():
    txt = subprocess.run(["curl", "-sf", f"{GDAC}/ar_index_global_meta.txt"],
                         capture_output=True, text=True, check=True).stdout
    idx = pd.read_csv(io.StringIO(txt), comment="#")
    idx["float_id"] = idx["file"].str.split("/").str[1].astype(int)
    return idx.drop_duplicates("float_id").set_index("float_id")["file"]


def download(wmos, workers=8):
    os.makedirs(META_DIR, exist_ok=True)
    idx = meta_index()

    def one(w):
        out = os.path.join(META_DIR, f"{w}_meta.nc")
        if os.path.exists(out) or w not in idx.index:
            return
        tmp = out + ".tmp"
        if subprocess.run(["curl", "-sf", "-o", tmp, f"{GDAC}/dac/{idx[w]}"]).returncode == 0:
            os.replace(tmp, out)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, wmos))
    return sum(os.path.exists(os.path.join(META_DIR, f"{w}_meta.nc")) for w in wmos)


def _text(v):
    v = v.item() if hasattr(v, "item") else v
    return (v.decode(errors="ignore") if isinstance(v, bytes) else str(v)).strip()


def launch_table(wmos):
    """float_id, launch_lat, launch_lon, launch_date; NaN where the file is missing or
    its launch fields are unparseable."""
    rows = []
    for w in wmos:
        p = os.path.join(META_DIR, f"{w}_meta.nc")
        r = {"float_id": int(w), "launch_lat": np.nan, "launch_lon": np.nan, "launch_date": pd.NaT}
        if os.path.exists(p):
            try:
                with xr.open_dataset(p, decode_times=False) as m:
                    lat, lon = float(m["LAUNCH_LATITUDE"]), float(m["LAUNCH_LONGITUDE"])
                    t = pd.to_datetime(_text(m["LAUNCH_DATE"].values), format="%Y%m%d%H%M%S", utc=True)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    r.update(launch_lat=lat, launch_lon=lon, launch_date=t)
            except Exception:
                pass
        rows.append(r)
    return pd.DataFrame(rows)


def build(pad=3.0, start="2009-10-01", end="2020-01-31"):
    d = read_argo()
    wmos = candidates(d, pad, start, end)
    n = download(wmos)
    lt = launch_table(wmos)
    lt.to_csv(LAUNCH_CSV, index=False)
    print(f"{len(wmos)} candidate floats, {n} meta files, "
          f"{lt['launch_date'].notna().sum()} parsed launches -> {LAUNCH_CSV}")
    return lt


def read_launch(path=LAUNCH_CSV):
    lt = pd.read_csv(path)
    lt["launch_date"] = pd.to_datetime(lt["launch_date"], format="ISO8601", utc=True)
    return lt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("--start", default="2009-10-01")
    ap.add_argument("--end", default="2020-01-31")
    a = ap.parse_args()
    build(a.pad, a.start, a.end)
