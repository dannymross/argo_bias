"""DUACS L4 altimetry (altimeter-only; SLA and its geostrophic anomaly velocities, no MDT) as 10-day means at
native 0.125 deg on bins aligned with the transition-matrix steps (1998-01-01 + 10 k days). Each chunk downloads its
daily fields to scratch, aggregates, writes one compressed file and deletes the raw download.

Stored per bin: means of sla, ugosa, vgosa; eke = mean of (ugosa^2 + vgosa^2) / 2; grad_sla = mean of |grad sla|
(centred differences on the sphere); sla2 = mean of sla^2 (block variances follow exactly from sla2 and sla);
n_valid = valid sla days. All second-moment fields are averaged linearly, so any coarser block is exact.

    python code/sat/ssh.py --chunk 0          # one array task
    python code/sat/ssh.py --n-chunks         # print the number of chunks
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import download_glorys as dg

DATASET = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
VARS = ("sla", "ugosa", "vgosa")
LAT, LON = (0.0, 66.0), (-100.0, 0.0)
PAD = 0.25
T0, STEP = pd.Timestamp("1998-01-01"), 10
START, END = pd.Timestamp("2003-07-01"), pd.Timestamp("2025-12-31")
BINS_PER_CHUNK = 36
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "sat", "ssh_natl_10d")
RAW = os.path.join("/scratch", os.environ.get("USER", "user"), "duacs_raw")
R_EARTH = 6371e3


def edges():
    k0 = int(np.floor((START - T0).days / STEP))
    k1 = int(np.ceil((END + pd.Timedelta(days=1) - T0).days / STEP))
    return T0 + pd.to_timedelta(np.arange(k0, k1 + 1) * STEP, "D")


def chunks():
    e = edges()
    return [e[i:i + BINS_PER_CHUNK + 1] for i in range(0, len(e) - 1, BINS_PER_CHUNK)]


def grad_mag(sla):
    """|grad sla| (m/m) by centred differences in lat/lon radians on the sphere; NaN next to masked cells."""
    phi, lam = np.radians(sla.latitude.values), np.radians(sla.longitude.values)
    dy, dx = np.gradient(sla.values, phi, lam, axis=(1, 2))
    return np.hypot(dx / (R_EARTH * np.cos(phi)[None, :, None]), dy / R_EARTH)


def aggregate(ds, e):
    """10-day bin means of the stored fields from daily data."""
    sla, u, v = (ds[k] for k in VARS)
    daily = xr.Dataset(dict(sla=sla, ugosa=u, vgosa=v, eke=0.5 * (u ** 2 + v ** 2),
                            grad_sla=(sla.dims, grad_mag(sla)), sla2=sla ** 2))
    b = xr.DataArray(np.searchsorted(e.values, ds.time.values, side="right") - 1, dims="time")
    m = daily.groupby(b.rename("bin")).mean(skipna=True)
    n = sla.notnull().groupby(b.rename("bin")).sum().astype(np.int8).rename("n_valid")
    out = xr.merge([m.astype(np.float32), n]).sel(bin=np.arange(len(e) - 1))
    out = out.assign_coords(time=("bin", e[:-1].values), time_end=("bin", e[1:].values)).swap_dims(bin="time")
    return out.drop_vars("bin").sel(latitude=slice(*LAT), longitude=slice(*LON))


def run(i):
    e = chunks()[i]
    out = os.path.join(OUT, f"ssh_{e[0]:%Y%m%d}.nc")
    if os.path.exists(out):
        return out
    raw = os.path.join(RAW, f"c{i:03d}")
    f = dg.fetch_subset(raw, "raw.nc", DATASET, VARS, (LON[0] - PAD, LON[1] + PAD), (LAT[0] - PAD, LAT[1] + PAD),
                        start_date=f"{e[0]:%Y-%m-%d}", end_date=f"{e[-1] - pd.Timedelta(days=1):%Y-%m-%d}")
    with xr.open_dataset(f) as ds:
        a = xr.concat([aggregate(ds.sel(time=slice(t0, t1 - pd.Timedelta(seconds=1))).load(), pd.DatetimeIndex([t0, t1]))
                       for t0, t1 in zip(e[:-1], e[1:])], "time")
    a.attrs.update(source=DATASET, bins=f"{STEP} d from {T0:%Y-%m-%d}", note="altimeter-only; no MDT")
    os.makedirs(OUT, exist_ok=True)
    a.to_netcdf(out + ".tmp", encoding={k: dict(zlib=True, complevel=4) for k in a.data_vars})
    os.replace(out + ".tmp", out)
    shutil.rmtree(raw)
    return out


def local_anom(x, half_deg):
    """x minus its (2 half_deg)-square boxcar mean over valid cells, per time slice."""
    k = (1, 2 * int(round(half_deg * 8)) + 1, 2 * int(round(half_deg * 8)) + 1)
    ok = np.isfinite(x)
    num, den = (uniform_filter(a, k, mode="constant") for a in (np.where(ok, x, 0.0), ok.astype(float)))
    return x - num / np.where(den > 0, den, np.nan)


def at(lat, lon, time, var=("sla", "eke", "grad_sla"), half_deg=()):
    """Bin values at points (bin containing time, nearest 0.125 deg cell); NaN outside the record. For each h in
    half_deg also sla_loc{h}: sla minus its local boxcar mean."""
    t = pd.DatetimeIndex(time).values
    out = {v: np.full(len(t), np.nan, np.float32) for v in var}
    for f in sorted(os.listdir(OUT)):
        with xr.open_dataset(os.path.join(OUT, f)) as ds:
            m = (t >= ds.time.values[0]) & (t < ds.time_end.values[-1])
            if not m.any():
                continue
            k = np.searchsorted(ds.time.values, t[m], side="right") - 1
            i = np.clip(np.round((np.asarray(lat)[m] - ds.latitude.values[0]) * 8).astype(int), 0, ds.latitude.size - 1)
            j = np.clip(np.round((np.asarray(lon)[m] - ds.longitude.values[0]) * 8).astype(int), 0, ds.longitude.size - 1)
            for v in var:
                out[v][m] = ds[v].values[k, i, j]
            for h in half_deg:
                out.setdefault(f"sla_loc{h:g}", np.full(len(t), np.nan, np.float32))[m] = \
                    local_anom(ds["sla"].values.astype(float), h)[k, i, j]
    return out


def mdt_at(lat, lon, path=os.path.join(REPO, "data", "sat", "mdt_natl.nc")):
    """CNES-CLS mean dynamic topography (m; same 1993-2012 reference as DUACS sla) at the nearest 0.125 deg cell."""
    with xr.open_dataset(path) as ds:
        m = ds.mdt.isel(time=0)
        i = np.clip(np.round((np.asarray(lat) - float(m.latitude[0])) * 8).astype(int), 0, m.latitude.size - 1)
        j = np.clip(np.round((np.asarray(lon) - float(m.longitude[0])) * 8).astype(int), 0, m.longitude.size - 1)
        return m.values[i, j]


def climatology(var="eke"):
    """Time mean of a stored field over the whole record (bins weighted equally)."""
    fs = sorted(os.listdir(OUT))
    tot = None
    for f in fs:
        with xr.open_dataset(os.path.join(OUT, f)) as ds:
            x = ds[var].sum("time", skipna=True), ds[var].notnull().sum("time")
            tot = x if tot is None else (tot[0] + x[0], tot[1] + x[1])
    return (tot[0] / tot[1]).rename(var)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--n-chunks", action="store_true")
    a = ap.parse_args()
    if a.n_chunks:
        print(len(chunks()))
    else:
        print(run(a.chunk))
