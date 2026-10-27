"""Argo profile positions -> lag-matched transition pairs, launches and float lifetimes."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import argo_meta

EPOCH = pd.Timestamp("1990-01-01", tz="UTC")


def _days(t):
    return (pd.to_datetime(t, utc=True) - EPOCH).dt.total_seconds().to_numpy() / 86400.0


def profiles(path=None, dedupe_days=2.0, cutoff=None):
    """Raw Argo positions sorted by float and date, dropping fixes < dedupe_days after the previous raw fix."""
    d = argo_meta.read_argo(path)
    if cutoff is not None:
        d = d[d["date"] <= pd.Timestamp(cutoff, tz="UTC")]
    day = _days(d["date"])
    new = np.r_[True, d["float_id"].to_numpy()[1:] != d["float_id"].to_numpy()[:-1]]
    keep = new | (np.diff(day, prepend=-np.inf) >= dedupe_days)
    d = d[keep].assign(day=day[keep], lon=((d["lon"][keep] + 180.0) % 360.0) - 180.0)
    return d.reset_index(drop=True)


def pairs(d, lag_days, tol_days):
    """Every profile as a start, paired with the same float's profile nearest start + lag (within +/- tol)."""
    f = pd.factorize(d["float_id"])[0]
    day = d["day"].to_numpy()
    key = f * 1e5 + day
    i = np.arange(len(d))
    j = np.searchsorted(key, key + lag_days)
    cand = np.clip(np.stack([j - 1, j]), 0, len(d) - 1)
    best = cand[np.argmin(np.abs(key[cand] - (key + lag_days)), axis=0), i]
    lag = day[best] - day
    ok = (f[best] == f) & (best > i) & (np.abs(lag - lag_days) <= tol_days)
    a, b = i[ok], best[ok]
    return pd.DataFrame({"float_id": d["float_id"].to_numpy()[a],
                         "lat0": d["lat"].to_numpy()[a], "lon0": d["lon"].to_numpy()[a], "day0": day[a],
                         "lat1": d["lat"].to_numpy()[b], "lon1": d["lon"].to_numpy()[b], "day1": day[b]})


def launches(d, max_km0=50.0, max_km_per_day=20.0):
    """Launch per float: GDAC metadata when consistent with the first profile (same rule as
    continuous_sim.write_resample_pool), else the first profile."""
    f = d.groupby("float_id")[["lat", "lon", "date"]].first()
    m = argo_meta.read_launch().set_index("float_id").reindex(f.index)
    days = (f["date"] - m["launch_date"]).dt.total_seconds() / 86400.0
    la1, la2 = np.deg2rad(m["launch_lat"]), np.deg2rad(f["lat"])
    km = 6371.0 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2)
                                    * np.cos(np.deg2rad(f["lon"] - m["launch_lon"])), -1, 1))
    ok = ((days >= -1) & (km <= max_km0 + max_km_per_day * days.clip(lower=0))).to_numpy()
    date = m["launch_date"].where(ok, f["date"])
    return pd.DataFrame({"float_id": f.index, "lat": np.where(ok, m["launch_lat"], f["lat"]),
                         "lon": np.where(ok, m["launch_lon"], f["lon"]), "date": date.dt.tz_convert(None).to_numpy(),
                         "day": _days(date), "source": np.where(ok, "meta", "first_profile")})


def lifetimes(d, cutoff, censor_days):
    """Days from launch to last profile; a death is observed only if the last profile precedes cutoff - censor_days."""
    last = d.groupby("float_id")["day"].max()
    start = launches(d).set_index("float_id")["day"].reindex(last.index)
    end_ok = _days(pd.Series([pd.Timestamp(cutoff, tz="UTC")]))[0] - censor_days
    return pd.DataFrame({"duration": np.maximum(last - np.minimum(start, last), 0.0),
                         "event": last.to_numpy() < end_ok}, index=last.index)
