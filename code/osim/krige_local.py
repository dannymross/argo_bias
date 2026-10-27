"""Locally stationary space-time covariance for the simple-kriging OHC estimator.

Kuusela & Stein's construction. For each centre (x*, y*, tau*) on the 1 deg x
1 deg x 1 month prediction grid, the window is

    W~ = [x* +/- lam_G] x [y* +/- lam_G] x [t* +/- lam_t],

and W collects W~ over every year at the same yearday. Within W the anomaly of
year i is a zero-mean GP with an anisotropic exponential covariance
k = phi_W exp(-d), d^2 = (dlon/th_lon)^2 + (dlat/th_lat)^2 + (dt/th_t)^2, plus an
iid nugget. Years are iid REPLICATES, not pooled: the joint log-likelihood is the
SUM of per-year log-likelihoods, the covariance is block diagonal by year, and no
cross-year lag ever enters. That is what keeps th_t anchored to within-window
lags -- a single pooled fit over the same data returns th_t = 154 d, against an
empirical e-fold of ~20 d, because it can see multi-year lags.

This is the ORACLE fit: the covariance comes from the complete truth field, so it
is fixed across deployment realizations and the kriging weights depend only on
design geometry. Refitting per realization from its own profiles is the separate
B_param arm.

Windows overlap by ~95%, so the field is subsampled to a 1 deg x 10 d lattice
ONCE and each window is a slice of it; reading per window would decompress
orders of magnitude more data.

    python code/osim/krige_local.py --prepare      # write the lattice + centres
    python code/osim/krige_local.py --gather       # concat, then median over tau*

Per Kuusela, the final plug-ins are the temporal MEDIAN of each parameter over the
monthly grid, so the fitted field is spatially varying but seasonally constant:
5,280 window fits collapse to 440 cell parameter sets.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg

LAMBDA_G = 10.0        # spatial half-width, degrees
LAMBDA_T = 45          # temporal half-width, days
GRID_DEG = 1.0         # PREDICTION grid (defines the estimator; do not coarsen)
PARAM_DEG = 2.0        # PARAMETER grid: where theta_W is estimated, then
                       # interpolated to the prediction grid. Kuusela estimates on
                       # the 1 deg grid; adjacent windows overlap ~95% (20 deg
                       # footprint, 1 deg apart) so the parameter field is very
                       # smooth spatially and 2 deg costs 4x fewer fits.
SAMPLE_DEG = 1.0       # within-window sampling
SAMPLE_DAYS = 10
EPOCH = np.datetime64("2010-01-01")


def lattice(tag="buffer", years=cfg.ANALYSIS_YEARS, deg=SAMPLE_DEG, days=SAMPLE_DAYS,
            pad=LAMBDA_T + 10):
    """Global subsample of the residual field: lon, lat, day, year, yearday, z.

    Padded `pad` days either side of the replicate years: a window centred in
    January reaches back into the previous December, since W~ is an interval in
    ABSOLUTE time, not in yearday. (The right pad past the last year does not
    exist, so December windows of the final year are truncated.)
    """
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{tag}.nc")) as ds:
        z = ds["z"]
        t = pd.DatetimeIndex(z["time"].values)
        lo = pd.Timestamp(f"{min(years)}-01-01") - pd.Timedelta(days=pad)
        hi = pd.Timestamp(f"{max(years)}-12-31") + pd.Timedelta(days=pad)
        sel = np.flatnonzero((t >= lo) & (t <= hi))
        sel = sel[:: max(1, days)]
        kla = int(round(deg / float(np.median(np.diff(z["latitude"].values)))))
        w = z.isel(time=sel, latitude=slice(None, None, kla), longitude=slice(None, None, kla))
        v = w.values
        lat, lon = w["latitude"].values, w["longitude"].values
        tt = pd.DatetimeIndex(w["time"].values)

    LO, LA = np.meshgrid(lon, lat)
    out = pd.DataFrame({
        "lon": np.tile(LO.ravel(), len(tt)),
        "lat": np.tile(LA.ravel(), len(tt)),
        "day": np.repeat((tt.values - EPOCH) / np.timedelta64(1, "D"), LO.size),
        "year": np.repeat(tt.year.to_numpy(), LO.size),
        "yearday": np.repeat(tt.dayofyear.to_numpy(), LO.size),
        "z": v.ravel(),
    }).dropna()
    # J/m2 are ~1e9; the optimizer is better conditioned on GJ.
    out["z"] *= cfg.__dict__.get("J_TO_GJ", 1e-9)
    return out.reset_index(drop=True)


def centres(boxes=("boxA", "boxB"), deg=PARAM_DEG, months=range(1, 13), bounds=None):
    """(box, lat, lon, month) of every PARAMETER grid cell; `bounds` {name: (lat, lon)} overrides the boxes."""
    rows = []
    bounds = bounds or {b: (cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]) for b in boxes}
    for b, ((la0, la1), (lo0, lo1)) in bounds.items():
        for la in np.arange(la0 + deg / 2, la1, deg):
            for lo in np.arange(lo0 + deg / 2, lo1, deg):
                for m in months:
                    rows.append({"box": b, "lat": round(float(la), 2),
                                 "lon": round(float(lo), 2), "month": int(m)})
    return pd.DataFrame(rows)


def prepare(tag="buffer", years=cfg.ANALYSIS_YEARS, param_deg=PARAM_DEG):
    lat_ = lattice(tag, years)
    cen = centres(deg=param_deg)
    p1 = os.path.join(cfg.OUT_DIR, "krige_local_lattice.csv")
    p2 = os.path.join(cfg.OUT_DIR, "krige_local_centres.csv")
    lat_.to_csv(p1, index=False)
    cen.to_csv(p2, index=False)
    print(f"lattice  {len(lat_):,} rows ({os.path.getsize(p1)/1e6:.1f} MB), "
          f"{lat_['year'].nunique()} years x {lat_['day'].nunique()} days x "
          f"{lat_[['lat','lon']].drop_duplicates().shape[0]:,} cells -> {p1}")
    print(f"centres  {len(cen):,} windows -> {p2}")
    return lat_, cen


PARS = ["phi", "th_lon", "th_lat", "th_t", "nugget_ratio", "nugget"]


def plugin_parameters(out_tag="oracle", save=True):
    """Kuusela's final plug-in estimates: the temporal MEDIAN over the monthly grid.

    theta_hat(x*, y*) = med_{tau*} theta_hat_W(x*, y*, tau*), collapsing the 12
    seasonal fits per cell to one non-seasonal parameter set. The median (not the
    mean) is what gives robustness against individual window fits that fail or run
    to a bound -- a cell survives as long as most of its months are sane. It is
    invariant under monotone transforms, so log vs natural scale is immaterial.

    Also returns a seasonal-spread diagnostic. Kuusela reports the seasonal effect
    as small; that is an empirical finding in their setting, so it is measured here
    rather than assumed. Note the 12 monthly fits per cell are NOT independent --
    consecutive windows are 90 d wide and 30 d apart, sharing ~2/3 of their data --
    so the spread understates true seasonal variation and the median buys
    robustness rather than a factor-of-sqrt(12) in precision.
    """
    d = pd.read_csv(os.path.join(cfg.OUT_DIR, f"krige_local_{out_tag}.csv"))
    g = d.groupby(["box", "lat", "lon"])
    med = g[PARS].median().reset_index()
    med["n_months"] = g.size().values
    spread = (g[PARS].quantile(0.75).values - g[PARS].quantile(0.25).values) / \
             np.maximum(np.abs(g[PARS].median().values), 1e-30)
    diag = pd.DataFrame(spread, columns=[f"iqr_over_med_{c}" for c in PARS])
    print("seasonal spread per cell, IQR/median over the 12 monthly fits:")
    print(diag.describe().loc[["25%", "50%", "75%"]].round(3).to_string())
    if save:
        out = os.path.join(cfg.OUT_DIR, f"krige_plugin_{out_tag}.csv")
        med.to_csv(out, index=False)
        print(f"\n{len(med)} cells (from {len(d)} windows) -> {out}")
    return med, diag


def gather(out_tag="oracle"):
    import glob
    fs = sorted(glob.glob(os.path.join(cfg.OUT_DIR, f"krige_local_{out_tag}_t*.csv")))
    if not fs:
        raise FileNotFoundError(f"krige_local_{out_tag}_t*.csv")
    d = pd.concat((pd.read_csv(f) for f in fs), ignore_index=True)
    out = os.path.join(cfg.OUT_DIR, f"krige_local_{out_tag}.csv")
    d.to_csv(out, index=False)
    print(f"{len(fs)} task files -> {len(d):,} windows -> {out}")
    print(d[["phi", "th_lon", "th_lat", "th_t", "nugget"]].describe().round(3).to_string())
    plugin_parameters(out_tag)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--gather", action="store_true")
    ap.add_argument("--tag", default="buffer")
    ap.add_argument("--out-tag", default="oracle")
    ap.add_argument("--param-deg", type=float, default=PARAM_DEG,
                    help="parameter-grid spacing (the prediction grid stays 1 deg)")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.tag, param_deg=a.param_deg)
    elif a.gather:
        gather(a.out_tag)
    else:
        ap.error("pass --prepare or --gather")
