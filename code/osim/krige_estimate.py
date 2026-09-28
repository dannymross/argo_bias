"""Simple-kriging estimate of the box-mean OHC anomaly, and its sampling bias.

Kuusela's estimator: predict on the 1 deg x 1 deg x 1 month grid with the locally
stationary anisotropic exponential covariance, then area-time average. Written as
a linear functional of the observed field,

    Zhat = sum_g c_g Zhat(x*_g) = sum_i a_i Z(x_i),   a_i = sum_g c_g alpha_i(x*_g),

so the estimator carries a signed measure Q = sum_i a_i delta_{x_i} and the bias
splits as E[B] = <Z - Zbar, E[Q]> + (m-1) Zbar with m = E[sum_i a_i]. Simple
kriging reverts to its zero prior mean where under-informed, so m <= 1 and the
second term is live -- unlike the simple mean, where m == 1 identically.

Two things make this affordable:
  * phi cancels. Sigma = phi(R + tau^2 I) and k* = phi rbar, so
    a = (R + tau^2 I)^{-1} rbar: the weights depend on the ranges and the nugget
    RATIO only, never on the sill.
  * Prediction points inside one 2 deg parameter cell share both theta and (by
    construction) their conditioning set, so their c_g-weighted cross-covariances
    collapse to ONE right-hand side per (cell, month) -- one Cholesky, not one
    per grid point.

    python code/osim/krige_estimate.py <out.csv> <task> <ntask> [--tag _pilot_wide]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.linalg as sla
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import continuous_sim as cs
from osim import osim_config as cfg

LAMBDA_G = 10.0     # conditioning half-width in space, degrees
# Half-width in time. +/-365 d was chosen before theta_t was known; at the fitted
# theta_t ~ 28 d that edge carries correlation 2e-6, so those profiles cost n^3 and
# contribute nothing. Measured against a +/-365 d reference on a box-B block:
# +/-180 d leaves m unchanged to 4 dp and Zhat to 7.5e-6 relative, for 4.5x the
# speed; +/-120 d already moves Zhat by 1.6e-3. Shrinking SPACE is the worse
# trade (+/-8 deg costs 4.8e-3), so the spatial half-width stays at 10 deg.
LAMBDA_T = 180
PARAM_DEG = 2.0
JITTER = 1e-6       # floor on the nugget ratio; the fitted nugget is ~0, which
                    # leaves K singular wherever two profiles nearly coincide
EPOCH = np.datetime64("2010-01-01")


def _corr(a, b, th):
    """exp(-d) between two point sets, d the range-scaled Euclidean distance."""
    d2 = np.zeros((len(a), len(b)))
    for m in range(3):
        d2 += ((a[:, m, None] - b[None, :, m]) / th[m]) ** 2
    return np.exp(-np.sqrt(d2))


def prediction_grid(box, year, area, deg=1.0):
    """1 deg x 1 month prediction points with their normalised area-time weights.

    c_g is the masked cell area times the month's day count, normalised over the
    box-year, so sum_g c_g = 1 and the grid average approximates the P-integral.
    """
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    a = area.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    lat, lon = a["latitude"].values, a["longitude"].values
    rows = []
    for cla in np.arange(la0 + deg / 2, la1, deg):
        for clo in np.arange(lo0 + deg / 2, lo1, deg):
            sub = a.sel(latitude=slice(cla - deg / 2, cla + deg / 2),
                        longitude=slice(clo - deg / 2, clo + deg / 2)).values
            w = np.nansum(sub)
            if w > 0:
                rows.append((float(cla), float(clo), float(w)))
    g = pd.DataFrame(rows, columns=["lat", "lon", "area"])
    out = []
    for m in range(1, 13):
        nd = (pd.Timestamp(year, m, 1) + pd.offsets.MonthEnd(0)).day
        mid = (pd.Timestamp(year, m, 15) - pd.Timestamp("2010-01-01")).days
        d = g.copy(); d["month"] = m; d["day"] = mid; d["w"] = d["area"] * nd
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out["c"] = out["w"] / out["w"].sum()
    return out


def aggregated_weights(prof_xyz, grid, params, jitter=JITTER):
    """a_i = sum_g c_g alpha_i(x*_g), accumulated over (parameter cell, month).

    Returns (a, n_solves, n_escalated): the weight per profile, how many linear
    systems were solved, and how many needed the jitter raised to factorise.
    """
    a = np.zeros(len(prof_xyz))
    nsolve = nesc = 0
    grid = grid.assign(clat=(np.floor((grid.lat - 22) / PARAM_DEG) * PARAM_DEG + 22 + 1),
                       clon=(np.floor((grid.lon + 60) / PARAM_DEG) * PARAM_DEG - 60 + 1))
    for (clat, clon, month), blk in grid.groupby(["clat", "clon", "month"], sort=False):
        p = params.iloc[np.argmin((params.lat - clat) ** 2 + (params.lon - clon) ** 2)]
        th = np.array([p.th_lon, p.th_lat, p.th_t])
        tau2 = max(float(p.nugget_ratio), jitter)
        day = blk["day"].iloc[0]
        sel = np.flatnonzero((np.abs(prof_xyz[:, 0] - clon) <= LAMBDA_G) &
                             (np.abs(prof_xyz[:, 1] - clat) <= LAMBDA_G) &
                             (np.abs(prof_xyz[:, 2] - day) <= LAMBDA_T))
        if sel.size < 5:
            continue
        X = prof_xyz[sel]
        R = _corr(X, X, th)
        R[np.diag_indices_from(R)] += tau2
        star = blk[["lon", "lat", "day"]].to_numpy(dtype=float)
        rbar = _corr(X, star, th) @ blk["c"].to_numpy()
        for k in range(6):                      # escalate jitter only if needed
            try:
                a[sel] += sla.cho_solve(sla.cho_factor(R, lower=True), rbar)
                break
            except sla.LinAlgError:
                R[np.diag_indices_from(R)] += 10.0 ** (-6 + k)
                nesc += 1
        nsolve += 1
    return a, nsolve, nesc


def run(tag="_pilot_wide", reps=None, boxes=("boxB", "boxA"), years=cfg.ANALYSIS_YEARS):
    params = pd.read_csv(os.path.join(cfg.OUT_DIR, "krige_plugin_oracle.csv"))
    z = xr.open_dataset(os.path.join(cfg.OUT_DIR, "residual_field_buffer.nc"))["z"]
    zt = pd.DatetimeIndex(z["time"].values)
    zlat, zlon = z["latitude"].values, z["longitude"].values
    zv = z.values                                # (time, lat, lon) float32, ~3.5 GB
    area = xr.open_dataset(os.path.join(cfg.OUT_DIR, "cell_area_buffer.nc"))["cell_area"]

    prof = cs.read_profiles(tag, ["rep", "lat", "lon", "time"])
    prof["time"] = pd.to_datetime(prof["time"])
    if reps is not None:
        prof = prof[prof["rep"].isin(reps)]

    # Z at every profile: containing cell, containing day. Positions off the mask
    # are dropped, never filled from a neighbour.
    it = zt.searchsorted(prof["time"].values, side="right") - 1
    ila = cfg.containing_index(prof["lat"].to_numpy(), zlat)
    ilo = cfg.containing_index(prof["lon"].to_numpy(), zlon)
    good = (it >= 0) & (it < len(zt)) & (ila >= 0) & (ilo >= 0)
    zi = np.full(len(prof), np.nan)
    zi[good] = zv[it[good], ila[good], ilo[good]]
    prof = prof.assign(z=zi, day=(prof["time"].values - EPOCH) / np.timedelta64(1, "D"))
    prof = prof[np.isfinite(prof["z"])].reset_index(drop=True)

    grids = {(b, y): prediction_grid(b, y, area) for b in boxes for y in years}
    zbar = {}
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        ii = (zlat >= la0) & (zlat <= la1); jj = (zlon >= lo0) & (zlon <= lo1)
        w = np.nan_to_num(area.values[np.ix_(ii, jj)])
        for y in years:
            k = np.flatnonzero(zt.year == y)
            sub = zv[np.ix_(k, ii, jj)]
            zbar[(b, y)] = float(np.nansum(sub * w) / (w.sum() * len(k)))

    rows = []
    for rep, pr in prof.groupby("rep", sort=True):
        xyz = pr[["lon", "lat", "day"]].to_numpy(dtype=float)
        zz = pr["z"].to_numpy()
        for b in boxes:
            for y in years:
                a, ns, ne = aggregated_weights(xyz, grids[(b, y)], params)
                zhat = float(a @ zz); m = float(a.sum())
                rows.append({"box": b, "rep": int(rep), "year": int(y),
                             "zhat_krige": zhat, "m": m, "zbar": zbar[(b, y)],
                             "B": zhat - zbar[(b, y)],
                             "B_align": zhat - m * zbar[(b, y)],
                             "B_cover": (m - 1) * zbar[(b, y)],
                             "n_cond": int(np.count_nonzero(a)),
                             "n_solve": ns, "n_jitter": ne})
        print(f"  rep {rep}: m = {np.mean([r['m'] for r in rows[-10:]]):.3f}", flush=True)
    return pd.DataFrame(rows)


def gather(tag="_pilot_wide"):
    """Concatenate the array-task output; partial arrays give a partial ensemble."""
    import glob
    fs = sorted(glob.glob(os.path.join(cfg.OUT_DIR, f"krige_est{tag}_t*.csv")))
    if not fs:
        raise FileNotFoundError(f"krige_est{tag}_t*.csv")
    d = pd.concat((pd.read_csv(f) for f in fs), ignore_index=True)
    out = os.path.join(cfg.OUT_DIR, f"krige_est{tag}.csv")
    d.to_csv(out, index=False)
    print(f"{len(fs)} task files -> {d['rep'].nunique()} reps, {len(d)} rows -> {out}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", nargs="?"); ap.add_argument("task", type=int, nargs="?")
    ap.add_argument("ntask", type=int, nargs="?")
    ap.add_argument("--tag", default="_pilot_wide")
    ap.add_argument("--gather", action="store_true")
    a = ap.parse_args()
    if a.gather:
        gather(a.tag); sys.exit(0)
    allreps = sorted(cs.read_profiles(a.tag, ["rep"])["rep"].unique())
    mine = allreps[a.task::a.ntask]
    print(f"task {a.task}/{a.ntask}: reps {mine}", flush=True)
    run(a.tag, reps=mine).to_csv(a.out_csv, index=False)
    print(f"wrote {a.out_csv}")
