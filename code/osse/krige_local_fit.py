"""Per-window MLE of Kuusela's locally stationary anisotropic exponential covariance.

Years are iid REPLICATES: the joint log-likelihood is the SUM of per-year
log-likelihoods, the covariance is block diagonal by year, no cross-year lag
enters, and th_t stays anchored to lags <= 2*lambda_t. (A single pooled fit over
the same field returns th_t = 154 d against an empirical e-fold of ~20 d.)
Blocks are small enough for an exact Cholesky -- no Vecchia. The GP is zero-mean:
the field is already a mean-removed residual.

Three things make this affordable:
  * numpy, not R. R here links its own reference BLAS, where a 3439x3439
    Cholesky takes 16.3 s against 0.31 s in numpy -- an 18x factor that no
    amount of optimizer tuning recovers.
  * phi profiled out analytically. With Sigma = phi*K(theta),
    phi_hat = sum_i z_i' K_i^-1 z_i / sum_i n_i, leaving 4 free parameters and
    removing the variance scale from the search.
  * warm starts along a contiguous run of windows, which overlap by ~95%.

    python code/osse/krige_local_fit.py <out.csv> <task> <ntask> [--sample-days 15]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg
from osse import krige_local as kl

LB = np.log([0.05, 0.05, 0.5, 1e-7])
UB = np.log([40.0, 40.0, 400.0, 10.0])
START = np.log([1.3, 1.0, 20.0, 0.05])


def _K(locs, r, tau2, want_parts=False):
    """exp(-d) + tau2*I with d the range-scaled Euclidean distance."""
    d2 = np.zeros((len(locs), len(locs)))
    parts = []
    for m in range(3):
        x = locs[:, m] / r[m]
        s = (x[:, None] - x[None, :]) ** 2
        d2 += s
        if want_parts:
            parts.append(s)
    d = np.sqrt(d2)
    K = np.exp(-d)
    K[np.diag_indices_from(K)] += tau2
    return K, d, parts          # parts is empty unless want_parts


def _obj(lp, blocks, grad=True):
    """Profiled negative log-likelihood (and gradient) on log parameters."""
    p = np.exp(lp)
    r, tau2 = p[:3], p[3]
    N = sum(len(b[1]) for b in blocks)
    logdet = 0.0
    q = 0.0
    cache = []
    for locs, z in blocks:
        K, d, parts = _K(locs, r, tau2, want_parts=grad)
        try:
            c = sla.cho_factor(K, lower=True)
        except sla.LinAlgError:
            return (1e10, np.zeros(4)) if grad else 1e10
        u = sla.cho_solve(c, z)
        logdet += 2.0 * np.log(np.diag(c[0])).sum()
        q += float(z @ u)
        if grad:
            cache.append((c, u, K, d, parts))
    if not np.isfinite(q) or q <= 0:
        return (1e10, np.zeros(4)) if grad else 1e10
    f = 0.5 * (N * np.log(q / N) + logdet + N)
    if not grad:
        return f

    g = np.zeros(4)
    for (c, u, K, d, parts), (locs, z) in zip(cache, blocks):
        Ki = sla.cho_solve(c, np.eye(len(z)))
        E = K.copy()
        E[np.diag_indices_from(E)] -= tau2          # exp(-d), nugget removed
        with np.errstate(invalid="ignore", divide="ignore"):
            inv_d = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        for m in range(3):
            dK = E * parts[m] * inv_d / r[m]        # dK/dr_m
            g[m] += 0.5 * (np.sum(Ki * dK) - (N / q) * float(u @ (dK @ u)))
        g[3] += 0.5 * (np.trace(Ki) - (N / q) * float(u @ u))   # dK/dtau2 = I
    return f, g * p


# Chosen by sweep, not by feel. The default (ftol 2.2e-9, gtol 1e-5) takes 15
# iterations; this takes 13 for a 0.63% max change in the estimates. Do NOT loosen
# further: the nugget is by far the slowest parameter to converge, so at ftol 1e-6
# the optimizer stops after ~5 iterations with the nugget still sitting at its
# WARM START (0.044 vs the converged 0.035 on a synthetic window whose truth was
# 0.050). That makes every cell's nugget depend on where the raster scan came
# from -- path-dependent parameters, not merely imprecise ones.
FTOL, GTOL = 1e-7, 1e-3


def fit_window(blocks, start=None, ftol=FTOL, gtol=GTOL):
    x0 = np.clip(START if start is None else start, LB + 1e-6, UB - 1e-6)
    res = minimize(_obj, x0, args=(blocks,), jac=True, method="L-BFGS-B",
                   bounds=list(zip(LB, UB)),
                   options={"maxiter": 200, "ftol": ftol, "gtol": gtol})
    p = np.exp(res.x)
    N = sum(len(b[1]) for b in blocks)
    qs = 0.0
    for locs, z in blocks:
        K = _K(locs, p[:3], p[3])[0]
        qs += float(z @ sla.cho_solve(sla.cho_factor(K, lower=True), z))
    return dict(phi=qs / N, th_lon=p[0], th_lat=p[1], th_t=p[2], nugget_ratio=p[3],
                nugget=(qs / N) * p[3], loglik=-res.fun, conv=int(res.status),
                nit=res.nit, n_obs=N), res.x


def main(out_csv, task, ntask, sample_days=kl.SAMPLE_DAYS):
    lat_ = pd.read_csv(os.path.join(cfg.OUT_DIR, "krige_local_lattice.csv"))
    cen = pd.read_csv(os.path.join(cfg.OUT_DIR, "krige_local_centres.csv"))
    if sample_days != kl.SAMPLE_DAYS:                 # thin the lattice further
        # Striding can only reach multiples of the lattice spacing; asking for 15 d
        # on a 10 d lattice silently computed a stride of 1 and thinned nothing.
        if sample_days % kl.SAMPLE_DAYS:
            raise ValueError(f"--sample-days must be a multiple of the lattice spacing "
                             f"({kl.SAMPLE_DAYS} d); rebuild the lattice for anything else")
        keep = np.sort(lat_["day"].unique())[:: sample_days // kl.SAMPLE_DAYS]
        lat_ = lat_[lat_["day"].isin(keep)]
    years = sorted(lat_["year"].unique())[1:]         # first year is the pad

    n = len(cen)
    lo, hi = n * task // ntask, n * (task + 1) // ntask
    mine = cen.iloc[lo:hi]                            # CONTIGUOUS: warm starts need neighbours
    print(f"task {task}/{ntask}: windows {lo}-{hi} ({len(mine)}), "
          f"lattice {len(lat_):,} rows, replicates {years}", flush=True)

    rows, warm, t0 = [], None, time.perf_counter()
    for k, (_, w) in enumerate(mine.iterrows(), 1):
        sub = lat_[(lat_["lat"].between(w.lat - kl.LAMBDA_G, w.lat + kl.LAMBDA_G)) &
                   (lat_["lon"].between(w.lon - kl.LAMBDA_G, w.lon + kl.LAMBDA_G))]
        blocks = []
        for y in years:
            tstar = (pd.Timestamp(f"{y}-{w.month:02d}-15") - pd.Timestamp("2010-01-01")).days
            d = sub[sub["day"].between(tstar - kl.LAMBDA_T, tstar + kl.LAMBDA_T)]
            if len(d) > 50:
                blocks.append((d[["lon", "lat", "day"]].to_numpy(), d["z"].to_numpy()))
        if len(blocks) < 2:
            continue
        try:
            out, warm = fit_window(blocks, warm)
        except Exception as e:
            print(f"  window {k} failed: {e}", flush=True); warm = None; continue
        rows.append({"box": w.box, "lat": w.lat, "lon": w.lon, "month": w.month,
                     "n_blocks": len(blocks), **out})
        if k % 10 == 0:
            print(f"  {k}/{len(mine)}  {(time.perf_counter()-t0)/k:.1f} s/window", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    el = time.perf_counter() - t0
    print(f"wrote {len(rows)} windows -> {out_csv} ({el/60:.1f} min, {el/max(len(mine),1):.1f} s/window)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv"); ap.add_argument("task", type=int); ap.add_argument("ntask", type=int)
    ap.add_argument("--sample-days", type=int, default=kl.SAMPLE_DAYS,
                    help="must be a multiple of the lattice spacing")
    a = ap.parse_args()
    main(a.out_csv, a.task, a.ntask, a.sample_days)
