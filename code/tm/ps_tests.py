"""Preferential-sampling detection on real Argo against the altimetric field.

Case-control intensity test: for each profile, K controls uniform in area within its deg x deg box at the same
10-day bin; the conditional logit of case vs controls on satellite covariates g estimates beta in the within-box
intensity exp(beta' g) (box x bin effects cancel). Placebos repeat it with the covariates taken at shifted times.
Mark-location test: own-float-out residual on log local density of other floats with box x period effects.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sat import ssh
from tm import validate as v

R_EARTH = 6371.0


def controls(lat, lon, deg=4.0, K=10, seed=0):
    """K points uniform in area in each case's deg x deg box: (n, K) lat, lon."""
    rng = np.random.default_rng(seed)
    la0, lo0 = np.floor(lat / deg)[:, None] * deg, np.floor(lon / deg)[:, None] * deg
    s0, s1 = np.sin(np.radians(la0)), np.sin(np.radians(la0 + deg))
    la = np.degrees(np.arcsin(s0 + (s1 - s0) * rng.random((len(lat), K))))
    return la, lo0 + deg * rng.random((len(lat), K))


def covariates(lat, lon, time, half_deg=1.0):
    a = ssh.at(lat, lon, time, var=("sla", "eke", "grad_sla"), half_deg=(half_deg,))
    s = a[f"sla_loc{half_deg:g}"] * 100
    return pd.DataFrame({"sla_loc": s, "sla_loc2": s ** 2, "log_eke": np.log10(a["eke"] * 1e4),
                         "grad": a["grad_sla"] * 1e7, "sla": a["sla"] * 100,
                         "adt": (a["sla"] + ssh.mdt_at(lat, lon)) * 100})


def sample(p, deg=4.0, K=10, shift_days=0, seed=0, half_deg=1.0):
    """Covariates at cases (n, m) and controls (n, K, m), optionally at time + shift_days."""
    t = p["time"].to_numpy() + np.timedelta64(int(shift_days), "D")
    cl, co = controls(p.lat.to_numpy(), p.lon.to_numpy(), deg, K, seed)
    gc = covariates(p.lat.to_numpy(), p.lon.to_numpy(), t, half_deg)
    gk = covariates(cl.ravel(), co.ravel(), np.repeat(t, K), half_deg)
    return gc, gk.to_numpy().reshape(len(p), K, -1)


def clogit(gc, gk, cluster, cols, it=30):
    """1-case-K-control conditional logit by Newton; CR1 sandwich clustered on `cluster`. Invalid controls dropped."""
    j = [list(gc.columns).index(c) for c in cols]
    X0, Xk = gc.to_numpy()[:, j], gk[:, :, j]
    ok = np.isfinite(X0).all(1) & (np.isfinite(Xk).all(2).sum(1) >= 2)
    X0, Xk, cl = X0[ok], Xk[ok], np.asarray(cluster)[ok]
    mk = np.isfinite(Xk).all(2)
    X = np.concatenate([X0[:, None], np.where(mk[..., None], Xk, 0.0)], 1)
    m = np.concatenate([np.ones((len(X0), 1), bool), mk], 1)
    mu, sd = X0.mean(0), X0.std(0)
    Z = (X - mu) / sd
    b = np.zeros(len(j))
    for _ in range(it):
        e = np.where(m, Z @ b, -np.inf)
        w = np.exp(e - e.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        zbar = np.einsum("ik,ikp->ip", w, Z)
        U = Z[:, 0] - zbar
        H = np.einsum("ik,ikp,ikq->pq", w, Z, Z) - zbar.T @ zbar
        step = np.linalg.solve(H, U.sum(0))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    S = pd.DataFrame(U).groupby(cl).sum().to_numpy()
    G = len(S)
    Hi = np.linalg.inv(H)
    V = Hi @ (S.T @ S) @ Hi * G / (G - 1)
    se = np.sqrt(np.diag(V))
    d = X0 - np.nanmean(np.where(mk[..., None], Xk, np.nan), 1)
    return pd.DataFrame({"beta": b / sd, "se": se / sd, "z": b / se, "diff": d.mean(0),
                         "se_diff": pd.DataFrame(d - d.mean(0)).groupby(cl).sum().pow(2).sum().pow(0.5).to_numpy() / len(d)},
                        index=cols).assign(n=len(X0), clusters=G)


def _xyzt(p, km_per_day):
    la, lo = np.radians(p.lat.to_numpy()), np.radians(p.lon.to_numpy())
    t = (p.time.to_numpy() - np.datetime64("2000-01-01")) / np.timedelta64(1, "D")
    return np.column_stack([R_EARTH * np.cos(la) * np.cos(lo), R_EARTH * np.cos(la) * np.sin(lo),
                            R_EARTH * np.sin(la), t * km_per_day])


def density(p, R=100.0, T=15.0):
    """Profiles of other floats within the (R km, T days) ellipsoid around each profile."""
    X = _xyzt(p, R / T)
    allc = cKDTree(X).query_ball_point(X, R, return_length=True)
    fid = pd.factorize(p.float_id)[0][:, None] * 1e8
    own = cKDTree(np.column_stack([X, fid])).query_ball_point(np.column_stack([X, fid]), R, return_length=True)
    return allc - own


def mark_test(p, x, deg=4.0, years=3, cluster="blk"):
    """Slope of r on x with deg box x period fixed effects; CR1 SE clustered on `cluster`."""
    g = [v.stratum(p, deg) * 100 + (p.time.dt.year.to_numpy() - 2004) // years]
    Z = v.demean(np.column_stack([p.r.to_numpy(), np.asarray(x, float)]), g, np.ones(len(p)))
    y, z = Z[:, 0], Z[:, 1]
    b = np.sum(z * y) / np.sum(z * z)
    u = pd.Series(z * (y - b * z)).groupby(p[cluster].to_numpy()).sum().to_numpy()
    se = np.sqrt(np.sum(u * u) * len(u) / (len(u) - 1)) / np.sum(z * z)
    return dict(slope=b, se=se, t=b / se, sd_x=np.std(z))


REGIONS = {"Gulf Stream": ((33, 42), (-78, -50)), "NAC": ((42, 53), (-50, -25)),
           "Subpolar": ((53, 66), (-65, -5)), "Subtropical E": ((15, 33), (-50, -10)),
           "Subtropical W": ((15, 33), (-78, -50)), "Tropical": ((0, 15), (-60, -10))}


def region(lat, lon):
    out = np.full(len(lat), "other", object)
    for k, ((a, b), (c, d)) in REGIONS.items():
        out[(lat >= a) & (lat < b) & (lon >= c) & (lon < d)] = k
    return out


def box_eke(p, deg=4.0):
    """Record-mean EKE (cm2/s2) averaged over each profile's deg x deg box (ocean cells)."""
    e = ssh.climatology("eke") * 1e4
    b = e.coarsen(latitude=int(deg * 8), longitude=int(deg * 8), boundary="trim").mean()
    i = np.clip(((p.lat.to_numpy() - float(e.latitude[0]) + 1 / 16) // deg).astype(int), 0, b.latitude.size - 1)
    j = np.clip(((p.lon.to_numpy() - float(e.longitude[0]) + 1 / 16) // deg).astype(int), 0, b.longitude.size - 1)
    return b.values[i, j]


def design(g, beke, form="log", thresh=200.0):
    """Tilt covariates from case (n, m) or control (n, K, m) arrays of `covariates` columns: local SLA and its
    interaction with box EKE (log10 EKE/thresh, or the indicator EKE >= thresh). Box effects cancel in the logit."""
    s = g[..., 0]
    h = np.log10(beke / thresh) if form == "log" else (beke >= thresh).astype(float)
    h = np.asarray(h).reshape((-1,) + (1,) * (s.ndim - 1))
    return np.stack([s, s * h], -1)


def tilt_weights(beta, Xc, Xk):
    """w_i = mean_k exp(beta' x_ik) / exp(beta' x_i): inverse within-box tilt, normalised by the box's own mean."""
    ek = np.exp(np.where(np.isfinite(Xk).all(-1), Xk @ beta, np.nan))
    return np.nanmean(ek, 1) / np.exp(Xc @ beta)
