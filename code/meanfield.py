"""Toy comparison of mean-field fits (global OLS / GLS + log q-hat, local RG/KS windows) ahead of kriging
(reports/meanfield.qmd)."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import csr_matrix

import simex as sx

YEAR, A_SEAS, T_PEAK = 365.0, 0.3, 60.0
MEANDER, K_M, ZONAL = 0.3, 2 * sx.K_N[0], 0.5
SPIN_C, N_SNAP = 37, 146
H, M_PART, LAM_T, TARGET_EVERY = 4, 200, 30.0, 3
M_BIG = 4000
BOUND_TERMS = ("cbias", "R", "Rpool", "Rsys", "Rmc", "Q", "unc", "res2", "res2bar", "res2hat")
N_FOLDS = sx.N_FLOATS
FITS = ("ols", "glsq", "loc", "locq")
FIT_LAB = dict(ols="global OLS", glsq="global GLS + log q-hat", loc="local LS (RG/KS)", locq="local LS + log q-hat")
FIT_COL = dict(ols="#2a78d6", glsq="#1baf7a", loc="#eb6834", locq="#4a3aa7")
NAIVE_COL = "#9a9893"
MECHS = dict(warm={}, front=dict(front=0.2))


def flows(mech="warm"):
    """Cloud models per scenario; the front drift scales with the warm drift, so "no_conv" omits both."""
    return {sc: dict(f, meander=MEANDER, k_m=K_M, **{k: v * f["delta"] / sx.TRUE["delta"] for k, v in MECHS[mech].items()})
            for sc, f in sx.SCENARIOS.items()}


TRUE = flows()["correct"]


def seas_amp(x, y):
    return A_SEAS * (1 + 0.5 * np.tanh(y / sx.L)) * (1 + ZONAL * np.sin(sx.K_N[0] * x))


def season(x, y, t):
    return seas_amp(x, y) * np.cos(2 * np.pi * (np.asarray(t) - T_PEAK) / YEAR)


def mean_true(x, y, t):
    """Time mean of the streamfunction part (jet + standing meander) plus the seasonal cycle."""
    return -np.tanh(y / sx.L) + MEANDER * np.cosh(y / sx.L) ** -2 * np.cos(K_M * x) + season(x, y, t)


def ohc(x, y, t):
    return sx.ohc(x, y, t, meander=MEANDER, k_m=K_M) + season(x, y, t)


def basis(y, t, lq=None):
    u, w = y / sx.LY, 2 * np.pi * np.asarray(t, float) / YEAR
    c, s = np.broadcast_arrays(np.cos(w), np.sin(w), u)[:2]
    cols = [np.ones_like(u), u, u**2, u**3, c, s, u * c, u * s] + ([lq] if lq is not None else [])
    return np.stack(cols, -1)


def anomaly_params(nug_frac=1e-3):
    """Oracle e-folding scales and sill of eta = Z - mu on the grid (daily over one year)."""
    t = np.arange(0.0, YEAR)
    E = np.stack([(ohc(sx.GX, sx.GY, ti) - mean_true(sx.GX, sx.GY, ti)).reshape(sx.NX, sx.NY) for ti in t])
    s2 = (E**2).mean()
    efold = lambda r, step: step * np.argmax(np.asarray(r) < np.exp(-1))
    rx = [(E * np.roll(E, k, 1)).mean() / s2 for k in range(sx.NX // 2)]
    ry = [(E[:, :, : sx.NY - k] * E[:, :, k:]).mean() / s2 for k in range(sx.NY)]
    rt = [(E[: len(t) - k] * E[k:]).mean() / s2 for k in range(len(t) // 2)]
    return dict(s2=s2, lx=efold(rx, sx.DX), ly=efold(ry, sx.DX), lt=efold(rt, 1.0), nug=nug_frac * s2)


def cov(a, b, kp):
    """Anisotropic exponential in (x, y, t); x periodic."""
    dx = np.abs(a[:, None, 0] - b[None, :, 0])
    dx = np.minimum(dx, sx.LX - dx)
    d = np.sqrt((dx / kp["lx"]) ** 2 + ((a[:, None, 1] - b[None, :, 1]) / kp["ly"]) ** 2
                + ((a[:, None, 2] - b[None, :, 2]) / kp["lt"]) ** 2)
    return kp["s2"] * np.exp(-d)


AREA = 2 * sx.LX * sx.LY
SCENARIOS = list(sx.SCENARIOS)


def log_rel(dens, n):
    """log of a stacked cloud density relative to uniform for n floats, floored at C_REL."""
    return np.log(np.maximum(dens * AREA / n, sx.C_REL))


def simulate(rng, mech="warm"):
    """One replicate of real floats: profiles (snap, float, x, y, t, z) and the full trajectories."""
    cyc = np.arange(SPIN_C, SPIN_C + N_SNAP)
    traj = sx.real_floats(1, rng, cycles=int(cyc[-1]), flow=flows(mech)["correct"])[0]
    n = traj.shape[0]
    xy = traj[:, cyc].transpose(1, 0, 2)
    t = np.repeat(cyc * sx.CYCLE, n)
    x, y = xy[..., 0].ravel(), xy[..., 1].ravel()
    prof = pd.DataFrame(dict(snap=np.repeat(np.arange(N_SNAP), n), float=np.tile(np.arange(n), N_SNAP),
                             x=x, y=y, t=t, z=ohc(x, y, t)))
    return prof, traj, cyc * sx.CYCLE


def propensity(traj, prof, flow, rng, folds=None, targets=()):
    """Clouds from anchors H cycles back under `flow`: pooled density grid per snapshot `tot` (S, NX, NY),
    log q-hat at the profiles `lq`, and with folds the leave-fold-out log q-hat `lq_out` (rows, K) and the
    per-fold densities `lam` (T, K, NX, NY) at the targets."""
    cyc = np.arange(SPIN_C, SPIN_C + N_SNAP)
    n = traj.shape[0]
    parts = sx.clouds(traj[:, cyc - H].transpose(1, 0, 2), ((cyc - H) * sx.CYCLE)[:, None],
                      H * sx.CYCLE, flow, rng, M_PART)
    ix, iy = sx.cell(prof[["x", "y"]].values.reshape(N_SNAP, n, 2))
    if folds is None:
        tot = np.stack([sx.density_grid(pp) for pp in parts])
        return dict(tot=tot, lq=log_rel(tot[np.arange(N_SNAP)[:, None], ix, iy], n).ravel())
    fold, nf = np.arange(n) % folds, np.bincount(np.arange(n) % folds)
    tot, lq_out, lam_t = [], [], []
    for k, pp in enumerate(parts):
        lam = np.stack([sx.density_grid(pp[fold == b]) for b in range(folds)])
        tot.append(lam.sum(0))
        lq_out.append(log_rel(tot[-1][ix[k], iy[k], None] - lam[:, ix[k], iy[k]].T, n - nf))
        if k in targets:
            lam_t.append(lam.astype(np.float32))
    tot = np.stack(tot)
    return dict(tot=tot, lq=log_rel(tot[np.arange(N_SNAP)[:, None], ix, iy], n).ravel(),
                lq_out=np.concatenate(lq_out), lam=np.stack(lam_t))


def fit_means(prof, kp, lqs):
    """OLS and GLS without q-hat, and GLS + log q-hat for each covariate in `lqs` (key -> values at the
    rows of `prof`); GLS uses year blocks of the oracle anomaly covariance."""
    y, t, z = prof.y.values, prof.t.values, prof.z.values
    p = prof[["x", "y", "t"]].values
    blocks = [np.flatnonzero(t // YEAR == yr) for yr in np.unique(t // YEAR)]
    chol = [np.linalg.cholesky(cov(p[i], p[i], kp) + kp["nug"] * np.eye(len(i))) for i in blocks]
    white = lambda A: np.concatenate([solve_triangular(c, A[i], lower=True) for c, i in zip(chol, blocks)])
    ls = lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]
    wz = white(z)
    return dict(ols=ls(basis(y, t), z)) | {f"glsq/{k}": ls(white(basis(y, t, v)), wz) for k, v in lqs.items()}


# ----------------------------------------------------------------------------
# local (Roemmich-Gilson / Kuusela-Stein) mean field: moving-window least squares on a centre grid
# ----------------------------------------------------------------------------

LAM_W, N_HARM, N_MIN = 150.0, 6, 100
T_D = (SPIN_C + (N_SNAP - 1) / 2) * sx.CYCLE
CX_N, CY_N = 45, 20
CENTERS = np.stack([a.ravel() for a in np.meshgrid((np.arange(CX_N) + 0.5) * sx.LX / CX_N,
                                                  (np.arange(CY_N) + 0.5) * 2 * sx.LY / CY_N - sx.LY,
                                                  indexing="ij")], -1)
N_T = 3 + 2 * N_HARM


def tbasis(t):
    """Intercept, N_HARM annual harmonics and a quadratic trend in (t - T_D) / YEAR: the part kept as mu-hat(x*, y*, t)."""
    t = np.asarray(t, float)[..., None]
    w, s = 2 * np.pi * t / YEAR * np.arange(1, N_HARM + 1), (t - T_D) / YEAR
    return np.concatenate([np.ones_like(t), np.sin(w), np.cos(w), s, s**2], -1)


def _wrap(d):
    return (d + sx.LX / 2) % sx.LX - sx.LX / 2


def interp_matrix(x, y):
    """Bilinear interpolation from the centre grid (periodic in x, clamped in y) as a sparse (n, C) matrix."""
    fx = np.asarray(x) / (sx.LX / CX_N) - 0.5
    i0 = np.floor(fx)
    wx, i0 = fx - i0, i0.astype(int) % CX_N
    i1 = (i0 + 1) % CX_N
    fy = np.clip((np.asarray(y) + sx.LY) / (2 * sx.LY / CY_N) - 0.5, 0, CY_N - 1)
    j0 = np.minimum(np.floor(fy).astype(int), CY_N - 2)
    wy = fy - j0
    cols = np.stack([i0 * CY_N + j0, i0 * CY_N + j0 + 1, i1 * CY_N + j0, i1 * CY_N + j0 + 1], -1)
    vals = np.stack([(1 - wx) * (1 - wy), (1 - wx) * wy, wx * (1 - wy), wx * wy], -1)
    return csr_matrix((vals.ravel(), (np.repeat(np.arange(cols.shape[0]), 4), cols.ravel())),
                      shape=(cols.shape[0], len(CENTERS)))


GRID_W = interp_matrix(sx.GX, sx.GY)


def local_fits(x, y, t, z, lqs, fold=None, folds=None, lq_outs=None):
    """Window fits at every centre: key "loc" and "locq/{k}" -> (C, N_T [+1]) coefficients of tbasis [and log q-hat];
    with folds also the leave-fold-out coefficients (folds, C, .), fold b's covariate from `lq_outs[k][:, b]`.
    A window holding fewer than N_MIN profiles is widened by 25% steps."""
    T = tbasis(t)
    p = 5 + N_T
    ridge = 1e-8 * np.eye(p + 1)
    full = {"loc": np.zeros((len(CENTERS), N_T))} | {f"locq/{k}": np.zeros((len(CENTERS), N_T + 1)) for k in lqs}
    lofo = {k: np.zeros((folds,) + v.shape) for k, v in full.items()} if folds else {}
    for c, (cx, cy) in enumerate(CENTERS):
        dx, dy, lam = _wrap(x - cx), y - cy, LAM_W
        while (inw := (np.abs(dx) <= lam) & (np.abs(dy) <= lam)).sum() < N_MIN:
            lam *= 1.25
        i = np.flatnonzero(inw)
        u, v = dx[i] / lam, dy[i] / lam
        X, zi = np.column_stack([u, v, u * v, u**2, v**2, T[i]]), z[i]
        G, g = X.T @ X, X.T @ zi
        full["loc"][c] = np.linalg.solve(G + ridge[:p, :p], g)[5:]
        for k, lq in lqs.items():
            Xq = np.column_stack([X, lq[i]])
            full[f"locq/{k}"][c] = np.linalg.solve(Xq.T @ Xq + ridge, Xq.T @ zi)[5:]
        if not folds:
            continue
        fb = fold[i]
        o = np.argsort(fb, kind="stable")
        st = np.flatnonzero(np.r_[True, fb[o][1:] != fb[o][:-1]])
        Gb, gb = np.zeros((folds, p, p)), np.zeros((folds, p))
        Gb[fb[o][st]] = np.add.reduceat(X[o, :, None] * X[o, None, :], st)
        gb[fb[o][st]] = np.add.reduceat(X[o] * zi[o, None], st)
        A, a = G - Gb, g - gb
        lofo["loc"][:, c] = np.linalg.solve(A + ridge[:p, :p], a[..., None])[..., 0][:, 5:]
        out = fb[:, None] != np.arange(folds)
        for k, Lq in lq_outs.items():
            Lm = Lq[i] * out
            cr = (X.T @ Lm).T
            Aq = np.zeros((folds, p + 1, p + 1))
            Aq[:, :p, :p], Aq[:, :p, p], Aq[:, p, :p], Aq[:, p, p] = A, cr, cr, (Lm**2).sum(0)
            rhs = np.column_stack([a, Lm.T @ zi])
            lofo[f"locq/{k}"][:, c] = np.linalg.solve(Aq + ridge, rhs[..., None])[..., 0][:, 5:]
    return full, lofo


def uses_q(key):
    return key.split("/")[0].endswith("q")


def mean_at(beta, key, x, y, t, lq=None):
    """Fitted mean at points; local fits interpolate the centre coefficients bilinearly (x may be sx.GX for the grid)."""
    b, lq = beta[key], lq if uses_q(key) else None
    if not key.startswith("loc"):
        return basis(y, t, lq) @ b
    W = GRID_W if x is sx.GX else interp_matrix(x, y)
    t = np.asarray(t, float)
    m = W @ (b[:, :N_T] @ tbasis(t)) if t.ndim == 0 else ((W @ b[:, :N_T]) * tbasis(t)).sum(-1)
    return m + (W @ b[:, N_T]) * lq if lq is not None else m


def fold_fits(prof, kp, prop, folds, fits=FITS):
    """Full-data and leave-fold-out mean fits and residuals; fold b's q-hat covariate uses the other folds' clouds."""
    x, y, t, z = prof.x.values, prof.y.values, prof.t.values, prof.z.values
    fold = prof.float.values % folds
    keys = [k for f in fits for k in ([f"{f}/{sc}" for sc in prop] if uses_q(f) else [f])]
    lq_of = lambda key, b=None: (None if not uses_q(key) else
                                 prop[key.split("/")[1]]["lq"] if b is None else prop[key.split("/")[1]]["lq_out"][:, b])
    sel = lambda f, d: d if f in fits else {}
    glob = any(not f.startswith("loc") for f in fits)
    beta = fit_means(prof, kp, sel("glsq", {sc: v["lq"] for sc, v in prop.items()})) if glob else {}
    beta_f = [fit_means(prof[fold != b], kp, sel("glsq", {sc: v["lq_out"][fold != b, b] for sc, v in prop.items()}))
              if glob else {} for b in range(folds)]
    full, lofo = local_fits(x, y, t, z, sel("locq", {sc: v["lq"] for sc, v in prop.items()}), fold, folds,
                            sel("locq", {sc: v["lq_out"] for sc, v in prop.items()}))
    beta |= full
    for b, bf in enumerate(beta_f):
        bf |= {k: v[b] for k, v in lofo.items()}
    return dict(keys=keys, names=fits, folds=folds, beta=beta, beta_f=beta_f,
                R=np.column_stack([z - mean_at(beta, k, x, y, t, lq_of(k)) for k in keys]),
                R_f=[np.column_stack([z - mean_at(bf, k, x, y, t, lq_of(k, b)) for k in keys])
                     for b, bf in enumerate(beta_f)])


def true_cloud(traj, k, flow, rng, m=M_BIG):
    """Particles (n, m, 2) from each float's anchor H cycles before snapshot k: its exact law at k given the anchor."""
    c = SPIN_C + k
    return sx.clouds(traj[:, c - H], (c - H) * sx.CYCLE, H * sx.CYCLE, flow, rng, m)


def hist_q(pts, m):
    """Unsmoothed intensity w.r.t. normalized area of particles (..., 2), m per float, on the flat grid."""
    ix, iy = sx.cell(pts.reshape(-1, 2))
    return np.bincount(ix * sx.NY + iy, minlength=sx.GX.size) * AREA / (m * sx.DX**2)


def target_maps(prof, kp, prop, fits, times, j, k, maps=False, parts=None):
    """One target snapshot k (index j into each prop's `lam`): per scenario and fit, the box means of the mean
    field, the kriging map and the cross-fit DR, and with `maps` the grids and weighted residuals behind them.

    DR: fold b's map is mixed on the grid by q_b / q and checked by fold b's profiles at k with weights
    1 / (q * area) on the covered set; off it the maps are averaged equally. Folds share one cross-covariance.
    With `parts` = (true_cloud at k, {scenario: model cloud at k}), also the bias-identity terms in BOUND_TERMS.
    """
    p, z, y = prof[["x", "y", "t"]].values, prof.z.values, prof.y.values
    n, folds, keys = sx.N_FLOATS, fits["folds"], fits["keys"]
    fold, nf = prof.float.values % folds, np.bincount(np.arange(n) % folds)
    key_of = lambda f, sc: keys.index(f"{f}/{sc}" if uses_q(f) else f)
    t0 = times[k]
    nb = np.flatnonzero(np.abs(prof.t.values - t0) <= LAM_T)
    g = np.column_stack([sx.GX, sx.GY, np.full(sx.GX.size, t0)])
    C, Kn = cov(p[nb], g, kp), cov(p[nb], p[nb], kp) + kp["nug"] * np.eye(len(nb))
    anom_full = C.T @ cho_solve(cho_factor(Kn), fits["R"][nb])
    here = np.flatnonzero(prof.snap.values == k)
    out = {}
    for sc, v in prop.items():
        tot = v["lam"][j].sum(0)
        covered = tot * AREA / n >= sx.C_REL
        ix, iy = sx.cell(p[here, :2])
        wt = np.where(covered[ix, iy], 1 / (np.where(covered[ix, iy], tot[ix, iy], 1.0) * AREA), 0.0)
        out[sc] = dict(cover=covered.mean(), xy=p[here, :2], z=z[here], weight=wt,
                       mix=[np.where(covered, v["lam"][j][b] / np.where(covered, tot, 1.0), 1 / folds).ravel()
                            for b in range(folds)],
                       lq_grid=[log_rel(tot - v["lam"][j][b], n - nf[b]).ravel() for b in range(folds)])
        for f in fits["names"]:
            i = key_of(f, sc)
            mu = mean_at(fits["beta"], keys[i], sx.GX, sx.GY, t0, log_rel(v["tot"][k], n).ravel())
            out[sc][f] = dict(mean_grid=mu, krige_grid=mu + anom_full[:, i],
                              dr_grid=np.zeros(sx.GX.size), resid=np.zeros(len(here)))
            if parts is not None:
                out[sc][f]["bnd"] = dict(res=0.0, R=0.0)
    if parts is not None:
        parts, mparts = parts
        zg, owner = ohc(sx.GX, sx.GY, t0), np.arange(n) % folds
        q_half = [hist_q(parts[:, h::2], parts.shape[1] / 2) for h in (0, 1)]
        qbar_half = {sc: [sx.density_grid(pp[:, h::2]).ravel() * AREA for h in (0, 1)] for sc, pp in mparts.items()}
        qhat = {sc: v["lam"][j].sum(0).ravel() * AREA for sc, v in prop.items()}
        iq = {sc: np.where(qh >= sx.C_REL * n, 1 / np.maximum(qh, sx.C_REL * n), 0.0) for sc, qh in qhat.items()}
    for b in range(folds):
        keep = fold[nb] != b
        chk = here[fold[here] == b]
        alpha = np.zeros((len(nb), len(keys)))
        alpha[keep] = cho_solve(cho_factor(Kn[np.ix_(keep, keep)]), fits["R_f"][b][nb[keep]])
        anom_grid, anom_chk = C.T @ alpha, cov(p[chk], p[nb], kp) @ alpha
        pos = np.searchsorted(here, chk)
        if parts is not None:
            pts = parts[owner == b].reshape(-1, 2)
            pix, piy = sx.cell(pts)
            pc, zp, qb = pix * sx.NY + piy, ohc(pts[:, 0], pts[:, 1], t0), hist_q(pts, parts.shape[1])
            anom_p = cov(np.column_stack([pts, np.full(len(pts), t0)]), p[nb], kp) @ alpha
            wq = {sc: c[pc] for sc, c in iq.items()}
        for sc, v in prop.items():
            o = out[sc]
            for f in fits["names"]:
                i = key_of(f, sc)
                m_grid = mean_at(fits["beta_f"][b], keys[i], sx.GX, sx.GY, t0, o["lq_grid"][b]) + anom_grid[:, i]
                o[f]["dr_grid"] += o["mix"][b] * m_grid
                o[f]["resid"][pos] = z[chk] - (mean_at(fits["beta_f"][b], keys[i], p[chk, 0], y[chk], t0,
                                                       v["lq_out"][chk, b])
                                               + anom_chk[:, i])
                if parts is not None:
                    mp = mean_at(fits["beta_f"][b], keys[i], pts[:, 0], pts[:, 1], t0, o["lq_grid"][b][pc]) + anom_p[:, i]
                    o[f]["bnd"]["res"] += ((zp - mp) * wq[sc]).sum() / parts.shape[1]
                    o[f]["bnd"]["R"] += (iq[sc] * (zg - m_grid) * (qb - v["lam"][j][b].ravel() * AREA)).mean()
    for sc, o in out.items():
        if parts is not None:
            qh, w, qt = qhat[sc], iq[sc], sum(q_half) / 2
            cv = w > 0
            qb = sum(qbar_half[sc]) / 2
            wb = np.where(cv, 1 / np.maximum(qb, 1e-12), 0.0)
            o["bnd"] = dict(chi2=((q_half[0] - qh) * (q_half[1] - qh) * w).mean(),
                            chi2sys=((q_half[0] - qbar_half[sc][0]) * (q_half[1] - qbar_half[sc][1]) * wb).mean())
            for f in fits["names"]:
                d, e = o[f], zg - o[f]["dr_grid"]
                bd = d["bnd"]
                on = (np.where(cv, -e, 0.0)).mean() + bd["res"]
                bd |= dict(unc=np.where(cv, 0.0, -e).mean(), Q=on - bd["R"],
                           Rpool=(e * (qt - qh) * w).mean(), Rsys=(e * (qt - qb) * wb).mean(), res2=(e**2 * w).mean(),
                           res2bar=(e**2 * wb).mean(), res2hat=(o["weight"] ** 2 * d["resid"] ** 2).sum())
                bd["Rmc"] = bd["Rpool"] - bd["Rsys"]
                bd["cbias"] = on + bd["unc"]
                del bd["res"]
        for f in fits["names"]:
            d = o[f]
            d |= dict(mean=d["mean_grid"].mean(), krige=d["krige_grid"].mean(),
                      dr=d["dr_grid"].mean() + o["weight"] @ d["resid"])
            if not maps:
                for key in ("mean_grid", "krige_grid", "dr_grid"):
                    del d[key]
        del o["mix"], o["lq_grid"]
    return out


def estimate(prof, kp, prop, times, targets, folds, true_parts=None, fits=FITS):
    """Rows per target and scenario: naive, mean only, mean + kriging, and cross-fit DR with `folds` folds;
    with `true_parts` (k -> true_cloud) also the bias-identity terms."""
    names, fits = fits, fold_fits(prof, kp, prop, folds, fits)
    rows = []
    for j, k in enumerate(targets):
        res = target_maps(prof, kp, prop, fits, times, j, k, parts=None if true_parts is None else true_parts(k))
        base = dict(snap=k, t=times[k], truth=ohc(sx.GX, sx.GY, times[k]).mean(),
                    naive=prof.z.values[prof.snap.values == k].mean())
        for sc, o in res.items():
            rows.append(dict(base, scenario=sc, cover=o["cover"]) |
                        {f"{f}_{st}": o[f][st] for f in names for st in ("mean", "krige", "dr")} |
                        o.get("bnd", {}) | {f"{f}_{b}": v for f in names for b, v in o[f].get("bnd", {}).items()})
    return pd.DataFrame(rows), {sc: fits["beta"][f"glsq/{sc}"][-1] for sc in prop if "glsq" in names}


# ----------------------------------------------------------------------------
# fitting the cloud diffusivity K to the floats' own transitions ("GLORYS" = a deterministic model velocity)
# ----------------------------------------------------------------------------

K_GRID, M_FIT, H_DIAG = np.geomspace(10.0, 3000.0, 13), 50, (1, 2, 4, 8)
C_FIT = SPIN_C - H
SEP_BINS = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
GLORYS = dict(diffusion="correct", no_conv="no_conv", shifted="shifted")


def misses(traj, flow, h, K, seed, m=M_FIT):
    """Cross-jet cloud-mean misses and cloud variances (C, n) for transitions c -> c + h ending by cycle C_FIT (the
    first analysis anchor), with common random numbers across K. Along-jet displacements alias around the periodic
    channel at these horizons, so only y is used."""
    c = np.arange(C_FIT - h + 1)
    parts = sx.clouds(traj[:, c].transpose(1, 0, 2), (c * sx.CYCLE)[:, None], h * sx.CYCLE, dict(flow, K=K),
                      np.random.default_rng(seed), m)[..., 1]
    return traj[:, c + h, 1].T - parts.mean(-1), parts.var(-1, ddof=1)


def _root(G):
    """log-linear root in K of a decreasing moment function on K_GRID (first sign change); nan if none."""
    i = np.flatnonzero((G[:-1] > 0) & (G[1:] <= 0))
    if not len(i):
        return np.nan
    i, lk = i[0], np.log(K_GRID)
    return np.exp(lk[i] + G[i] / (G[i] - G[i + 1]) * (lk[i + 1] - lk[i]))


def fit_K(traj, flow, seed, h=H, boot=400, m=M_FIT):
    """Simulated-moments K: per-float sums of miss^2 - s/m - s, zero in expectation at the true K when the model
    velocity is right; float-bootstrap 95% interval."""
    g = np.stack([(mi**2 - s / m - s).sum(0) for mi, s in (misses(traj, flow, h, K, seed, m) for K in K_GRID)])
    rng = np.random.default_rng(seed)
    kb = [_root(g[:, rng.integers(0, g.shape[1], g.shape[1])].sum(1)) for _ in range(boot)]
    return (_root(g.sum(1)), *(np.nanpercentile(kb, [2.5, 97.5]) if boot else (np.nan, np.nan)))


def k_diagnostics(traj, flow, K, seed):
    """K-hat by horizon; mean cross-jet miss (km) overall and by band; neighbour miss correlation by anchor separation."""
    out = {f"K_h{h}": fit_K(traj, flow, seed, h, boot=0)[0] for h in H_DIAG}
    miss, _ = misses(traj, flow, H, K, seed)
    a = traj[:, np.arange(C_FIT - H + 1)].transpose(1, 0, 2)
    pf = miss.mean(0)
    out |= dict(miss_y=pf.mean(), miss_y_se=pf.std(ddof=1) / np.sqrt(len(pf)))
    band = np.digitize(a[..., 1], np.linspace(-sx.LY, sx.LY, 6)[1:-1])
    out |= {f"miss_y_band{b}": miss[band == b].mean() for b in range(5)}
    i, j = np.triu_indices(a.shape[1], 1)
    b = np.digitize(np.hypot(_wrap(a[:, i, 0] - a[:, j, 0]), a[:, i, 1] - a[:, j, 1]), SEP_BINS) - 1
    dot, nrm = miss[:, i] * miss[:, j], (miss[:, i] ** 2 + miss[:, j] ** 2) / 2
    out |= {f"corr_{SEP_BINS[k]:.0f}_{SEP_BINS[k + 1]:.0f}": dot[b == k].sum() / nrm[b == k].sum()
            for k in range(len(SEP_BINS) - 1)}
    return out


def kfit_models(traj, seed):
    """Cloud models for the K-fitting toy: guessed and fitted K for each "GLORYS" velocity, plus the oracle; and a
    table of K-hat, its interval and the diagnostics per "GLORYS" velocity."""
    base, rows, models = flows(), [], dict(oracle=flows()["correct"])
    for g, sc in GLORYS.items():
        k, lo, hi = fit_K(traj, base[sc], seed)
        rows.append(dict(glorys=g, K_hat=k, K_lo=lo, K_hi=hi) | k_diagnostics(traj, base[sc], k, seed))
        guesses = dict(K25=base["K_small"], K400=base["K_large"]) if g == "diffusion" else dict(K100=base[sc])
        models |= {f"{g}_{n}": f for n, f in guesses.items()} | {f"{g}_Kfit": dict(base[sc], K=k)}
    return models, pd.DataFrame(rows)


def one_rep(r, seed=1, folds=N_FOLDS, every=TARGET_EVERY, mech="warm", m_true=0, n_targets=None, fits=FITS,
            kfit=False):
    """One replicate; m_true > 0 adds the bias-identity terms from true and per-scenario model clouds of m_true
    particles per float; kfit replaces the scenario clouds by kfit_models and returns its K table."""
    rng = np.random.default_rng([seed, r])
    prof, traj, times = simulate(rng, mech)
    targets = np.arange(0, N_SNAP, every)[:n_targets]
    models, info = kfit_models(traj, [seed, r, 77]) if kfit else (flows(mech), None)
    prop = {sc: propensity(traj, prof, flow, np.random.default_rng([seed, r, i]), folds, set(targets))
            for i, (sc, flow) in enumerate(models.items())}
    rt = np.random.default_rng([seed, r, 99])
    true_parts = (lambda k: (true_cloud(traj, k, flows(mech)["correct"], rt, m_true),
                             {sc: true_cloud(traj, k, fl, rt, m_true) for sc, fl in models.items()})) if m_true else None
    df, gam = estimate(prof, anomaly_params(), prop, times, targets, folds, true_parts, fits)
    extra = info if kfit else pd.DataFrame(dict(scenario=list(gam), gamma=list(gam.values())))
    return df.assign(rep=r), extra.assign(rep=r)


def run(R=30, seed=1, path=None, workers=6, mech="warm"):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(one_rep, range(R), [seed] * R, [N_FOLDS] * R, [TARGET_EVERY] * R, [mech] * R))
    df, gam = (pd.concat(x, ignore_index=True) for x in zip(*res))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        gam.to_parquet(path.replace("results", "gamma"))
    return df, gam


def load_or_run(R=30, seed=1, mech="warm"):
    tag = "meander" if mech == "warm" else f"meander_{mech}"
    path = os.path.join(sx.OUT_DIR, f"meanfield_results_{tag}_R{R}_s{seed}_K{N_FOLDS}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path), pd.read_parquet(path.replace("results", "gamma"))
    return run(R, seed, path, mech=mech)


def example(seed=7):
    """One replicate under the correct cloud model, kept whole for the figures (full-data fits only)."""
    kp = anomaly_params()
    prof, traj, times = simulate(np.random.default_rng(seed))
    prop = propensity(traj, prof, TRUE, np.random.default_rng([seed, 0]))
    beta = fit_means(prof, kp, {"correct": prop["lq"]})
    beta |= local_fits(prof.x.values, prof.y.values, prof.t.values, prof.z.values, {"correct": prop["lq"]})[0]
    beta = {k.split("/")[0]: v for k, v in beta.items()}
    prof = prof.assign(lq=prop["lq"])
    est = pd.DataFrame(dict(snap=np.arange(N_SNAP), t=times,
                            truth=[ohc(sx.GX, sx.GY, t0).mean() for t0 in times],
                            naive=prof.groupby("snap").z.mean().values))
    for f in FITS:
        est[f"{f}_mean"] = [mean_at(beta, f, sx.GX, sx.GY, t0, log_rel(prop["tot"][k], sx.N_FLOATS).ravel()).mean()
                            for k, t0 in enumerate(times)]
    return dict(kp=kp, prof=prof, tot=prop["tot"], times=times, beta=beta, est=est)


def grid_lq(ex, k):
    return log_rel(ex["tot"][k], sx.N_FLOATS).ravel()


def region_split(ex, every=TARGET_EVERY):
    """Example replicate: contribution of the south and north halves to each full-data map's box-mean error."""
    kp, beta, prof, times = ex["kp"], ex["beta"], ex["prof"], ex["times"]
    p, north = prof[["x", "y", "t"]].values, sx.GY > 0
    rows = []
    for k in range(0, N_SNAP, every):
        nb = np.flatnonzero(np.abs(prof.t.values - times[k]) <= LAM_T)
        g = np.column_stack([sx.GX, sx.GY, np.full(sx.GX.size, times[k])])
        W = cho_solve(cho_factor(cov(p[nb], p[nb], kp) + kp["nug"] * np.eye(len(nb))), cov(p[nb], g, kp))
        Z = ohc(sx.GX, sx.GY, times[k])
        for f in FITS:
            m = mean_at(beta, f, sx.GX, sx.GY, times[k], grid_lq(ex, k))
            r = prof.z.values[nb] - mean_at(beta, f, prof.x.values[nb], prof.y.values[nb], prof.t.values[nb],
                                            prof.lq.values[nb])
            for st, mp in (("mean field only", m), ("mean field + kriging", m + W.T @ r)):
                e = (mp - Z) / Z.size
                rows.append(dict(stage=st, fit=FIT_LAB[f], south=e[~north].sum(), north=e[north].sum()))
    out = pd.DataFrame(rows).groupby(["stage", "fit"], sort=False).mean()
    return out.assign(total=out.south + out.north).reset_index()


# ----------------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------------


STAGE_LAB = dict(mean="mean field only", krige="mean field + kriging", dr="cross-fit DR (LOFO)")
SEASONS = ("warm peak", "cooling", "cold", "warming")
ORDER = ["naive"] + [f"{f}_{k}" for k in STAGE_LAB for f in FITS]


def long_table(df):
    d = df.melt(id_vars=[c for c in ("rep", "snap", "t", "truth", "scenario") if c in df], value_vars=ORDER, var_name="estimator", value_name="estimate")
    phase = ((d.t - T_PEAK) % YEAR) / YEAR
    return d.assign(err=d.estimate - d.truth, season=pd.Categorical(np.array(SEASONS)[((phase + 0.125) % 1 * 4).astype(int)], SEASONS, ordered=True))


def summary(df, by=()):
    """Bias, its MC standard error across replicates (rep-level means), sd and rmse over replicate-days."""
    d, keys = long_table(df), ["estimator", *by]
    per_rep = d.groupby([*keys, "rep"], observed=True).err.mean()
    s = d.groupby(keys, observed=True).agg(bias=("err", "mean"), sd=("err", "std"), rmse=("err", lambda e: np.sqrt((e**2).mean())))
    s = s.join(per_rep.groupby(keys, observed=True).sem().rename("se")).reset_index()
    s = s.assign(bias_over_se=s.bias / s.se,
                 stage=s.estimator.map(lambda e: "simple mean of profiles" if e == "naive" else STAGE_LAB[e.split("_")[1]]),
                 fit=s.estimator.map(lambda e: "" if e == "naive" else FIT_LAB[e.split("_")[0]]),
                 o=s.estimator.map(ORDER.index))
    return s.sort_values([*by, "o"])[[*by, "stage", "fit", "bias", "se", "bias_over_se", "sd", "rmse"]].reset_index(drop=True)


def plot_setup(ex, figsize=(13, 3.8)):
    times = ex["times"]
    warm, cold = (int(np.argmin(np.abs((times - T_PEAK - s) % YEAR))) for s in (0.0, YEAR / 2))
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(width_ratios=[2, 2, 1.6]))
    for a, k, lab in ((ax[0], warm, "warm season"), (ax[1], cold, "cold season")):
        Z = ohc(sx.GX, sx.GY, times[k]).reshape(sx.NX, sx.NY).T
        im = a.imshow(Z, origin="lower", extent=[0, sx.LX, -sx.LY, sx.LY], cmap="RdBu_r", vmin=-1.6, vmax=1.6,
                      aspect="auto")
        pr = ex["prof"][ex["prof"].snap == k]
        a.plot(pr.x, pr.y, "o", ms=3.5, mfc="k", mec="w", mew=0.5)
        a.set(title=f"Z on day {times[k]:.0f} ({lab})", xlabel="x (km)", ylabel="y (km)")
        a.grid(False)
    fig.colorbar(im, ax=ax[:2], pad=0.01, label="Z")
    e = ex["est"]
    ax[2].plot(e.t, e.truth, color="k", lw=2, label="truth (box mean)")
    ax[2].plot(e.t, e.naive, color=NAIVE_COL, lw=1, label="simple mean of profiles")
    ax[2].set(xlabel="day", title="Box mean through four years")
    ax[2].legend(fontsize=8, frameon=False)
    return fig


def plot_fits(ex, k=None, figsize=(13, 3.9)):
    """Zonal-mean fitted mean field against the truth at one snapshot, and its box mean through time."""
    times, beta, prof = ex["times"], ex["beta"], ex["prof"]
    k = int(np.argmin(np.abs((times - T_PEAK - YEAR / 2) % YEAR))) if k is None else k
    yy = np.arange(sx.NY) * sx.DX - sx.LY + sx.DX / 2
    fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(width_ratios=[1.2, 1, 2]))
    ax[0].plot(mean_true(sx.GX, sx.GY, times[k]).reshape(sx.NX, sx.NY).mean(0), yy, color="k", lw=2.5,
               label="true mean field")
    for f in FITS:
        m = mean_at(beta, f, sx.GX, sx.GY, times[k], grid_lq(ex, k)).reshape(sx.NX, sx.NY).mean(0)
        ax[0].plot(m, yy, color=FIT_COL[f], lw=1.6, label=FIT_LAB[f])
    ax[0].set(xlabel="zonal-mean mu", ylabel="y (km)", title=f"Fitted mean, day {times[k]:.0f}")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].hist(prof.y, bins=25, orientation="horizontal", density=True, color=NAIVE_COL, edgecolor="w")
    ax[1].axvline(1 / (2 * sx.LY), color="k", ls="--", lw=1)
    ax[1].set(xlabel="profile density", title="Where the profiles are", yticklabels=[])
    e = ex["est"]
    ax[2].plot(e.t, e.truth, color="k", lw=2.5, label="truth")
    for f in FITS:
        ax[2].plot(e.t, e[f"{f}_mean"], color=FIT_COL[f], lw=1.6, label=FIT_LAB[f])
    ax[2].set(xlabel="day", ylabel="box mean", title="Box mean of the fitted mean field alone")
    ax[2].legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_q_term(ex, k=None, figsize=(12, 3.8)):
    """log q-hat on the grid next to the true anomaly."""
    times = ex["times"]
    k = int(np.argmin(np.abs((times - T_PEAK - YEAR / 2) % YEAR))) if k is None else k
    lq = grid_lq(ex, k).reshape(sx.NX, sx.NY).T
    eta = (ohc(sx.GX, sx.GY, times[k]) - mean_true(sx.GX, sx.GY, times[k])).reshape(sx.NX, sx.NY).T
    ext = [0, sx.LX, -sx.LY, sx.LY]
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    im = ax[0].imshow(lq / np.log(10), origin="lower", extent=ext, cmap="Blues", vmin=-1.3, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax[0], pad=0.01, label="log10 q-hat / uniform")
    ax[0].set(title=f"Expected float density q-hat, day {times[k]:.0f}", xlabel="x (km)", ylabel="y (km)")
    im = ax[1].imshow(eta, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax[1].contour(lq, levels=[np.log(2)], extent=ext, colors="k", linewidths=0.8)
    fig.colorbar(im, ax=ax[1], pad=0.01, label="eta")
    ax[1].set(title="True anomaly eta, with q-hat = 2 x uniform contour", xlabel="x (km)")
    for a in ax:
        a.grid(False)
    fig.tight_layout()
    return fig


def plot_errors(df, figsize=(9, 6)):
    s = summary(df[df.scenario == "correct"] if "scenario" in df else df)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="k", lw=0.8)
    lab = [st if not f else f"{st}: {f}" for st, f in zip(s.stage, s.fit)]
    col = [NAIVE_COL] + [FIT_COL[f] for _ in STAGE_LAB for f in FITS]
    for i, (r, c) in enumerate(zip(s.itertuples(), col)):
        ax.errorbar(r.bias, i, xerr=1.96 * r.se, fmt="o", ms=8, color=c, capsize=3, lw=2)
        ax.plot([r.bias - r.sd, r.bias + r.sd], [i, i], color=c, lw=6, alpha=0.2, solid_capstyle="butt")
    ax.set_yticks(range(len(lab)), lab)
    ax.invert_yaxis()
    ax.set(xlabel="error of the box-mean estimate (dot: bias with 95% MC bar; band: +/- 1 sd)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_scenarios(df, figsize=(10, 4.6)):
    """Bias +/- 1.96 MC se of each DR estimator and the local + log q-hat kriging map under each cloud model."""
    s = summary(df, by=("scenario",))
    s = s[(s.stage == STAGE_LAB["dr"]) | ((s.stage == STAGE_LAB["krige"]) & (s.fit == FIT_LAB["locq"]))]
    s = s.assign(label=np.where(s.stage == STAGE_LAB["dr"], "DR: " + s.fit, "kriging: " + s.fit))
    labels = list(dict.fromkeys(s.label))
    col = {l: FIT_COL[[f for f in FITS if FIT_LAB[f] in l][0]] for l in labels}
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="k", lw=0.8)
    for i, l in enumerate(labels):
        d = s[s.label == l].set_index("scenario").loc[SCENARIOS]
        yy = np.arange(len(SCENARIOS)) + (i - (len(labels) - 1) / 2) * 0.18
        ax.errorbar(d.bias, yy, xerr=1.96 * d.se, fmt="o" if l.startswith("DR") else "D", ms=6, color=col[l],
                    mfc=col[l] if l.startswith("DR") else "w", capsize=2, lw=1.5, label=l)
    ax.set_yticks(range(len(SCENARIOS)), [sx.SCEN_LAB[k] for k in SCENARIOS])
    ax.invert_yaxis()
    ax.set(xlabel="bias of the box-mean estimate (bars: 95% MC interval)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# spatial walkthrough of one target day
# ----------------------------------------------------------------------------

EXT = [0, sx.LX, -sx.LY, sx.LY]
YY = np.arange(sx.NY) * sx.DX - sx.LY + sx.DX / 2
ZK = dict(cmap="RdBu_r", vmin=-1.6, vmax=1.6)


def cold_day(times):
    return int(np.argmin(np.abs((times - T_PEAK - YEAR / 2) % YEAR)))


def spatial_story(seed=7, k=None):
    """The example replicate on one day: every estimator's grids under the correct cloud model, LOFO DR."""
    kp = anomaly_params()
    prof, traj, times = simulate(np.random.default_rng(seed))
    k = cold_day(times) if k is None else k
    prop = {"correct": propensity(traj, prof, TRUE, np.random.default_rng([seed, 0]), N_FOLDS, {k})}
    o = target_maps(prof, kp, prop, fold_fits(prof, kp, prop, N_FOLDS), times, 0, k, maps=True)["correct"]
    t0 = times[k]
    Z = ohc(sx.GX, sx.GY, t0)
    err = np.concatenate([o[f][g] - Z for f in FITS for g in ("krige_grid", "dr_grid")])
    return dict(o, t=t0, Z=Z, mu=mean_true(sx.GX, sx.GY, t0), truth=Z.mean(), naive=o["z"].mean(),
                emax=float(np.round(np.quantile(np.abs(err), 0.99), 1)))


def _img(ax, grid, **kw):
    ax.grid(False)
    return ax.imshow(grid.reshape(sx.NX, sx.NY).T, origin="lower", extent=EXT, aspect="auto", **kw)


def _floats(ax, st, **kw):
    ax.plot(*st["xy"].T, "o", ms=3, mfc="k", mec="w", mew=0.5, **kw)


def _zonal_errors(ax, st, grids, title):
    for f in FITS:
        ax.plot((st[f][grids] - (st["mu"] if grids == "mean_grid" else st["Z"])).reshape(sx.NX, sx.NY).mean(0),
                YY, color=FIT_COL[f], lw=2, label=FIT_LAB[f])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="zonal-mean error", ylabel="y (km)", title=title, ylim=(-sx.LY, sx.LY))
    ax.legend(fontsize=8, frameon=False)


def _panel_grid(st, row1, grids, truth, row1_title, row2_title, figsize=(18, 6.6)):
    """Row 1: the truth and each fit's grid; row 2: zonal-mean error by fit and each fit's error map."""
    ek = dict(cmap="PuOr_r", vmin=-st["emax"], vmax=st["emax"])
    fig, ax = plt.subplots(2, 1 + len(FITS), figsize=figsize, sharey=True)
    im = _img(ax[0, 0], truth, **ZK)
    ax[0, 0].set_title(row1)
    for c, f in enumerate(FITS, 1):
        _img(ax[0, c], st[f][grids], **ZK)
        ax[0, c].set_title(row1_title(f))
        ime = _img(ax[1, c], st[f][grids] - truth, **ek)
        ax[1, c].set_title(row2_title(f))
    _zonal_errors(ax[1, 0], st, grids, "Error by latitude")
    fig.colorbar(im, ax=ax[0, :], pad=0.01, fraction=0.025, label="OHC (Z)")
    fig.colorbar(ime, ax=ax[1, :], pad=0.01, fraction=0.025, label="estimate - truth")
    for a in ax[:, 1:].ravel():
        a.set_xlabel("x (km)")
    ax[0, 0].set(ylabel="y (km)", xlabel="x (km)")
    return fig, ax


def plot_mean_maps(st):
    fig, ax = _panel_grid(
        st, f"True mean field mu (box mean {st['mu'].mean():+.3f})", "mean_grid", st["mu"],
        lambda f: f"{FIT_LAB[f]} (box mean {st[f]['mean']:+.3f})",
        lambda f: f"error, {FIT_LAB[f]}")
    return fig


def plot_krige_maps(st):
    fig, ax = _panel_grid(
        st, f"True OHC Z and floats (box mean {st['truth']:+.3f})", "krige_grid", st["Z"],
        lambda f: f"mean + kriging, {FIT_LAB[f]} ({st[f]['krige']:+.3f})",
        lambda f: f"error, {FIT_LAB[f]}")
    _floats(ax[0, 0], st)
    for c in range(1, 1 + len(FITS)):
        _floats(ax[0, c], st, alpha=0.6)
    return fig


def plot_dr_maps(st, figsize=(18, 6.6)):
    """Row 1: the q_f/q mixture of leave-one-out maps; row 2: each float's check, sized by its 1/q weight."""
    ek = dict(cmap="PuOr_r", vmin=-st["emax"], vmax=st["emax"])
    share = st["weight"] / st["weight"].sum()
    fig, ax = plt.subplots(2, 1 + len(FITS), figsize=figsize, sharey=True)
    sc = ax[0, 0].scatter(*st["xy"].T, s=2500 * share + 4, c="#2a78d6", edgecolors="k", linewidths=0.5)
    ax[0, 0].set(xlim=(0, sx.LX), ylim=(-sx.LY, sx.LY), title="Weight of each float's check (area ~ 1/q)",
                 xlabel="x (km)", ylabel="y (km)")
    for c, f in enumerate(FITS, 1):
        d = st[f]
        im = _img(ax[0, c], d["dr_grid"], **ZK)
        ax[0, c].set_title(f"mixed leave-one-out map, {FIT_LAB[f]} ({d['dr_grid'].mean():+.3f})")
        _img(ax[1, c], d["dr_grid"] - st["Z"], alpha=0.35, **ek)
        ime = ax[1, c].scatter(*st["xy"].T, c=d["resid"], s=2500 * share + 4, edgecolors="k", linewidths=0.5, **ek)
        corr = st["weight"] @ d["resid"]
        ax[1, c].set_title(f"checks {corr:+.3f} -> DR {d['dr']:+.3f}")
        ax[1, c].set_xlabel("x (km)")
        ax[0, c].set_xlabel("x (km)")
    _zonal_errors(ax[1, 0], st, "dr_grid", "Map error by latitude")
    fig.colorbar(im, ax=ax[0, :], pad=0.01, fraction=0.025, label="OHC (Z)")
    fig.colorbar(ime, ax=ax[1, :], pad=0.01, fraction=0.025, label="error (map) / residual (floats)")
    return fig


def plot_day_summary(st, figsize=(9, 4.6)):
    """That day's box-mean estimates against the truth, one row per estimator."""
    stages = (("mean", "mean field only", "s"), ("krige", "mean + kriging", "D"), ("dr", "cross-fit DR", "o"))
    rows = [("simple mean of profiles", st["naive"], "#898781", "o")]
    rows += [(f"{lab}: {FIT_LAB[f]}", st[f][key], FIT_COL[f], mk) for key, lab, mk in stages for f in FITS]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(st["truth"], color="k", lw=1)
    ax.text(st["truth"], -0.9, " truth", fontsize=9, va="center")
    for i, (lab, v, c, mk) in enumerate(rows):
        ax.plot(v, i, mk, ms=9, color=c, mec="w", mew=1.5)
        ax.annotate(f"{v:+.3f}", (v, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1.4)
    ax.set_xlabel(f"box-mean estimate on day {st['t']:.0f} (truth {st['truth']:+.3f})")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
