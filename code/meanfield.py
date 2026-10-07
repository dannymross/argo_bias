"""Toy comparison of mean-field fits (OLS, GLS, GLS + log q-hat) ahead of kriging (reports/meanfield.qmd)."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve, solve_triangular

import simex as sx

YEAR, A_SEAS, T_PEAK = 365.0, 0.3, 60.0
SPIN_C, N_SNAP = 37, 146
H, M_PART, LAM_T, TARGET_EVERY = 4, 200, 30.0, 3
N_FOLDS = sx.N_FLOATS
FITS = ("ols", "gls", "glsq")
FIT_LAB = dict(ols="OLS, no q", gls="GLS, no q", glsq="GLS + log q-hat")
FIT_COL = dict(ols="#2a78d6", gls="#eb6834", glsq="#1baf7a")
NAIVE_COL = "#9a9893"


def seas_amp(y):
    return A_SEAS * (1 + 0.5 * np.tanh(y / sx.L))


def season(y, t):
    return seas_amp(y) * np.cos(2 * np.pi * (np.asarray(t) - T_PEAK) / YEAR)


def mean_true(y, t):
    return -np.tanh(y / sx.L) + season(y, t)


def ohc(x, y, t):
    return sx.ohc(x, y, t) + season(y, t)


def basis(y, t, lq=None):
    u, w = y / sx.LY, 2 * np.pi * np.asarray(t, float) / YEAR
    c, s = np.broadcast_arrays(np.cos(w), np.sin(w), u)[:2]
    cols = [np.ones_like(u), u, u**2, u**3, c, s, u * c, u * s] + ([lq] if lq is not None else [])
    return np.stack(cols, -1)


def anomaly_params(nug_frac=1e-3):
    """Oracle e-folding scales and sill of eta = Z - mu on the grid (daily over one year)."""
    t = np.arange(0.0, YEAR)
    E = np.stack([(ohc(sx.GX, sx.GY, ti) - mean_true(sx.GY, ti)).reshape(sx.NX, sx.NY) for ti in t])
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


def simulate(rng):
    """One replicate of real floats: profiles (snap, float, x, y, t, z) and the full trajectories."""
    cyc = np.arange(SPIN_C, SPIN_C + N_SNAP)
    traj = sx.real_floats(1, rng, cycles=int(cyc[-1]))[0]
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
    X, wz = basis(y, t), white(z)
    return dict(ols=ls(X, z), gls=ls(white(X), wz)) | {f"glsq/{k}": ls(white(basis(y, t, v)), wz) for k, v in lqs.items()}


def mean_at(beta, key, y, t, lq=None):
    return basis(y, t, lq if key.startswith("glsq") else None) @ beta[key]


def fold_fits(prof, kp, prop, folds):
    """Full-data and leave-fold-out mean fits and residuals; fold b's q-hat covariate uses the other folds' clouds."""
    z, y, t = prof.z.values, prof.y.values, prof.t.values
    fold = prof.float.values % folds
    keys = ["ols", "gls"] + [f"glsq/{sc}" for sc in prop]
    lq_of = lambda key, b=None: (None if not key.startswith("glsq") else
                                 prop[key[5:]]["lq"] if b is None else prop[key[5:]]["lq_out"][:, b])
    beta = fit_means(prof, kp, {sc: v["lq"] for sc, v in prop.items()})
    beta_f = [fit_means(prof[fold != b], kp, {sc: v["lq_out"][fold != b, b] for sc, v in prop.items()})
              for b in range(folds)]
    return dict(keys=keys, folds=folds, beta=beta, beta_f=beta_f,
                R=np.column_stack([z - mean_at(beta, k, y, t, lq_of(k)) for k in keys]),
                R_f=[np.column_stack([z - mean_at(bf, k, y, t, lq_of(k, b)) for k in keys])
                     for b, bf in enumerate(beta_f)])


def target_maps(prof, kp, prop, fits, times, j, k, maps=False):
    """One target snapshot k (index j into each prop's `lam`): per scenario and fit, the box means of the mean
    field, the kriging map and the cross-fit DR, and with `maps` the grids and weighted residuals behind them.

    DR: fold b's map is mixed on the grid by q_b / q and checked by fold b's profiles at k with weights
    1 / (q * area) on the covered set; off it the maps are averaged equally. Folds share one cross-covariance.
    """
    p, z, y = prof[["x", "y", "t"]].values, prof.z.values, prof.y.values
    n, folds, keys = sx.N_FLOATS, fits["folds"], fits["keys"]
    fold, nf = prof.float.values % folds, np.bincount(np.arange(n) % folds)
    key_of = lambda f, sc: keys.index(f if f != "glsq" else f"glsq/{sc}")
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
        for f in FITS:
            i = key_of(f, sc)
            mu = mean_at(fits["beta"], keys[i], sx.GY, t0, log_rel(v["tot"][k], n).ravel())
            out[sc][f] = dict(mean_grid=mu, krige_grid=mu + anom_full[:, i],
                              dr_grid=np.zeros(sx.GX.size), resid=np.zeros(len(here)))
    for b in range(folds):
        keep = fold[nb] != b
        chk = here[fold[here] == b]
        alpha = np.zeros((len(nb), len(keys)))
        alpha[keep] = cho_solve(cho_factor(Kn[np.ix_(keep, keep)]), fits["R_f"][b][nb[keep]])
        anom_grid, anom_chk = C.T @ alpha, cov(p[chk], p[nb], kp) @ alpha
        pos = np.searchsorted(here, chk)
        for sc, v in prop.items():
            o = out[sc]
            for f in FITS:
                i = key_of(f, sc)
                m_grid = mean_at(fits["beta_f"][b], keys[i], sx.GY, t0, o["lq_grid"][b]) + anom_grid[:, i]
                o[f]["dr_grid"] += o["mix"][b] * m_grid
                o[f]["resid"][pos] = z[chk] - (mean_at(fits["beta_f"][b], keys[i], y[chk], t0, v["lq_out"][chk, b])
                                               + anom_chk[:, i])
    for o in out.values():
        for f in FITS:
            d = o[f]
            d |= dict(mean=d["mean_grid"].mean(), krige=d["krige_grid"].mean(),
                      dr=d["dr_grid"].mean() + o["weight"] @ d["resid"])
            if not maps:
                for key in ("mean_grid", "krige_grid", "dr_grid"):
                    del d[key]
        del o["mix"], o["lq_grid"]
    return out


def estimate(prof, kp, prop, times, targets, folds):
    """Rows per target and scenario: naive, mean only, mean + kriging, and cross-fit DR with `folds` folds."""
    fits = fold_fits(prof, kp, prop, folds)
    rows = []
    for j, k in enumerate(targets):
        res = target_maps(prof, kp, prop, fits, times, j, k)
        base = dict(snap=k, t=times[k], truth=ohc(sx.GX, sx.GY, times[k]).mean(),
                    naive=prof.z.values[prof.snap.values == k].mean())
        for sc, o in res.items():
            rows.append(dict(base, scenario=sc, cover=o["cover"]) |
                        {f"{f}_{st}": o[f][st] for f in FITS for st in ("mean", "krige", "dr")})
    return pd.DataFrame(rows), {sc: fits["beta"][f"glsq/{sc}"][-1] for sc in prop}


def one_rep(r, seed=1, folds=N_FOLDS, every=TARGET_EVERY):
    rng = np.random.default_rng([seed, r])
    prof, traj, times = simulate(rng)
    targets = np.arange(0, N_SNAP, every)
    prop = {sc: propensity(traj, prof, flow, np.random.default_rng([seed, r, i]), folds, set(targets))
            for i, (sc, flow) in enumerate(sx.SCENARIOS.items())}
    df, gam = estimate(prof, anomaly_params(), prop, times, targets, folds)
    return df.assign(rep=r), pd.DataFrame(dict(rep=r, scenario=list(gam), gamma=list(gam.values())))


def run(R=30, seed=1, path=None, workers=6):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(one_rep, range(R), [seed] * R))
    df, gam = (pd.concat(x, ignore_index=True) for x in zip(*res))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        gam.to_parquet(path.replace("results", "gamma"))
    return df, gam


def load_or_run(R=30, seed=1):
    path = os.path.join(sx.OUT_DIR, f"meanfield_results_R{R}_s{seed}_K{N_FOLDS}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path), pd.read_parquet(path.replace("results", "gamma"))
    return run(R, seed, path)


def example(seed=7):
    """One replicate under the correct cloud model, kept whole for the figures (full-data fits only)."""
    kp = anomaly_params()
    prof, traj, times = simulate(np.random.default_rng(seed))
    prop = propensity(traj, prof, sx.TRUE, np.random.default_rng([seed, 0]))
    beta = fit_means(prof, kp, {"correct": prop["lq"]})
    beta["glsq"] = beta.pop("glsq/correct")
    prof = prof.assign(lq=prop["lq"])
    est = pd.DataFrame(dict(snap=np.arange(N_SNAP), t=times,
                            truth=[ohc(sx.GX, sx.GY, t0).mean() for t0 in times],
                            naive=prof.groupby("snap").z.mean().values))
    for f in FITS:
        est[f"{f}_mean"] = [mean_at(beta, f, sx.GY, t0, log_rel(prop["tot"][k], sx.N_FLOATS).ravel()).mean()
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
            m = mean_at(beta, f, sx.GY, times[k], grid_lq(ex, k))
            r = prof.z.values[nb] - mean_at(beta, f, prof.y.values[nb], prof.t.values[nb], prof.lq.values[nb])
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
    ax[0].plot(mean_true(yy, times[k]), yy, color="k", lw=2.5, label="true mean field")
    for f in FITS:
        m = mean_at(beta, f, sx.GY, times[k], grid_lq(ex, k)).reshape(sx.NX, sx.NY).mean(0)
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
    eta = (ohc(sx.GX, sx.GY, times[k]) - mean_true(sx.GY, times[k])).reshape(sx.NX, sx.NY).T
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


def plot_errors(df, figsize=(9, 4.8)):
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
    """Bias +/- 1.96 MC se of each DR estimator and the GLS + log q-hat kriging map under each cloud model."""
    s = summary(df, by=("scenario",))
    s = s[(s.stage == STAGE_LAB["dr"]) | ((s.stage == STAGE_LAB["krige"]) & (s.fit == FIT_LAB["glsq"]))]
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
    prop = {"correct": propensity(traj, prof, sx.TRUE, np.random.default_rng([seed, 0]), N_FOLDS, {k})}
    o = target_maps(prof, kp, prop, fold_fits(prof, kp, prop, N_FOLDS), times, 0, k, maps=True)["correct"]
    t0 = times[k]
    Z = ohc(sx.GX, sx.GY, t0)
    err = np.concatenate([o[f][g] - Z for f in FITS for g in ("krige_grid", "dr_grid")])
    return dict(o, t=t0, Z=Z, mu=mean_true(sx.GY, t0), truth=Z.mean(), naive=o["z"].mean(),
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


def _panel_grid(st, row1, grids, truth, row1_title, row2_title, figsize=(15, 6.6)):
    """Row 1: the truth and each fit's grid; row 2: zonal-mean error by fit and each fit's error map."""
    ek = dict(cmap="PuOr_r", vmin=-st["emax"], vmax=st["emax"])
    fig, ax = plt.subplots(2, 4, figsize=figsize, sharey=True)
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
    for c in range(1, 4):
        _floats(ax[0, c], st, alpha=0.6)
    return fig


def plot_dr_maps(st, figsize=(15, 6.6)):
    """Row 1: the q_f/q mixture of leave-one-out maps; row 2: each float's check, sized by its 1/q weight."""
    ek = dict(cmap="PuOr_r", vmin=-st["emax"], vmax=st["emax"])
    share = st["weight"] / st["weight"].sum()
    fig, ax = plt.subplots(2, 4, figsize=figsize, sharey=True)
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
