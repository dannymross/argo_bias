"""Toy sequential-IPW experiment on a stochastically forced Bickley jet (reports/simex.qmd).

Units are km and days. The "real" ocean and the "model" share this code and differ only in the
flow dict (convergence strength, eddy phases, diffusivity).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simex")

# Rypina et al. (2007) Bickley jet rescaled to L = 100 km, U0 = 40 km/d (0.46 m/s).
L, U0 = 100.0, 40.0
LX, LY = 1130.0, 250.0
K_N = 2 * np.pi * np.arange(1, 4) / LX
C_N = np.array([0.1446, 0.205, 0.461]) * U0
EPS = np.array([0.075, 0.4, 0.3])

CYCLE, DT = 10.0, 0.5
DX = 5.0
NX, NY = int(LX / DX), int(2 * LY / DX)
GX, GY = [a.ravel() for a in np.meshgrid(np.arange(NX) * DX + DX / 2,
                                         np.arange(NY) * DX - LY + DX / 2, indexing="ij")]

N_FLOATS, SPINUP, WIN_GAP, N_WIN = 60, 360.0, 90.0, 5
M_PART, BW, C_REL = 100, 10.0, 0.05

TRUE = dict(delta=0.03, phase=np.zeros(3), K=100.0)
SCENARIOS = {
    "correct":  dict(TRUE),
    "no_conv":  dict(TRUE, delta=0.0),
    "shifted":  dict(TRUE, phase=np.full(3, np.pi / 2)),
    "K_small":  dict(TRUE, K=TRUE["K"] / 4),
    "K_large":  dict(TRUE, K=TRUE["K"] * 4),
}
HORIZONS = (1, 2, 4, 8, 16, 36)
H_MAIN = 4
SWEEP = ("correct", "no_conv", "shifted")
ESTIMATORS = ("naive", "krige", "hajek", "dr")


def _theta(x, t, phase):
    return K_N * (np.asarray(x)[..., None] - C_N * np.asarray(t, float)[..., None]) + phase


def psi(x, y, t, phase=TRUE["phase"]):
    s2 = np.cosh(y / L) ** -2
    return U0 * L * (-np.tanh(y / L) + s2 * (EPS * np.cos(_theta(x, t, phase))).sum(-1))


def ohc(x, y, t, phase=TRUE["phase"]):
    return psi(x, y, t, phase) / (U0 * L)


def velocity(x, y, t, flow):
    th = _theta(x, t, flow["phase"])
    s2 = np.cosh(y / L) ** -2
    u = U0 * s2 * (1 + 2 * np.tanh(y / L) * (EPS * np.cos(th)).sum(-1))
    v = -U0 * L * s2 * (EPS * K_N * np.sin(th)).sum(-1)
    # delta * grad(psi): a divergent drift up the OHC gradient, the Langevin form of exp(beta Z) intensity
    return u + flow["delta"] * v, v - flow["delta"] * u


def advect(x, y, t0, days, flow, rng):
    """RK2 advection plus isotropic random walk of diffusivity K; periodic in x, reflecting in y."""
    t = np.broadcast_to(np.asarray(t0, float), x.shape).copy()
    sd = np.sqrt(2 * flow["K"] * DT)
    for _ in range(int(round(days / DT))):
        u1, v1 = velocity(x, y, t, flow)
        u2, v2 = velocity(x + 0.5 * DT * u1, y + 0.5 * DT * v1, t + 0.5 * DT, flow)
        x = (x + DT * u2 + sd * rng.standard_normal(x.shape)) % LX
        y = y + DT * v2 + sd * rng.standard_normal(y.shape)
        y = np.where(y > LY, 2 * LY - y, np.where(y < -LY, -2 * LY - y, y))
        t += DT
    return x, y


def window_cycles():
    return (SPINUP + WIN_GAP * np.arange(N_WIN)) / CYCLE


def real_floats(R, rng, n=N_FLOATS):
    """Profile positions (R, n, cycles + 1, 2) from uniform deployment at t = 0."""
    x, y = rng.uniform(0, LX, (R, n)), rng.uniform(-LY, LY, (R, n))
    out = [np.stack([x, y], -1)]
    for c in range(int(window_cycles()[-1])):
        x, y = advect(x, y, c * CYCLE, CYCLE, TRUE, rng)
        out.append(np.stack([x, y], -1))
    return np.stack(out, 2)


def clouds(anchors, t0, days, flow, rng, m=M_PART):
    """Advect m particles from each anchor (..., 2); returns (..., m, 2)."""
    x = np.repeat(anchors[..., None, 0], m, -1)
    y = np.repeat(anchors[..., None, 1], m, -1)
    x, y = advect(x, y, np.asarray(t0)[..., None], days, flow, rng)
    return np.stack([x, y], -1)


def density_grid(parts):
    """Mixture density (per km2) on the grid of the clouds (n_anchor, m, 2): sum of per-anchor KDEs."""
    p = parts.reshape(-1, 2)
    H, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[NX, NY], range=[[0, LX], [-LY, LY]])
    return ndimage.gaussian_filter(H, BW / DX, mode=("wrap", "reflect")) / (parts.shape[1] * DX**2)


def cell(xy):
    return (xy[..., 0] // DX).astype(int) % NX, np.clip(((xy[..., 1] + LY) // DX).astype(int), 0, NY - 1)


def truth_grid(t, phase=TRUE["phase"]):
    return ohc(GX, GY, t, phase)


# ----------------------------------------------------------------------------
# ordinary kriging with an anisotropic exponential covariance
# ----------------------------------------------------------------------------


def krige_params(times=None):
    """Sill and e-folding scales of the truth field, fixed once; the zonal scale is of the eddy part only."""
    times = window_cycles() * CYCLE if times is None else times
    Z = np.stack([truth_grid(t).reshape(NX, NY) for t in times])
    A = Z - Z.mean((1, 2), keepdims=True)
    s2 = (A**2).mean()
    E = Z - Z.mean(1, keepdims=True)
    rx = np.array([(E * np.roll(E, k, 1)).mean() / (E**2).mean() for k in range(NX // 2)])
    ry = np.array([(A[:, :, : NY - k] * A[:, :, k:]).mean() / s2 for k in range(NY)])
    return dict(s2=s2, lx=DX * np.argmax(rx < np.exp(-1)), ly=DX * np.argmax(ry < np.exp(-1)), nug=1e-3 * s2)


def _cov(a, b, kp):
    dx = np.abs(a[:, None, 0] - b[None, :, 0])
    dx = np.minimum(dx, LX - dx)
    return kp["s2"] * np.exp(-np.hypot(dx / kp["lx"], (a[:, None, 1] - b[None, :, 1]) / kp["ly"]))


def _ok_weights(p, rhs, kp):
    n = len(p)
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = _cov(p, p, kp) + kp["nug"] * np.eye(n)
    A[n, n] = 0.0
    return np.linalg.solve(A, np.vstack([rhs, np.ones((1, rhs.shape[1]))]))[:n]


_GRID = np.stack([GX, GY], -1)


def krige_block(p, z, kp):
    return _ok_weights(p, _cov(p, _GRID, kp).mean(1, keepdims=True), kp)[:, 0] @ z


def krige_point(p, z, q, kp):
    return _ok_weights(p, _cov(p, q, kp), kp).T @ z


# ----------------------------------------------------------------------------
# estimators for one replicate-window
# ----------------------------------------------------------------------------


def _weighted(lam_grid, n_anchor, xy, v):
    """Hajek mean of v over profiles in the covered region, the covered fraction, and the Kish ESS."""
    covered = lam_grid * (2 * LX * LY) / n_anchor >= C_REL
    ix, iy = cell(xy)
    keep = covered[ix, iy]
    w = 1 / lam_grid[ix, iy][keep]
    return (w @ v[keep]) / w.sum(), covered.mean(), w.sum() ** 2 / (w**2).sum()


def seq_estimates(xy, z, parts, kp):
    """Hajek and cross-fit DR (kriging map from the other fold + Hajek-weighted residual correction)."""
    hajek, cover, ess = _weighted(density_grid(parts), len(parts), xy, z)
    fold = np.arange(len(xy)) % 2 == 0
    dr = []
    for a, b in ((fold, ~fold), (~fold, fold)):
        block = krige_block(xy[a], z[a], kp)
        r = z[b] - krige_point(xy[a], z[a], xy[b], kp)
        corr, cov_b, _ = _weighted(density_grid(parts[b]), b.sum(), xy[b], r)
        dr.append(block + cov_b * corr)
    return dict(hajek=hajek, dr=np.mean(dr), cover=cover, ess=ess)



def pit(xy, parts):
    """Highest-density-region PIT of each real landing in its own cloud: Uniform(0,1) iff calibrated."""
    def dens(q, p):
        dx = np.abs(q[:, :, None, 0] - p[:, None, :, 0])
        d2 = np.minimum(dx, LX - dx) ** 2 + (q[:, :, None, 1] - p[:, None, :, 1]) ** 2
        return np.exp(-d2 / (2 * BW**2)).sum(-1)
    m = parts.shape[1]
    f_real = dens(xy[:, None], parts)[:, 0] / m
    return ((dens(parts, parts) - 1) / (m - 1) >= f_real[:, None]).mean(-1)  # leave-one-out


def run(R=40, seed=1, path=None):
    """Full experiment: per (rep, window, scenario, horizon, estimator) estimates, and PIT values."""
    rng = np.random.default_rng(seed)
    kp = krige_params()
    traj = real_floats(R, rng)
    cw = window_cycles().astype(int)
    truth = np.array([truth_grid(c * CYCLE).mean() for c in cw])
    rows, pits = [], []
    for r in range(R):
        for w, c in enumerate(cw):
            xy = traj[r, :, c]
            z = ohc(xy[:, 0], xy[:, 1], c * CYCLE)
            rows += [dict(rep=r, window=w, truth=truth[w], scenario=s, h=np.nan, estimator=e, estimate=v)
                     for s in SCENARIOS for e, v in (("naive", z.mean()), ("krige", krige_block(xy, z, kp)))]
    for s, flow in SCENARIOS.items():
        for h in (HORIZONS if s in SWEEP else (H_MAIN,)):
            anchors = traj[:, :, cw - h].transpose(0, 2, 1, 3)
            parts = clouds(anchors, ((cw - h) * CYCLE)[None, :, None], h * CYCLE, flow, rng)
            for r in range(R):
                for w, c in enumerate(cw):
                    xy = traj[r, :, c]
                    est = seq_estimates(xy, ohc(xy[:, 0], xy[:, 1], c * CYCLE), parts[r, w], kp)
                    base = dict(rep=r, window=w, truth=truth[w], scenario=s, h=h, cover=est["cover"], ess=est["ess"])
                    rows += [dict(base, estimator=e, estimate=est[e]) for e in ("hajek", "dr")]
                    if h == H_MAIN:
                        pits.append(pd.DataFrame(dict(scenario=s, rep=r, window=w, pit=pit(xy, parts[r, w]))))
            print(s, h, flush=True)
    df = pd.DataFrame(rows).assign(bias=lambda d: d.estimate - d.truth)
    pt = pd.concat(pits, ignore_index=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        pt.to_parquet(path.replace("results", "pit"))
    return df, pt


def load_or_run(R=40, seed=1):
    path = os.path.join(OUT_DIR, f"results_R{R}_s{seed}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path), pd.read_parquet(path.replace("results", "pit"))
    return run(R, seed, path)


# ----------------------------------------------------------------------------
# presentation
# ----------------------------------------------------------------------------

COL = dict(hajek="#2a78d6", dr="#eb6834", naive="#52514e", krige="#52514e")
LAB = dict(naive="simple mean", krige="kriging", hajek="sequential Hajek", dr="cross-fit DR")
SCEN_LAB = dict(correct="correct model", no_conv="model misses convergence", shifted="model eddies displaced",
                K_small="noise too small (K/4)", K_large="noise too large (4K)")
ANCHOR_COL = ("#000000", "#008300", "#eda100")


def _field(ax, t, vmax=1.3, alpha=1.0):
    Z = truth_grid(t).reshape(NX, NY).T
    ext = [0, LX, -LY, LY]
    im = ax.imshow(Z, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", alpha=alpha)
    ax.contour(Z, levels=np.linspace(-1.2, 1.2, 9), extent=ext, colors="0.35", linewidths=0.4)
    ax.set(xlabel="x (km)", ylabel="y (km)")
    return im


def plot_setup(traj, figsize=(12, 3.8)):
    cw = window_cycles().astype(int)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw=dict(width_ratios=[4, 1, 1]))
    im = _field(ax[0], cw[0] * CYCLE)
    ax[0].plot(*traj[0, :, cw[0]].T, "o", ms=3.5, mfc="k", mec="w", mew=0.5)
    ax[0].set_title("OHC anomaly Z and real floats (one replicate, first window)")
    fig.colorbar(im, ax=ax[0], pad=0.01, label="Z")
    yb = np.linspace(-LY, LY, 21)
    yc = 0.5 * (yb[1:] + yb[:-1])
    zbar = np.mean([truth_grid(c * CYCLE).reshape(NX, NY).mean(0) for c in cw], 0)
    ax[1].plot(zbar, np.arange(NY) * DX - LY + DX / 2, color="k", lw=2)
    ax[1].set(xlabel="zonal-mean Z", title="Warm south")
    rel = np.histogram(traj[:, :, cw, 1].ravel(), yb)[0] / (traj[:, :, cw].size / 2 / len(yc))
    ax[2].step(rel, yc, where="mid", color=COL["hajek"], lw=2)
    ax[2].axvline(1, color="0.5", ls="--", lw=1)
    ax[2].set(xlabel="float density / uniform", title="Floats pile up south")
    fig.tight_layout()
    return fig


def example_clouds(anchors=((300.0, 0.0), (300.0, -120.0), (300.0, 200.0)), t0=3600.0,
                   horizons=(1, 4, 16), flow=TRUE, m=400, seed=3):
    rng = np.random.default_rng(seed)
    a = np.array(anchors)
    return a, t0, {h: clouds(a, np.full(len(a), t0), h * CYCLE, flow, rng, m) for h in horizons}


def plot_clouds(example=None, figsize=(13, 3.6)):
    a, t0, cl = example_clouds() if example is None else example
    fig, ax = plt.subplots(1, len(cl), figsize=figsize, sharey=True)
    for axi, (h, p) in zip(ax, cl.items()):
        _field(axi, t0 + h * CYCLE, alpha=0.45)
        for k in range(len(a)):
            axi.plot(*p[k].T, ".", ms=3, color=ANCHOR_COL[k], alpha=0.8)
            axi.plot(*a[k], "*", ms=13, color=ANCHOR_COL[k], mec="k", mew=0.8)
        axi.set_title(f"h = {h} cycle{'s' if h > 1 else ''} ({int(h * CYCLE)} d)")
    for axi in ax[1:]:
        axi.set_ylabel("")
    fig.tight_layout()
    return fig


def snapshot(traj, rep=0, window=0, h=H_MAIN, flow=TRUE, seed=5, m=M_PART):
    """One replicate-window: stacked-cloud density, profile weights, and the three box-mean numbers."""
    rng = np.random.default_rng(seed)
    c = int(window_cycles()[window])
    parts = clouds(traj[rep, :, c - h], np.full(traj.shape[1], (c - h) * CYCLE), h * CYCLE, flow, rng, m)
    lam = density_grid(parts)
    xy = traj[rep, :, c]
    z = ohc(xy[:, 0], xy[:, 1], c * CYCLE)
    ix, iy = cell(xy)
    share = (1 / lam[ix, iy]) / (1 / lam[ix, iy]).sum()
    return dict(h=h, t=c * CYCLE, parts=parts, lam=lam, xy=xy, z=z, share=share, naive=z.mean(),
                weighted=share @ z, truth=truth_grid(c * CYCLE).mean())


def plot_weights(traj, rep=0, window=0, h=H_MAIN, flow=TRUE, seed=5, figsize=(12, 4.0), snap=None):
    sn = snapshot(traj, rep, window, h, flow, seed) if snap is None else snap
    h, lam, xy, z, share = sn["h"], sn["lam"], sn["xy"], sn["z"], sn["share"]
    rel = lam * (2 * LX * LY) / len(xy)
    fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[3, 1.3]))
    im = ax[0].imshow(np.log10(np.maximum(rel, 1e-3)).T, origin="lower", extent=[0, LX, -LY, LY],
                      cmap="Blues", vmin=-2, vmax=1, aspect="auto")
    ax[0].contour(rel.T, levels=[C_REL], extent=[0, LX, -LY, LY], colors="k", linewidths=1.2)
    ax[0].scatter(*xy.T, s=600 * share, c=COL["dr"], edgecolors="k", linewidths=0.5)
    fig.colorbar(im, ax=ax[0], pad=0.01, label="log10 expected density / uniform")
    ax[0].set(xlabel="x (km)", ylabel="y (km)",
              title=f"Stacked clouds (h = {h}); marker area = weight; black line = gap edge")
    ax[1].scatter(z, share * len(xy), s=18, color=COL["dr"], edgecolors="k", linewidths=0.4)
    ax[1].axhline(1, color="0.5", ls="--", lw=1)
    ax[1].set(xlabel="Z at profile", ylabel="weight / equal weight", yscale="log",
              title="Warm profiles count less")
    fig.tight_layout()
    return fig


def rep_summary(df):
    """Window-averaged bias per replicate, then mean, MC standard error and sd across replicates."""
    per_rep = (df.groupby(["scenario", "h", "estimator", "rep"], dropna=False)
                 .agg(bias=("bias", "mean"), cover=("cover", "mean"), ess=("ess", "mean")).reset_index())
    sd_win = df.groupby(["scenario", "h", "estimator"], dropna=False).bias.std().rename("sd")
    out = (per_rep.groupby(["scenario", "h", "estimator"], dropna=False)
                  .agg(bias=("bias", "mean"), se=("bias", "sem"), cover=("cover", "mean"), ess=("ess", "mean")))
    return out.join(sd_win).assign(rmse=lambda d: np.hypot(d.bias, d.sd)).reset_index()


def summary_table(df, h=H_MAIN):
    s = rep_summary(df)
    s = s[(s.h == h) | s.h.isna()]
    s = s[(s.scenario == "correct") | s.h.notna()]
    order = {k: i for i, k in enumerate(SCENARIOS)}
    s = s.assign(o1=lambda d: np.where(d.h.isna(), -1, d.scenario.map(order)),
                 o2=lambda d: d.estimator.map({e: i for i, e in enumerate(ESTIMATORS)})).sort_values(["o1", "o2"])
    s = s.assign(scenario=lambda d: np.where(d.h.isna(), "(any)", d.scenario.map(SCEN_LAB)),
                 estimator=lambda d: d.estimator.map(LAB))
    return s[["scenario", "estimator", "bias", "se", "sd", "rmse", "cover", "ess"]]


def plot_bias(df, h=H_MAIN, figsize=(9, 4.2)):
    s = rep_summary(df)
    ref = s[s.h.isna() & (s.scenario == "correct")].set_index("estimator")
    s = s[s.h == h]
    fig, ax = plt.subplots(figsize=figsize)
    for e, ls in (("naive", "--"), ("krige", ":")):
        ax.axvline(ref.loc[e, "bias"], color=COL[e], ls=ls, lw=1.5, label=LAB[e])
    ax.axvline(0, color="k", lw=0.8)
    scen = list(SCENARIOS)
    for j, (e, mk) in enumerate((("hajek", "o"), ("dr", "s"))):
        d = s[s.estimator == e].set_index("scenario").loc[scen]
        ypos = np.arange(len(scen)) + (j - 0.5) * 0.25
        ax.errorbar(d.bias, ypos, xerr=1.96 * d.se, fmt=mk, ms=8, color=COL[e], lw=2, capsize=3, label=LAB[e])
    ax.set_yticks(range(len(scen)), [SCEN_LAB[k] for k in scen])
    ax.invert_yaxis()
    ax.set(xlabel=f"bias of the box-mean estimate (h = {h} cycles; bars = 95% MC interval)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def plot_horizon(df, figsize=(12, 7.5)):
    s = rep_summary(df)
    ref = s[s.h.isna() & (s.scenario == "correct")].set_index("estimator")
    fig, ax = plt.subplots(3, len(SWEEP), figsize=figsize, sharex=True, sharey="row")
    for j, sc in enumerate(SWEEP):
        for e, mk in (("hajek", "o"), ("dr", "s")):
            d = s[(s.scenario == sc) & (s.estimator == e)].sort_values("h")
            ax[0, j].errorbar(d.h, d.bias, yerr=1.96 * d.se, marker=mk, color=COL[e], lw=2, ms=6, capsize=2, label=LAB[e])
            ax[1, j].plot(d.h, d.sd, marker=mk, color=COL[e], lw=2, ms=6)
        ax[2, j].plot(d.h, d.cover, marker="o", color="k", lw=2, ms=6)
        for e, ls in (("naive", "--"), ("krige", ":")):
            ax[0, j].axhline(ref.loc[e, "bias"], color=COL[e], ls=ls, lw=1.2, label=LAB[e])
            ax[1, j].axhline(ref.loc[e, "sd"], color=COL[e], ls=ls, lw=1.2)
        ax[0, j].axhline(0, color="k", lw=0.8)
        ax[0, j].set_title(SCEN_LAB[sc])
        ax[2, j].set(xscale="log", xlabel="horizon h (cycles of 10 d)")
        ax[2, j].set_xticks(HORIZONS, [str(h) for h in HORIZONS])
    ax[0, 0].set_ylabel("bias")
    ax[1, 0].set_ylabel("sd across windows")
    ax[2, 0].set_ylabel("covered fraction")
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_pit(pt, figsize=(13, 2.8)):
    scen = list(SCENARIOS)
    fig, ax = plt.subplots(1, len(scen), figsize=figsize, sharey=True)
    for axi, sc in zip(ax, scen):
        axi.hist(pt[pt.scenario == sc].pit, bins=10, range=(0, 1), density=True,
                 color=COL["hajek"], edgecolor="w", linewidth=2)
        axi.axhline(1, color="k", ls="--", lw=1)
        axi.set(title=SCEN_LAB[sc], xlabel="PIT")
        axi.title.set_fontsize(9)
    ax[0].set_ylabel("density")
    fig.tight_layout()
    return fig


def mc_check(R=12, h=H_MAIN, settings=((100, 10.0), (400, 10.0), (1600, 10.0), (400, 20.0)), seed=11):
    """Hajek bias under the correct model against particle count M and KDE bandwidth (cached)."""
    global BW
    path = os.path.join(OUT_DIR, f"mc_check_R{R}_s{seed}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    tr = real_floats(R, np.random.default_rng(seed))
    cw = window_cycles().astype(int)
    rows, bw0 = [], BW
    for m, bw in settings:
        BW = bw
        parts = clouds(tr[:, :, cw - h].transpose(0, 2, 1, 3), ((cw - h) * CYCLE)[None, :, None], h * CYCLE,
                       TRUE, np.random.default_rng(m), m=m)
        for r in range(R):
            for w, c in enumerate(cw):
                xy = tr[r, :, c]
                est = _weighted(density_grid(parts[r, w]), tr.shape[1], xy, ohc(xy[:, 0], xy[:, 1], c * CYCLE))[0]
                rows.append(dict(M=m, bandwidth_km=bw, rep=r, bias=est - truth_grid(c * CYCLE).mean()))
    BW = bw0
    out = (pd.DataFrame(rows).groupby(["M", "bandwidth_km", "rep"]).bias.mean()
             .groupby(["M", "bandwidth_km"]).agg(bias="mean", se="sem").reset_index())
    out.to_parquet(path)
    return out


def rect_weights(rects, pts):
    """Weights from flat rectangular clouds (x0, x1, y0, y1) in the unit box: 1 / summed cloud density."""
    r, p = np.asarray(rects, float), np.asarray(pts, float)
    inside = ((p[:, None, 0] >= r[None, :, 0]) & (p[:, None, 0] <= r[None, :, 1])
              & (p[:, None, 1] >= r[None, :, 2]) & (p[:, None, 1] <= r[None, :, 3]))
    dens = (inside / ((r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2]))).sum(1)
    return pd.DataFrame(dict(cloud_area=(r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2]), density=dens, weight=1 / dens))


def plot_rects(rects, pts, names, weights, warm_south=False, figsize=(5, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    if warm_south:
        ax.axhspan(0, 0.5, color="#e34948", alpha=0.15)
        ax.axhspan(0.5, 1, color="#2a78d6", alpha=0.15)
        ax.text(0.02, 0.03, "warm (Z = +1)", fontsize=9)
        ax.text(0.02, 0.94, "cold (Z = -1)", fontsize=9)
    for k, ((x0, x1, y0, y1), (px, py), n, w) in enumerate(zip(rects, pts, names, weights)):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2,
                                   ls=("-", "--", ":", "-.")[k % 4], ec="k"))
        ax.plot(px, py, "o", ms=9, color="k")
        ax.annotate(f"{n}: weight {w:.0%}", (px, py), xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect="equal")
    ax.grid(False)
    fig.tight_layout()
    return fig


def dr_story(snap, kp=None, mapper=0):
    """One cross-fit direction on a snapshot: team A = even floats, team B = odd; mapper 0 means A maps, B checks."""
    kp = krige_params() if kp is None else kp
    xy, z, parts = snap["xy"], snap["z"], snap["parts"]
    a = np.arange(len(xy)) % 2 == mapper
    b = ~a
    m = krige_point(xy[a], z[a], _GRID, kp)
    r = z[b] - krige_point(xy[a], z[a], xy[b], kp)
    lam = density_grid(parts[b])
    covered = lam * (2 * LX * LY) / b.sum() >= C_REL
    ix, iy = cell(xy[b])
    w = np.zeros(len(ix))
    w[covered[ix, iy]] = 1 / lam[ix, iy][covered[ix, iy]]
    share = w / w.sum()
    out = dict(snap, a=a, b=b, names=("AB", "BA")[mapper], marks=(("o", "^"), ("^", "o"))[mapper], map=m, map_mean=m.mean(), resid=r, lam_b=lam, covered=covered, share_b=share,
               cover=covered.mean(), corr_plain=r.mean(), corr_w=covered.mean() * (share @ r),
               truth_grid=truth_grid(snap["t"]))
    out["final"] = out["map_mean"] + out["corr_w"]
    out["dr_both"] = seq_estimates(xy, z, parts, kp)["dr"]
    return out


def plot_dr_story(st, figsize=(13, 7.2), zmax=1.3, rmax=0.8):
    ext = [0, LX, -LY, LY]
    img = lambda ax, f, **k: ax.imshow(f.reshape(NX, NY).T, origin="lower", extent=ext, aspect="auto", **k)
    xa, xb = st["xy"][st["a"]], st["xy"][st["b"]]
    za, zb, r = st["z"][st["a"]], st["z"][st["b"]], st["resid"]
    zk = dict(cmap="RdBu_r", vmin=-zmax, vmax=zmax)
    rk = dict(cmap="PuOr_r", vmin=-rmax, vmax=rmax)
    pk = dict(edgecolors="k", linewidths=0.6)
    fig, ax = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    ax = ax.ravel()
    im = img(ax[0], st["truth_grid"], **zk)
    ax[0].set_title(f"1. True field (box mean {st['truth']:+.2f})")
    fig.colorbar(im, ax=ax[0], pad=0.01, label="Z")
    (na, nb), (ma, mb) = st["names"], st["marks"]
    ax[1].scatter(*xa.T, c=za, marker=ma, s=45, **zk, **pk, label=f"team {na} (builds map)")
    sc = ax[1].scatter(*xb.T, c=zb, marker=mb, s=45, **zk, **pk, label=f"team {nb} (checks map)")
    ax[1].set_title(f"2. Observations (simple average {st['naive']:+.2f})")
    ax[1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.colorbar(sc, ax=ax[1], pad=0.01, label="Z")
    rel = st["lam_b"] * (2 * LX * LY) / st["b"].sum()
    im = img(ax[2], np.log10(np.maximum(rel, 1e-3)).ravel(), cmap="Blues", vmin=-2, vmax=1)
    ax[2].contour(rel.T, levels=[C_REL], extent=ext, colors="k", linewidths=1.2)
    ax[2].scatter(*xb.T, marker=mb, s=40, c=COL["dr"], **pk)
    ax[2].set_title(f"3. Team {nb}'s possible locations (stacked clouds)")
    fig.colorbar(im, ax=ax[2], pad=0.01, label="log10 density / uniform")
    im = img(ax[3], st["map"], **zk)
    ax[3].scatter(*xa.T, c=za, marker=ma, s=40, **zk, **pk)
    ax[3].set_title(f"4. Kriging map from team {na} (mean {st['map_mean']:+.2f})")
    fig.colorbar(im, ax=ax[3], pad=0.01, label="Z")
    im = img(ax[4], st["truth_grid"] - st["map"], alpha=0.45, **rk)
    ax[4].scatter(*xb.T, c=r, marker=mb, s=50, **rk, **pk)
    ax[4].set_title(f"5. Residuals at team {nb} (plain average {st['corr_plain']:+.2f})")
    fig.colorbar(im, ax=ax[4], pad=0.01, label="measured - map")
    img(ax[5], st["truth_grid"] - st["map"], alpha=0.2, **rk)
    ax[5].scatter(*xb.T, c=r, marker=mb, s=1500 * st["share_b"] + 2, **rk, **pk)
    ax[5].set_title(f"6. Reweighted residuals (correction {st['corr_w']:+.2f})")
    fig.colorbar(im, ax=ax[5], pad=0.01, label="measured - map")
    for k in (0, 3):
        ax[k].set_ylabel("y (km)")
    for k in (3, 4, 5):
        ax[k].set_xlabel("x (km)")
    for a_ in ax:
        a_.grid(False)
    fig.tight_layout()
    return fig


def plot_dr_summary(st1, st2, figsize=(9, 3.6)):
    """From the simple average to the final DR estimate: each direction's map, then map + weighted check."""
    rows = [("simple average", st1["naive"], None),
            (f"team {st1['names'][0]} map", st1["map_mean"], None),
            (f"team {st1['names'][0]} map + team {st1['names'][1]} check", st1["final"], st1["map_mean"]),
            (f"team {st2['names'][0]} map", st2["map_mean"], None),
            (f"team {st2['names'][0]} map + team {st2['names'][1]} check", st2["final"], st2["map_mean"]),
            ("final: average of the two", 0.5 * (st1["final"] + st2["final"]), None)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(st1["truth"], color="k", lw=1)
    ax.text(st1["truth"], -0.7, " truth", fontsize=9, va="center")
    for i, (lab, v, frm) in enumerate(rows):
        if frm is not None:
            ax.annotate("", (v, i), (frm, i), arrowprops=dict(arrowstyle="->", color=COL["dr"], lw=1.5))
        ax.plot(v, i, "o", ms=9, color=COL["dr"] if "check" in lab or "final" in lab else COL["naive"])
        ax.text(v, i - 0.32, f"{v:+.2f}", ha="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -1)
    ax.set_xlabel("box-mean estimate (truth = %.2f)" % st1["truth"])
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
