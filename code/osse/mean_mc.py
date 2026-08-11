"""Monte-Carlo sampling distributions of the residual-field mean.

The estimand is the domain mean Zbar_k = int Z_k dP, with P the normalised
area x time measure on box B's wet native cells. A design {x_i} of size n gives
Zhat = mean_i Z_k(x_i) and bias B = Zhat - Zbar_k.

Three ensembles are compared, each advective design paired with synthetic
designs of exactly its own n so that only the point *locations* differ:

  advective  the advected uniform deployment, from sampled_residuals_*.csv
  U-full     n iid draws, cell ~ area, doy ~ Uniform{0..364}; E[B] = 0
  U-doy      n iid draws over cells, holding the advective design's own doys,
             which isolates the bias attributable to *when* the array samples

E[B] for U-doy has a closed form -- mean_i Zbar_k(doy_i) - Zbar_k, where
Zbar_k(d) is the spatial mean on day d -- so it is computed exactly rather than
simulated. Values are J/m^2 throughout; J_TO_GJ converts for display.

    python code/osse/mean_mc.py [--box boxB]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg

J_TO_GJ = 1e-9
YEARS = list(cfg.ANALYSIS_YEARS)
ZCOLS = [f"z_{y}" for y in YEARS]


def load_field(box="boxB", tag=None):
    """Flattened residual field, the wet-cell index, and the P weights on it."""
    z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{tag or box}.nc"))["z"]
    area = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    zf = z.values.reshape(z.sizes["year"], z.sizes["doy"], -1)
    a = area.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return zf, wet, a[wet] / a[wet].sum()


def domain_means(zf, wet, w):
    """Zbar_k and sd_P(Z_k), plus the spatial mean on each day of year.

    Both moments are taken under P, so sd_Z is area-weighted; the cached
    build_residual_field value is unweighted and differs by well under 1%.
    """
    by_doy = np.empty((zf.shape[0], zf.shape[1]))
    m2 = np.empty_like(by_doy)
    for i in range(zf.shape[0]):
        zw = zf[i][:, wet].astype(np.float64)
        by_doy[i] = zw @ w
        m2[i] = (zw * zw) @ w
    zbar = by_doy.mean(axis=1)
    return pd.DataFrame({"year": YEARS, "zbar": zbar,
                         "sd_Z": np.sqrt(m2.mean(axis=1) - zbar**2)}), by_doy


def advective_estimates(box="boxB", tag="uniform_mc", path=None):
    """Per-design Zhat, sample size, and the float-clustered design effect.

    The design effect is sum_f (cluster sum of centred z)^2 / sum_i (centred
    z)^2: consecutive cycles of one float are near-replicates, so it is the
    factor by which the effective sample size falls short of n.
    """
    path = path or os.path.join(cfg.OUT_DIR, f"sampled_residuals_{tag}.csv")
    df = pd.read_csv(path, usecols=["vel_year", "rep", "float_id", "doy"] + ZCOLS)
    key = ["vel_year", "rep"]
    g = df.groupby(key, sort=True)

    out = g[ZCOLS].mean().rename(columns=lambda c: c.replace("z_", "zhat_"))
    out["n"] = g.size()
    out["n_floats"] = g["float_id"].nunique()

    ctr = df[ZCOLS] - g[ZCOLS].transform("mean")
    ss = ctr.pow(2).groupby([df[k] for k in key]).sum()
    cl = ctr.groupby([df[k] for k in key] + [df["float_id"]]).sum()
    ss_cl = cl.pow(2).groupby(level=[0, 1]).sum()
    deff = (ss_cl / ss.where(ss > 0)).rename(columns=lambda c: c.replace("z_", "deff_"))

    return out.join(deff).reset_index(), df[key + ["doy"]]


def temporal_component(doys, by_doy, zbar):
    """Exact E[B] for a design that keeps its own doys but samples space from P."""
    key = ["vel_year", "rep"]
    per_doy = pd.DataFrame(by_doy.T[doys["doy"].to_numpy().astype(int)],
                           columns=[f"temporal_{y}" for y in YEARS],
                           index=doys.index)
    m = per_doy.groupby([doys[k] for k in key]).mean()
    return (m - zbar).reset_index()


def uniform_estimates(zf, wet, w, n_vec, doys=None, seed=0):
    """iid draws from P (or from P in space at fixed doys), n_vec points each."""
    rng = np.random.default_rng(seed)
    n_vec = np.asarray(n_vec, dtype=np.int64)
    total = int(n_vec.sum())
    cells = wet[np.searchsorted(np.cumsum(w), rng.random(total)).clip(max=len(w) - 1)]
    d = (rng.integers(0, cfg.DAYS_PER_YEAR, total) if doys is None
         else np.asarray(doys, dtype=int))
    vals = zf[:, d, cells]
    off = np.concatenate([[0], np.cumsum(n_vec)[:-1]])
    return (np.add.reduceat(vals, off, axis=1) / n_vec).T


def bias_grid(adv, unif, zbar):
    """Long cells-table summary of every (analysis year, velocity year) cell."""
    zb = pd.Series(zbar["zbar"].to_numpy(), index=YEARS)
    rows = []
    for k in YEARS:
        for l, ga in adv.groupby("vel_year"):
            gu = unif[unif["vel_year"] == l]
            b = ga[f"zhat_{k}"] - zb[k]
            rows.append({
                "analysis_year": k, "vel_year": l, "aligned": k == l,
                "n_designs": len(ga), "mean_n": ga["n"].mean(),
                "mean_B": b.mean(), "se_B": b.sem(), "sd_B": b.std(ddof=1),
                "mean_temporal": ga[f"temporal_{k}"].mean(),
                "mean_spatial": (b - ga[f"temporal_{k}"]).mean(),
                "deff": ga[f"deff_{k}"].mean(),
                "unif_mean_B": (gu[f"zhat_{k}"] - zb[k]).mean(),
                "unif_sd_B": (gu[f"zhat_{k}"] - zb[k]).std(ddof=1),
            })
    return pd.DataFrame(rows)


# ------------------------------- plotting --------------------------------

def _hist(ax, x, **kw):
    kw.setdefault("bins", 24)
    kw.setdefault("color", "0.35")
    ax.hist(np.asarray(x) * J_TO_GJ, density=True, **kw)


def plot_reference_hists(unif, zbar, figsize=(13, 3.0)):
    """Uniform-sampling estimates around the truth, one panel per analysis year."""
    import matplotlib.pyplot as plt
    zb = pd.Series(zbar["zbar"].to_numpy(), index=YEARS)
    fig, axes = plt.subplots(1, len(YEARS), figsize=figsize, sharey=True)
    for ax, k in zip(axes, YEARS):
        _hist(ax, unif[f"zhat_{k}"])
        ax.axvline(zb[k] * J_TO_GJ, color="crimson", lw=1.6)
        ax.set_title(str(k), fontsize=10)
        ax.set_xlabel("Zhat (GJ/m2)")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_sample_sizes(adv, figsize=(11, 3.4)):
    """Profiles and distinct floats retained in the box, by velocity year."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    years = sorted(adv["vel_year"].unique())
    for ax, col, lab in ((a1, "n", "profiles per design"),
                         (a2, "n_floats", "distinct floats per design")):
        ax.boxplot([adv.loc[adv["vel_year"] == y, col] for y in years],
                   tick_labels=[str(y) for y in years], showfliers=False)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    return fig


def plot_advective_grid(adv, unif, zbar, figsize=(18, 9.5)):
    """Zhat by analysis year (rows) x velocity year (columns).

    Aligned cells (velocity year == analysis year) are outlined in red; the
    dashed curve is the n-matched uniform reference for the same cell.
    """
    import matplotlib.pyplot as plt
    zb = pd.Series(zbar["zbar"].to_numpy(), index=YEARS)
    vy = sorted(adv["vel_year"].unique())
    fig, axes = plt.subplots(len(YEARS), len(vy), figsize=figsize,
                             sharex="row", sharey="row", squeeze=False)
    for i, k in enumerate(YEARS):
        lo, hi = np.percentile(adv[f"zhat_{k}"] * J_TO_GJ, [0.2, 99.8])
        pad = 0.1 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 26)
        for j, l in enumerate(vy):
            ax = axes[i, j]
            a = adv.loc[adv["vel_year"] == l, f"zhat_{k}"]
            u = unif.loc[unif["vel_year"] == l, f"zhat_{k}"]
            ax.hist(a * J_TO_GJ, bins=bins, density=True, color="0.35")
            ax.hist(u * J_TO_GJ, bins=bins, density=True, histtype="step",
                    color="tab:blue", lw=1.2, ls="--")
            ax.axvline(zb[k] * J_TO_GJ, color="crimson", lw=1.2)
            ax.set_xlim(bins[0], bins[-1])
            ax.tick_params(labelsize=6)
            if k == l:
                for s in ax.spines.values():
                    s.set(color="crimson", linewidth=2.0)
            if i == 0:
                ax.set_title(str(l), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Z {k}", fontsize=9)
            if i == len(YEARS) - 1:
                ax.set_xlabel("Zhat (GJ/m2)", fontsize=7)
    fig.suptitle("rows: analysis year of the field   columns: velocity year of the flow",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_bias_heatmap(grid, figsize=(7.6, 6.4)):
    """Mean B per cell in GJ/m2, annotated with its Monte-Carlo standard error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    m = grid.pivot(index="vel_year", columns="analysis_year", values="mean_B") * J_TO_GJ
    s = grid.pivot(index="vel_year", columns="analysis_year", values="se_B") * J_TO_GJ
    hi = float(np.abs(m.to_numpy()).max())
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi))
    ax.set_xticks(range(m.shape[1]), [str(c) for c in m.columns])
    ax.set_yticks(range(m.shape[0]), [str(r) for r in m.index])
    ax.set_xlabel("analysis year (field)"); ax.set_ylabel("velocity year (flow)")
    for i, l in enumerate(m.index):
        for j, k in enumerate(m.columns):
            ax.text(j, i, f"{m.iat[i, j]:+.3f}\n({s.iat[i, j]:.3f})", ha="center",
                    va="center", fontsize=7,
                    fontweight="bold" if l == k else "normal")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean B (GJ/m2)")
    fig.tight_layout()
    return fig


def plot_decomposition(grid, figsize=(13, 3.4)):
    """Total bias split into its temporal-coverage and spatial-given-time parts."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(YEARS), figsize=figsize, sharey=True)
    for ax, k in zip(axes, YEARS):
        g = grid[grid["analysis_year"] == k].sort_values("vel_year")
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["mean_temporal"] * J_TO_GJ, 0.4, color="tab:orange",
               label="temporal")
        ax.bar(x + 0.2, g["mean_spatial"] * J_TO_GJ, 0.4, color="tab:blue",
               label="spatial | time")
        ax.plot(x, g["mean_B"] * J_TO_GJ, "k.", ms=7, label="total")
        for xi, al in zip(x, g["aligned"]):
            if al:
                ax.axvspan(xi - 0.5, xi + 0.5, color="crimson", alpha=0.12)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x, [str(v) for v in g["vel_year"]], rotation=90, fontsize=7)
        ax.set_title(str(k), fontsize=10)
    axes[0].set_ylabel("GJ/m2")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--tag", default="uniform_mc")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    zf, wet, w = load_field(a.box)
    zbar, by_doy = domain_means(zf, wet, w)
    cached = pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{a.box}.csv"))
    assert np.allclose(zbar["zbar"], cached["zbar"], rtol=1e-5, atol=1e3), "Zbar mismatch"
    assert np.allclose(zbar["sd_Z"], cached["sd_Z"], rtol=0.02), "sd_Z mismatch"

    adv, doys = advective_estimates(a.box, a.tag)
    adv = adv.merge(temporal_component(doys, by_doy, zbar["zbar"].to_numpy()),
                    on=["vel_year", "rep"])

    n_vec = adv["n"].to_numpy()
    zu = uniform_estimates(zf, wet, w, n_vec, seed=a.seed)
    unif = adv[["vel_year", "rep", "n"]].copy()
    unif[[f"zhat_{y}" for y in YEARS]] = zu

    grid = bias_grid(adv, unif, zbar)
    for name, df in (("advective", adv), ("uniform", unif), ("grid", grid)):
        df.to_csv(os.path.join(cfg.OUT_DIR, f"mean_mc_{name}_{a.box}.csv"), index=False)

    zb = zbar["zbar"].to_numpy()
    bu = zu - zb
    print(f"{len(adv)} designs, n {n_vec.min()}-{n_vec.max()} (median {int(np.median(n_vec))})")
    print("\nU-full check: mean B +/- MC se, and sd vs sigma*sqrt(mean 1/n)   [GJ/m2]")
    for j, k in enumerate(YEARS):
        pred = zbar["sd_Z"][j] * np.sqrt(np.mean(1.0 / n_vec))
        print(f"  {k}: {bu[:, j].mean()*J_TO_GJ:+.4f} +/- "
              f"{bu[:, j].std(ddof=1)/np.sqrt(len(bu))*J_TO_GJ:.4f}   "
              f"sd {bu[:, j].std(ddof=1)*J_TO_GJ:.3f} vs {pred*J_TO_GJ:.3f}")
    d = grid[grid["aligned"]]
    print("\naligned cells: mean B (MC se), temporal share   [GJ/m2]")
    for _, r in d.iterrows():
        print(f"  {int(r.analysis_year)}: {r.mean_B*J_TO_GJ:+.4f} ({r.se_B*J_TO_GJ:.4f})"
              f"   temporal {r.mean_temporal*J_TO_GJ:+.4f}")
    print(f"\nwrote mean_mc_{{advective,uniform,grid}}_{a.box}.csv")
