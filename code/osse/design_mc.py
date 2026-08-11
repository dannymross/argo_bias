"""Monte-Carlo summaries of the design distribution w, per velocity year.

Each seeding omega_{l,d} ~ D advected through V_l gives one draw of w^(l,d), so
many seedings per year estimate the law of w^(l). Two things are then separable
that a single replicate cannot separate:

  E[w^(l)] - p        advective sampling bias: where the flow concentrates a
                      deployment that was uniform with respect to P
  sd_l(E[w^(l)])      the D1a violation: how far the design mean moves between
                      velocity years, judged against the Monte-Carlo standard
                      error of each year's mean

Reads the tagged per-year files written by submit_uniform_mc.slurm.

    python code/osse/design_mc.py [--box boxB] [--variant uniform]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg


def load_mc_weights(box="boxB", variant="uniform", out_dir=None, pattern=None):
    out_dir = out_dir or cfg.OUT_DIR
    pattern = pattern or os.path.join(out_dir, f"design_weights_{box}_{variant}_y*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def weight_matrix(w, space_only=True):
    """(design x cell) matrix, rows summing to 1; index (vel_year, rep)."""
    keys = ["cell_lat", "cell_lon"] + ([] if space_only else ["doy_bin"])
    g = w.groupby(["vel_year", "rep"] + keys, as_index=False)["w"].sum()
    W = g.pivot_table(index=["vel_year", "rep"], columns=keys, values="w",
                      fill_value=0.0)
    return W.div(W.sum(axis=1), axis=0)


def area_measure(box="boxB", deg=cfg.COARSE_DEG, path=None, out_dir=None):
    """p_j and wet fraction on the coarse grid, binned as the weights are.

    Profiles are binned by position alone, but p counts only water below the
    analysis depth, so a partly-masked shelf cell shows a large spurious w/p
    excess -- on the Grand Banks the wet fraction falls to 2%, inflating the
    ratio 40-fold. ``coverage`` is what lets callers exclude those cells.
    """
    out_dir = out_dir or cfg.OUT_DIR
    a = xr.open_dataset(path or os.path.join(out_dir, f"cell_area_{box}.nc"))["cell_area"]
    df = a.to_dataframe().reset_index()
    df["cell_lat"] = np.floor(df["latitude"] / deg) * deg + deg / 2
    df["cell_lon"] = np.floor(df["longitude"] / deg) * deg + deg / 2
    g = df.groupby(["cell_lat", "cell_lon"])["cell_area"]
    out = pd.DataFrame({"area": g.sum(), "coverage": g.count() / g.size()})
    out["p"] = out["area"] / out["area"].sum()
    return out


def restrict(W, area, min_coverage=1.0):
    """Common fully-wet support, with w and p each renormalised over it."""
    keep = area.index[(area["coverage"] >= min_coverage) & (area["p"] > 0)]
    keep = W.columns.intersection(keep)
    Wr = W[keep]
    p = area.loc[keep, "p"]
    return Wr.div(Wr.sum(axis=1), axis=0), p / p.sum()


def design_moments(W, p=None):
    """Per-year MC mean of w and its standard error, aligned to p if given."""
    yr = W.index.get_level_values("vel_year")
    mean, se = W.groupby(yr).mean(), W.groupby(yr).sem()
    n_rep = W.groupby(yr).size()
    if p is not None:
        p = p.reindex(mean.columns).fillna(0.0)
        p = p / p.sum()
    return mean, se, n_rep, p


def d1a_table(mean, se, p=None):
    """Between-year spread of the design mean against Monte-Carlo error.

    ratio > 1 in a cell means the year-to-year movement of E[w] is larger than
    the noise with which each year's mean is estimated, i.e. a real D1a effect
    rather than a sampling artefact.
    """
    between = mean.std(axis=0, ddof=1)
    within = se.mean(axis=0)
    out = pd.DataFrame({"between_year_sd": between, "mc_se": within,
                        "ratio": between / within.replace(0.0, np.nan)})
    if p is not None:
        out["p"] = p
        out["mean_w"] = mean.mean(axis=0)
        out["excess"] = out["mean_w"] / p.replace(0.0, np.nan)
    return out.reset_index()


def hellinger_pcoa(W, k=2):
    """Classical MDS of the Hellinger distance between designs."""
    X = np.sqrt(W.to_numpy()) / np.sqrt(2.0)
    G = X @ X.T
    d = np.diag(G)
    D2 = d[:, None] + d[None, :] - 2 * G
    n = len(D2)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ D2 @ J
    val, vec = np.linalg.eigh(B)
    idx = np.argsort(val)[::-1][:k]
    coords = vec[:, idx] * np.sqrt(np.clip(val[idx], 0, None))
    frac = np.clip(val, 0, None)[np.argsort(val)[::-1]][:k] / np.clip(val, 0, None).sum()
    return coords, frac


# ------------------------------- plotting --------------------------------

def _grid(series, ax, **kw):
    df = series.reset_index()
    df.columns = ["cell_lat", "cell_lon", "v"]
    piv = df.pivot(index="cell_lat", columns="cell_lon", values="v")
    m = ax.pcolormesh(piv.columns, piv.index, piv.to_numpy(), shading="auto", **kw)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(piv.index.to_series().mean()))))
    return m


def plot_excess_maps(mean, p, ncol=5, figsize=(16, 7)):
    """E[w^(l)] / p per year: where the flow concentrates a P-uniform deployment."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    years = list(mean.index)
    r = mean.div(p.replace(0.0, np.nan), axis=1)
    hi = float(np.nanpercentile(np.abs(np.log2(r.to_numpy())), 98))
    norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
    nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    for ax, y in zip(axes.ravel(), years):
        m = _grid(np.log2(r.loc[y]), ax, cmap="RdBu_r", norm=norm)
        ax.set_title(str(y), fontsize=10)
    for ax in axes.ravel()[len(years):]:
        ax.axis("off")
    fig.colorbar(m, ax=axes, shrink=0.8, label="log2( E[w] / p )")
    return fig


def plot_d1a_snr(tab, figsize=(11, 4.2)):
    """Between-year spread of E[w] relative to Monte-Carlo error."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    s = tab.set_index(["cell_lat", "cell_lon"])["ratio"]
    m = _grid(s, a1, cmap="viridis", vmin=0, vmax=float(np.nanpercentile(s, 98)))
    a1.set_title("between-year sd of E[w] / MC se")
    fig.colorbar(m, ax=a1, shrink=0.85)
    a2.hist(s.dropna(), bins=40, color="0.4")
    a2.axvline(1.0, color="crimson", lw=1.5, label="MC noise level")
    a2.set_xlabel("ratio"); a2.set_ylabel("cells"); a2.legend()
    fig.tight_layout()
    return fig


def plot_ordination(W, figsize=(6.4, 5.6), alpha=0.45):
    """PCoA of the designs, coloured by velocity year: the visual PERMANOVA."""
    import matplotlib.pyplot as plt
    coords, frac = hellinger_pcoa(W)
    yr = np.asarray(W.index.get_level_values("vel_year"))
    years = np.unique(yr)
    cmap = plt.get_cmap("viridis", len(years))
    fig, ax = plt.subplots(figsize=figsize)
    for i, y in enumerate(years):
        s = yr == y
        ax.scatter(coords[s, 0], coords[s, 1], s=9, alpha=alpha, color=cmap(i), label=str(y))
        ax.scatter(*coords[s].mean(0), s=140, marker="*", color=cmap(i),
                   edgecolor="k", linewidth=0.6, zorder=3)
    ax.set_xlabel(f"PCoA 1 ({100*frac[0]:.1f}%)")
    ax.set_ylabel(f"PCoA 2 ({100*frac[1]:.1f}%)")
    ax.legend(fontsize=7, ncol=2, markerscale=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--variant", default="uniform")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="drop coarse cells less than this fraction wet")
    a = ap.parse_args()

    w = load_mc_weights(a.box, a.variant)
    W = weight_matrix(w, space_only=True)
    area = area_measure(a.box)
    W, p = restrict(W, area, a.min_coverage)
    mean, se, n_rep, p = design_moments(W, p)
    tab = d1a_table(mean, se, p)

    out = os.path.join(cfg.OUT_DIR, f"design_mc_{a.box}_{a.variant}.csv")
    tab.to_csv(out, index=False)
    # Long form ("cells table"): a MultiIndex-column CSV does not round-trip.
    long = (mean.stack(list(range(mean.columns.nlevels)), future_stack=True)
                .rename("mean_w").reset_index()
                .merge(se.stack(list(range(se.columns.nlevels)), future_stack=True)
                         .rename("se_w").reset_index())
                .merge(p.rename("p").reset_index()))
    long.to_csv(os.path.join(cfg.OUT_DIR,
                f"design_mc_mean_{a.box}_{a.variant}.csv"), index=False)

    # Cache the ordination so the report need not reload the full weight table.
    coords, frac = hellinger_pcoa(W)
    pd.DataFrame({"vel_year": W.index.get_level_values("vel_year"),
                  "rep": W.index.get_level_values("rep"),
                  "pcoa1": coords[:, 0], "pcoa2": coords[:, 1],
                  "frac1": frac[0], "frac2": frac[1]}).to_csv(
        os.path.join(cfg.OUT_DIR, f"design_mc_pcoa_{a.box}_{a.variant}.csv"), index=False)

    print(f"{len(W)} designs, {n_rep.min()}-{n_rep.max()} reps/year, {W.shape[1]} cells")
    print(f"\nE[w]/p over cells: median {tab['excess'].median():.2f}, "
          f"5-95% {tab['excess'].quantile(.05):.2f}-{tab['excess'].quantile(.95):.2f}")
    print(f"between-year sd / MC se: median {tab['ratio'].median():.2f}, "
          f"frac cells > 1: {(tab['ratio'] > 1).mean():.3f}, "
          f"> 2: {(tab['ratio'] > 2).mean():.3f}")
    print(f"\nwrote {out}")
