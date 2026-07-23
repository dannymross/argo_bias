"""Orchestrate the R field-distribution diagnostics for
reports/field_diagnostics.qmd.

The bound of paper/sections/methods.tex needs four data-only ingredients:
sigma_w* (counts), R = Var_pi(Y), a DECLARED anomaly-field kurtosis kappa, and
the within-cell kappa_perp / sigma_Y_perp (float pairs). This module renders the
evidence for choosing kappa (and judging whether Gaussian kappa=3 is defensible)
and for the feasibility gate Delta<1, computed in code/field_diagnostics.R and
read back here. No mean field is fit: seasonality and the secular trend are
carried by the partition (they inflate the between-cell sigma_Y*), and the only
"detrending" is subtracting each cell's own empirical mean (the within-cell
residual). Floats realize the sampling law pi, so kappa_hat/R_hat here are
design-law; kappa is affine-invariant, which is what licenses reading its shape
off the floats and cross-checking against the dense GLORYS (nu) field.

Same Python<->R subprocess bridge as resolution.py / ohc_climatology's run_gp_*.
"""

import os
import subprocess

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_DIAG_SCRIPT = os.path.join(REPO_ROOT, "code", "field_diagnostics.R")

DEPTH_LABEL = {"700": "0-700 m", "2000": "0-2000 m"}
DEPTH_COLOR = {"700": "firebrick", "2000": "steelblue"}
VIEW_LABEL = {"raw": "raw OHC", "residual": "within-cell residual"}
# reference kurtoses (kappa, not excess): affine-invariant anchors
KAPPA_ANCHORS = {"Gaussian": 3.0, "Laplace": 6.0, "Exponential": 9.0}
TABLE_NAMES = ("moments", "residual_shape", "variance_split", "season_trend",
               "feasibility", "samples", "glorys_cross")


def run_field_diag(out_dir, depths=("700", "2000")):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["Rscript", os.path.abspath(FIELD_DIAG_SCRIPT), "tables",
           os.path.abspath(out_dir), ",".join(depths)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_tables(out_dir):
    """Read whatever diagnostic CSVs are present into a name->DataFrame dict."""
    tables = {}
    for name in TABLE_NAMES:
        p = os.path.join(out_dir, name + ".csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "depth" in df.columns:
                df["depth"] = df["depth"].astype(str)
            tables[name] = df
    return tables


def kappa_of(values):
    """Population kurtosis (kappa, Gaussian = 3), matching the R field_moments."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    m = x.mean()
    v = np.mean((x - m) ** 2)
    return float(np.mean((x - m) ** 4) / v ** 2)


def kurtosis_verdict(kappa):
    if not np.isfinite(kappa):
        return "indeterminate"
    if kappa < 2.5:
        return "platykurtic (< Gaussian; seasonal / spatial-gradient signature)"
    if kappa <= 3.5:
        return "~Gaussian (kappa=3 defensible)"
    if kappa <= 6:
        return "mildly leptokurtic (Gaussian..Laplace); declare kappa above the estimate"
    return "heavy-tailed (>= Laplace); declare a large kappa"


def delta(sigma_wstar, kappa):
    """Envelope feasibility functional Delta = sw* sqrt(kappa-1) + sw*^2."""
    sw = np.asarray(sigma_wstar, float)
    return sw * np.sqrt(np.maximum(0.0, kappa - 1.0)) + sw ** 2


# ---- QQ / distribution shape (raw vs within-cell residual) ---------------
def plot_qq(samples_df, depth, figsize=(11, 4.4)):
    import matplotlib.pyplot as plt
    from scipy import stats

    s = samples_df[samples_df["depth"] == str(depth)]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, view, colr in zip(axes, ("raw", "residual"), ("firebrick", "steelblue")):
        y = s[s["view"] == view]["value"].to_numpy()
        (osm, osr), (slope, inter, _) = stats.probplot(y, dist="norm")
        ax.plot(osm, osr, ".", ms=3, color=colr, alpha=0.6)
        ax.plot(osm, slope * osm + inter, "-", color="0.2", lw=1.5)
        ax.set_xlabel("standard normal quantile")
        ax.set_ylabel("ordered OHC (GJ m$^{-2}$)")
        ax.set_title(f"{VIEW_LABEL[view]}\n"
                     rf"$\kappa={kappa_of(y):.2f}$ (excess {kappa_of(y) - 3:+.2f})")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Normal QQ: raw field is platykurtic, eddy residual leptokurtic  "
                 f"({DEPTH_LABEL[str(depth)]})")
    fig.tight_layout()
    return fig


# ---- eddy-scale kappa_perp / sigma_Y_perp vs resolution ------------------
def plot_residual_shape(shape_df, depths, figsize=(11, 4.4)):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for d in depths:
        s = shape_df[shape_df["depth"] == str(d)].sort_values("K_t")
        c = DEPTH_COLOR[str(d)]
        ax1.plot(s["K_t"], s["kappa_perp_pair"], "o-", color=c,
                 label=f"{DEPTH_LABEL[str(d)]} (pairs)")
        ax1.plot(s["K_t"], s["kappa_perp_cellmean"], "^--", color=c, mfc="white",
                 label=f"{DEPTH_LABEL[str(d)]} (cell-demeaned)")
        ax2.plot(s["K_t"], s["sigmaYperp_pair"], "o-", color=c, label=DEPTH_LABEL[str(d)])
    for name, k in KAPPA_ANCHORS.items():
        ax1.axhline(k, ls=":", color="0.6", lw=1)
        ax1.text(ax1.get_xlim()[1], k, f" {name}", va="center", fontsize=7, color="0.4")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(sorted(shape_df["K_t"].unique()))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"$K_t$ (time cells over 2020-2022)")
        ax.grid(True, which="both", alpha=0.25)
    ax1.set_ylabel(r"within-cell kurtosis $\kappa^\perp$")
    ax1.set_title("Eddy-scale kurtosis vs resolution")
    ax1.legend(frameon=False, fontsize=7, ncol=2)
    ax2.set_ylabel(r"$\sigma_{Y^\perp}$ (GJ m$^{-2}$)")
    ax2.set_ylim(0, None)
    ax2.set_title("Within-cell dispersion vs resolution")
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("Within-cell (design-based) anomaly shape from float pairs")
    fig.tight_layout()
    return fig


# ---- variance split + the resolution tension -----------------------------
def plot_variance_split(split_df, depth, figsize=(7.4, 4.6)):
    """L (perp variance fraction, falling) and plug-in sigma_w* (rising) vs K_t."""
    import matplotlib.pyplot as plt

    s = split_df[split_df["depth"] == str(depth)].sort_values("K_t")
    fig, axL = plt.subplots(figsize=figsize)
    axL.plot(s["K_t"], s["L"], "o-", color="darkorange",
             label=r"$\sigma_{Y^\perp}^2/\sigma_Y^2$ (within-cell fraction)")
    axL.set_xscale("log")
    axL.set_xticks(sorted(s["K_t"].unique()))
    axL.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axL.set_xlabel(r"$K_t$ (time cells)")
    axL.set_ylabel(r"$\sigma_{Y^\perp}^2/\sigma_Y^2$", color="darkorange")
    axL.tick_params(axis="y", labelcolor="darkorange")
    axL.set_ylim(0, None)
    axR = axL.twinx()
    axR.plot(s["K_t"], s["sigma_wstar"], "s--", color="purple",
             label=r"$\sigma_{w^\star}$ (plug-in)")
    axR.axhline(1.0, ls=":", color="0.5", lw=1)
    axR.set_ylabel(r"$\sigma_{w^\star}$ (plug-in)", color="purple")
    axR.tick_params(axis="y", labelcolor="purple")
    axL.grid(True, which="both", alpha=0.2)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center")
    axL.set_title(f"The resolution tension: season/trend move into $\\sigma_{{Y^\\star}}$ as\n"
                  f"$K_t$ rises, but $\\sigma_{{w^\\star}}$ (coincidence starvation) rises too  "
                  f"({DEPTH_LABEL[str(depth)]})")
    fig.tight_layout()
    return fig


# ---- feasibility Delta < 1 by resolution ---------------------------------
def plot_feasibility(feas_df, depth, kappa_grid=None, figsize=(7.4, 5.0)):
    import matplotlib.pyplot as plt

    if kappa_grid is None:
        kappa_grid = np.linspace(1.5, 12, 200)
    s = feas_df[feas_df["depth"] == str(depth)].sort_values("cell_km", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("viridis")
    for i, (_, r) in enumerate(s.iterrows()):
        swU = r["sigma_wstar_hi"]
        ax.plot(kappa_grid, delta(swU, kappa_grid), lw=2,
                color=cmap(i / max(1, len(s) - 1)),
                label=f"{r['cell_km']:.0f} km (K={int(r['K'])})")
    ax.axhline(1.0, ls="--", color="0.3")
    for name, k in KAPPA_ANCHORS.items():
        ax.axvline(k, ls=":", color="0.7", lw=1)
        ax.text(k, 1.55, name, rotation=90, va="top", ha="right", fontsize=7, color="0.4")
    ax.set_xlabel(r"$\kappa$ (declared anomaly-field kurtosis)")
    ax.set_ylabel(r"$\Delta = \sigma_{w^\star}\sqrt{\kappa-1} + \sigma_{w^\star}^2$")
    ax.set_ylim(0, 1.6)
    ax.set_title(f"Envelope feasibility $\\Delta<1$ by resolution\n"
                 f"(upper jackknife $\\sigma_{{w^\\star}}$, conservative)  "
                 f"({DEPTH_LABEL[str(depth)]})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, title="cell size", loc="upper left")
    fig.tight_layout()
    return fig


def feasible_kappa_max(feas_df, depth, use_hi=True):
    """Largest declared kappa that keeps Delta<1 at each resolution (upper sw*)."""
    s = feas_df[feas_df["depth"] == str(depth)].copy()
    col = "sigma_wstar_hi" if use_hi else "sigma_wstar"
    s["kappa_feasible_max"] = np.where(s[col] < 1, 1 + ((1 - s[col] ** 2) / s[col]) ** 2, 1.0)
    return s
