"""Orchestrate the R cell-resolution analysis for reports/resolution.qmd.

The cell scale K = K_lon * K_lat * K_t is fixed from the dense GLORYS field
(code/export_glorys_ohc.py), so it is free of the Argo sampling pattern. Three
factors, computed in R and read back here for plotting:

  * SPATIAL (K_lon, K_lat) -- the fraction of spatial field variance below the
    cell scale, sigma_{Y^perp}^2 / sigma_Y^2, per snapshot (seasonality-free);
    the intuitive driver of the spatial grid (code/leakage_curve.R spatial view).
  * spatial DECORRELATION length -- directional OHC variogram, corroborating the
    above and exposing anisotropy (code/variogram.R).
  * TEMPORAL (K_t) and seasonality -- residual of the regional-mean series vs the
    number of time bins, plus its ACF (code/leakage_curve.R time view + ACF).

Same Python<->R subprocess bridge as ohc_climatology's run_gp_*. Notation
follows paper/sections/methodology.tex (nu, P_K, y^star/y^perp,
sigma_{Y^star}, sigma_{Y^perp}, eq:sigmaYK-hat/eq:sigmaYperp-hat,
eq:residual-diagnostic). The plotted quantity is the *variance* fraction
sigma_{Y^perp}^2/sigma_Y^2 (the `L` column); the SD ratio sigma_{Y^perp}/sigma_Y
is the `resid` column.
"""

import os
import subprocess

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAKAGE_SCRIPT = os.path.join(REPO_ROOT, "code", "leakage_curve.R")
VARIOGRAM_SCRIPT = os.path.join(REPO_ROOT, "code", "variogram.R")

DEPTH_LABEL = {"700": "0-700 m", "2000": "0-2000 m"}
DEPTH_COLOR = {"700": "firebrick", "2000": "steelblue"}
FIELD_LABEL = {"raw": "raw OHC", "deseas": "deseasonalized"}
DIR_STYLE = {"omni": ("grey", "-", "omnidirectional"),
             "zonal": ("firebrick", "-", "zonal (along-front)"),
             "merid": ("steelblue", "-", "meridional (across-front)")}
SAMPLING_SCALE_KM = 75  # float-count floor s_F = sqrt(|A|/F); paper eq:verdict
FRAC_YLABEL = r"$\sigma_{Y^\perp}^2/\sigma_Y^2$ (variance fraction below cell scale)"


def _run(r_script, out_paths, depths):
    cmd = ["Rscript", os.path.abspath(r_script)] + \
          [os.path.abspath(p) for p in out_paths] + [",".join(depths)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_leakage_curves(out_time, out_full, out_spatial, depths=("700", "2000")):
    _run(LEAKAGE_SCRIPT, (out_time, out_full, out_spatial), depths)


def run_variogram(out_ranges, out_curves, out_acf, depths=("700", "2000")):
    _run(VARIOGRAM_SCRIPT, (out_ranges, out_curves, out_acf), depths)


def load_tables(*paths):
    for p in paths:
        df = pd.read_csv(p)
        if "depth" in df.columns:
            df["depth"] = df["depth"].astype(str)
        yield df


def _log_x(ax, ticks):
    import matplotlib.pyplot as plt
    ax.set_xscale("log")
    ax.set_xticks(sorted(ticks))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())


# ---- SPATIAL: variance fraction vs cell size (sets K_lon, K_lat) ---------
def plot_spatial_fraction(spatial_df, depths, figsize=(7, 4.6)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for d in depths:
        s = spatial_df[spatial_df["depth"] == str(d)].sort_values("cell_km")
        ax.plot(s["cell_km"], s["L"], marker="o", color=DEPTH_COLOR[str(d)],
                label=DEPTH_LABEL[str(d)])
    ax.axvline(SAMPLING_SCALE_KM, ls="--", color="0.4", lw=1,
               label=rf"float floor $s_F\approx{SAMPLING_SCALE_KM:.0f}$ km")
    ax.set_xscale("log")
    ax.set_xlabel(r"cell size $\sqrt{\Delta x\,\Delta y}$ (km)")
    ax.set_ylabel(FRAC_YLABEL)
    ax.set_ylim(0, None)
    ax.set_title("Fraction of spatial OHC variance below the cell scale (per snapshot)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def spatial_fraction_at(spatial_df, depth, cell_km_max):
    """Row nearest to (and no coarser than) a target cell size."""
    s = spatial_df[spatial_df["depth"] == str(depth)]
    s = s[s["cell_km"] <= cell_km_max]
    return None if s.empty else s.sort_values("cell_km").iloc[-1]


# ---- SPATIAL: directional variogram --------------------------------------
def plot_variograms(curves_df, depth, figsize=(12, 4.4)):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, field in zip(axes, ("raw", "deseas")):
        d = curves_df[(curves_df["depth"] == str(depth)) & (curves_df["field"] == field)]
        sill = float(d["sill"].iloc[0])
        for dir_, (colr, ls, lab) in DIR_STYLE.items():
            g = d[d["direction"] == dir_].sort_values("dist_km")
            ax.plot(g["dist_km"], g["gamma"], color=colr, ls=ls, marker=".", label=lab)
        ax.axhline(sill, ls=":", color="grey", lw=1, label="sill")
        ax.axvline(SAMPLING_SCALE_KM, ls="--", color="0.5", lw=1,
                   label=rf"$s_F\approx{SAMPLING_SCALE_KM:.0f}$ km")
        ax.set_xlabel("separation (km)")
        ax.set_ylabel(r"semivariance $\gamma(h)$")
        ax.set_title(FIELD_LABEL[field])
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Directional OHC variogram  ({DEPTH_LABEL[str(depth)]})")
    fig.tight_layout()
    return fig


# ---- TEMPORAL / seasonality ----------------------------------------------
def plot_temporal_fraction(time_df, depth, figsize=(7, 4.6)):
    """Variance fraction of the regional-mean series below the time bin, vs K_t."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    d = time_df[time_df["depth"] == str(depth)]
    for field, colr, mk in (("raw", "firebrick", "o"), ("deseas", "steelblue", "^")):
        s = d[d["field"] == field].sort_values("K_t")
        ax.plot(s["K_t"], s["L"], marker=mk, color=colr, label=FIELD_LABEL[field])
    _log_x(ax, d["K_t"].unique())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$K_t$ (time bins over 2020-2022)")
    ax.set_ylabel(FRAC_YLABEL)
    ax.set_title(f"Temporal variance leaked below the time bin  ({DEPTH_LABEL[str(depth)]})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def temporal_fraction_at(time_df, depth, K_t):
    d = time_df[(time_df["depth"] == str(depth)) & (time_df["K_t"] == K_t)]
    return {r["field"]: r["L"] for _, r in d.iterrows()}


def plot_temporal_acf(acf_df, depths, figsize=(6.8, 4.4)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for d in depths:
        s = acf_df[acf_df["depth"] == str(d)].sort_values("lag_days")
        ef = float(s["efold_days"].iloc[0])
        ax.plot(s["lag_days"], s["acf"], marker=".", color=DEPTH_COLOR[str(d)],
                label=f"{DEPTH_LABEL[str(d)]} (e-fold {ef:.0f} d)")
        if np.isfinite(ef):
            ax.axvline(ef, ls="--", color=DEPTH_COLOR[str(d)], lw=1)
    ax.axhline(1 / np.e, ls=":", color="grey", lw=1, label=r"$1/e$")
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Temporal ACF of the deseasonalized regional-mean OHC")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---- checks --------------------------------------------------------------
def max_identity_gap(*dfs):
    return max(float(np.abs(df["identity_gap"]).max()) for df in dfs if "identity_gap" in df)


def max_quad_gap_frac(full_df):
    """|equal-measure sigma_Y^2 - direct area-weighted sigma_Y^2| / sigma_Y^2."""
    return float((full_df["quad_gap"].abs() / full_df["sigmaY2"]).max())
