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


def run_argo_comparison(out_argo, out_occ, out_sweep, depths=("700", "2000")):
    cmd = ["Rscript", os.path.abspath(LEAKAGE_SCRIPT), "argo"] + \
          [os.path.abspath(p) for p in (out_argo, out_occ, out_sweep)] + \
          [",".join(depths)]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


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


# ---- observed Argo vs GLORYS truth (preferential sampling) ---------------
def plot_argo_occupancy(occ_df, depth, figsize=(7.4, 4.4)):
    """Operating-grid map of Argo sampling weight wbar_j = K*pi_j (n_j profiles)."""
    import matplotlib.pyplot as plt

    o = occ_df[occ_df["depth"] == str(depth)]
    lons = np.sort(o["cell_lon"].unique()); lats = np.sort(o["cell_lat"].unique())
    W = np.full((len(lats), len(lons)), np.nan)
    for _, r in o.iterrows():
        W[np.where(lats == r["cell_lat"])[0][0],
          np.where(lons == r["cell_lon"])[0][0]] = r["wbar"]
    fig, ax = plt.subplots(figsize=figsize)
    vmax = float(np.nanmax(np.abs(W - 1))) + 1
    im = ax.imshow(W, origin="lower", cmap="RdBu_r", vmin=2 - vmax, vmax=vmax,
                   extent=[lons.min() - 0.5, lons.max() + 0.5,
                           lats.min() - 0.4, lats.max() + 0.4], aspect="auto")
    for _, r in o.iterrows():
        ax.text(r["cell_lon"], r["cell_lat"], f"{int(r['n'])}",
                ha="center", va="center", fontsize=8, color="0.1")
    fig.colorbar(im, ax=ax, label=r"sampling weight $\bar w_j=K\pi_j$")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Argo profile occupancy on the operating grid  ({DEPTH_LABEL[str(depth)]})\n"
                 "cell counts $n_j$; $\\bar w_j=1$ is uniform sampling")
    fig.tight_layout()
    return fig


def plot_within_between_compare(argo_df, full_df, depth, figsize=(11, 4.4)):
    """Argo-sample vs GLORYS-truth field factors (sqrt, GJ) and L, across K_t."""
    import matplotlib.pyplot as plt

    a = argo_df[argo_df["depth"] == str(depth)].sort_values("K_t")
    g = full_df[(full_df["depth"] == str(depth)) & (full_df["field"] == "raw")].sort_values("K_t")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for factor, colr in (("sigmaYstar2", "firebrick"), ("sigmaYperp2", "steelblue")):
        lab = r"$\sigma_{Y^\star}$" if "star" in factor else r"$\sigma_{Y^\perp}$"
        ax1.plot(g["K_t"], np.sqrt(g[factor]), "-", color=colr, label=f"{lab} GLORYS")
        ax1.plot(a["K_t"], np.sqrt(a[factor]), "o--", color=colr, mfc="white",
                 label=f"{lab} Argo")
    _log_x(ax1, g["K_t"].unique())
    ax1.set_xlabel(r"$K_t$"); ax1.set_ylabel(r"field factor (GJ m$^{-2}$)")
    ax1.set_ylim(0, None); ax1.grid(True, which="both", alpha=0.25)
    ax1.set_title("Between/within-cell OHC variation")
    ax1.legend(frameon=False, fontsize=8, ncol=2)

    ax2.plot(g["K_t"], g["L"], "-", color="0.3", label="GLORYS truth")
    ax2.plot(a["K_t"], a["L"], "o--", color="darkorange", mfc="white", label="Argo sample")
    _log_x(ax2, g["K_t"].unique())
    ax2.set_ylim(0, None); ax2.set_xlabel(r"$K_t$")
    ax2.set_ylabel(r"$\sigma_{Y^\perp}^2/\sigma_Y^2$")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_title("Within-cell variance fraction")
    ax2.legend(frameon=False)
    fig.suptitle(f"Observed Argo vs GLORYS truth  ({DEPTH_LABEL[str(depth)]})")
    fig.tight_layout()
    return fig


def plot_argo_sweep(sweep_df, depth, figsize=(7, 4.6)):
    """Spatial sampling floor: fraction of cells with >=2 profiles vs cell size."""
    import matplotlib.pyplot as plt

    s = sweep_df[sweep_df["depth"] == str(depth)].sort_values("cell_km")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s["cell_km"], s["frac_ge2"], "o-", color="seagreen")
    ax.axvline(SAMPLING_SCALE_KM, ls="--", color="0.4", lw=1,
               label=rf"float floor $s_F\approx{SAMPLING_SCALE_KM:.0f}$ km")
    ax.set_xscale("log")
    ax.set_xlabel(r"cell size $\sqrt{\Delta x\,\Delta y}$ (km)")
    ax.set_ylabel(r"fraction of cells with $\geq 2$ profiles")
    ax.set_ylim(0, 1.03)
    ax.set_title("Argo spatial sampling floor (pooled over 2020-2022)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def argo_bias_summary(argo_df, full_df, depth, K_t):
    """Scalar comparison at one (depth, K_t): realized bias, sigma_w*, between-cell bound."""
    a = argo_df[(argo_df["depth"] == str(depth)) & (argo_df["K_t"] == K_t)].iloc[0]
    g = full_df[(full_df["depth"] == str(depth)) & (full_df["field"] == "raw")
                & (full_df["K_t"] == K_t)].iloc[0]
    sig_Ystar = float(np.sqrt(g["sigmaYstar2"]))
    return {"mu_truth": float(a["mu_truth"]), "mu_naive": float(a["mu_naive"]),
            "mu_grid": float(a["mu_grid"]), "bias_naive": float(a["bias_naive"]),
            "bias_grid": float(a["bias_grid"]),
            "sigma_wstar": float(a["sigma_wstar"]),
            "sigma_Ystar_glorys": sig_Ystar,
            "bound_between": float(a["sigma_wstar"]) * sig_Ystar,
            "L_argo": float(a["L"]), "L_glorys": float(g["L"])}


# ---- checks --------------------------------------------------------------
def max_identity_gap(*dfs):
    return max(float(np.abs(df["identity_gap"]).max()) for df in dfs if "identity_gap" in df)


def max_quad_gap_frac(full_df):
    """|equal-measure sigma_Y^2 - direct area-weighted sigma_Y^2| / sigma_Y^2."""
    return float((full_df["quad_gap"].abs() / full_df["sigmaY2"]).max())
