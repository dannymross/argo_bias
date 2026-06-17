"""Quantify OHC sampling bias: synthetic-Argo estimate vs GLORYS truth.

Given the gridded truth OHC (from :func:`ohc.coarsen_truth`) and the
synthetic-float OHC binned to the same cells (from :func:`ohc.grid_cells`), this
computes the two biases that matter for the preferential-sampling question:

* **Representation error** -- in each cell a float actually sampled, how far is
  the float estimate from the truth in that cell/month?
* **Coverage / sampling bias** (the headline number) -- the domain-mean OHC you
  would report from the *float-sampled cells only*, minus the *true* domain mean
  over *all* cells. Non-uniform float coverage makes these differ; that gap is
  the bias from preferential sampling.

Also emits the sampled-cell fraction over time and simple diagnostic plots.
All OHC in J/m2 unless converted to GJ/m2 for display.
"""

import os

import numpy as np
import pandas as pd

import ohc

J_TO_GJ = 1e-9


# ---- BIAS METRICS --------------------------------------------------------
def compute_bias(float_cells, truth_cells, value_cols=("ohc_700", "ohc_2000"),
                 true_domain_mean=None):
    """Join float and truth cells and compute per-cell and domain-level bias.

    Parameters
    ----------
    float_cells, truth_cells : DataFrame
        Output of :func:`ohc.grid_cells` / :func:`ohc.coarsen_truth`, each with
        columns ``month, cell_lat, cell_lon, <value_cols>, n``.
    value_cols : tuple
        OHC columns to analyse.
    true_domain_mean : DataFrame, optional
        Fixed per-month true domain mean (columns ``month`` + value_cols, J/m2)
        from :func:`ohc.truth_domain_mean`. If given, it is used as the truth
        reference for total and coverage bias instead of the cell-mean over all
        truth cells -- which removes the resolution-dependent drift of the
        reference across a cell-size sweep. The total = coverage + representation
        decomposition is preserved either way.

    Returns
    -------
    dict with:
        ``cells`` -- per (month, cell) truth, float, and representation error
            (float - truth), only where a float sampled the cell;
        ``domain`` -- per month: true domain mean (all truth cells), sampled
            domain mean (float-sampled cells), the float estimate, the coverage
            bias (sampled estimate - true mean), and sampled-cell fraction.
    """
    keys = ["month", "cell_lat", "cell_lon"]
    tcols = {c: f"{c}_truth" for c in value_cols}
    fcols = {c: f"{c}_float" for c in value_cols}

    truth = truth_cells.rename(columns=tcols)
    flt = float_cells.rename(columns={**fcols, "n": "n_float"})

    merged = truth.merge(
        flt[keys + list(fcols.values()) + ["n_float"]], on=keys, how="left"
    )
    sampled = merged["n_float"].notna()

    cell_rows = merged[sampled].copy()
    for c in value_cols:
        cell_rows[f"{c}_bias"] = cell_rows[f"{c}_float"] - cell_rows[f"{c}_truth"]

    # Domain-level summary per month, via vectorised groupby aggregation.
    month = merged["month"]
    gb = merged.groupby("month")
    domain = pd.DataFrame({
        "n_truth_cells": gb.size(),
        "n_sampled_cells": gb["n_float"].count(),  # non-NaN = sampled cells
    })
    domain["sampled_fraction"] = domain["n_sampled_cells"] / domain["n_truth_cells"]

    ref = true_domain_mean.set_index("month") if true_domain_mean is not None else None

    for c in value_cols:
        cell_true_mean = gb[f"{c}_truth"].mean()                     # truth, all cells (this deg)
        # Fixed area-weighted reference if supplied, else the cell-mean over all cells.
        true_mean = ref[c].reindex(cell_true_mean.index) if ref is not None else cell_true_mean
        float_mean = gb[f"{c}_float"].mean()                         # float estimate (NaN-skip = sampled)
        true_at_sampled = merged[f"{c}_truth"].where(sampled).groupby(month).mean()
        domain[f"{c}_true_mean"] = true_mean
        domain[f"{c}_float_mean"] = float_mean
        domain[f"{c}_true_at_sampled"] = true_at_sampled
        # Total bias splits into coverage (which cells) + representation (value error).
        domain[f"{c}_bias"] = float_mean - true_mean
        domain[f"{c}_coverage_bias"] = true_at_sampled - true_mean
        domain[f"{c}_repr_bias"] = float_mean - true_at_sampled

    domain = domain.reset_index().sort_values("month").reset_index(drop=True)

    return {"cells": cell_rows, "domain": domain}


def bias_summary(domain, value_cols=("ohc_700", "ohc_2000")):
    """Time-averaged bias summary in GJ/m2, for a quick headline table."""
    rows = []
    for c in value_cols:
        rows.append({
            "depth": c,
            "mean_true_GJ": domain[f"{c}_true_mean"].mean() * J_TO_GJ,
            "bias_GJ": domain[f"{c}_bias"].mean() * J_TO_GJ,
            "coverage_bias_GJ": domain[f"{c}_coverage_bias"].mean() * J_TO_GJ,
            "repr_bias_GJ": domain[f"{c}_repr_bias"].mean() * J_TO_GJ,
            "mean_sampled_fraction": domain["sampled_fraction"].mean(),
        })
    return pd.DataFrame(rows)


# ---- CELL-SIZE SWEEP -----------------------------------------------------
def sweep_resolution(truth_field, sim, degs, value_cols=("ohc_700", "ohc_2000"),
                     weighted_reference=True):
    """Sweep the analysis cell size and report bias vs resolution.

    The grid cell size is a parameter of the estimator (a box-kernel proxy for
    a Gaussian-process correlation length), not an intrinsic feature of the
    ungridded float data. Coverage and representation bias both depend on it:
    coarse cells -> small coverage bias but large representation/smoothing bias;
    fine cells -> the reverse. This traces that curve.

    The expensive pieces (truth OHC field and synthetic-float profiles) are
    computed once by the caller and re-gridded cheaply at each ``deg``.

    Parameters
    ----------
    truth_field : xarray.Dataset
        Output of :func:`ohc.truth_ohc_field` (ohc_700/ohc_2000 on the native grid).
    sim : pandas.DataFrame
        Per-profile synthetic-Argo OHC from :func:`ohc.float_ohc`.
    degs : iterable of float
        Cell sizes (degrees) to sweep, e.g. ``[1/12, 0.25, 0.5, 1]``.
    weighted_reference : bool
        If True (default), measure bias against the fixed cos(lat) area-weighted
        true domain mean (native resolution), so only the estimator changes
        across the sweep. If False, use the unweighted native-cell mean.

    Returns
    -------
    pandas.DataFrame
        One row per (deg, depth) with time-averaged bias terms in GJ/m2,
        the mean sampled-cell fraction, and the mean cell count.
    """
    ref = ohc.truth_domain_mean(truth_field, weighted=weighted_reference,
                                value_cols=list(value_cols))
    rows = []
    for deg in degs:
        truth_cells = ohc.coarsen_truth(truth_field, deg=deg)
        float_cells = ohc.grid_cells(sim, list(value_cols), deg=deg)
        res = compute_bias(float_cells, truth_cells, value_cols, true_domain_mean=ref)
        summary = bias_summary(res["domain"], value_cols)
        summary.insert(0, "deg", deg)
        summary["mean_n_truth_cells"] = res["domain"]["n_truth_cells"].mean()
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    cols = ["deg", "depth", "mean_n_truth_cells", "mean_sampled_fraction",
            "mean_true_GJ", "bias_GJ", "coverage_bias_GJ", "repr_bias_GJ"]
    return out[cols].sort_values(["depth", "deg"]).reset_index(drop=True)


# ---- PLOTS ---------------------------------------------------------------
def plot_domain_timeseries(domain, value_col="ohc_2000", out_path=None):
    """Truth vs synthetic-Argo domain-mean OHC over time, with coverage."""
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    m = domain["month"]
    ax.plot(m, domain[f"{value_col}_true_mean"] * J_TO_GJ, "-o",
            color="#1a4f8a", label="truth (all cells)")
    ax.plot(m, domain[f"{value_col}_float_mean"] * J_TO_GJ, "-s",
            color="#e07b39", label="synthetic Argo (sampled cells)")
    ax.set_ylabel(f"{value_col}  (GJ m$^{{-2}}$)")
    ax.legend(frameon=False)
    ax.set_title(f"Domain-mean OHC: truth vs synthetic Argo ({value_col})")
    ax.grid(alpha=0.2)

    ax2.bar(m, domain["sampled_fraction"], width=20, color="#4a90d9", alpha=0.7)
    ax2.set_ylabel("sampled\ncell frac")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("month")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_bias_vs_resolution(sweep, value_col="ohc_2000", out_path=None):
    """Bias terms vs cell size (the box-kernel proxy for a GP length scale)."""
    import matplotlib.pyplot as plt

    df = sweep[sweep["depth"] == value_col].sort_values("deg")
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(df["deg"], df["bias_GJ"], "-o", color="#1a1a1a", label="total bias")
    ax.plot(df["deg"], df["coverage_bias_GJ"], "-s", color="#4a90d9", label="coverage bias")
    ax.plot(df["deg"], df["repr_bias_GJ"], "-^", color="#e07b39", label="representation bias")
    ax.set_ylabel(f"{value_col} bias  (GJ m$^{{-2}}$)")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    ax.set_title(f"OHC sampling bias vs analysis cell size ({value_col})")
    ax.grid(alpha=0.2, which="both")

    ax2.plot(df["deg"], df["mean_sampled_fraction"], "-o", color="#4a90d9")
    ax2.set_ylabel("sampled\ncell frac")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("cell size (deg)")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.2, which="both")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_bias_map(cells, value_col="ohc_2000", month=None, out_path=None):
    """Per-cell representation error (float - truth) as a scatter/heat map."""
    import matplotlib.pyplot as plt

    df = cells if month is None else cells[cells["month"] == month]
    bias = df[f"{value_col}_bias"] * J_TO_GJ
    vmax = np.nanpercentile(np.abs(bias), 95) if len(bias) else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["cell_lon"], df["cell_lat"], c=bias, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=140, marker="s", edgecolor="0.6")
    fig.colorbar(sc, ax=ax, label=f"{value_col} bias (GJ m$^{{-2}}$)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    title = f"Per-cell OHC bias ({value_col})"
    if month is not None:
        title += f" — {pd.Timestamp(month):%Y-%m}"
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig
