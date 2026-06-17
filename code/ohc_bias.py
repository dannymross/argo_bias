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

J_TO_GJ = 1e-9


# ---- BIAS METRICS --------------------------------------------------------
def compute_bias(float_cells, truth_cells, value_cols=("ohc_700", "ohc_2000")):
    """Join float and truth cells and compute per-cell and domain-level bias.

    Parameters
    ----------
    float_cells, truth_cells : DataFrame
        Output of :func:`ohc.grid_cells` / :func:`ohc.coarsen_truth`, each with
        columns ``month, cell_lat, cell_lon, <value_cols>, n``.
    value_cols : tuple
        OHC columns to analyse.

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

    # Domain-level summary per month.
    dom = []
    for month, g in merged.groupby("month"):
        n_all = len(g)
        s = g[g["n_float"].notna()]
        row = {"month": month, "n_truth_cells": n_all,
               "n_sampled_cells": len(s),
               "sampled_fraction": len(s) / n_all if n_all else np.nan}
        for c in value_cols:
            true_mean = g[f"{c}_truth"].mean()              # truth over all cells
            float_mean = s[f"{c}_float"].mean()             # float estimate
            true_at_sampled = s[f"{c}_truth"].mean()        # truth where floats are
            row[f"{c}_true_mean"] = true_mean
            row[f"{c}_float_mean"] = float_mean
            row[f"{c}_true_at_sampled"] = true_at_sampled
            # Total bias splits into coverage (which cells) + representation (value error).
            row[f"{c}_bias"] = float_mean - true_mean
            row[f"{c}_coverage_bias"] = true_at_sampled - true_mean
            row[f"{c}_repr_bias"] = float_mean - true_at_sampled
        dom.append(row)
    domain = pd.DataFrame(dom).sort_values("month").reset_index(drop=True)

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
