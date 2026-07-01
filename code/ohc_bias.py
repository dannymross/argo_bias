import numpy as np
import pandas as pd

import ohc

J_TO_GJ = 1e-9


# ---- BIAS METRICS --------------------------------------------------------
def compute_bias(
    float_cells, truth_cells, value_cols=("ohc_700", "ohc_2000"), true_domain_mean=None
):
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
        reference for total and sampling bias instead of the cell-mean over all
        truth cells -- which removes the resolution-dependent drift of the
        reference across a cell-size sweep. The total = sampling + grid
        decomposition is preserved either way.

    Returns
    -------
    dict with:
        ``cells`` -- per (month, cell) truth, float, and grid error
            (float - truth), only where a float sampled the cell;
        ``domain`` -- per month: true domain mean (all truth cells), sampled
            domain mean (float-sampled cells), the float estimate, the sampling
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
    domain = pd.DataFrame(
        {
            "n_truth_cells": gb.size(),
            "n_sampled_cells": gb["n_float"].count(),  # non-NaN = sampled cells
        }
    )
    domain["sampled_fraction"] = domain["n_sampled_cells"] / domain["n_truth_cells"]

    ref = true_domain_mean.set_index("month") if true_domain_mean is not None else None

    for c in value_cols:
        cell_true_mean = gb[f"{c}_truth"].mean()  # truth, all cells (this deg)
        # Fixed area-weighted reference if supplied, else the cell-mean over all cells.
        true_mean = (
            ref[c].reindex(cell_true_mean.index) if ref is not None else cell_true_mean
        )
        float_mean = gb[f"{c}_float"].mean()  # float estimate (NaN-skip = sampled)
        true_at_sampled = merged[f"{c}_truth"].where(sampled).groupby(month).mean()
        domain[f"{c}_true_mean"] = true_mean
        domain[f"{c}_float_mean"] = float_mean
        domain[f"{c}_true_at_sampled"] = true_at_sampled
        # Total bias splits into sampling (which cells) + grid (value error).
        domain[f"{c}_bias"] = float_mean - true_mean
        domain[f"{c}_sampling_bias"] = true_at_sampled - true_mean
        domain[f"{c}_grid_bias"] = float_mean - true_at_sampled

    domain = domain.reset_index().sort_values("month").reset_index(drop=True)

    return {"cells": cell_rows, "domain": domain}


def bias_summary(domain, value_cols=("ohc_700", "ohc_2000")):
    """Time-averaged bias summary in GJ/m2, for a quick headline table."""
    rows = []
    for c in value_cols:
        rows.append(
            {
                "depth": c,
                "mean_true_GJ": domain[f"{c}_true_mean"].mean() * J_TO_GJ,
                "bias_GJ": domain[f"{c}_bias"].mean() * J_TO_GJ,
                "sampling_bias_GJ": domain[f"{c}_sampling_bias"].mean() * J_TO_GJ,
                "grid_bias_GJ": domain[f"{c}_grid_bias"].mean() * J_TO_GJ,
                "mean_sampled_fraction": domain["sampled_fraction"].mean(),
            }
        )
    return pd.DataFrame(rows)


def monthly_bias_table(float_cells, truth_cells, value_col, true_domain_mean=None):
    """Per-month preferential-sampling summary for one depth + resolution.

    Tells the full bias story for a single depth layer at the cell size implied
    by ``float_cells``/``truth_cells``. OHC columns are in GJ/m2:

    * ``samp_cells``     -- cells a float sampled that month
    * ``profiles``       -- float profiles that month (sum of per-cell counts)
    * ``prof_per_cell``  -- profiles / sampled cells
    * ``true_ohc``       -- mean true OHC over **all** cells
    * ``true_ohc_samp``  -- mean true OHC over the **sampled** cells
    * ``synth_ohc_samp`` -- mean synthetic-Argo OHC over the sampled cells
    * ``samp_bias``      -- true_ohc_samp - true_ohc  (sampling/coverage: which
      cells get sampled, true values only)
    * ``grid_bias``      -- synth_ohc_samp - true_ohc_samp  (within-cell estimate
      error)

    ``samp_bias + grid_bias`` is the total estimate error
    (synth_ohc_samp - true_ohc). A final ``YEAR`` row summarises the year:
    profiles are summed, everything else is the monthly mean (so the YEAR bias
    terms are the time-averaged biases).
    """
    dom = compute_bias(
        float_cells,
        truth_cells,
        value_cols=(value_col,),
        true_domain_mean=true_domain_mean,
    )["domain"]
    prof = float_cells.groupby("month")["n"].sum()
    g = J_TO_GJ
    tbl = pd.DataFrame(
        {
            "month": pd.to_datetime(dom["month"]).dt.strftime("%Y-%m"),
            "samp_cells": dom["n_sampled_cells"].to_numpy(),
            "profiles": dom["month"].map(prof).to_numpy(),
            "true_ohc": dom[f"{value_col}_true_mean"].to_numpy() * g,
            "true_ohc_samp": dom[f"{value_col}_true_at_sampled"].to_numpy() * g,
            "synth_ohc_samp": dom[f"{value_col}_float_mean"].to_numpy() * g,
            "samp_bias": dom[f"{value_col}_sampling_bias"].to_numpy() * g,
            "grid_bias": dom[f"{value_col}_grid_bias"].to_numpy() * g,
        }
    )
    tbl["prof_per_cell"] = tbl["profiles"] / tbl["samp_cells"]
    tbl["bias"] = tbl["samp_bias"] + tbl["grid_bias"]
    tbl = tbl[
        [
            "month",
            "profiles",
            "true_ohc",
            "true_ohc_samp",
            "synth_ohc_samp",
            "samp_bias",
            "grid_bias",
            "bias",
        ]
    ]

    year = {"month": "YEAR", "profiles": tbl["profiles"].sum()}
    for c in tbl.columns:
        if c not in ("month", "profiles"):
            year[c] = tbl[c].mean()
    out = pd.concat([tbl, pd.DataFrame([year])], ignore_index=True)

    out["profiles"] = out["profiles"].round().astype(int)
    for c, n in {
        "true_ohc": 3,
        "true_ohc_samp": 3,
        "synth_ohc_samp": 3,
        "samp_bias": 3,
        "grid_bias": 3,
        "bias": 3,
    }.items():
        out[c] = out[c].round(n)
    return out


# ---- CELL-SIZE SWEEP -----------------------------------------------------
def sweep_resolution(
    truth_field, sim, degs, value_cols=("ohc_700", "ohc_2000"), weighted_reference=True
):
    """Sweep the analysis cell size and report bias vs resolution.

    The grid cell size is a parameter of the estimator (a box-kernel proxy for
    a Gaussian-process correlation length), not an intrinsic feature of the
    ungridded float data. sampling and grid bias both depend on it:
    coarse cells -> small sampling bias but large grid/smoothing bias;
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
    ref = ohc.truth_domain_mean(
        truth_field, weighted=weighted_reference, value_cols=list(value_cols)
    )
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
    cols = [
        "deg",
        "depth",
        "mean_n_truth_cells",
        "mean_sampled_fraction",
        "mean_true_GJ",
        "bias_GJ",
        "sampling_bias_GJ",
        "grid_bias_GJ",
    ]
    return out[cols].sort_values(["depth", "deg"]).reset_index(drop=True)


# ---- ANALYSIS-REGION SWEEP ----------------------------------------------
def subset_region(truth_field, sim, bounds):
    """Restrict the truth field and synthetic profiles to a lat/lon box.

    ``bounds`` is ``(lat_min, lat_max, lon_min, lon_max)``. Used to evaluate the
    bias over a defined analysis region rather than the whole download domain --
    e.g. starting at the float deployment footprint and expanding outward, so
    never-sampled shelf / far-field cells don't dominate the sampling term.
    """
    la0, la1, lo0, lo1 = bounds
    tf = truth_field.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    s = sim[sim["lat"].between(la0, la1) & sim["lon"].between(lo0, lo1)]
    return tf, s


def sweep_region(
    truth_field,
    sim,
    regions,
    deg=1.0,
    value_cols=("ohc_700", "ohc_2000"),
    weighted_reference=True,
):
    """Compute bias over a sequence of analysis regions at a fixed cell size.

    Parameters
    ----------
    regions : list of (bounds, label)
        ``bounds`` is ``(lat_min, lat_max, lon_min, lon_max)``; ``label`` names
        the region (e.g. "deploy", "+2deg").
    deg : float
        Analysis cell size held fixed across the region sweep.
    weighted_reference : bool
        cos(lat) area-weighted true domain mean per region (default) vs unweighted.

    Returns
    -------
    pandas.DataFrame
        One row per (region, depth) with time-averaged bias terms (GJ/m2), the
        region box, and its area in deg^2.
    """
    rows = []
    for bounds, label in regions:
        tf, s = subset_region(truth_field, sim, bounds)
        ref = ohc.truth_domain_mean(
            tf, weighted=weighted_reference, value_cols=list(value_cols)
        )
        truth_cells = ohc.coarsen_truth(tf, deg=deg)
        float_cells = ohc.grid_cells(s, list(value_cols), deg=deg)
        res = compute_bias(float_cells, truth_cells, value_cols, true_domain_mean=ref)
        summary = bias_summary(res["domain"], value_cols)
        la0, la1, lo0, lo1 = bounds
        summary.insert(0, "region", label)
        summary["area_deg2"] = (la1 - la0) * (lo1 - lo0)
        summary["n_profiles"] = len(s)
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    cols = [
        "region",
        "area_deg2",
        "depth",
        "n_profiles",
        "mean_sampled_fraction",
        "mean_true_GJ",
        "bias_GJ",
        "sampling_bias_GJ",
        "grid_bias_GJ",
    ]
    return out[cols].reset_index(drop=True)


def monthly_float_counts(sim):
    """Distinct floats and profile counts per month in a (region-filtered) sim.

    ``sim`` is the per-profile table from :func:`ohc.float_ohc`, already
    restricted to the analysis region (e.g. via :func:`subset_region`). Returns
    columns ``month, n_floats, n_profiles``.
    """
    s = sim.copy()
    s["month"] = pd.to_datetime(s["date"]).dt.to_period("M").dt.to_timestamp()
    g = s.groupby("month")
    out = pd.DataFrame({"n_floats": g["float_id"].nunique(), "n_profiles": g.size()})
    return out.reset_index()


# ---- PLOTS ---------------------------------------------------------------
def plot_domain_timeseries(
    domain,
    value_col="ohc_2000",
    out_path=None,
    title=None,
    ylim=None,
    real=None,
    en4=None,
    figsize=None,
):
    """Domain-mean OHC over time: GLORYS truth (all cells), GLORYS truth (sampled
    cells), and synthetic Argo (sampled cells), with the sampled-cell fraction below.

    The gap between truth-all and truth-sampled is the **sampling bias** (which
    cells get sampled); the gap between truth-sampled and synthetic Argo is the
    **grid bias** (within-cell estimate error). Pass ``ylim=(lo, hi)`` (GJ/m2) to
    fix the OHC axis -- e.g. a shared range across resolutions for one depth.

    ``real`` optionally adds a fourth line for the **real** Argo array: a
    DataFrame with a ``month`` column and a ``value_col`` column (J/m2), e.g. the
    monthly mean of ``data/argo_ohc.csv`` over the cells real floats sampled.

    ``en4`` optionally adds a fifth line for the **EN4** gridded product: a
    DataFrame with a ``month`` column and a ``value_col`` column (J/m2), e.g. the
    monthly domain mean of the EN4 cells at their native resolution.

    Pass ``figsize`` explicitly when embedding this in a context (e.g. a quarto
    panel-tabset built via ``display()``) where the chunk's ``fig-width``/
    ``fig-height`` options don't reach ``matplotlib.rcParams`` -- the default
    (``None``) falls back to ``rcParams["figure.figsize"]``.
    """
    import matplotlib.pyplot as plt

    fig, (ax, ax3, ax2) = plt.subplots(
        3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]}
    )
    pal = plt.get_cmap("Dark2").colors
    lw, ms = 1.2, 3
    m = domain["month"]
    ax.plot(
        m,
        domain[f"{value_col}_true_mean"] * J_TO_GJ,
        "-o",
        color=pal[0],
        lw=lw,
        markersize=ms,
        label="GLORYS truth (all cells)",
    )
    ax.plot(
        m,
        domain[f"{value_col}_true_at_sampled"] * J_TO_GJ,
        "-^",
        color=pal[1],
        lw=lw,
        markersize=ms,
        label="GLORYS truth (sampled cells)",
    )
    ax.plot(
        m,
        domain[f"{value_col}_float_mean"] * J_TO_GJ,
        "-s",
        color=pal[2],
        lw=lw,
        markersize=ms,
        label="synthetic Argo (sampled cells)",
    )
    if real is not None and value_col in real:
        ax.plot(
            real["month"],
            real[value_col] * J_TO_GJ,
            "-D",
            color=pal[3],
            lw=lw,
            markersize=ms,
            label="real Argo (sampled cells)",
        )
    if en4 is not None and value_col in en4:
        ax.plot(
            en4["month"],
            en4[value_col] * J_TO_GJ,
            "-v",
            color=pal[4],
            lw=lw,
            markersize=ms,
            label="EN4",
        )
    ax.set_ylabel(f"{value_col}  (GJ m$^{{-2}}$)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(frameon=False, fontsize="small")
    ax.set_title(title or f"Domain-mean OHC: truth vs synthetic Argo ({value_col})")
    ax.grid(alpha=0.2)

    samp_bias = domain[f"{value_col}_sampling_bias"] * J_TO_GJ
    grid_bias = domain[f"{value_col}_grid_bias"] * J_TO_GJ
    ax3.axhline(0, color="0.6", lw=0.8)
    ax3.plot(
        m,
        samp_bias + grid_bias,
        "-o",
        color="red",
        lw=lw,
        markersize=ms,
        label="total bias",
    )
    ax3.plot(
        m, samp_bias, "-s", color="#4a90d9", lw=lw, markersize=ms, label="sampling bias"
    )
    ax3.plot(
        m, grid_bias, "-^", color="#1a4f8a", lw=lw, markersize=ms, label="grid bias"
    )
    ax3.set_ylabel(f"bias  (GJ m$^{{-2}}$)")
    ax3.legend(frameon=False, fontsize="small")
    ax3.grid(alpha=0.2)

    ax2.bar(m, domain["sampled_fraction"], width=20, color="#4a90d9", alpha=0.7)
    for mi, frac in zip(m, domain["sampled_fraction"]):
        ax2.text(
            mi, frac, f"{frac * 100:.0f}%", ha="center", va="bottom", fontsize="x-small"
        )
    ax2.set_ylabel("sampled\ncell frac")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("month")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_bias_vs_resolution(
    sweep, value_col="ohc_2000", xscale="linear", out_path=None
):
    """Bias terms vs cell size (the box-kernel proxy for a GP length scale)."""
    import matplotlib.pyplot as plt

    df = sweep[sweep["depth"] == value_col].sort_values("deg")
    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(df["deg"], df["bias_GJ"], "-o", color="red", label="total bias")
    ax.plot(
        df["deg"], df["sampling_bias_GJ"], "-s", color="#4a90d9", label="sampling bias"
    )
    ax.plot(
        df["deg"],
        df["grid_bias_GJ"],
        "-^",
        color="#1a4f8a",
        label="grid bias",
    )
    ax.set_ylabel(f"{value_col} bias  (GJ m$^{{-2}}$)")
    ax.set_xscale(xscale)
    ax.legend(frameon=False)
    ax.set_title(f"OHC sampling bias vs analysis cell size ({value_col})")
    ax.grid(alpha=0.2, which="both")

    bar_width = (
        df["deg"] * 0.3
        if xscale == "log"
        else (df["deg"].max() - df["deg"].min()) * 0.06
    )
    ax2.bar(
        df["deg"],
        df["mean_sampled_fraction"],
        width=bar_width,
        color="#4a90d9",
        alpha=0.7,
    )
    for deg, frac in zip(df["deg"], df["mean_sampled_fraction"]):
        ax2.text(
            deg,
            frac,
            f"{frac * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize="x-small",
        )
    ax2.set_ylabel("sampled\ncell frac")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("cell size (deg)")
    ax2.set_xscale(xscale)
    ax2.grid(alpha=0.2, which="both")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_monthly_cell_maps(
    cells,
    value_col="ohc_2000",
    ncols=4,
    year=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    title=None,
    out_path=None,
    value_scale=J_TO_GJ,
    cbar_label=None,
    discrete=False,
    lats=None,
    lons=None,
    scatter_dots=False,
    width=None,
    xlim=None,
    ylim=None,
):
    """Facet of monthly cell maps from a gridded cells table.

    Works for both the truth cells (dense) and the synthetic-float cells
    (sparse) -- unsampled cells are left blank, so a float-cell panel doubles as
    a coverage map. Pass a shared ``vmin``/``vmax`` (e.g. from the truth) to make
    float and truth panels directly comparable.

    By default the cell grid (and hence the map extent) is taken from the cells
    present, so a sparse float panel spans a *smaller* extent than the dense
    truth. Pass explicit ``lats``/``lons`` (e.g. the truth cell centres) to force
    every source onto the **same grid and extent** -- essential when flipping
    between maps in tabs, so they line up exactly.

    ``xlim``/``ylim`` (each a ``(min, max)`` tuple) instead fix the *view*
    window directly, independent of the underlying data/grid resolution --
    useful for lining up tabs whose cell tables span different native
    resolutions (e.g. a coarse 1° climatology grid vs a native 1/12° truth
    grid): both can be cropped to the same on-screen extent even though their
    ``lats``/``lons`` differ.

    Defaults plot OHC in GJ/m2. To plot another quantity (e.g. the per-cell
    profile count ``n``) pass ``value_col="n"``, ``value_scale=1`` and a
    ``cbar_label``. With ``discrete=True`` the values are treated as integers
    and shown on a discrete colour scale with one band per integer level (use
    for counts). At fine resolutions (e.g. 1/12 deg) a sampled cell can be
    smaller than a pixel on screen; pass ``scatter_dots=True`` to overlay a
    fixed-size marker per sampled cell so sparse sources stay visible
    regardless of grid resolution.

    ``width`` (inches) sets the figure width directly -- needed in contexts
    (e.g. a quarto panel-tabset built via ``display()``) where the chunk's
    ``fig-width`` option doesn't reach ``matplotlib.rcParams``. Defaults to
    ``rcParams["figure.figsize"][0]`` when not given.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cell_dt = pd.to_datetime(cells["month"])
    if year is not None:
        months = [pd.Timestamp(year=year, month=i, day=1) for i in range(1, 13)]
    else:
        months = sorted(cell_dt.unique())

    if lats is None:
        lats = np.sort(cells["cell_lat"].unique())
    if lons is None:
        lons = np.sort(cells["cell_lon"].unique())
    vals = cells[value_col] * value_scale

    norm = ticks = None
    cmap_used = cmap
    if discrete:
        finite = vals[np.isfinite(vals)]
        lo = int(np.floor(vmin if vmin is not None else finite.min()))
        hi = int(np.ceil(vmax if vmax is not None else finite.max()))
        ncol = max(1, hi - lo + 1)
        cmap_used = plt.get_cmap(cmap, ncol)
        norm = mcolors.BoundaryNorm(np.arange(lo - 0.5, hi + 1.5, 1.0), ncol)
        ticks = np.arange(lo, hi + 1)
    else:
        if vmin is None:
            vmin = float(np.nanpercentile(vals, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(vals, 98))

    nrows = int(np.ceil(len(months) / ncols))
    if width is None:
        width = plt.rcParams["figure.figsize"][0]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, width / ncols * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    mesh = None
    for ax, m in zip(axes, months):
        if year is not None:
            d = cells[(cell_dt.dt.year == m.year) & (cell_dt.dt.month == m.month)]
        else:
            d = cells[cell_dt == m]
        grid = (
            d.pivot_table(
                index="cell_lat", columns="cell_lon", values=value_col
            ).reindex(index=lats, columns=lons)
        ) * value_scale
        mesh_kw = dict(cmap=cmap_used, shading="nearest")
        if norm is not None:
            mesh_kw["norm"] = norm
        else:
            mesh_kw.update(vmin=vmin, vmax=vmax)
        mesh = ax.pcolormesh(lons, lats, grid.values, **mesh_kw)
        if scatter_dots and len(d):
            scatter_kw = dict(
                cmap=cmap_used, edgecolors="black", linewidths=0.5, zorder=3
            )
            if norm is not None:
                scatter_kw["norm"] = norm
            else:
                scatter_kw.update(vmin=vmin, vmax=vmax)
            ax.scatter(
                d["cell_lon"],
                d["cell_lat"],
                c=d[value_col] * value_scale,
                s=30,
                **scatter_kw,
            )
        ax.set_aspect("equal")  # 1 deg lon == 1 deg lat, so cells render square
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(pd.Timestamp(m).strftime("%Y-%m"), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(months) :]:
        ax.axis("off")
    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.7, pad=0.02, ticks=ticks)
        cb.set_label(cbar_label or f"{value_col}  (GJ m$^{{-2}}$)")
    if title:
        fig.suptitle(title, y=1.0, fontsize=12)
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_bias_vs_region(sweep, value_col="ohc_2000", out_path=None):
    """Bias terms vs analysis-region size (expanding from the deployment box)."""
    import matplotlib.pyplot as plt

    df = sweep[sweep["depth"] == value_col].sort_values("area_deg2")
    x = df["area_deg2"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(x, df["bias_GJ"], "-o", color="#1a1a1a", label="total bias")
    ax.plot(x, df["sampling_bias_GJ"], "-s", color="#4a90d9", label="sampling bias")
    ax.plot(x, df["grid_bias_GJ"], "-^", color="#e07b39", label="grid bias")
    for xi, lab in zip(x, df["region"]):
        ax.annotate(
            lab,
            (xi, 0),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=7,
            color="0.4",
        )
    ax.set_ylabel(f"{value_col} bias  (GJ m$^{{-2}}$)")
    ax.legend(frameon=False)
    ax.set_title(f"OHC sampling bias vs analysis region ({value_col})")
    ax.grid(alpha=0.2)

    ax2.plot(x, df["mean_sampled_fraction"], "-o", color="#4a90d9")
    ax2.set_ylabel("sampled\ncell frac")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("analysis-region area (deg$^2$)")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig


def plot_bias_map(cells, value_col="ohc_2000", month=None, out_path=None):
    """Per-cell grid error (float - truth) as a scatter/heat map."""
    import matplotlib.pyplot as plt

    df = cells if month is None else cells[cells["month"] == month]
    bias = df[f"{value_col}_bias"] * J_TO_GJ
    vmax = np.nanpercentile(np.abs(bias), 95) if len(bias) else 1.0

    fig, ax = plt.subplots()
    sc = ax.scatter(
        df["cell_lon"],
        df["cell_lat"],
        c=bias,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        s=140,
        marker="s",
        edgecolor="0.6",
    )
    fig.colorbar(sc, ax=ax, label=f"{value_col} bias (GJ m$^{{-2}}$)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
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


def plot_bias_se_violin(
    bias_cells,
    se_cells,
    bias_col,
    se_col,
    quantiles=[0.05,.25,.75, 0.95],
    se_labels=None,
    se_colors=None,
    title=None,
    figsize=None,
    out_path=None,
):
    """Side-by-side horizontal violins per month: a bias distribution vs one or more SE distributions.

    For an interpolation method that fills in a value everywhere (not just at
    binned cells a float happened to sample), the natural per-month "bias" is
    a full spatial distribution -- |predicted minus truth| at every grid point
    -- directly comparable to that same method's per-pixel SE distribution,
    rather than a single domain-level scalar compared against a domain-mean
    SE. ``bias_cells``/``se_cells`` are cells tables (``month, cell_lat,
    cell_lon, <col>``) sharing the same ``month`` values; ``bias_col``/
    ``se_col`` select which column of each to plot. Bias is shown as its
    absolute value, so both distributions are >= 0 and live on the same
    GJ/m2 axis with directly comparable spreads. Each violin marks its
    median (solid), mean (dashed), and 5th/95th percentiles (solid) -- the
    percentiles in place of the true min/max, which are sensitive to single
    outlier grid cells.

    ``se_cells`` and ``se_col`` may each be a single DataFrame/str (one SE
    violin) or a list of DataFrames/strs (multiple SE violins, e.g. ``se_A``
    and ``se_0`` for Levitus). ``se_labels`` and ``se_colors`` customise the
    legend entry and fill colour for each SE source; defaults are provided
    when omitted. Violins are evenly spaced across a fixed span so they
    never overlap regardless of how many SE sources are passed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Normalise se_cells/se_col to lists.
    if isinstance(se_cells, list):
        se_cells_list = se_cells
        se_col_list = se_col
    else:
        se_cells_list = [se_cells]
        se_col_list = [se_col]
    n_se = len(se_cells_list)

    _default_colors = ["#4a90d9", "#f39c12", "#2ecc71", "#9b59b6"]
    if se_labels is None:
        se_labels = (
            ["interpolation SE"] if n_se == 1 else [f"SE {i + 1}" for i in range(n_se)]
        )
    if se_colors is None:
        se_colors = _default_colors[:n_se]

    months = sorted(bias_cells["month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%b") for m in months]
    bias_data = [
        (bias_cells.loc[bias_cells["month"] == mo, bias_col] * J_TO_GJ)
        .abs()
        .dropna()
        .to_numpy()
        for mo in months
    ]
    se_data_list = [
        [
            (cells.loc[cells["month"] == mo, col] * J_TO_GJ).dropna().to_numpy()
            for mo in months
        ]
        for cells, col in zip(se_cells_list, se_col_list)
    ]

    # Drop months where any distribution is empty — violinplot crashes on zero-size arrays.
    valid = [
        len(bias_data[i]) > 1
        and all(len(sd[i]) > 1 for sd in se_data_list)
        for i in range(len(months))
    ]
    months = [m for m, v in zip(months, valid) if v]
    month_labels = [l for l, v in zip(month_labels, valid) if v]
    bias_data = [d for d, v in zip(bias_data, valid) if v]
    se_data_list = [
        [d for d, v in zip(sd, valid) if v]
        for sd in se_data_list
    ]

    positions = np.arange(1, len(months) + 1)
    n_total = 1 + n_se
    # Evenly space all violins (bias + SE sources) within a fixed span.
    span = min(0.70, 0.20 * n_total)
    offsets = np.linspace(-span / 2, span / 2, n_total)
    width = (span / (n_total - 1)) * 0.80 if n_total > 1 else span

    quantiles = [quantiles] * len(months)
    fig, ax = plt.subplots(figsize=figsize or (7, 8))

    parts_bias = ax.violinplot(
        bias_data,
        positions=positions + offsets[0],
        vert=False,
        widths=width,
        showmedians=True,
        showmeans=True,
        showextrema=False,
        quantiles=quantiles,
    )
    for body in parts_bias["bodies"]:
        body.set_facecolor("#c0392b")
        body.set_alpha(0.5)
    for key in ("cmedians", "cmeans", "cquantiles"):
        parts_bias[key].set_color("black")
        parts_bias[key].set_linewidth(0.5)

    parts_bias["cquantiles"].set_color("gray")
    parts_bias["cmedians"].set_linestyle(":")

    for i, (se_data, color) in enumerate(zip(se_data_list, se_colors)):
        parts_se = ax.violinplot(
            se_data,
            positions=positions + offsets[1 + i],
            vert=False,
            widths=width,
            showmedians=True,
            showmeans=True,
            showextrema=False,
            quantiles=quantiles,
        )
        for body in parts_se["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        for key in ("cmedians", "cmeans", "cquantiles"):
            parts_se[key].set_color("black")
            parts_se[key].set_linewidth(0.5)

        parts_se["cmedians"].set_linestyle(":")
        parts_se["cquantiles"].set_color("gray")

    ax.axvline(0, color="0.6", lw=0.8, zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels(month_labels)
    ax.invert_yaxis()  # January at top
    ax.set_xlabel("GJ m$^{-2}$")
    if title:
        ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="#c0392b", alpha=0.5, label="bias"),
            *[
                Patch(facecolor=c, alpha=0.5, label=lbl)
                for c, lbl in zip(se_colors, se_labels)
            ],
        ],
        frameon=False,
        fontsize="small",
        loc="lower right",
    )
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig
