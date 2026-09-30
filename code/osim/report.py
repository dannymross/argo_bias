"""Summary tables and figures for `reports/osim.qmd`.

The presentation layer: everything here formats or draws a quantity that
mean_mc.py computes. Imported by the report, never run.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg
from osim.mean_mc import (
    load_krige,
    empirical_efolds,
    load_krige_params,
    union_bias,
    area_share,
    J_TO_GJ,
    M_S_TO_CM_S,
    NW_BOXB_2015,
    VEL_COMPS,
    _decorrelation_km,
    _sd_Z,
    _velocity_year,
    argo_actual_profile_bias,
    argo_region_summary,
    bathymetry,
    clean_argo,
    combined_quantile_days,
    combined_velocity_quantile_days,
    domain_exit_fates,
    domain_extents,
    float_bias_profile,
    mean_field_months,
    ohc_decomposition,
    ohc_domain_series,
    raw_deployments,
    sampling_weight_ratio,
    sampling_weight_space,
    trajectory_divergence,
)


# ----------------------------------------------------------------------------
# map scaffolding
# ----------------------------------------------------------------------------


def _both_box_extent(boxes=("boxB", "boxA"), pad=1.0):
    """Map extent covering every box, padded, for a shared both-box view."""
    la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes)
    la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes)
    lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes)
    lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes)
    return [lo0 - pad, lo1 + pad, la0 - pad, la1 + pad]


def _cluster_colors(float_ids):
    import matplotlib.pyplot as plt
    cyc = plt.get_cmap("tab10")
    return {int(f): cyc(i % 10) for i, f in enumerate(float_ids)}


def _cluster_track_map(paths, float_ids, boxes=("boxA", "boxB"), figsize=(8.5, 7.2),
                       title=""):
    """Draw one colour-per-float trajectory set on the domain base map, with a black
    star at each deployment (first point) and an open circle at the last point."""
    fig, ax, pc, e = _domain_base(boxes, figsize)
    cols = _cluster_colors(float_ids)
    for fid in float_ids:
        g = paths.get(int(fid))
        if g is None or len(g) == 0:
            continue
        c = cols[int(fid)]
        ax.plot(g["lon"], g["lat"], "-", color=c, lw=0.9, alpha=0.9, transform=pc,
                zorder=3, label=str(int(fid)))
        ax.plot(g["lon"].iloc[0], g["lat"].iloc[0], "*", color="k", mfc=c, ms=13,
                mew=0.8, transform=pc, zorder=5)
        ax.plot(g["lon"].iloc[-1], g["lat"].iloc[-1], "o", mfc="none", mec=c, mew=1.4,
                ms=7, transform=pc, zorder=4)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, title="float")
    return fig


def _domain_base(boxes=("boxA", "boxB"), figsize=(7.5, 6.4)):
    """Base analysis-domain map: land, the velocity/thetao regions, and the boxes.

    Returns (fig, ax, pc, extents); the caller adds the legend and any overlay.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    e = domain_extents(boxes)
    pc = ccrs.PlateCarree()
    # tight_layout collapses the GeoAxes boundary that the gridliner draws
    # against, so layout is left to constrained mode here and below.
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": pc},
                           layout="constrained")
    v = e["velocity"]
    ext = [v[2] - 1.5, v[3] + 1.5, v[0] - 1.5, v[1] + 1.5]
    ax.set_extent(ext, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.5)

    style = {"velocity": ("tab:blue", "velocity (advection domain)"),
             "thetao": ("tab:green", "thetao (OHC truth + mean field)"),
             "boxA": ("tab:orange", "Box A (subtropical gyre)"),
             "boxB": ("crimson", "Box B (NAC / Gulf Stream)")}
    for name in ("velocity", "thetao") + tuple(boxes):
        la0, la1, lo0, lo1 = e[name]
        color, label = style[name]
        fill = name in boxes
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                               fill=fill, facecolor=color if fill else "none",
                               alpha=0.18 if fill else 1.0, edgecolor=color,
                               lw=2.0, ls="-" if fill else "--", zorder=2, label=label))
    _geo_ticks(ax, ext, True, True, step=10.0)
    return fig, ax, pc, e


def _draw_domain_map(ax, das, norm, cmap, extent, title, bottom_labels, left_labels,
                     scale=J_TO_GJ, boxes=("boxB", "boxA")):
    """Draw one or more per-box field patches on a shared both-box base map.

    Each DataArray in `das` carries its own lat/lon, so the disjoint boxes land
    in place on one map covering both; box outlines are added for orientation.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle
    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.4)
    im = None
    for da in das:
        im = ax.pcolormesh(da["longitude"], da["latitude"], da.values * scale,
                           cmap=cmap, norm=norm, transform=pc, shading="nearest",
                           antialiased=False, linewidth=0, rasterized=True)
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                               fill=False, edgecolor="0.1", lw=0.8, zorder=4))
    ax.set_title(title, fontsize=9)
    _geo_ticks(ax, extent, bottom_labels, left_labels)
    return im


def _geo_ticks(ax, extent, bottom_labels, left_labels, step=5.0):
    """Lat/lon ticks via the ordinary matplotlib tick machinery.

    Deliberately NOT ax.gridlines(draw_labels=True): the cartopy gridliner makes
    GeoAxes.get_tightbbox raise on a NaN label position, matplotlib swallows
    that, and the figure's tight bbox then collapses onto whatever axes did
    succeed -- so a bbox_inches="tight" save (what the Jupyter inline backend
    does, hence Quarto) drops every map and keeps only the colorbars.
    """
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    lo0, lo1, la0, la1 = extent
    ax.set_xticks(np.arange(np.ceil(lo0 / step) * step, lo1 + 1e-9, step),
                  crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(la0 / step) * step, la1 + 1e-9, step),
                  crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7, labelbottom=bottom_labels, labelleft=left_labels)
    ax.grid(lw=0.25, color="0.6", alpha=0.4)


def _map_grid(nrow, ncol, figsize, cbar=True, left=0.05, right=0.885, bottom=0.05,
              top=0.90, hgap=0.015, vgap=0.11):
    """GeoAxes grid at explicit positions, optionally a colorbar slot per row.

    Positions are set by hand rather than by a layout engine: cartopy gridliners
    raise inside tight_layout, and constrained layout resolves row-spanning
    colorbars inconsistently, which silently collapses the map axes.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    right = right if cbar else 0.985
    fig = plt.figure(figsize=figsize)
    w = (right - left - (ncol - 1) * hgap) / ncol
    h = (top - bottom - (nrow - 1) * vgap) / nrow
    axes, caxes = [], []
    for r in range(nrow):
        y = top - (r + 1) * h - r * vgap
        axes.append([fig.add_axes([left + c * (w + hgap), y, w, h], projection=pc)
                     for c in range(ncol)])
        if cbar:
            caxes.append(fig.add_axes([right + 0.012, y + 0.04 * h, 0.014, h * 0.92]))
    return fig, axes, caxes


def _raw_area(box):
    """Wet-cell index and unnormalized cell areas (m^2) on the analysis mask."""
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    a = a.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return wet, a[wet]


def _trajectory_map(tracks, boxes=("boxA", "boxB"), max_floats=60, figsize=(8.5, 7.2),
                    order="time", years=None, subset_seed=None):
    """Float drift tracks + profile points on the domain base map, coloured by the
    calendar year of each profile. `tracks` needs float_id, lat, lon, year and an
    `order` column; `years` fixes the colour scale so two maps stay comparable."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import LineCollection

    tracks = tracks.dropna(subset=["lat", "lon"]).sort_values(["float_id", order])
    ids = tracks["float_id"].unique()
    if max_floats is not None and len(ids) > max_floats:
        ids = (np.random.default_rng(subset_seed).choice(ids, max_floats, replace=False)
               if subset_seed is not None else np.sort(ids)[:max_floats])
        tracks = tracks[tracks["float_id"].isin(ids)]
    yr = years or sorted(int(y) for y in tracks["year"].unique())
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)

    fig, ax, pc, e = _domain_base(boxes, figsize)
    for _, g in tracks.groupby("float_id"):
        p = g[["lon", "lat"]].to_numpy()
        if len(p) < 2:
            continue
        segs = np.stack([p[:-1], p[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=cmap(norm(g["year"].to_numpy()[:-1])),
                                         lw=0.35, alpha=0.5, transform=pc, zorder=1.5))
    ax.scatter(tracks["lon"], tracks["lat"], c=tracks["year"], cmap=cmap, norm=norm,
               s=1.5, alpha=0.75, transform=pc, zorder=3)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        ticks=yr, shrink=0.8, pad=0.02)
    cbar.set_label("profile year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


# ----------------------------------------------------------------------------
# summary tables
# ----------------------------------------------------------------------------


def argo_comparison_table(uniform, actual, real, boot=None, boxes=("boxB", "boxA")):
    """Bias by box/year across deployment laws (GJ/m2): the uniform-seeded ensemble
    mean, the single actual-deployment realization, and the real Argo array. Pass
    `boot` (the resampled real-deployment ensemble) to add its mean/sd columns."""
    rows = []
    for b in boxes:
        u = uniform[uniform["box"] == b].groupby("year")["B"].mean()
        ac = actual[actual["box"] == b].set_index("year")["B"]
        bt = boot[boot["box"] == b].groupby("year")["B"] if boot is not None else None
        rr = real[b]
        for y in sorted(int(v) for v in u.index):
            row = {"box": b, "year": y, "uniform": u[y] * J_TO_GJ}
            if bt is not None:
                row["argo-boot mean"] = bt.mean().get(y, np.nan) * J_TO_GJ
                row["argo-boot sd"] = bt.std().get(y, np.nan) * J_TO_GJ
            row["argo-actual"] = ac.get(y, np.nan) * J_TO_GJ
            row["real Argo"] = rr.get(y, np.nan) * J_TO_GJ
            rows.append(row)
    return pd.DataFrame(rows)


def continuous_bias_table(est, box, ref=None):
    """Per-year advective bias distribution, with the uniform-null mean alongside.

    The `uniform mean B` column is the mean of the large `ref` ensemble when given,
    else the aligned per-rep B_unif in `est`.
    """
    sd = _sd_Z(box)
    rows = []
    for k, gk in est[est["box"] == box].groupby("year"):
        b = gk["B"].to_numpy()
        um = (ref[(ref["box"] == box) & (ref["year"] == k)]["B_unif"].mean()
              if ref is not None else gk["B_unif"].mean())
        rows.append({"year": int(k), "reps": b.size,
                     "mean B": b.mean() * J_TO_GJ,
                     "MC se": b.std(ddof=1) / np.sqrt(b.size) * J_TO_GJ,
                     "sd over deployments": b.std(ddof=1) * J_TO_GJ,
                     "5%": np.percentile(b, 5) * J_TO_GJ,
                     "95%": np.percentile(b, 95) * J_TO_GJ,
                     "P(B < 0)": float((b < 0).mean()),
                     "sd_k": sd[int(k)] * J_TO_GJ,
                     "mean B / sd_k": b.mean() / sd[int(k)],
                     "uniform mean B": um * J_TO_GJ})
    return pd.DataFrame(rows)


def continuous_pilot_size_table(est, boxes=("boxB", "boxA")):
    """Per box/year pilot sample sizes: mean in-box standing stock and distinct
    floats per realization, mean profile observations per realization, and the
    profile total over all realizations."""
    rows = []
    for b in boxes:
        g = est[est["box"] == b]
        for y in sorted(int(v) for v in g["year"].unique()):
            gy = g[g["year"] == y]
            rows.append({"box": b, "year": y,
                         "in-box stock": gy["standing_stock"].mean(),
                         "floats / realization": gy["n_floats"].mean(),
                         "profiles / realization": gy["n"].mean(),
                         "profiles, all reps": int(gy["n"].sum())})
    return pd.DataFrame(rows)


def continuous_stock_table(est, boxes=("boxB", "boxA")):
    """Virtual against real Argo standing stock, both boxes, by analysis year.

    Standing stock is the mean number of distinct floats reporting in a 10-day
    window -- the quantity that compares across the two, since real and virtual
    floats cycle on different schedules. Real stock is from the Argo profile
    record restricted to the box; virtual stock is the ensemble-mean in-box stock.
    """
    out = None
    for b in boxes:
        sim = est[est["box"] == b].groupby("year")["standing_stock"].mean()
        real = argo_region_summary(b).set_index("year")["argo_stock"].reindex(sim.index)
        t = pd.DataFrame({"year": sim.index.astype(int),
                          f"Argo {b}": real.to_numpy(),
                          f"virtual {b}": sim.to_numpy(),
                          f"ratio {b}": sim.to_numpy() / real.to_numpy()})
        out = t if out is None else out.merge(t, on="year")
    out.loc["mean"] = ["mean", *out.iloc[:, 1:].mean()]
    return out


def domain_exit_table(fates=None):
    """Counts and fractions of each fate (domain_exit / lifespan / record_end)."""
    f = domain_exit_fates() if fates is None else fates
    t = f["fate"].value_counts().reindex(["domain_exit", "lifespan", "record_end"]).fillna(0).astype(int)
    return pd.DataFrame({"fate": t.index, "floats": t.to_numpy(),
                         "percent": (100 * t / len(f)).round(1).to_numpy()})


def truth_table(boxes=("boxB", "boxA")):
    """Domain mean and dispersion of the residual field, both boxes side by side."""
    out = None
    for b in boxes:
        t = pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{b}.csv"))
        t = pd.DataFrame({"year": t["year"],
                          f"Zbar {b}": t["zbar"] * J_TO_GJ,
                          f"sd_k {b}": t["sd_Z"] * J_TO_GJ,
                          f"Zbar/sd_k {b}": t["zbar"] / t["sd_Z"]})
        out = t if out is None else out.merge(t, on="year")
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------


def plot_argo_actual_bias_cohort(deploy_year, bias=None, boxes=("boxB", "boxA"), figsize=(13, 5.0)):
    """One panel per box for a deployment-year cohort: a thin line per argo-actual
    float (deployed in `deploy_year`) of its per-profile bias Z(x_i)-Zbar_k (GJ/m2)
    over its whole life, in-box profiles joined in age order across out-of-domain
    gaps and year boundaries. Pass a precomputed `bias` frame
    (argo_actual_profile_bias) to avoid recomputing per cohort."""
    import matplotlib.pyplot as plt
    s = argo_actual_profile_bias(boxes) if bias is None else bias
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize, sharex=True)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        g = s[(s["box"] == b) & (s["deploy_year"] == int(deploy_year))]
        for _, gg in g.groupby("float_id"):
            gg = gg.sort_values("age")
            ax.plot(gg["age"], gg["delta"] * J_TO_GJ, color="0.35", lw=0.4, alpha=0.55)
        ax.axhline(0, color="crimson", lw=0.8)
        ax.set_title(f"{b} ({g['float_id'].nunique()} floats, n={len(g)})", fontsize=10)
        ax.set_xlabel("days since deployment")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel(r"single-profile bias $Z(x_i)-\bar Z_k$ (GJ/m$^2$)")
    fig.suptitle(f"Argo-actual per-float bias over life, {deploy_year} deployment cohort", fontsize=11)
    fig.tight_layout()
    return fig


def plot_argo_comparison(uniform, actual, real, boot=None, boxes=("boxB", "boxA"), figsize=(11, 4.2)):
    """Deployment-law comparison, one panel per box: dodged 95% CI + 5-95% intervals
    for the uniform ensemble (and the real-deployment bootstrap if `boot` is given),
    with the single actual-deployment realization and real Argo overlaid as markers."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ens = [(uniform, "0.45", -0.18 if boot is not None else 0.0)]
    if boot is not None:
        ens.append((boot, "tab:purple", 0.18))
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    for ax, box in zip(np.atleast_1d(axes), boxes):
        years = sorted(int(y) for y in uniform.loc[uniform["box"] == box, "year"].unique())
        for src, color, dx in ens:
            g = src[src["box"] == box]
            for y in years:
                b = g.loc[g["year"] == y, "B"].to_numpy() * J_TO_GJ
                if not len(b):
                    continue
                m, se = b.mean(), b.std(ddof=1) / np.sqrt(len(b))
                q05, q95 = np.percentile(b, [5, 95])
                ax.plot([y + dx] * 2, [q05, q95], color=color, lw=1.0, alpha=0.5, zorder=2)
                ax.errorbar(y + dx, m, yerr=1.96 * se, fmt="o", color=color, ms=4,
                            elinewidth=1.4, capsize=3, zorder=4)
        ga = actual[actual["box"] == box].set_index("year")["B"] * J_TO_GJ
        rr = real[box] * J_TO_GJ
        yy = [y for y in years if y in ga.index]
        ax.plot(yy, [ga[y] for y in yy], "D", mfc="none", mec="tab:orange", ms=6, zorder=5)
        yr = [y for y in years if y in rr.index]
        ax.plot(yr, [rr[y] for y in yr], "X", color="tab:green", ms=8, zorder=6)
        ax.axhline(0.0, color="crimson", lw=1.4, zorder=1)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_title(box, fontsize=10)
        ax.set_xlabel("analysis year", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("sampling bias B (GJ/m2)")
    handles = [Line2D([0], [0], color="0.45", lw=1.4, marker="o", label="uniform sim")]
    if boot is not None:
        handles.append(Line2D([0], [0], color="tab:purple", lw=1.4, marker="o", label="argo-bootstrap"))
    handles += [Line2D([0], [0], color="tab:orange", lw=0, marker="D", mfc="none", label="argo-actual (1 real.)"),
                Line2D([0], [0], color="tab:green", lw=0, marker="X", label="real Argo")]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_argo_trajectory_map(boxes=("boxA", "boxB"), years=(2010, 2019),
                             buffer=((15, 50), (-70, -30)), max_floats=60,
                             figsize=(8.5, 7.2), path=None):
    """Real Argo trajectories over the advection domain, same base map and year
    colouring as the simulated map, for side-by-side comparison. Positions are the
    profile (surface) locations in data/argo_ohc.csv; a random subset of the floats
    present in the domain is drawn for legibility."""
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d["year"] = d["date"].dt.year
    (la0, la1), (lo0, lo1) = buffer
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)
          & d["year"].between(*years)].rename(columns={"date": "time"})
    return _trajectory_map(d, boxes, max_floats, figsize, order="time",
                           years=list(range(years[0], years[1] + 1)), subset_seed=0)


def plot_bathymetry(boxes=("boxA", "boxB"), threshold=2000.0, figsize=(13.5, 6.2),
                    cap=6000.0):
    """Ocean depth over the advection domain, and what the depth cut removes.

    Left: deptho, capped for display so the shelf break is resolved rather than
    compressed against the abyssal plain. Right: the cells the cut excludes.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.patches import Rectangle, Patch

    dep = bathymetry()
    e = domain_extents(boxes)
    v = e["velocity"]
    ext = [v[2], v[3], v[0], v[1]]
    pc = ccrs.PlateCarree()

    fig, axes, caxes = _map_grid(1, 2, figsize, top=0.92, bottom=0.08, right=0.90)
    shallow = dep.where(dep <= threshold)
    for j, ax in enumerate(axes[0]):
        ax.set_extent(ext, crs=pc)
        ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=3)
        ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=4)
        if j == 0:
            im = ax.pcolormesh(dep["longitude"], dep["latitude"],
                               np.minimum(dep.values, cap) * 1e-3, cmap="Blues",
                               norm=Normalize(0.0, cap * 1e-3), transform=pc,
                               shading="nearest", antialiased=False, linewidth=0,
                               rasterized=True)
            ax.set_title(f"GLORYS deptho (capped at {cap * 1e-3:.0f} km)", fontsize=9)
        else:
            # NaN, not 0, off the painted set: a one-colour ListedColormap maps
            # every finite value to its single colour, so a 0/1 indicator paints
            # the whole panel. Only NaN is left unpainted.
            for m, col in ((np.isfinite(dep.values), "0.93"),
                           (np.isfinite(shallow.values), "tab:red")):
                ax.pcolormesh(dep["longitude"], dep["latitude"],
                              np.where(m, 1.0, np.nan), cmap=ListedColormap([col]),
                              transform=pc, shading="nearest", antialiased=False,
                              linewidth=0, rasterized=True, zorder=2)
            ax.set_title(f"ocean shallower than {threshold:.0f} m (excluded)", fontsize=9)
            ax.legend(handles=[Patch(facecolor="tab:red", label=f"< {threshold:.0f} m"),
                               Patch(facecolor="0.93", label="analysis-eligible")],
                      loc="lower left", fontsize=7, framealpha=0.9)
        for b in boxes:
            la0, la1, lo0, lo1 = e[b]
            ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                                   fill=False, edgecolor="k", lw=1.8, zorder=5))
            ax.text(lo0 + 0.4, la1 - 1.4, b, transform=pc, fontsize=8, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
        _geo_ticks(ax, ext, True, j == 0, step=10.0)
    fig.colorbar(im, cax=caxes[0], label="depth (km)")
    return fig


def plot_bias_intervals(est, ref, boxes=("boxB", "boxA"), figsize=(11, 4.0)):
    """Headline bias chart: per year, dodged intervals for the advective bias (from
    `est`, 100 realizations) and the uniform null (from `ref`, the large n-matched
    ensemble), one panel per box (own y-axis).

    For each (box, year, source): point = ensemble mean; capped bar = 95% CI
    (mean +/- 1.96 MC se), so an interval clearing zero flags a mean distinguishable
    from zero at the 5% level; thin whisker = 5-95% (the scatter of a single
    deployment). Red line at 0 is the target.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    sources = [(est, "B", "0.35", "advective", -0.15),
               (ref, "B_unif", "tab:blue", "uniform (P)", 0.15)]
    for ax, box in zip(np.atleast_1d(axes), boxes):
        years = sorted(int(y) for y in est.loc[est["box"] == box, "year"].unique())
        for src, col, color, _lab, dx in sources:
            gs = src[src["box"] == box]
            for y in years:
                b = gs.loc[gs["year"] == y, col].to_numpy() * J_TO_GJ
                m, se = b.mean(), b.std(ddof=1) / np.sqrt(b.size)
                q05, q95 = np.percentile(b, [5, 95])
                x = y + dx
                ax.plot([x, x], [q05, q95], color=color, lw=1.0, alpha=0.6, zorder=2)
                ax.errorbar(x, m, yerr=1.96 * se, fmt="o", color=color, ms=4.5,
                            elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=4)
        ax.axhline(0.0, color="crimson", lw=1.4, zorder=1)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_title(box, fontsize=10)
        ax.set_xlabel("analysis year", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("sampling bias B (GJ/m2)")
    handles = [Line2D([0], [0], color=c, lw=1.4, marker="o", label=l)
               for _, _, c, l, _ in sources]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_cluster_real_tracks(float_ids=NW_BOXB_2015, boxes=("boxA", "boxB"),
                             figsize=(8.5, 7.2), path=None):
    """Real Argo (raw) trajectories of the named floats, one colour each."""
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d[d["float_id"].isin([int(f) for f in float_ids])].dropna(subset=["lat", "lon"])
    paths = {int(f): g.sort_values("date") for f, g in d.groupby("float_id")}
    return _cluster_track_map(paths, float_ids, boxes, figsize,
                              "Real Argo trajectories (deployment star, last-fix circle)")


def plot_cluster_sim_tracks(float_ids=NW_BOXB_2015, boxes=("boxA", "boxB"),
                            figsize=(8.5, 7.2)):
    """GLORYS park-advection sim-twin trajectories of the named floats (argo-actual
    arm), matched to float_id through the deployment pool, one colour each."""
    pool = pd.read_csv(os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"))
    tr = pd.read_csv(os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"),
                     parse_dates=["time"])
    sid = pool.set_index("float_id")["seed_id"]
    paths = {}
    for fid in float_ids:
        s = int(sid.get(int(fid), -1))
        g = tr[tr["seed_id"] == s].sort_values("time")
        paths[int(fid)] = g.dropna(subset=["lat", "lon"])
    return _cluster_track_map(paths, float_ids, boxes, figsize,
                              "GLORYS park-advection sim-twin trajectories")


def plot_continuous_bias(est, box, ref=None, argo=None, figsize=(13, 3.0)):
    """Bias distribution over realizations, one panel per year; uniform null overlaid
    (from the large `ref` ensemble when given). If `argo` (a Series year -> B in
    J/m^2 from argo_bias_by_year) is given, the realized real-Argo bias is drawn as
    a green reference line, and the x-axis is widened to include it."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    g = est[est["box"] == box]
    gu = ref[ref["box"] == box] if ref is not None else g
    years = sorted(int(y) for y in g["year"].unique())
    allB = g["B"].to_numpy() * J_TO_GJ
    lo, hi = np.percentile(allB, [0.5, 99.5])
    pad = 0.08 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 28)
    xlo, xhi = lo - pad, hi + pad
    if argo is not None:
        av = argo * J_TO_GJ
        xlo, xhi = min(xlo, av.min() - pad), max(xhi, av.max() + pad)
    fig, axes = plt.subplots(1, len(years), figsize=figsize, sharex=True, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), years):
        gk = g[g["year"] == k]
        ax.hist(gk["B"] * J_TO_GJ, bins=bins, density=True, color="0.35")
        ax.hist(gu[gu["year"] == k]["B_unif"] * J_TO_GJ, bins=bins, density=True,
                histtype="step", color="tab:blue", lw=1.2, ls="--")
        ax.axvline(0.0, color="crimson", lw=1.6)
        ax.axvline(gk["B"].mean() * J_TO_GJ, color="k", lw=1.4, ls=":")
        if argo is not None and k in argo.index:
            ax.axvline(argo[k] * J_TO_GJ, color="tab:green", lw=1.8)
        ax.set_xlim(xlo, xhi)
        ax.set_title(str(k), fontsize=10)
        ax.set_xlabel("B (GJ/m2)", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("density")
    if argo is not None:
        np.atleast_1d(axes)[0].legend(
            handles=[Line2D([0], [0], color="tab:green", lw=1.8, label="real Argo"),
                     Line2D([0], [0], color="tab:blue", lw=1.2, ls="--", label="uniform"),
                     Line2D([0], [0], color="crimson", lw=1.6, label="zero")],
            fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_continuous_stock(est, boxes=("boxB", "boxA"), figsize=(11, 3.6)):
    """Virtual (mean +/- sd over realizations) against real Argo standing stock."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        g = est[est["box"] == b]
        yrs = sorted(int(y) for y in g["year"].unique())
        m = g.groupby("year")["standing_stock"].mean().reindex(yrs)
        s = g.groupby("year")["standing_stock"].std().reindex(yrs)
        real = argo_region_summary(b).set_index("year")["argo_stock"].reindex(yrs)
        ax.errorbar(yrs, m.to_numpy(), yerr=s.to_numpy(), fmt="o-", color="0.2",
                    capsize=3, label="virtual (mean +/- sd)")
        ax.plot(yrs, real.to_numpy(), "s--", color="tab:red", label="real Argo")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("year")
        ax.set_ylim(bottom=0)
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel("in-box stock (floats per 10-day epoch)")
    a0.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_deployment_advection_map(tracks=None, boxes=("boxA", "boxB"),
                                  max_floats=60, figsize=(8.5, 7.2)):
    """One simulated realization's advection: drift tracks + profile points by year."""
    if tracks is None:
        tracks = pd.read_csv(os.path.join(cfg.OUT_DIR, "illustration_tracks.csv"))
    return _trajectory_map(tracks, boxes, max_floats, figsize, order="time",
                           years=list(range(2010, 2020)))


def plot_deployment_positions_map(years=(2010, 2019), domain="velocity",
                                  boxes=("boxA", "boxB"), figsize=(8.5, 7.2)):
    """Raw (not QC-cleaned) float deployment positions in the advection domain,
    coloured by deployment year, on the shared domain base map."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    dep = raw_deployments(years, domain, boxes)
    yr = list(range(years[0], years[1] + 1))
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)
    fig, ax, pc, e = _domain_base(boxes, figsize)
    ax.scatter(dep["lon"], dep["lat"], c=dep["year"], cmap=cmap, norm=norm, s=30,
               alpha=0.9, edgecolor="k", linewidth=0.3, transform=pc, zorder=3)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        ticks=yr, shrink=0.8, pad=0.02)
    cbar.set_label("deployment year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def plot_domain_exit_map(fates=None, boxes=("boxA", "boxB"), figsize=(8.5, 7.2)):
    """Last in-domain position of every float that advected OUT of the velocity
    domain, on the shared base map -- where virtual floats hit the boundary."""
    import matplotlib.pyplot as plt
    f = domain_exit_fates() if fates is None else fates
    ex = f[f["fate"] == "domain_exit"]
    fig, ax, pc, e = _domain_base(boxes, figsize)
    ax.scatter(ex["lon"], ex["lat"], c="red", s=45, marker="X", edgecolor="k",
               linewidth=0.4, transform=pc, zorder=4,
               label=f"domain exit (n={len(ex)})")
    ax.set_title("Where virtual floats leave the velocity domain (last in-domain position)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return fig


def plot_domain_map(boxes=("boxA", "boxB"), figsize=(7.5, 6.4)):
    """Analysis boxes inside the velocity and thetao download regions."""
    fig, ax, pc, e = _domain_base(boxes, figsize)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return fig


def plot_float_bias_profile(float_ids=NW_BOXB_2015, box="boxB", x="pnum",
                            source="real", figsize=(11, 5.0)):
    """Single-profile bias delta vs profile (or age) for each named float + the
    cluster mean, in GJ/m2. Only in-box profiles carry a defined bias."""
    import matplotlib.pyplot as plt
    s = float_bias_profile(float_ids, box, source)
    cols = _cluster_colors(float_ids)
    fig, ax = plt.subplots(figsize=figsize)
    for fid, g in s.groupby("float_id"):
        g = g.sort_values(x)
        ax.plot(g[x], g["delta"] * J_TO_GJ, "-o", ms=3, lw=1.0, color=cols[int(fid)],
                alpha=0.85, label=f"{int(fid)} (n={len(g)})")
    xb = np.linspace(s[x].min(), s[x].max(), 9)
    s = s.assign(_b=pd.cut(s[x], xb))
    m = s.groupby("_b", observed=True).agg(xc=(x, "mean"), mu=("delta", "mean"))
    ax.plot(m["xc"], m["mu"] * J_TO_GJ, "k-", lw=2.5, zorder=5, label="cluster mean")
    ax.axhline(0, color="0.4", lw=0.8)
    xlab = {"pnum": "profile number (cleaned record, ~10-day cycles)",
            "age": "days since deployment"}[x]
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"single-profile bias  $Z(x_i)-\bar Z_k$  (GJ/m$^2$)")
    src = {"real": "real Argo positions", "sim": "GLORYS park-advection positions"}[source]
    ax.set_title(f"Per-float sampling bias over life, {box} ({src}, deployed 27-28 Sep 2015)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_mean_field_months(boxes=("boxB", "boxA"), year=2015, months=(1, 4, 7, 10),
                           figsize=(14.5, 5.2)):
    """Fitted mean field at mid-month, both boxes on one map per month, shared scale."""
    from matplotlib.colors import Normalize
    names = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}

    mfs = {b: mean_field_months(b, year, months) for b in boxes}
    flat = np.concatenate([np.ravel(mfs[b].values) for b in boxes])
    lo, hi = (float(np.nanquantile(flat, q)) * J_TO_GJ for q in (0.01, 0.99))
    norm = Normalize(vmin=lo, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(months), figsize)
    for c, ax in enumerate(axes[0]):
        im = _draw_domain_map(ax, [mfs[b].isel(month=c) for b in boxes], norm,
                              "viridis", ext, f"{names.get(months[c], months[c])} {year}",
                              True, c == 0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="mean field (GJ/m2)")
    return fig


def plot_ohc_decomposition(boxes=("boxB", "boxA"), q=0.5, figsize=(13.5, 6.4),
                           clip=0.995):
    """OHC = mean field + residual on one shared day, both boxes on one map per field.

    The day is the combined-domain q-quantile of daily spatial spread, so both
    boxes are shown on the same date; OHC and mean field share one sequential
    scale (the residual's size is then readable against the OHC it came from),
    the residual its own diverging scale.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import Normalize, TwoSlopeNorm

    d = combined_quantile_days((q,), boxes).iloc[0]
    parts = {b: ohc_decomposition(b, int(d.year), int(d.doy)) for b in boxes}
    date = parts[boxes[0]][3]

    flat = np.concatenate([np.ravel(parts[b][i].values) for b in boxes for i in (0, 1)])
    lo, hi = (float(np.nanquantile(flat, x)) * J_TO_GJ for x in (0.005, 0.995))
    res = max(float(np.nanquantile(np.abs(parts[b][2].values), clip)) for b in boxes) * J_TO_GJ
    norms = [Normalize(lo, hi), Normalize(lo, hi),
             TwoSlopeNorm(vmin=-res, vcenter=0.0, vmax=res)]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    labels = ["OHC 0-2000 m (GJ/m2)", "mean field (GJ/m2)", "residual Z (GJ/m2)"]
    names = ["OHC (observed)", "fitted mean field", "residual Z = OHC - mean"]

    ext = _both_box_extent(boxes)
    pc = ccrs.PlateCarree()
    left, right, top, bottom, hgap = 0.04, 0.99, 0.90, 0.14, 0.03
    w = (right - left - 2 * hgap) / 3
    fig = plt.figure(figsize=figsize)
    for c in range(3):
        x = left + c * (w + hgap)
        ax = fig.add_axes([x, bottom, w, top - bottom], projection=pc)
        im = _draw_domain_map(ax, [parts[b][c] for b in boxes], norms[c], cmaps[c], ext,
                              f"{names[c]}\n{date:%Y-%m-%d}", True, c == 0, boxes=boxes)
        cax = fig.add_axes([x, bottom - 0.08, w, 0.02])
        fig.colorbar(im, cax=cax, orientation="horizontal", label=labels[c])
    return fig


def plot_ohc_timeseries(boxes=("boxA", "boxB"), figsize=(11, 6.5)):
    """Two stacked panels sharing the date axis: (top) domain-mean total OHC
    (solid) and fitted mean field (dashed) per box; (bottom) the domain-mean
    anomaly (total - mean field) per box. Daily over 2010-2019, GJ/m^2."""
    import matplotlib.pyplot as plt
    color = {"boxA": "tab:orange", "boxB": "tab:blue"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for b in boxes:
        dates, total, mhat = ohc_domain_series(b)
        c = color.get(b)
        a1.plot(dates, total * J_TO_GJ, color=c, lw=0.7, alpha=0.9, label=f"{b} total OHC")
        a1.plot(dates, mhat * J_TO_GJ, color=c, lw=1.4, ls="--", label=f"{b} mean field")
        a2.plot(dates, (total - mhat) * J_TO_GJ, color=c, lw=0.8, label=b)
    a1.set_ylabel("domain-mean OHC 0-2000 m (GJ/m2)")
    a1.legend(fontsize=8, ncol=2)
    a2.axhline(0.0, color="0.4", lw=1.0, ls=":")
    a2.set_ylabel("anomaly (GJ/m2)")
    a2.set_xlabel("date")
    a2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_quantile_days(boxes=("boxB", "boxA"), qs=(0.05, 0.5, 0.95),
                       figsize=(13.5, 5.8), clip=0.995):
    """Single-day residual fields at quantiles of the combined daily spread.

    Both boxes on one both-box base map per column, one shared diverging scale, so
    the western-boundary and subtropical fields contrast directly; the quantile
    days are chosen on the union-domain spatial dispersion.
    """
    from matplotlib.colors import TwoSlopeNorm

    sel = combined_quantile_days(qs, boxes)
    opened = {b: xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{b}.nc"))
              for b in boxes}
    try:
        cols = [[opened[b]["z"].sel(year=int(d.year)).isel(doy=int(d.doy)).load()
                 for b in boxes] for _, d in sel.iterrows()]
    finally:
        for ds in opened.values():
            ds.close()
    hi = max(float(np.nanquantile(np.abs(da.values), clip))
             for das in cols for da in das) * J_TO_GJ
    norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(qs), figsize)
    for c, (das, (_, d)) in enumerate(zip(cols, sel.iterrows())):
        im = _draw_domain_map(axes[0][c], das, norm, "RdBu_r", ext,
                              f"{int(d.q * 100)}th pctile day\n"
                              f"{int(d.year)} doy {int(d.doy) + 1}, "
                              f"sd {d.sd * J_TO_GJ:.2f} GJ/m2",
                              True, c == 0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="Z (GJ/m2)")
    return fig


def plot_sampling_weight_hist(box, tag="_pilot", years=None, figsize=None):
    """Distribution of the per-cell sampling weight E[w]/uniform, one panel per year.

    A histogram over native wet cells; the red line at 1 is the uniform weight.
    Concentration near 1 means near-uniform sampling; a right tail is over-sampled
    cells, mass below 1 under-sampled. All panels share bins set from the pooled
    99.5th percentile.
    """
    import matplotlib.pyplot as plt
    ratio = sampling_weight_ratio(box, tag)
    years = years or sorted(ratio)
    hi = np.max(np.concatenate([ratio[y] for y in years]))
    bins = np.linspace(0, hi, 40)
    fig, axes = plt.subplots(1, len(years), figsize=figsize or (2.4 * len(years) + 1.0, 3.0),
                             sharex=True, sharey=True)
    for ax, y in zip(np.atleast_1d(axes), years):
        r = ratio[y]
        ax.hist(r, bins=bins, density=True, color="0.4")
        ax.axvline(1.0, color="crimson", lw=1.6)
        ax.axvline(r.mean(), color="k", lw=1.2, ls=":")
        ax.set_title(str(y), fontsize=10)
        ax.set_xlabel("E[w] / uniform", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("density (native cells)")
    fig.tight_layout()
    return fig


def plot_sampling_weight_space_combined(boxes=("boxB", "boxA"), tag="_pilot",
                                        years=None, figsize=None):
    """Over-/under-sampling map log2(E[w]/uniform), both boxes on one both-box base
    map per year and one shared symmetric scale, so the two arrays' spatial
    preference contrasts directly. Native 1/12 deg grid; red over-, blue under-sampled.
    """
    from matplotlib.colors import TwoSlopeNorm

    data = {}
    for b in boxes:
        Ew, lat, lon, p = sampling_weight_space(b, tag)
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, np.nan)), np.nan)
        yrs = years or sorted(int(y) for y in Ew.index.get_level_values("year").unique())
        per = {}
        for y in yrs:
            w = Ew.xs(y, level="year")
            i = w.index.get_level_values("ilat").to_numpy()
            j = w.index.get_level_values("ilon").to_numpy()
            v = w.to_numpy()
            G = np.full(p.shape, np.nan)
            pos = v > 0
            G[i[pos], j[pos]] = np.log2(v[pos]) - logp[i[pos], j[pos]]
            per[y] = xr.DataArray(G, dims=("latitude", "longitude"),
                                  coords={"latitude": lat, "longitude": lon})
        data[b] = per
    years = years or sorted(next(iter(data.values())).keys())
    finite = np.concatenate([data[b][y].values[np.isfinite(data[b][y].values)]
                             for b in boxes for y in years])
    vmax = float(np.percentile(np.abs(finite), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ext = _both_box_extent(boxes)

    figsize = figsize or (3.0 * len(years) + 1.5, 4.6)
    fig, axes, caxes = _map_grid(1, len(years), figsize)
    for c, y in enumerate(years):
        im = _draw_domain_map(axes[0][c], [data[b][y] for b in boxes], norm, "RdBu_r",
                              ext, str(y), True, c == 0, scale=1.0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="log2( E[w] / uniform )")
    return fig


def plot_trajectory_divergence(div=None, bin_days=20, figsize=(9.5, 5.2)):
    """Sim-vs-real separation vs days since deployment: median + IQR, with the
    decorrelated-floats floor. Rising to that floor = park advection loses the float."""
    import matplotlib.pyplot as plt
    div = trajectory_divergence() if div is None else div
    b = (np.floor(div["elapsed_days"] / bin_days) * bin_days + bin_days / 2)
    g = div.assign(b=b).groupby("b")["sep_km"]
    med, q1, q3, n = g.median(), g.quantile(0.25), g.quantile(0.75), g.size()
    x = med.index.to_numpy()[n.to_numpy() >= 20]
    med, q1, q3 = med.loc[x], q1.loc[x], q3.loc[x]
    sat = _decorrelation_km(div)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, q1, q3, alpha=0.22, color="tab:blue", label="IQR")
    ax.plot(x, med, color="tab:blue", lw=1.8, label="median separation")
    ax.axhline(sat, ls="--", color="0.4",
               label=f"decorrelated floats (~{sat:.0f} km)")
    ax.set_xlabel("days since deployment")
    ax.set_ylabel("real vs GLORYS-park separation (km)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_trajectory_twins(pairs=((295, 6901117), (68, 4901217)), tracks=None,
                          pool=None, cleaned=None, boxes=("boxA", "boxB"),
                          figsize=(8.5, 7.2)):
    """Real Argo float vs its GLORYS-park sim twin for a few long-lived floats,
    to show the two tracks drift apart from a shared deployment. Each pair shares a
    colour; real is solid, sim twin dashed; the shared deployment is a black star,
    the real end a filled circle and the sim end an open square."""
    import matplotlib.pyplot as plt
    tr = tracks if tracks is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
    pl = pool if pool is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"), parse_dates=["date"])
    real = clean_argo() if cleaned is None else cleaned
    to_naive = lambda s: pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)
    real = real.assign(date=to_naive(real["date"]))
    pl = pl.assign(date=to_naive(pl["date"]))
    tr = tr.assign(time=pd.to_datetime(tr["time"], format="ISO8601").astype("datetime64[ns]"))
    realg = {int(f): g.sort_values("date") for f, g in real.groupby("float_id")}
    simg = {int(s): g.sort_values("time") for s, g in tr.groupby("seed_id")}
    dep = {int(r.seed_id): r for r in pl.itertuples()}

    fig, ax, pc, e = _domain_base(boxes, figsize)
    # (real, sim twin) colour per float: matched hue families so the pairing reads,
    # distinct shades so real vs twin stay apart where the tracks overlap.
    pair_colors = [("#08519c", "#41b6c4"), ("#6a1b9a", "#ec7fc3"),
                   ("#1b7837", "#a6d96a"), ("#8c510a", "#dfc27d")]
    for k, (sid, fid) in enumerate(pairs):
        cr, cs = pair_colors[k % len(pair_colors)]
        d = dep[int(sid)]
        rt = realg[int(fid)]; rt = rt[rt["date"] >= d.date]
        st = simg[int(sid)]
        yrs = (rt["date"].max() - d.date).days / 365.25
        ax.plot(rt["lon"], rt["lat"], "-", color=cr, lw=1.0, transform=pc, zorder=3,
                label=f"real {fid} ({yrs:.1f} yr)")
        ax.plot(st["lon"], st["lat"], "--", color=cs, lw=1.0, transform=pc,
                zorder=3, label=f"sim twin {fid}")
        ax.plot(rt["lon"].iloc[-1], rt["lat"].iloc[-1], "o", color=cr, ms=6,
                transform=pc, zorder=4)
        ax.plot(st["lon"].iloc[-1], st["lat"].iloc[-1], "s", mfc="none", mec=cs,
                mew=1.4, ms=7, transform=pc, zorder=4)
        ax.plot(d.lon, d.lat, "*", color="k", ms=13, mec="w", mew=0.6,
                transform=pc, zorder=5, label="deployment" if k == 0 else None)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    return fig


def plot_velocity_quantile_days(comp="uo", boxes=("boxB", "boxA"), qs=(0.05, 0.5, 0.95),
                                figsize=(13.5, 5.8), clip=0.995):
    """Single-day park-depth velocity at quantiles of the combined daily spread.

    Both boxes on one both-box base map per column, one shared diverging scale.
    """
    from matplotlib.colors import TwoSlopeNorm

    sel = combined_velocity_quantile_days(comp, qs, boxes)
    cols = {}
    for y, g in sel.groupby("year", sort=False):
        perbox = {b: _velocity_year(b, int(y))[comp] for b in boxes}
        for idx, dd in zip(g.index, g["doy"]):
            cols[idx] = [perbox[b].isel(time=int(dd)) * M_S_TO_CM_S for b in boxes]
    cols = [cols[i] for i in sel.index]
    hi = max(float(np.nanquantile(np.abs(da.values), clip))
             for das in cols for da in das)
    norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(qs), figsize)
    for c, (das, (_, s)) in enumerate(zip(cols, sel.iterrows())):
        im = _draw_domain_map(axes[0][c], das, norm, "RdBu_r", ext,
                              f"{int(s.q * 100)}th pctile day\n"
                              f"{int(s.year)} doy {int(s.doy) + 1}, "
                              f"sd {s.sd:.1f} cm/s", True, c == 0, scale=1.0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label=f"{VEL_COMPS[comp]} velocity at 1000 m (cm/s)")
    return fig


# ----------------------------------------------------------------------------
# the simple-kriging estimator
# ----------------------------------------------------------------------------


def krige_param_table(tag="oracle"):
    """Fitted covariance per box: median and inter-quartile range over cells.

    sd is sqrt(phi), comparable to the field's own sd_P(Z); the nugget is
    reported as the ratio tau^2, the absolute nugget variance being phi tau^2.
    """
    p = load_krige_params(tag)
    p = p.assign(sd=np.sqrt(p["phi"]) * J_TO_GJ)
    cols = ["sd", "th_lon", "th_lat", "th_t", "nugget_ratio"]
    g = p.groupby("box")[cols]
    out = g.median().round(4)
    iqr = (g.quantile(0.75) - g.quantile(0.25)).round(4)
    out.columns = ["sd (GJ/m2)", "th_lon (deg)", "th_lat (deg)", "th_t (d)", "nugget ratio"]
    for c, s_ in zip(out.columns, cols):
        out[c] = [f"{m:g} [{q:g}]" for m, q in zip(out[c], iqr[s_])]
    out.insert(0, "cells", g.size())
    return out.reset_index()


def krige_vs_empirical_table(tag="oracle", boxes=("boxB", "boxA")):
    """Fitted decorrelation scales against the field's own e-folding scales.

    For an exponential covariance the range IS the e-folding distance, so these
    should agree. They do not: the ratio column is the range inflation the
    kernel-shape mismatch produces.
    """
    p = load_krige_params(tag)
    rows = []
    for b in boxes:
        e = empirical_efolds(b)
        f = p[p["box"] == b][["th_lon", "th_lat", "th_t"]].median()
        for lab, emp, fit, unit in (("zonal", e["zonal"], f.th_lon, "deg"),
                                    ("meridional", e["merid"], f.th_lat, "deg"),
                                    ("temporal", e["time"], f.th_t, "d")):
            rows.append({"box": b, "scale": f"{lab} ({unit})", "empirical e-fold": emp,
                         "fitted range": fit, "ratio": fit / emp})
    return pd.DataFrame(rows)


def krige_fit_diagnostics(tag="oracle"):
    """Per-box fit quality: windows, convergence, seasonal spread, nugget at bound."""
    w = pd.read_csv(os.path.join(cfg.OUT_DIR, f"krige_local_{tag}.csv"))
    g = w.groupby("box")
    iqr = (w.groupby(["box", "lat", "lon"])[["th_lon", "th_t"]]
             .agg(lambda x: (x.quantile(.75) - x.quantile(.25)) / max(abs(x.median()), 1e-30))
             .groupby("box").median())
    return pd.DataFrame({
        "box": g.size().index, "windows": g.size().to_numpy(),
        "cells": g["lat"].apply(lambda x: len(x.unique())).to_numpy() * 0 +
                 w.groupby("box").apply(lambda d: len(d[["lat", "lon"]].drop_duplicates()),
                                        include_groups=False).to_numpy(),
        "converged": g["conv"].apply(lambda x: (x == 0).mean()).to_numpy(),
        "median iters": g["nit"].median().to_numpy(),
        "obs per window": g["n_obs"].median().to_numpy(),
        "seasonal IQR/med th_lon": iqr["th_lon"].to_numpy(),
        "seasonal IQR/med th_t": iqr["th_t"].to_numpy(),
        "nugget at bound": g["nugget_ratio"].apply(lambda x: (x < 1e-6).mean()).to_numpy(),
    })


def plot_krige_params(tag="oracle", figsize=(12, 6.4)):
    """The fitted covariance across the parameter grid, one panel per parameter."""
    p = load_krige_params(tag).assign(sd=lambda d: np.sqrt(d["phi"]) * J_TO_GJ)
    panels = [("sd", "sqrt(phi)  GJ per m2"), ("th_lon", "range lon  deg"),
              ("th_lat", "range lat  deg"), ("th_t", "range time  d")]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (col, lab) in zip(axes.ravel(), panels):
        s = ax.scatter(p["lon"], p["lat"], c=p[col], s=190, marker="s",
                       cmap="viridis", edgecolor="none")
        for b in ("boxA", "boxB"):
            (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
            ax.add_patch(plt.Rectangle((lo0, la0), lo1 - lo0, la1 - la0,
                                       fill=False, ec="w", lw=1.2))
        fig.colorbar(s, ax=ax, fraction=0.046)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def krige_bias_table(krige, box):
    """Per-year kriging mass, bias and its alignment / coverage split."""
    g = krige[krige["box"] == box].groupby("year")
    t = g.agg(m=("m", "mean"), mean_B=("B", "mean"), se_B=("B", "sem"),
              sd_B=("B", "std"), alignment=("B_align", "mean"),
              coverage=("B_cover", "mean"), zbar=("zbar", "mean")).reset_index()
    a = krige[krige["box"] == box]
    t.loc["all"] = ["all", a["m"].mean(), a["B"].mean(), a["B"].sem(), a["B"].std(),
                    a["B_align"].mean(), a["B_cover"].mean(), a["zbar"].mean()]
    for c in ("mean_B", "se_B", "sd_B", "alignment", "coverage", "zbar"):
        t[c] *= J_TO_GJ
    return t


def estimator_table(est, krige, box):
    """Simple mean against simple kriging, per year, standardised by sd_P(Z)."""
    sd = _sd_Z(box) * J_TO_GJ
    s = est[est["box"] == box].groupby("year")["B"].agg(["mean", "sem", "std"]) * J_TO_GJ
    k = krige[krige["box"] == box].groupby("year")["B"].agg(["mean", "sem", "std"]) * J_TO_GJ
    m = krige[krige["box"] == box].groupby("year")["m"].mean()
    return pd.DataFrame({
        "year": s.index, "sd_k": sd.reindex(s.index).to_numpy(),
        "mean B": s["mean"].to_numpy(), "mean se": s["sem"].to_numpy(),
        "mean sd": s["std"].to_numpy(),
        "krige B": k["mean"].to_numpy(), "krige se": k["sem"].to_numpy(),
        "krige sd": k["std"].to_numpy(), "krige m": m.to_numpy(),
        "mean B / sd": s["mean"].to_numpy() / sd.reindex(s.index).to_numpy(),
        "krige B / sd": k["mean"].to_numpy() / sd.reindex(s.index).to_numpy(),
    })


def union_table(est, krige):
    """Bias over A u B for both estimators, with the simple mean's allocation term."""
    u = union_bias(est, krige)
    g = u.groupby("year")
    t = g.agg(**{"mean B": ("mean_B", "mean"), "mean se": ("mean_B", "sem"),
                 "within": ("mean_within", "mean"), "allocation": ("mean_alloc", "mean"),
                 "krige B": ("krige_B", "mean"), "krige se": ("krige_B", "sem"),
                 "alignment": ("krige_B_align", "mean"), "coverage": ("krige_B_cover", "mean"),
                 "krige m": ("krige_m", "mean")}).reset_index()
    t.loc["all"] = ["all", u["mean_B"].mean(), u["mean_B"].sem(), u["mean_within"].mean(),
                    u["mean_alloc"].mean(), u["krige_B"].mean(), u["krige_B"].sem(),
                    u["krige_B_align"].mean(), u["krige_B_cover"].mean(), u["krige_m"].mean()]
    for c in t.columns.drop(["year", "krige m"]):
        t[c] *= J_TO_GJ
    return t


def plot_krige_decomposition(krige, boxes=("boxB", "boxA"), figsize=(11, 3.8)):
    """Alignment and coverage contributions to E[B], per box-year."""
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize, sharex=True)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        g = krige[krige["box"] == b].groupby("year")
        yr = np.array(sorted(krige["year"].unique()))
        al = g["B_align"].mean().to_numpy() * J_TO_GJ
        co = g["B_cover"].mean().to_numpy() * J_TO_GJ
        se = g["B"].sem().to_numpy() * J_TO_GJ
        ax.bar(yr - 0.18, al, 0.36, label="alignment", color="#4477aa")
        ax.bar(yr + 0.18, co, 0.36, label="coverage", color="#cc6677")
        ax.errorbar(yr, al + co, yerr=1.96 * se, fmt="ko", ms=4, lw=1, capsize=3, label="E[B]")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(b, fontsize=10); ax.set_xlabel("year"); ax.set_xticks(yr)
    np.atleast_1d(axes)[0].set_ylabel("GJ per m2")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_estimator_comparison(est, krige, boxes=("boxB", "boxA"), figsize=(11, 4.0)):
    """E[B] with 95 percent CI for the simple mean and simple kriging, per box-year."""
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize, sharex=True)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        yr = np.array(sorted(krige["year"].unique()))
        for src, off, col, lab in ((est, -0.15, "0.45", "simple mean"),
                                   (krige, 0.15, "#117733", "simple kriging")):
            g = src[src["box"] == b].groupby("year")["B"]
            ax.errorbar(yr + off, g.mean() * J_TO_GJ, yerr=1.96 * g.sem() * J_TO_GJ,
                        fmt="o", ms=5, lw=1.4, capsize=4, color=col, label=lab)
        ax.axhline(0, color="r", lw=0.8)
        ax.set_title(b, fontsize=10); ax.set_xlabel("year"); ax.set_xticks(yr)
    np.atleast_1d(axes)[0].set_ylabel("E[B]  (GJ per m2)")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# inverse-intensity / doubly robust estimators (reports/ipw.qmd)
# ----------------------------------------------------------------------------

IPW_DIR = os.path.join(cfg.OUT_DIR, "ipw_testbed")
Q_PER_1E4KM2_YR = 1e10 * 365.25   # profiles per m2 per day -> per 10^4 km2 per year


def load_testbed(label="sim_k0.25_7.5"):
    """Per-replicate testbed rows (ipw.testbed output) for one setting, with a rep column."""
    import glob
    files = sorted(glob.glob(os.path.join(IPW_DIR, f"testbed_{label}_r*_box*_*.csv")))
    if not files:
        return None
    t = pd.concat(map(pd.read_csv, files), ignore_index=True)
    return t.assign(rep=t["positions"].str.extract(r"_r(\d+)$")[0].astype(int))


def bias_summary(df, est, box, truth="zbar"):
    """Per-year mean bias, MC se, t and single-replicate RMSE (GJ/m2) per estimator,
    plus an all-years row computed on per-replicate year averages over replicates with
    every year present (replicates are the independent units; years share floats)."""
    d = df[df["box"] == box]
    b = d[["rep", "year"]].assign(**{c: (d[c] - d[truth]) * J_TO_GJ for c in est})
    stats = lambda g: pd.concat({c: pd.Series({"mean": g[c].mean(), "se": g[c].sem(),
                                               "rmse": np.sqrt((g[c] ** 2).mean())})
                                 for c in est})
    t = b.groupby("year").apply(stats)
    full = b.groupby("rep").filter(lambda g: g["year"].nunique() == b["year"].nunique())
    t.loc["all"] = stats(full.groupby("rep")[est].mean())
    for c in est:
        t[(c, "t")] = t[(c, "mean")] / t[(c, "se")]
    t = t[[(c, s) for c in est for s in ("mean", "se", "t", "rmse")]]
    t.columns = [f"{c} {s}" for c, s in t.columns]
    reps = b.groupby("year")["rep"].nunique()
    t.insert(0, "reps", reps.reindex(t.index).fillna(full["rep"].nunique()).astype(int))
    return t.reset_index().assign(box=box)


BOX_LABELS = {"boxA": "Box A", "boxB": "Box B", "boxAB": "Box A u B"}


def bias_removed_table(df, box, est=("ht", "hajek"), base="naive", truth="truth", n_boot=2000, seed=0):
    """Per year (+ "all" on per-replicate year means): base bias and t, and for each estimator
    its bias, the bias removed sign(E[B_base]) (E[B_base] - E[B_est]) (positive = toward zero)
    with paired se, and the percent removed
    100 (1 - E[B_est]/E[B_base]) with a replicate-bootstrap 95% CI, blank unless |t_base| > 2."""
    d = df[df["box"] == box]
    b = d[["rep", "year"]].assign(**{c: (d[c] - d[truth]) * J_TO_GJ for c in (base, *est)})
    full = b.groupby("rep").filter(lambda g: g["year"].nunique() == b["year"].nunique())
    groups = list(b.groupby("year")) + [("all", full.groupby("rep")[[base, *est]].mean())]
    rng = np.random.default_rng(seed)
    rows = []
    for y, g in groups:
        x0 = g[base].to_numpy()
        n = len(x0)
        t0 = x0.mean() / (x0.std(ddof=1) / np.sqrt(n))
        idx = rng.integers(0, n, (n_boot, n))
        row = {"year": y, "reps": n, f"{base} bias": x0.mean(), f"{base} t": t0}
        for c in est:
            x = g[c].to_numpy()
            dd = np.sign(x0.mean()) * (x0 - x)
            pct = 100 * (1 - x[idx].mean(1) / x0[idx].mean(1))
            ok = abs(t0) > 2
            row |= {f"{c} bias": x.mean(), f"{c} removed": dd.mean(), f"{c} removed se": dd.std(ddof=1) / np.sqrt(n),
                    f"{c} % removed": 100 * (1 - x.mean() / x0.mean()) if ok else np.nan,
                    f"{c} % lo": np.percentile(pct, 2.5) if ok else np.nan,
                    f"{c} % hi": np.percentile(pct, 97.5) if ok else np.nan}
        rows.append(row)
    return pd.DataFrame(rows).assign(box=BOX_LABELS.get(box, box))


EST_STYLE = {"naive": ("Simple mean", "#2a78d6", "o"), "ht": ("Horvitz-Thompson", "#eb6834", "s"),
             "hajek": ("Hajek", "#1baf7a", "^")}


def _year_axis(ax, years):
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.axvline(len(years) - 1.5, color="0.8", lw=0.8)


def plot_mean_bias(df, est=("naive", "ht", "hajek"), boxes=("boxA", "boxB", "boxAB"), truth="truth",
                   figsize=None):
    """Mean realized bias (GJ/m2) +/- 2 se over replicates per year and pooled, one panel per box."""
    figsize = figsize or (3.4 * len(boxes), 3.2)
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize, sharey=True, squeeze=False)
    for ax, b in zip(axes[0], boxes):
        t = bias_summary(df, list(est), b, truth=truth)
        years = t["year"].tolist()
        for k, e in enumerate(est):
            lab, col, mk = EST_STYLE[e]
            x = np.arange(len(years)) + (k - (len(est) - 1) / 2) * 0.22
            ax.errorbar(x, t[f"{e} mean"], yerr=2 * t[f"{e} se"], fmt=mk, color=col, ms=6, lw=1.5,
                        capsize=0, label=lab)
        ax.axhline(0, color="0.3", lw=0.8)
        _year_axis(ax, years)
        ax.set_title(BOX_LABELS.get(b, b), fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("mean bias (GJ per m2)")
    axes[0][-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_bias_removed(df, est=("ht", "hajek"), boxes=("boxA", "boxB", "boxAB"), figsize=None):
    """Percent of the simple-mean bias removed, with replicate-bootstrap 95% CI; years where
    the simple-mean bias is not resolved (|t| <= 2) are marked n.r."""
    figsize = figsize or (3.4 * len(boxes), 3.2)
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize, sharey=True, squeeze=False)
    w = 0.36
    for ax, b in zip(axes[0], boxes):
        t = bias_removed_table(df, b, est)
        years = t["year"].tolist()
        for k, e in enumerate(est):
            lab, col, _ = EST_STYLE[e]
            x = np.arange(len(years)) + (k - (len(est) - 1) / 2) * (w + 0.03)
            v = t[f"{e} % removed"].to_numpy()
            ax.bar(x, v, width=w, color=col, label=lab, edgecolor="white", lw=1)
            ax.errorbar(x, v, yerr=[v - t[f"{e} % lo"], t[f"{e} % hi"] - v], fmt="none", ecolor="0.25", lw=1)
        for i, v in enumerate(t[f"{est[0]} % removed"]):
            if np.isnan(v):
                ax.text(i, 5, "n.r.", ha="center", fontsize=8, color="0.4")
        ax.axhline(100, color="0.3", lw=0.8, ls="--")
        ax.axhline(0, color="0.3", lw=0.8)
        _year_axis(ax, years)
        ax.set_title(BOX_LABELS.get(b, b), fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("bias removed (percent)")
    axes[0][-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def aipw_decomposition_table(df, box):
    """AIPW = fold-mean integral term + fold-mean Hajek residual correction; both as
    deviations from truth where meaningful (GJ/m2), with the KS map and its mass."""
    d = df[df["box"] == box].copy()
    mean_of = lambda s: s.astype(str).str.split(";").apply(lambda x: np.mean(np.asarray(x, float)))
    d = d.assign(integral=(mean_of(d["fold_int"]) - d["zbar"]) * J_TO_GJ,
                 correction=mean_of(d["fold_cor"]) * J_TO_GJ,
                 aipw_B=(d["aipw"] - d["zbar"]) * J_TO_GJ, ks_B=(d["ks"] - d["zbar"]) * J_TO_GJ,
                 fold_sd=d["aipw_fold_sd"] * J_TO_GJ)
    cols = ["ks_B", "ks_mass", "integral", "correction", "aipw_B", "fold_sd"]
    t = d.groupby("year")[cols].mean()
    t.loc["all"] = d.groupby("rep")[cols].mean().mean()
    t.insert(0, "reps", d.groupby("year")["rep"].nunique().reindex(t.index).fillna(d["rep"].nunique()))
    return t.reset_index()


def plot_qhat_window(qds, t0, t1, kind="nominal", boxes=("boxB", "boxA"), figsize=(6.2, 6.0), points=None,
                     title=None):
    """Kernel q-hat averaged over [t0, t1) on its fine grid, both boxes on one map, with
    optional (lat, lon) points overlaid. Support area is time-invariant and time bins are
    equal, so the plain time mean is the mu-weighted mean."""
    from matplotlib.colors import LogNorm
    q = qds[f"q_{kind}"].sel(time=slice(np.datetime64(t0), np.datetime64(t1) - np.timedelta64(1, "s")))
    da = (q.mean("time") * Q_PER_1E4KM2_YR).rename(lat="latitude", lon="longitude")
    v = da.values[np.isfinite(da.values) & (da.values > 0)]
    norm = LogNorm(*np.percentile(v, [1, 99.5]))
    fig, axes, caxes = _map_grid(1, 1, figsize)
    im = _draw_domain_map(axes[0][0], [da], norm, "viridis", _both_box_extent(boxes, pad=1.5),
                          title or f"{t0} to {t1}", True, True, scale=1.0, boxes=boxes)
    if points is not None:
        import cartopy.crs as ccrs
        axes[0][0].plot(points[1], points[0], "o", ms=2.5, mfc="white", mec="0.1", mew=0.4,
                        transform=ccrs.PlateCarree(), zorder=5)
    fig.colorbar(im, cax=caxes[0], label="q-hat, profiles per 10^4 km2 per year")
    return fig


def plot_replicate_bias(df, est, labels=None, truth="zbar", boxes=("boxB", "boxA"), figsize=None):
    """Replicate distribution of estimate minus truth (GJ/m2) per box-year and estimator."""
    labels = labels or est
    years = sorted(df["year"].unique())
    figsize = figsize or (2.3 * len(years) + 1, 3.0 * len(boxes))
    fig, axes = plt.subplots(len(boxes), len(years), figsize=figsize, sharey="row", squeeze=False)
    cols = plt.get_cmap("tab10")
    for r, b in enumerate(boxes):
        for c, y in enumerate(years):
            ax = axes[r][c]
            d = df[(df["box"] == b) & (df["year"] == y)]
            data = [((d[e] - d[truth]) * J_TO_GJ).dropna().to_numpy() for e in est]
            bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
            for k, patch in enumerate(bp["boxes"]):
                patch.set(facecolor=cols(k), alpha=0.35)
            for k, x in enumerate(data):
                ax.plot(np.full(len(x), k + 1) + np.random.default_rng(k).uniform(-0.15, 0.15, len(x)),
                        x, ".", ms=2.5, color=cols(k), alpha=0.6)
                ax.plot(k + 1, x.mean() if len(x) else np.nan, "kD", ms=4)
            ax.axhline(0, color="r", lw=0.8)
            ax.set_xticks(range(1, len(est) + 1), labels, rotation=45, fontsize=7)
            ax.set_title(f"{b} {y}", fontsize=9)
        axes[r][0].set_ylabel("estimate - truth (GJ per m2)")
    fig.tight_layout()
    return fig


def extrap_bias(reps):
    """Mean over test replicates of the Hajek bias per (R, bandwidth, box, year), with MC se."""
    return (reps.groupby(["R", "sigma_deg", "sigma_days", "box", "year"])["bias"]
            .agg(["mean", "sem"]).reset_index())


def plot_bias_vs_bandwidth(reps, boxes=("boxB", "boxA"), figsize=None):
    """Mean over box-years of |mean Hajek bias| against spatial bandwidth, one line per R,
    one column per temporal bandwidth; dashed grey is the mean MC se, the level
    |mean bias| cannot resolve below."""
    m = extrap_bias(reps)
    days = sorted(m["sigma_days"].unique())
    Rs = sorted(m["R"].unique())
    figsize = figsize or (3.6 * len(days), 3.0 * len(boxes))
    fig, axes = plt.subplots(len(boxes), len(days), figsize=figsize, sharey="row", squeeze=False)
    cmap = plt.get_cmap("viridis", len(Rs))
    for r, b in enumerate(boxes):
        for c, st in enumerate(days):
            ax = axes[r][c]
            g = m[(m["box"] == b) & (m["sigma_days"] == st)]
            for k, R in enumerate(Rs):
                s = g[g["R"] == R].groupby("sigma_deg")["mean"].apply(lambda x: x.abs().mean())
                ax.plot(s.index, s.to_numpy(), "o-", color=cmap(k), ms=4, label=f"R = {R}")
            se = g[g["R"] == Rs[-1]].groupby("sigma_deg")["sem"].mean()
            ax.plot(se.index, se.to_numpy(), "--", color="0.5", lw=1, label="MC se")
            ax.set_title(f"{b}, sigma_t = {st:g} d", fontsize=9)
            ax.set_xlabel("sigma_s (deg)")
        axes[r][0].set_ylabel("mean |bias| (GJ per m2)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def richardson_table(reps, pairs=(((0.125, 3.75), (0.25, 7.5)), ((0.1875, 3.75), (0.375, 7.5)),
                                  ((0.25, 7.5), (0.5, 15.0)), ((0.1875, 7.5), (0.375, 15.0)))):
    """Two-bandwidth Richardson Hajek (4 H_h - H_2h)/3 against H_h alone: per R, box and
    pair, mean over years of |mean bias| and of the single-replicate sd (GJ/m2)."""
    w = reps.pivot_table(index=["R", "rep", "box", "year"], columns=["sigma_deg", "sigma_days"],
                         values="bias")
    rows = []
    for h, h2 in pairs:
        if h not in w or h2 not in w:
            continue
        for lab, b in (("H_h", w[h]), ("Richardson", (4 * w[h] - w[h2]) / 3)):
            g = b.groupby(["R", "box", "year"])
            s = pd.DataFrame({"abs_mean": g.mean().abs(), "sd": g.std()}).groupby(["R", "box"]).mean()
            rows.append(s.assign(pair=f"{h[0]:g}/{h[1]:g} & {h2[0]:g}/{h2[1]:g}", estimator=lab))
    t = pd.concat(rows).reset_index()
    return t.pivot_table(index=["pair", "R"], columns=["box", "estimator"],
                         values=["abs_mean", "sd"]).round(3)
