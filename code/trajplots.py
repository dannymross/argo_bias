"""Plotting helpers for Argo trajectory output.

Separate from ``trajsim.py`` so the simulation library has no matplotlib /
cartopy dependency on HPC compute nodes. Import this only for local analysis.
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def open_trajectories(output):
    """Open one zarr path, or concat a glob of batch outputs on trajectory.

    Each batch zarr numbers its floats 0..n-1 locally, so concatenating leaves
    duplicate ``trajectory`` labels across batches. We reassign a unique
    trajectory index after concat, otherwise any per-float groupby would merge
    floats that share a local id across batches.
    """
    paths = (
        sorted(glob.glob(output))
        if isinstance(output, str) and "*" in output
        else [output]
    )
    if len(paths) == 1:
        return xr.open_zarr(paths[0])
    ds = xr.concat([xr.open_zarr(p) for p in paths], dim="trajectory")
    return ds.assign_coords(trajectory=np.arange(ds.sizes["trajectory"]))


def _zarr_stem(output):
    """Return a filename stem from a zarr path (directory or file)."""
    return os.path.splitext(os.path.basename(str(output).rstrip("/")))[0]


def _deployed_color_values(lat_vals, lon_vals, color_by):
    """Per-float deployed lat/lon used to colour tracks (None disables colouring).

    ``lat_vals``/``lon_vals`` are (trajectory, obs); the deployed position is the
    first observation of each float.
    """
    if color_by == "lat":
        return lat_vals[:, 0]
    if color_by == "lon":
        return lon_vals[:, 0]
    if color_by is None:
        return None
    raise ValueError(f"color_by must be 'lat', 'lon', or None (got {color_by!r})")


def _color_by_label(color_by):
    return "deployed latitude (°N)" if color_by == "lat" else "deployed longitude (°E)"


def _discrete_color(cvals, cmap):
    """Discrete BoundaryNorm + colormap keyed to the unique values in ``cvals``.

    With a cell-aligned deployment there are only a handful of distinct launch
    latitudes/longitudes, so each gets its own colour band. Returns
    ``(norm, discrete_cmap, ticks)`` where ``ticks`` are the unique values.
    """
    import matplotlib.colors as mcolors

    vals = np.round(np.asarray(cvals, dtype=float), 3)
    uniq = np.unique(vals[np.isfinite(vals)])
    n = len(uniq)
    cmap_d = plt.get_cmap(cmap, n)
    if n == 1:
        bounds = np.array([uniq[0] - 0.5, uniq[0] + 0.5])
    else:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        lo = uniq[0] - (uniq[1] - uniq[0]) / 2.0
        hi = uniq[-1] + (uniq[-1] - uniq[-2]) / 2.0
        bounds = np.concatenate([[lo], mids, [hi]])
    return mcolors.BoundaryNorm(bounds, n), cmap_d, uniq


def _auto_extent(lat_vals, lon_vals, margin=3):
    """Compute [lon_min, lon_max, lat_min, lat_max] with a margin.

    NaN-aware: deleted floats leave NaN positions in the trajectory arrays.
    """
    return [
        np.nanmin(lon_vals) - margin,
        np.nanmax(lon_vals) + margin,
        np.nanmin(lat_vals) - margin,
        np.nanmax(lat_vals) + margin,
    ]


def _add_boxes(ax, boxes, lw=1, alpha=1, zorder=4):
    for box in boxes or []:
        la0, la1, lo0, lo1, *opts = box

        label = opts[0] if len(opts) > 0 else None
        color = opts[1] if len(opts) > 1 else "k"
        ls = opts[2] if len(opts) > 2 else "--"

        ax.plot(
            [lo0, lo1, lo1, lo0, lo0],
            [la0, la0, la1, la1, la0],
            lw=lw,
            ls=ls,
            color=color,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=zorder,
            label=label,
        )


def map_trajectories(
    output,
    lon="lon",
    lat="lat",
    figsize=None,
    linewidth=1.0,
    title=None,
    extent=None,
    margin=3,
    save_path=None,
    dpi=300,
    show=True,
):
    """Plot float trajectories on a map, one colour per float.

    Parameters
    ----------
    output : str
        Path to a zarr directory, or a glob matching multiple batch zarrs.
    extent : list of [lon_min, lon_max, lat_min, lat_max], optional
        Map extent. Auto-computed from the data + margin if not given.
    margin : float
        Degrees of padding added around the data when auto-computing extent.
    save_path : str, optional
        Directory to save the figure (as <stem>.png). Not saved if None.
    """
    ds = open_trajectories(output)
    stem = _zarr_stem(output)

    lat_vals = ds[lat].values.astype(float)  # (trajectory, obs)
    lon_vals = ds[lon].values.astype(float)
    n_traj = lat_vals.shape[0]

    # Auto title from date range
    if title is None:
        t_start = np.datetime_as_string(ds.time.values[:, 0].min(), unit="D")
        t_end = np.datetime_as_string(ds.time.values[:, -1].max(), unit="D")
        title = f"Argo float trajectories  |  {n_traj} floats  |  {t_start} → {t_end}"

    colors = cm.tab10(np.linspace(0, 1, max(n_traj, 1)))

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#d5e9f5")
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d0", edgecolor="grey", linewidth=0.4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgrey")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey", linestyle=":")

    for i in range(n_traj):
        lons_i = lon_vals[i]
        lats_i = lat_vals[i]
        color = colors[i]

        # Trajectory line
        ax.plot(
            lons_i,
            lats_i,
            linewidth=linewidth,
            color=color,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Deployment marker (circle)
        ax.scatter(
            lons_i[0],
            lats_i[0],
            s=40,
            color=color,
            marker="o",
            edgecolors="k",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=5,
            label=f"Float {i + 1}",
        )

        # End marker (star)
        ax.scatter(
            lons_i[-1],
            lats_i[-1],
            s=60,
            color=color,
            marker="*",
            edgecolors="k",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    map_extent = (
        extent if extent is not None else _auto_extent(lat_vals, lon_vals, margin)
    )
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        title="○ start  ★ end",
        title_fontsize=7,
    )
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()

    if save_path is not None:
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)
        fig_file = os.path.join(save_path, f"{stem}.png")
        fig.savefig(fig_file, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure -> {fig_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def map_trajectories_minimal(
    output,
    extent=None,
    margin=1.5,
    boxes=None,
    line_color="#1a4f8a",
    color_by="lat",
    cmap="viridis",
    figsize=None,
    save_path=None,
    dpi=200,
    title=None,
    show=True,
):
    """Minimal map of many float trajectories.

    Unlike :func:`map_trajectories` (one colour + legend entry per float, made
    for a handful of floats), this draws all tracks as thin translucent lines
    with deployment dots -- readable for a 100-float pilot. Optional ``boxes``
    (list of ``(lat_min, lat_max, lon_min, lon_max, label)``) outline e.g. the
    truth domain and the deployment region.

    ``color_by`` colours each float's track and deployment dot by its **deployed
    position** -- ``"lat"`` (default) or ``"lon"`` -- so floats launched at the
    same latitude/longitude share a colour and can be tracked as they disperse.
    Set ``color_by=None`` for a single ``line_color``.
    """
    ds = open_trajectories(output)
    lat_vals = ds["lat"].values.astype(float)
    lon_vals = ds["lon"].values.astype(float)
    n_traj = lat_vals.shape[0]

    cvals = _deployed_color_values(lat_vals, lon_vals, color_by)
    norm = cmap_obj = ticks = None
    if cvals is not None:
        norm, cmap_obj, ticks = _discrete_color(cvals, cmap)  # discrete by launch band

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#ededed", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey", zorder=2)

    for i in range(n_traj):
        c = cmap_obj(norm(cvals[i])) if cvals is not None else line_color
        ax.plot(
            lon_vals[i],
            lat_vals[i],
            lw=0.5,
            color=c,
            alpha=0.45,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
    ax.scatter(
        lon_vals[:, 0],
        lat_vals[:, 0],
        c=cvals if cvals is not None else "#e84040",
        cmap=cmap_obj if cvals is not None else None,
        norm=norm,
        s=12,
        zorder=5,
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        label=None if cvals is not None else "deployment",
    )
    if cvals is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, ticks=ticks)
        cb.set_label(_color_by_label(color_by))

    _add_boxes(ax, boxes)

    map_extent = (
        extent if extent is not None else _auto_extent(lat_vals, lon_vals, margin)
    )
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="grey", alpha=0.4, linestyle=":"
    )
    gl.top_labels = gl.right_labels = False

    if title is None:
        t0 = np.datetime_as_string(np.nanmin(ds.time.values), unit="D")
        t1 = np.datetime_as_string(np.nanmax(ds.time.values), unit="D")
        title = f"{n_traj} float trajectories  |  {t0} → {t1}"
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path), dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def map_point_trajectories(
    df,
    lat_col="lat",
    lon_col="lon",
    id_col="float_id",
    date_col="date",
    extent=None,
    margin=1.5,
    boxes=None,
    cmap="tab10",
    figsize=None,
    save_path=None,
    dpi=200,
    title=None,
    show=True,
):
    """Map float tracks from a LONG-format DataFrame (one row per observation).

    Unlike :func:`map_trajectories_minimal` (which reads a Parcels/VirtualFleet
    trajectory zarr), this plots real Argo-style point observations: it groups
    ``df`` by ``id_col``, sorts each float by ``date_col``, and draws its track
    as a lon/lat line with a start dot, one colour per float. Pass the same
    ``extent``/``boxes`` as the simulated-trajectory map for comparison.
    """
    floats = list(df.groupby(id_col))
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(floats), 1)))

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#ededed", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey", zorder=2)

    all_lat, all_lon = [], []
    for (fid, g), color in zip(floats, colors):
        g = g.sort_values(date_col)
        lon, lat = g[lon_col].to_numpy(), g[lat_col].to_numpy()
        all_lat.append(lat)
        all_lon.append(lon)
        ax.plot(
            lon,
            lat,
            lw=0.9,
            color=color,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.scatter(
            lon[0],
            lat[0],
            s=20,
            color=color,
            edgecolors="k",
            linewidths=0.4,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    _add_boxes(ax, boxes)

    lat_arr = np.concatenate(all_lat) if all_lat else np.array([np.nan])
    lon_arr = np.concatenate(all_lon) if all_lon else np.array([np.nan])
    map_extent = (
        extent if extent is not None else _auto_extent(lat_arr, lon_arr, margin)
    )
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="grey", alpha=0.4, linestyle=":"
    )
    gl.top_labels = gl.right_labels = False

    ax.set_title(title or f"{len(floats)} float tracks", fontsize=11, pad=8)
    if any(len(b) > 4 for b in (boxes or [])):
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path), dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def map_field(
    da,
    extent=None,
    margin=1.5,
    boxes=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
    title=None,
    lon_name="longitude",
    lat_name="latitude",
    figsize=None,
    save_path=None,
    dpi=200,
    show=True,
    ax=None,
):
    """Plot a 2-D lat/lon field (e.g. an annual-mean OHC map) on a cartopy map.

    ``da`` is a 2-D xarray DataArray with ``lat_name``/``lon_name`` coordinates.
    Coastline and ``boxes`` (``(lat_min, lat_max, lon_min, lon_max, label)``) are
    drawn on top; pass the same ``extent``/``boxes`` as the trajectory maps to
    compare. Land is drawn over the field, so only ocean values show. Pass an
    existing cartopy ``ax`` (with a PlateCarree projection) to draw into a panel
    of a larger figure (e.g. via :func:`map_fields_row`).
    """
    lons = da[lon_name].values
    lats = da[lat_name].values

    owns_fig = ax is None
    if owns_fig:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        lons,
        lats,
        da.values,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.add_feature(cfeature.LAND, facecolor="#ededed", edgecolor="none", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey", zorder=3)

    _add_boxes(ax, boxes)

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label)

    map_extent = extent if extent is not None else _auto_extent(lats, lons, margin)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="grey", alpha=0.4, linestyle=":"
    )
    gl.top_labels = gl.right_labels = False

    if title:
        ax.set_title(title, fontsize=11, pad=8)
    if any(len(b) > 4 for b in (boxes or [])):
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    if owns_fig:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(os.path.expanduser(save_path), dpi=dpi, bbox_inches="tight")
            print(f"saved {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig, ax


def map_fields_row(
    das,
    titles=None,
    cmaps=None,
    cbar_labels=None,
    vmin=None,
    vmax=None,
    extent=None,
    margin=1.5,
    boxes=None,
    figsize=None,
    suptitle=None,
    save_path=None,
    dpi=200,
    show=True,
):
    """Plot several 2-D fields side by side (one cartopy panel each).

    Reuses :func:`map_field` per panel. ``titles``/``cmaps``/``cbar_labels`` are
    per-field lists; ``vmin``/``vmax`` (and ``extent``/``boxes``) are shared.
    """
    n = len(das)
    if figsize is None:
        # Height follows quarto's fig-height (rcParams); width scales with
        # panel count at the same per-panel aspect ratio as before (6.5:6).
        height = plt.rcParams["figure.figsize"][1]
        figsize = (height * (6.5 / 6) * n, height)
    fig, axes = plt.subplots(
        1, n, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = np.atleast_1d(axes).ravel()
    for i, da in enumerate(das):
        map_field(
            da,
            extent=extent,
            margin=margin,
            boxes=boxes,
            cmap=(cmaps[i] if cmaps else "viridis"),
            vmin=vmin,
            vmax=vmax,
            cbar_label=(cbar_labels[i] if cbar_labels else None),
            title=(titles[i] if titles else None),
            ax=axes[i],
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path), dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def map_positions_by_month(
    output,
    extent=None,
    margin=1.5,
    boxes=None,
    ncols=4,
    color="#1a4f8a",
    color_by="lat",
    cmap="viridis",
    save_path=None,
    dpi=150,
    title=None,
):
    """Facet of monthly float positions: one snapshot dot per float per month.

    Complements :func:`map_trajectories_minimal` -- instead of the full tangle of
    tracks, each panel shows where the floats are in a given month (each float's
    last position that month), so the cloud's month-over-month migration and
    thinning (as floats exit the domain) are easy to read. ``boxes`` outlines
    (e.g. deployment / advection domain) are drawn on every panel; the panel
    title notes the float count still present that month.

    ``color_by`` colours each dot by its float's **deployed position** --
    ``"lat"`` (default) or ``"lon"`` -- with a shared colour scale across panels,
    so a launch latitude/longitude band can be followed month to month. Set
    ``color_by=None`` for a single ``color``.
    """
    import pandas as pd

    ds = open_trajectories(output)
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds["time"].values
    ntraj, nobs = lat.shape

    df = pd.DataFrame(
        {
            "float_id": np.repeat(np.arange(ntraj), nobs),
            "lat": lat.ravel(),
            "lon": lon.ravel(),
            "date": pd.to_datetime(time.ravel()),
        }
    ).dropna(subset=["lat", "lon", "date"])
    df["month"] = df["date"].dt.to_period("M")
    # one snapshot per float per month = its last recorded position that month
    snap = df.loc[df.groupby(["float_id", "month"])["date"].idxmax()].copy()
    months = sorted(snap["month"].unique())

    cvals = _deployed_color_values(lat, lon, color_by)
    norm = cmap_obj = ticks = None
    if cvals is not None:
        snap["cval"] = cvals[snap["float_id"].values]
        norm, cmap_obj, ticks = _discrete_color(cvals, cmap)  # discrete by launch band

    if extent is None:
        extent = _auto_extent(lat, lon, margin)
    nrows = int(np.ceil(len(months) / ncols))
    # Width follows quarto's fig-width (rcParams); height keeps panels square.
    width = plt.rcParams["figure.figsize"][0]
    fig = plt.figure(figsize=(width, width / ncols * nrows))
    for k, m in enumerate(months):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#ededed", edgecolor="none", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="grey", zorder=2)
        d = snap[snap["month"] == m]
        _add_boxes(ax, boxes, lw=0.8, alpha=0.6, zorder=3)
        if cvals is not None:
            ax.scatter(
                d["lon"],
                d["lat"],
                c=d["cval"],
                cmap=cmap_obj,
                norm=norm,
                s=7,
                alpha=0.85,
                edgecolors="none",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        else:
            ax.scatter(
                d["lon"],
                d["lat"],
                s=7,
                color=color,
                alpha=0.75,
                edgecolors="none",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_title(f"{m}  (n={len(d)})", fontsize=8)
        ax.gridlines(draw_labels=False, linewidth=0.2, color="grey", alpha=0.3)
    if title:
        fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    if cvals is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cb = fig.colorbar(sm, ax=fig.get_axes(), shrink=0.5, pad=0.02, ticks=ticks)
        cb.set_label(_color_by_label(color_by))
    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path), dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    return fig


PHASE_LABELS = {
    0: "init descend",
    1: "drift",
    2: "profile descend",
    3: "profile ascend",
    4: "transmit",
}
PHASE_COLORS = {
    0: "#e07b39",  # orange
    1: "#4a90d9",  # blue
    2: "#1a4f8a",  # dark blue
    3: "#4caf73",  # green
    4: "#e84040",  # red
}


def plot_depth_profiles(
    output,
    figsize=None,
    ncols=2,
    linewidth=1.2,
    title=None,
    save_path=None,
    dpi=300,
    show=True,
    floats=None,
    max_days=None,
    mark_profiles=False,
):
    """Plot float depth vs days since deployment, coloured by cycle phase.

    Each segment of the continuous line is coloured by its cycle phase so the
    transitions are visible without breaking the line. Y-axis is inverted
    (surface at top). Vertical dashed lines mark cycle boundaries.

    Parameters
    ----------
    output : str
        Path to a zarr directory, or a glob matching multiple batch zarrs.
    ncols : int
        Number of subplot columns (rows auto-calculated).
    linewidth : float
        Width of the depth profile line.
    save_path : str, optional
        Directory to save the figure (as <stem>_depth.png).
    floats : list of int, optional
        Trajectory indices to plot (default: all floats).
    max_days : float, optional
        Crop the x-axis to the first ``max_days`` days (e.g. ~22 for two cycles).
    mark_profiles : bool
        Overlay a star at each **recorded** profile position -- the shallowest
        point of each cycle, excluding the deployment cycle, matching
        :func:`ohc._one_position_per_cycle` (the point whose lat/lon/time become
        the synthetic Argo observation).
    """
    from matplotlib.collections import LineCollection

    ds = open_trajectories(output)
    stem = _zarr_stem(output)

    z = ds.z.values.astype(float)  # (trajectory, obs)
    phase = ds.cycle_phase.values.astype(float)  # may contain NaN
    cyc_num = ds.cycle_number.values
    times = ds.time.values  # datetime64, (trajectory, obs)

    sel = list(range(z.shape[0])) if floats is None else list(floats)
    z, phase, cyc_num, times = z[sel], phase[sel], cyc_num[sel], times[sel]

    # Days since the start of each float's own deployment.
    t0_ns = times[:, 0:1].astype("i8")  # (trajectory, 1), nanoseconds
    days = (times.astype("i8") - t0_ns) / 8.64e13  # (trajectory, obs)

    n_traj = z.shape[0]
    nrows = int(np.ceil(n_traj / ncols))

    if figsize is None:
        # Width follows quarto's fig-width (rcParams); height keeps the
        # original per-panel aspect ratio (7:3.2).
        width = plt.rcParams["figure.figsize"][0]
        figsize = (width, width / ncols * (3.2 / 7) * nrows)

    if title is None:
        t_start = np.datetime_as_string(times[:, 0].min(), unit="D")
        t_end = np.datetime_as_string(times[:, -1].max(), unit="D")
        title = f"Float depth profiles by cycle phase  |  {t_start} → {t_end}"

    # Build a LineCollection for one float: consecutive (day, z) pairs
    # coloured by the phase at the *start* of each segment.
    def _phase_lines(days_i, z_i, phase_i):
        pts = np.stack([days_i, z_i], axis=1)  # (obs, 2)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (obs-1, 2, 2)
        colors = [
            PHASE_COLORS.get(int(p), "#dddddd") if np.isfinite(p) else "#dddddd"
            for p in phase_i[:-1]
        ]
        return LineCollection(segs, colors=colors, linewidth=linewidth, zorder=3)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        sharey=True,
    )
    axes = np.array(axes).ravel()

    for i in range(n_traj):
        ax = axes[i]
        days_i = days[i]
        zi = z[i]
        pi = phase[i]
        ci = cyc_num[i]

        ax.add_collection(_phase_lines(days_i, zi, pi))

        finite_cycles = np.unique(ci[np.isfinite(ci)])
        # Cycle boundary markers (skip the first/deployment cycle boundary).
        for c in finite_cycles[1:]:
            first_obs = np.where(ci == c)[0][0]
            xb = days_i[first_obs]
            if max_days and xb > max_days:  # don't place artists outside the crop
                continue
            ax.axvline(xb, color="#aaaaaa", linewidth=0.7, linestyle="--", zorder=2)
            ax.text(
                xb + 0.3,
                np.nanmax(zi) * 0.97,
                f"C{int(c)}",
                fontsize=6.5,
                color="#888888",
                va="top",
            )

        # Recorded profile positions: shallowest point of each cycle, excluding
        # the deployment cycle (matches ohc._one_position_per_cycle).
        if mark_profiles and finite_cycles.size > 1:
            for c in finite_cycles[1:]:
                idxs = np.where((ci == c) & np.isfinite(zi))[0]
                if idxs.size:
                    j = idxs[np.argmin(zi[idxs])]
                    if max_days and days_i[j] > max_days:
                        continue
                    ax.scatter(
                        days_i[j],
                        zi[j],
                        marker="*",
                        s=160,
                        color="k",
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=6,
                    )

        ax.set_xlim(0, max_days) if max_days else ax.set_xlim(days_i[0], days_i[-1])
        zmax = np.nanmax(zi)
        ax.set_ylim(zmax * 1.05, -zmax * 0.02)  # inverted, with headroom
        ax.set_title(f"Float {sel[i]}", fontsize=10)
        ax.set_xlabel("Days since deployment", fontsize=8)
        ax.set_ylabel("Depth (m)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#cccccc")
        ax.grid(False)

    # Hide unused subplots.
    for j in range(n_traj, len(axes)):
        axes[j].set_visible(False)

    # Build a shared legend from the phase palette.
    legend_handles = [
        plt.Line2D([0], [0], color=PHASE_COLORS[ph], linewidth=2, label=label)
        for ph, label in PHASE_LABELS.items()
    ]
    if mark_profiles:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="k",
                markeredgecolor="white",
                linestyle="none",
                markersize=11,
                label="profile recorded",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(title, fontsize=12, y=0.99)
    # Reserve space for the suptitle (top) and the shared legend (bottom); robust
    # for a single panel where constrained_layout would collapse the axes.
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    if save_path is not None:
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)
        fig_file = os.path.join(save_path, f"{stem}_depth.png")
        fig.savefig(fig_file, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure -> {fig_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_deployment_plan(
    plan, title="Float Deployment Plan", figsize=None, margin=3
):
    lats = plan["lat"]
    lons = plan["lon"]

    extent = [
        lons.min() - margin,
        lons.max() + margin,
        lats.min() - margin,
        lats.max() + margin,
    ]

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#d5e9f5")
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d0", edgecolor="grey", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgrey")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", linestyle=":")
    ax.gridlines(
        draw_labels=True, linewidth=0.4, color="grey", alpha=0.6, linestyle="--"
    )

    ax.scatter(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        s=20,
        color="steelblue",
        edgecolors="k",
        linewidths=0.5,
        alpha=0.9,
        zorder=5,
        label=f"{len(lats)} floats",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.show()
