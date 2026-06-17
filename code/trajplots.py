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
    paths = sorted(glob.glob(output)) if isinstance(output, str) and "*" in output else [output]
    if len(paths) == 1:
        return xr.open_zarr(paths[0])
    ds = xr.concat([xr.open_zarr(p) for p in paths], dim="trajectory")
    return ds.assign_coords(trajectory=np.arange(ds.sizes["trajectory"]))


def _zarr_stem(output):
    """Return a filename stem from a zarr path (directory or file)."""
    return os.path.splitext(os.path.basename(str(output).rstrip("/")))[0]


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


def map_trajectories(
    output,
    lon="lon",
    lat="lat",
    figsize=(13, 8),
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

    lat_vals = ds[lat].values.astype(float)   # (trajectory, obs)
    lon_vals = ds[lon].values.astype(float)
    n_traj = lat_vals.shape[0]

    # Auto title from date range
    if title is None:
        t_start = np.datetime_as_string(ds.time.values[:, 0].min(), unit="D")
        t_end   = np.datetime_as_string(ds.time.values[:, -1].max(), unit="D")
        title = f"Argo float trajectories  |  {n_traj} floats  |  {t_start} → {t_end}"

    colors = cm.tab10(np.linspace(0, 1, max(n_traj, 1)))

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#d5e9f5")
    ax.add_feature(cfeature.LAND,  facecolor="#e8e0d0", edgecolor="grey", linewidth=0.4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgrey")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="grey", linestyle=":")

    for i in range(n_traj):
        lons_i = lon_vals[i]
        lats_i = lat_vals[i]
        color  = colors[i]

        # Trajectory line
        ax.plot(lons_i, lats_i,
                linewidth=linewidth,
                color=color,
                transform=ccrs.PlateCarree(),
                zorder=3)

        # Deployment marker (circle)
        ax.scatter(lons_i[0], lats_i[0],
                   s=40, color=color, marker="o", edgecolors="k", linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=5,
                   label=f"Float {i+1}")

        # End marker (star)
        ax.scatter(lons_i[-1], lats_i[-1],
                   s=60, color=color, marker="*", edgecolors="k", linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=5)

    map_extent = extent if extent is not None else _auto_extent(lat_vals, lon_vals, margin)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False

    ax.legend(loc="lower left", fontsize=8, framealpha=0.85,
              title="○ start  ★ end", title_fontsize=7)
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
    figsize=(8, 8),
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
    """
    ds = open_trajectories(output)
    lat_vals = ds["lat"].values.astype(float)
    lon_vals = ds["lon"].values.astype(float)
    n_traj = lat_vals.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#ededed", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey", zorder=2)

    for i in range(n_traj):
        ax.plot(lon_vals[i], lat_vals[i], lw=0.5, color=line_color, alpha=0.35,
                transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter(lon_vals[:, 0], lat_vals[:, 0], s=12, color="#e84040", zorder=5,
               edgecolors="none", transform=ccrs.PlateCarree(), label="deployment")

    for box in (boxes or []):
        la0, la1, lo0, lo1 = box[:4]
        label = box[4] if len(box) > 4 else None
        ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0],
                lw=1.2, ls="--", color="k", alpha=0.7, transform=ccrs.PlateCarree(),
                zorder=4, label=label)

    map_extent = extent if extent is not None else _auto_extent(lat_vals, lon_vals, margin)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.4, linestyle=":")
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


PHASE_LABELS = {
    0: "init descend",
    1: "drift",
    2: "profile descend",
    3: "profile ascend",
    4: "transmit",
}
PHASE_COLORS = {
    0: "#e07b39",   # orange
    1: "#4a90d9",   # blue
    2: "#1a4f8a",   # dark blue
    3: "#4caf73",   # green
    4: "#e84040",   # red
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
    """
    from matplotlib.collections import LineCollection

    ds = open_trajectories(output)
    stem = _zarr_stem(output)

    z       = ds.z.values.astype(float)       # (trajectory, obs)
    phase   = ds.cycle_phase.values.astype(int)
    cyc_num = ds.cycle_number.values
    times   = ds.time.values                  # datetime64, (trajectory, obs)

    # Days since the start of each float's own deployment.
    t0_ns  = times[:, 0:1].astype("i8")      # (trajectory, 1), nanoseconds
    days   = (times.astype("i8") - t0_ns) / 8.64e13   # (trajectory, obs)

    n_traj = z.shape[0]
    nrows  = int(np.ceil(n_traj / ncols))

    if figsize is None:
        figsize = (ncols * 7, nrows * 3.2)

    if title is None:
        t_start = np.datetime_as_string(times[:, 0].min(), unit="D")
        t_end   = np.datetime_as_string(times[:, -1].max(), unit="D")
        title = f"Float depth profiles by cycle phase  |  {t_start} → {t_end}"

    # Build a LineCollection for one float: consecutive (day, z) pairs
    # coloured by the phase at the *start* of each segment.
    def _phase_lines(days_i, z_i, phase_i):
        pts  = np.stack([days_i, z_i], axis=1)          # (obs, 2)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)    # (obs-1, 2, 2)
        colors = [PHASE_COLORS[p] for p in phase_i[:-1]]
        return LineCollection(segs, colors=colors, linewidth=linewidth, zorder=3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=False, sharey=True,
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i in range(n_traj):
        ax     = axes[i]
        days_i = days[i]
        zi     = z[i]
        pi     = phase[i]
        ci     = cyc_num[i]

        ax.add_collection(_phase_lines(days_i, zi, pi))

        # Cycle boundary markers.
        for c in np.unique(ci)[1:]:
            first_obs = np.where(ci == c)[0][0]
            xb = days_i[first_obs]
            ax.axvline(xb, color="#aaaaaa", linewidth=0.7, linestyle="--", zorder=2)
            ax.text(xb + 0.3, zi.max() * 0.97,
                    f"C{int(c)}", fontsize=6.5, color="#888888", va="top")

        ax.set_xlim(days_i[0], days_i[-1])
        ax.set_ylim(zi.max() * 1.05, -zi.max() * 0.02)   # inverted, with headroom
        ax.set_title(f"Float {i + 1}", fontsize=10)
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
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(PHASE_LABELS), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=12, y=1.01)

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


def plot_deployment_plan(plan, title="Float Deployment Plan", figsize=(12, 8), margin=3):
    lats = plan["lat"]
    lons = plan["lon"]

    extent = [
        lons.min() - margin,
        lons.max() + margin,
        lats.min() - margin,
        lats.max() + margin,
    ]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="#d5e9f5")
    ax.add_feature(cfeature.LAND,  facecolor="#e8e0d0", edgecolor="grey", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgrey")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.6, linestyle="--")

    ax.scatter(lons, lats,
               transform=ccrs.PlateCarree(),
               s=20, color="steelblue", edgecolors="k", linewidths=0.5,
               alpha=0.9, zorder=5, label=f"{len(lats)} floats")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)
    plt.tight_layout()
    plt.show()
