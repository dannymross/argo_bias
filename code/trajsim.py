"""Argo float trajectory simulation in the North Atlantic.

Compute-only library: no plotting, no notebook side effects, so it imports
cleanly on an HPC compute node. Plotting lives in ``trajplots.py``; data
download lives behind a lazy import in :func:`fetch_velocity_data`.

The expensive piece of a long simulation is the velocity field. Always build
the field from a file path / glob (see :func:`build_velocity_field`) so parcels
loads it lazily (``deferred_load=True``) instead of pulling the whole multi-year
field into memory.
"""

import os
import uuid
from datetime import timedelta

import numpy as np
import xarray as xr
from global_land_mask import globe
from virtualargofleet import Velocity, FloatConfiguration, VirtualFleet

# Default CMEMS GLORYS12V1 variable / coordinate names.
_VARIABLES = {"U": "uo", "V": "vo"}
_DIMENSIONS = {"time": "time", "depth": "depth", "lat": "latitude", "lon": "longitude"}


# ---- DEPLOYMENT ----------------------------------------------------------
def deploy_float_grid(
    top_left,
    bottom_right=None,
    nfloats=None,
    spacing_deg=None,
    deploy_time="2020-01-01",
    ocean_only=True,
    snap_deg=None,
    cell_deg=None,
):
    """Deploy floats on a regular lat/lon grid within a bounding box.

    Mode 1 — top_left + bottom_right + nfloats:
        Aspect-ratio-preserving even grid across the full box.
    Mode 2 — top_left + bottom_right + spacing_deg:
        Fixed degree spacing edge-to-edge across the full box.
    Mode 3 — top_left + spacing_deg [+ nfloats]:
        Fixed degree spacing expanding to North Atlantic default bounds.
        If nfloats is also given, the grid is built first then capped to
        the first nfloats ocean points (row-major: W->E, N->S).

    The result is deterministic for fixed inputs, so independent workers can
    rebuild the same plan and each take a slice via :func:`select_floats`.

    Parameters
    ----------
    top_left : tuple of (float, float)
        (lat, lon) of the NW corner; always the first grid point.
    bottom_right : tuple of (float, float), optional
        (lat, lon) of the SE corner.
    nfloats : int, optional
        Cap on total floats deployed.
    spacing_deg : float, optional
        Distance between floats in degrees.
    deploy_time : str
        ISO-8601 deployment date for all floats.
    ocean_only : bool
        Drop grid points that fall on land (default True).
    snap_deg : float, optional
        If set, snap deployment latitudes/longitudes to the nearest multiple of
        ``snap_deg`` so floats sit at model cell centres (use 1/12 for GLORYS12,
        whose grid points lie on exact multiples of 1/12 deg). Positions that are
        already centred are left unchanged (the snap is idempotent).
    cell_deg : float, optional
        If set, deploy exactly one float at the centre of every analysis cell of
        size ``cell_deg`` within the box, using the same floor-binning as
        :func:`ohc.grid_cells` (centres at ``floor(x/cell_deg)*cell_deg +
        cell_deg/2``). This removes any grid-spacing mismatch between the
        deployment and the analysis grid (one float per cell). Overrides the
        nfloats / spacing_deg grid. Requires ``bottom_right``.

    Returns
    -------
    dict with keys "lat", "lon", "time".
    """
    lat_max, lon_min = top_left

    # Resolve SE corner
    if bottom_right is not None:
        lat_min, lon_max = bottom_right
    elif spacing_deg is not None:
        # Expand to full North Atlantic default bounds
        lat_min, lon_max = 25.0, 30.0
    else:
        raise ValueError("Provide either bottom_right, spacing_deg, or both.")

    # Build 1-D grid arrays
    if cell_deg is not None:
        # One float per analysis-cell centre (floor-binning, matching
        # ohc.grid_cells), so deployment tiles the analysis grid exactly.
        def _cell_centers_in(lo, hi, c):
            start = np.floor(lo / c) * c + c / 2.0
            cs = np.arange(start, hi, c)
            return cs[(cs >= lo) & (cs <= hi)]
        lat_pts = _cell_centers_in(lat_min, lat_max, cell_deg)[::-1]  # N -> S
        lon_pts = _cell_centers_in(lon_min, lon_max, cell_deg)
    elif spacing_deg is not None:
        lat_pts = np.arange(lat_max, lat_min - 1e-9, -spacing_deg)
        lon_pts = np.arange(lon_min, lon_max + 1e-9, spacing_deg)
    else:
        # nfloats only: split proportional to aspect ratio
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        aspect = lon_span / lat_span
        n_lat = max(1, round((nfloats / aspect) ** 0.5))
        n_lon = max(1, round(n_lat * aspect))
        lat_pts = np.linspace(lat_max, lat_min, n_lat)
        lon_pts = np.linspace(lon_min, lon_max, n_lon)

    # Snap to model cell centres (e.g. GLORYS 1/12 deg grid). Idempotent for
    # positions already on the grid.
    if snap_deg:
        lat_pts = np.round(lat_pts / snap_deg) * snap_deg
        lon_pts = np.round(lon_pts / snap_deg) * snap_deg

    lon_grid, lat_grid = np.meshgrid(lon_pts, lat_pts)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    # Land-sea mask
    if ocean_only:
        is_ocean = globe.is_ocean(lat_flat, lon_flat)
        lat_flat = lat_flat[is_ocean]
        lon_flat = lon_flat[is_ocean]

    # Cap to nfloats. Rounding in the aspect-ratio grid can produce more points
    # than requested, and the land mask never adds points, so this is always safe.
    if nfloats is not None:
        lat_flat = lat_flat[:nfloats]
        lon_flat = lon_flat[:nfloats]

    nfloats_actual = len(lat_flat)
    tim = np.array([deploy_time] * nfloats_actual, dtype="datetime64")

    print(f"{nfloats_actual} floats deployed")
    return {"lat": lat_flat, "lon": lon_flat, "time": tim}


def select_floats(plan, start, count):
    """Return a contiguous slice of a deployment plan.

    Used to hand each parallel worker its own batch of floats. ``start`` and
    ``count`` are clamped to the plan size, so an over-long final batch is fine.
    """
    stop = start + count
    return {k: v[start:stop] for k, v in plan.items()}


def split_indices(nfloats, ntasks):
    """Split ``nfloats`` into ``ntasks`` near-equal (start, count) batches."""
    base, extra = divmod(nfloats, ntasks)
    batches, start = [], 0
    for t in range(ntasks):
        count = base + (1 if t < extra else 0)
        batches.append((start, count))
        start += count
    return batches


# ---- VELOCITY FIELD ------------------------------------------------------
def fetch_velocity_data(
    out_dir="~/data/",
    out_file="velocity.nc",
    lon_bounds=(-78, 17),
    lat_bounds=(18, 80),
    start_date="2020-01-01",
    end_date="2020-01-31",
    depth_bounds=(0, 2300),
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=("uo", "vo", "thetao"),
    force_download=False,
):
    """Download a velocity subset from Copernicus Marine; return the file path.

    ``copernicusmarine`` is imported lazily so the module stays importable on
    compute nodes without network access or the package installed. Returns the
    NetCDF path (not an open Dataset) so the caller can hand it straight to
    :func:`build_velocity_field` for lazy loading.
    """
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_file)

    readme_path = path + ".json"

    if not force_download and os.path.exists(path):
        print(f"Using cached file: {path}")
        return path

    import copernicusmarine  # lazy: only needed when actually downloading

    print(f"Downloading velocity data -> {path}")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=lon_bounds[0],
        maximum_longitude=lon_bounds[1],
        minimum_latitude=lat_bounds[0],
        maximum_latitude=lat_bounds[1],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_depth=depth_bounds[0],
        maximum_depth=depth_bounds[1],
        output_filename=out_file,
        output_directory=out_dir,
    )

    import datetime as _dt, json as _json
    readme = {
        "file": out_file,
        "downloaded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "variables": list(variables),
        "lon_bounds": list(lon_bounds),
        "lat_bounds": list(lat_bounds),
        "depth_bounds": list(depth_bounds),
        "start_date": start_date,
        "end_date": end_date,
    }
    with open(readme_path, "w") as f:
        _json.dump(readme, f, indent=2)
    print(f"Metadata written -> {readme_path}")

    return path


def _month_range(start_month, end_month):
    """Yield (year, month) tuples from 'YYYY-MM' start to end, inclusive."""
    sy, sm = (int(x) for x in start_month.split("-"))
    ey, em = (int(x) for x in end_month.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_velocity_months(
    start_month,
    end_month,
    out_dir="~/data/velocity/",
    file_prefix="velocity",
    lon_bounds=(-78, 17),
    lat_bounds=(18, 80),
    depth_bounds=(0, 2300),
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=("uo", "vo", "thetao"),
    force_download=False,
):
    """Download a velocity subset one file per month over a date span.

    One file per month keeps each download (and each lazily-read time chunk)
    manageable for multi-year fields, and lets parcels stitch them along time
    via a glob. Existing files are skipped unless ``force_download`` is set, so
    interrupted downloads resume cleanly.

    Parameters
    ----------
    start_month, end_month : str
        Inclusive 'YYYY-MM' bounds, e.g. "2015-01" to "2024-12".
    out_dir : str
        Directory for the monthly NetCDF files.
    file_prefix : str
        Files are named ``<prefix>_YYYY_MM.nc``.
    (remaining args mirror :func:`fetch_velocity_data`.)

    Returns
    -------
    tuple of (list[str], str)
        The list of monthly file paths and a glob pattern matching them
        (``<out_dir>/<prefix>_*.nc``) — pass the glob to
        :func:`build_velocity_field`.
    """
    import calendar

    out_dir = os.path.expanduser(out_dir)
    paths = []
    for year, month in _month_range(start_month, end_month):
        last_day = calendar.monthrange(year, month)[1]
        out_file = f"{file_prefix}_{year:04d}_{month:02d}.nc"
        path = fetch_velocity_data(
            out_dir=out_dir,
            out_file=out_file,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            start_date=f"{year:04d}-{month:02d}-01",
            end_date=f"{year:04d}-{month:02d}-{last_day:02d}",
            depth_bounds=depth_bounds,
            dataset_id=dataset_id,
            variables=variables,
            force_download=force_download,
        )
        paths.append(path)

    glob_pattern = os.path.join(out_dir, f"{file_prefix}_*.nc")
    print(f"{len(paths)} monthly files -> {glob_pattern}")
    return paths, glob_pattern


def build_velocity_field(
    src,
    model="GLORYS12V1",
    variables=_VARIABLES,
    dimensions=_DIMENSIONS,
):
    """Wrap a velocity source in a VirtualArgoFleet ``Velocity`` fieldset.

    Parameters
    ----------
    src : str or xarray.Dataset
        Preferred: a file path or glob string (e.g. ``"data/velocity/*.nc"``).
        This routes through ``FieldSet.from_netcdf(deferred_load=True)``, so
        parcels reads time chunks lazily and memory stays bounded regardless of
        how many years the field spans. Passing an open ``Dataset`` instead
        forces an eager in-memory load and will not scale to multi-year fields.
    model : str
        Model identifier passed to ``Velocity`` (default "GLORYS12V1").
    variables, dimensions : dict
        NetCDF variable / coordinate name maps. Defaults match CMEMS GLORYS12V1.

    Returns
    -------
    Velocity
    """
    if isinstance(src, xr.Dataset):
        print(
            "WARNING: building the field from an in-memory Dataset loads the "
            "whole field into RAM. Pass a file path/glob for lazy loading."
        )
        return Velocity(model=model, src=src, variables=variables, dimensions=dimensions)

    src = os.path.expanduser(src)
    return Velocity(model=model, src=src, variables=variables, dimensions=dimensions)


# ---- SIMULATION ----------------------------------------------------------
def run_simulation(
    plan,
    velocity_field,
    duration_days=30,
    step_seconds=300,
    record_seconds=3600,
    out_dir="data/virtualfleet/",
    out_file=None,
    cfg=None,
    nc_save=False,
):
    """Run a VirtualArgoFleet simulation for one batch of floats.

    Parameters
    ----------
    plan : dict
        Deployment plan with keys "lat", "lon", "time".
    velocity_field : Velocity
        Fieldset built by :func:`build_velocity_field`.
    duration_days : int or float
        Simulation duration in days.
    step_seconds : int
        Integration time step (default 300 s).
    record_seconds : int
        Output recording period (default 3600 s); must be a multiple of step.
    out_dir : str
        Directory to write the trajectory output.
    out_file : str, optional
        Output ``.zarr`` name. A random name is generated if omitted.
    cfg : FloatConfiguration, optional
        Mission configuration. Defaults to ``FloatConfiguration("default")``.
    nc_save : bool
        Also write a ``.nc`` copy of the zarr output.

    Returns
    -------
    str
        Full path to the trajectory output (.zarr).
    """
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if cfg is None:
        cfg = FloatConfiguration("default")

    if out_file is None:
        out_file = f"trajectory_{uuid.uuid4().hex[:8]}.zarr"

    fleet = VirtualFleet(plan=plan, fieldset=velocity_field, mission=cfg)
    output = os.path.join(out_dir, out_file)

    print(f"Running simulation: {duration_days} days -> {output}")
    fleet.simulate(
        duration=timedelta(days=duration_days),
        step=timedelta(seconds=step_seconds),
        record=timedelta(seconds=record_seconds),
        output=True,
        output_folder=out_dir,
        output_file=out_file,
    )

    if nc_save:
        nc_path = os.path.splitext(output.rstrip("/"))[0] + ".nc"
        xr.open_zarr(output).to_netcdf(nc_path)

    return output
