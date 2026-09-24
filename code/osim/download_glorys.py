"""GLORYS12V1 download helpers for the Lagrangian float simulation.

Generalized copernicusmarine.subset wrappers: cache-by-path, a JSON metadata
sidecar per file, resumable (existing files skipped). Standalone -- imports only
copernicusmarine (lazily) so it runs under the argo_bias env without VirtualFleet.

Datasets (all GLORYS12V1, 1/12 deg):
    static   cmems_mod_glo_phy_my_0.083deg_static        (deptho, e3t, mask, ...)
    daily    cmems_mod_glo_phy_my_0.083deg_P1D-m         (thetao, uo/vo, zos, ...)
    monthly  cmems_mod_glo_phy_my_0.083deg_P1M-m         (thetao, zos)
"""

import calendar
import datetime as _dt
import json
import os

STATIC_ID = "cmems_mod_glo_phy_my_0.083deg_static"
DAILY_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
MONTHLY_ID = "cmems_mod_glo_phy_my_0.083deg_P1M-m"


def fetch_subset(
    out_dir,
    out_file,
    dataset_id,
    variables,
    lon_bounds,
    lat_bounds,
    depth_bounds=None,
    start_date=None,
    end_date=None,
    dataset_part=None,
    compression_level=0,
    force=False,
):
    """Download one copernicusmarine subset to out_dir/out_file; return the path.

    depth_bounds / start_date / end_date are optional so this also serves the
    2-D and time-less (static) datasets. dataset_part selects a part of a
    multi-part dataset (the static grid splits into 'bathy' and 'coords').
    compression_level > 0 writes deflate-compressed NetCDF (lossless; the int16
    packing is unchanged), halving on-disk size at a small read-time cost.
    Writes a <file>.json sidecar recording the request; skips the download if
    the file already exists and not force.
    """
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_file)

    if not force and os.path.exists(path):
        print(f"skip (exists): {path}")
        return path

    import copernicusmarine  # lazy: only needed when actually downloading

    kw = dict(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=lon_bounds[0],
        maximum_longitude=lon_bounds[1],
        minimum_latitude=lat_bounds[0],
        maximum_latitude=lat_bounds[1],
        output_filename=out_file,
        output_directory=out_dir,
    )
    if depth_bounds is not None:
        kw["minimum_depth"], kw["maximum_depth"] = depth_bounds
    if dataset_part is not None:
        kw["dataset_part"] = dataset_part
    if compression_level:
        kw["netcdf_compression_level"] = compression_level
    if start_date is not None:
        kw["start_datetime"] = start_date
    if end_date is not None:
        kw["end_datetime"] = end_date

    print(f"downloading -> {path}")
    copernicusmarine.subset(**kw)

    with open(path + ".json", "w") as f:
        json.dump(
            {
                "file": out_file,
                "downloaded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                **{k: v for k, v in kw.items()
                   if k not in ("output_filename", "output_directory")},
            },
            f,
            indent=2,
        )
    return path


def _month_range(start_month, end_month):
    """Yield (year, month) from 'YYYY-MM' start to end, inclusive."""
    sy, sm = (int(x) for x in start_month.split("-"))
    ey, em = (int(x) for x in end_month.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_monthly(
    out_dir,
    file_prefix,
    dataset_id,
    variables,
    lon_bounds,
    lat_bounds,
    start_month,
    end_month,
    depth_bounds=None,
    compression_level=0,
    force=False,
):
    """One file per calendar month over [start_month, end_month]; return (paths, glob).

    Chunking by month keeps each download and each lazy time-chunk manageable
    and makes the pull resumable at month granularity. Files are named
    ``<prefix>_YYYY_MM.nc``.
    """
    out_dir = os.path.expanduser(out_dir)
    paths = []
    for year, month in _month_range(start_month, end_month):
        last = calendar.monthrange(year, month)[1]
        path = fetch_subset(
            out_dir=out_dir,
            out_file=f"{file_prefix}_{year:04d}_{month:02d}.nc",
            dataset_id=dataset_id,
            variables=variables,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            depth_bounds=depth_bounds,
            start_date=f"{year:04d}-{month:02d}-01",
            end_date=f"{year:04d}-{month:02d}-{last:02d}",
            compression_level=compression_level,
            force=force,
        )
        paths.append(path)
    glob_pattern = os.path.join(out_dir, f"{file_prefix}_*.nc")
    print(f"{len(paths)} monthly files -> {glob_pattern}")
    return paths, glob_pattern


def fetch_static(out_dir, lon_bounds, lat_bounds, parts, depth_bounds=None,
                 file_prefix="static", force=False):
    """One-time static grid; one file per dataset part -> {part: path}.

    The GLORYS static dataset is split into parts with disjoint variables:
    'bathy' (deptho, deptho_lev, mask) and 'coords' (e1t, e2t, e3t). ``parts``
    maps each wanted part to its variable list. depth_bounds trims the 3-D
    fields (mask, e3t) to the OHC range; the 2-D fields ignore it.
    """
    out = {}
    for part, variables in parts.items():
        out[part] = fetch_subset(
            out_dir=out_dir,
            out_file=f"{file_prefix}_{part}.nc",
            dataset_id=STATIC_ID,
            variables=variables,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            depth_bounds=depth_bounds,
            dataset_part=part,
            force=force,
        )
    return out


def fetch_velocity_bands(
    out_dir,
    file_prefix,
    lon_bounds,
    lat_bounds,
    start_month,
    end_month,
    bands,
    variables=("uo", "vo"),
    dataset_id=DAILY_ID,
    force=False,
):
    """Daily uo/vo over thin, non-contiguous depth bands.

    ``bands`` maps a short label to a (min_depth, max_depth) pair, e.g.
    ``{"surf": (0, 3), "park": (900, 1100)}`` -- the surface drift levels and
    the pair bracketing the 1000 m park depth. Each band is a separate monthly
    file set (``<prefix>_<label>_YYYY_MM.nc``); returns {label: (paths, glob)}.
    """
    out = {}
    for label, depth_bounds in bands.items():
        out[label] = fetch_monthly(
            out_dir=out_dir,
            file_prefix=f"{file_prefix}_{label}",
            dataset_id=dataset_id,
            variables=variables,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            start_month=start_month,
            end_month=end_month,
            depth_bounds=depth_bounds,
            force=force,
        )
    return out
