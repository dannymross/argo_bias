"""Ocean heat content (OHC) from GLORYS12 temperature, for the sampling-bias study.

Two roles, one integration rule:

* **Truth** -- integrate the full GLORYS ``thetao`` field over depth at every grid
  cell/time to get the complete-coverage OHC the floats are compared against.
* **Synthetic Argo** -- sample the GLORYS temperature column at each virtual
  float's profile position/time and integrate the same way, mimicking what an
  Argo float "observes".

OHC follows the advisor's convention::

    OHC = trapz(z, T * cp * rho)   with cp = 3989.411 J/kg/K, rho = 1028.319 kg/m3

integrated 0-700 m and 0-2000 m. Depth in metres is treated as pressure in dbar
(1 dbar ~ 1 m). Results are returned in J/m2; divide by 1e9 for GJ/m2 to match
``data/argo_ohc.csv`` and ``data/ohc_en4_gridded.rds``.

Pure xarray/numpy: no VirtualFleet import, so this stays importable for analysis
on any machine.
"""

import numpy as np
import pandas as pd
import xarray as xr

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Advisor's seawater constants (specific heat, reference density).
CP = 3989.411  # J / (kg K)
RHO = 1028.319  # kg / m^3

J_TO_GJ = 1e-9

# Standard integration depths.
DEPTHS = (700, 2000)


# ---- CORE INTEGRATION ----------------------------------------------------
def profile_ohc(depth, theta, zmax):
    """Integrate ``T * cp * rho`` from the surface to ``zmax`` (J/m2).

    Vectorised over leading axes: ``theta`` may be 1-D ``(nz,)`` for a single
    profile or N-D ``(..., nz)`` for a stack of profiles / a gridded field, with
    depth along the last axis. ``depth`` is the 1-D level vector (increasing).

    The integral spans exactly ``[0, zmax]``:

    * a surface segment ``[0, depth[0]]`` using the shallowest level's value
      (GLORYS' top level is ~0.49 m, so this is a sub-metre correction);
    * the native levels with ``depth <= zmax``, integrated trapezoidally;
    * a bottom segment from the deepest in-range level up to ``zmax``, with
      ``T`` linearly interpolated to ``zmax`` from the bracketing levels.

    A column containing NaN within ``[0, zmax]`` (e.g. below the seafloor on the
    shelf) returns NaN -- that water column cannot be integrated to ``zmax``.
    """
    depth = np.asarray(depth, dtype=float)
    theta = np.asarray(theta, dtype=float)

    in_range = depth <= zmax
    n_in = int(in_range.sum())
    if n_in == 0:
        raise ValueError(
            f"No levels at or above zmax={zmax} m (shallowest {depth[0]} m)."
        )

    d_in = depth[in_range]
    t_in = theta[..., in_range]

    # Native trapezoidal part over levels within [depth[0], zmax].
    integ = _trapz(t_in, d_in, axis=-1)

    # Surface segment [0, depth[0]] assuming constant = shallowest value.
    if d_in[0] > 0:
        integ = integ + t_in[..., 0] * d_in[0]

    # Bottom segment (d_lo, zmax] interpolating T to zmax, if a deeper level exists.
    if n_in < depth.size and d_in[-1] < zmax:
        d_lo = d_in[-1]
        d_hi = depth[n_in]  # first level deeper than zmax
        t_lo = t_in[..., -1]
        t_hi = theta[..., n_in]
        t_zmax = t_lo + (t_hi - t_lo) * (zmax - d_lo) / (d_hi - d_lo)
        integ = integ + 0.5 * (t_lo + t_zmax) * (zmax - d_lo)

    return integ * CP * RHO


def _profile_ohc_core(theta, depth, zmax):
    """theta-first adapter so ``apply_ufunc`` can pass the field array positionally."""
    return profile_ohc(depth, theta, zmax)


# ---- TRUTH FIELD ---------------------------------------------------------
def truth_ohc_field(theta_ds, depths=DEPTHS, theta_var="thetao", depth_dim="depth"):
    """Compute complete-coverage OHC fields from a GLORYS temperature dataset.

    Parameters
    ----------
    theta_ds : xarray.Dataset or DataArray
        Must contain ``thetao`` (or ``theta_var``) on a ``depth`` axis plus
        ``time``/``latitude``/``longitude``.
    depths : iterable of int
        Integration limits in metres (default 700 and 2000).

    Returns
    -------
    xarray.Dataset
        ``ohc_700``, ``ohc_2000`` (J/m2) on (time, latitude, longitude).
    """
    theta = theta_ds[theta_var] if isinstance(theta_ds, xr.Dataset) else theta_ds
    # Move depth to the last axis so profile_ohc integrates along it.
    theta = theta.transpose(..., depth_dim)
    z = theta[depth_dim].values

    out = {}
    for zmax in depths:
        ohc = xr.apply_ufunc(
            _profile_ohc_core,
            theta,
            input_core_dims=[[depth_dim]],
            kwargs={"depth": z, "zmax": zmax},
            dask="parallelized",
            output_dtypes=[float],
        )
        out[f"ohc_{zmax}"] = ohc
    ds = xr.Dataset(out)
    ds.attrs["units"] = "J m-2"
    ds.attrs["cp"] = CP
    ds.attrs["rho"] = RHO
    return ds


# ---- GRIDDING TO 1-DEGREE CELLS -----------------------------------------
def grid_cells(df, value_cols, deg=1, lat_col="lat", lon_col="lon", date_col="date"):
    """Bin point OHC into deg-degree monthly cells, returning the cell means.

    Works for both synthetic floats and a truth field flattened to points, so
    truth and floats land on identical cells. Returns one row per
    (month, cell_lat, cell_lon) with the mean of each value column and a count.
    """
    df = df.copy()
    df["cell_lat"] = np.floor(df[lat_col].to_numpy() / deg) * deg + deg / 2.0
    df["cell_lon"] = np.floor(df[lon_col].to_numpy() / deg) * deg + deg / 2.0
    df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()

    agg = {c: "mean" for c in value_cols}
    cells = (
        df.groupby(["month", "cell_lat", "cell_lon"], as_index=False)
        .agg({**agg, lat_col: "size"})
        .rename(columns={lat_col: "n"})
    )
    return cells


def truth_domain_mean(truth_field, weighted=True, value_cols=None):
    """True domain-mean OHC per month from the native-resolution field.

    This is the fixed reference for the bias-vs-resolution sweep: unlike the
    coarsened cell-mean (which drifts as the analysis cell size changes), it is
    computed once on the native grid and does not depend on the estimator.

    Parameters
    ----------
    truth_field : xarray.Dataset
        Output of :func:`truth_ohc_field` (ohc_700/ohc_2000 on the native grid).
    weighted : bool
        If True (default), cos(lat) area-weighted mean over valid (deep-water)
        cells -- the gold standard. If False, a simple unweighted mean over
        native cells (kept as an option for comparison).
    value_cols : list, optional
        OHC variables to average (default: all data variables).

    Returns
    -------
    pandas.DataFrame
        Columns ``['month', *value_cols]`` in J/m2, one row per month.
    """
    if value_cols is None:
        value_cols = list(truth_field.data_vars)
    monthly = truth_field.resample(time="1MS").mean()
    if weighted:
        w = np.cos(np.deg2rad(monthly["latitude"]))
        dm = monthly.weighted(w).mean(dim=("latitude", "longitude"))
    else:
        dm = monthly.mean(dim=("latitude", "longitude"))
    df = dm.to_dataframe().reset_index().rename(columns={"time": "month"})
    df["month"] = pd.to_datetime(df["month"])
    return df[["month"] + list(value_cols)]


def coarsen_truth(truth_ds, deg=1, value_cols=None):
    """Monthly-mean the truth field and average onto deg-degree cells.

    Uses the same floor-based cell definition as :func:`grid_cells` (via a
    flatten-to-points step) so truth and float cells align exactly.
    """
    if value_cols is None:
        value_cols = list(truth_ds.data_vars)
    monthly = truth_ds.resample(time="1MS").mean()
    df = (
        monthly.to_dataframe()
        .reset_index()
        .rename(columns={"latitude": "lat", "longitude": "lon", "time": "date"})
        .dropna(subset=value_cols, how="all")
    )
    return grid_cells(df, value_cols, deg=deg)


def load_en4_cells(path, bounds=None, year=None):
    """Load EN4 gridded OHC (monthly 1-deg) into the cell format used here.

    Reads the CSV produced by ``code/convert_en4.R`` and returns columns
    ``month, cell_lat, cell_lon, ohc_700, ohc_2000`` (+ ``cell_area_m2``) in
    J/m2, matching :func:`coarsen_truth` / :func:`grid_cells` output so EN4 can
    be mapped and compared the same way as the GLORYS truth.

    Note EN4 cells are centred on integer degrees, i.e. offset 0.5 deg from the
    floor-binned GLORYS 1-deg cells (``cell_centres at x.5``); keep that in mind
    before differencing EN4 against GLORYS cell-by-cell.

    Parameters
    ----------
    path : str
        CSV path (columns lon, lat, year, month, date, ohc_700, ohc_2000, ...).
    bounds : tuple, optional
        (lat_min, lat_max, lon_min, lon_max) to subset.
    year : int, optional
        Restrict to a single calendar year.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    if bounds is not None:
        la0, la1, lo0, lo1 = bounds
        df = df[df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)]
    if year is not None:
        df = df[df["year"] == year]
    # Drop the integer year/month columns (the latter collides with the renamed date).
    df = df.drop(columns=[c for c in ("year", "month") if c in df.columns])
    return df.rename(columns={"lat": "cell_lat", "lon": "cell_lon", "date": "month"})


# ---- SYNTHETIC FLOAT SAMPLING -------------------------------------------
def _one_position_per_cycle(traj, traj_dim="trajectory"):
    """Reduce a trajectory dataset to one (float, cycle) surfacing position.

    For each float and each ``cycle_number``, takes the shallowest observation
    (minimum ``z``) of the cycle -- the surfacing point where the float reports
    its profile. This is robust to the brief ascent/transmit phases being missed
    at the output cadence (the surface points are reliably recorded).

    The first (deployment) cycle is dropped: a float starts at the surface at its
    launch position, which would otherwise register a spurious profile at the
    deployment cell on day 0. A real Argo float does not report a profile until
    it surfaces at the end of its first ~10-day cycle, at its drifted position --
    so the first kept observation is that post-ascent surfacing.

    Returns a tidy DataFrame with float_id, cycle, lat, lon, date. ``traj_dim``
    is the per-float dimension of the VirtualFleet/Parcels zarr (``trajectory``).
    """
    df = (
        traj[["cycle_number", "z", "lat", "lon", "time"]]
        .to_dataframe()
        .reset_index()
        .rename(columns={traj_dim: "float_id", "cycle_number": "cycle", "time": "date"})
        .dropna(subset=["cycle", "z", "lat", "lon"])
    )
    # Shallowest (min z) observation of each (float, cycle): one profile per cycle.
    idx = df.groupby(["float_id", "cycle"])["z"].idxmin()
    out = df.loc[idx, ["float_id", "cycle", "lat", "lon", "date"]].copy()
    out["cycle"] = out["cycle"].astype(int)
    out = out.sort_values(["float_id", "cycle"]).reset_index(drop=True)
    # Drop the deployment cycle (smallest cycle per float).
    first_cycle = out.groupby("float_id")["cycle"].transform("min")
    return out[out["cycle"] > first_cycle].reset_index(drop=True)


def sample_float_profiles(traj, theta_ds, theta_var="thetao", depth_dim="depth"):
    """Extract a GLORYS temperature column at each float surfacing position.

    Returns the per-profile positions DataFrame (from
    :func:`_one_position_per_cycle`) and the sampled temperature array of shape
    ``(nprofiles, ndepth)`` plus the depth vector.
    """
    pos = _one_position_per_cycle(traj)
    theta = theta_ds[theta_var] if isinstance(theta_ds, xr.Dataset) else theta_ds

    cols = theta.interp(
        time=xr.DataArray(pos["date"].values, dims="profile"),
        latitude=xr.DataArray(pos["lat"].values, dims="profile"),
        longitude=xr.DataArray(pos["lon"].values, dims="profile"),
    )
    cols = cols.transpose("profile", depth_dim)
    return pos, cols.values, cols[depth_dim].values


def float_ohc(traj, theta_ds, depths=DEPTHS):
    """Compute per-profile synthetic-Argo OHC from a trajectory + GLORYS field.

    Returns a tidy DataFrame (float_id, cycle, lat, lon, date, ohc_700,
    ohc_2000) in J/m2 -- the same schema as ``data/argo_ohc.csv``.
    """
    pos, theta_cols, z = sample_float_profiles(traj, theta_ds)
    for zmax in depths:
        pos[f"ohc_{zmax}"] = profile_ohc(z, theta_cols, zmax)
    return pos
