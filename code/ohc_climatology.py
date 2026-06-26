"""Climatological OHC mean field from the RG Argo Climatology, and OHC anomalies.

The RG monthly extension files (``data/rg_climatology/RG_ArgoClim_<YYYYMM>_2019.nc``)
contain only the *anomaly* relative to the 2004-2018 climatology -- not the mean
itself. The mean lives in a separate, large file
(``RG_ArgoClim_Temperature_2019.nc``, fetched by
:func:`code.download_rg_monthly.download_mean`): ``ARGO_TEMPERATURE_MEAN`` is a
single **static** 15-year (Jan 2004 - Dec 2018) mean field on ``(pressure,
latitude, longitude)`` -- no time/month dimension, so using it alone (Option B,
below) folds the real seasonal OHC cycle into "the anomaly" along with
everything else.

That same file also has ``ARGO_TEMPERATURE_ANOMALY`` resolved over all 180
months (15 years x 12) of 2004-2018, relative to that static mean -- i.e. the
ingredients for a proper calendar-month climatology are already here, just not
pre-computed. :func:`load_rg_anomaly_series` + :func:`seasonal_climatology_mean`
average those 180 months within each calendar month and add the result back to
the static mean, giving Option A: a 12-month seasonal mean temperature field.

Two climatology options, both integrated by the same
:func:`climatological_ohc_field` (it doesn't care whether its input has a
``month`` dimension or not):

* **Option A (seasonal, recommended default)** -- ``climatological_ohc_field(
  seasonal_climatology_mean(mean_da, anomaly_da))``, on ``(month, latitude,
  longitude)``. Anomalies relative to this isolate real interannual variability
  from the seasonal cycle.
* **Option B (static)** -- ``climatological_ohc_field(mean_da)``, on
  ``(latitude, longitude)`` only. Simpler, but its anomaly includes the seasonal
  cycle as part of the "signal."

``climatology_at``/``climatology_on_grid`` sample either one (by calendar month,
when the field has one) at arbitrary points or onto another grid -- the building
block for computing OHC anomalies for both the synthetic-float profiles and the
GLORYS truth field.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

import ohc

RG_MEAN_PATH = "../data/rg_climatology/RG_ArgoClim_Temperature_2019.nc"
RG_EXT_PATH_TEMPLATE = "../data/rg_climatology/RG_ArgoClim_{year}{month:02d}_2019.nc"
RG_MEAN_VAR = "ARGO_TEMPERATURE_MEAN"
RG_ANOMALY_VAR = "ARGO_TEMPERATURE_ANOMALY"
GP_INTERP_SCRIPT = os.path.join(os.path.dirname(__file__), "ohc_gp_interp.R")
LEVITUS_INTERP_SCRIPT = os.path.join(os.path.dirname(__file__), "ohc_levitus_interp.R")
LEVITUS_R_DEFAULT_KM = 666.0  # literal Levitus (2012) value -- smoother interpolation given this small domain.


def _to_180(lon):
    """Convert 0-360 longitude (RG's convention) to -180..180."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180) % 360) - 180


def load_rg_mean(path=RG_MEAN_PATH, mean_var=RG_MEAN_VAR):
    """Load the RG climatological mean temperature, with lon in -180..180.

    Returns a DataArray on ``(latitude, longitude, pressure)`` -- a single static
    field, since the RG mean has no seasonal/monthly dimension. ``pressure`` is
    RG's 58 standard levels (dbar), treated as metres via the same dbar~m
    approximation used throughout :mod:`ohc`.
    """
    # decode_times=False: the file's TIME axis uses "months since" units that
    # xarray can't decode by default, and we don't need TIME for the (time-
    # independent) mean variable anyway.
    ds = xr.open_dataset(path, decode_times=False)
    da = ds[mean_var]
    da = da.rename(
        {"LATITUDE": "latitude", "LONGITUDE": "longitude", "PRESSURE": "pressure"}
    )
    da = da.assign_coords(longitude=_to_180(da["longitude"]))
    return da.sortby("longitude")


def load_rg_anomaly_series(
    path=RG_MEAN_PATH, anomaly_var=RG_ANOMALY_VAR, time_chunk=12
):
    """Load the 2004-2018 monthly ``ARGO_TEMPERATURE_ANOMALY`` time series.

    Same file as :func:`load_rg_mean` -- it has both the static mean and this
    180-month (15 years x 12 months) anomaly series relative to that mean.
    Loaded lazily (dask-chunked along time): the full array is ~2 GB, but
    :func:`seasonal_climatology_mean` only ever materializes the much smaller
    per-calendar-month average.

    Returns
    -------
    xarray.DataArray
        On ``(time, pressure, latitude, longitude)``, with an extra
        ``month_of_year`` (1-12) coordinate on ``time`` for the groupby in
        :func:`seasonal_climatology_mean`.
    """
    ds = xr.open_dataset(path, decode_times=False, chunks={"TIME": time_chunk})
    da = ds[anomaly_var]
    da = da.rename(
        {
            "TIME": "time",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "PRESSURE": "pressure",
        }
    )
    da = da.assign_coords(longitude=_to_180(da["longitude"]))
    da = da.sortby("longitude")
    # TIME = 0.5, 1.5, ... (mid-month offsets, months since Jan 2004) -> 1-12.
    month_of_year = (np.floor(da["time"].to_numpy()).astype(int) % 12) + 1
    return da.assign_coords(month_of_year=("time", month_of_year))


def load_rg_extensions(year=2020, months=range(1, 13), path_template=RG_EXT_PATH_TEMPLATE):
    """Load the monthly RG extension files for *year* (default 2020).

    Each file (e.g. ``RG_ArgoClim_202001_2019.nc``) contains one month of
    ``ARGO_TEMPERATURE_ANOMALY`` relative to the 2004-2018 mean from
    :func:`load_rg_mean`.  Adding the result to that mean gives the actual
    observed RG temperature field for each month.

    Returns
    -------
    xarray.DataArray
        On ``(month: 1..12, pressure, latitude, longitude)`` with lon in
        -180..180.
    """
    months = list(months)
    das = []
    for m in months:
        path = path_template.format(year=year, month=m)
        ds = xr.open_dataset(path, decode_times=False)
        da = ds[RG_ANOMALY_VAR].squeeze("TIME").drop_vars("TIME")
        da = da.rename(
            {"LATITUDE": "latitude", "LONGITUDE": "longitude", "PRESSURE": "pressure"}
        )
        da = da.assign_coords(longitude=_to_180(da["longitude"]))
        da = da.sortby("longitude")
        das.append(da)
    return xr.concat(das, dim=xr.DataArray(months, dims="month", name="month"))


def seasonal_climatology_mean(mean_da, anomaly_da):
    """Calendar-month climatological mean temperature (Option A).

    Averages ``anomaly_da`` (the 180-month 2004-2018 series from
    :func:`load_rg_anomaly_series`) within each calendar month and adds
    ``mean_da`` back, giving a 12-month seasonal mean -- as opposed to using
    ``mean_da`` alone (Option B), a single time-invariant field with no seasonal
    cycle at all.

    Returns
    -------
    xarray.DataArray
        Mean temperature on ``(month: 1..12, pressure, latitude, longitude)``.
    """
    seasonal_anom = anomaly_da.groupby("month_of_year").mean("time")
    return (mean_da + seasonal_anom).rename({"month_of_year": "month"})


def climatological_ohc_field(mean_da, depths=ohc.DEPTHS):
    """Integrate a climatological mean temperature field over depth.

    Same integration rule as :func:`ohc.truth_ohc_field`, applied to RG's mean
    temperature on its native 1-degree / 58-level grid. Works for either
    climatology option unchanged: ``mean_da`` may be the static field from
    :func:`load_rg_mean` (Option B, on ``(latitude, longitude, pressure)``) or
    the seasonal field from :func:`seasonal_climatology_mean` (Option A, with an
    extra leading ``month`` dimension) -- ``apply_ufunc`` below broadcasts over
    whatever leading dimensions are present.

    Returns
    -------
    xarray.Dataset
        ``ohc_700``, ``ohc_2000`` (J/m2), with the same non-depth dimensions as
        ``mean_da`` (``(latitude, longitude)`` for Option B, ``(month,
        latitude, longitude)`` for Option A).
    """
    theta = mean_da.transpose(..., "pressure")
    z = theta["pressure"].values

    out = {}
    for zmax in depths:
        out[f"ohc_{zmax}"] = xr.apply_ufunc(
            ohc._profile_ohc_core,
            theta,
            input_core_dims=[["pressure"]],
            kwargs={"depth": z, "zmax": zmax},
            dask="parallelized",
            output_dtypes=[float],
        )
    ds = xr.Dataset(out)
    ds.attrs["units"] = "J m-2"
    return ds


def climatology_at(clim_field, lat, lon, month=None):
    """Bilinear-interpolate the climatological OHC field onto matched points.

    Parameters
    ----------
    clim_field : xarray.Dataset
        Output of :func:`climatological_ohc_field` -- either Option A (with a
        ``month`` dimension) or Option B (without one).
    lat, lon : array-like
        Matched 1-D point arrays (e.g. float profile positions).
    month : array-like, optional
        Calendar month (1-12) per point. Required if ``clim_field`` has a
        ``month`` dimension (Option A); ignored otherwise (Option B).

    Returns
    -------
    xarray.Dataset
        Interpolated on dim ``"points"``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    spatial = clim_field.interp(
        latitude=xr.DataArray(lat, dims="points"),
        longitude=xr.DataArray(lon, dims="points"),
    )
    if "month" not in clim_field.dims:
        return spatial
    if month is None:
        raise ValueError("clim_field has a month dimension; month= is required")
    # idx shares the "points" dim name with `spatial`, so .sel does a diagonal
    # (pointwise) selection: for point i, pick month=month[i] *at* point i --
    # not an outer product of every month against every point.
    idx = xr.DataArray(np.asarray(month, dtype=int), dims="points")
    return spatial.sel(month=idx)


def climatology_on_grid(clim_field, lats, lons):
    """Climatological OHC field re-gridded (bilinear) onto an arbitrary lat/lon grid.

    Preserves a ``month`` dimension if ``clim_field`` has one (Option A) --
    only latitude/longitude are touched.
    """
    return clim_field.interp(
        latitude=xr.DataArray(lats, dims="latitude"),
        longitude=xr.DataArray(lons, dims="longitude"),
    )


def _select_month(field, month_timestamps):
    """Select ``field``'s ``month`` (1-12) dim to match an array of calendar-month timestamps.

    For each entry in ``month_timestamps`` (e.g. truth/GP-field month
    timestamps), picks the matching calendar-month slice out of ``field``'s
    12-entry seasonal ``month`` dimension, broadcasting across ``field``'s other
    dims (e.g. latitude/longitude) -- an outer replacement, not the pointwise
    selection in :func:`climatology_at`. The indexer's dim is named differently
    from ``"month"`` to avoid a coordinate-name collision in ``.sel`` (xarray
    would otherwise try to align the dropped 1-12 ``month`` coordinate against
    the new timestamp one).
    """
    month_of_year = pd.DatetimeIndex(np.asarray(month_timestamps)).month
    idx = xr.DataArray(
        month_of_year, dims="_t", coords={"_t": np.asarray(month_timestamps)}
    )
    return field.sel(month=idx).drop_vars("month").rename({"_t": "month"})


# ---- ANOMALY CONSTRUCTION -------------------------------------------------
def float_ohc_anomalies(sim, clim_field, value_cols=("ohc_700", "ohc_2000")):
    """Per-profile OHC anomaly: synthetic-Argo OHC minus the RG climatological mean.

    Parameters
    ----------
    sim : pandas.DataFrame
        Per-profile synthetic-Argo OHC, the output of :func:`ohc.float_ohc`
        (columns ``float_id, cycle, lat, lon, date, ohc_700, ohc_2000``).
    clim_field : xarray.Dataset
        Output of :func:`climatological_ohc_field`.

    Returns
    -------
    pandas.DataFrame
        ``sim`` plus a ``month`` (calendar-month timestamp) column and
        ``<col>_anom`` for each of ``value_cols``.
    """
    df = sim.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    clim_vals = climatology_at(
        clim_field,
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
        month=df["date"].dt.month.to_numpy(),
    )
    for c in value_cols:
        df[f"{c}_anom"] = df[c].to_numpy() - clim_vals[c].to_numpy()
    return df


def truth_ohc_anomaly(truth_field, clim_field, value_cols=("ohc_700", "ohc_2000")):
    """Monthly-mean GLORYS truth OHC anomaly relative to the RG climatology.

    Returns the truth field's monthly mean (native grid) minus the climatology
    (Option A: that calendar month's seasonal mean; Option B: the one static
    field), on ``(month, latitude, longitude)`` -- same grid as ``truth_field``.
    """
    monthly = truth_field.resample(time="1MS").mean().rename({"time": "month"})
    lats = monthly["latitude"].to_numpy()
    lons = monthly["longitude"].to_numpy()
    clim_grid = climatology_on_grid(clim_field, lats, lons)
    if "month" in clim_grid.dims:
        clim_grid = _select_month(clim_grid, monthly["month"].to_numpy())
    return monthly[list(value_cols)] - clim_grid[list(value_cols)]


def build_pred_grid(lats, lons):
    """Flatten a lat/lon grid to the (lon, lat) point table the R script expects.

    Cast to float64: ``truth_field``'s lat/lon are GLORYS's native float32, and
    writing float32 values to CSV prints fewer significant digits, which loses
    enough precision in the Python -> R -> Python round-trip to break exact
    coordinate-equality alignment later (xarray silently inner-joins on the
    overlap instead of raising, so this is easy to miss). See also
    :func:`snap_grid_coords`, the second line of defence for the same issue.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return pd.DataFrame({"lon": lon_grid.ravel(), "lat": lat_grid.ravel()})


# ---- R SUBPROCESS GLUE -----------------------------------------------------
def write_profile_csv(anom_df, path, value_cols=("ohc_700", "ohc_2000")):
    """Write the per-profile anomaly table the R Vecchia-GP script reads.

    The temporal coordinate written is the actual observation ``date`` (YYYY-MM-DD),
    not the calendar month. The R script converts this to day-of-year for the
    spatio-temporal GP fit and uses a configurable day within each calendar month
    (first/middle/last, controlled by :func:`run_gp_interp`'s ``month_day``
    argument) as the prediction temporal coordinate.
    """
    out = anom_df[["date", "lon", "lat"] + [f"{c}_anom" for c in value_cols]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def run_gp_interp(
    profiles_csv,
    grid_csv,
    out_csv,
    fit_summary_csv,
    model_cache=None,
    month_day="middle",
    r_script=GP_INTERP_SCRIPT,
):
    """Shell out to ``code/ohc_gp_interp.R`` to fit/predict the Vecchia GP.

    ``fit_summary_csv`` is where the pooled fit's parameter table gets written.
    Pass ``model_cache`` (path to a ``.rds`` file) to save the fitted model on
    the first call and reload it on subsequent calls -- so predicting at a
    second grid (e.g. 1°) skips the expensive fit entirely.

    ``month_day`` controls the temporal prediction point within each month:
    ``"first"`` (1st), ``"middle"`` (15th, default), or ``"last"`` (last day).
    """
    if month_day not in ("first", "middle", "last"):
        raise ValueError(f"month_day must be 'first', 'middle', or 'last'; got {month_day!r}")
    cmd = ["Rscript", r_script, profiles_csv, grid_csv, out_csv, fit_summary_csv]
    if model_cache is not None:
        cmd.append(model_cache)
    cmd.append(month_day)
    subprocess.run(cmd, check=True)


def run_levitus_interp(
    profiles_csv,
    grid_csv,
    out_csv,
    R=LEVITUS_R_DEFAULT_KM,
    pooled=False,
    r_script=LEVITUS_INTERP_SCRIPT,
):
    """Shell out to ``code/ohc_levitus_interp.R`` to run the Levitus-style kernel smoother.

    Mirrors :func:`run_gp_interp`'s subprocess pattern. ``profiles_csv``/
    ``grid_csv`` are the same files :func:`write_profile_csv`/
    :func:`build_pred_grid` produce for the GP path -- both estimators read the
    identical input format.

    ``pooled=True`` pools every month's profiles into one fixed set (reasonable
    since these are de-seasonalized anomalies) and writes one static,
    time-invariant prediction per grid point -- ``out_csv`` then has no
    ``month`` column at all; load it with
    :func:`load_levitus_pooled_anomaly_field` instead of
    :func:`load_levitus_anomaly_field`.
    """
    cmd = ["Rscript", r_script, profiles_csv, grid_csv, out_csv, str(R)]
    if pooled:
        cmd.append("pooled")
    subprocess.run(cmd, check=True)


def load_gp_fit_summary(fit_summary_csv):
    """Load the pooled GP fit's parameter table written by ``ohc_gp_interp.R``.

    One row per (depth, parameter): ``mean (intercept)`` plus the
    ``matern_spheretime`` covariance parameters (variance, spatial_range,
    temporal_range, smoothness, nugget). ``std_error``/``z_stat`` are NaN for
    ``smoothness``, which is fixed rather than estimated (see
    ``code/ohc_gp_interp.R:FIXED_SMOOTHNESS`` -- letting it float left the
    likelihood unconverged). ``loglik``/``converged``/``n_obs`` repeat per row
    for convenience but are constant within a depth.
    """
    return pd.read_csv(fit_summary_csv)


def snap_grid_coords(field, lats, lons):
    """Replace ``field``'s latitude/longitude coordinates with authoritative arrays.

    Guards against the float32 -> CSV -> float64 precision loss described in
    :func:`build_pred_grid`: even after that fix, a CSV round-trip isn't
    guaranteed to reproduce the exact bit pattern xarray needs for coordinate
    alignment, and a near-miss fails *silently* (an inner join on the overlap,
    not an error). Only valid when ``field``'s grid is known by construction to
    be the same grid as ``(lats, lons)``, just re-derived from text -- asserts
    the shapes match so a real mismatch still raises loudly.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    assert field.sizes["latitude"] == len(lats), (
        f"latitude size mismatch: field has {field.sizes['latitude']}, expected {len(lats)}"
    )
    assert field.sizes["longitude"] == len(lons), (
        f"longitude size mismatch: field has {field.sizes['longitude']}, expected {len(lons)}"
    )
    return field.assign_coords(latitude=lats, longitude=lons)


def _load_anomaly_field(
    df, lats, lons, se_suffixes, value_cols=("ohc_700", "ohc_2000")
):
    """Convert a long (month, lon, lat, ...) prediction table into an xarray field.

    Shared by :func:`load_gp_anomaly_field` and :func:`load_levitus_anomaly_field`
    -- they differ only in how many SE columns each estimator produces
    (``se_suffixes``, e.g. ``("_se",)`` for the GP, ``("_se_a", "_se_0")`` for
    the Levitus-style estimator).

    ``lats``/``lons`` are the authoritative prediction-grid coordinates (e.g.
    ``truth_field``'s native lat/lon) -- passed straight to
    :func:`snap_grid_coords` to undo any precision loss from the CSV round-trip
    (both estimators run as standalone R scripts via subprocess), since this
    field must align exactly with the truth grid for later comparison.

    Returns a Dataset on ``(month, latitude, longitude)`` with ``<col>_anom`` and
    ``<col>_anom<suffix>`` for each suffix in ``se_suffixes``, for each of
    ``value_cols``, plus ``too_few_profiles``.
    """
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    cols = [f"{c}_anom_pred" for c in value_cols]
    cols += [f"{c}_anom{suf}" for c in value_cols for suf in se_suffixes]
    cols += ["too_few_profiles"]
    ds = df.set_index(["month", "latitude", "longitude"])[cols].to_xarray()
    ds = ds.rename({f"{c}_anom_pred": f"{c}_anom" for c in value_cols})
    # Sort ascending so position-for-position alignment with sorted lats/lons
    # (below) is correct regardless of the row order .to_xarray() produced.
    ds = ds.sortby(["latitude", "longitude"])
    return snap_grid_coords(ds, np.sort(lats), np.sort(lons))


def load_gp_anomaly_field(out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000")):
    """Load the R script's prediction CSV into an xarray field.

    ``<col>_anom_se`` is the GP's Vecchia-approximation standard error (see
    ``code/ohc_gp_interp.R:kriging_predict``; NaN for any (month, depth) where
    there were too few profiles to fit a GP at all -- see ``too_few_profiles``).
    See :func:`_load_anomaly_field` for the shared loading logic.
    """
    pred = pd.read_csv(out_csv, parse_dates=["month"])
    return _load_anomaly_field(
        pred, lats, lons, se_suffixes=("_se",), value_cols=value_cols
    )


def load_levitus_anomaly_field(out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000")):
    """Load the Levitus-style R script's prediction CSV into an xarray field.

    ``<col>_anom_se_a``/``<col>_anom_se_0`` are the two SE variants the
    Levitus-style estimator produces -- sigma_A (independence assumption) and
    sigma_0 (the source doc's eq. 7, taken literally) -- see
    ``code/ohc_levitus_interp.R``. See :func:`_load_anomaly_field` for the
    shared loading logic.
    """
    pred = pd.read_csv(out_csv, parse_dates=["month"])
    return _load_anomaly_field(
        pred, lats, lons, se_suffixes=("_se_a", "_se_0"), value_cols=value_cols
    )


def load_levitus_pooled_anomaly_field(
    out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000")
):
    """Load the Levitus R script's *pooled* prediction CSV (``run_levitus_interp(..., pooled=True)``).

    Same columns as :func:`load_levitus_anomaly_field` minus ``month`` -- one
    static, time-invariant prediction per grid point, pooling every month's
    profiles into a single set. Returns a Dataset on ``(latitude, longitude)``
    only (no ``month`` dim); pass it to :func:`assemble_levitus_pooled_field`
    (not :func:`assemble_levitus_field`, which expects a ``month`` dim).
    """
    pred = pd.read_csv(out_csv)
    pred = pred.rename(columns={"lat": "latitude", "lon": "longitude"})
    se_suffixes = ("_se_a", "_se_0")
    cols = [f"{c}_anom_pred" for c in value_cols]
    cols += [f"{c}_anom{suf}" for c in value_cols for suf in se_suffixes]
    cols += ["too_few_profiles"]
    ds = pred.set_index(["latitude", "longitude"])[cols].to_xarray()
    ds = ds.rename({f"{c}_anom_pred": f"{c}_anom" for c in value_cols})
    ds = ds.sortby(["latitude", "longitude"])
    return snap_grid_coords(ds, np.sort(lats), np.sort(lons))


# ---- FIELD ASSEMBLY ---------------------------------------------------------
def _assemble_field(
    anom_field, clim_field, se_suffixes, value_cols=("ohc_700", "ohc_2000")
):
    """Add the climatological mean back to a predicted-anomaly field.

    Shared by :func:`assemble_gp_field` and :func:`assemble_levitus_field` --
    they differ only in which SE columns get carried over (``se_suffixes``).
    SE is unchanged by this step: the climatology is a fixed offset, so it adds
    no uncertainty.

    Where the anomaly itself is NaN (e.g. the Levitus estimator outside its
    influence radius -- there are no observations to predict an anomaly from
    at all), the absolute field falls back to the climatology alone (anomaly
    treated as 0) rather than propagating NaN -- a real, fully-populated OHC
    map rather than one with holes wherever a profile didn't happen to be
    nearby. The anomaly field itself is untouched, so it still correctly shows
    where that gap is.
    """
    lats = anom_field["latitude"].to_numpy()
    lons = anom_field["longitude"].to_numpy()
    clim_grid = climatology_on_grid(clim_field, lats, lons)
    if "month" in clim_grid.dims:
        clim_grid = _select_month(clim_grid, anom_field["month"].to_numpy())
    out = xr.Dataset(
        {
            **{c: anom_field[f"{c}_anom"].fillna(0) + clim_grid[c] for c in value_cols},
            **{
                f"{c}{suf}": anom_field[f"{c}_anom{suf}"]
                for c in value_cols
                for suf in se_suffixes
            },
        }
    )
    out.attrs["units"] = "J m-2"
    return out


def assemble_gp_field(gp_anom_field, clim_field, value_cols=("ohc_700", "ohc_2000")):
    """Add the climatological mean back to the GP-predicted anomaly field.

    Returns absolute OHC (``ohc_700``, ``ohc_2000``) on the same grid as
    ``gp_anom_field`` -- the GP estimator, directly comparable to
    ``truth_field.resample(time="1MS").mean()`` -- plus ``<col>_se``. See
    :func:`_assemble_field` for the shared assembly logic.
    """
    return _assemble_field(
        gp_anom_field, clim_field, se_suffixes=("_se",), value_cols=value_cols
    )


def assemble_levitus_field(
    levitus_anom_field, clim_field, value_cols=("ohc_700", "ohc_2000")
):
    """Add the climatological mean back to the Levitus-predicted anomaly field.

    Mirrors :func:`assemble_gp_field`, carrying both SE variants
    (``<col>_se_a``, ``<col>_se_0``) over unchanged.
    """
    return _assemble_field(
        levitus_anom_field,
        clim_field,
        se_suffixes=("_se_a", "_se_0"),
        value_cols=value_cols,
    )


def assemble_levitus_pooled_field(
    pooled_anom_field, clim_field, months, value_cols=("ohc_700", "ohc_2000")
):
    """Add the climatological mean back to the pooled (time-invariant) Levitus anomaly field.

    ``pooled_anom_field`` (from :func:`load_levitus_pooled_anomaly_field`) has
    no ``month`` dim -- one static anomaly correction for the whole year.
    Broadcasting it onto ``months`` (actual calendar timestamps) before adding
    the climatology means every month gets the *same* anomaly correction added
    to that month's own seasonal climatology -- so the only month-to-month
    variation in the result comes from the climatology, not the float data.
    """
    anom_field = pooled_anom_field.expand_dims(month=pd.DatetimeIndex(months))
    return _assemble_field(
        anom_field, clim_field, se_suffixes=("_se_a", "_se_0"), value_cols=value_cols
    )


def climatology_cells_table(
    clim_field, lats, lons, months=None, value_cols=("ohc_700", "ohc_2000")
):
    """Climatological mean field on an arbitrary grid, as a cells table.

    The mean-field counterpart to :func:`to_cells_table` on the GP/Levitus
    anomaly fields -- this is exactly what :func:`assemble_gp_field`/
    :func:`assemble_levitus_field` add the predicted anomaly to, so plotting it
    alongside them on the same colour scale shows how the two sum to the final
    OHC map. ``months`` (actual calendar timestamps, e.g. a GP/Levitus field's
    ``month`` coordinate) selects each one's matching calendar-month slice if
    ``clim_field`` has a seasonal ``month`` dimension (Option A); ignored for
    the static Option B field.
    """
    grid = climatology_on_grid(clim_field, lats, lons)
    if "month" in grid.dims and months is not None:
        grid = _select_month(grid, months)
    return to_cells_table(grid, value_cols=value_cols)


def to_cells_table(field, value_cols=("ohc_700", "ohc_2000")):
    """Flatten a ``(month, latitude, longitude)`` field to a cells table.

    Matches the ``month, cell_lat, cell_lon, <value_cols>`` shape produced by
    :func:`ohc.grid_cells` / :func:`ohc.coarsen_truth`, so
    :func:`ohc_bias.plot_monthly_cell_maps` works on it unchanged -- even though
    this grid is the continuous GP field (or the native truth grid), not a
    floor-binned cell average.
    """
    df = field[list(value_cols)].to_dataframe().reset_index()
    return df.rename(columns={"latitude": "cell_lat", "longitude": "cell_lon"})
