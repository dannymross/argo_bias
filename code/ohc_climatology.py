"""Climatological OHC mean field from the RG Argo Climatology, and OHC anomalies.

The RG monthly extension files (``RG_ArgoClim_<YYYYMM>_2019.nc``) hold only
the *anomaly* relative to the 2004-2018 climatology, not the mean itself. The
mean lives in a separate file (``RG_ArgoClim_Temperature_2019.nc``, fetched by
:func:`code.download_rg_monthly.download_mean`): ``ARGO_TEMPERATURE_MEAN`` is a
single static 2004-2018 mean with no month dimension, so using it alone
(Option B) folds the seasonal cycle into "the anomaly." The same file's
``ARGO_TEMPERATURE_ANOMALY`` (all 180 months, relative to that mean) lets
:func:`load_rg_anomaly_series` + :func:`seasonal_climatology_mean` build a
proper 12-month seasonal mean instead (Option A, the recommended default).

Both options integrate through the same :func:`climatological_ohc_field`
(with or without a ``month`` dimension); :func:`climatology_at`/
:func:`climatology_on_grid` sample either one at points or onto a grid, for
computing OHC anomalies for both synthetic-float profiles and GLORYS truth.
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
GP_FIT_SCRIPT = os.path.join(os.path.dirname(__file__), "ohc_gp_fit.R")
GP_PREDICT_SCRIPT = os.path.join(os.path.dirname(__file__), "ohc_gp_predict.R")
LEVITUS_INTERP_SCRIPT = os.path.join(os.path.dirname(__file__), "ohc_levitus_interp.R")
GP_AUDIT_SCRIPT = os.path.join(os.path.dirname(__file__), "gp_audit_fields.R")
# Repo root (parent of code/) -- R subprocesses below run with this as cwd so
# renv's project-local library activates via .Rprofile (which R only auto-sources
# from the exact cwd, not ancestor directories -- and reports/ has no .Rprofile
# of its own). File args are made absolute first since cwd is changing.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    Same file as :func:`load_rg_mean`. Dask-chunked along time (full array
    ~2 GB) since :func:`seasonal_climatology_mean` only needs the per-month
    average. Adds a ``month_of_year`` (1-12) coordinate on ``time`` for that
    groupby.
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

    Each file has one month of ``ARGO_TEMPERATURE_ANOMALY`` relative to
    :func:`load_rg_mean`'s static mean; add the two together for the actual
    observed field for that month.
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

    Averages ``anomaly_da`` within each calendar month and adds ``mean_da``
    back -- a 12-month seasonal mean, vs. ``mean_da`` alone (Option B), which
    has no seasonal cycle.
    """
    seasonal_anom = anomaly_da.groupby("month_of_year").mean("time")
    return (mean_da + seasonal_anom).rename({"month_of_year": "month"})


def climatological_ohc_field(mean_da, depths=ohc.DEPTHS):
    """Integrate a climatological mean temperature field over depth.

    Same integration rule as :func:`ohc.truth_ohc_field`. Works for either
    climatology option unchanged -- Option A's ``mean_da`` carries an extra
    leading ``month`` dimension, Option B doesn't; ``apply_ufunc`` broadcasts
    over whichever is present.
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

    ``month`` (calendar month 1-12 per point) is required if ``clim_field``
    has a ``month`` dim (Option A); ignored otherwise (Option B).
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
    """Select ``field``'s ``month`` (1-12) dim to match calendar-month timestamps.

    Outer replacement (broadcast across ``field``'s other dims), not the
    pointwise selection in :func:`climatology_at`. The indexer's dim is named
    differently from ``"month"`` to avoid a coordinate-name collision in
    ``.sel``.
    """
    month_of_year = pd.DatetimeIndex(np.asarray(month_timestamps)).month
    idx = xr.DataArray(
        month_of_year, dims="_t", coords={"_t": np.asarray(month_timestamps)}
    )
    return field.sel(month=idx).drop_vars("month").rename({"_t": "month"})


# ---- ANOMALY CONSTRUCTION -------------------------------------------------
def float_ohc_anomalies(sim, clim_field, value_cols=("ohc_700", "ohc_2000")):
    """Per-profile OHC anomaly: synthetic-Argo OHC minus the RG climatological mean.

    Adds a ``month`` (calendar-month timestamp) column and ``<col>_anom`` for
    each of ``value_cols`` to ``sim`` (output of :func:`ohc.float_ohc`).
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
def write_profile_csv(df, path, value_cols=("ohc_700", "ohc_2000"), suffix="_anom"):
    """Write the per-profile table the R Vecchia-GP script reads.

    ``suffix="_anom"`` (default) writes ``<col>_anom`` (anomaly-modeling
    workflow); ``suffix=""`` writes raw ``<col>`` values so GpGp fits/predicts
    absolute OHC directly (see :func:`load_gp_anomaly_field`). Writes the
    actual observation date, not calendar month -- ``ohc_gp_fit.R`` converts
    it to a continuous day count; ``ohc_gp_predict.R`` uses a configurable day
    within each month (:func:`run_gp_predict`'s ``month_day``) for prediction.
    """
    out = df[["date", "lon", "lat"] + [f"{c}{suffix}" for c in value_cols]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def run_gp_fit(profiles_csv, fit_summary_csv, model_cache, r_script=GP_FIT_SCRIPT):
    """Shell out to ``code/ohc_gp_fit.R`` to fit the pooled Vecchia GP per depth.

    Writes ``fit_summary_csv`` (the fit's parameter table) and ``model_cache``
    (an ``.rds`` of the fitted GpGp model objects). Prediction is a separate
    step -- see :func:`run_gp_predict` -- so one fit here can be reused across
    any number of prediction grids/resolutions without refitting.
    """
    cmd = ["Rscript", os.path.abspath(r_script)] + [
        os.path.abspath(p) for p in (profiles_csv, fit_summary_csv, model_cache)
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_gp_predict(
    profiles_csv,
    grid_csv,
    model_cache,
    out_csv,
    m="exact",
    month_day="middle",
    r_script=GP_PREDICT_SCRIPT,
):
    """Shell out to ``code/ohc_gp_predict.R`` to predict a fitted GP onto a grid.

    ``m="exact"`` (default) conditions every point on all observations --
    correct but scales poorly past a few thousand profiles. A small integer
    (e.g. 30, matching ``ohc_gp_fit.R``'s ``m_seq``) switches to a fast
    bounded-Vecchia approximation that still returns a valid SE, unlike
    ``GpGp::predictions()``. ``month_day`` sets the day used within each
    month: ``"first"``, ``"middle"`` (default), or ``"last"``.
    """
    if month_day not in ("first", "middle", "last"):
        raise ValueError(f"month_day must be 'first', 'middle', or 'last'; got {month_day!r}")
    cmd = ["Rscript", os.path.abspath(r_script)] + [
        os.path.abspath(p) for p in (profiles_csv, grid_csv, model_cache, out_csv)
    ] + [str(m), month_day]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_levitus_interp(
    profiles_csv,
    grid_csv,
    out_csv,
    R=LEVITUS_R_DEFAULT_KM,
    pooled=False,
    r_script=LEVITUS_INTERP_SCRIPT,
):
    """Shell out to ``code/ohc_levitus_interp.R`` to run the Levitus-style kernel smoother.

    Mirrors :func:`run_gp_predict`'s subprocess pattern and input files.
    ``pooled=True`` pools every month's profiles into one static prediction
    per grid point (no ``month`` column in ``out_csv``) -- load with
    :func:`load_levitus_pooled_anomaly_field` instead of
    :func:`load_levitus_anomaly_field`.
    """
    cmd = ["Rscript", os.path.abspath(r_script)] + [
        os.path.abspath(p) for p in (profiles_csv, grid_csv, out_csv)
    ] + [str(R)]
    if pooled:
        cmd.append("pooled")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_gp_audit_fields(
    profiles_csv,
    grid_csv,
    gpgp_out_csv,
    model_cache,
    audit_out_csv=None,
    r_script=GP_AUDIT_SCRIPT,
):
    """Shell out to ``code/gp_audit_fields.R`` to audit the GpGp predictions.

    Reproduces ``kriging_predict``'s output using ``fields::Krig`` with a
    hand-coded matern_spheretime correlation and the fitted ``model_cache``
    parameters; compares GpGp's kriging_predict, a direct Cholesky simple
    kriging, and fields::Krig ordinary kriging. Returns the audit CSV path
    (defaults to ``<gpgp_out_csv stem>_fields_audit.csv``).
    """
    if audit_out_csv is None:
        stem = gpgp_out_csv.replace(".csv", "")
        audit_out_csv = f"{stem}_fields_audit.csv"
    cmd = ["Rscript", os.path.abspath(r_script)] + [
        os.path.abspath(p)
        for p in (profiles_csv, grid_csv, gpgp_out_csv, model_cache, audit_out_csv)
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return audit_out_csv


def load_gp_audit(audit_csv):
    """Load the fields-audit comparison CSV written by ``run_gp_audit_fields``.

    Columns: month/lon/lat/depth; pred_gpgp/pred_sk/pred_fields and
    se_gpgp/se_sk/se_fields (J/m2); pred_sk_diff/pred_fields_diff;
    se_sk_ratio/se_fields_ratio; sigma2_gpgp/sigma2_fields.
    """
    df = pd.read_csv(audit_csv, parse_dates=["month"], low_memory=False)
    return df


def load_gp_fit_summary(fit_summary_csv):
    """Load the pooled GP fit's parameter table written by ``ohc_gp_fit.R``.

    One row per (depth, parameter). ``std_error``/``z_stat`` are NaN for
    ``smoothness``, which is fixed rather than estimated -- letting it float
    left the likelihood unconverged (see ``ohc_gp_fit.R:FIXED_SMOOTHNESS``).
    """
    return pd.read_csv(fit_summary_csv)


def snap_grid_coords(field, lats, lons):
    """Replace ``field``'s latitude/longitude coordinates with authoritative arrays.

    Guards against the float32->CSV->float64 precision loss in
    :func:`build_pred_grid`: even after that fix, a near-miss coordinate
    mismatch fails silently (xarray inner-joins on the overlap instead of
    raising). Only valid when ``field``'s grid is known by construction to
    match ``(lats, lons)``; asserts shapes so a real mismatch still raises.
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
    df, lats, lons, se_suffixes, value_cols=("ohc_700", "ohc_2000"), suffix="_anom"
):
    """Convert a long (month, lon, lat, ...) prediction table into an xarray field.

    Shared by :func:`load_gp_anomaly_field`/:func:`load_levitus_anomaly_field`,
    differing only in ``se_suffixes`` (how many SE columns each estimator
    produces). ``lats``/``lons`` are passed to :func:`snap_grid_coords` to undo
    CSV round-trip precision loss. ``suffix`` must match what
    :func:`write_profile_csv` wrote (``"_anom"`` default, ``""`` for raw values).

    Returns ``(month, latitude, longitude)`` with ``<col><suffix>`` and
    ``<col><suffix><se_suffix>`` per ``se_suffixes``, plus ``too_few_profiles``.
    """
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    cols = [f"{c}{suffix}_pred" for c in value_cols]
    cols += [f"{c}{suffix}{suf}" for c in value_cols for suf in se_suffixes]
    cols += ["too_few_profiles"]
    ds = df.set_index(["month", "latitude", "longitude"])[cols].to_xarray()
    ds = ds.rename({f"{c}{suffix}_pred": f"{c}{suffix}" for c in value_cols})
    # Sort ascending so position-for-position alignment with sorted lats/lons
    # (below) is correct regardless of the row order .to_xarray() produced.
    ds = ds.sortby(["latitude", "longitude"])
    return snap_grid_coords(ds, np.sort(lats), np.sort(lons))


def load_gp_anomaly_field(out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000"), suffix="_anom"):
    """Load the R script's prediction CSV into an xarray field.

    ``<col>_anom_se`` (or ``<col>_se`` if ``suffix=""``) is the GP's
    Vecchia-approximation SE (NaN wherever ``too_few_profiles``). See
    :func:`_load_anomaly_field` for the shared loading logic.
    """
    pred = pd.read_csv(out_csv, parse_dates=["month"])
    return _load_anomaly_field(
        pred, lats, lons, se_suffixes=("_se",), value_cols=value_cols, suffix=suffix
    )


def load_levitus_anomaly_field(out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000")):
    """Load the Levitus-style R script's prediction CSV into an xarray field.

    ``<col>_anom_se_a``/``<col>_anom_se_0`` are the estimator's two SE variants
    (independence-assumption sigma_A, and sigma_0 per the source doc's eq. 7).
    """
    pred = pd.read_csv(out_csv, parse_dates=["month"])
    return _load_anomaly_field(
        pred, lats, lons, se_suffixes=("_se_a", "_se_0"), value_cols=value_cols
    )


def load_levitus_pooled_anomaly_field(
    out_csv, lats, lons, value_cols=("ohc_700", "ohc_2000")
):
    """Load the Levitus R script's *pooled* prediction CSV (``run_levitus_interp(..., pooled=True)``).

    One static prediction per grid point, no ``month`` dim -- pass to
    :func:`assemble_levitus_pooled_field`, not :func:`assemble_levitus_field`.
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

    Shared by :func:`assemble_gp_field`/:func:`assemble_levitus_field`,
    differing only in ``se_suffixes``. SE is unchanged (climatology is a
    fixed offset). Where the anomaly is NaN (e.g. Levitus outside its
    influence radius), the absolute field falls back to the climatology
    alone (anomaly treated as 0) instead of propagating NaN -- a
    fully-populated map; the anomaly field itself is left untouched.
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

    Returns absolute OHC on the same grid as ``gp_anom_field`` plus ``<col>_se``.
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

    Broadcasts the single static anomaly onto ``months`` before adding each
    month's own seasonal climatology, so month-to-month variation comes only
    from the climatology, not the float data.
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
    anomaly fields -- exactly what :func:`assemble_gp_field`/
    :func:`assemble_levitus_field` add the anomaly to. ``months`` selects each
    row's calendar-month slice for a seasonal (Option A) ``clim_field``.
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
