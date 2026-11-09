"""Build the native-grid OHC residual field for the float-simulation study.

OHC (0-2000 m) on the GLORYS 1/12 deg grid, minus a per-cell mean field fitted
from the cfg.FIT_START_YEAR-FIT_END_YEAR monthly record (quadratic trend + annual and semiannual
harmonics). The residual Z is the field the test samples; the analysis grid is
GLORYS' own grid, so the simple-function representation of theory.tex is exact
and no quadrature error term arises.

Run from the repo root: ``python code/osim/build_residual_field.py [box] [--mean-only]``.
"""

import calendar
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ohc
from osim import osim_config as cfg


def _store_years(cadence="daily"):
    """Years present in the layer store.

    The field must span the whole simulated record, not just the scored years:
    a prediction is conditioned on profiles up to a year either side of it, so a
    field cut to cfg.ANALYSIS_YEARS silently starves the window at both ends.
    """
    pat = os.path.join(cfg.GLORYS_DIR, cfg.LAYER_DIR, f"ohc_layers_{cadence}_*.nc")
    years = sorted({int(re.search(r"_(\d{4})_\d{2}\.nc$", f).group(1))
                    for f in glob.glob(pat)})
    if not years:
        raise FileNotFoundError(pat)
    return tuple(years)


def _ohc_from_layers(years, box, cadence, drop_leap=True):
    """Total 0-2000 m OHC over the box from the layer store.

    The layers sum to the full integral; a layer is NaN below the seafloor, so the
    sum is NaN exactly where the floor is shallower than 2000 m. ``drop_leap``
    removes Feb 29 for the day-of-year-aligned layout; the real-calendar layout
    keeps it.
    """
    out = []
    for y in years:
        pat = os.path.join(cfg.GLORYS_DIR, cfg.LAYER_DIR, f"ohc_layers_{cadence}_{y}_*.nc")
        for f in sorted(glob.glob(pat)):
            with xr.open_dataset(f) as ds:
                ds = cfg.box_slice(ds, box)
                out.append(sum(ds[v] for v in ds.data_vars).rename("ohc").load())
    if not out:
        raise FileNotFoundError(f"no {cadence} layer files for {years} in {cfg.LAYER_DIR}")
    field = xr.concat(out, dim="time").sortby("time").to_dataset(name="ohc")
    return cfg.drop_leap_day(field) if drop_leap else field


def load_static(box):
    static = os.path.join(cfg.GLORYS_DIR, os.path.dirname(cfg.STATIC_PREFIX))
    pre = os.path.basename(cfg.STATIC_PREFIX)
    bathy = cfg.box_slice(xr.open_dataset(os.path.join(static, f"{pre}_bathy.nc")), box)
    # The coords file can lack a row of the bathymetry grid (natl starts at 0.083 N).
    coords = xr.open_dataset(os.path.join(static, f"{pre}_coords.nc")).reindex_like(
        bathy, method="nearest", tolerance=1e-3).assign_coords(latitude=bathy.latitude, longitude=bathy.longitude)
    wet = (bathy["deptho"] > cfg.MIN_DEPTH).rename("wet")
    area = (coords["e1t"] * coords["e2t"]).where(wet).rename("cell_area")
    area.attrs.update(units="m2", long_name="e1t*e2t, masked to deptho > 2000 m")
    return area, wet


def fit_mean_field(monthly, quadratic=True):
    """Per-cell OLS of OHC on the trend/harmonic basis.

    The regressors are averaged over each file's month, matching what the
    monthly means actually are; see cfg.design_matrix_averaged. All cells share
    one design matrix, so this is a single pinv and one matmul.
    """
    t_start = monthly["time"].values.astype("datetime64[M]")
    t0 = cfg.decimal_year(t_start.astype("datetime64[ns]"))
    t1 = cfg.decimal_year((t_start + np.timedelta64(1, "M")).astype("datetime64[ns]"))
    X, t_ref = cfg.design_matrix_averaged(t0, t1, quadratic=quadratic)
    Y = monthly["ohc"].values.reshape(len(t0), -1)
    ok = np.isfinite(Y).all(axis=0)
    beta = np.full((X.shape[1], Y.shape[1]), np.nan)
    beta[:, ok] = np.linalg.pinv(X) @ Y[:, ok]
    return beta, t_ref, ok


def evaluate_mean_field(beta, times, shape, quadratic=True, t_ref=None):
    X, _ = cfg.design_matrix(cfg.decimal_year(times), quadratic=quadratic, t_ref=t_ref)
    return (X @ beta).reshape((len(times),) + shape)


def _bin_index(values, deg):
    idx = np.floor(np.asarray(values) / deg).astype(int)
    return idx - idx.min()


def _indicator(idx, ncol, weights=None):
    """Dense (len(idx), ncol) 0/1 (or weighted) aggregation matrix."""
    M = np.zeros((len(idx), ncol), dtype=np.float32)
    M[np.arange(len(idx)), idx] = 1.0 if weights is None else weights
    return M


def coarsen(zda, area, deg=cfg.COARSE_DEG, days=cfg.COARSE_DAYS):
    """Area-weighted aggregate of Z onto coarse space-time cells.

    Only ever used as the surrogate generator's proposal field, so it needs to
    be a sensible place-to-put-points rather than an accurate field.
    """
    ilat = _bin_index(zda["latitude"].values, deg)
    ilon = _bin_index(zda["longitude"].values, deg)
    nlat, nlon = ilat.max() + 1, ilon.max() + 1
    nsp = nlat * nlon

    idoy = np.minimum(np.arange(cfg.DAYS_PER_YEAR) // days, cfg.DAYS_PER_YEAR // days - 1)
    ndoy = idoy.max() + 1

    # Fine cell -> coarse spatial cell, carrying the area weight.
    flat = (ilat[:, None] * nlon + ilon[None, :]).ravel()
    a = np.nan_to_num(area.values).ravel().astype(np.float32)
    S = _indicator(flat, nsp, weights=a)

    z = np.nan_to_num(zda.values).reshape(-1, len(flat)).astype(np.float32)
    wet = (np.isfinite(zda.values).reshape(-1, len(flat))).astype(np.float32)
    num = z @ S
    den = wet @ S
    zsp = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    zsp = zsp.reshape(zda.sizes["year"], cfg.DAYS_PER_YEAR, nsp)

    T = _indicator(idoy, ndoy)
    zc = np.einsum("ydc,dt->ytc", zsp, T) / T.sum(axis=0)[None, :, None]

    lat_c = np.floor(zda["latitude"].values.min() / deg) * deg + deg / 2 + np.arange(nlat) * deg
    lon_c = np.floor(zda["longitude"].values.min() / deg) * deg + deg / 2 + np.arange(nlon) * deg
    out = xr.Dataset(
        {
            "z": (("year", "doy_bin", "cell_lat", "cell_lon"),
                  zc.reshape(zda.sizes["year"], ndoy, nlat, nlon).astype(np.float32)),
            "cell_area": (("cell_lat", "cell_lon"), S.sum(axis=0).reshape(nlat, nlon)),
            "n_days": (("doy_bin",), T.sum(axis=0).astype(np.int32)),
        },
        coords={"year": zda["year"].values, "doy_bin": np.arange(ndoy),
                "cell_lat": lat_c, "cell_lon": lon_c},
    )
    out["z"].attrs.update(units="J m-2", long_name="coarse OHC residual (surrogate proposal field)")
    out["cell_area"].attrs.update(units="m2", long_name="masked area per coarse cell")
    return out


def main(box="boxB", quadratic=True, years=None, coarse=None, real_calendar=None, mean_only=False):
    """Mean field and residual field for one region.

    Two layouts. ``real_calendar`` gives a flat ``time`` axis on the true calendar,
    Feb 29 included -- what a continuous-time estimator (the GP) wants, and the
    default for the conditioning buffer. Otherwise the field is (year, doy) on the
    Feb-29-dropped 365-day calendar, which is what the day-of-year-aligned designs
    index. Which layout a cached field uses is readable from its dims.
    """
    years = tuple(years) if years else _store_years()
    real_calendar = (box == "buffer") if real_calendar is None else real_calendar
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    print(f"[{box}] residual years {years[0]}-{years[-1]} ({len(years)}), "
          f"calendar={'real' if real_calendar else 'leap-dropped'}")
    area, wet = load_static(box)

    print(f"[{box}] mean field: monthly layer store {cfg.FIT_START_YEAR}-{cfg.FIT_END_YEAR}, "
          f"quadratic={quadratic}")
    monthly = _ohc_from_layers(range(cfg.FIT_START_YEAR, cfg.FIT_END_YEAR + 1), box, "monthly")
    shape = monthly["ohc"].shape[1:]
    beta, t_ref, ok = fit_mean_field(monthly, quadratic=quadratic)
    print(f"  {ok.sum():,}/{ok.size:,} cells fitted, {beta.shape[0]} coefficients")
    del monthly
    tag = f"{box}{'' if quadratic else '_lin'}"
    if mean_only:
        _write(_beta_ds(beta, shape, area, t_ref, quadratic), os.path.join(cfg.OUT_DIR, f"mean_field_coef_{tag}.nc"))
        _write(area.to_dataset(), os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))
        return None

    # Filled a year at a time: the buffer is ~10x a box, so holding the whole daily
    # record and its mean-field evaluation at once is several GB more than needed.
    ndays = [366 if (real_calendar and calendar.isleap(y)) else cfg.DAYS_PER_YEAR for y in years]
    zarr_ = np.empty((sum(ndays),) + shape, dtype=np.float32)
    times, off = [], 0
    for y, n in zip(years, ndays):
        daily = _ohc_from_layers([y], box, "daily", drop_leap=not real_calendar)
        if daily.sizes["time"] != n:
            raise ValueError(f"{y}: layer store has {daily.sizes['time']} days, expected {n} "
                             f"-- the year is incomplete")
        mhat = evaluate_mean_field(beta, daily["time"].values, shape,
                                   quadratic=quadratic, t_ref=t_ref)
        zarr_[off:off + n] = (daily["ohc"].values - mhat).astype(np.float32)
        times.append(daily["time"].values)
        off += n
        print(f"  {y}: {n} days", flush=True)
        del daily, mhat
    times = np.concatenate(times)
    zarr_[:, ~wet.values] = np.nan

    coords = {"latitude": area["latitude"], "longitude": area["longitude"]}
    if real_calendar:
        zda = xr.DataArray(zarr_, dims=("time", "latitude", "longitude"),
                           coords={"time": times, **coords}, name="z")
        space = ("latitude", "longitude")
        tdims = ("time",)
    else:
        zda = xr.DataArray(zarr_.reshape((len(years), cfg.DAYS_PER_YEAR) + shape),
                           dims=("year", "doy", "latitude", "longitude"),
                           coords={"year": list(years),
                                   "doy": np.arange(cfg.DAYS_PER_YEAR), **coords}, name="z")
        space = ("latitude", "longitude")
        tdims = ("year", "doy")
    zda.attrs.update(units="J m-2", calendar="real" if real_calendar else "leap-dropped-365",
                     long_name="OHC 0-2000 m residual (OHC minus fitted mean field)")

    # The bathymetry mask alone is not the data support: the layer store needs
    # finite theta at every level to 2533 m (PCHIP locality), so a cell with
    # 2000 < deptho < 2533 passes deptho > MIN_DEPTH but is NaN. Take the
    # intersection so the area weights and the field share one mask -- otherwise
    # Zbar normalises by more area than the field actually covers.
    valid = np.isfinite(zda).all(tdims)
    lost = 1.0 - float(area.where(valid).sum() / area.sum())
    print(f"  data-support mask drops {int((~valid & np.isfinite(area)).sum())} bathymetrically "
          f"wet cells ({lost:.3%} of box area)")
    area = area.where(valid)
    zarr_[:, ~valid.values] = np.nan

    w = area.fillna(0.0)
    per_day = (zda * w).sum(space) / w.sum()
    if real_calendar:
        yr = pd.DatetimeIndex(times).year.to_numpy()
        means = pd.DataFrame({
            "year": list(years),
            "zbar": [float(per_day.values[yr == y].mean()) for y in years],
            "sd_Z": [float(np.nanstd(zarr_[yr == y])) for y in years],
        })
    else:
        means = pd.DataFrame({"year": list(years), "zbar": per_day.mean("doy").values,
                              "sd_Z": zda.std(("doy",) + space).values})

    _write(zda.to_dataset(), os.path.join(cfg.OUT_DIR, f"residual_field_{tag}.nc"))
    _write(area.to_dataset(), os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))
    _write(_beta_ds(beta, shape, zda, t_ref, quadratic), os.path.join(cfg.OUT_DIR, f"mean_field_coef_{tag}.nc"))
    if coarse if coarse is not None else not real_calendar:
        _write(coarsen(zda, area), os.path.join(cfg.OUT_DIR, f"residual_coarse_{tag}.nc"))
    means.to_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{tag}.csv"), index=False)

    print(means.to_string(index=False))
    return means


def _beta_ds(beta, shape, zda, t_ref, quadratic):
    ds = xr.DataArray(
        beta.reshape((beta.shape[0],) + shape).astype(np.float32),
        dims=("coef", "latitude", "longitude"),
        coords={"latitude": zda["latitude"], "longitude": zda["longitude"]},
        name="beta",
    ).to_dataset()
    ds["beta"].attrs.update(units="J m-2", long_name="mean-field regression coefficients")
    ds.attrs["t_ref"] = t_ref
    ds.attrs["basis"] = ("1, t, t^2, sin(2pi t), cos(2pi t), sin(4pi t), cos(4pi t)"
                         if quadratic else "1, t, sin(2pi t), cos(2pi t), sin(4pi t), cos(4pi t)")
    return ds


# netCDF's default chunking picks a near-cubic shape in INDEX space, which for a
# (3652, 493, 481) field is ~(609, 71, 69) -- 609 time steps deep. Reading one
# lat/lon slice then touches 49 chunks and decompresses ~600x the data wanted.
# Cap every dimension instead, so time-slice, block and scattered-point reads are
# all within a small factor of optimal.
_CHUNK_CAP = {"time": 32, "year": 1, "doy": 64, "latitude": 64, "longitude": 64}


def _write(ds, path):
    enc = {}
    for v in ds.data_vars:
        da = ds[v]
        enc[v] = {"zlib": True, "complevel": 4,
                  "chunksizes": tuple(min(da.sizes[d], _CHUNK_CAP.get(d, da.sizes[d]))
                                      for d in da.dims)}
    ds.to_netcdf(path, encoding=enc)
    print(f"  -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("box", nargs="?", default="boxB")
    ap.add_argument("--years", help="comma-separated; default is every year in the layer store")
    ap.add_argument("--linear", action="store_true", help="drop the quadratic trend term")
    ap.add_argument("--coarse", action="store_true", help="also write the surrogate proposal field")
    ap.add_argument("--mean-only", action="store_true",
                    help="write only the mean-field coefficients and cell area (natl: the daily residual is ~22 GB)")
    ap.add_argument("--real-calendar", dest="real_calendar", action="store_true", default=None,
                    help="flat time axis on the true calendar (default for the buffer)")
    a = ap.parse_args()
    main(a.box, not a.linear,
         [int(y) for y in a.years.split(",")] if a.years else None,
         coarse=True if a.coarse else None, real_calendar=a.real_calendar, mean_only=a.mean_only)
