"""Build the native-grid OHC residual field for the OSSE permutation test.

OHC (0-2000 m) on the GLORYS 1/12 deg grid, minus a per-cell mean field fitted
from the 1995-2019 monthly record (quadratic trend + annual and semiannual
harmonics). The residual Z is the field the test samples; the analysis grid is
GLORYS' own grid, so the simple-function representation of theory.tex is exact
and no quadrature error term arises.

Run from the repo root: ``python code/osse/build_residual_field.py [box]``.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ohc
from osse import osse_config as cfg


def _ohc_from_files(files, box):
    """OHC 0-2000 m over the box, concatenated over time, Feb 29 removed."""
    out = []
    for f in sorted(files):
        ds = cfg.box_slice(xr.open_dataset(f), box)
        out.append(ohc.truth_ohc_field(ds, depths=(cfg.ZMAX,))[f"ohc_{cfg.ZMAX}"].compute())
        ds.close()
    field = xr.concat(out, dim="time").sortby("time").to_dataset(name="ohc")
    return cfg.drop_leap_day(field)


def load_static(box):
    static = os.path.join(cfg.GLORYS_DIR, "static")
    bathy = cfg.box_slice(xr.open_dataset(os.path.join(static, "static_bathy.nc")), box)
    coords = cfg.box_slice(xr.open_dataset(os.path.join(static, "static_coords.nc")), box)
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


def main(box="boxB", quadratic=True):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    area, wet = load_static(box)

    monthly_files = [
        f
        for y in range(cfg.FIT_START_YEAR, cfg.FIT_END_YEAR + 1)
        for f in glob.glob(os.path.join(cfg.GLORYS_DIR, "thetao_zos_monthly",
                                        f"thetao_zos_monthly_{y}_*.nc"))
    ]
    print(f"[{box}] mean field: {len(monthly_files)} monthly files "
          f"({cfg.FIT_START_YEAR}-{cfg.FIT_END_YEAR}), quadratic={quadratic}")
    monthly = _ohc_from_files(monthly_files, box)
    shape = monthly["ohc"].shape[1:]
    beta, t_ref, ok = fit_mean_field(monthly, quadratic=quadratic)
    print(f"  {ok.sum():,}/{ok.size:,} cells fitted, {beta.shape[0]} coefficients")

    daily_files = [
        f
        for y in cfg.ANALYSIS_YEARS
        for f in glob.glob(os.path.join(cfg.GLORYS_DIR, "thetao_daily", f"thetao_daily_{y}_*.nc"))
    ]
    print(f"[{box}] residual field: {len(daily_files)} daily files")
    daily = _ohc_from_files(daily_files, box)
    mhat = evaluate_mean_field(beta, daily["time"].values, shape, quadratic=quadratic, t_ref=t_ref)
    z = (daily["ohc"] - xr.DataArray(mhat, coords=daily["ohc"].coords, dims=daily["ohc"].dims))
    z = z.where(wet)

    # (year, doy, lat, lon): the layout the day-of-year-aligned gather indexes.
    years = daily["time"].dt.year.values
    zda = xr.DataArray(
        np.stack([z.values[years == y] for y in cfg.ANALYSIS_YEARS]).astype(np.float32),
        dims=("year", "doy", "latitude", "longitude"),
        coords={"year": list(cfg.ANALYSIS_YEARS), "doy": np.arange(cfg.DAYS_PER_YEAR),
                "latitude": z["latitude"], "longitude": z["longitude"]},
        name="z",
    )
    zda.attrs.update(units="J m-2", long_name="OHC 0-2000 m residual (OHC minus fitted mean field)")

    # The bathymetry mask alone is not the data support: a cell with
    # 2000 < deptho < 2225 m passes deptho > 2000 but has no model level below
    # 2000 m to interpolate the bottom segment against, so its OHC is NaN. Take
    # the intersection so the area weights and the field share one mask --
    # otherwise Zbar normalises by more area than the field actually covers.
    valid = np.isfinite(zda).all(("year", "doy"))
    lost = 1.0 - float(area.where(valid).sum() / area.sum())
    print(f"  data-support mask drops {int((~valid & np.isfinite(area)).sum())} bathymetrically "
          f"wet cells ({lost:.3%} of box area)")
    area = area.where(valid)
    zda = zda.where(valid)

    w = area.fillna(0.0)
    means = pd.DataFrame({
        "year": list(cfg.ANALYSIS_YEARS),
        "zbar": ((zda * w).sum(("latitude", "longitude")) / w.sum()).mean("doy").values,
        "sd_Z": zda.std(("doy", "latitude", "longitude")).values,
    })

    tag = f"{box}{'' if quadratic else '_lin'}"
    _write(zda.to_dataset(), os.path.join(cfg.OUT_DIR, f"residual_field_{tag}.nc"))
    _write(area.to_dataset(), os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))
    _write(_beta_ds(beta, shape, zda, t_ref, quadratic), os.path.join(cfg.OUT_DIR, f"mean_field_coef_{tag}.nc"))
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


def _write(ds, path):
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"  -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "boxB")
