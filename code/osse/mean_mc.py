"""Estimator, diagnostics and report figures for the OSSE sampling-bias study.

The estimand is the domain mean Zbar_k = int Z_k dP, with P the normalised
area x time measure on a box's wet native cells. A design {x_i} of size n gives
Zhat = mean_i Z_k(x_i) and bias B = Zhat - Zbar_k. Values are J/m^2 throughout;
J_TO_GJ converts for display.

This module backs `reports/osse.qmd`; it is imported, never run. The sampling it
scores comes from the continuous-seeding arm (`continuous_sim.py`), against the
residual field built by `build_residual_field.py`.

Sections below, in order: configuration and field loading; the truth field; the
real Argo record; the continuous-seeding ensemble; the sampling weight E[w]; map
scaffolding; plots.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg

J_TO_GJ = 1e-9
VEL_COMPS = {"uo": "eastward", "vo": "northward"}
M_S_TO_CM_S = 100.0
# The six floats deployed together on 27-28 Sep 2015 in the NW corner of box B:
# a common release point and time, so their tracks isolate deployment geometry.
NW_BOXB_2015 = (6902632, 6902634, 6902635, 6902633, 4902099, 4902100)


# ----------------------------------------------------------------------------
# configuration and field loading
# ----------------------------------------------------------------------------


def _field_years(box="boxB"):
    """Analysis years actually present in the residual field, not the config.

    The field is what every downstream column name is keyed to, so rebuilding it
    over a wider window must not require editing cfg.ANALYSIS_YEARS in lockstep.
    """
    p = os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc")
    if os.path.exists(p):
        with xr.open_dataset(p) as ds:
            return [int(y) for y in ds["year"].values]
    return list(cfg.ANALYSIS_YEARS)


YEARS = _field_years()
ZCOLS = [f"z_{y}" for y in YEARS]


def set_box(box):
    """Rebind the analysis-year list to `box`'s residual field.

    Every z_/zhat_/temporal_/deff_ column name is keyed to YEARS, so a run on a
    box other than boxB must switch it before anything else -- otherwise the
    columns are silently keyed to boxB's window.
    """
    global YEARS, ZCOLS
    YEARS = _field_years(box)
    ZCOLS = [f"z_{y}" for y in YEARS]
    return YEARS


def area_weights(box="boxB"):
    """Flat index of the analysis mask and the P weights on it."""
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    a = a.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return wet, a[wet] / a[wet].sum()


def load_field(box="boxB", tag=None):
    """Flattened residual field, the wet-cell index, and the P weights on it."""
    z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{tag or box}.nc"))["z"]
    wet, w = area_weights(box)
    return z.values.reshape(z.sizes["year"], z.sizes["doy"], -1), wet, w


def domain_means(zf, wet, w):
    """Zbar_k and sd_P(Z_k), plus the spatial mean on each day of year.

    Both moments are taken under P, so sd_Z is area-weighted; the cached
    build_residual_field value is unweighted and differs by well under 1%.
    """
    by_doy = np.empty((zf.shape[0], zf.shape[1]))
    m2 = np.empty_like(by_doy)
    for i in range(zf.shape[0]):
        zw = zf[i][:, wet].astype(np.float64)
        by_doy[i] = zw @ w
        m2[i] = (zw * zw) @ w
    zbar = by_doy.mean(axis=1)
    return pd.DataFrame({"year": YEARS, "zbar": zbar,
                         "sd_Z": np.sqrt(m2.mean(axis=1) - zbar**2)}), by_doy


def _sd_Z(box):
    return pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{box}.csv")).set_index("year")["sd_Z"]


def bathymetry(tag="static_vel"):
    """GLORYS deptho over the velocity domain, on the native advection grid."""
    with xr.open_dataset(os.path.join(cfg.GLORYS_DIR, "static", f"{tag}_bathy.nc")) as ds:
        return ds["deptho"].load()


def _raw_area(box):
    """Wet-cell index and unnormalized cell areas (m^2) on the analysis mask."""
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    a = a.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return wet, a[wet]


# ----------------------------------------------------------------------------
# the truth field: domain, mean field, daily dispersion
# ----------------------------------------------------------------------------


def _file_extent(pattern):
    """(lat0, lat1, lon0, lon1) of the first GLORYS file matching a glob."""
    path = sorted(glob.glob(os.path.join(cfg.GLORYS_DIR, pattern)))[0]
    with xr.open_dataset(path) as ds:
        return (float(ds["latitude"].min()), float(ds["latitude"].max()),
                float(ds["longitude"].min()), float(ds["longitude"].max()))


def domain_extents(boxes=("boxA", "boxB")):
    """Analysis boxes and the two downloaded regions, read from the files.

    Read rather than taken from the download config so the drawn extents are the
    data actually on disk.
    """
    out = {"velocity": _file_extent("velocity_daily/velocity_daily_park_*.nc"),
           "thetao": _file_extent(f"{cfg.LAYER_DIR}/ohc_layers_daily_*.nc")}
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        out[b] = (la0, la1, lo0, lo1)
    return out


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


def ohc_domain_series(box):
    """Daily domain-mean OHC and fitted mean field over `box`, 2010-2019 (J/m^2).

    Returns (dates, total, mhat) on the 365-day leap-dropped calendar. The mean
    field domain mean is phi(t)^T beta-bar with beta-bar the area-weighted mean
    coefficients; total = mhat + the area-weighted daily residual (domain_means),
    so total - mhat is the domain-mean anomaly.
    """
    zf, wet, w = load_field(box)
    _, by_doy = domain_means(zf, wet, w)
    years = _field_years(box)
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        beta = ds["beta"].values.reshape(ds.sizes["coef"], -1)
        t_ref = float(ds.attrs["t_ref"])
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"].values.ravel()
    m = np.isfinite(a)
    betabar = (beta[:, m] * (a[m] / a[m].sum())).sum(axis=1)

    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    X, _ = cfg.design_matrix(cfg.decimal_year(dates.values), quadratic=True, t_ref=t_ref)
    mhat = X @ betabar
    return dates, mhat + by_doy.reshape(-1), mhat


def mean_field_months(box, year, months=(1, 4, 7, 10)):
    """Fitted mean field evaluated at mid-month, as a (month, lat, lon) array.

    The basis is smooth in time -- day-to-day change is ~0.03 GJ/m^2 against a
    ~4 GJ/m^2 seasonal range -- so month is the finest unit at which it varies
    visibly.
    """
    from osse import build_residual_field as brf
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        shape = (ds.sizes["latitude"], ds.sizes["longitude"])
        beta = ds["beta"].values.reshape(ds.sizes["coef"], -1)
        t = np.array([np.datetime64(f"{year}-{m:02d}-15") for m in months],
                     dtype="datetime64[ns]")
        m = brf.evaluate_mean_field(beta, t, shape, quadratic=True,
                                    t_ref=ds.attrs["t_ref"])
        return xr.DataArray(m, dims=("month", "latitude", "longitude"),
                            coords={"month": list(months),
                                    "latitude": ds["latitude"], "longitude": ds["longitude"]})


def ohc_decomposition(box, year, doy):
    """OHC, fitted mean field, and residual on one day.

    OHC is reconstructed as mhat + Z rather than re-integrated from thetao: the
    residual is defined as OHC - mhat on the same mask, so the sum recovers the
    GLORYS 0-2000 m OHC exactly, with no re-read of the daily temperature.
    """
    from osse import build_residual_field as brf
    t = _doy_date(year, doy)
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc")) as ds:
        z = ds["z"].sel(year=int(year)).isel(doy=int(doy)).load()
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        m = brf.evaluate_mean_field(
            ds["beta"].values.reshape(ds.sizes["coef"], -1),
            np.array([t], dtype="datetime64[ns]"),
            (ds.sizes["latitude"], ds.sizes["longitude"]),
            quadratic=True, t_ref=ds.attrs["t_ref"])[0]
    mhat = xr.DataArray(m, coords=z.coords, dims=z.dims).where(np.isfinite(z))
    return (mhat + z).rename("ohc"), mhat.rename("mhat"), z, t


def _doy_date(year, doy):
    """Calendar date of a day index on the 365-day leap-stripped calendar."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return days[~((days.month == 2) & (days.day == 29))][int(doy)]


def _velocity_year(box, year):
    """Park-depth (u, v) over one year on the leap-stripped 365-day calendar.

    Interpolated to PARK_DEPTH between the two archived levels, which is what
    Parcels does to the same files when it advects the floats.
    """
    from osse import deployment as di
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    files = [os.path.join(cfg.GLORYS_DIR, "velocity_daily",
                          f"velocity_daily_park_{year}_{m:02d}.nc") for m in range(1, 13)]
    with xr.open_mfdataset(files, combine="by_coords") as ds:
        d = (ds[["uo", "vo"]]
             .sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
             .interp(depth=di.PARK_DEPTH).load())
    t = d["time"].dt
    return d.isel(time=~((t.month == 2) & (t.day == 29)).values)


def combined_daily_spread(boxes=("boxB", "boxA"), path=None):
    """sd_P(Z) per (year, doy) over the union of both boxes under one area measure."""
    path = path or os.path.join(cfg.OUT_DIR, "daily_spread_combined.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    fields = {}
    for b in boxes:
        z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{b}.nc"))["z"]
        zf = z.values.reshape(z.sizes["year"], z.sizes["doy"], -1)
        wet, area = _raw_area(b)
        fields[b] = (zf, wet, area, [int(y) for y in z["year"].values])
    years = sorted(set.intersection(*[set(f[3]) for f in fields.values()]))
    w = np.concatenate([fields[b][2] for b in boxes])
    w = w / w.sum()
    ndoy = fields[boxes[0]][0].shape[1]
    rows = []
    for y in years:
        vals = np.concatenate(
            [fields[b][0][fields[b][3].index(y)][:, fields[b][1]].astype(np.float64)
             for b in boxes], axis=1)
        m1 = vals @ w
        rows.append(pd.DataFrame({"year": y, "doy": np.arange(ndoy), "mean": m1,
                                  "sd": np.sqrt(np.clip((vals * vals) @ w - m1**2, 0, None))}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def combined_quantile_days(qs=(0.05, 0.5, 0.95), boxes=("boxB", "boxA")):
    """The (year, doy) whose combined spatial sd is nearest each quantile."""
    s = combined_daily_spread(boxes)
    sd = s["sd"].to_numpy()
    return pd.DataFrame([
        {"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]), "sd": sd[i]}
        for q in qs for i in [int(np.argmin(np.abs(sd - np.quantile(sd, q))))]])


def combined_velocity_daily_spread(comp="uo", boxes=("boxB", "boxA"), path=None):
    """Spatial sd of a velocity component per (year, doy) over both boxes (cm/s)."""
    path = path or os.path.join(cfg.OUT_DIR, f"daily_vel_spread_combined_{comp}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    areas = {b: _raw_area(b) for b in boxes}
    w = np.concatenate([areas[b][1] for b in boxes])
    w = w / w.sum()
    years = sorted(set.intersection(*[set(_field_years(b)) for b in boxes]))
    rows = []
    for y in years:
        cols, ndoy = [], None
        for b in boxes:
            d = _velocity_year(b, y)[comp]
            ndoy = d.sizes["time"]
            cols.append(d.values.reshape(ndoy, -1)[:, areas[b][0]].astype(np.float64))
        vals = np.concatenate(cols, axis=1) * M_S_TO_CM_S
        m1 = vals @ w
        rows.append(pd.DataFrame({"year": y, "doy": np.arange(ndoy), "mean": m1,
                                  "sd": np.sqrt(np.clip((vals * vals) @ w - m1**2, 0, None))}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def combined_velocity_quantile_days(comp="uo", qs=(0.05, 0.5, 0.95),
                                    boxes=("boxB", "boxA")):
    """The (year, doy) whose combined velocity spatial sd is nearest each quantile."""
    s = combined_velocity_daily_spread(comp, boxes)
    sd = s["sd"].to_numpy()
    return pd.DataFrame([
        {"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]), "sd": sd[i]}
        for q in qs for i in [int(np.argmin(np.abs(sd - np.quantile(sd, q))))]])


# ----------------------------------------------------------------------------
# the real Argo record
# ----------------------------------------------------------------------------


def clean_argo(path=None, min_spacing=10, max_gap=90, min_span=90):
    """QC-cleaned real Argo profile record (float_id, lat, lon, date, year, doy).

    Applied to each float's FULL global record: (1) thin to >= min_spacing days
    between kept profiles (drops near-duplicate surface fixes); (2) drop the whole
    float if any consecutive gap exceeds max_gap days OR its total span is under
    min_span days. doy is on the 365-day leap-dropped calendar (Feb 29 dropped).
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d.dropna(subset=["lat", "lon", "date"]).sort_values(["float_id", "date"])

    # Greedy thin: keep a profile only if >= min_spacing days since the last KEPT one
    # (per float), so runs of sub-cycle fixes are thinned to ~min_spacing rather than
    # collapsed to a single point.
    fid = pd.factorize(d["float_id"].to_numpy())[0]
    day = (d["date"] - d["date"].min()).dt.total_seconds().to_numpy() / 86400.0
    keep = np.zeros(len(d), dtype=bool)
    last = -1e18
    for i in range(len(d)):
        if i and fid[i] != fid[i - 1]:
            last = -10 ** 9
        if day[i] - last >= min_spacing:
            keep[i] = True
            last = day[i]
    d = d[keep]
    gap = d.groupby("float_id")["date"].diff().dt.total_seconds() / 86400.0
    g = d.groupby("float_id")
    max_gap_f = gap.groupby(d["float_id"]).transform("max")      # NaN for single-profile floats
    span_f = (g["date"].transform("max") - g["date"].transform("min")).dt.total_seconds() / 86400.0
    d = d[(max_gap_f <= max_gap) & (span_f >= min_span)]         # NaN comparisons -> False -> dropped

    dt = d["date"].dt
    doy0 = (dt.dayofyear - 1).to_numpy()
    leap = dt.is_leap_year.to_numpy()
    d = d.assign(year=dt.year.to_numpy(),
                 doy=np.where(leap & (doy0 >= 60), doy0 - 1, doy0))
    d.loc[leap & (dt.month.to_numpy() == 2) & (dt.day.to_numpy() == 29), "doy"] = -1
    return d[d["doy"] >= 0]


def argo_region_summary(box="boxB", years=None, path=None, cache=None):
    """Real Argo occupancy of the box, per analysis year; cached.

    Standing stock is the mean number of distinct floats reporting within a
    10-day window, which is what compares against the virtual array: real floats
    cycle on their own schedules, so a straight profile count would confound
    array size with cycling cadence.
    """
    years = years or _field_years(box)
    cache = cache or os.path.join(cfg.OUT_DIR, f"argo_region_{box}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    from osse import deployment as di

    d = pd.read_csv(path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv"),
                    usecols=["float_id", "lat", "lon", "date", "ohc_2000"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)
          & d["ohc_2000"].notna()]
    d["year"] = d["date"].dt.year
    d = d[d["year"].isin(years)]
    d["bin"] = (d["date"].dt.dayofyear - 1) // di.CYCLE_DAYS

    stock = (d.groupby(["year", "bin"])["float_id"].nunique()
              .groupby("year").mean())
    out = (d.groupby("year")
             .agg(argo_profiles=("float_id", "size"), argo_floats=("float_id", "nunique"))
             .join(stock.rename("argo_stock")).reset_index())
    out.to_csv(cache, index=False)
    return out


def argo_bias_by_year(box, cleaned=None, path=None):
    """Realized bias of the REAL Argo array sampling the GLORYS anomaly field:
    B_k = mean over that box's year-k Argo profiles of Z_k, minus Zbar_k (J/m^2).

    Same estimator as the simulated arm, on the QC-cleaned Argo profile positions
    (surface locations; see clean_argo). Returns a Series year -> B_argo for the
    years present in both the Argo record and the residual field.
    """
    from osse import sample_residuals as sr
    d = clean_argo(path) if cleaned is None else cleaned
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)]

    z = sr.load_residual_field(box)
    years = [int(y) for y in z["year"].values]
    sampled, _ = sr.gather(d, z)
    sampled = sampled[sampled["year"].isin(years)].copy()
    col = {y: j for j, y in enumerate(years)}
    zc = sampled[[f"z_{y}" for y in years]].to_numpy()
    sampled["z_own"] = zc[np.arange(len(sampled)),
                          [col[y] for y in sampled["year"].to_numpy()]]
    zf, wet, w = load_field(box)
    zbar = domain_means(zf, wet, w)[0].set_index("year")["zbar"]
    ghat = sampled.groupby("year")["z_own"].mean()
    return (ghat - zbar).dropna()


def raw_deployments(years=(2010, 2019), domain="velocity", boxes=("boxA", "boxB"), path=None):
    """First profile (deployment) of every float in argo_ohc.csv, NOT QC-cleaned.

    Restricted to deployment years in [years[0], years[1]] and to `domain`:
    'velocity' the advection domain, 'analysis' the union of the analysis boxes.
    Deployment is the earliest profile on record, so a float first observed before
    the file window is not resolved as deployed in-window -- see the report note.
    Returns float_id, lat, lon, date, year.
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d.dropna(subset=["lat", "lon", "date"])
    dep = d.loc[d.groupby("float_id")["date"].idxmin()].copy()
    dep["year"] = dep["date"].dt.year
    e = domain_extents(boxes)
    if domain == "velocity":
        la0, la1, lo0, lo1 = e["velocity"]
        m = dep["lat"].between(la0, la1) & dep["lon"].between(lo0, lo1)
    else:
        m = np.zeros(len(dep), bool)
        for b in boxes:
            la0, la1, lo0, lo1 = e[b]
            m = m | (dep["lat"].between(la0, la1) & dep["lon"].between(lo0, lo1)).to_numpy()
    dep = dep[m & dep["year"].between(years[0], years[1])]
    return dep[["float_id", "lat", "lon", "date", "year"]].reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between arrays of lat/lon points."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _decorrelation_km(div, seed=0, n=6000):
    """Median great-circle distance between random pairs of the array's profile
    positions -- the separation two independent floats reach, i.e. the no-skill floor."""
    pts = div[["rlat", "rlon"]].drop_duplicates().to_numpy()
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(pts), n), r.integers(0, len(pts), n)
    return float(np.median(_haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1])))


def trajectory_divergence(tol_days=6, max_days=None, tracks=None, pool=None, cleaned=None):
    """Separation between each real Argo float and its GLORYS-park sim twin.

    The argo-actual full-trajectory run advects each real deployment verbatim; this
    matches every real profile (after deployment) to the nearest sim position in
    time (within tol_days) and returns the great-circle separation vs days since
    deployment. Isolates real-vs-reanalysis drift: a park-only sim that tracked
    real floats would stay small; one that diverges to the array's own scale shows
    the residual bias gap is advection error, which a dive cycle cannot fix.
    """
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

    rows = []
    for pr in pl.itertuples():
        st = simg.get(int(pr.seed_id))
        rt = realg.get(int(pr.float_id))
        if st is None or rt is None:
            continue
        rt = rt[rt["date"] >= pr.date]
        if len(rt) < 2 or len(st) < 2:
            continue
        s_t = st["time"].to_numpy()
        r_t = rt["date"].to_numpy()
        j = np.clip(np.searchsorted(s_t, r_t), 1, len(s_t) - 1)
        pick = np.where(np.abs(r_t - s_t[j - 1]) <= np.abs(s_t[j] - r_t), j - 1, j)
        gap = np.abs((s_t[pick] - r_t) / np.timedelta64(1, "D"))
        ok = gap <= tol_days
        if not ok.any():
            continue
        rows.append(pd.DataFrame({
            "seed_id": int(pr.seed_id), "float_id": int(pr.float_id),
            "elapsed_days": (r_t[ok] - np.datetime64(pr.date)) / np.timedelta64(1, "D"),
            "sep_km": _haversine_km(rt["lat"].to_numpy()[ok], rt["lon"].to_numpy()[ok],
                                    st["lat"].to_numpy()[pick][ok],
                                    st["lon"].to_numpy()[pick][ok]),
            "rlat": rt["lat"].to_numpy()[ok], "rlon": rt["lon"].to_numpy()[ok]}))
    div = pd.concat(rows, ignore_index=True)
    return div[div["elapsed_days"] <= max_days] if max_days else div


def trajectory_divergence_summary(div=None, marks=(10, 30, 90, 180, 365, 730)):
    """Median (IQR) sim-vs-real separation and n at a few elapsed-time marks."""
    div = trajectory_divergence() if div is None else div
    rows = []
    for m in marks:
        w = div[(div["elapsed_days"] > m - 15) & (div["elapsed_days"] <= m + 15)]["sep_km"]
        if len(w):
            rows.append({"days since deployment": m, "median sep (km)": w.median(),
                         "IQR (km)": w.quantile(0.75) - w.quantile(0.25), "n": len(w)})
    out = pd.DataFrame(rows)
    out.attrs["decorrelation_km"] = _decorrelation_km(div)
    return out


# ----------------------------------------------------------------------------
# the continuous-seeding ensemble
# ----------------------------------------------------------------------------


def load_continuous(tag="_pilot"):
    return pd.read_csv(os.path.join(cfg.OUT_DIR, f"estimates_continuous{tag}.csv"))


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


def uniform_reference_ensemble(est, R=2000, seed=0):
    """Large n-matched uniform null: R iid draws from P per (box, year), each at the
    median advective sample size for that cell. E[B]=0, so the ensemble mean is a
    precise zero reference (its 95% CI covers zero) and its 5-95% spread is the
    single-deployment scatter of a uniform array. Returns box, year, B_unif (R rows
    per group). Decoupled from the 100 advective reps, which are limited by the
    advection cost; the null is not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for box in sorted(est["box"].unique()):
        zf, wet, w = load_field(box)
        yidx = {y: j for j, y in enumerate(_field_years(box))}
        zbar = domain_means(zf, wet, w)[0].set_index("year")["zbar"]
        cw = np.cumsum(w)
        g = est[est["box"] == box]
        for y in sorted(int(v) for v in g["year"].unique()):
            n = int(np.median(g.loc[g["year"] == y, "n"].to_numpy()))
            cells = wet[np.searchsorted(cw, rng.random(R * n)).clip(max=len(w) - 1)].reshape(R, n)
            d = rng.integers(0, cfg.DAYS_PER_YEAR, (R, n))
            b = zf[yidx[y]][d, cells].mean(axis=1) - zbar[y]
            rows += [(box, y, float(v)) for v in b]
    return pd.DataFrame(rows, columns=["box", "year", "B_unif"])


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


def argo_actual_profile_bias(boxes=("boxB", "boxA")):
    """Per-profile bias for every argo-actual (GLORYS park-advection of real
    deployments) float, tagged by deployment-year cohort.

    For an in-box profile in year k, delta = Z(x_i) - Zbar_k (J/m2), unnormalized:
    the single-profile bias, comparable across years. Age is days since the float's
    real deployment and runs continuously through any spell outside the domain
    (out-of-box profiles carry no bias), so a float's in-box points join across
    those gaps and across year boundaries over its whole life. Returns float_id,
    box, deploy_year, year, age, delta.
    """
    from osse.continuous_sim import leap_dropped_doy
    pool = pd.read_csv(os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"))
    f2s = pool.set_index("seed_id")["float_id"]
    dep = pd.to_datetime(pool.set_index("seed_id")["date"], format="ISO8601",
                         utc=True).dt.tz_localize(None)
    tr = pd.read_csv(os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
    tr = tr[tr["cycle"] > 0].copy()
    tr["float_id"] = tr["seed_id"].map(f2s).astype(int)
    tr["deploy_year"] = tr["seed_id"].map(dep).dt.year
    tr["age"] = (tr["time"] - tr["seed_id"].map(dep)).dt.total_seconds() / 86400.0
    yr, doy = leap_dropped_doy(tr["time"].to_numpy())
    tr["year"], tr["doy"], tr["pnum"] = yr, doy, 0
    tr = tr[tr["doy"] >= 0]
    dep_by_fid = tr.groupby("float_id")["deploy_year"].first()
    cols = ["float_id", "lat", "lon", "year", "doy", "age", "pnum"]
    out = []
    for b in boxes:
        d = _delta_frame(tr[cols], b)
        d["box"] = b
        d["deploy_year"] = d["float_id"].map(dep_by_fid)
        out.append(d[["float_id", "box", "deploy_year", "year", "age", "delta"]])
    return pd.concat(out, ignore_index=True)


def _delta_frame(d, box):
    """Attach per-profile delta = Z(x_i) - Zbar_k to positions with float_id, lat,
    lon, year, doy, age, pnum. Restricts to `box` support and the field's years."""
    from osse import sample_residuals as sr
    set_box(box)
    (la0, la1) = cfg.BOXES[box]["lat_bounds"]; (lo0, lo1) = cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)]
    z = sr.load_residual_field(box)
    zyears = [int(y) for y in z["year"].values]
    d = d[d["year"].isin(zyears)]
    sampled, _ = sr.gather(d, z)
    col = {y: j for j, y in enumerate(zyears)}
    zc = sampled[[f"z_{y}" for y in zyears]].to_numpy()
    sampled["z_own"] = zc[np.arange(len(sampled)), [col[y] for y in sampled["year"].to_numpy()]]
    zf, wet, w = load_field(box)
    dm = domain_means(zf, wet, w)[0].set_index("year")
    sampled["delta"] = sampled["z_own"] - sampled["year"].map(dm["zbar"])
    sampled["delta_std"] = sampled["delta"] / sampled["year"].map(dm["sd_Z"])
    return sampled[["float_id", "year", "doy", "lat", "lon", "age", "pnum",
                    "delta", "delta_std"]].reset_index(drop=True)


def float_bias_profile(float_ids, box="boxB", source="real", cleaned=None):
    """Per-profile single-profile bias delta = Z(x_i) - Zbar_k for named floats.

    source='real': QC-cleaned real Argo profiles (pnum = ordinal in the float's full
    cleaned record). source='sim': GLORYS 1000 m park-advection sim-twin positions
    (argo-actual arm, matched via the deployment pool; pnum = 10-day cycle). Both are
    restricted to `box` (the field's support), so out-of-box profiles drop out.
    """
    ids = [int(f) for f in float_ids]
    if source == "real":
        d = (clean_argo() if cleaned is None else cleaned).copy()
        d = d[d["float_id"].isin(ids)].sort_values(["float_id", "date"])
        d["age"] = (d["date"] - d.groupby("float_id")["date"].transform("min")).dt.total_seconds() / 86400.0
        d["pnum"] = d.groupby("float_id").cumcount() + 1
    elif source == "sim":
        from osse.continuous_sim import leap_dropped_doy
        pool = pd.read_csv(os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"))
        f2s = pool.set_index("seed_id")["float_id"]
        tr = pd.read_csv(os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
        tr = tr[tr["seed_id"].isin(pool.loc[pool["float_id"].isin(ids), "seed_id"])].copy()
        tr["float_id"] = tr["seed_id"].map(f2s).astype(int)
        tr = tr.sort_values(["float_id", "time"])
        tr["age"] = (tr["time"] - tr.groupby("float_id")["time"].transform("min")).dt.total_seconds() / 86400.0
        yr, doy = leap_dropped_doy(tr["time"].to_numpy())
        tr["year"], tr["doy"], tr["pnum"] = yr, doy, tr["cycle"].astype(int)
        d = tr[(tr["cycle"] > 0) & (tr["doy"] >= 0)]     # drop release record and Feb-29
    else:
        raise ValueError(f"source must be 'real' or 'sim', got {source!r}")
    return _delta_frame(d, box)


def domain_exit_fates(tracks_path=None, pool_path=None, end="2019-12-26", cycle_days=10.0):
    """Classify each virtual float's fate from its FULL park-advection track.

    A float leaves via one of three exits: deleted for advecting out of the
    velocity domain (early, before lifespan and before the record ends), retired
    at its drawn lifespan, or still alive at the record end. Classification is by
    time, not boundary proximity (the out-of-bounds step itself is unrecorded), so
    the exit point returned is the last in-domain 10-day position. Returns per-float
    seed_id, float_id, last lat/lon/time, age_d, lifespan, fate.
    """
    tracks_path = tracks_path or os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv")
    pool_path = pool_path or os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv")
    pool = pd.read_csv(pool_path)
    life = pool.set_index("seed_id")["lifespan"]
    dep = pd.to_datetime(pool.set_index("seed_id")["date"], format="ISO8601",
                         utc=True).dt.tz_localize(None)
    fid = pool.set_index("seed_id")["float_id"]
    tr = pd.read_csv(tracks_path, parse_dates=["time"])
    tr = tr[tr["cycle"] > 0].sort_values(["seed_id", "time"])
    last = tr.groupby("seed_id").tail(1).set_index("seed_id")
    age = (last["time"] - dep.reindex(last.index)).dt.total_seconds() / 86400.0
    lf = life.reindex(last.index)
    endt = pd.Timestamp(end)
    retired = age >= lf - cycle_days
    survived = (~retired) & (last["time"] >= endt - pd.Timedelta(days=cycle_days))
    fate = np.where(retired, "lifespan", np.where(survived, "record_end", "domain_exit"))
    return pd.DataFrame({"seed_id": last.index, "float_id": fid.reindex(last.index).to_numpy(),
                         "lat": last["lat"].to_numpy(), "lon": last["lon"].to_numpy(),
                         "time": last["time"].to_numpy(), "age_d": age.to_numpy(),
                         "lifespan": lf.to_numpy(), "fate": fate})


def domain_exit_table(fates=None):
    """Counts and fractions of each fate (domain_exit / lifespan / record_end)."""
    f = domain_exit_fates() if fates is None else fates
    t = f["fate"].value_counts().reindex(["domain_exit", "lifespan", "record_end"]).fillna(0).astype(int)
    return pd.DataFrame({"fate": t.index, "floats": t.to_numpy(),
                         "percent": (100 * t / len(f)).round(1).to_numpy()})


# ----------------------------------------------------------------------------
# the sampling weight E[w]
# ----------------------------------------------------------------------------


def _profile_weights(box, tag):
    """Native-cell-tagged profiles, each carrying weight 1/(n R) of its (year, rep).

    Profiles are binned to the containing 1/12 deg cell and restricted to the wet
    analysis mask -- the same support the estimator's w and the field live on, so n
    is the on-mask count per realization. Summing wt over any set of (year, cell or
    doy) bins gives E[w] = mean_rep (fraction of that rep's profiles in the bin),
    since a bin a rep misses contributes 0 to the mean. Returns (prof, lat, lon, p)
    with p = A_c / sum(A) the native uniform area weight (NaN off-mask).
    """
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    lat, lon, A = a["latitude"].values, a["longitude"].values, a.values
    from osse import continuous_sim as cs
    prof = cs.read_profiles(tag, ["rep", "box", "year", "lat", "lon", "doy"])
    # doy = -1 marks Feb 29, which has no field slot; it must not become a
    # day-of-year bin, nor count toward the per-realization normalizer n.
    prof = prof[(prof["box"] == box) & (prof["doy"] >= 0)].copy()
    ii = cfg.containing_index(prof["lat"].to_numpy(), lat)
    jj = cfg.containing_index(prof["lon"].to_numpy(), lon)
    wet = (ii >= 0) & (jj >= 0)
    wet[wet] = np.isfinite(A[ii[wet], jj[wet]])
    prof = prof.loc[wet].assign(ilat=ii[wet], ilon=jj[wet])
    n = prof.groupby(["year", "rep"])["lat"].transform("size")
    R = prof.groupby("year")["rep"].transform("nunique")
    prof["wt"] = 1.0 / (n * R)
    return prof, lat, lon, A / np.nansum(A)


def sampling_weight_space(box, tag="_pilot"):
    """E[w] spatial marginal and the uniform area reference p, on the NATIVE grid.

    No coarsening: the 1/12 deg cells the residual field and the inner product
    B = <Z, w - p> live on. Returns (Ew, lat, lon, p): Ew a Series indexed by
    (year, ilat, ilon) summing to 1 per year; p the (nlat, nlon) area weight.
    """
    prof, lat, lon, p = _profile_weights(box, tag)
    Ew = prof.groupby(["year", "ilat", "ilon"])["wt"].sum()
    return Ew, lat, lon, p


def sampling_weight_ratio(box, tag="_pilot"):
    """Per-year array of E[w]_c / p_c over ALL wet native cells (empty cells -> 0).

    The over-/under-sampling factor per cell: 1 is the uniform weight, >1
    over-sampled, <1 under-sampled. Returns {year: ndarray over wet cells}.
    """
    Ew, lat, lon, p = sampling_weight_space(box, tag)
    wi, wj = np.where(np.isfinite(p))
    wflat, pw = wi * p.shape[1] + wj, p[wi, wj]
    out = {}
    for y in sorted(int(y) for y in Ew.index.get_level_values("year").unique()):
        w = Ew.xs(y, level="year")
        f = (w.index.get_level_values("ilat").to_numpy() * p.shape[1]
             + w.index.get_level_values("ilon").to_numpy())
        r = np.zeros(len(wflat))
        pos = np.searchsorted(wflat, f)
        r[pos] = w.to_numpy() / pw[pos]
        out[y] = r
    return out


# ----------------------------------------------------------------------------
# map scaffolding
# ----------------------------------------------------------------------------


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


def _both_box_extent(boxes=("boxB", "boxA"), pad=1.0):
    """Map extent covering every box, padded, for a shared both-box view."""
    la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes)
    la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes)
    lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes)
    lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes)
    return [lo0 - pad, lo1 + pad, la0 - pad, la1 + pad]


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


# ----------------------------------------------------------------------------
# plots
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
