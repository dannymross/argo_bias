"""Estimand, sampled estimates and diagnostics for the float-simulation study.

The estimand is the domain mean Zbar_k = int Z_k dP, with P the normalised
area x time measure on a box's wet native cells. A design {x_i} of size n gives
Zhat = mean_i Z_k(x_i) and bias B = Zhat - Zbar_k. Values are J/m^2 throughout;
J_TO_GJ converts for display.

Computation only -- the summary tables and figures that the report calls live in
report.py, which imports from here.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg

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


# ----------------------------------------------------------------------------
# the truth field
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
    from osim import build_residual_field as brf
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
    from osim import build_residual_field as brf
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
    from osim import deployment as di
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
    from osim import deployment as di

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
    from osim import sample_residuals as sr
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
    from osim.continuous_sim import leap_dropped_doy
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
    from osim import sample_residuals as sr
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
        from osim.continuous_sim import leap_dropped_doy
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
    from osim import continuous_sim as cs
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
