"""Grid estimate of the Argo profile intensity q-hat from the resample ensemble.

On cells c of `deg` x `months` calendar months (working default 2 deg x 2 months),

    q_hat(c) = (1/R) sum_r sum_{sim profiles p in c} w_p / mu(c & S),

with mu area x days and S the float support (deptho >= park depth). w_p = 1 gives
the nominal 10-day profile intensity; w_p = rho_k, the eligible-profile rate of the
real float the simulated float was resampled from, gives the intensity of eligible
profiles (thinning theorem: retention independent of position given the float).
rho_age uses rho_k within yearly age bins. MC standard errors come from the
spread over replicates.

    python code/osim/intensity.py [--deg 2 --months 2] [--shift]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import continuous_sim as cs
from osim import deployment as di
from osim import osim_config as cfg

R_EARTH = 6371008.8
BOX_LAT = (22.0, 46.0)       # grid aligned to the box edges (22, 34 N; -60, -40 E),
BOX_LON = (-60.0, -40.0)     # plus a one-cell margin
PERIOD = ("2015-01", "2019-12")   # analysis years; 2010-2014 deployments are the ramp-up
MIN_GAP_DAYS = 2.0
MIN_DEPTH = 1900.0
AGE_BIN_DAYS = 365.25


def eligible(d, min_gap=MIN_GAP_DAYS, min_depth=MIN_DEPTH):
    """Real profiles the estimator uses: greedy per-float thinning to >= min_gap days
    (drops duplicate fixes) then max_depth >= min_depth."""
    d = d.sort_values(["float_id", "date"])
    fid = d["float_id"].to_numpy()
    day = (d["date"] - d["date"].min()).dt.total_seconds().to_numpy() / 86400.0
    keep = np.zeros(len(d), bool)
    last = -np.inf
    for i in range(len(d)):
        if i and fid[i] != fid[i - 1]:
            last = -np.inf
        if day[i] - last >= min_gap:
            keep[i] = True
            last = day[i]
    return d[keep & (d["max_depth"].to_numpy() >= min_depth)]


def eligibility_rates(pool, d, cycle_days=cs.CYCLE_DAYS, age_bin=AGE_BIN_DAYS):
    """Per pool entry: rho = eligible real profiles over the simulated life divided by
    nominal cycles floor(lifespan/cycle); rho_age[j] the same within age bin j."""
    e = eligible(d[d["float_id"].isin(pool["float_id"])])
    e = e.merge(pool[["pool_idx", "float_id", "date", "lifespan"]].rename(columns={"date": "t0"}),
                on="float_id")
    age = (e["date"] - e["t0"]).dt.total_seconds() / 86400.0
    e = e[(age > 0.5 * cycle_days) & (age <= e["lifespan"] + 0.5 * cycle_days)]
    age = age.loc[e.index]
    nb = int(np.ceil(pool["lifespan"].max() / age_bin)) + 1
    n_elig = e.groupby("pool_idx").size().reindex(pool["pool_idx"], fill_value=0).to_numpy()
    n_nom = np.floor(pool["lifespan"].to_numpy() / cycle_days)
    rho = np.divide(n_elig, n_nom, out=np.zeros(len(pool)), where=n_nom > 0)

    jb = np.minimum((age.to_numpy() // age_bin).astype(int), nb - 1)
    ea = np.zeros((len(pool), nb))
    np.add.at(ea, (e["pool_idx"].to_numpy(), jb), 1.0)
    edges = np.arange(nb + 1) * age_bin
    life = pool["lifespan"].to_numpy()[:, None]
    nom = np.floor(np.clip(np.minimum(life, edges[1:]) - edges[:-1], 0, None) / cycle_days)
    rho_age = np.divide(ea, nom, out=np.zeros_like(ea), where=nom > 0)
    return pd.DataFrame({"pool_idx": pool["pool_idx"], "rho": rho}), rho_age


def profile_rho_age(p, rho_age, cycle_days=cs.CYCLE_DAYS, age_bin=AGE_BIN_DAYS):
    """Eligible-profile rate of each simulated profile's source float in its age bin."""
    jb = np.minimum((p["cycle"].to_numpy() * cycle_days // age_bin).astype(int), rho_age.shape[1] - 1)
    return rho_age[p["pool_idx"].to_numpy().astype(int), jb]


def box_year_counts(tag="_argo_resample", boxes=("boxA", "boxB"), years=cfg.ANALYSIS_YEARS,
                    sources=("deploy",)):
    """Calibration: expected eligible in-box profiles per replicate and box-year (sim
    nominal profiles weighted by rho_age) against the real eligible count, with the real
    count's mid-rank percentile among replicates."""
    from osim import argo_meta as am
    files = sorted(glob.glob(cs.profile_path(tag, chunk="*")))
    if not files:
        raise FileNotFoundError(cs.profile_path(tag, chunk="*"))
    d = am.read_argo()
    _, rho_age = eligibility_rates(cs.read_resample_pool(), d)
    sim = []
    for f in files:
        p = pd.read_parquet(f, columns=["cycle", "pool_idx", "source", "box", "year"])
        p = p[p["source"].isin(sources) & p["box"].isin(boxes) & p["year"].isin(years)]
        sim.append(p.assign(w=profile_rho_age(p, rho_age), rep=int(f.rsplit("_c", 1)[1].split(".")[0]))
                   .groupby(["rep", "box", "year"])["w"].agg(nominal="size", eligible="sum"))
    sim = pd.concat(sim).reset_index()
    e = eligible(d)
    real = pd.concat([e[di.in_box(e, b)].assign(box=b) for b in boxes])
    real = (real.assign(year=real["date"].dt.year).query("year in @years")
            .groupby(["box", "year"]).size().rename("real"))
    s = sim.merge(real.reset_index(), on=["box", "year"], how="left").fillna({"real": 0})
    s["below"] = (s["eligible"] < s["real"]) + 0.5 * (s["eligible"] == s["real"])
    out = s.groupby(["box", "year"]).agg(sim_nominal=("nominal", "mean"), sim_eligible=("eligible", "mean"),
                                         sim_sd=("eligible", "std"), real=("real", "first"),
                                         pct=("below", "mean"))
    out["pct"] *= 100
    return out.reset_index(), sim


def support_area(lat_edges, lon_edges, min_depth=di.PARK_DEPTH):
    """Area (m2) of native bathymetry cells with deptho >= min_depth, summed into the
    (lat, lon) grid cells by native-cell centre."""
    with xr.open_dataset(cs.BATHY_PATH) as ds:
        dep = ds["deptho"].values
        lat, lon = ds["latitude"].values.astype(float), ds["longitude"].values.astype(float)
    le, oe = cfg.cell_edges(lat), cfg.cell_edges(lon)
    a = (R_EARTH ** 2 * np.deg2rad(np.diff(oe))[None, :]
         * np.diff(np.sin(np.deg2rad(le)))[:, None])
    a = np.where(np.isfinite(dep) & (dep >= min_depth), a, 0.0)
    i = np.searchsorted(lat_edges, lat, side="right") - 1
    j = np.searchsorted(lon_edges, lon, side="right") - 1
    oki, okj = (i >= 0) & (i < len(lat_edges) - 1), (j >= 0) & (j < len(lon_edges) - 1)
    out = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))
    np.add.at(out, (i[oki][:, None], j[okj][None, :]), a[np.ix_(oki, okj)])
    return out


def grid(deg=2.0, months=2, shift_deg=0.0, shift_days=0):
    lat_e = np.arange(BOX_LAT[0] - deg, BOX_LAT[1] + 1e-9, deg) + shift_deg
    lon_e = np.arange(BOX_LON[0] - deg, BOX_LON[1] + deg + 1e-9, deg) + shift_deg
    m = pd.period_range(*PERIOD, freq="M")
    starts = m.start_time[::months]
    t_e = np.r_[starts.values, (m[-1].end_time + pd.Timedelta(seconds=1)).to_datetime64()]
    t_e = t_e + np.timedelta64(int(shift_days), "D")
    return lat_e, lon_e, t_e.astype("datetime64[s]")


def build(tag="_argo_resample", deg=2.0, months=2, shift_deg=0.0, shift_days=0,
          sources=("deploy",), exclude=()):
    from osim import argo_meta as am
    files = sorted(glob.glob(cs.profile_path(tag, chunk="*")))
    files = [f for f in files if int(f.rsplit("_c", 1)[1].split(".")[0]) not in set(exclude)]
    if not files:
        raise FileNotFoundError(cs.profile_path(tag, chunk="*"))
    pool = cs.read_resample_pool()
    rates, rho_age = eligibility_rates(pool, am.read_argo())
    rho = rates.set_index("pool_idx")["rho"]

    lat_e, lon_e, t_e = grid(deg, months, shift_deg, shift_days)
    nt, ny, nx = len(t_e) - 1, len(lat_e) - 1, len(lon_e) - 1
    kinds = ("nominal", "rho", "rho_age")
    acc = {k: np.zeros((len(files), nt, ny, nx)) for k in kinds}
    for r, f in enumerate(files):
        p = pd.read_parquet(f, columns=["time", "lat", "lon", "cycle", "pool_idx", "source"])
        p = p[p["source"].isin(sources)]
        it = np.searchsorted(t_e, p["time"].to_numpy().astype("datetime64[s]"), side="right") - 1
        iy = np.searchsorted(lat_e, p["lat"].to_numpy(), side="right") - 1
        ix = np.searchsorted(lon_e, p["lon"].to_numpy(), side="right") - 1
        ok = (it >= 0) & (it < nt) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        k = p["pool_idx"].to_numpy()[ok].astype(int)
        w = {"nominal": np.ones(ok.sum()), "rho": rho.reindex(k).to_numpy(),
             "rho_age": profile_rho_age(p[ok], rho_age)}
        for kind in kinds:
            np.add.at(acc[kind][r], (it[ok], iy[ok], ix[ok]), w[kind])

    days = np.diff(t_e).astype("timedelta64[s]").astype(float) / 86400.0
    mu = support_area(lat_e, lon_e)[None] * days[:, None, None]          # m2 * day
    R = len(files)
    coords = {"time": t_e[:-1], "lat": 0.5 * (lat_e[:-1] + lat_e[1:]),
              "lon": 0.5 * (lon_e[:-1] + lon_e[1:])}
    dims = ("time", "lat", "lon")
    ds = xr.Dataset(coords=coords)
    ds["time_edge"] = ("time_edge", t_e)
    ds["lat_edge"] = ("lat_edge", lat_e)
    ds["lon_edge"] = ("lon_edge", lon_e)
    with np.errstate(invalid="ignore", divide="ignore"):
        for kind in kinds:
            m, s = acc[kind].mean(0), acc[kind].std(0, ddof=1)
            ds[f"q_{kind}"] = (dims, np.where(mu > 0, m / mu, np.nan))
            ds[f"se_{kind}"] = (dims, np.where(mu > 0, s / np.sqrt(R) / mu, np.nan))
            ds[f"count_{kind}"] = (dims, m)
    ds["n_sim"] = (dims, acc["nominal"].sum(0))
    ds["mu"] = (dims, mu)
    ds.attrs.update(n_reps=R, tag=tag, deg=deg, months=months, shift_deg=shift_deg, shift_days=shift_days,
                    sources=",".join(sources),
                    units="q: profiles per m2 per day; count: expected profiles per replicate",
                    eligibility=f"gap >= {MIN_GAP_DAYS} d, max_depth >= {MIN_DEPTH} m")
    return ds


def qhat_path(tag="_argo_resample", deg=2.0, months=2, shift=False):
    sfx = f"_{deg:g}deg_{months}mo" + ("_shift" if shift else "")
    return os.path.join(cfg.OUT_DIR, f"qhat{tag}{sfx}.nc")


def at(ds, lat, lon, time, kind="rho"):
    """q-hat of the containing grid cell at each point (NaN off the grid)."""
    t_e = ds["time_edge"].values.astype("datetime64[s]")
    it = np.searchsorted(t_e, np.asarray(time, dtype="datetime64[s]"), side="right") - 1
    iy = np.searchsorted(ds["lat_edge"].values, np.asarray(lat), side="right") - 1
    ix = np.searchsorted(ds["lon_edge"].values, np.asarray(lon), side="right") - 1
    q = ds[f"q_{kind}"].values
    ok = ((it >= 0) & (it < q.shape[0]) & (iy >= 0) & (iy < q.shape[1])
          & (ix >= 0) & (ix < q.shape[2]))
    out = np.full(len(it), np.nan)
    out[ok] = q[it[ok], iy[ok], ix[ok]]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_argo_resample")
    ap.add_argument("--deg", type=float, default=2.0)
    ap.add_argument("--months", type=int, default=2)
    ap.add_argument("--shift", action="store_true", help="half-cell shift: deg/2 and months*15 d")
    a = ap.parse_args()
    ds = build(a.tag, a.deg, a.months, a.deg / 2 if a.shift else 0.0, 15 * a.months if a.shift else 0)
    out = qhat_path(a.tag, a.deg, a.months, a.shift)
    ds.to_netcdf(out)
    print(f"{ds.attrs['n_reps']} reps -> {out}")


# --------------------------- kernel (binned) q-hat ---------------------------

FINE_DEG, FINE_DAYS = 0.25, 5
KERNEL_LAT = (20.0, 48.0)       # boxes plus a margin for the kernel's reach
KERNEL_LON = (-62.0, -38.0)
KERNEL_T = ("2014-10-01", "2020-01-01")


def fine_grid(deg=FINE_DEG, days=FINE_DAYS):
    lat_e = np.arange(KERNEL_LAT[0], KERNEL_LAT[1] + 1e-9, deg)
    lon_e = np.arange(KERNEL_LON[0], KERNEL_LON[1] + 1e-9, deg)
    step = np.timedelta64(int(round(days * 24)), "h")
    t_e = np.arange(np.datetime64(KERNEL_T[0], "h"), np.datetime64(KERNEL_T[1], "h") + step,
                    step).astype("datetime64[s]")
    return lat_e, lon_e, t_e


def native_grid(lat=KERNEL_LAT, lon=KERNEL_LON, t=KERNEL_T):
    """Native GLORYS cells (midpoint edges of the bathymetry centres) x calendar days."""
    with xr.open_dataset(cs.BATHY_PATH) as ds:
        le = cfg.cell_edges(ds["latitude"].values.astype(float))
        oe = cfg.cell_edges(ds["longitude"].values.astype(float))
    t_e = np.arange(np.datetime64(t[0], "D"), np.datetime64(t[1], "D") + 1).astype("datetime64[s]")
    return (le[(le >= lat[0]) & (le <= lat[1])], oe[(oe >= lon[0]) & (oe <= lon[1])], t_e)


def _grid(deg, days):
    return native_grid() if deg == "native" else fine_grid(deg, days)


def binned_counts(files, rho_age, grid, sources=("deploy",), dtype=np.float32):
    """Per-replicate-mean weighted counts on the grid, for kinds nominal / rho_age."""
    lat_e, lon_e, t_e = grid
    shape = (len(t_e) - 1, len(lat_e) - 1, len(lon_e) - 1)
    flat, w = [], []
    for f in files:
        p = pd.read_parquet(f, columns=["time", "lat", "lon", "cycle", "pool_idx", "source"])
        p = p[p["source"].isin(sources)]
        it = np.searchsorted(t_e, p["time"].to_numpy().astype("datetime64[s]"), side="right") - 1
        iy = np.searchsorted(lat_e, p["lat"].to_numpy(), side="right") - 1
        ix = np.searchsorted(lon_e, p["lon"].to_numpy(), side="right") - 1
        ok = ((it >= 0) & (it < shape[0]) & (iy >= 0) & (iy < shape[1]) & (ix >= 0) & (ix < shape[2]))
        flat.append(np.ravel_multi_index((it[ok], iy[ok], ix[ok]), shape))
        w.append(profile_rho_age(p[ok], rho_age))
    flat, w = np.concatenate(flat), np.concatenate(w)
    n = int(np.prod(shape))
    return {"nominal": (np.bincount(flat, minlength=n) / len(files)).astype(dtype).reshape(shape),
            "rho_age": (np.bincount(flat, weights=w, minlength=n) / len(files)).astype(dtype).reshape(shape)}


def fine_support(grid, t_end=None):
    """Float-support area x days per fine cell (zero after the advection ends)."""
    lat_e, lon_e, t_e = grid
    t_end = np.datetime64(t_end or cs._velocity_end(), "s")
    days = (np.minimum(t_e[1:], t_end) - np.minimum(t_e[:-1], t_end)).astype(float) / 86400.0
    return support_area(lat_e, lon_e)[None] * days[:, None, None]


def smooth(counts, mu, sigma_deg, sigma_days, grid):
    """Binned Gaussian kernel intensity with edge correction: G*counts / G*mu."""
    from scipy.ndimage import gaussian_filter
    lat_e, lon_e, t_e = grid
    dlat = float(np.diff(lat_e)[0])
    dt = float(np.diff(t_e)[0].astype("timedelta64[s]").astype(float) / 86400.0)
    s = (sigma_days / dt, sigma_deg / dlat, sigma_deg / dlat)
    num = gaussian_filter(counts, s, mode="constant")
    den = gaussian_filter(mu, s, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((den > 0) & (mu > 0), num / den, np.nan)


def build_kernel(sigma_deg, sigma_days, tag="_argo_resample", exclude=(), counts=None, mu=None,
                 deg=FINE_DEG, days=FINE_DAYS):
    """Kernel q-hat on the fine (deg="native": GLORYS cell x day) grid, same layout as
    build() so at() applies. `counts`/`mu` may be precomputed to sweep bandwidths cheaply."""
    from osim import argo_meta as am
    grid = _grid(deg, days)
    if counts is None:
        files = [f for f in sorted(glob.glob(cs.profile_path(tag, chunk="*")))
                 if int(f.rsplit("_c", 1)[1].split(".")[0]) not in set(exclude)]
        _, rho_age = eligibility_rates(cs.read_resample_pool(), am.read_argo())
        counts = binned_counts(files, rho_age, grid)
    mu = (fine_support(grid) if mu is None else mu).astype(np.float32)
    lat_e, lon_e, t_e = grid
    ds = xr.Dataset(coords={"time": t_e[:-1], "lat": 0.5 * (lat_e[:-1] + lat_e[1:]),
                            "lon": 0.5 * (lon_e[:-1] + lon_e[1:])})
    ds["time_edge"], ds["lat_edge"], ds["lon_edge"] = ("time_edge", t_e), ("lat_edge", lat_e), ("lon_edge", lon_e)
    for kind, c in counts.items():
        ds[f"q_{kind}"] = (("time", "lat", "lon"), smooth(c, mu, sigma_deg, sigma_days, grid))
    ds.attrs.update(estimator="binned Gaussian kernel, edge-corrected", sigma_deg=sigma_deg,
                    sigma_days=sigma_days, fine_deg=deg, fine_days=days)
    return ds


def kernel_path(sigma_deg, sigma_days, tag="_argo_resample", deg=0.125, days=2.5):
    f = "native" if deg == "native" else f"{deg:g}_{days:g}"
    return os.path.join(cfg.OUT_DIR, f"qhat_kernel{tag}_s{sigma_deg:.4g}_{sigma_days:g}_f{f}.nc")


def load_kernel(sigma_deg, sigma_days, tag="_argo_resample", deg=0.125, days=2.5):
    """Kernel q-hat from all replicates, cached as float32 on first build."""
    p = kernel_path(sigma_deg, sigma_days, tag, deg, days)
    if not os.path.exists(p):
        ds = build_kernel(sigma_deg, sigma_days, tag, deg=deg, days=days)
        ds.to_netcdf(p, encoding={v: {"dtype": "float32", "zlib": True} for v in ds.data_vars
                                  if v.startswith("q_")})
    return xr.open_dataset(p)
