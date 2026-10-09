"""Anchored GLORYS clouds for real Argo profiles.

Each profile's float is advected from its anchor (its last fix at least MIN_LAG days earlier, or its
launch) to the profile time at park depth, plus an anisotropic random walk whose (kx, ky) each particle
carries. Work is chunked by anchor month.

    python code/osim/clouds.py profiles
    python code/osim/clouds.py fit  --month 2016-03     # dispersion-fit grid, 2015-2019 box anchors
    python code/osim/clouds.py kfit                      # K-hat, drift and diagnostics from the fit chunks
    python code/osim/clouds.py prod --month 2016-07 --kfit data/osim/clouds/prod_arms.csv
"""

import argparse
import math  # noqa: F401 (parcels kernels)
import os
import sys
import warnings
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import argo_meta as am
from osim import continuous_sim as cs
from osim import deployment as di
from osim import osim_config as cfg

OUT = os.path.join(cfg.OUT_DIR, "clouds")
PROFILES = os.path.join(OUT, "profiles.parquet")
ARGO_CSV = os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")

DEDUP_DAYS, MIN_LAG, MAX_H = 1.0, 35.0, 120.0
PROD_YEARS = (2015, 2019)
K_GRID = 100.0 * 2.0 ** np.arange(8)   # m2/s; 1 km2/d = 11.57 m2/s
M_FIT, M_PROD = 30, 100
K_MULT = 2.0 ** np.arange(-2, 2)
ETA_MU, ETA_SD = (0.0, -0.3, -0.6, -0.9, -1.2), (0.0, 0.25, 0.5)
SPEED_GRIDS = {"fit2": dict(regions=("boxA", "boxB"), mx=K_MULT, my=K_MULT, mu=ETA_MU, sd=ETA_SD),
               "fit3": dict(regions=("boxA",), mx=2.0 ** np.arange(-1, 3), my=2.0 ** np.arange(0, 4),
                            mu=(0.0, -0.6, -0.9, -1.2), sd=(0.0, 0.5, 0.75, 1.0))}
DT_S = 7200
VEL_END = pd.Timestamp("2019-12-31")   # last daily velocity slice
NORTH = cfg.BOXES["boxB"]["lat_bounds"][0]   # anchors at or north of 34 N take box B's (kx, ky)


def load_fixes(dedup=DEDUP_DAYS):
    """Every fix inside the park-velocity domain; a fix within `dedup` days of its float's previous fix is dropped."""
    (la0, la1), (lo0, lo1) = di.velocity_extent(cs._window_files("2010-01", "2010-01")[0])
    d = pd.read_csv(ARGO_CSV, usecols=["float_id", "lat", "lon", "date"]).dropna()
    d["time"] = pd.to_datetime(d["date"], format="ISO8601", utc=True).dt.tz_localize(None)
    d = d[d.lat.between(la0, la1) & d.lon.between(lo0, lo1)].sort_values(["float_id", "time"])
    gap = d.groupby("float_id").time.diff() / pd.Timedelta(days=1)
    return d[~(gap < dedup)][["float_id", "lat", "lon", "time"]].reset_index(drop=True)


def anchor_profiles(fixes, launch, lag=MIN_LAG, max_h=MAX_H):
    """Each fix's anchor: the float's latest fix or launch at least `lag` days earlier, else its launch.
    Fixes with no anchor, or a horizon over `max_h` days, get NaN anchors."""
    la = launch.dropna().rename(columns=dict(launch_lat="lat", launch_lon="lon", launch_date="time"))
    la = la.assign(time=la.time.dt.tz_localize(None))
    pool = pd.concat([fixes, la[["float_id", "lat", "lon", "time"]]]).sort_values("time")
    q = fixes.assign(t_q=fixes.time - pd.Timedelta(days=lag)).reset_index(names="prof_id").sort_values("t_q")
    a = pd.merge_asof(q, pool.rename(columns=dict(lat="a_lat", lon="a_lon", time="a_time")), left_on="t_q",
                      right_on="a_time", by="float_id", direction="backward")
    young = a.a_time.isna()
    a.loc[young, ["a_lat", "a_lon", "a_time"]] = (
        a[young][["float_id"]].merge(la, on="float_id", how="left")[["lat", "lon", "time"]].to_numpy())
    a["a_time"] = pd.to_datetime(a.a_time)
    a["horizon"] = (a.time - a.a_time) / pd.Timedelta(days=1)
    bad = ~((a.horizon > 0) & (a.horizon <= max_h))
    a.loc[bad, ["a_lat", "a_lon", "a_time", "horizon"]] = np.nan
    a["a_launch"] = young & ~bad
    return a.drop(columns="t_q").sort_values("prof_id").reset_index(drop=True)


def box_of(lat, lon):
    out = np.full(len(lat), "", object)
    for b in ("boxA", "boxB"):
        bb = cfg.BOXES[b]
        out[(lat >= bb["lat_bounds"][0]) & (lat <= bb["lat_bounds"][1])
            & (lon >= bb["lon_bounds"][0]) & (lon <= bb["lon_bounds"][1])] = b
    return out


def build_profiles():
    """Anchored profiles 2010-2019 whose anchor lies in the buffer region."""
    p = anchor_profiles(load_fixes(), am.read_launch())
    (la0, la1), (lo0, lo1) = cfg.BOXES["buffer"]["lat_bounds"], cfg.BOXES["buffer"]["lon_bounds"]
    p = p[p.a_lat.between(la0, la1) & p.a_lon.between(lo0, lo1) & p.time.dt.year.between(2010, 2019)]
    p = p.assign(box=box_of(p.lat.values, p.lon.values), a_box=box_of(p.a_lat.values, p.a_lon.values),
                 region=np.where(p.a_lat >= NORTH, "boxB", "boxA"))
    os.makedirs(OUT, exist_ok=True)
    p.reset_index(drop=True).to_parquet(PROFILES)
    return p


def advect(a_lat, a_lon, a_time, horizon, kx, ky, seed, fac=1.0):
    """Particle positions after `horizon` days from (a_lat, a_lon, a_time) under `fac` times the GLORYS park
    velocity plus a random walk of diffusivities kx, ky (m2/s). Particles that leave the velocity domain come
    back NaN."""
    from parcels import FieldSet, JITParticle, ParcelsRandom, ParticleSet, StatusCode, Variable

    class Cloud(JITParticle):
        kx = Variable("kx", dtype=np.float32, to_write=False)
        ky = Variable("ky", dtype=np.float32, to_write=False)
        fac = Variable("fac", dtype=np.float32, to_write=False)
        horizon = Variable("horizon", dtype=np.float32, to_write=False)
        age = Variable("age", dtype=np.float32, initial=0.0, to_write=False)
        pid = Variable("pid", dtype=np.int32, to_write=False)

    def ScaledRK4(particle, fieldset, time):  # noqa: N802 (parcels AdvectionRK4 with velocity times fac)
        (u1, v1) = fieldset.UV[particle]
        u1, v1 = u1 * particle.fac, v1 * particle.fac
        lon1, lat1 = (particle.lon + u1 * 0.5 * particle.dt, particle.lat + v1 * 0.5 * particle.dt)
        (u2, v2) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat1, lon1, particle]
        u2, v2 = u2 * particle.fac, v2 * particle.fac
        lon2, lat2 = (particle.lon + u2 * 0.5 * particle.dt, particle.lat + v2 * 0.5 * particle.dt)
        (u3, v3) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat2, lon2, particle]
        u3, v3 = u3 * particle.fac, v3 * particle.fac
        lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        u4, v4 = u4 * particle.fac, v4 * particle.fac
        particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particle.dt  # noqa: F821
        particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particle.dt  # noqa: F821

    def Diffuse(particle, fieldset, time):  # noqa: N802
        if particle.age < particle.horizon:
            particle_dlon += (math.sqrt(2 * particle.kx * math.fabs(particle.dt)) * ParcelsRandom.normalvariate(0, 1)  # noqa: F821
                              / (1852 * 60 * math.cos(particle.lat * math.pi / 180)))  # noqa: F821
            particle_dlat += (math.sqrt(2 * particle.ky * math.fabs(particle.dt)) * ParcelsRandom.normalvariate(0, 1)  # noqa: F821
                              / (1852 * 60))  # noqa: F821

    def Freeze(particle, fieldset, time):  # noqa: N802
        if particle.age >= particle.horizon:
            particle_dlon = 0  # noqa: F841
            particle_dlat = 0  # noqa: F841
        particle.age += particle.dt

    def DeleteOutOfBounds(particle, fieldset, time):  # noqa: N802
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    t0 = np.asarray(a_time, "datetime64[s]")
    h = np.round(np.asarray(horizon) * 86400 / DT_S) * DT_S
    t1 = (t0 + h.astype("timedelta64[s]")).max()
    files = cs._window_files(str(t0.min())[:7], str(t1)[:7])
    fs = FieldSet.from_netcdf(files, variables={"U": "uo", "V": "vo"},
                              dimensions={"time": "time", "lat": "latitude", "lon": "longitude"})
    n = len(t0)
    sl, sn = cs.SENTINEL_LATLON
    ps = ParticleSet(fieldset=fs, pclass=Cloud, lon=np.r_[a_lon, sn], lat=np.r_[a_lat, sl],
                     time=np.r_[t0, t0.min()], kx=np.r_[kx, 0.0], ky=np.r_[ky, 0.0],
                     fac=np.r_[np.broadcast_to(fac, n), 1.0], horizon=np.r_[h, 0.0],
                     pid=np.arange(n + 1, dtype=np.int32))
    ParcelsRandom.seed(int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps.execute([ScaledRK4, Diffuse, Freeze, DeleteOutOfBounds], endtime=t1,
                   dt=np.timedelta64(DT_S, "s"), verbose_progress=False)
    d = ps.particledata
    out = np.full((n, 2), np.nan, np.float32)
    pid = d.data["pid"]
    keep = pid < n
    out[pid[keep]] = np.column_stack([d.data["lat"][keep], d.data["lon"][keep]])
    return out


def speed_arms(kind="fit2"):
    """Per-box grid for the speed-factor fit: K-hat (kfit.csv) times the multipliers mx, my in each component,
    crossed with the log speed-factor mean and sd, for the regions of SPEED_GRIDS[kind]."""
    gr = SPEED_GRIDS[kind]
    k = pd.read_csv(os.path.join(OUT, "kfit.csv"))
    k = k[k.box.isin(gr["regions"])]
    g = pd.MultiIndex.from_product([gr["mx"], gr["my"], gr["mu"], gr["sd"]], names=["mx", "my", "eta_mu", "eta_sd"])
    g = g.to_frame(index=False).reset_index(names="arm")
    a = k[["box", "kx", "ky"]].rename(columns=dict(box="region")).merge(g, how="cross")
    return a.assign(kx=a.kx * a.mx, ky=a.ky * a.my)


def run_chunk(kind, month, kfit=None, m=None, seed=0, limit=None):
    """Advect every anchor in `month` for one job. kind "fit" crosses the K grid and "fit2" the speed_arms grid,
    both for box anchors; kind "prod" runs the arms in `kfit` (rows: arm, region, kx, ky, eta_mu, eta_sd) for
    every anchor. Particle j of m takes the log speed factor eta_mu + eta_sd * Phi^-1((j + 1/2) / m)."""
    from scipy.stats import norm
    p = pd.read_parquet(PROFILES)
    p = p[(p.a_time.dt.to_period("M") == pd.Period(month)) & p.time.dt.year.between(*PROD_YEARS)
          & (p.time <= VEL_END)]
    if kind == "fit":
        p = p[p.a_box != ""]
        kx, ky = (g.ravel() for g in np.meshgrid(K_GRID, K_GRID, indexing="ij"))
        arms = pd.DataFrame(dict(arm=np.arange(kx.size), kx=kx, ky=ky, eta_mu=0.0, eta_sd=0.0))
        per = p[["prof_id"]].merge(arms, how="cross")
        m = m or M_FIT
    else:
        speed = kind in SPEED_GRIDS
        p = p[p.a_box.isin(SPEED_GRIDS[kind]["regions"])] if speed else p
        arms = speed_arms(kind) if speed else pd.read_csv(kfit)
        on = ("a_box", "region") if speed else ("region", "region")
        per = p[["prof_id", on[0]]].merge(arms, left_on=on[0], right_on=on[1])
        m = m or (M_FIT if speed else M_PROD)
    per = per[["prof_id", "arm", "kx", "ky", "eta_mu", "eta_sd"]]
    if limit:
        per = per[per.prof_id.isin(per.prof_id.unique()[:limit])]
    per = per.merge(p[["prof_id", "a_lat", "a_lon", "a_time", "horizon"]], on="prof_id")
    rows = per.loc[per.index.repeat(m)].reset_index(drop=True)
    z = norm.ppf((np.arange(m) + 0.5) / m)[np.arange(len(rows)) % m]
    xy = advect(rows.a_lat.values, rows.a_lon.values, rows.a_time.values, rows.horizon.values,
                rows.kx.values, rows.ky.values, seed=zlib.crc32(f"{kind}{month}{seed}".encode()) % 2**31,
                fac=np.exp(rows.eta_mu.values + rows.eta_sd.values * z))
    res = pd.DataFrame(dict(prof_id=rows.prof_id.values.astype(np.int32), arm=rows.arm.values.astype(np.int16),
                            lat=xy[:, 0], lon=xy[:, 1]))
    path = chunk_path(kind, month, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    res.to_parquet(path)
    if kind != "prod":
        arms.to_csv(os.path.join(os.path.dirname(path), "arms.csv"), index=False)
    return res


# ----------------------------------------------------------------------------
# dispersion fit: moments of real-landing misses against the grid clouds
# ----------------------------------------------------------------------------

KM_PER_DEG = 111.195
FINE = np.linspace(np.log(K_GRID[0]), np.log(K_GRID[-1]), 141)
BIN_DEG = 2.0


def months(kind):
    p = pd.read_parquet(PROFILES)
    p = p[p.time.dt.year.between(*PROD_YEARS) & (p.time <= VEL_END)]
    p = p[p.a_box.isin(SPEED_GRIDS[kind]["regions"])] if kind in SPEED_GRIDS else p[p.a_box != ""] if kind == "fit" else p
    return sorted(p.a_time.dt.to_period("M").astype(str).unique())


def transition_moments(kind="fit"):
    """Per transition and arm: miss d = real landing - cloud mean (km), cloud variances s, particle count m."""
    import glob
    c = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(OUT, kind, f"{kind}_*.parquet")))])
    p = pd.read_parquet(PROFILES).set_index("prof_id")
    c = c.join(p[["lat", "lon"]].rename(columns=lambda k: f"r_{k}"), on="prof_id")
    c["dx"] = ((c.r_lon - c.lon + 180) % 360 - 180) * KM_PER_DEG * np.cos(np.radians(c.r_lat))
    c["dy"] = (c.r_lat - c.lat) * KM_PER_DEG
    g = c.dropna(subset=["dx"]).groupby(["prof_id", "arm"])
    t = g.agg(dx=("dx", "mean"), dy=("dy", "mean"), sx=("dx", "var"), sy=("dy", "var"), m=("dx", "size")).reset_index()
    lost = c.dx.isna().groupby(c.arm).mean().rename("lost")
    return t.join(p[["float_id", "a_box", "a_lat", "a_lon", "horizon"]], on="prof_id"), lost


def _solve(Gx, Gy, Dx, Dy):
    """(log kx, log ky) minimizing the scaled moment residuals, by bilinear interpolation on the K grid."""
    from scipy.interpolate import RegularGridInterpolator
    lk = np.log(K_GRID)
    X, Y = np.meshgrid(FINE, FINE, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    f = lambda G: RegularGridInterpolator((lk, lk), G.reshape(len(lk), len(lk)))(pts)
    obj = (f(Gx) / f(Dx)) ** 2 + (f(Gy) / f(Dy)) ** 2
    i = np.argmin(obj)
    edge = X.ravel()[i] in FINE[[0, -1]] or Y.ravel()[i] in FINE[[0, -1]]
    return np.exp(pts[i]), obj[i], edge


def fit_dispersion(t, drift=None, boot=400, seed=0):
    """K-hat per box from sum_j [d_j^2 - (1 + 1/m_j) s_j] = 0 in x and y, with a float-bootstrap 95% interval.
    `drift` (box -> (ux, uy) in km/d) removes u * h_j from each miss first."""
    arms = pd.read_csv(os.path.join(OUT, "fit", "arms.csv")).set_index("arm")
    rows = []
    for box, tb in t[t.a_box != ""].groupby("a_box"):
        u = (0.0, 0.0) if drift is None else drift[box]
        dx, dy = tb.dx - u[0] * tb.horizon, tb.dy - u[1] * tb.horizon
        terms = pd.DataFrame(dict(gx=dx**2 - (1 + 1 / tb.m) * tb.sx, gy=dy**2 - (1 + 1 / tb.m) * tb.sy,
                                  ex=dx**2, ey=dy**2, float_id=tb.float_id, arm=tb.arm))
        F = terms.groupby(["float_id", "arm"])[["gx", "gy", "ex", "ey"]].sum().unstack("arm").fillna(0.0)
        A = {k: F[k][arms.index].to_numpy() for k in ("gx", "gy", "ex", "ey")}
        k, obj, edge = _solve(*(A[k].sum(0) for k in ("gx", "gy", "ex", "ey")))
        W = np.random.default_rng(seed).multinomial(len(F), np.full(len(F), 1 / len(F)), boot)
        kb = np.array([_solve(*(w @ A[k] for k in ("gx", "gy", "ex", "ey")))[0] for w in W]).reshape(-1, 2)
        lo, hi = np.percentile(kb, [2.5, 97.5], axis=0) if boot else np.full((2, 2), np.nan)
        rows.append(dict(box=box, kx=k[0], ky=k[1], kx_lo=lo[0], kx_hi=hi[0], ky_lo=lo[1], ky_hi=hi[1],
                         obj=obj, edge=edge, n_trans=tb.prof_id.nunique(), n_floats=len(F)))
    return pd.DataFrame(rows)


def nearest_arm(kx, ky):
    arms = pd.read_csv(os.path.join(OUT, "fit", "arms.csv"))
    return int(arms.arm[np.argmin(np.hypot(np.log(arms.kx / kx), np.log(arms.ky / ky)))])


def mean_miss(t, kfit, boot=400, seed=0):
    """Mean miss and drift u = sum d / sum h (km/d) per box at the grid arm nearest K-hat, float-bootstrap se;
    and the mean miss in BIN_DEG anchor bins."""
    rows, bins = [], []
    for r in kfit.itertuples():
        tb = t[(t.a_box == r.box) & (t.arm == nearest_arm(r.kx, r.ky))]
        F = tb.groupby("float_id")[["dx", "dy", "horizon"]].sum()
        W = np.random.default_rng(seed).multinomial(len(F), np.full(len(F), 1 / len(F)), boot)
        ub = (W @ F[["dx", "dy"]].to_numpy()) / (W @ F.horizon.to_numpy())[:, None]
        u = F[["dx", "dy"]].sum().to_numpy() / F.horizon.sum()
        rows.append(dict(box=r.box, miss_x=tb.dx.mean(), miss_y=tb.dy.mean(), rms_x=np.sqrt((tb.dx**2).mean()),
                         rms_y=np.sqrt((tb.dy**2).mean()), ux=u[0], uy=u[1], ux_se=ub[:, 0].std(), uy_se=ub[:, 1].std()))
        b = tb.assign(blat=np.floor(tb.a_lat / BIN_DEG) * BIN_DEG, blon=np.floor(tb.a_lon / BIN_DEG) * BIN_DEG)
        bins.append(b.groupby(["blat", "blon"]).agg(miss_x=("dx", "mean"), miss_y=("dy", "mean"), n=("dx", "size"),
                                                   n_floats=("float_id", "nunique")).reset_index().assign(box=r.box))
    return pd.DataFrame(rows), pd.concat(bins, ignore_index=True)


def log_scores(t):
    """Mean Gaussian log score of the real landings under each grid arm's cloud (mean, (1 + 1/m) s), per box."""
    v = lambda s: (1 + 1 / t.m) * s
    ls = -0.5 * (np.log(2 * np.pi * v(t.sx)) + t.dx**2 / v(t.sx) + np.log(2 * np.pi * v(t.sy)) + t.dy**2 / v(t.sy))
    arms = pd.read_csv(os.path.join(OUT, "fit", "arms.csv"))
    return t.assign(ls=ls)[t.a_box != ""].groupby(["a_box", "arm"]).ls.mean().reset_index().merge(arms, on="arm")


def run_fit(by_year=True):
    """K-hat with and without the box drift, mean-miss tables, log scores and the per-year K-hat."""
    t, lost = transition_moments()
    k0 = fit_dispersion(t)
    mm, bins = mean_miss(t, k0)
    drift = {r.box: (r.ux, r.uy) for r in mm.itertuples()}
    k1 = fit_dispersion(t, drift)
    ls = log_scores(t)
    yr = pd.read_parquet(PROFILES).set_index("prof_id").time.dt.year
    ky = pd.concat([fit_dispersion(g, boot=0).assign(year=y) for y, g in t.groupby(t.prof_id.map(yr))]) if by_year else None
    out = dict(kfit=k0, kfit_drift=k1, mean_miss=mm, miss_bins=bins, log_scores=ls, lost=lost.reset_index(),
               kfit_year=ky)
    for name, df in out.items():
        if df is not None:
            df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    return out


# ----------------------------------------------------------------------------
# speed-factor fit: along/cross-flow moments relative to the GLORYS displacement
# ----------------------------------------------------------------------------

N_FINE = 21
SPEED_Q = 5


def _km(lat, lon, lat0, lon0):
    return (((lon - lon0 + 180) % 360 - 180) * KM_PER_DEG * np.cos(np.radians(lat0)), (lat - lat0) * KM_PER_DEG)


def along_moments(kind="fit2"):
    """Per transition and arm: misses d (real - cloud mean, km) in x, y and along/cross the GLORYS direction,
    cloud variances s, particle count m, and the GLORYS displacement L (km) that sets the direction. The
    direction is the cloud-mean displacement of the reference arm (K-hat, speed factor 1)."""
    import glob
    arms = pd.read_csv(os.path.join(OUT, kind, "arms.csv"))
    ref = arms[(arms.mx == 1) & (arms.my == 1) & (arms.eta_mu == 0) & (arms.eta_sd == 0)].arm.unique()
    p = pd.read_parquet(PROFILES).set_index("prof_id")[["lat", "lon", "a_lat", "a_lon", "a_box", "float_id", "horizon"]]
    out = []
    for f in sorted(glob.glob(os.path.join(OUT, kind, f"{kind}_*.parquet"))):
        c = pd.read_parquet(f).join(p[["lat", "lon", "a_lat", "a_lon"]].add_prefix("r_"), on="prof_id")
        c = c.dropna(subset=["lat"])
        c["px"], c["py"] = _km(c.lat.values, c.lon.values, c.r_lat.values, c.r_lon.values)
        gx, gy = _km(c.lat.values, c.lon.values, c.r_a_lat.values, c.r_a_lon.values)
        g = c.assign(gx=gx, gy=gy)[c.arm.isin(ref)].groupby("prof_id")[["gx", "gy"]].mean()
        g["L"] = np.hypot(g.gx, g.gy)
        c = c.join(g.assign(ex=g.gx / g.L, ey=g.gy / g.L)[["ex", "ey", "L"]], on="prof_id")
        c["pa"], c["pc"] = c.px * c.ex + c.py * c.ey, -c.px * c.ey + c.py * c.ex
        t = c.groupby(["prof_id", "arm"]).agg(
            **{f"d{k}": (f"p{k}", "mean") for k in "xyac"}, **{f"s{k}": (f"p{k}", "var") for k in "xyac"},
            m=("px", "size"), L=("L", "first")).reset_index()
        t[["dx", "dy", "da", "dc"]] *= -1
        out.append(t)
    return pd.concat(out, ignore_index=True).join(p[["a_box", "float_id", "horizon"]], on="prof_id")


def _equations(t):
    """Per float x arm sums of the four estimating functions and their scales."""
    f = 1 + 1 / t.m
    e = pd.DataFrame(dict(
        g1=t.L * t.da, g2=t.dx**2 - f * t.sx, g3=t.dy**2 - f * t.sy, g4=t.L**2 * (t.da**2 - f * t.sa),
        n1=t.L * t.da.abs(), n2=t.dx**2, n3=t.dy**2, n4=t.L**2 * t.da**2, float_id=t.float_id, arm=t.arm))
    return e.groupby(["float_id", "arm"]).sum().unstack("arm").fillna(0.0)


def _solve4(G, N, axes):
    """Grid point minimizing sum_k (G_k / N_k)^2 after linear interpolation of G, N over the 4-D arm grid."""
    from scipy.interpolate import RegularGridInterpolator
    fine = np.stack([a.ravel() for a in np.meshgrid(*[np.linspace(a[0], a[-1], N_FINE) for a in axes],
                                                    indexing="ij")], -1)
    shape = tuple(len(a) for a in axes)
    I = lambda v: RegularGridInterpolator(axes, v.reshape(shape))(fine)
    obj = sum((I(g) / I(n)) ** 2 for g, n in zip(G, N))
    i = np.argmin(obj)
    edge = any(fine[i, k] in (axes[k][0], axes[k][-1]) for k in range(4))
    return fine[i], obj[i], edge


def fit_speed(t, kind="fit2", boot=200, seed=0):
    """(kx, ky, eta_mu, eta_sd) per box from: mean along-flow miss weighted by L = 0, x and y variance
    calibration, and along-flow variance calibration weighted by L^2; float-bootstrap 95% intervals."""
    arms = pd.read_csv(os.path.join(OUT, kind, "arms.csv"))
    rows = []
    for box, tb in t[t.a_box.isin(SPEED_GRIDS[kind]["regions"])].groupby("a_box"):
        ab = arms[arms.region == box].assign(lmx=lambda d: np.log(d.mx), lmy=lambda d: np.log(d.my))
        ab = ab.sort_values(["lmx", "lmy", "eta_mu", "eta_sd"])
        axes = [np.unique(ab[c]) for c in ("lmx", "lmy", "eta_mu", "eta_sd")]
        F = _equations(tb)
        A = {k: F[k][ab.arm].to_numpy() for k in ("g1", "g2", "g3", "g4", "n1", "n2", "n3", "n4")}
        solve = lambda w: _solve4([w @ A[f"g{k}"] for k in range(1, 5)], [w @ A[f"n{k}"] for k in range(1, 5)], axes)
        x, obj, edge = solve(np.ones(len(F)))
        W = np.random.default_rng(seed).multinomial(len(F), np.full(len(F), 1 / len(F)), boot)
        xb = np.array([solve(w)[0] for w in W])
        lo, hi = np.percentile(xb, [2.5, 97.5], axis=0)
        k0 = ab.iloc[0]
        kx0, ky0 = k0.kx / k0.mx, k0.ky / k0.my
        tr = lambda v: (kx0 * np.exp(v[0]), ky0 * np.exp(v[1]), v[2], v[3])
        rows.append(dict(box=box, **dict(zip(("kx", "ky", "eta_mu", "eta_sd"), tr(x))),
                         **dict(zip(("kx_lo", "ky_lo", "eta_mu_lo", "eta_sd_lo"), tr(lo))),
                         **dict(zip(("kx_hi", "ky_hi", "eta_mu_hi", "eta_sd_hi"), tr(hi))),
                         obj=obj, edge=edge, n_trans=tb.prof_id.nunique(), n_floats=len(F)))
    return pd.DataFrame(rows)


def calibration_by_speed(t, fit, kind="fit2"):
    """Along/cross variance ratio and mean along miss by GLORYS-speed quintile, at the grid arm nearest the
    fit and at the K-only reference arm."""
    arms = pd.read_csv(os.path.join(OUT, kind, "arms.csv"))
    out = []
    for r in fit.itertuples():
        ab = arms[arms.region == r.box]
        dist = (np.log(ab.kx / r.kx) ** 2 + np.log(ab.ky / r.ky) ** 2 + (ab.eta_mu - r.eta_mu) ** 2
                + (ab.eta_sd - r.eta_sd) ** 2)
        ref = ab[(ab.mx == 1) & (ab.my == 1) & (ab.eta_mu == 0) & (ab.eta_sd == 0)].arm.iloc[0]
        for lab, arm in (("fitted", ab.arm.iloc[np.argmin(dist)]), ("K only", ref)):
            tb = t[(t.a_box == r.box) & (t.arm == arm)]
            f = 1 + 1 / tb.m
            q = pd.qcut(tb.L / tb.horizon * 1e5 / 86400, SPEED_Q)
            g = pd.DataFrame(dict(da2=tb.da**2, sa=f * tb.sa, dc2=tb.dc**2, sc=f * tb.sc, da=tb.da)).groupby(q, observed=True)
            s = g.sum()
            out.append(pd.DataFrame(dict(box=r.box, arms=lab, speed_cm_s=[iv.mid for iv in s.index],
                                         ratio_along=s.da2 / s.sa, ratio_cross=s.dc2 / s.sc,
                                         mean_along_km=g.da.mean().values)))
    return pd.concat(out, ignore_index=True)


def run_fit2(kinds={"boxA": "fit3", "boxB": "fit2"}):
    """Speed-model fit per box from the grid run `kinds[box]`, its calibration table, and the production arms."""
    fits, cals = [], []
    for kind in dict.fromkeys(kinds.values()):
        t = along_moments(kind)
        t = t[t.a_box.isin([b for b, k in kinds.items() if k == kind])]
        f = fit_speed(t, kind)
        fits.append(f.assign(grid=kind))
        cals.append(calibration_by_speed(t, f, kind).assign(grid=kind))
    fit, cal = pd.concat(fits, ignore_index=True), pd.concat(cals, ignore_index=True)
    k1 = pd.read_csv(os.path.join(OUT, "kfit.csv"))
    arms = []
    for i, (lab, src, kscale, speed) in enumerate([("speed", fit, 1, True), ("K_only", k1, 1, False),
                                                   ("speed_K/4", fit, 0.25, True), ("speed_4K", fit, 4, True)]):
        for r in src.itertuples():
            arms.append(dict(arm=i, label=lab, region=r.box, kx=r.kx * kscale, ky=r.ky * kscale,
                             eta_mu=r.eta_mu if speed else 0.0, eta_sd=r.eta_sd if speed else 0.0))
    out = dict(kfit_speed=fit, calib_speed=cal, prod_arms=pd.DataFrame(arms))
    for name, df in out.items():
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    return out


def chunk_path(kind, month, seed=0):
    return os.path.join(OUT, kind, f"{kind}_{month}_s{seed}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("profiles", "months", "fit", "kfit", "fit2", "fit3", "kfit2", "prod"))
    ap.add_argument("--month")
    ap.add_argument("--kfit")
    ap.add_argument("--m", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.what == "profiles":
        p = build_profiles()
        print(p.groupby([p.time.dt.year, "box"]).size().unstack(), "\n", p.a_launch.mean(), p.horizon.describe())
    elif a.what == "months":
        print(" ".join(months(a.month or "fit")))
    elif a.what == "kfit":
        o = run_fit()
        for k in ("kfit", "kfit_drift", "mean_miss"):
            print(k, "\n", o[k].round(3).to_string(index=False))
    elif a.what == "kfit2":
        o = run_fit2()
        for k, v in o.items():
            print(k, "\n", v.round(3).to_string(index=False))
    else:
        run_chunk(a.what, a.month, a.kfit, a.m, a.seed, a.limit)
