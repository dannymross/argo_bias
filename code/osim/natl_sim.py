"""North Atlantic case study: idealized uniform deployment advected in GLORYS, then truth, q-hat, map and
cross-fit DR over the deep domain (water >= 2000 m connected to the open basin, north of 10 N).

Deployment law per replicate: N_y floats in year y (real core-mission launches whose first profile is in the
domain: deep profilers and floats on short regional cycles excluded), release uniform in the year, position uniform
in area over the domain, 10 d cycle, lifetime from the Kaplan-Meier curve of real launches in years y-1..y+1.
Advection ignores lifetime; it is applied to the stored candidate profiles (fleet, select). Floats released before 2010 (no velocity) that are still alive on
2010-01-01 start there at a uniform position; together they form cohort INIT. Each float draws from its
own RNG keyed on (rep, year, index), so adding floats or replicates never changes existing ones.

    python code/osim/natl_sim.py design
    python code/osim/natl_sim.py advect --rep 0       # or as an array task: the index is the replicate
    python code/osim/natl_sim.py fields               # truth + oracle lattice, one pass over the daily store
    python code/osim/natl_sim.py oracle_prepare --layer ohc_15_700   # anomaly lattice + centres for krige_local_fit.py
    python code/osim/natl_sim.py profiles --reps 0,49 # candidates (5 d grid) + selection; select alone after a fleet change
    python code/osim/natl_sim.py sample --reps 0,49   # OHC at the candidates, one pass over the store, then select
    python code/osim/natl_sim.py qpoints --reps 0,49 --all-reps 200
    python code/osim/natl_sim.py map --rep 0 --layer ohc_15_700   # map of one layer (needs that layer's plug-ins)
    python code/osim/natl_sim.py estimate --rep 0     # per layer and their sum (needs qpoints and both maps)

Target: 15-2000 m OHC as the sum of two layers mapped separately, as Kuusela et al. (15-975 + 975-1850),
each with its own local mean, oracle covariance and kriging; every estimator is linear, so the total is the sum.
"""

import argparse
import functools
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import costlog
from osim import gyre_mask as gm
from osim import osim_config as cfg

os.environ.setdefault("GLORYS_DR_DOMAIN", "natl_sim")   # before any glorys_dr import
OUT = os.path.join(cfg.OUT_DIR, "natl_sim")
COSTS = os.path.join(OUT, "costs.csv")
TRACKS = os.path.join(os.path.dirname(cfg.NATL_RAW), "natl_sim_tracks")
INDEX = os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_index", "ar_index_global_prof.txt.gz")
VEL = os.path.join(cfg.NATL_DIR, "velocity_1000")
SOUTH = 10.0
T0, T1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31")
FIRST_YEAR = 2002          # earliest deployment year that can still be alive in 2010
ACTIVE_DAYS = 60           # a float whose last profile is this recent is censored (still alive)
META = INDEX.replace("_prof.", "_meta.")
DEEP_TYPES = {838, 849, 862, 864, 874, 881, 883}   # R08 deep profilers: they park well below 1000 m
CYCLE = 10                 # every simulated float profiles every 10 d (core Argo mission)
SHORT_CYCLE = 7.5          # real floats with a shorter median profile interval (regional 5 d missions) are excluded
KM_HALF = 1                # lifetime curve of launch year y pools launches y-1..y+1
LMAX_YEARS = 12            # advection span cap: beyond every observed lifetime
BASE = 5                   # candidate profile spacing (days): keeps 5 d cycles open as a sensitivity
INIT = T0.year - 1         # cohort label of the initial stock
SEED = 7100
ID_STRIDE = 10_000         # float_id = launch year * ID_STRIDE + index within the year: stable under design changes


@functools.lru_cache(1)
def domain():
    """Native grid, deep-domain mask and cell area (m2)."""
    b = xr.open_dataset(os.path.join(cfg.NATL_DIR, "static", "natl_bathy.nc")).deptho.load()
    la, lo = b.latitude.values.astype(float), b.longitude.values.astype(float)
    return la, lo, gm.deep_domain(b.values, la, lo, south=SOUTH), gm.cell_area(la, lo) * 1e6


def in_domain(lat, lon):
    la, lo, m, _ = domain()
    i, j = cfg.containing_index(lat, la), cfg.containing_index(lon, lo)
    out = np.zeros(len(i), bool)
    ok = (i >= 0) & (j >= 0)
    out[ok] = m[i[ok], j[ok]]
    return out


def argo_floats():
    """Per float in the GDAC index: first ascending profile (date, lat, lon), last profile date, count, median
    interval between profiles (days) and profiler type (meta index)."""
    d = pd.read_csv(INDEX, comment="#", usecols=["file", "date", "latitude", "longitude"]).dropna()
    d = d[~d.file.str.endswith("D.nc")]
    d = d.assign(float_id=d.file.str.split("/").str[1].astype(int),
                 date=pd.to_datetime(d.date.astype("int64").astype(str), format="%Y%m%d%H%M%S", errors="coerce"))
    d = d.dropna(subset=["date"]).sort_values(["float_id", "date"])
    g = d.groupby("float_id")
    gap = d.date.diff().where(d.float_id.eq(d.float_id.shift())).dt.total_seconds() / 86400
    m = pd.read_csv(META, comment="#", usecols=["file", "profiler_type"])
    f = (g[["date", "latitude", "longitude"]].first().join(g.date.max().rename("last")).join(g.size().rename("n"))
         .join(gap.groupby(d.float_id).median().rename("interval"))
         .join(m.set_index(m.file.str.split("/").str[1].astype(int).rename("float_id")).profiler_type))
    return f.assign(inside=in_domain(f.latitude.to_numpy(), f.longitude.to_numpy()))


def index_date():
    import gzip
    with gzip.open(INDEX, "rt") as h:
        line = next(l for l in h if l.startswith("# Date of update"))
    return pd.to_datetime(line.split(":")[1].strip(), format="%Y%m%d%H%M%S")


def km_sf(g):
    from scipy.stats import CensoredData, ecdf
    return ecdf(CensoredData(uncensored=g.life[~g.alive].to_numpy(), right=g.life[g.alive].to_numpy())).sf


def design():
    """Per launch year: N_y and the Kaplan-Meier lifetime curve of launches y-1..y+1. Real launches whose first
    profile is in the domain; deep profilers and short-cycle (regional) floats excluded."""
    with costlog.step(COSTS, "design"):
        f = argo_floats()
        f = f[f.inside & f.date.dt.year.between(FIRST_YEAR, T1.year)]
        drop = pd.Series(dict(deep=f.profiler_type.isin(DEEP_TYPES).sum(),
                              short_cycle=(~f.profiler_type.isin(DEEP_TYPES) & (f.interval < SHORT_CYCLE)).sum()))
        f = f[~f.profiler_type.isin(DEEP_TYPES) & ~(f.interval < SHORT_CYCLE)]
        f = f.assign(year=f.date.dt.year, life=(f["last"] - f.date).dt.total_seconds() / 86400,
                     alive=f["last"] >= index_date() - pd.Timedelta(days=ACTIVE_DAYS))
        n = f.groupby("year").size().rename("n").reset_index()
        km = pd.concat([pd.DataFrame(dict(year=y, n=len(g), days=(s := km_sf(g)).quantiles, sf=s.probabilities))
                        for y in n.year for g in [f[f.year.between(y - KM_HALF, y + KM_HALF)]]], ignore_index=True)
        os.makedirs(OUT, exist_ok=True)
        n.to_csv(os.path.join(OUT, "design_counts.csv"), index=False)
        km.to_csv(os.path.join(OUT, "lifetime_km.csv"), index=False)
        la, lo, m, a = domain()
        xr.Dataset(dict(domain=(("latitude", "longitude"), m.astype("i1")),
                        cell_area=(("latitude", "longitude"), np.where(m, a, np.nan))),
                   coords=dict(latitude=la, longitude=lo), attrs=dict(south=SOUTH)).to_netcdf(
            os.path.join(OUT, "domain.nc"))
    print(n.to_string(index=False), f"\n{len(f)} floats, {f.alive.sum()} censored; excluded {drop.to_dict()}", flush=True)
    return n, km


def seeds(rep):
    """Every nominal deployment of one replicate: cohort, release, position, the uniform its lifetime is drawn
    with, and its advection start and span. Advection ignores lifetime (fleet() applies it to the tracks); it runs
    from max(release, T0) to the earlier of release + LMAX_YEARS and T1. All deployments are kept: q-hat is per
    nominal deployment, so folds partition them."""
    n = pd.read_csv(os.path.join(OUT, "design_counts.csv")).set_index("year").n
    la, lo, m, a = domain()
    cells = np.flatnonzero(m)
    cum = np.cumsum(a.ravel()[cells])
    lae, loe = cfg.cell_edges(la), cfg.cell_edges(lo)
    rows = []
    for y, ny in n.items():
        u = np.stack([np.random.default_rng([SEED, rep, y, j]).random(5) for j in range(ny)])
        start = pd.Timestamp(f"{y}-01-01")
        span = (pd.Timestamp(f"{y + 1}-01-01") - start).total_seconds()
        rel = start + pd.to_timedelta(u[:, 0] * span, "s")
        c = cells[np.minimum(np.searchsorted(cum, u[:, 1] * cum[-1]), len(cells) - 1)]
        i, j = np.unravel_index(c, m.shape)
        s0, s1 = np.sin(np.deg2rad(lae[i])), np.sin(np.deg2rad(lae[i + 1]))
        rows.append(pd.DataFrame(dict(year=y, index=np.arange(ny), release=rel, u_life=u[:, 4],
                                      lat=np.rad2deg(np.arcsin(s0 + u[:, 2] * (s1 - s0))),
                                      lon=loe[j] + u[:, 3] * (loe[j + 1] - loe[j]))))
    d = pd.concat(rows, ignore_index=True)
    d["cohort"] = np.where(d.year < T0.year, INIT, d.year)
    d["start"] = d.release.clip(lower=T0)
    stop = (d.release + pd.DateOffset(years=LMAX_YEARS)).clip(upper=T1 + pd.Timedelta(days=1))
    d["span"] = (stop - d.start).dt.total_seconds() / 86400
    return d.assign(float_id=d.year * ID_STRIDE + d["index"])


def fleet(rep, tag=""):
    """The deployed floats under the current design (the first N_y of each year: draws are keyed per float, so a
    smaller design is a subset of an advected larger one), with cycle and the lifetime of their launch year's KM
    curve at u_life (inf past the curve: alive beyond the index date). No re-advection needed after a change."""
    s = pd.read_parquet(seeds_path(rep, tag))
    n = pd.read_csv(os.path.join(OUT, "design_counts.csv")).set_index("year").n
    s = s[s["index"] < s.year.map(n).fillna(0)]
    life = np.empty(len(s))
    for y, g in pd.read_csv(os.path.join(OUT, "lifetime_km.csv")).groupby("year"):
        idx = np.flatnonzero(s.year.to_numpy() == y)
        k = np.searchsorted(1 - g.sf.to_numpy(), s.u_life.to_numpy()[idx], side="left")
        life[idx] = np.append(g.days.to_numpy(), np.inf)[k]
    return pd.DataFrame(dict(float_id=s.float_id, cohort=s.cohort, cycle_days=CYCLE, life=life))


def seeds_path(rep, tag=""):
    return os.path.join(OUT, f"profiles{tag}", f"seeds_r{rep}.parquet")


def profile_path(rep, tag=""):
    return os.path.join(OUT, f"profiles{tag}", f"r{rep}.parquet")


def track_path(rep, tag=""):
    return os.path.join(TRACKS, f"r{rep}{tag}.parquet")


def advect(rep, dt_hours=1, tag=""):
    """One continuous run of every deployment without mortality, daily output, tracks kept (keyed by float_id);
    profiles then follow from fleet(). `tag` keeps a test run (e.g. another dt_hours) apart from production."""
    from osim import continuous_sim as cs
    s = seeds(rep)
    os.makedirs(os.path.dirname(profile_path(rep, tag)), exist_ok=True)
    s.to_parquet(seeds_path(rep, tag))
    zarr = os.path.join(cs.scratch_dir(), f"r{rep}.zarr")   # node-local: per-step zarr writes crawl on network disk
    os.makedirs(TRACKS, exist_ok=True)
    with costlog.step(COSTS, f"advect{tag}", rep=rep, n=len(s)):
        shutil.rmtree(zarr, ignore_errors=True)
        cs.run_continuous(rep, zarr, len(s), T0, T1, vel_dir=VEL, output_days=1, dt_hours=dt_hours,
                          seeds=(s.lat.to_numpy(), s.lon.to_numpy(), s.start.to_numpy(), s.span.to_numpy()))
        tr = cs.tracks_from_zarr(zarr, rep)
        tr["float_id"] = s.float_id.to_numpy()[tr.float_id.to_numpy()]    # particle index -> deployment id
        tr.to_parquet(track_path(rep, tag))
        shutil.rmtree(zarr)
    return profiles(rep, tag)


def cand_path(rep, tag=""):
    return os.path.join(OUT, f"candidates{tag}", f"r{rep}.parquet")


def candidates(rep, tag=""):
    """Every float's would-be profiles at release + k * BASE days (k >= 1) over its whole advection span, linearly
    interpolated in the daily track, in the domain or not. Any lifetime law and any cycle that is a multiple of
    BASE is a subset of this table (select), so it replaces the tracks for reuse."""
    s = pd.read_parquet(seeds_path(rep, tag))
    tr = pd.read_parquet(track_path(rep, tag), columns=["float_id", "time", "lat", "lon"])
    sid = s.year * ID_STRIDE + s["index"]
    if not (s.float_id == sid).all():   # written before float ids were stable: re-key seeds and tracks
        tr["float_id"] = tr.float_id.map(pd.Series(sid.to_numpy(), index=s.float_id))
        s = s.assign(float_id=sid)
        s.to_parquet(seeds_path(rep, tag))
        tr.sort_values(["float_id", "time"]).to_parquet(track_path(rep, tag))
        tr = pd.read_parquet(track_path(rep, tag), columns=["float_id", "time", "lat", "lon"])
    with costlog.step(COSTS, f"candidates{tag}", rep=rep) as meta:
        fid, t = tr.float_id.to_numpy(np.int64), tr.time.to_numpy().astype("datetime64[s]").astype(np.int64)
        lat, lon = tr.lat.to_numpy(float), tr.lon.to_numpy(float)
        ends = tr.groupby("float_id").time.agg(["min", "max"]).astype("datetime64[s]").astype(np.int64)
        f = s.set_index("float_id").join(ends, how="inner")
        r0 = f.release.to_numpy().astype("datetime64[s]").astype(np.int64)
        c = BASE * 86400
        kmin = np.maximum(1, -((r0 - f["min"].to_numpy()) // c))
        nk = np.maximum((f["max"].to_numpy() - r0) // c - kmin + 1, 0)
        pid = np.repeat(f.index.to_numpy(np.int64), nk)
        k = np.repeat(kmin, nk) + np.arange(nk.sum()) - np.repeat(np.cumsum(nk) - nk, nk)
        tk = np.repeat(r0, nk) + k * c
        K = np.int64(2) ** 34
        key = fid * K + t
        i = np.clip(np.searchsorted(key, pid * K + tk, side="right") - 1, 0, len(t) - 1)
        j = np.minimum(i + 1, len(t) - 1)
        dt = (t[j] - t[i]).astype(float)
        w = np.where((fid[j] == pid) & (dt > 0), (tk - t[i]) / np.where(dt > 0, dt, 1), 0.0)
        p = pd.DataFrame(dict(rep=rep, float_id=pid, k=k, time=tk.astype("datetime64[s]").astype("datetime64[ns]"),
                              lat=lat[i] + w * (lat[j] - lat[i]), lon=lon[i] + w * (lon[j] - lon[i])))
        p = p.merge(s[["float_id", "cohort"]], on="float_id")
        p["inside"] = in_domain(p.lat.to_numpy(), p.lon.to_numpy())
        os.makedirs(os.path.dirname(cand_path(rep, tag)), exist_ok=True)
        p.to_parquet(cand_path(rep, tag))
        meta["n"] = len(p)
    print(f"rep {rep}: {len(f)} floats advected, {len(p)} candidate profiles", flush=True)
    return p


def select(rep, tag=""):
    """A replicate's profiles under the current fleet law: candidates on the float's cycle (k a multiple of
    cycle_days / BASE) before its end of life; `cycle` counts its own cycles. Carries any sampled OHC columns."""
    fl = pd.read_parquet(seeds_path(rep, tag), columns=["float_id", "release"]).merge(
        fleet(rep, tag).drop(columns="cohort"), on="float_id")
    p = pd.read_parquet(cand_path(rep, tag)).merge(fl, on="float_id")
    m = p.cycle_days.to_numpy() // BASE
    end = p.release + pd.to_timedelta(np.minimum(p.life.to_numpy(), 4e4), "D")
    p = p[(p.k.to_numpy() % m == 0) & (p.time <= end)]
    p = p.assign(cycle=p.k // (p.cycle_days // BASE)).drop(columns=["k", "release", "life"])
    p.to_parquet(profile_path(rep, tag))
    print(f"rep {rep}: {len(p)} profiles ({p.float_id.nunique()} floats), {p.inside.sum()} in the domain", flush=True)
    return p


def profiles(rep, tag=""):
    candidates(rep, tag)
    return select(rep, tag)


# ------------------------------------------- truth and the oracle lattice -------------------------------------------

LAYERS = os.path.join(cfg.GLORYS_DIR, cfg.LAYER_DIR, "ohc_layers_daily_{y}_{m:02d}.nc")
LAT_DEG, LAT_DAYS = 1.0, 10          # oracle lattice spacing, as krige_local
ORACLE_MONTHS = (1, 4, 7, 10)        # plug-ins are the median over months; 4 of 12 (plan)
ORACLE_FROM = pd.Timestamp("2014-11-01")   # pad before 2015 for January windows (krige_local_fit drops year 1)


# Mapped layers as sums of stored ones; 0-15 m is left out, as in Kuusela et al.
LAYER_SUMS = {"ohc_15_700": ("ohc_15_700",), "ohc_700_2000": ("ohc_700_975", "ohc_975_1850", "ohc_1850_2000")}
LAYER_NAMES = tuple(LAYER_SUMS)


def ohc_month(y, m):
    """Layer OHC (J/m2), one variable per LAYER_SUMS key; ohc_700_2000 is NaN where the floor is above 2000 m."""
    with xr.open_dataset(LAYERS.format(y=y, m=m)) as d:
        return xr.Dataset({k: sum(d[v] for v in vs) for k, vs in LAYER_SUMS.items()}).load()


def fields():
    """One pass over the daily store: daily domain-mean OHC (truth) and the 1 deg x 10 d lattice of OHC."""
    la, lo, m, a = domain()
    w = np.where(m, a, 0.0)
    k = int(round(LAT_DEG / float(np.median(np.diff(la)))))
    T, L = [], []
    with costlog.step(COSTS, "fields"):
        for y in range(T0.year, T1.year + 1):
            for mo in range(1, 13):
                z = ohc_month(y, mo)
                v = {c: z[c].values for c in LAYER_NAMES}
                if any(np.isnan(x[:, m]).any() for x in v.values()):
                    raise ValueError(f"{y}-{mo:02d}: NaN inside the domain")
                t = pd.DatetimeIndex(z.time.values)
                T.append(pd.DataFrame(dict(time=t, **{c: np.einsum("tij,ij->t", np.nan_to_num(x), w) / w.sum()
                                                      for c, x in v.items()})))
                sel = np.flatnonzero(((t - T0).days % LAT_DAYS) == 0)
                LA, LO = np.meshgrid(la[::k], lo[::k], indexing="ij")
                L.append(pd.DataFrame(dict(time=np.repeat(t[sel], LA.size), lat=np.tile(LA.ravel(), len(sel)),
                                           lon=np.tile(LO.ravel(), len(sel)),
                                           **{c: x[sel][:, ::k, ::k].ravel() for c, x in v.items()})).dropna())
            print(y, flush=True)
        truth = pd.concat(T, ignore_index=True)
        lat_ = pd.concat(L, ignore_index=True)
        truth.to_parquet(os.path.join(OUT, "truth_daily.parquet"))
        lat_.to_parquet(os.path.join(OUT, "lattice.parquet"))
    return truth, lat_


def truth():
    """theta_y (J/m2) per layer and their sum ("total"): the annual mean of the daily domain means; long format."""
    d = pd.read_parquet(os.path.join(OUT, "truth_daily.parquet"))
    d = d.assign(total=d[list(LAYER_NAMES)].sum(axis=1)).melt(id_vars="time", var_name="layer", value_name="ohc")
    return d.groupby([d.time.dt.year.rename("year"), "layer"]).ohc.agg(theta="mean", n_days="size").reset_index()


def oracle_path(layer, kind):
    return os.path.join(OUT, f"oracle_{kind}_{layer}.{'npz' if kind == 'mean' else 'csv'}")


def oracle_prepare(layer):
    """Anomaly lattice for the oracle covariance fit: lattice OHC minus the local LS mean (glorys_dr.fit_mean, the
    form every replicate fits) fitted once to the whole lattice; written in the krige_local_fit layout (GJ/m2), with
    the parameter-grid centres whose 2 deg cell touches the domain, ORACLE_MONTHS only."""
    from osim import glorys_dr as gd
    from osim import krige_local as kl
    with costlog.step(COSTS, "oracle_prepare", rep=layer):
        d = pd.read_parquet(os.path.join(OUT, "lattice.parquet"), columns=["time", "lat", "lon", layer])
        d = d.rename(columns={layer: "z"}).assign(day=lambda x: (x.time - gd.EPOCH) / pd.Timedelta(days=1), one=0)
        full = gd.fit_mean(d, "one")[0]
        np.savez_compressed(oracle_path(layer, "mean"), full=full, c_lat=gd.C_LAT, c_lon=gd.C_LON)
        d["anom"] = d.z - gd.mean_at(full, d.lat.values, d.lon.values, d.day.values)
        d = d[(d.time >= ORACLE_FROM) & d.anom.notna()]
        out = pd.DataFrame(dict(lon=d.lon, lat=d.lat, day=(d.time - pd.Timestamp(kl.EPOCH)).dt.days,
                                year=d.time.dt.year, yearday=d.time.dt.dayofyear, z=d.anom * 1e-9))
        out.to_csv(oracle_path(layer, "lattice"), index=False)
        la, lo, m, _ = domain()
        cen = kl.centres(deg=kl.PARAM_DEG, months=ORACLE_MONTHS, bounds={"natl": (cfg.NATL_BOX["lat_bounds"],
                                                                                    cfg.NATL_BOX["lon_bounds"])})
        h = kl.PARAM_DEG / 2
        touch = [m[np.ix_((la >= c.lat - h) & (la < c.lat + h), (lo >= c.lon - h) & (lo < c.lon + h))].any()
                 for c in cen.itertuples()]
        cen[touch].to_csv(os.path.join(OUT, "oracle_centres.csv"), index=False)
    print(f"{layer} oracle lattice {len(out):,} rows; {sum(touch)} windows "
          f"({sum(touch) // len(ORACLE_MONTHS)} cells x {len(ORACLE_MONTHS)} months)", flush=True)


# ---------------------------------------------- OHC at the profiles ----------------------------------------------

def sample(reps):
    """GLORYS layer OHC at the containing cell and day of every candidate profile of `reps`, one pass over the daily
    store, written into the candidate tables (one column per layer, NaN off the deep field); then select."""
    la, lo = domain()[:2]
    P = pd.concat([pd.read_parquet(cand_path(r)) for r in reps], ignore_index=True)
    t = pd.DatetimeIndex(P.time)
    i, j = cfg.containing_index(P.lat.to_numpy(), la), cfg.containing_index(P.lon.to_numpy(), lo)
    z = np.full((len(P), len(LAYER_NAMES)), np.nan)
    with costlog.step(COSTS, "sample", rep=f"{min(reps)}-{max(reps)}", n=len(P)):
        for (y, m), idx in pd.DataFrame(dict(y=t.year, m=t.month)).groupby(["y", "m"]).indices.items():
            idx = idx[(i[idx] >= 0) & (j[idx] >= 0)]
            d = ohc_month(y, m)
            z[idx] = np.column_stack([d[c].values[t.day.values[idx] - 1, i[idx], j[idx]] for c in LAYER_NAMES])
        P = P.drop(columns=[c for c in P if c.startswith("ohc_")]).assign(**dict(zip(LAYER_NAMES, z.T)))
        for r, g in P.groupby("rep"):
            g.to_parquet(cand_path(r))
            select(r)


# ------------------------------- q-hat: cohort kernel intensity, leave one replicate out -------------------------------

def cohort_sizes():
    n = pd.read_csv(os.path.join(OUT, "design_counts.csv")).set_index("year").n
    return pd.concat([pd.Series({INIT: n[n.index < T0.year].sum()}), n[n.index >= T0.year]])


@functools.lru_cache(1)
def support():
    """Float support for the kernel's edge correction: native cells deep enough to park (m2 per cell)."""
    from osim import deployment as di
    b = xr.open_dataset(os.path.join(cfg.NATL_DIR, "static", "natl_bathy.nc")).deptho.values
    la, lo, _, a = domain()
    return cfg.cell_edges(la), cfg.cell_edges(lo), np.where(np.nan_to_num(b) >= di.PARK_DEPTH, a, 0.0)


QSIG, TRUNC = (1.0, 30.0), 4.0   # q-hat kernel sd (deg, days), chosen by the cross-half Hajek test; truncation in sd


def q_dir(sig=QSIG):
    return os.path.join(OUT, f"q_s{sig[0]:g}_t{sig[1]:g}")


def q_points(reps, all_reps=range(200), sigs=(QSIG,), map_year=None):
    """Leave-one-replicate-out q-hat (in-domain profiles per km2 per day per replicate) at the in-domain
    analysis-year profiles of `reps`, from the in-domain profile positions of `all_reps`: Gaussian kernel on the
    native grid block-summed as qhat_diag, edge-corrected by the kernel mass over the domain x [T0, T1].
    map_year: also save the annual mean of the all-replicate q-hat / q-bar in that year per sd (OUT/qhat_maps)."""
    from scipy import ndimage
    from osim import qhat_diag as qd
    lat, lon, _, A = qd.grid()
    dx = float(np.median(np.diff(lat)))
    yrs = cfg.ANALYSIS_YEARS
    d0, d1 = ((pd.Timestamp(f"{y}-01-01") - T0).days for y in (yrs[0], yrs[-1] + 1))
    cols = ["rep", "float_id", "cycle", "time", "lat", "lon", "inside"]
    P = pd.concat([pd.read_parquet(profile_path(r), columns=cols).query("inside") for r in all_reps], ignore_index=True)
    P = P.assign(day=((P.time - T0) / pd.Timedelta(days=1)).to_numpy(),
                 i=np.floor((P.lat.to_numpy() - lat[0]) / dx + 0.5).astype(np.int64),
                 j=np.floor((P.lon.to_numpy() - lon[0]) / dx + 0.5).astype(np.int64))
    ev = P.time.dt.year.isin(yrs).to_numpy() & P.rep.isin(reps).to_numpy()
    R = len(all_reps)
    for sig in sigs:
        out = q_dir(sig)
        os.makedirs(out, exist_ok=True)
        with costlog.step(COSTS, "q_points", rep=f"{min(reps, default='')}-{max(reps, default='')} s{sig[0]:g}_t{sig[1]:g}",
                          n=int(ev.sum())):
            fs, ft = max(1, round(sig[0] / dx / 2)), max(1, int(sig[1] / 2 + 1e-6))
            pad = -(-int(TRUNC * sig[1] + 1) // ft) * ft
            ta, nt = d0 - pad, -(-(d1 - d0 + 2 * pad) // ft)
            days = ta + np.arange(nt * ft)
            tsup = qd.block(((days >= 0) & (days <= (T1 - T0).days)).astype(np.float32), ft, 0)
            asup = qd.block(qd.block(A.astype(np.float32), fs, 0), fs, 1)
            sg = (sig[1] / ft, sig[0] / dx / fs, sig[0] / dx / fs)
            f = lambda c: ndimage.gaussian_filter(c, sg, truncate=TRUNC, mode="constant")
            tden = ndimage.gaussian_filter1d(tsup, sg[0], truncate=TRUNC, mode="constant")
            aden = ndimage.gaussian_filter(asup, sg[1:], truncate=TRUNC, mode="constant")
            shape = (nt, *asup.shape)
            cnt = lambda g: qd.counts(g.day.to_numpy(), g.i.to_numpy(), g.j.to_numpy(), ta, ft, fs, shape)
            S = f(cnt(P))
            if map_year:
                ya = (pd.Timestamp(f"{map_year}-01-01") - T0).days
                ny_ = pd.Timestamp(f"{map_year}-12-31").dayofyear
                kb = np.flatnonzero((ta + np.arange(nt) * ft >= ya) & (ta + np.arange(nt) * ft < ya + ny_))
                ok = (asup > 0) & (aden > 0)
                qa = np.where(ok, (S[kb] / tden[kb, None, None]).mean(0) / (R * np.where(ok, aden, 1)), np.nan)
                qa /= np.nansum(qa * asup) / asup[ok].sum()
                c = lambda x, n: x[: n * fs].reshape(n, fs).mean(1) if len(x) >= n * fs else np.r_[x, x[-1] + dx * np.arange(1, n * fs - len(x) + 1)].reshape(n, fs).mean(1)
                tag = f"s{sig[0]:g}_t{sig[1]:g}"
                mp = os.path.join(OUT, "qhat_maps", f"maps_{map_year}.npz")
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                old = dict(np.load(mp)) if os.path.exists(mp) else {}
                np.savez_compressed(mp, **old, **{tag: qa.astype(np.float32), f"lat_{tag}": c(lat, qa.shape[0]),
                                                  f"lon_{tag}": c(lon, qa.shape[1])})
            for r, g in P[ev].groupby("rep"):
                k = (np.floor((g.day.to_numpy() - ta) / ft).astype(np.int64), g.i.to_numpy() // fs, g.j.to_numpy() // fs)
                q = ((S - f(cnt(P[P.rep == r])))[k] / ((R - 1) * tden[k[0]] * aden[k[1], k[2]])).astype(np.float32)
                g[["float_id", "cycle"]].assign(q=q).to_parquet(os.path.join(out, f"r{r}.parquet"))


K = 5


@functools.lru_cache(1)
def nodes():
    """0.5 deg map cells of the domain at the area centroid of their domain cells, with that area (m2)."""
    from osim import glorys_dr as gd
    la, lo, m, a = domain()
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    w = a[m]
    d = pd.DataFrame(dict(i=np.floor(LA[m] / gd.NODE_DEG), j=np.floor(LO[m] / gd.NODE_DEG), area=w,
                          wlat=w * LA[m], wlon=w * LO[m])).groupby(["i", "j"]).sum()
    return pd.DataFrame(dict(lat=d.wlat / d.area, lon=d.wlon / d.area, area=d.area, box="natl")).reset_index(drop=True)


def rep_obs(rep, layer, path=None):
    """A replicate's profiles with one layer's OHC as z, in the glorys_dr obs layout. Profiles are those valid in
    every layer, so prof_id is shared across layers."""
    from osim import glorys_dr as gd
    p = pd.read_parquet(path or profile_path(rep)).dropna(subset=list(LAYER_NAMES)).rename(columns={layer: "z"})
    fl = deployments(rep)
    p = p.assign(fold=p.float_id.map(fl.set_index("float_id").fold), box=np.where(p.inside, "natl", ""),
                 day=(p.time - gd.EPOCH) / pd.Timedelta(days=1))
    return p.reset_index(drop=True).reset_index(names="prof_id")


def deployments(rep):
    """All nominal deployments of a replicate with cohort and fold (a random permutation within cohort)."""
    from osim import ipw
    s = fleet(rep)[["float_id", "cohort"]].reset_index(drop=True)
    return s.assign(fold=ipw.folds(s.float_id.to_numpy(), K, seed=rep, strata=s.cohort.to_numpy()))


def cov_points(rep, layer, path=None):
    """A replicate's residuals about its own full-data local mean (GJ/m2), in the covariance-fit point layout."""
    from osim import chain as ch
    obs = ch.fit_map(rep_obs(rep, layer, path))[0]
    return pd.DataFrame(dict(lon=obs.lon, lat=obs.lat, day=(obs.time - T0) / pd.Timedelta(days=1),
                             year=obs.time.dt.year, z=(obs.z - obs.mu) * 1e-9)).dropna(subset=["z"])  # mu undefined off the mean grid


def gpgp_fit(points, centres, out, years=cfg.ANALYSIS_YEARS, m=30):
    """Per-window covariance MLE with GpGp (krige_local_gpgp.R), as an R subprocess over all `centres`."""
    import subprocess
    from osim import continuous_sim as cs
    d = cs.scratch_dir()
    pp, cp = os.path.join(d, "cov_points.csv"), os.path.join(d, "cov_centres.csv")
    points[["lon", "lat", "day", "year", "z"]].to_csv(pp, index=False)
    centres[["lat", "lon", "month"]].to_csv(cp, index=False)
    r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krige_local_gpgp.R")
    subprocess.run(["Rscript", r, pp, cp, out, ",".join(map(str, years)), str(m)], check=True, cwd=cfg.REPO_ROOT)
    return pd.read_csv(out)


COV_CHUNKS = 10


COV_DIR = "cov_exact"   # exact calendar-season fits; "cov" holds the earlier GpGp +/-45 d fits


def cov_path(rep, layer, chunk=None, kind=COV_DIR):
    d = os.path.join(OUT, kind, f"r{rep}")
    return os.path.join(d, f"{layer}_t{chunk}.csv" if chunk is not None else f"plugin_{layer}.csv")


def covfit(reps, task, chunks=COV_CHUNKS):
    """One array task of the per-replicate covariance fits: task -> (rep, layer, chunk of the parameter-cell x
    season windows). The windows are the oracle_centres grid (2 deg cells touching the domain, ORACLE_MONTHS)."""
    rep = reps[task // (len(LAYER_NAMES) * chunks)]
    layer = LAYER_NAMES[(task // chunks) % len(LAYER_NAMES)]
    k = task % chunks
    cen = pd.read_csv(os.path.join(OUT, "oracle_centres.csv"))
    cen = cen.iloc[len(cen) * k // chunks: len(cen) * (k + 1) // chunks]
    out = cov_path(rep, layer, k)
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part.jsonl"   # one line per finished window, so a rerun resumes where the last stopped
    done = pd.read_json(part, lines=True) if os.path.exists(part) else pd.DataFrame(columns=["lat", "lon", "month"])
    todo = cen.merge(done[["lat", "lon", "month"]], how="left", indicator=True).query("_merge == 'left_only'")
    with costlog.step(COSTS, f"covfit_{layer}", rep=rep, n=len(todo)):
        pts = cov_points(rep, layer)
        with open(part, "a") as f:
            for i, c in enumerate(todo.itertuples()):
                w = cov_window(pts, c.lat, c.lon, c.month)
                r = dict(lat=c.lat, lon=c.lon, month=c.month, n_obs=len(w), n_blocks=w.year.nunique())
                try:
                    r.update(exact_fit(w, EXACT_STARTS) if r["n_blocks"] >= 2 else {})
                except Exception as e:   # record the failure and move on; the window's parameters stay NaN
                    r["error"] = repr(e)[:200]
                f.write(pd.Series(r).to_json() + "\n")
                f.flush()
                print(f"{len(done) + i + 1}/{len(cen)} windows", flush=True)
    pd.read_json(part, lines=True).to_csv(out, index=False)
    os.remove(part)


def covgather(rep):
    """Per layer: the plug-ins of each parameter cell, the median of its seasonal fits (Kuusela's practice)."""
    P = ["phi", "th_lon", "th_lat", "th_t", "nugget_ratio"]
    for layer in LAYER_NAMES:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(cov_path(rep, layer, "*")))], ignore_index=True)
        g = d.dropna(subset=P).groupby(["lat", "lon"])
        g[P].median().join(g.size().rename("n_seasons")).reset_index().to_csv(cov_path(rep, layer), index=False)
        print(f"rep {rep} {layer}: {len(d)} windows, {d[P].isna().any(axis=1).sum()} failed, "
              f"{g.ngroups} cells", flush=True)


WIN_DEG, MIN_BLOCK = 10.0, 50
EXACT_START = (1.0, 1.3, 1.0, 20.0, 0.05)       # phi, th_lon, th_lat, th_t, nugget ratio on sd-scaled z (GpGp's start)
EXACT_STARTS = (EXACT_START,)   # a second start (2, 1.5, 70) reached the same optimum in 40/40 windows (|dll| < 0.01)


def cov_window(pts, lat, lon, month, years=cfg.ANALYSIS_YEARS):
    """The profiles of one fitting window: +/-WIN_DEG of the cell centre in calendar months month-1..month+1 (DJF,
    MAM, JJA, SON for 1, 4, 7, 10; December belongs to the following year's block), keeping blocks with more than
    MIN_BLOCK profiles; `year` becomes the block and dt the days from its centre month's 15th."""
    s = pts[pts.lat.between(lat - WIN_DEG, lat + WIN_DEG) & pts.lon.between(lon - WIN_DEG, lon + WIN_DEG)]
    t = T0 + pd.to_timedelta(s.day.to_numpy(), "D")
    k = t.year * 12 + t.month - month + 1           # 12 * block + 0, 1, 2 inside the window
    blk = k // 12
    s = s.assign(year=blk)[(k % 12 <= 2) & np.isin(blk, years)]
    tc = (pd.to_datetime(dict(year=s.year, month=month, day=15)) - T0).dt.days
    s = s.assign(dt=s.day - tc.values)
    n = s.year.value_counts()
    return s[s.year.isin(n.index[n > MIN_BLOCK])]


def exact_nll(s):
    """Negative exact log-likelihood (and gradient) of the sd-scaled window residuals as a function of the
    log-parameters theta = log(phi, th_lon, th_lat, th_t, nugget variance) of phi * exp(-|(dlon, dlat, dt) / th|)
    + nugget, years independent: the sum over years of sklearn's exact per-year log marginal likelihood."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    sy = s.z.std()
    kern = ConstantKernel(1.0, (1e-6, 1e6)) * Matern([1.0, 1.0, 1.0], (1e-3, 1e5), nu=0.5) + WhiteKernel(1e-2, (1e-10, 1e2))
    gps = [GaussianProcessRegressor(kern, optimizer=None).fit(g[["lon", "lat", "dt"]].values, g.z.values / sy)
           for _, g in s.groupby("year")]

    def nll(th):
        v = [gp.log_marginal_likelihood(th, eval_gradient=True) for gp in gps]
        return -sum(l for l, _ in v), -sum(g for _, g in v)
    return nll, kern.bounds, sy, len(gps)


def to_theta(p):
    """(phi, th_lon, th_lat, th_t, nugget ratio) on the scaled data -> sklearn's log-parameters."""
    return np.log([p[0], p[1], p[2], p[3], p[4] * p[0]])


def exact_fit(s, starts=(EXACT_START,)):
    """Exact MLE: L-BFGS-B on exact_nll from each start (scaled-data parameters), keeping the best optimum."""
    import time
    from scipy.optimize import minimize
    t0 = time.time()
    nll, bounds, sy, nb = exact_nll(s)
    fits = [minimize(nll, np.clip(to_theta(p), bounds[:, 0], bounds[:, 1]), jac=True, method="L-BFGS-B", bounds=bounds)
            for p in starts]
    f = min(fits, key=lambda r: r.fun)
    e = np.exp(f.x)
    return dict(phi=e[0] * sy ** 2, th_lon=e[1], th_lat=e[2], th_t=e[3], nugget_ratio=e[4] / e[0],
                loglik=-f.fun - len(s) * np.log(sy), conv=f.success, best_start=fits.index(f),
                ll_gap=max(r.fun for r in fits) - f.fun,   # log-lik lost by the worst start
                n_eval=sum(r.nfev for r in fits), at_bound=bool(np.isclose(f.x, bounds.T).any()),
                secs=time.time() - t0, n_obs=len(s), n_blocks=nb)


def cov_exact_test(n_windows=30, seed=0, out="starts_calendar.csv"):
    """Start check for the exact calendar-season fit on a random sample of GpGp window fits of reps 0-4 (half per
    layer): EXACT_STARTS alone vs EXACT_STARTS plus the GpGp estimate as a third start."""
    out = os.path.join(OUT, "pilot_exact", out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    w = pd.concat([cov_windows(range(5), l, kind="cov").dropna(subset=list(COV_PARAMS)).assign(layer=l)
                   .sample(n_windows // len(LAYER_NAMES), random_state=seed) for l in LAYER_NAMES])
    rows = []
    for (rep, layer), ww in w.groupby(["rep", "layer"]):
        pts = cov_points(rep, layer)
        for _, g in ww.iterrows():
            with costlog.step(COSTS, f"cov_exact_test_{layer}", rep=rep, n=1):
                s = cov_window(pts, g.lat, g.lon, g.month)
                e = exact_fit(s, EXACT_STARTS)
                e3 = exact_fit(s, (*EXACT_STARTS, (g.phi / s.z.var(), g.th_lon, g.th_lat, g.th_t, g.nugget_ratio)))
            rows.append(dict(rep=rep, layer=layer, lat=g.lat, lon=g.lon, month=g.month, n_obs_gpgp_45d=g.n_obs,
                             **{f"gpgp_{p}": g[p] for p in COV_PARAMS}, **{f"exact_{k}": v for k, v in e.items()},
                             **{f"exact3_{k}": e3[k] for k in [*COV_PARAMS, "loglik", "best_start"]}))
            print(pd.Series(rows[-1]).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def uniform_sampling(R=200, seed=0, n_from=range(20)):
    """Comparison design: each replicate-year draws n_y iid points uniform on domain x year (cell with probability
    proportional to area, uniform day), n_y the mean in-domain profile count per replicate-year of the Lagrangian
    replicates `n_from`; returns per layer-year the sample means' errors (GJ/m2) and the exact iid SD sigma_y/sqrt(n_y)."""
    la, lo, m, a = domain()
    w = a[m] / a[m].sum()
    n = pd.concat([pd.read_parquet(profile_path(r), columns=["time", "inside", *LAYER_NAMES]) for r in n_from])
    n = n[n.inside].dropna(subset=list(LAYER_NAMES)).time.dt.year.value_counts() / len(n_from)
    tr = truth().set_index(["year", "layer"]).theta
    rng = np.random.default_rng(seed)
    rows = []
    for y in cfg.ANALYSIS_YEARS:
        with costlog.step(COSTS, "uniform_sampling", rep=y, n=R):
            z = [{k: d[k].values[:, m].astype(np.float32) * 1e-9 for k in LAYER_NAMES}
                 for d in map(lambda mo: ohc_month(y, mo), range(1, 13))]
            Z = {k: np.concatenate([d[k] for d in z]) for k in LAYER_NAMES}
            Z["total"] = sum(Z[k] for k in LAYER_NAMES)
            ny = int(round(n[y]))
            c = rng.choice(len(w), size=(R, ny), p=w)
            t = rng.integers(0, Z["total"].shape[0], size=(R, ny))
            for k, v in Z.items():
                mu = (v * w).sum(1).mean()
                sd = np.sqrt(((v - mu) ** 2 * w).sum(1).mean() / ny)
                est = v[t, c].mean(1)
                rows.append(pd.DataFrame(dict(year=y, layer=k, rep=np.arange(R), n=ny, est=est, theta=tr[y, k] * 1e-9,
                                              theta_grid=mu, sd_iid=sd)))
    d = pd.concat(rows, ignore_index=True).assign(err=lambda x: x.est - x.theta)
    d.to_csv(os.path.join(OUT, "uniform_errors.csv"), index=False)
    return d


def stationary_sampling(reps=range(200)):
    """Comparison design: the deployed fleet of each replicate (same seeds, lifetimes and 10 d cycle) with every float
    held at its start position (no advection); per replicate, layer and year, the in-domain sample mean (GJ/m2)."""
    la, lo, m, _ = domain()
    P = []
    for r in reps:
        s = pd.read_parquet(seeds_path(r), columns=["float_id", "release", "start", "lat", "lon"]).merge(
            fleet(r)[["float_id", "life"]])
        end = (s.release + pd.to_timedelta(np.minimum(s.life.to_numpy(), 4e4), "D")).clip(upper=T1)
        nk = ((end - s.release) / pd.Timedelta(days=CYCLE)).astype(int).to_numpy()
        k = np.concatenate([np.arange(1, n + 1) for n in nk])
        i = np.repeat(np.arange(len(s)), nk)
        t = s.release.to_numpy()[i] + (k * CYCLE).astype("timedelta64[D]")
        ok = (t >= s.start.to_numpy()[i]) & (pd.DatetimeIndex(t).year.isin(cfg.ANALYSIS_YEARS))
        P.append(pd.DataFrame(dict(rep=r, time=t[ok], lat=s.lat.to_numpy()[i][ok], lon=s.lon.to_numpy()[i][ok])))
    P = pd.concat(P, ignore_index=True)
    t = pd.DatetimeIndex(P.time)
    ii, jj = cfg.containing_index(P.lat.to_numpy(), la), cfg.containing_index(P.lon.to_numpy(), lo)
    z = np.full((len(P), len(LAYER_NAMES)), np.nan)
    with costlog.step(COSTS, "stationary_sampling", rep=f"{min(reps)}-{max(reps)}", n=len(P)):
        for (y, mo), idx in pd.DataFrame(dict(y=t.year, m=t.month)).groupby(["y", "m"]).indices.items():
            d = ohc_month(y, mo)
            z[idx] = np.column_stack([d[c].values[t.day.values[idx] - 1, ii[idx], jj[idx]] for c in LAYER_NAMES])
    P = P.assign(**dict(zip(LAYER_NAMES, z.T * 1e-9)), year=t.year)
    P = P[m[ii, jj]].dropna(subset=list(LAYER_NAMES))
    P["total"] = P[list(LAYER_NAMES)].sum(1)
    e = P.groupby(["rep", "year"])[[*LAYER_NAMES, "total"]].mean().stack().rename("est").reset_index()
    e.columns = ["rep", "year", "layer", "est"]
    e = e.merge(P.groupby(["rep", "year"]).size().rename("n").reset_index()).merge(
        truth().assign(theta=lambda x: x.theta * 1e-9)[["year", "layer", "theta"]]).assign(err=lambda x: x.est - x.theta)
    e.to_csv(os.path.join(OUT, "stationary_errors.csv"), index=False)
    return e


def node_cells(nd):
    """Bilinear weights of the native cell centres around each node, renormalized over in-domain corners (nearest
    in-domain cell if none). Returns node, i, j, w."""
    from scipy.spatial import cKDTree
    la, lo, m, _ = domain()
    dx = float(np.median(np.diff(la)))
    u, v = (nd.lat.to_numpy() - la[0]) / dx, (nd.lon.to_numpy() - lo[0]) / dx
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    n = np.arange(len(nd))
    c = pd.concat([pd.DataFrame(dict(node=n, i=i0 + a, j=j0 + b, w=(fu if a else 1 - fu) * (fv if b else 1 - fv)))
                   for a in (0, 1) for b in (0, 1)], ignore_index=True)
    c = c[(c.w > 0) & m[c.i, c.j]]
    miss = np.setdiff1d(n, c.node)
    di, dj = np.nonzero(m)
    k = cKDTree(np.column_stack([la[di], lo[dj]])).query(np.column_stack([u[miss] * dx + la[0], v[miss] * dx + lo[0]]))[1]
    c = pd.concat([c, pd.DataFrame(dict(node=miss, i=di[k], j=dj[k], w=1.0))], ignore_index=True)
    return c.assign(w=c.w / c.groupby("node").w.transform("sum"))


def quadrature():
    """Q_y(Z) - theta_y per layer: the day-weighted annual mean of the area-weighted map-node average of the true
    field at the T_NODES times minus the true annual mean. Z at a node interpolates the native field: bilinear in
    space (node_cells), linear in time between daily means centred at noon."""
    from osim import glorys_dr as gd
    nd = nodes()
    c = node_cells(nd)
    rows = []
    for y in range(cfg.ANALYSIS_YEARS[0], cfg.ANALYSIS_YEARS[-1] + 1):
        for mo in range(1, 13):
            m0 = pd.Timestamp(f"{y}-{mo:02d}-01")
            with xr.open_dataset(LAYERS.format(y=y, m=mo)) as d:
                for f in gd.T_NODES:
                    t = np.clip(m0.days_in_month * f - 0.5, 0, m0.days_in_month - 1)
                    d0, a = int(np.floor(t)), t - np.floor(t)
                    z = d.isel(time=d0).load() * (1 - a) + (d.isel(time=min(d0 + 1, m0.days_in_month - 1)).load() * a)
                    rows.append(dict(year=y, wd=m0.days_in_month / len(gd.T_NODES), **{
                        k: np.average(np.bincount(c.node, c.w * sum(z[v].values for v in vs)[c.i, c.j], len(nd)),
                                      weights=nd.area) for k, vs in LAYER_SUMS.items()}))
    q = pd.DataFrame(rows).assign(total=lambda x: x[list(LAYER_NAMES)].sum(axis=1))
    q = q.groupby("year").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.wd) for c in [*LAYER_NAMES, "total"]}))
    q = q.reset_index().melt(id_vars="year", var_name="layer", value_name="Q").merge(truth(), on=["year", "layer"])
    q["remainder"] = q.Q - q.theta
    q.to_csv(os.path.join(OUT, "quadrature.csv"), index=False)
    return q


COV_PARAMS = ("phi", "th_lon", "th_lat", "th_t")
COV_LABELS = {"phi": "variance phi [(GJ/m2)^2]", "th_lon": "zonal range [km]", "th_lat": "meridional range [km]",
              "th_t": "temporal range [days]"}
RUNAWAY = {"th_lon": 40.0, "th_lat": 40.0, "th_t": 400.0}   # the bounds of krige_local_fit; beyond them a fit ran away
KM_PER_DEG = 111.195   # mean Earth radius 6371 km


def cov_windows(reps, layer, kind=COV_DIR):
    """Every window fit (cell x season) of the given replicates for one layer."""
    return pd.concat([pd.read_csv(f).assign(rep=r) for r in reps for f in glob.glob(cov_path(r, layer, "*", kind))],
                     ignore_index=True)


def cov_plugins(reps, layer, km=False, kind=COV_DIR, cols=COV_PARAMS):
    """Plug-ins per 2 deg cell: each replicate's median over seasons, and "pooled", the median over all window fits
    of all `reps` (seasons x replicates); runaway values (beyond RUNAWAY, in deg) set to NaN; `km` converts the ranges
    at the cell-centre latitude."""
    w = cov_windows(reps, layer, kind)
    d = pd.concat([w.groupby(["rep", "lat", "lon"])[list(cols)].median().reset_index(),
                   w.groupby(["lat", "lon"])[list(cols)].median().reset_index().assign(rep="pooled")])
    for p, ub in RUNAWAY.items():
        d[p] = d[p].where(d[p] <= ub)
    if km:
        d["th_lon"] *= KM_PER_DEG * np.cos(np.deg2rad(d.lat))
        d["th_lat"] *= KM_PER_DEG
    return d


KUUSELA = os.path.join(cfg.REPO_ROOT, "data", "kuusela_mle", "raw", "MLE", "intTempLocalMLESpaceTimeTrend_{}_2004_2022.mat")
KUUSELA_LAYERS = {"ohc_15_700": "15_975", "ohc_700_2000": "975_1850"}   # nearest of their two layers
KUUSELA_RHO_CP = 1030 * 3989.244   # their rho0 * cp0; files are unscaled degC dbar, dbar taken as m


def kuusela_plugins(layer, km=False):
    """Sukianto, Kuusela et al. (2026) Zenodo 18273718 plug-ins (median over 12 monthly fits, 2004-2022, 1 deg) on our
    2 deg cells: median of the 1 deg values in each cell, phi in (GJ/m2)^2, runaway rule as cov_plugins."""
    from scipy.io import loadmat
    m = loadmat(KUUSELA.format(KUUSELA_LAYERS[layer]))
    lo, la = (a.ravel() for a in np.meshgrid(np.arange(20.5, 380), np.arange(-89.5, 90), indexing="ij"))
    d = pd.DataFrame(dict(lat=2 * np.floor(la / 2) + 1, lon=2 * np.floor(((lo + 180) % 360 - 180) / 2) + 1,
                          phi=m["thetasOpt"].ravel() * (KUUSELA_RHO_CP * 1e-9) ** 2, th_lon=m["thetaLongOpt"].ravel(),
                          th_lat=m["thetaLatOpt"].ravel(), th_t=m["thetatOpt"].ravel()))
    cells = pd.read_csv(os.path.join(OUT, "oracle_centres.csv"))[["lat", "lon"]].drop_duplicates()
    d = cells.merge(d.dropna().groupby(["lat", "lon"]).median().reset_index(), how="left")
    for p, ub in RUNAWAY.items():
        d[p] = d[p].where(d[p] <= ub)
    if km:
        d["th_lon"] *= KM_PER_DEG * np.cos(np.deg2rad(d.lat))
        d["th_lat"] *= KM_PER_DEG
    return d


def plot_cov_maps(panels, par, nc=3):
    """Maps of one parameter on the 2 deg grid, one per (title, frame) in `panels`, drawn on the native grid and
    clipped to the domain, on one viridis colour scale spanning all non-NaN values; NaN (runaway or missing) cells
    left blank."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    la, lo, m, _ = domain()
    nr = -(-len(panels) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.5 * nr), squeeze=False,
                           subplot_kw=dict(projection=ccrs.PlateCarree()), layout="constrained")
    v = pd.concat([d[par] for _, d in panels])
    for a, (title, d) in zip(ax.flat, panels):
        g = d.pivot(index="lat", columns="lon", values=par).reindex(index=2 * np.floor(la / 2) + 1,
                                                                    columns=2 * np.floor(lo / 2) + 1)
        pc = a.pcolormesh(lo, la, np.where(m, g.values, np.nan), shading="nearest", vmin=v.min(), vmax=v.max(),
                          cmap="viridis", rasterized=True, transform=ccrs.PlateCarree())
        a.contour(lo, la, m.astype(float), levels=[0.5], colors="k", linewidths=0.6, transform=ccrs.PlateCarree())
        a.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        a.coastlines(lw=0.4, zorder=3)
        a.set_extent([lo.min(), lo.max(), SOUTH, la.max()], crs=ccrs.PlateCarree())
        a.set_title(title, fontsize=9)
    for a in ax.flat[len(panels):]:
        a.set_visible(False)
    fig.colorbar(pc, ax=ax, shrink=0.6).set_label(COV_LABELS[par])
    return fig


def plot_cov(reps, layer, par, pooled=True, nc=3):
    """One map per replicate (median over seasons) and/or the pooled median; runaway cells blank."""
    d = cov_plugins(reps, layer, km=True)
    def title(src, g):
        nb = int(g[par].isna().sum())
        return ((f"pooled ({len(reps)} replicates x 4 seasons)" if src == "pooled" else f"replicate {src}")
                + (f", {nb} runaway cells blank" if nb else ""))
    srcs = ([*reps] if pooled != "only" else []) + (["pooled"] if pooled else [])
    return plot_cov_maps([(title(src, g), g) for src in srcs for g in [d[d.rep.astype(str) == str(src)]]], par, nc)


def plot_cov_kuusela(reps, layer, par):
    """Our pooled median beside the Sukianto, Kuusela et al. (2026) real-Argo plug-in, on one colour scale."""
    d = cov_plugins(reps, layer, km=True)
    k = kuusela_plugins(layer, km=True)
    lo, hi = KUUSELA_LAYERS[layer].split("_")
    return plot_cov_maps([(f"GLORYS sim, pooled ({layer.split('_', 1)[1].replace('_', '-')} m)",
                           d[d.rep.astype(str) == "pooled"]),
                          (f"Kuusela, real Argo 2004-2022 ({lo}-{hi} dbar)", k)], par, nc=2)


def cov_summary(reps):
    """Per layer and parameter: the domain median of the pooled plug-ins, the range over replicates of the domain
    medians of their own plug-ins, the median over cells of the between-replicate coefficient of variation, and the
    count of blank (unfitted or runaway) replicate-cells."""
    rows = []
    for layer in LAYER_NAMES:
        d = cov_plugins(reps, layer, km=True)
        r = d[d.rep != "pooled"]
        for p in COV_PARAMS:
            g = r.groupby(["lat", "lon"])[p]
            m = r.groupby("rep")[p].median()
            rows.append(dict(layer=layer, parameter=p, pooled=d.loc[d.rep == "pooled", p].median(), rep_min=m.min(),
                             rep_max=m.max(), cv_between=(g.std() / g.mean()).median(), blank=int(r[p].isna().sum())))
    return pd.DataFrame(rows)


def cov_vs_gpgp(reps):
    """Exact vs Vecchia (GpGp, +/-45 d windows) pooled plug-ins of the same replicates: domain medians and the median
    over cells of the ratio exact/GpGp."""
    rows = []
    for layer in LAYER_NAMES:
        e, g = (x[x.rep == "pooled"].set_index(["lat", "lon"])[list(COV_PARAMS)]
                for x in (cov_plugins(reps, layer, km=True), cov_plugins(reps, layer, km=True, kind="cov")))
        for p in COV_PARAMS:
            rows.append(dict(layer=layer, parameter=p, exact=e[p].median(), gpgp=g[p].median(),
                             ratio=(e[p] / g[p]).median()))
    return pd.DataFrame(rows)


def covpool(reps):
    """Pooled plug-ins for kriging (median over the window fits of `reps`), in the krige_plugin column layout."""
    for layer in LAYER_NAMES:
        w = cov_windows(reps, layer)
        g = w.dropna(subset=[*COV_PARAMS, "nugget_ratio"]).groupby(["lat", "lon"])
        d = g[[*COV_PARAMS, "nugget_ratio"]].median().join(g.size().rename("n_fits")).reset_index()
        d.assign(box="natl", nugget=d.nugget_ratio * d.phi).to_csv(cov_pooled_path(layer, reps), index=False)


COV_POOL_REPS = range(10)   # kriging plug-ins are fitted on these; map only other replicates


def cov_pooled_path(layer, reps):
    return os.path.join(OUT, f"cov_pooled_{layer}_r{min(reps)}-{max(reps)}.csv")


def gpgp_pilot(n_windows=20):  # GpGp +/-45 d windows (superseded by exact fits)
    """GpGp on r0's residuals (snapshot) for a sample of cells x the ORACLE_MONTHS seasons per layer, plus the
    49N 21W January window on the oracle lattice, with the exact likelihood (krige_local_fit) at the GpGp estimate."""
    from osim import krige_local_fit as kf
    d = os.path.join(OUT, "pilot_gpgp")
    allc = pd.read_csv(os.path.join(OUT, "oracle_centres.csv"))
    cells = allc[["lat", "lon"]].drop_duplicates()
    cells = pd.concat([cells[(cells.lat == 49) & (cells.lon == -21)],
                       cells[(cells.lat != 49) | (cells.lon != -21)].sample(n_windows - 1, random_state=0)])
    cen = allc.merge(cells, on=["lat", "lon"])
    stuck = cen[(cen.lat == 49) & (cen.lon == -21) & (cen.month == 1)]
    for layer in LAYER_NAMES:
        with costlog.step(COSTS, f"gpgp_pilot_rep_{layer}", rep=0, n=len(cen)):
            pts = cov_points(0, layer, os.path.join(d, "profiles_r0.parquet"))
            gpgp_fit(pts, cen, os.path.join(d, f"r0_{layer}.csv"))
    lat_ = pd.read_csv(oracle_path("ohc_700_2000", "lattice"))
    with costlog.step(COSTS, "gpgp_pilot_oracle", rep="49N21W", n=1):
        g = gpgp_fit(lat_, stuck, os.path.join(d, "oracle_49N21W.csv")).iloc[0]
    years = sorted(lat_.year.unique())[1:]
    sub = lat_[lat_.lat.between(39, 59) & lat_.lon.between(-31, -11)]
    B = []
    for y in years:
        ts = (pd.Timestamp(f"{y}-01-15") - T0).days
        b = sub[sub.day.between(ts - 45, ts + 45)]
        B.append((b[["lon", "lat", "day"]].to_numpy(), b.z.to_numpy()))
    ll = lambda p: -kf._obj(np.log(p), B, grad=False)
    print(f"oracle 49N21W GpGp: th=({g.th_lon:.2f}, {g.th_lat:.2f}, {g.th_t:.0f}) nugget {g.nugget_ratio:.2e}, "
          f"{g.secs:.0f} s; exact loglik at GpGp estimate {ll([g.th_lon, g.th_lat, g.th_t, max(g.nugget_ratio, 1e-7)]):.1f} "
          f"vs exact MLE (scaled L-BFGS) {ll([1.4711, 0.9204, 62.4629, 1e-7]):.1f}", flush=True)


def map_path(rep, layer=""):
    return os.path.join(OUT, "maps", f"r{rep}", layer)


def run_map(rep, layer, months=None):
    """Mean field (full and leave-one-fold-out), kriging, and the domain-mean of every map per month; held-out
    residuals e = z - m_(-k) at the in-domain profiles. Depends on the field and covariance, not on q-hat."""
    if rep in COV_POOL_REPS and "NATL_SIM_PARAMS" not in os.environ:
        raise ValueError(f"rep {rep} was used to fit the pooled covariance plug-ins")
    from osim import chain as ch
    d = map_path(rep, layer)
    os.makedirs(d, exist_ok=True)
    months = months or pd.date_range(f"{cfg.ANALYSIS_YEARS[0]}-01", f"{cfg.ANALYSIS_YEARS[-1]}-12", freq="MS")
    params, nd = pd.read_csv(os.environ.get("NATL_SIM_PARAMS", cov_pooled_path(layer, COV_POOL_REPS)).format(
        layer=layer)), nodes()
    with costlog.step(COSTS, f"map_mean_{layer}", rep=rep) as meta:
        obs, mean = ch.fit_map(rep_obs(rep, layer))
        meta["n"] = len(obs)
    B, Pf = [], []
    with costlog.step(COSTS, f"map_krige_{layer}", rep=rep, n=len(months)):
        for m in months:
            bm, pf = ch.map_month(m, obs, mean, params, nd)
            B.append(bm)
            Pf.append(pf)
            print(f"  {m:%Y-%m}", flush=True)
    pd.concat(B, ignore_index=True).to_parquet(os.path.join(d, "maps.parquet"))
    pd.concat(Pf, ignore_index=True).merge(obs[["prof_id", "cycle", "cohort", "time"]], on="prof_id").to_parquet(
        os.path.join(d, "prof.parquet"))
    deployments(rep).to_parquet(os.path.join(map_path(rep), "folds.parquet"))


def jk_terms(h, fl):
    """Hajek correction C of one fold-year and its delete-one-float values C_j (floats in cohorts of >1).
    h: held-out profiles with v, e; fl: the fold's floats."""
    t = pd.DataFrame(dict(float_id=h.float_id.values, a=h.v.values * h.e.values, d=h.v.values)).groupby("float_id").sum()
    t = t.reindex(fl.float_id, fill_value=0.0).assign(cohort=fl.cohort.values)
    A, D = t.a.sum(), t.d.sum()
    C = A / D
    n = t.groupby("cohort").a.transform("size")
    t = t[n > 1].assign(n=n[n > 1])
    rest = D - t.d
    return C, t.assign(Cj=((A - t.a) / rest.where(rest > 0)).fillna(C))


def jk_summary(C, t):
    """Delete-one-float jackknife within cohort: Vjk and the terms c_j (their sums give the Satterthwaite df)."""
    cj = (t.n - 1) / t.n * (t.Cj - t.Cj.groupby(t.cohort).transform("mean")) ** 2
    return dict(C=C, Vjk=cj.sum(), c_sum=cj.sum(), c2_sum=(cj ** 2).sum())


def fold_stats(h, fl, keep=None):
    """jk_terms + jk_summary, with profile and float counts; `keep` (list) receives (C, t) for combining layers."""
    C, t = jk_terms(h, fl)
    if keep is not None:
        keep.append((C, t))
    return jk_summary(C, t) | dict(n=len(h), n_floats=int((t.d > 0).sum()))


def layer_maps(rep):
    """Each layer's maps and held-out profile residuals, plus their sum as layer "total" (all linear in z)."""
    M = {l: [pd.read_parquet(os.path.join(map_path(rep, l), f)) for f in ("maps.parquet", "prof.parquet")]
         for l in LAYER_NAMES}
    num = ["full", "mu", *[f"fold{k}" for k in range(K)]]
    key = ["box", "month"]
    m0, p0 = M[LAYER_NAMES[0]]
    tot_m = sum(M[l][0].set_index(key)[num] for l in LAYER_NAMES).reset_index()
    tot_p = p0.drop(columns=["z", "e"]).merge(sum(M[l][1].set_index("prof_id")[["z", "e"]] for l in LAYER_NAMES)
                                             .reset_index(), on="prof_id", validate="1:1")
    if len(tot_p) != len(p0) or tot_m.isna().any().any():
        raise ValueError(f"rep {rep}: layer maps do not align")
    return M | {"total": [tot_m, tot_p]}


def estimate(rep, sig=QSIG):
    """Per layer, per year of one replicate: naive, Hajek, mean field, map and cross-fit DR (jackknife variance).
    The "total" rows are the layer sums: identical to estimating from the summed field, by linearity."""
    fl = pd.read_parquet(os.path.join(map_path(rep), "folds.parquet"))
    tr = truth().set_index(["layer", "year"])
    out, folds = zip(*[estimate_layer(rep, l, maps, pf, fl, tr.loc[l], sig) for l, (maps, pf) in layer_maps(rep).items()])
    out, folds = pd.concat(out, ignore_index=True), pd.concat(folds, ignore_index=True)
    tag = os.path.basename(q_dir(sig))
    out.to_parquet(os.path.join(map_path(rep), f"estimates_{tag}.parquet"))
    folds.to_parquet(os.path.join(map_path(rep), f"fold_stats_{tag}.parquet"))
    return out


QSIG_LAYER = {"ohc_15_700": (1.0, 30.0), "ohc_700_2000": (0.3333, 5.0)}   # per-layer DR kernels (dr_bandwidth sweep)
QTAG = "q_layer"


def estimate_layers(rep, sigs=QSIG_LAYER, tag=QTAG):
    """estimate() with a q-hat kernel per layer. The "total" rows are the layer sums; DR's total variance is the
    jackknife of the summed per-float delete-one values, so it keeps the covariance between layers."""
    fl = pd.read_parquet(os.path.join(map_path(rep), "folds.parquet"))
    tr = truth().set_index(["layer", "year"])
    M, jk, out, folds = layer_maps(rep), {}, [], []
    for l in LAYER_NAMES:
        o, f = estimate_layer(rep, l, *M[l], fl, tr.loc[l], sigs[l], jk)
        out.append(o), folds.append(f)
    num = ["naive", "hajek", "mean_field", "krige", "dr"]
    tot = sum(o.set_index("year")[num] for o in out)
    fs = []
    for (y, k), parts in jk.items():
        C = sum(c for c, _ in parts)
        t = parts[0][1][["cohort", "n"]].assign(Cj=sum(t.Cj for _, t in parts))
        fs.append(dict(year=y, fold=k, **jk_summary(C, t)))
    fs = pd.DataFrame(fs).groupby("year")
    tot = tot.assign(rep=rep, layer="total", theta=tr.loc["total"].theta.reindex(tot.index),
                     dr_V=fs.Vjk.sum() / K ** 2, dr_df=fs.c_sum.sum() ** 2 / fs.c2_sum.sum(),
                     n=out[0].set_index("year").n, n_floats=out[0].set_index("year").n_floats).reset_index()
    out = pd.concat([*out, tot[out[0].columns]], ignore_index=True)
    out.to_parquet(os.path.join(map_path(rep), f"estimates_{tag}.parquet"))
    pd.concat(folds, ignore_index=True).to_parquet(os.path.join(map_path(rep), f"fold_stats_{tag}.parquet"))
    return out


def spatial_diag(reps, sigs=(QSIG_LAYER, {l: QSIG for l in LAYER_NAMES}), out=None, by_year=False):
    """Per layer, kernel set and 1 deg cell, pooled over `reps` x analysis years: the cell's area share a_j/A; the
    mean held-out residual e = z - mu_lofo - anom_lofo and its two parts; and, per fold-year averaged over folds,
    the cell's share of the 1/q-hat weight (r_j = share / area share) and of DR's correction C. Then
    dr_contrib = c_share - area_share * e_mean approximates the cell's contribution to DR's bias (exact up to the
    within-cell variation of e and the quadrature remainder) and map_contrib = -area_share * e_mean the map's.
    by_year: per year as well (shares then per replicate of that year)."""
    la, lo, m, a = domain()
    I, J = np.nonzero(m)
    cell = lambda lat, lon: (np.floor(lat) + 0.5, np.floor(lon) + 0.5)
    ar = pd.DataFrame(dict(zip(("lat", "lon"), cell(la[I], lo[J])), area=a[I, J])).groupby(["lat", "lon"]).area.sum()
    ar = (ar / ar.sum()).rename("area_share")
    E, W = [], []
    for r in reps:
        for l in LAYER_NAMES:
            p = pd.read_parquet(os.path.join(map_path(r, l), "prof.parquet"),
                                columns=["float_id", "cycle", "fold", "lat", "lon", "time", "z", "mu_lofo", "anom_lofo", "e"])
            p = p.assign(year=p.time.dt.year, emf=p.z - p.mu_lofo, lat=cell(p.lat, p.lon)[0], lon=cell(p.lat, p.lon)[1])
            gk = ["year", "lat", "lon"] if by_year else ["lat", "lon"]
            E.append(p.groupby(gk).agg(n=("e", "size"), e=("e", "sum"), emf=("emf", "sum"),
                                                   anom=("anom_lofo", "sum")).assign(layer=l))
            for i, sg in enumerate(sigs):
                q = pd.read_parquet(os.path.join(q_dir(sg[l]), f"r{r}.parquet"))
                x = p.merge(q, on=["float_id", "cycle"], validate="1:1").assign(v=lambda d: 1 / d.q)
                x["v"] /= x.groupby(["year", "fold"]).v.transform("sum") * K
                x["ve"] = x.v * x.e
                W.append(x.groupby(gk)[["v", "ve"]].sum().assign(layer=l, kernel=i))
    ny = len(reps) * (1 if by_year else len(cfg.ANALYSIS_YEARS))
    key = ["year", "lat", "lon"] if by_year else ["lat", "lon"]
    E = pd.concat(E).groupby(["layer", *key]).sum()
    E = E.assign(e_mean=E.e / E.n, emf_mean=E.emf / E.n, anom_mean=E.anom / E.n)[["n", "e_mean", "emf_mean", "anom_mean"]]
    W = (pd.concat(W).groupby(["layer", "kernel", *key]).sum() / ny).rename(columns=dict(v="w_share", ve="c_share"))
    d = W.reset_index().merge(E.reset_index(), on=["layer", *key], how="left").join(ar, on=["lat", "lon"])
    d["kernel"] = d.kernel.map(lambda i: ",".join(f"{sigs[i][l][0]:g}/{sigs[i][l][1]:g}" for l in LAYER_NAMES))
    d = d.assign(r=d.w_share / d.area_share, map_contrib=-d.area_share * d.e_mean * 1e-9,
                 dr_contrib=(d.c_share - d.area_share * d.e_mean) * 1e-9, e_mean=d.e_mean * 1e-9,
                 emf_mean=d.emf_mean * 1e-9, anom_mean=d.anom_mean * 1e-9, c_share=d.c_share * 1e-9)
    d.to_csv(out or os.path.join(OUT, "spatial_diag.csv"), index=False)
    return d


def estimate_layer(rep, layer, maps, pf, fl, tr, sig=QSIG, jk=None):
    q = pd.read_parquet(os.path.join(q_dir(sig), f"r{rep}.parquet"))
    pf = pf.merge(q, on=["float_id", "cycle"], how="inner", validate="1:1").assign(year=lambda x: x.time.dt.year)
    pf["v"] = pf["w"] = 1 / pf.q   # folds are stratified by cohort and C_k, Hajek are ratios: the scale cancels
    m = maps.assign(year=maps.month.dt.year, wd=maps.month.dt.days_in_month)
    cols = ["full", "mu", *[f"fold{k}" for k in range(K)]]
    ann = m.groupby("year").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.wd) for c in cols}))
    rows, folds = [], []
    for y, g in pf.groupby("year"):
        fk = [dict(layer=layer, year=y, fold=k, Q=ann.loc[y, f"fold{k}"],
                   **fold_stats(g[g.fold == k], fl[fl.fold == k], None if jk is None else jk.setdefault((y, k), [])))
              for k in range(K)]
        folds += fk
        fk = pd.DataFrame(fk)
        rows.append(dict(rep=rep, layer=layer, year=y, theta=tr.loc[y, "theta"], naive=g.z.mean(),
                         hajek=np.average(g.z, weights=g.w), mean_field=ann.loc[y, "mu"], krige=ann.loc[y, "full"],
                         dr=float((fk.Q + fk.C).mean()), dr_V=float(fk.Vjk.sum() / K ** 2),
                         dr_df=float(fk.c_sum.sum() ** 2 / fk.c2_sum.sum()), n=len(g), n_floats=g.float_id.nunique()))
    return pd.DataFrame(rows), pd.DataFrame(folds).assign(rep=rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "advect", "profiles", "select", "gpgp_pilot", "cov_exact_test", "uniform", "stationary", "covfit", "covgather", "covpool", "estimate_layers", "spatial_diag", "quadrature", "fields", "oracle_prepare", "sample", "qpoints", "map",
                                    "estimate", "estimate_layers", "spatial_diag", "qmaps"])
    ap.add_argument("--rep", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--reps", help="first,last (inclusive): sample/qpoints chunk; q-hat's evaluation reps")
    ap.add_argument("--all-reps", type=int, help="qpoints: q-hat from replicates 0..N-1")
    ap.add_argument("--year", type=int, help="map: only this year's months (timing pilot)")
    ap.add_argument("--layer", choices=LAYER_NAMES, action="append", help="oracle_prepare/map (repeatable; default all)")
    ap.add_argument("--sig", action="append", help="qpoints/estimate: kernel sd 'deg,days' (repeatable; default QSIG)")
    ap.add_argument("--dt-hours", type=int, default=1, help="advect: RK4 step; a non-default step writes profiles_dt{h}")
    a = ap.parse_args()
    rr = range(*(lambda x: (x[0], x[1] + 1))([int(v) for v in a.reps.split(",")])) if a.reps else [a.rep]
    layers = a.layer or LAYER_NAMES
    sigs = [tuple(float(v) for v in x.split(",")) for x in a.sig] if a.sig else [QSIG]
    {"design": design, "fields": fields, "oracle_prepare": lambda: [oracle_prepare(l) for l in layers], "advect": lambda: advect(a.rep, a.dt_hours, "" if a.dt_hours == 1 else f"_dt{a.dt_hours}"),
     "profiles": lambda: [profiles(r) for r in rr], "select": lambda: [select(r) for r in rr], "gpgp_pilot": gpgp_pilot, "cov_exact_test": cov_exact_test, "uniform": uniform_sampling, "stationary": stationary_sampling,
     "covfit": lambda: covfit(list(rr), a.rep), "covgather": lambda: [covgather(r) for r in rr], "covpool": lambda: covpool(list(rr)),
     "quadrature": quadrature, "sample": lambda: sample(rr), "qpoints": lambda: q_points(rr, range(a.all_reps or 200), sigs),
     "qmaps": lambda: q_points([], range(a.all_reps or 200), sigs, map_year=a.year or 2020),
     "map": lambda: [run_map(a.rep, l, a.year and list(pd.date_range(f"{a.year}-01", f"{a.year}-12", freq="MS")))
                     for l in layers], "estimate_layers": lambda: [estimate_layers(r) for r in rr if os.path.exists(os.path.join(map_path(r), "folds.parquet"))],
     "spatial_diag": lambda: (spatial_diag(list(rr)), spatial_diag(list(rr), out=os.path.join(OUT, "spatial_diag_year.csv"), by_year=True)), "estimate": lambda: [estimate(r, g) for g in sigs for r in rr if os.path.exists(os.path.join(map_path(r), "folds.parquet"))]}[a.cmd]()
