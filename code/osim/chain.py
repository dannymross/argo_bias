"""Identical-twin GLORYS chain (paper/sections/theory.tex): deployment-resample replicates read full 0-2000 m
GLORYS OHC; per box-year the naive mean, HT/Hajek, the two-stage map (local mean + local kriging, glorys_dr) and
cross-fit DR with cohort-matched, leave-one-replicate-out kernel q-hat.

    python code/osim/chain.py points --chunk 0 --nchunk 20   # in-box profiles: Z and cohort q-hats (in-sample, LORO)
    python code/osim/chain.py nodes             # cohort q-hats at the map nodes (coverage check)
    python code/osim/chain.py align             # Cov(Z, w-hat) on native cells
    python code/osim/chain.py rep --rep 3       # map + DR for one replicate
    python code/osim/chain.py summary
    CHAIN_SIGMA=0.08333,2 python code/osim/chain.py points --maxrep 200 ...   # bandwidth sensitivity
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import continuous_sim as cs
from osim import glorys_dr as gd
from osim import intensity as qi
from osim import ipw
from osim import osim_config as cfg

TAG = "_argo_resample"
POOL = os.path.join(cfg.OUT_DIR, "argo_resample_pool_with_init.csv")   # the pool the replicates were drawn from
PARAMS = os.path.join(cfg.OUT_DIR, "krige_plugin_argomean.csv")
OUT = os.path.join(cfg.OUT_DIR, "chain")
# "init": initial-state floats (alive on 2010-01-01) join every fold map's data, outside the folds; "deploy": not used
MAP = os.environ.get("CHAIN_MAP", "init")
RUN = os.path.join(OUT, MAP)
LEGACY = (1 / 6, 4.0)   # first production kernel sd (deg, days); its outputs carry no suffix
SIGMA = (1 / 6, 2.0)    # production q-hat, chosen from the bandwidth sensitivity
if os.environ.get("CHAIN_SIGMA"):   # "deg,days"
    SIGMA = tuple(float(x) for x in os.environ["CHAIN_SIGMA"].split(","))
KS = "" if SIGMA == LEGACY else f"_s{SIGMA[0]:.4g}_t{SIGMA[1]:g}"
TRUNC = 4.0
K = 5
YEARS = cfg.ANALYSIS_YEARS
BOXES = ("boxA", "boxB")
UNION = {"boxA": ("boxA",), "boxB": ("boxB",), "boxAB": BOXES}
SIG = np.array([SIGMA[1], SIGMA[0], SIGMA[0]])
NORM = (2 * np.pi) ** 1.5 * SIG.prod()


def rep_files():
    return sorted(glob.glob(cs.profile_path(TAG, chunk="*")), key=lambda f: int(f.rsplit("_c", 1)[1].split(".")[0]))


def pool():
    p = cs.read_resample_pool(POOL)
    return p[p.source == "deploy"]


def cohort_sizes():
    return pool().groupby("year").size()


def deployments(rep):
    """Every deployed float of a replicate with its cohort (launch year of its pool entry) and its fold, a random
    permutation within cohort."""
    p = cs.read_resample_pool(POOL)
    tab = cs.resample_seeds(p, rep, "2010-01-01", cs._velocity_end())[1]
    tab = tab[tab.source == "deploy"].merge(p[["pool_idx", "year"]], on="pool_idx").rename(columns={"year": "cohort"})
    return tab[["float_id", "cohort"]].assign(fold=ipw.folds(tab.float_id.to_numpy(), K, seed=rep,
                                                             strata=tab.cohort.to_numpy()))


@__import__("functools").lru_cache(1)
def _area():
    return xr.open_dataset(gd.AREA)["cell_area"].load()


def domain_box(lat, lon):
    """Box of the containing native cell if it is a deep (estimand) cell, else ''."""
    a = _area()
    i, j = cfg.containing_index(np.asarray(lat), a.latitude.values), cfg.containing_index(np.asarray(lon), a.longitude.values)
    ok = (i >= 0) & (j >= 0)
    i, j = np.where(ok, i, 0), np.where(ok, j, 0)
    out = np.full(len(i), "", object)
    deep = np.isfinite(a.values)
    for b, mk in gd.box_masks(a.latitude.values, a.longitude.values).items():
        if b in BOXES:
            out[ok & (mk & deep)[i, j]] = b
    return out


def sample_z(time, lat, lon):
    """GLORYS OHC (J/m2) at the containing native cell and day; NaN off the field."""
    t = pd.DatetimeIndex(time)
    z = np.full(len(t), np.nan)
    for (y, m), idx in pd.DataFrame(dict(y=t.year, m=t.month)).groupby(["y", "m"]).indices.items():
        f = gd.ohc_month(y, m)
        i = cfg.containing_index(np.asarray(lat)[idx], f.latitude.values)
        j = cfg.containing_index(np.asarray(lon)[idx], f.longitude.values)
        ok = (i >= 0) & (j >= 0)
        z[idx[ok]] = f.values[t.day.values[idx[ok]] - 1, i[ok], j[ok]]
    return z


def node_z(time, lat, lon):
    """Z at map nodes: mean of the native cells a quarter cell away in each quadrant. Node centroids sit on
    native cell corners, where the containing-cell lookup breaks ties to one side (a 1/24 deg shift)."""
    d = 1 / 48
    return np.nanmean([sample_z(time, np.asarray(lat) + s * d, np.asarray(lon) + t * d)
                       for s in (-1, 1) for t in (-1, 1)], axis=0)


def truth():
    """theta_y (J/m2) and mu(X_y) (m2 day) per box-year, boxes A, B and A u B."""
    d = pd.read_parquet(os.path.join(gd.BASE, "truth_daily.parquet"))
    d = d[d.time.dt.year.isin(YEARS)].assign(year=lambda x: x.time.dt.year)
    t = d.groupby(["box", "year"]).agg(theta=("ohc", "mean"), n_days=("ohc", "size"), area_m2=("area_m2", "first"))
    return t.assign(mu_T=t.area_m2 * t.n_days).reset_index()


# ------------------------------- kernel q-hat, cohort by cohort, leave one replicate out -------------------------------

def _unit(time, lat, lon):
    return np.column_stack([qi._days(np.asarray(time, "datetime64[s]")), lat, lon]) / SIG


def _psum(ev, src):
    return qi._pair_sum(ev, src, TRUNC) if len(ev) and len(src) else np.zeros(len(ev))


def _reach(lat, lon, time):
    r = TRUNC * SIGMA[0]
    la = np.array([cfg.BOXES[b]["lat_bounds"] for b in BOXES])
    lo = np.array([cfg.BOXES[b]["lon_bounds"] for b in BOXES])
    la0, la1, lo0, lo1 = la.min(), la.max(), lo.min(), lo.max()
    t0 = np.datetime64(f"{YEARS[0]}-01-01") - np.timedelta64(int(np.ceil(TRUNC * SIGMA[1])) + 2, "D")
    t1 = np.datetime64(f"{YEARS[-1] + 1}-01-01")
    return (lat >= la0 - r) & (lat <= la1 + r) & (lon >= lo0 - r) & (lon <= lo1 + r) & (time >= t0) & (time < t1)


def all_profiles():
    """Deployed-float profiles of every replicate within kernel reach of the box-years, with cohort and box."""
    yr = pool().set_index("pool_idx").year
    P = []
    for f in rep_files():
        p = pd.read_parquet(f, columns=["rep", "float_id", "cycle", "time", "lat", "lon", "pool_idx", "source"])
        p = p[p.source == "deploy"]
        t = pd.to_datetime(p.time).to_numpy().astype("datetime64[s]")
        P.append(p[_reach(p.lat.to_numpy(), p.lon.to_numpy(), t)].drop(columns="source"))
    P = pd.concat(P, ignore_index=True)
    P["time"] = pd.to_datetime(P.time)
    P["cohort"] = P.pool_idx.map(yr).astype(int)
    P["box"] = domain_box(P.lat.to_numpy(), P.lon.to_numpy())
    return P


def cohort_q(ev, P, own=None):
    """q-hat^(1)_c at unit-scaled points for every cohort c (profiles per m2 per day, (n, n_cohorts)) from all
    replicates in P; with own (the replicate of each point) also the leave-own-replicate-out version."""
    den = qi._support_mass(ev[:, 1] * SIG[1], ev[:, 2] * SIG[2], ev[:, 0] * SIG[0], *SIGMA, TRUNC) * NORM
    R, Nc = P.rep.nunique(), cohort_sizes()
    src = _unit(P.time.to_numpy(), P.lat.to_numpy(), P.lon.to_numpy())
    q_in, q_lo = np.zeros((len(ev), len(Nc))), np.zeros((len(ev), len(Nc)))
    for ci, c in enumerate(Nc.index):
        sc = P.cohort.to_numpy() == c
        num = _psum(ev, src[sc])
        q_in[:, ci] = num / (R * Nc[c])
        if own is not None:
            for r, idx in pd.Series(own).groupby(own).indices.items():
                num[idx] -= _psum(ev[idx], src[sc & (P.rep.to_numpy() == r)])
            q_lo[:, ci] = num / ((R - 1) * Nc[c])
    with np.errstate(invalid="ignore", divide="ignore"):
        q = [np.where(den[:, None] > 0, x / den[:, None], np.nan) for x in (q_in, q_lo)]
    return q[0] if own is None else q


def node_points():
    """Map nodes x time nodes of every month of YEARS, with box and area."""
    nodes = gd.map_nodes()
    rows = []
    for m0 in pd.date_range(f"{YEARS[0]}-01", f"{YEARS[-1]}-12", freq="MS"):
        d = (m0 - gd.EPOCH) / pd.Timedelta(days=1) + m0.days_in_month * np.asarray(gd.T_NODES)
        rows.append(nodes.loc[nodes.index.repeat(len(d))].assign(node=lambda x: x.index, day=np.tile(d, len(nodes)),
                                                                 month=m0))
    return pd.concat(rows, ignore_index=True)


def build_points(chunk=0, nchunk=1, maxrep=None):
    """In-box profiles (2015-2019) of the replicates r = chunk mod nchunk, with Z and the in-sample and LORO
    cohort q-hats and their array totals q = sum_c N_c q^(1)_c."""
    os.makedirs(os.path.join(OUT, "points" + KS), exist_ok=True)
    t0 = time.time()
    P = all_profiles()
    pts = P[(P.box != "") & P.time.dt.year.isin(YEARS) & (P.rep % nchunk == chunk)
            & (P.rep < (maxrep or np.inf))].reset_index(drop=True)
    print(f"{len(P)} profiles within reach; {len(pts)} points ({time.time() - t0:.0f} s)", flush=True)
    Nc = cohort_sizes()
    q_in, q1 = cohort_q(_unit(pts.time.to_numpy(), pts.lat.to_numpy(), pts.lon.to_numpy()), P, own=pts.rep.to_numpy())
    pts = pts.assign(**{f"q1_{c}": q1[:, i].astype(np.float32) for i, c in enumerate(Nc.index)},
                     q_loro=q1 @ Nc.to_numpy(float), q_in=q_in @ Nc.to_numpy(float))
    print(f"q-hats ({time.time() - t0:.0f} s)", flush=True)
    pts["z"] = sample_z(pts.time, pts.lat.to_numpy(), pts.lon.to_numpy())
    pts.drop(columns=["pool_idx"]).to_parquet(os.path.join(OUT, "points" + KS, f"c{chunk}.parquet"))
    return pts


def build_node_q():
    """In-sample cohort q-hats at the map nodes: the coverage check of Ass. cover."""
    P = all_profiles()
    nd = node_points()
    tt = (gd.EPOCH + pd.to_timedelta(nd.day, "D")).to_numpy()
    qn = cohort_q(_unit(tt, nd.lat.to_numpy(), nd.lon.to_numpy()), P)
    nd = nd.assign(**{f"q1_{c}": qn[:, i].astype(np.float32) for i, c in enumerate(cohort_sizes().index)})
    nd.to_parquet(os.path.join(OUT, f"node_q{KS}.parquet"))
    return nd


def load_points(reps=None, ks=KS):
    return pd.read_parquet(os.path.join(OUT, "points" + ks), filters=None if reps is None else [("rep", "in", list(reps))])


# --------------------------------------------- design-only estimators (1) ---------------------------------------------

def _years(d, col="time"):
    return d.assign(year=d[col].dt.year)


def _with_union(d):
    return pd.concat([d, d.assign(box="boxAB")], ignore_index=True)


def design_estimates(pts=None):
    """Naive, HT and Hajek (LORO and in-sample q-hat) per (rep, box, year), boxes A, B and A u B."""
    pts = _with_union(_years(load_points() if pts is None else pts))
    pts = pts.assign(v=1 / pts.q_loro, vi=1 / pts.q_in)
    pts = pts.assign(vz=pts.v * pts.z, viz=pts.vi * pts.z)
    e = pts.groupby(["rep", "box", "year"]).agg(n=("z", "size"), naive=("z", "mean"), sv=("v", "sum"), svz=("vz", "sum"),
                                                svi=("vi", "sum"), sviz=("viz", "sum"), n_floats=("float_id", "nunique"))
    e = e.reset_index().merge(truth(), on=["box", "year"])
    return e.assign(ht=e.svz / e.mu_T, hajek=e.svz / e.sv, hajek_in=e.sviz / e.svi)[
        ["rep", "box", "year", "theta", "n", "n_floats", "naive", "ht", "hajek", "hajek_in"]]


# ---------------------------------------------- alignment Cov(Z, w-hat) (1a) ----------------------------------------------

def alignment():
    """Per box-year on native cell-days: Cov_P(w-hat, Z) with w-hat = q-hat / mean q-hat (the binned all-replicate
    kernel), the realized minimum of q-hat / mean, and the replicate-pooled ratio sum Z / n, which equals
    theta + Cov_P(w-tilde, Z) for the pooled occupancy w-tilde exactly."""
    qd = qi.load_kernel(*LEGACY, TAG, deg="native")
    qv = qd.q_nominal.values
    a = _area()
    deep = np.isfinite(a.values)
    masks = {b: m & deep for b, m in gd.box_masks(a.latitude.values, a.longitude.values).items()}
    il = np.searchsorted(qd.lat_edge.values, a.latitude.values, side="right") - 1
    jl = np.searchsorted(qd.lon_edge.values, a.longitude.values, side="right") - 1
    rows = []
    for m0 in pd.date_range(f"{YEARS[0]}-01", f"{YEARS[-1]}-12", freq="MS"):
        z = gd.ohc_month(m0.year, m0.month)
        it = np.searchsorted(qd.time_edge.values, z.time.values.astype("datetime64[s]"), side="right") - 1
        for b, mk in masks.items():
            ii, jj = np.nonzero(mk)
            q = qv[it[:, None], il[ii][None], jl[jj][None]]
            zz, w = z.values[:, ii, jj], a.values[ii, jj][None]
            rows.append(dict(box=b, year=m0.year, S=float((w * np.ones_like(zz)).sum()), SZ=float((w * zz).sum()),
                             Sq=float((w * q).sum()), SqZ=float((w * q * zz).sum()), qmin=float(np.nanmin(q))))
        print(m0.strftime("%Y-%m"), flush=True)
    d = pd.DataFrame(rows).groupby(["box", "year"]).agg(S=("S", "sum"), SZ=("SZ", "sum"), Sq=("Sq", "sum"),
                                                        SqZ=("SqZ", "sum"), qmin=("qmin", "min")).reset_index()
    d = d.assign(theta=d.SZ / d.S, cov_w=d.SqZ / d.Sq - d.SZ / d.S, qmin_rel=d.qmin / (d.Sq / d.S))
    pts = _with_union(_years(load_points(ks="")))
    pooled = pts.groupby(["box", "year"]).z.mean().rename("pooled_ratio").reset_index()
    d = d.merge(pooled, on=["box", "year"]).assign(cov_w_occ=lambda x: x.pooled_ratio - x.theta)
    d.to_parquet(os.path.join(OUT, "alignment.parquet"))
    return d


# ------------------------------------------- map + cross-fit DR per replicate -------------------------------------------

def rep_obs(rep, dep):
    """A replicate's profiles over the conditioning buffer, 2010-2019, in the glorys_dr obs layout. Initial-state
    floats (MAP == "init") get fold = cohort = -1: in every fold map's data, never held out or 1/q-hat weighted."""
    p = pd.read_parquet(cs.profile_path(TAG, chunk=rep), columns=["float_id", "cycle", "time", "lat", "lon", "source"])
    p["time"] = pd.to_datetime(p.time)
    src = ("deploy", "init") if MAP == "init" else ("deploy",)
    p = p[p.source.isin(src) & gd.in_buffer(p) & p.time.dt.year.between(*gd.YEARS)].drop(columns="source")
    p = p.merge(dep, on="float_id", how="left").fillna({"fold": -1, "cohort": -1}).astype({"fold": int, "cohort": int})
    p["z"] = sample_z(p.time, p.lat.to_numpy(), p.lon.to_numpy())
    p = p.dropna(subset=["z"]).reset_index(drop=True)
    p["box"] = domain_box(p.lat.to_numpy(), p.lon.to_numpy())
    p["day"] = (p.time - gd.EPOCH) / pd.Timedelta(days=1)
    return p.reset_index(names="prof_id")


def fit_map(obs):
    """Full-data and leave-one-fold-out local mean field, and each profile's mean without its own fold."""
    full, lofo, gl = gd.fit_mean(obs, "fold")
    gl = pd.Index(gl)
    mu_lofo = np.full(len(obs), np.nan)
    for k, idx in obs.groupby("fold").indices.items():
        mu_lofo[idx] = gd.mean_at(lofo[gl.get_loc(k)], obs.lat.values[idx], obs.lon.values[idx], obs.day.values[idx])
    obs = obs.assign(mu=gd.mean_at(full, obs.lat.values, obs.lon.values, obs.day.values), mu_lofo=mu_lofo)
    return obs, (full, lofo, gl.to_numpy())


def node_month(month, obs, mean, params, nodes):
    """One month at the map nodes: full map, mean field and every fold map, plus the in-box profiles'
    own-fold-out residuals e_k = Z - m_{-k}."""
    nd, lf, pf = gd.krige_month(month, obs, params, nodes, mean=mean, group="fold")
    full, lofo, gl = mean
    nd = nd.assign(full=nd.mu + nd.anom)
    if len(lf):
        dk = lf.pivot_table(index=["node", "day"], columns="fold", values="d_anom", aggfunc="first")
    for i, k in enumerate(gl):
        if k < 0:
            continue
        d = (dk[k].reindex(pd.MultiIndex.from_frame(nd[["node", "day"]])).fillna(0.0).to_numpy()
             if len(lf) and k in dk else 0.0)
        nd[f"fold{k}"] = gd.mean_at(lofo[i], nd.lat.values, nd.lon.values, nd.day.values) + nd.anom.values + d
    pf = pf[pf.fold >= 0].assign(e=lambda x: x.z - x.mu_lofo - x.anom_lofo)
    return nd, pf


def map_month(month, obs, mean, params, nodes):
    """Box-month node averages of node_month's maps, and its in-box residuals."""
    nd, pf = node_month(month, obs, mean, params, nodes)
    cols = ["full", "mu", *[f"fold{k}" for k in mean[2] if k >= 0]]
    bm = nd.groupby("box").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.area) for c in cols}
                                                     | dict(area=g.area.sum() / len(gd.T_NODES))))
    return bm.reset_index().assign(month=pd.Timestamp(month)), pf


def run_rep(rep, months=None):
    """Map + DR inputs for one replicate: box-month map averages and in-box residuals, cached under OUT/r{rep}."""
    d = os.path.join(RUN, f"r{rep}")
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    dep = deployments(rep)
    obs, mean = fit_map(rep_obs(rep, dep))
    print(f"rep {rep}: {len(obs)} profiles, mean field fitted ({time.time() - t0:.0f} s)", flush=True)
    params, nodes = pd.read_csv(PARAMS), gd.map_nodes()
    months = pd.date_range(f"{YEARS[0]}-01", f"{YEARS[-1]}-12", freq="MS") if months is None else months
    B, Pf = [], []
    for m in months:
        bm, pf = map_month(m, obs, mean, params, nodes)
        B.append(bm)
        Pf.append(pf)
        print(f"  {m:%Y-%m} ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(B, ignore_index=True).to_parquet(os.path.join(d, "maps.parquet"))
    pf = pd.concat(Pf, ignore_index=True).merge(obs[["prof_id", "cycle", "cohort", "time"]], on="prof_id")
    pf.to_parquet(os.path.join(d, "prof.parquet"))
    dep.to_parquet(os.path.join(d, "deployments.parquet"))
    return estimate_rep(rep)


def node_maps(rep, year):
    """Annual node means of one replicate's full, mean-field and fold maps with the true Z at the same
    node-times and the production fold q-hat_k, cached under OUT/r{rep}/nodes_{year}.parquet."""
    f = os.path.join(RUN, f"r{rep}", f"nodes_{year}{KS}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    dep = deployments(rep)
    obs, mean = fit_map(rep_obs(rep, dep))
    params, nodes = pd.read_csv(PARAMS), gd.map_nodes()
    nd = pd.concat([node_month(m, obs, mean, params, nodes)[0]
                    for m in pd.date_range(f"{year}-01", f"{year}-12", freq="MS")], ignore_index=True)
    nd["z"] = node_z(gd.EPOCH + pd.to_timedelta(nd.day, "D"), nd.lat.to_numpy(), nd.lon.to_numpy())
    nq = pd.read_parquet(os.path.join(OUT, f"node_q{KS}.parquet"))
    nq = nq[nq.month.dt.year == year]
    Nc = cohort_sizes()
    n_kc = dep.groupby(["fold", "cohort"]).size().unstack(fill_value=0).reindex(columns=Nc.index, fill_value=0)
    q1 = nq[[f"q1_{c}" for c in Nc.index]].to_numpy(float)
    nq = nq[["node", "day"]].assign(q=q1 @ Nc.to_numpy(float), **{f"q{k}": q1 @ n_kc.loc[k].to_numpy(float)
                                                                  for k in n_kc.index})
    nd = nd.merge(nq, on=["node", "day"], how="left")
    cols = ["full", "mu", "z", "q", *[c for c in nd.columns if c.startswith("fold") or c[1:].isdigit()]]
    nd = nd.assign(w=nd.month.dt.days_in_month).groupby(["node", "lat", "lon", "box", "area"]).apply(
        lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in cols})).reset_index()
    nd.to_parquet(f)
    return nd


def _annual(maps, tr):
    """Day-weighted annual box means of the map columns; box A u B weighted by the boxes' estimand areas."""
    m = maps.assign(year=maps.month.dt.year, w=maps.month.dt.days_in_month)
    cols = [c for c in m.columns if c in ("full", "mu") or c.startswith("fold")]
    y = m.groupby(["box", "year"]).apply(lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in cols}))
    y = y.reset_index().merge(tr[["box", "year", "area_m2"]], on=["box", "year"])
    ab = y.groupby("year").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.area_m2) for c in cols}))
    return pd.concat([y.drop(columns="area_m2"), ab.reset_index().assign(box="boxAB")], ignore_index=True)


def rep_profiles(rep, pts=None):
    """A replicate's held-out in-box residuals (YEARS) with their weight v = 1 / q-hat_k for their own fold k,
    the deployments and the fold-by-cohort counts n_kc."""
    d = os.path.join(RUN, f"r{rep}")
    pf, dep = (pd.read_parquet(os.path.join(d, f)) for f in ("prof.parquet", "deployments.parquet"))
    pts = load_points([rep]) if pts is None else pts
    Nc = cohort_sizes()
    qc = [f"q1_{c}" for c in Nc.index]
    n_kc = dep.groupby(["fold", "cohort"]).size().unstack(fill_value=0).reindex(columns=Nc.index, fill_value=0)
    pf = pf[pf.time.dt.year.isin(YEARS)].merge(pts[["float_id", "cycle", *qc]], on=["float_id", "cycle"], how="left")
    if pf[qc[0]].isna().any():
        raise ValueError(f"rep {rep}: {pf[qc[0]].isna().sum()} in-box profiles without q-hat")
    pf["v"] = 1 / np.einsum("ij,ij->i", pf[qc].to_numpy(float), n_kc.loc[pf.fold].to_numpy(float))
    return _years(pf.drop(columns=qc)), dep, n_kc


def estimate_rep(rep, pts=None, nodeq=None):
    """Per box-year of one replicate: map, mean field and cross-fit DR with the float-cluster variance (one row);
    per fold: Q_y(m_{-k}), the Hajek correction, its variance, T_k and the coverage ratio (fold table)."""
    d = os.path.join(RUN, f"r{rep}")
    maps = pd.read_parquet(os.path.join(d, "maps.parquet"))
    pts = load_points([rep]) if pts is None else pts[pts.rep == rep]
    pf, dep, n_kc = rep_profiles(rep, pts)
    nodeq = pd.read_parquet(os.path.join(OUT, f"node_q{KS}.parquet")) if nodeq is None else nodeq
    tr = truth()
    qc = [f"q1_{c}" for c in cohort_sizes().index]
    pf, nodeq = _with_union(pf), _with_union(_years(nodeq, "month"))
    ann = _annual(maps, tr).set_index(["box", "year"])
    rows, folds = [], []
    for (b, y), g in pf.groupby(["box", "year"]):
        th = tr.set_index(["box", "year"]).loc[(b, y), "theta"]
        a = ann.loc[(b, y)]
        ng = nodeq[(nodeq.box == b) & (nodeq.year == y)]
        est = []
        for k in n_kc.index:
            nk = n_kc.loc[k].to_numpy(float)
            h = g[g.fold == k]
            v = h.v.to_numpy()
            C = float((v * h.e).sum() / v.sum()) if len(h) else 0.0
            u = pd.DataFrame(dict(float_id=h.float_id, u=v * (h.e.to_numpy() - C))).groupby("float_id").u.sum()
            u = u.reindex(dep.float_id[dep.fold == k], fill_value=0.0).to_frame().join(dep.set_index("float_id").cohort)
            V = float(sum(len(x) / (len(x) - 1) * ((x - x.mean()) ** 2).sum() for _, x in u.groupby("cohort").u
                          if len(x) > 1) / v.sum() ** 2) if len(h) else np.nan
            qn = ng[qc].to_numpy(float) @ nk
            Q = a[f"fold{k}"]
            folds.append(dict(rep=rep, box=b, year=y, fold=k, Q=Q, C=C, V=V, T=th - Q, n=len(h),
                              n_floats=h.float_id.nunique(), ess=v.sum() ** 2 / (v ** 2).sum() if len(h) else 0.0,
                              cover=float(qn.min() / np.average(qn, weights=ng.area))))
            est.append(Q + C)
        fk = pd.DataFrame(folds[-len(n_kc):])
        rows.append(dict(rep=rep, box=b, year=y, theta=th, krige=a["full"], mean_field=a["mu"], dr=float(np.mean(est)),
                         dr_V=float(fk.V.sum() / len(fk) ** 2), dr_fold_sd=float(np.std(est, ddof=1)),
                         n_inbox=len(g)))
    out = pd.DataFrame(rows).merge(design_estimates(pts), on=["rep", "box", "year", "theta"], how="left")
    out.to_parquet(os.path.join(d, f"estimates{KS}.parquet"))
    pd.DataFrame(folds).to_parquet(os.path.join(d, f"folds{KS}.parquet"))
    return out


def quadrature_error():
    """Q_y(Z) - theta_y (GJ/m2): the map quadrature applied to the true field, the remainder in the map's error
    that no interpolation can remove; cached."""
    f = os.path.join(OUT, "quadrature.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    nd = node_points()
    nd["z"] = node_z(gd.EPOCH + pd.to_timedelta(nd.day, "D"), nd.lat.to_numpy(), nd.lon.to_numpy())
    m = (nd.dropna(subset=["z"]).groupby(["box", "month"])
         .apply(lambda d: pd.Series(dict(full=np.average(d.z, weights=d.area), area=d.area.sum() / len(gd.T_NODES))))
         .reset_index())
    tr = truth()
    y = _annual(m, tr).merge(tr[["box", "year", "theta"]], on=["box", "year"])
    y = y.assign(quad_err=(y.full - y.theta) * 1e-9)[["box", "year", "quad_err"]]
    y.to_parquet(f)
    return y


# ------------------------------------------------------- summary -------------------------------------------------------

EST = ("naive", "hajek", "mean_field", "krige", "dr")


def load_estimates(ks=KS):
    e = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(RUN, "r*", f"estimates{ks}.parquet")))
    f = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(RUN, "r*", f"folds{ks}.parquet")))
    return e.sort_values(["rep", "box", "year"]).reset_index(drop=True), f


def _err_stats(x):
    n = len(x)
    return pd.Series(dict(R=n, bias=x.mean(), bias_se=x.std() / np.sqrt(n), sd=x.std(), mse=(x ** 2).mean(),
                          mse_se=(x ** 2).std() / np.sqrt(n), rmse=np.sqrt((x ** 2).mean())))


def summary(e=None):
    """Bias, sd, MSE (with MC se over replicates) per estimator and box-year, plus all years pooled (GJ/m2)."""
    e = load_estimates()[0] if e is None else e
    err = e.melt(id_vars=["rep", "box", "year"], value_vars=list(EST), var_name="estimator")
    err["value"] = (err.value - e.set_index(["rep", "box", "year"]).theta.reindex(
        pd.MultiIndex.from_frame(err[["rep", "box", "year"]])).to_numpy()) * 1e-9
    by = err.groupby(["box", "estimator", "year"]).value.apply(_err_stats).unstack()
    al = err.groupby(["box", "estimator"]).value.apply(_err_stats).unstack().assign(year="all")
    return pd.concat([by.reset_index(), al.reset_index()], ignore_index=True)


def paired(e=None):
    """DR minus map per replicate: mean and se of the difference, and of the MSE difference (map - DR), plus the
    coverage of dr +/- 1.96 sqrt(V-hat) and the map test C / sqrt(V)."""
    e = load_estimates()[0] if e is None else e
    x = e.assign(D=(e.dr - e.krige) * 1e-9, dMSE=((e.krige - e.theta) ** 2 - (e.dr - e.theta) ** 2) * 1e-18,
                 cover=(np.abs(e.dr - e.theta) <= 1.96 * np.sqrt(e.dr_V)).astype(float),
                 z_map=(e.dr - e.krige) / np.sqrt(e.dr_V))
    f = lambda g: pd.Series(dict(R=len(g), D=g.D.mean(), D_se=g.D.std() / np.sqrt(len(g)), dMSE=g.dMSE.mean(),
                                 dMSE_se=g.dMSE.std() / np.sqrt(len(g)), coverage=g.cover.mean(),
                                 z_map_mean=g.z_map.mean(), z_map_sd=g.z_map.std()))
    by = x.groupby(["box", "year"]).apply(f).reset_index()
    return pd.concat([by, x.groupby("box").apply(f).reset_index().assign(year="all")], ignore_index=True)


def mse_terms(f=None):
    """Prop. mse terms per box-year (GJ2/m4): E T^2 for the fold maps, E (C - T)^2 = E beta^2 + E V, mean V-hat."""
    f = load_estimates()[1] if f is None else f
    x = f.assign(T2=(f["T"] * 1e-9) ** 2, CT2=((f.C - f["T"]) * 1e-9) ** 2, Vh=f.V * 1e-18, CT=(f.C - f["T"]) * 1e-9)
    agg = dict(ET2=("T2", "mean"), E_CT2=("CT2", "mean"), E_V=("Vh", "mean"), mean_CT=("CT", "mean"),
               cover_min=("cover", "min"), ess=("ess", "mean"))
    return pd.concat([x.groupby(["box", "year"]).agg(**agg).reset_index(),
                      x.groupby("box").agg(**agg).reset_index().assign(year="all")], ignore_index=True)


def _paired_err(x, k, th):
    return dict(R=len(th), map_bias=(x["krige"] - th).mean(), dr_bias=(x["dr"] - th).mean(),
                map_rmse=np.sqrt(((x["krige"] - th) ** 2).mean()), dr_rmse=np.sqrt(((x["dr"] - th) ** 2).mean()),
                hajek_rmse=np.sqrt(((x["hajek"] - th) ** 2).mean()),
                dMSE=((x["krige"] - th) ** 2 - (x["dr"] - th) ** 2).mean(),
                dMSE_se=((x["krige"] - th) ** 2 - (x["dr"] - th) ** 2).std() / np.sqrt(len(th)), target=k)


def derived(e=None):
    """Map vs DR for linear functionals of the annual means within a replicate: year-to-year changes, the
    2015-2019 change and the day-weighted five-year mean (GJ/m2)."""
    e = load_estimates()[0] if e is None else e
    d = e.set_index(["rep", "box", "year"])[["theta", "krige", "dr", "hajek"]].mul(1e-9).unstack("year")
    days = np.array([pd.Timestamp(f"{y}-12-31").dayofyear for y in YEARS], float)
    funs = {f"{y}-{y + 1}": (lambda v, y=y: v[y + 1] - v[y]) for y in YEARS[:-1]}
    funs[f"{YEARS[0]}-{YEARS[-1]}"] = lambda v: v[YEARS[-1]] - v[YEARS[0]]
    funs[f"{YEARS[0]}-{YEARS[-1]} mean"] = lambda v: v[list(YEARS)] @ days / days.sum()
    rows = []
    for b, g in d.groupby(level="box"):
        for k, fn in funs.items():
            x = {c: fn(g[c]) for c in ("theta", "krige", "dr", "hajek")}
            rows.append(dict(box=b, **_paired_err(x, k, x["theta"])))
    return pd.DataFrame(rows)[["box", "target", "R", "map_bias", "dr_bias", "map_rmse", "dr_rmse", "hajek_rmse",
                               "dMSE", "dMSE_se"]]


def bandwidth_sensitivity(maxrep=200):
    """DR and Hajek error, coverage and variance ratio per kernel bandwidth with estimates on disk, pooled over
    years, plus the fold-level coverage ratio and ESS."""
    sfx = {re.search(r"estimates(.*)\.parquet", os.path.basename(f)).group(1)
           for f in glob.glob(os.path.join(RUN, "r0", "estimates*.parquet"))}
    rows = []
    for k in sorted(sfx):
        e, f = load_estimates(k)
        e, f = e[e.rep < maxrep], f[f.rep < maxrep]
        s, t = (k or f"_s{LEGACY[0]:.4g}_t{LEGACY[1]:g}")[2:].split("_t")
        for b, g in e.groupby("box"):
            err, hj = (g.dr - g.theta) * 1e-9, (g.hajek - g.theta) * 1e-9
            fb = f[f.box == b]
            rows.append(dict(sigma_deg=float(s), sigma_days=float(t), box=b, dr_bias=err.mean(), dr_bias_se=err.sem(),
                             dr_rmse=np.sqrt((err ** 2).mean()), hajek_bias=hj.mean(), hajek_rmse=np.sqrt((hj ** 2).mean()),
                             coverage=(err.abs() <= 1.96e-9 * np.sqrt(g.dr_V)).mean(),
                             var_ratio=err.var() / (g.dr_V * 1e-18).mean(), cover_min=fb.cover.min(), ess=fb.ess.mean()))
    return pd.DataFrame(rows).sort_values(["box", "sigma_deg", "sigma_days"]).reset_index(drop=True)


def plot_errors(e, est_lab, box, split=("naive",), figsize=(11, 6.5)):
    """Replicate errors (GJ/m2) of one box per estimator, grouped by year, box plots over jittered replicate
    points; the split estimators get their own top panel and scale."""
    import matplotlib.pyplot as plt
    groups = [{k: v for k, v in est_lab.items() if k in split}, {k: v for k, v in est_lab.items() if k not in split}]
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=(1, 1.6))
    x, years = e[e.box == box], sorted(e.year.unique())
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n, j0 = len(est_lab), 0
    rng = np.random.default_rng(0)
    for a, grp in zip(ax, groups):
        for j, (c, lab) in enumerate(grp.items(), start=j0):
            pos = np.arange(len(years)) * (n + 1) + j
            d = [((x[x.year == y][c] - x[x.year == y].theta) * 1e-9).to_numpy() for y in years]
            bp = a.boxplot(d, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set(facecolor=cols[j], alpha=0.35)
            for p, v in zip(pos, d):
                a.plot(p + rng.uniform(-0.25, 0.25, len(v)), v, ".", ms=2.5, color=cols[j], alpha=0.4)
            a.plot(pos, [v.mean() for v in d], "D", color=cols[j], ms=4, label=lab)
        j0 += len(grp)
        a.axhline(0, color="#c0392b", lw=0.8)
        a.set_ylabel("estimate - truth (GJ/m2)")
    ax[-1].set_xticks(np.arange(len(years)) * (n + 1) + (n - 1) / 2, years)
    fig.legend(loc="upper center", ncol=n, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fold_means(maxrep=200):
    """Per replicate, box-year and fold: the unweighted mean of the held-out residuals next to the Hajek
    correction C_k and its target T_k (GJ/m2), cached."""
    f = os.path.join(OUT, f"fold_means{KS}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    out = []
    for r in range(maxrep):
        pf = _with_union(rep_profiles(r)[0])
        m = pf.groupby(["box", "year", "fold"]).e.mean().mul(1e-9).rename("e_mean").reset_index()
        fk = pd.read_parquet(os.path.join(RUN, f"r{r}", f"folds{KS}.parquet"))
        out.append(fk.merge(m, on=["box", "year", "fold"]).assign(C=lambda d: d.C * 1e-9, T=lambda d: d["T"] * 1e-9))
    out = pd.concat(out, ignore_index=True)[["rep", "box", "year", "fold", "T", "C", "e_mean", "n", "ess"]]
    out.to_parquet(f)
    return out


def residual_cells(maxrep=200):
    """Held-out residuals pooled over replicates and years on the 0.5 deg node cells: count, mean residual,
    mean relative weight r = v / mean v within its box-year-fold, and the sum of (r - 1) e, the cell's share
    of C_k minus the unweighted mean (GJ/m2), cached."""
    f = os.path.join(OUT, f"residual_cells{KS}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    out = []
    for r in range(maxrep):
        pf = rep_profiles(r)[0]
        g = pf.groupby(["box", "year", "fold"])
        pf = pf.assign(rel=pf.v / g.v.transform("mean"), e=pf.e * 1e-9, cnt=g.e.transform("size"))
        out.append(pf.assign(shift=(pf.rel - 1) * pf.e / pf.cnt)[["box", "lat", "lon", "e", "rel", "shift"]])
    p = pd.concat(out, ignore_index=True)
    p["lat"] = (np.floor(p.lat / gd.NODE_DEG) + 0.5) * gd.NODE_DEG
    p["lon"] = (np.floor(p.lon / gd.NODE_DEG) + 0.5) * gd.NODE_DEG
    c = p.groupby(["box", "lat", "lon"]).agg(n=("e", "size"), e=("e", "mean"), rel=("rel", "mean"),
                                            shift=("shift", "sum")).reset_index()
    c["shift"] /= maxrep * len(YEARS) * K
    c.to_parquet(f)
    return c


def plot_residual_cells(c, n_min=30, figsize=(14, 4.8)):
    """Per 0.5 deg cell: the mean relative weight 1 / q-hat_k, the mean held-out map residual, and the cell's
    contribution to the average gap between C_k and the unweighted residual mean."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    c = c[c.n >= n_min]
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    xe, ye, g = _cells(c, c.rel)
    lim = np.nanmax(np.abs(np.log(g)))
    m = ax[0].pcolormesh(xe, ye, g, cmap="PuOr_r", norm=LogNorm(np.exp(-lim), np.exp(lim)))
    fig.colorbar(m, ax=ax[0], shrink=0.7, orientation="horizontal", label="relative weight (log scale)")
    for a, v, lab in ((ax[1], c.e, "mean residual Z - m(-k) (GJ/m2)"),
                      (ax[2], c["shift"] * 1e3, "contribution to C_k - mean e (1e-3 GJ/m2)")):
        xe, ye, g = _cells(c, v)
        lim = np.nanquantile(np.abs(g), 0.99)
        m = a.pcolormesh(xe, ye, g, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
        fig.colorbar(m, ax=a, shrink=0.7, orientation="horizontal", label=lab)
    for a, t in zip(ax, ("weight 1/q-hat_k relative to its fold mean", "held-out map residual",
                         "reweighting shift")):
        _box_frame(a)
        a.set_title(t, fontsize=10)
    for a in ax[1:]:
        a.set_ylabel("")
    return fig


def _cells(nd, v):
    """Node values on the regular 0.5 deg cell grid (edges, 2-d array) for pcolormesh."""
    i = np.floor(nd.lat.to_numpy() / gd.NODE_DEG).astype(int)
    j = np.floor(nd.lon.to_numpy() / gd.NODE_DEG).astype(int)
    g = np.full((i.max() - i.min() + 1, j.max() - j.min() + 1), np.nan)
    g[i - i.min(), j - j.min()] = np.asarray(v)
    return (np.arange(j.min(), j.max() + 2) * gd.NODE_DEG, np.arange(i.min(), i.max() + 2) * gd.NODE_DEG, g)


def _box_frame(a, boxes=BOXES):
    from matplotlib.patches import Rectangle
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        a.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec="k", lw=0.8))
    a.set_aspect(1 / np.cos(np.deg2rad(34)))
    a.set_xlabel("lon")
    a.set_ylabel("lat")


def plot_intensity(nd, prof, figsize=(12, 5.5)):
    """Annual-mean true Z at the map nodes with one replicate's in-box profiles, and the production
    w-hat = q-hat / mean q-hat over A u B with Z contours."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    xe, ye, z = _cells(nd, nd.z * 1e-9)
    m = ax[0].pcolormesh(xe, ye, z, cmap="viridis")
    ax[0].plot(prof.lon, prof.lat, ".", color="w", ms=2.5, mec="k", mew=0.2)
    fig.colorbar(m, ax=ax[0], shrink=0.8, label="annual-mean OHC (GJ/m2)")
    w = nd.q / np.average(nd.q, weights=nd.area)
    xe, ye, g = _cells(nd, w)
    lim = np.nanmax(np.abs(np.log(g)))
    m = ax[1].pcolormesh(xe, ye, g, cmap="magma", norm=LogNorm(np.exp(-lim), np.exp(lim)))
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    cs_ = ax[1].contour(xc, yc, z, levels=np.arange(60, 90, 5), colors="w", linewidths=0.6)
    ax[1].clabel(cs_, fontsize=7, fmt="%d")
    fig.colorbar(m, ax=ax[1], shrink=0.8, label="w-hat = q-hat / mean (log scale)")
    for a in ax:
        _box_frame(a)
    fig.tight_layout()
    return fig


def plot_advection(rep, n_floats=80, figsize=(8.5, 7.2)):
    """A random subset of one replicate's deployed floats, 2010-2019 tracks at the 10-day profiles over the
    velocity domain, coloured by profile year."""
    from osim import report as rp
    p = pd.read_parquet(cs.profile_path(TAG, chunk=rep), columns=["float_id", "time", "lat", "lon"])
    return rp._trajectory_map(p.assign(year=p.time.dt.year), max_floats=n_floats, figsize=figsize,
                              years=list(range(2010, 2020)), subset_seed=rep)


def plot_deployments(figsize=(8.5, 7.2)):
    """The deployment pool: every launch position, coloured by launch year, on the domain base map."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from osim import report as rp
    d, yr = pool(), list(range(YEARS[0] - 5, YEARS[-1] + 1))
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)
    fig, ax, pc, _ = rp._domain_base(BOXES, figsize)
    ax.scatter(d.lon, d.lat, c=d.year, cmap=cmap, norm=norm, s=14, edgecolor="k", linewidth=0.25, transform=pc,
               zorder=3)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=yr, shrink=0.8, pad=0.02,
                 label="launch year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def plot_deployment_draw(rep, figsize=(8.5, 7.2)):
    """One replicate's bootstrap draw: the real launch positions (grey, transparent) and the replicate's jittered
    seeds (solid, coloured by launch-year cohort)."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from osim import report as rp
    p = cs.read_resample_pool(POOL)
    tab = cs.resample_seeds(p, rep, "2010-01-01", cs._velocity_end())[1]
    tab = tab[tab.source == "deploy"].merge(p[["pool_idx", "year"]], on="pool_idx")
    real, yr = pool(), list(range(YEARS[0] - 5, YEARS[-1] + 1))
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)
    fig, ax, pc, _ = rp._domain_base(BOXES, figsize)
    ax.scatter(real.lon, real.lat, s=30, color="0.45", alpha=0.3, lw=0, transform=pc, zorder=3,
               label=f"real launches ({len(real)})")
    ax.scatter(tab.seed_lon, tab.seed_lat, c=tab.year, cmap=cmap, norm=norm, s=9, edgecolor="k", linewidth=0.2,
               transform=pc, zorder=4, label=f"replicate {rep} ({tab.pool_idx.nunique()} distinct launches)")
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=yr, shrink=0.8, pad=0.02,
                 label="launch year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def profile_density(deg=1.0):
    """Deployed-float profiles of 2015-2019 per replicate, year and deg x deg cell over the velocity domain, all
    replicates, cached."""
    f = os.path.join(OUT, f"profile_density_{deg:g}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    out = []
    for fn in rep_files():
        p = pd.read_parquet(fn, columns=["time", "lat", "lon"])
        p = p[p.time.dt.year.isin(YEARS)]
        out.append(pd.DataFrame(dict(lat=(np.floor(p.lat / deg) + 0.5) * deg, lon=(np.floor(p.lon / deg) + 0.5) * deg))
                   .value_counts().rename("n"))
    d = pd.concat(out).groupby(level=[0, 1]).sum().div(len(out) * len(YEARS)).rename("n").reset_index()
    d.to_parquet(f)
    return d


def plot_profile_density(d, deg=1.0, span=4.0, figsize=(8.5, 7.2)):
    """Mean profiles per replicate and year in each cell, 2015-2019, relative to the mean over sampled cells, on a
    diverging log scale centred at 1, on the domain base map."""
    from matplotlib.colors import LogNorm
    from osim import report as rp
    fig, ax, pc, _ = rp._domain_base(BOXES, figsize)
    la, lo = (np.arange(d[c].min(), d[c].max() + deg / 2, deg) for c in ("lat", "lon"))
    g = d.pivot(index="lat", columns="lon", values="n").reindex(index=la, columns=lo, method="nearest", tolerance=deg / 4)
    r = g.to_numpy() / d.n.mean()
    m = ax.pcolormesh(lo, la, r, cmap="RdBu_r", transform=pc, norm=LogNorm(1 / span, span), zorder=1,
                      shading="nearest")
    fig.colorbar(m, ax=ax, shrink=0.8, pad=0.02, extend="both",
                 label=f"profiles per {deg:g} deg cell relative to the mean ({d.n.mean():.1f} per year), log scale")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def plot_intensity_grid(nq=None, figsize=(12, 13)):
    """Quarterly-mean design intensity q-hat = sum_c N_c q-hat_c, years by quarters, relative to its
    area- and day-weighted mean over A u B in that year."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    nq = pd.read_parquet(os.path.join(OUT, f"node_q{KS}.parquet")) if nq is None else nq
    nq = nq[nq.box != "boxAB"]
    Nc = cohort_sizes()
    nq = nq.assign(q=nq[[f"q1_{c}" for c in Nc.index]].to_numpy(float) @ Nc.to_numpy(float),
                   year=nq.month.dt.year, qtr=nq.month.dt.quarter, w=nq.month.dt.days_in_month)
    qbar = nq.groupby("year").apply(lambda g: np.average(g.q, weights=g.w * g.area))
    g = nq.assign(qw=nq.q * nq.w).groupby(["year", "qtr", "node", "lat", "lon"])[["qw", "w"]].sum().reset_index()
    g["wh"] = g.qw / g.w / g.year.map(qbar)
    pos = g.wh[g.wh > 0]
    lim = np.abs(np.log(np.nanquantile(pos, [0.005, 0.995]))).max()
    norm = LogNorm(np.exp(-lim), np.exp(lim))
    years = sorted(g.year.unique())
    fig, ax = plt.subplots(len(years), 4, figsize=figsize, sharex=True, sharey=True)
    for i, y in enumerate(years):
        for j in range(4):
            d = g[(g.year == y) & (g.qtr == j + 1)]
            xe, ye, v = _cells(d, np.where(d.wh > 0, d.wh, np.nan))
            m = ax[i, j].pcolormesh(xe, ye, v, cmap="magma", norm=norm)
            _box_frame(ax[i, j])
            ax[i, j].set(xlabel="", ylabel="")
            if i == 0:
                ax[i, j].set_title(("Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec")[j], fontsize=10)
        ax[i, 0].set_ylabel(f"{y}\nlat")
    fig.colorbar(m, ax=ax, shrink=0.5, label="w-hat = q-hat / annual mean (log scale)")
    return fig


def plot_dr_example(nd, pf, fm, k, box="boxB", figsize=(13, 4.8)):
    """One replicate-year: the fold-k map error field Z - m_{-k}, the held-out fold-k residuals sized by their
    weight 1 / q-hat_k, and per fold the target T_k, the unweighted residual mean and the Hajek correction C_k."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    nb, h = nd[nd.box == box], pf[(pf.box == box) & (pf.fold == k)]
    r = (nb.z - nb[f"fold{k}"]) * 1e-9
    lim = np.nanquantile(np.abs(r), 0.98)
    norm = TwoSlopeNorm(0, -lim, lim)
    fig, ax = plt.subplots(1, 3, figsize=figsize, width_ratios=(1, 1, 0.8))
    xe, ye, g = _cells(nb, r)
    m = ax[0].pcolormesh(xe, ye, g, cmap="RdBu_r", norm=norm)
    ax[0].set_title(f"Z - m(-{k}) at the nodes", fontsize=10)
    ax[1].pcolormesh(xe, ye, g, cmap="RdBu_r", norm=norm, alpha=0.25)
    sz = 40 * h.v / h.v.mean()
    ax[1].scatter(h.lon, h.lat, c=h.e * 1e-9, s=sz, cmap="RdBu_r", norm=norm, ec="k", lw=0.3)
    ax[1].set_title(f"fold {k} residuals e, size ~ 1/q-hat", fontsize=10)
    fig.colorbar(m, ax=ax[:2], shrink=0.8, label="GJ/m2")
    for a in ax[:2]:
        _box_frame(a, (box,))
        (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
        a.set(xlim=(lo0 - 0.5, lo1 + 0.5), ylim=(la0 - 0.5, la1 + 0.5))
    f = fm[fm.box == box].sort_values("fold")
    f = pd.concat([f, f[["T", "e_mean", "C"]].mean().to_frame().T.assign(fold="mean")], ignore_index=True)
    x = np.arange(len(f)) + np.r_[np.zeros(len(f) - 1), 0.5]
    for dx, c, lab, col in ((-0.25, "T", "T_k (truth)", "k"), (0, "e_mean", "unweighted mean e", "#999999"),
                            (0.25, "C", "C_k (1/q-hat weighted)", "#1f77b4")):
        ax[2].bar(x + dx, f[c], width=0.25, color=col, label=lab)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].axvline(x[-1] - 0.75, color="0.6", lw=0.6, ls=":")
    ax[2].set_xticks(x, [f"fold {i}" if i != "mean" else "mean over\nfolds" for i in f.fold], fontsize=8)
    ax[2].set_ylabel("GJ/m2")
    ax[2].legend(fontsize=8, frameon=False)
    return fig


def plot_corrections(fm, box_lab, figsize=(12, 4.2)):
    """Per replicate-year, the fold-averaged map error -T-bar and the errors left after adding the fold-averaged
    unweighted residual mean or C-bar; the last is the DR error exactly."""
    import matplotlib.pyplot as plt
    g = fm.groupby(["rep", "box", "year"])[["T", "e_mean", "C"]].mean().reset_index()
    cols = (("fold maps: -T", "#1f77b4", lambda d: -d["T"]), ("+ unweighted mean e", "#999999", lambda d: d.e_mean - d["T"]),
            ("+ C (DR)", "#d62728", lambda d: d.C - d["T"]))
    fig, ax = plt.subplots(1, len(box_lab), figsize=figsize, sharey=True)
    rng = np.random.default_rng(0)
    for a, b in zip(ax, box_lab):
        d = g[g.box == b]
        for i, (lab, col, fn) in enumerate(cols):
            v = fn(d).to_numpy()
            bp = a.boxplot([v], positions=[i], widths=0.6, patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=col, alpha=0.35)
            a.plot(i + rng.uniform(-0.2, 0.2, len(v)), v, ".", ms=2.5, color=col, alpha=0.4)
            a.plot(i, v.mean(), "D", color="k", ms=5)
            a.text(i, 1.0, f"bias {v.mean():+.3f}\nsd {v.std():.3f}", transform=a.get_xaxis_transform(), ha="center",
                   va="bottom", fontsize=7)
        a.axhline(0, color="k", lw=0.6)
        a.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=8)
        a.set_title(box_lab[b], pad=24)
    ax[0].set_ylabel("error (GJ/m2)")
    fig.tight_layout()
    return fig


def monthly_maps(maxrep=200):
    """Monthly box means of the full map and the mean field alone per replicate, with the true box-month mean
    (GJ/m2), cached."""
    f = os.path.join(OUT, f"monthly_maps{KS}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    m = pd.concat(pd.read_parquet(os.path.join(RUN, f"r{r}", "maps.parquet"), columns=["box", "month", "full", "mu"])
                  .assign(rep=r) for r in range(maxrep))
    tr = pd.read_parquet(os.path.join(gd.OUT, "truth_monthly.parquet"))[["box", "month", "ohc"]]
    m = m.merge(tr, on=["box", "month"])
    m[["full", "mu", "ohc"]] *= 1e-9
    m.to_parquet(f)
    return m


def plot_mean_series(m, box_lab, figsize=(11, 6.5)):
    """Monthly true box mean (solid) and the replicate median mean field (dashed, 5-95% band), and below the
    mean field's and the map's error (median and 5-95% band over replicates)."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=(2, 1.2))
    q = m.groupby(["box", "month"]).agg(ohc=("ohc", "first"), **{f"{c}_{p}": (c, lambda x, p=p: x.quantile(p))
                                                                   for c in ("mu", "full") for p in (0.05, 0.5, 0.95)})
    e = m.assign(mu=m.mu - m.ohc, full=m.full - m.ohc).groupby(["box", "month"])[["mu", "full"]].quantile(
        [0.05, 0.5, 0.95]).unstack()
    for i, b in enumerate(box_lab):
        x, c = q.loc[b], f"C{i}"
        a1.plot(x.index, x.ohc, color=c, lw=1.3, label=f"{box_lab[b]} truth")
        a1.plot(x.index, x["mu_0.5"], color=c, lw=1.4, ls="--", label=f"{box_lab[b]} mean field")
        a1.fill_between(x.index, x["mu_0.05"], x["mu_0.95"], color=c, alpha=0.2, lw=0)
        y = e.loc[b]
        a2.plot(y.index, y[("mu", 0.5)], color=c, ls="--", lw=1.2, label=f"{box_lab[b]} mean field")
        a2.fill_between(y.index, y[("mu", 0.05)], y[("mu", 0.95)], color=c, alpha=0.15, lw=0)
        a2.plot(y.index, y[("full", 0.5)], color=c, lw=1.2, label=f"{box_lab[b]} map")
    a1.set_ylabel("box-mean OHC 0-2000 m (GJ/m2)")
    a1.legend(fontsize=8, ncol=2, frameon=False)
    a2.axhline(0, color="k", lw=0.7)
    a2.set(ylabel="estimate - truth (GJ/m2)", xlabel="month")
    a2.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


EST_COL = {"naive": "#7f7f7f", "hajek": "#9467bd", "mean_field": "#2ca02c", "krige": "#1f77b4", "dr": "#d62728"}


def plot_naive_bias(de, al, box_lab, figsize=(7, 4)):
    """Naive bias per box and year (+/- 2 MC se) with the kernel alignment Cov(w-hat, Z)."""
    import matplotlib.pyplot as plt
    nb = de.assign(b=(de.naive - de.theta) * 1e-9).groupby(["box", "year"]).b.agg(["mean", "sem"]).reset_index()
    fig, a = plt.subplots(figsize=figsize)
    for i, b in enumerate(box_lab):
        x, c = nb[nb.box == b], f"C{i}"
        a.errorbar(x.year, x["mean"], 2 * x["sem"], marker="o", color=c, capsize=3, label=f"{box_lab[b]}: naive bias")
        y = al[al.box == b]
        a.plot(y.year, y.cov_w * 1e-9, "x--", color=c, ms=7, lw=0.8, label=f"{box_lab[b]}: Cov(w-hat, Z)")
    a.axhline(0, color="k", lw=0.6)
    a.set(xlabel="year", ylabel="GJ/m2", xticks=sorted(nb.year.unique()))
    a.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    return fig


def plot_tradeoff(s, est_lab, box_lab, figsize=(13, 4.3)):
    """Bias-variance plane per box: variance against |bias| for each estimator, pooled over years (large) and per
    year (small), on log axes with iso-MSE curves variance = MSE - bias^2."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(box_lab), figsize=figsize, sharex=True, sharey=True)
    lo, hi = 3e-3, 4.0
    bb = np.geomspace(lo, hi, 400)
    for a, b in zip(ax, box_lab):
        for M in (0.01, 0.04, 0.25, 1, 4):
            v = M - bb ** 2
            a.plot(bb[v > 0], v[v > 0], color="0.8", lw=0.7, zorder=0)
            a.text(lo * 1.15, M * 1.1, f"MSE {M:g}", fontsize=6, color="0.5")
        for c, lab in est_lab.items():
            x = s[(s.box == b) & (s.estimator == c)]
            yr, al = x[x.year != "all"], x[x.year == "all"]
            a.scatter(yr.bias.abs().clip(lower=lo), yr.sd ** 2, s=12, color=EST_COL[c], alpha=0.45)
            a.scatter(al.bias.abs().clip(lower=lo), al.sd ** 2, s=70, color=EST_COL[c], ec="k", lw=0.6, label=lab,
                      zorder=3)
        a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(2e-3, hi ** 2), title=box_lab[b], xlabel="|bias| (GJ/m2)")
    ax[0].set_ylabel("variance over deployments (GJ/m2)^2")
    ax[0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def plot_by_year(s, est_lab, box_lab, figsize=(13, 8)):
    """Bias (+/- 2 MC se), variance and MSE per year for each estimator; rows are statistics, columns boxes."""
    import matplotlib.pyplot as plt
    rows = (("bias", "bias (GJ/m2)"), ("var", "variance (GJ/m2)^2"), ("mse", "MSE (GJ/m2)^2"))
    fig, ax = plt.subplots(len(rows), len(box_lab), figsize=figsize, sharex=True, sharey="row")
    for j, b in enumerate(box_lab):
        for c, lab in est_lab.items():
            x = s[(s.box == b) & (s.estimator == c) & (s.year != "all")].sort_values("year")
            yr = x.year.astype(int)
            ax[0, j].errorbar(yr, x.bias, 2 * x.bias_se, marker="o", ms=4, capsize=2, color=EST_COL[c], label=lab)
            ax[1, j].plot(yr, x.sd ** 2, "o-", ms=4, color=EST_COL[c])
            ax[2, j].plot(yr, x.rmse ** 2, "o-", ms=4, color=EST_COL[c])
        ax[0, j].axhline(0, color="k", lw=0.6)
        ax[0, j].set_title(box_lab[b])
        ax[-1, j].set(xlabel="year", xticks=sorted(yr))
    for a in ax[1:, 0]:
        a.set_ylim(bottom=0)
    for i, (_, lab) in enumerate(rows):
        ax[i, 0].set_ylabel(lab)
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_derived(dv, box_lab, figsize=(13, 4.3)):
    """Map and DR MSE (solid) and squared bias (dashed) for the multi-year targets."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(box_lab), figsize=figsize)
    for a, b in zip(ax, box_lab):
        x = dv[dv.box == b]
        i = np.arange(len(x))
        for c, lab in (("map", "map"), ("dr", "DR")):
            col = EST_COL["krige" if c == "map" else "dr"]
            a.plot(i, x[f"{c}_rmse"] ** 2, "o-", color=col, label=f"{lab} MSE")
            a.plot(i, x[f"{c}_bias"] ** 2, "s--", color=col, mfc="none", label=f"{lab} bias^2")
        a.set_xticks(i, x.target.str.replace(" mean", "\nmean"), rotation=45, ha="right", fontsize=8)
        a.set_title(box_lab[b])
    ax[0].set_ylabel("(GJ/m2)^2")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_bandwidth(bw, box_lab, prod=SIGMA, figsize=(13, 6.5)):
    """DR bias (+/- 2 MC se) and MSE against the kernel time sd, one line per space sd; star at production."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(box_lab), figsize=figsize, sharex=True, sharey="row")
    for j, b in enumerate(box_lab):
        x = bw[bw.box == b]
        for k, (sd, g) in enumerate(x.groupby("sigma_deg")):
            g = g.sort_values("sigma_days")
            lab = f"sigma_s = 1/{round(1 / sd)} deg"
            ax[0, j].errorbar(g.sigma_days, g.dr_bias, 2 * g.dr_bias_se, marker="o", capsize=2, color=f"C{k}", label=lab)
            ax[1, j].plot(g.sigma_days, g.dr_rmse ** 2, "o-", color=f"C{k}")
        p = x[np.isclose(x.sigma_deg, prod[0], atol=1e-3) & np.isclose(x.sigma_days, prod[1])]
        ax[0, j].plot(p.sigma_days, p.dr_bias, "*", color="k", ms=13, zorder=4, label="production")
        ax[1, j].plot(p.sigma_days, p.dr_rmse ** 2, "*", color="k", ms=13, zorder=4)
        ax[0, j].axhline(0, color="k", lw=0.6)
        ax[0, j].set_title(box_lab[b])
        ax[1, j].set(xlabel="kernel time sd (days)", xticks=(2, 4, 8))
    ax[0, 0].set_ylabel("DR bias (GJ/m2)")
    ax[1, 0].set_ylabel("DR MSE (GJ/m2)^2")
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("points", "nodes", "align", "rep", "estimate", "summary"))
    ap.add_argument("--rep", type=int)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunk", type=int, default=1)
    ap.add_argument("--maxrep", type=int)
    a = ap.parse_args()
    if a.what == "points":
        build_points(a.chunk, a.nchunk, a.maxrep)
    elif a.what == "nodes":
        build_node_q()
    elif a.what == "align":
        print(alignment().round(4).to_string())
    elif a.what == "rep":
        print(run_rep(a.rep).round(3).to_string())
    elif a.what == "estimate":
        print(estimate_rep(a.rep).round(3).to_string())
    else:
        pd.set_option("display.width", 200)
        print(summary().round(3).to_string())
        print(paired().round(3).to_string())
        print(mse_terms().round(4).to_string())
