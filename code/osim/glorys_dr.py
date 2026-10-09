"""Real-Argo-positions testbed on the full GLORYS OHC field: box-month truth, OHC at the profiles, and the
leave-one-float-out local least-squares mean field (Roemmich-Gilson style) ahead of kriging and DR.

    python code/osim/glorys_dr.py truth
    python code/osim/glorys_dr.py obs
    python code/osim/glorys_dr.py mean
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import clouds as cl
from osim import osim_config as cfg

BASE = os.path.join(cfg.OUT_DIR, "glorys_dr")
TAG = os.environ.get("GLORYS_DR_TAG", "")   # "_synth": the cloud-drawn positions check
OUT = BASE + TAG
LAYERS = os.path.join(cfg.GLORYS_DIR, cfg.LAYER_DIR, "ohc_layers_daily_{y}_{m:02d}.nc")
AREA = os.path.join(cfg.OUT_DIR, "cell_area_buffer.nc")
BOXES = {"boxA": ("boxA",), "boxB": ("boxB",), "boxAB": ("boxA", "boxB")}
YEARS = (2010, 2019)
EPOCH = pd.Timestamp("2015-01-01")
YEAR_D = 365.25

LAM_MEAN, N_MIN, N_HARM = 5.0, 300, 2   # window half-width (deg), minimum profiles (window widened by 25%)
C_DEG = 1.0
C_LAT = np.arange(cfg.BOXES["buffer"]["lat_bounds"][0] + C_DEG / 2, cfg.BOXES["buffer"]["lat_bounds"][1], C_DEG)
C_LON = np.arange(cfg.BOXES["buffer"]["lon_bounds"][0] + C_DEG / 2, cfg.BOXES["buffer"]["lon_bounds"][1], C_DEG)
N_T = 2 + 2 * N_HARM


def ohc_month(y, m):
    with xr.open_dataset(LAYERS.format(y=y, m=m)) as d:
        return sum(d[v] for v in d.data_vars).load()


def box_masks(lat, lon):
    """Cell-centre membership of each box on the native grid (box A stops at 33.99 N, see osim_config)."""
    out = {}
    for b, parts in BOXES.items():
        mk = np.zeros((len(lat), len(lon)), bool)
        for p in parts:
            (la0, la1), (lo0, lo1) = cfg.BOXES[p]["lat_bounds"], cfg.BOXES[p]["lon_bounds"]
            mk |= ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None, :]
        out[b] = mk
    return out


def build_truth(years=YEARS):
    """Daily area-weighted box means of 0-2000 m OHC (J/m2) on the deep-cell mask, and their monthly means."""
    area = xr.open_dataset(AREA)["cell_area"].values
    rows = []
    for y in range(years[0], years[1] + 1):
        for m in range(1, 13):
            z = ohc_month(y, m)
            mk = box_masks(z.latitude.values, z.longitude.values)
            v = z.values
            for b, k in mk.items():
                w = np.where(k, np.nan_to_num(area), 0.0)
                rows.append(pd.DataFrame(dict(box=b, time=z.time.values,
                                              ohc=np.nansum(v * w, axis=(1, 2)) / w.sum(), area_m2=w.sum())))
    d = pd.concat(rows, ignore_index=True)
    mo = d.assign(month=d.time.dt.to_period("M").dt.to_timestamp()).groupby(["box", "month"]).agg(
        ohc=("ohc", "mean"), n_days=("ohc", "size"), area_m2=("area_m2", "first")).reset_index()
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(os.path.join(BASE, "truth_daily.parquet"))
    mo.to_parquet(os.path.join(BASE, "truth_monthly.parquet"))
    return mo


def sample_ohc(f):
    """GLORYS OHC at the containing cell and day of each row (lat, lon, time) and its box; masked rows NaN."""
    f = f.assign(z=np.nan, box="")
    for (y, m), g in f.groupby([f.time.dt.year, f.time.dt.month]):
        z = ohc_month(y, m)
        la, lo = z.latitude.values, z.longitude.values
        i, j = cfg.containing_index(g.lat.values, la), cfg.containing_index(g.lon.values, lo)
        f.loc[g.index, "z"] = z.values[g.time.dt.day.values - 1, i, j]
        for b, mk in box_masks(la, lo).items():
            if b != "boxAB":
                f.loc[g.index[mk[i, j]], "box"] = b
    return f


def in_buffer(f):
    (la0, la1), (lo0, lo1) = cfg.BOXES["buffer"]["lat_bounds"], cfg.BOXES["buffer"]["lon_bounds"]
    return f.lat.between(la0, la1) & f.lon.between(lo0, lo1)


def build_obs():
    """Every de-duplicated fix in the buffer over YEARS with GLORYS OHC at its containing cell and day; prof_id
    matches clouds.PROFILES. Fixes on masked cells are dropped."""
    f = cl.load_fixes().reset_index(names="prof_id")
    f = sample_ohc(f[in_buffer(f) & f.time.dt.year.between(*YEARS)].copy()).dropna(subset=["z"])
    f["day"] = (f.time - EPOCH) / pd.Timedelta(days=1)
    os.makedirs(OUT, exist_ok=True)
    f.reset_index(drop=True).to_parquet(os.path.join(OUT, "obs.parquet"))
    return f


def build_synth(arm=0, seed=0):
    """Cloud-drawn positions check: each production profile moves to one random particle of its own `arm`
    cloud (that particle is then left out of Q-hat); OHC is resampled there at the real profile time. Other
    fixes keep their real positions."""
    import glob
    c = pd.concat(pd.read_parquet(f, filters=[("arm", "==", arm)]) for f in sorted(glob.glob(os.path.join(cl.OUT, "prod", "prod_*.parquet"))))
    c["pidx"] = c.groupby("prof_id").cumcount()
    rng = np.random.default_rng(seed)
    m = c.groupby("prof_id").size()
    pick = pd.DataFrame(dict(prof_id=m.index, pidx=rng.integers(0, m.values)))
    drawn = pick.merge(c, on=["prof_id", "pidx"]).dropna(subset=["lat"])
    real = pd.read_parquet(os.path.join(BASE, "obs.parquet"))
    fix = cl.load_fixes().reset_index(names="prof_id").set_index("prof_id")
    syn = drawn[["prof_id", "lat", "lon"]].join(fix[["float_id", "time"]], on="prof_id")
    syn = sample_ohc(syn[in_buffer(syn)].copy()).dropna(subset=["z"])
    syn["day"] = (syn.time - EPOCH) / pd.Timedelta(days=1)
    obs = pd.concat([real[~real.prof_id.isin(drawn.prof_id)], syn[real.columns]], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    obs.sort_values("prof_id").reset_index(drop=True).to_parquet(os.path.join(OUT, "obs.parquet"))
    drawn[["prof_id", "arm", "pidx"]].to_parquet(os.path.join(OUT, "drawn.parquet"))
    return obs, drawn


def tbasis(day):
    """Intercept, N_HARM annual harmonics and a linear trend in years since EPOCH: the part kept at a centre."""
    t = np.asarray(day, float)[..., None]
    w = 2 * np.pi * t / YEAR_D * np.arange(1, N_HARM + 1)
    return np.concatenate([np.ones_like(t), np.sin(w), np.cos(w), t / YEAR_D], -1)


def interp_matrix(lat, lon):
    """Bilinear interpolation from the centre grid (clamped at its edges) as a sparse (n, C) matrix."""
    fy = np.clip((np.asarray(lat) - C_LAT[0]) / C_DEG, 0, len(C_LAT) - 1)
    fx = np.clip((np.asarray(lon) - C_LON[0]) / C_DEG, 0, len(C_LON) - 1)
    i0, j0 = np.minimum(fy.astype(int), len(C_LAT) - 2), np.minimum(fx.astype(int), len(C_LON) - 2)
    wy, wx = fy - i0, fx - j0
    nl = len(C_LON)
    cols = np.stack([i0 * nl + j0, i0 * nl + j0 + 1, (i0 + 1) * nl + j0, (i0 + 1) * nl + j0 + 1], -1)
    vals = np.stack([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx], -1)
    return csr_matrix((vals.ravel(), (np.repeat(np.arange(len(cols)), 4), cols.ravel())),
                      shape=(len(cols), len(C_LAT) * len(C_LON)))


def fit_mean(obs):
    """Window least squares at every centre: spatial quadratic in the window plus tbasis; keeps the tbasis
    coefficients. Returns the full-data coefficients (C, N_T), the leave-one-float-out ones (F, C, N_T) and the
    float ids indexing F. Leaving float f out drops its Gram contribution (exact refit)."""
    lat, lon, z = obs.lat.values, obs.lon.values, obs.z.values
    T = tbasis(obs.day.values)
    fid, fl = pd.factorize(obs.float_id.values)
    nF, p = len(fl), 5 + N_T
    ridge = 1e-8 * np.eye(p)
    full = np.full((len(C_LAT) * len(C_LON), N_T), np.nan)
    lofo = np.full((nF,) + full.shape, np.nan, np.float32)
    for c, (cy, cx) in enumerate(np.stack(np.meshgrid(C_LAT, C_LON, indexing="ij"), -1).reshape(-1, 2)):
        dy, dx, lam = lat - cy, lon - cx, LAM_MEAN
        while (inw := (np.abs(dy) <= lam) & (np.abs(dx) <= lam)).sum() < N_MIN and lam < 4 * LAM_MEAN:
            lam *= 1.25
        i = np.flatnonzero(inw)
        if len(i) < N_MIN:
            continue
        u, v = dx[i] / lam, dy[i] / lam
        X = np.column_stack([u, v, u * v, u**2, v**2, T[i]])
        G, g = X.T @ X, X.T @ z[i]
        full[c] = np.linalg.solve(G + ridge * np.trace(G), g)[5:]
        fb = fid[i]
        o = np.argsort(fb, kind="stable")
        st = np.flatnonzero(np.r_[True, fb[o][1:] != fb[o][:-1]])
        u_f = fb[o][st]
        Gb = np.add.reduceat(X[o, :, None] * X[o, None, :], st)
        gb = np.add.reduceat(X[o] * z[i][o, None], st)
        lofo[:, c] = full[c]
        lofo[u_f, c] = np.linalg.solve(G - Gb + ridge * np.trace(G), (g - gb)[..., None])[..., 0][:, 5:]
    return full, lofo, fl


def mean_at(coef, lat, lon, day):
    """Fitted mean at points from centre coefficients (C, N_T)."""
    W = interp_matrix(lat, lon)
    return ((W @ np.nan_to_num(coef)) * tbasis(day)).sum(-1)


def build_mean():
    obs = pd.read_parquet(os.path.join(OUT, "obs.parquet"))
    full, lofo, fl = fit_mean(obs)
    np.savez_compressed(os.path.join(OUT, "mean_field.npz"), full=full, lofo=lofo, float_id=np.asarray(fl),
                        c_lat=C_LAT, c_lon=C_LON)
    mu = mean_at(full, obs.lat.values, obs.lon.values, obs.day.values)
    fid = pd.Index(fl).get_indexer(obs.float_id.values)
    W = interp_matrix(obs.lat.values, obs.lon.values)
    T = tbasis(obs.day.values)
    mu_lofo = np.array([(W[k] @ np.nan_to_num(lofo[fid[k]])) @ T[k] for k in range(len(obs))]).ravel()
    out = obs.assign(mu=mu, mu_lofo=mu_lofo)
    out.to_parquet(os.path.join(OUT, "obs_mean.parquet"))
    return out


def load_mean():
    d = np.load(os.path.join(OUT, "mean_field.npz"), allow_pickle=True)
    return d["full"], d["lofo"], d["float_id"]


# ----------------------------------------------------------------------------
# kriging of the anomaly: continuous-time simple kriging with the oracle Kuusela-Stein plug-ins,
# full data and leave one float out, on a 0.5 deg x 3-node-per-month map grid and at the in-box profiles
# ----------------------------------------------------------------------------

KTAG = os.environ.get("GLORYS_DR_KTAG", "")   # "_argomean": plug-ins refitted to our own anomaly
PARAMS = os.path.join(cfg.OUT_DIR, f"krige_plugin_{KTAG.lstrip('_') or 'oracle'}.csv")
LAMBDA_G, LAMBDA_T, JITTER = 10.0, 180.0, 1e-6   # as krige_estimate
NODE_DEG, T_NODES = 0.5, (1 / 6, 1 / 2, 5 / 6)   # map grid: 0.5 deg cells, 3 time nodes per month
LOFO_TOL = 1e6                                    # J/m2; smaller leave-one-out changes at a node are not stored


def map_nodes():
    """0.5 deg map cells of each box with their deep-cell area (m2) on the native mask, located at the area
    centroid of their deep native cells."""
    area = xr.open_dataset(AREA)["cell_area"]
    la, lo = area.latitude.values, area.longitude.values
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    out = []
    for b, mk in box_masks(la, lo).items():
        if b == "boxAB":
            continue
        k = mk & np.isfinite(area.values)
        w = area.values[k]
        d = pd.DataFrame(dict(i=np.floor(LA[k] / NODE_DEG), j=np.floor(LO[k] / NODE_DEG), area=w,
                              wlat=w * LA[k], wlon=w * LO[k])).groupby(["i", "j"]).sum()
        out.append(pd.DataFrame(dict(lat=d.wlat / d.area, lon=d.wlon / d.area, area=d.area, box=b)).reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


def param_index(params, lat, lon):
    """Nearest plug-in parameter cell of each point."""
    return np.argmin((params.lat.values[None] - np.asarray(lat)[:, None]) ** 2
                     + (params.lon.values[None] - np.asarray(lon)[:, None]) ** 2, axis=1)


def _corr(a, b, th):
    d2 = sum(((a[:, k, None] - b[None, :, k]) / th[k]) ** 2 for k in range(3))
    return np.exp(-np.sqrt(d2))


def _inverse(R):
    import scipy.linalg as sla
    for k in range(6):
        try:
            c = sla.cho_factor(R, lower=True)
            return sla.cho_solve(c, np.eye(len(R)))
        except sla.LinAlgError:
            R[np.diag_indices_from(R)] += 10.0 ** (-6 + k)
    raise sla.LinAlgError("kriging matrix not positive definite")


def mu_lofo_at(lofo, fidx, lat, lon, day):
    """Leave-one-float-out mean at points for each float index in fidx: (n, len(fidx))."""
    W, T = interp_matrix(lat, lon), tbasis(day)
    return np.stack([((W @ lofo[f].astype(float)) * T).sum(-1) for f in fidx], -1)


def krige_month(month, obs=None, params=None, nodes=None):
    """One target month: full-data and leave-one-float-out anomaly predictions at the map nodes (LOFO stored as
    sparse changes from the full map) and each in-box profile's prediction without its own float.

    Per plug-in cell: one inverse B of the conditioning correlation matrix; leaving float f out uses
    B_{-f,-f} - B_{-f,f} B_{ff}^{-1} B_{f,-f}, so alpha_{-f} = y - B_{.,f} B_{ff}^{-1} y_f with y = B r^{(-f)}
    (r^{(-f)} zeroed on f), exact up to rounding."""
    obs = pd.read_parquet(os.path.join(OUT, "obs_mean.parquet")) if obs is None else obs
    params = pd.read_csv(PARAMS) if params is None else params
    nodes = map_nodes() if nodes is None else nodes
    full, lofo, fl = load_mean()
    fpos = pd.Index(fl)
    m0 = pd.Timestamp(month)
    nd = m0.days_in_month
    d0 = (m0 - EPOCH) / pd.Timedelta(days=1)
    tn = d0 + nd * np.asarray(T_NODES)
    mid = d0 + nd / 2
    nodes = nodes.assign(cell=param_index(params, nodes.lat, nodes.lon))
    here = obs[(obs.time.dt.to_period("M") == m0.to_period("M")) & (obs.box != "")]
    here = here.assign(cell=param_index(params, here.lat, here.lon))
    xyz = obs[["lon", "lat", "day"]].to_numpy(float)
    node_rows, lofo_rows, prof_rows = [], [], []
    for c in np.union1d(nodes.cell.unique(), here.cell.unique()):
        p = params.iloc[c]
        th = np.array([p.th_lon, p.th_lat, p.th_t])
        sel = np.flatnonzero((np.abs(xyz[:, 0] - p.lon) <= LAMBDA_G) & (np.abs(xyz[:, 1] - p.lat) <= LAMBDA_G)
                             & (np.abs(xyz[:, 2] - mid) <= LAMBDA_T))
        X, ob = xyz[sel], obs.iloc[sel]
        R = _corr(X, X, th)
        R[np.diag_indices_from(R)] += max(float(p.nugget_ratio), JITTER)
        B = _inverse(R)
        a_full = B @ (ob.z.values - ob.mu.values)
        fids, fcol = np.unique(ob.float_id.values, return_inverse=True)
        fidx = fpos.get_indexer(fids)
        Rm = ob.z.values[:, None] - mu_lofo_at(lofo, fidx, ob.lat.values, ob.lon.values, ob.day.values)
        Rm[np.arange(len(ob)), fcol] = 0.0
        Y = B @ Rm
        A = Y.copy()
        for j in range(len(fids)):
            i = np.flatnonzero(fcol == j)
            A[:, j] -= B[:, i] @ np.linalg.solve(B[np.ix_(i, i)], Y[i, j])
            A[i, j] = 0.0
        nc = nodes[nodes.cell == c]
        if len(nc):
            P = np.column_stack([np.repeat(nc.lon.values, len(tn)), np.repeat(nc.lat.values, len(tn)),
                                 np.tile(tn, len(nc))])
            C = _corr(P, X, th)
            an_full, an_lofo = C @ a_full, C @ A
            nid = np.repeat(nc.index.values, len(tn))
            node_rows.append(pd.DataFrame(dict(node=nid, day=P[:, 2], anom=an_full,
                                               mu=mean_at(full, P[:, 1], P[:, 0], P[:, 2]))))
            dif = an_lofo - an_full[:, None]
            r, k = np.nonzero(np.abs(dif) > LOFO_TOL)
            lofo_rows.append(pd.DataFrame(dict(node=nid[r], day=P[r, 2], float_id=fids[k],
                                               d_anom=dif[r, k].astype(np.float32))))
        hc = here[here.cell == c]
        if len(hc):
            Ch = _corr(hc[["lon", "lat", "day"]].to_numpy(float), X, th)
            own = np.searchsorted(fids, hc.float_id.values)
            prof_rows.append(hc[["prof_id", "float_id", "box", "lat", "lon", "day", "z", "mu", "mu_lofo"]].assign(
                anom=Ch @ a_full, anom_lofo=(Ch * A[:, own].T).sum(1), n_cond=len(sel)))
    cat = lambda r: pd.concat(r, ignore_index=True) if r else pd.DataFrame()
    nodes_out = cat(node_rows).merge(nodes[["lat", "lon", "area", "box"]], left_on="node", right_index=True)
    return nodes_out.assign(month=m0), cat(lofo_rows).assign(month=m0), cat(prof_rows).assign(month=m0)


def anomaly_lattice(years=(2015, 2019), pad=55):
    """Our kriging anomaly, GLORYS OHC minus the full-data local mean field, on the 1 deg x 10 d lattice
    krige_local fits (GJ/m2; padded `pad` days before the first year, as krige_local.lattice)."""
    from osim import krige_local as kl
    t0 = pd.Timestamp(f"{years[0]}-01-01") - pd.Timedelta(days=pad)
    days = pd.date_range(t0, f"{years[1]}-12-31", freq=f"{kl.SAMPLE_DAYS}D")
    full = load_mean()[0]
    out = []
    for (y, m), dd in pd.Series(days, index=days).groupby([days.year, days.month]):
        z = ohc_month(y, m)
        k = int(round(kl.SAMPLE_DEG / float(np.median(np.diff(z.latitude.values)))))
        w = z.isel(time=dd.dt.day.values - 1, latitude=slice(None, None, k), longitude=slice(None, None, k))
        LA, LO = np.meshgrid(w.latitude.values, w.longitude.values, indexing="ij")
        for ti, t in enumerate(pd.DatetimeIndex(w.time.values)):
            day = (t - EPOCH) / pd.Timedelta(days=1)
            v = w.values[ti] - mean_at(full, LA.ravel(), LO.ravel(), np.full(LA.size, day)).reshape(LA.shape)
            out.append(pd.DataFrame(dict(lon=LO.ravel(), lat=LA.ravel(), day=(t - pd.Timestamp(kl.EPOCH)).days,
                                         year=t.year, yearday=t.dayofyear, z=v.ravel() * 1e-9)))
    d = pd.concat(out, ignore_index=True).dropna()
    d.to_csv(os.path.join(OUT, "krige_local_lattice_argomean.csv"), index=False)
    return d


def krige_path(month, what):
    return os.path.join(OUT, f"krige{KTAG}", f"{what}_{pd.Timestamp(month):%Y-%m}.parquet")


def run_krige(month):
    os.makedirs(os.path.join(OUT, f"krige{KTAG}"), exist_ok=True)
    for what, d in zip(("nodes", "lofo", "prof"), krige_month(month)):
        d.to_parquet(krige_path(month, what))


# ----------------------------------------------------------------------------
# DR: Q-hat from the production clouds, the q_f / Q mixture of leave-one-float-out maps, and the
# 1 / Q-weighted correction from each in-box profile's own-float-out residual
# ----------------------------------------------------------------------------

H_KDE, KAPPA_D, C_REL = 25.0, 10.0, 0.05   # spatial kernel sd (km), time-kernel width (d), coverage floor
R_EARTH_KM = 6371.0


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return R_EARTH_KM * np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def load_clouds(arm):
    """Production particles of one arm with their profile's day and float, minus any particle drawn as a
    synthetic position (build_synth)."""
    import glob
    p = pd.read_parquet(cl.PROFILES, columns=["prof_id", "float_id", "time"]).set_index("prof_id")
    c = pd.concat(pd.read_parquet(f, filters=[("arm", "==", arm)]) for f in sorted(glob.glob(os.path.join(cl.OUT, "prod", "prod_*.parquet"))))
    c["pidx"] = c.groupby("prof_id").cumcount()
    dp = os.path.join(OUT, "drawn.parquet")
    if os.path.exists(dp):
        dr = pd.read_parquet(dp)
        c = c.merge(dr[dr.arm == arm][["prof_id", "pidx"]], on=["prof_id", "pidx"], how="left", indicator=True)
        c = c[c._merge == "left_only"].drop(columns="_merge")
    c = c.dropna(subset=["lat"]).join(p, on="prof_id")
    c["day"] = (c.time - EPOCH) / pd.Timedelta(days=1)
    c["w"] = 1 / (c.groupby("prof_id").lat.transform("size") * KAPPA_D)
    return c.drop(columns="time").reset_index(drop=True)


def intensity(parts, lat, lon, day, by_float=False):
    """Q-hat (expected profiles per km2 per day) at points: each cloud a Gaussian KDE (sd H_KDE) with mass 1
    spread uniformly over +/- KAPPA_D / 2 days around its profile day. With by_float, the per-float parts
    q_f as a long table (point, float_id, q) whose sums over floats are Q."""
    from scipy.spatial import cKDTree
    lat, lon, day = (np.asarray(v, float) for v in (lat, lon, day))
    near = parts[(parts.day >= day.min() - KAPPA_D / 2) & (parts.day <= day.max() + KAPPA_D / 2)]
    D = cKDTree(_xyz(lat, lon)).sparse_distance_matrix(cKDTree(_xyz(near.lat.values, near.lon.values)),
                                                       4 * H_KDE, output_type="coo_matrix")
    i, j, d = D.row, D.col, D.data
    ok = np.abs(near.day.values[j] - day[i]) <= KAPPA_D / 2
    i, j, d = i[ok], j[ok], d[ok]
    q = near.w.values[j] * np.exp(-0.5 * (d / H_KDE) ** 2) / (2 * np.pi * H_KDE**2)
    tot = np.bincount(i, q, len(lat))
    if not by_float:
        return tot
    return tot, pd.DataFrame(dict(point=i, float_id=near.float_id.values[j], q=q)).groupby(
        ["point", "float_id"], as_index=False).q.sum()


def dr_month(month, parts, full=None, lofo=None, fl=None):
    """Box-month estimates for one arm: naive, mean only, kriging, and cross-fit DR with its map term,
    correction and covered fraction. Off the covered set C = {Q >= C_REL * mean Q} the full-data map is used."""
    if full is None:
        full, lofo, fl = load_mean()
    fpos = pd.Index(fl)
    m0 = pd.Timestamp(month)
    nd = pd.read_parquet(krige_path(month, "nodes"))
    lf = pd.read_parquet(krige_path(month, "lofo"))
    pf = pd.read_parquet(krige_path(month, "prof"))
    pf = pf[pf.prof_id.isin(parts.prof_id.unique())]
    Q, qf = intensity(parts, nd.lat, nd.lon, nd.day, by_float=True)
    nd = nd.reset_index(drop=True).assign(Q=Q)
    qf = qf.merge(nd[["node", "day", "lat", "lon"]], left_on="point", right_index=True)
    qf = qf.merge(lf[["node", "day", "float_id", "d_anom"]], on=["node", "day", "float_id"], how="left").fillna({"d_anom": 0.0})
    qf["mu_f"] = 0.0
    for f, g in qf.groupby("float_id"):
        coef = lofo[fpos.get_loc(f)].astype(float) if f in fpos else full   # no profile in the fit: own-out = full
        qf.loc[g.index, "mu_f"] = mean_at(coef, g.lat.values, g.lon.values, g.day.values)
    mix = (qf.q * (qf.mu_f + qf.d_anom)).groupby(qf.point).sum().reindex(nd.index, fill_value=0.0)
    pf = pf.assign(Q=intensity(parts, pf.lat, pf.lon, pf.day))
    rows = []
    for b, nb in nd.groupby("box"):
        thr = C_REL * np.average(nb.Q, weights=nb.area)
        cov = nb.Q >= thr
        krig = nb.mu + nb.anom
        dr_map = np.where(cov, nb.anom + mix[nb.index] / np.where(cov, nb.Q, 1.0), krig)
        area = nb.area.groupby(nb.node).first().sum() / 1e6
        pb = pf[pf.box == b]
        cp = pb.Q >= thr
        corr = ((pb.z - pb.mu_lofo - pb.anom_lofo)[cp] / pb.Q[cp]).sum() / (area * m0.days_in_month)
        avg = lambda v: np.average(v, weights=nb.area)
        w = 1 / pb.Q[cp]
        rows.append(dict(box=b, month=m0, area_km2=area, naive=pb.z.mean(), mean_only=avg(nb.mu), krige=avg(krig),
                         dr_map=avg(dr_map), dr_corr=corr, dr=avg(dr_map) + corr, cover=avg(cov),
                         n_prof=len(pb), n_cov=int(cp.sum()), ess=w.sum() ** 2 / (w**2).sum() if cp.any() else 0.0,
                         max_share=(w.max() / w.sum()) if cp.any() else np.nan))
    return pd.DataFrame(rows)


def run_dr(arms=None, months=None):
    """All box-months for each production arm, with box A u B as the area-weighted combination."""
    labels = pd.read_csv(os.path.join(cl.OUT, "prod_arms.csv")).groupby("arm").label.first()
    arms = labels.index if arms is None else arms
    months = pd.date_range("2015-01", "2019-12", freq="MS") if months is None else months
    full, lofo, fl = load_mean()
    out = []
    for a in arms:
        parts = load_clouds(a)
        out += [dr_month(m, parts, full, lofo, fl).assign(arm=a, label=labels[a]) for m in months]
    d = pd.concat(out, ignore_index=True)
    cols = ["naive", "mean_only", "krige", "dr_map", "dr_corr", "dr", "cover"]
    ab = d.groupby(["arm", "label", "month"]).apply(lambda g: pd.Series(
        {c: np.average(g[c], weights=g.area_km2) for c in cols} | dict(n_prof=g.n_prof.sum()))).reset_index()
    d = pd.concat([d, ab.assign(box="boxAB")], ignore_index=True)
    t = pd.read_parquet(os.path.join(BASE, "truth_monthly.parquet"))
    d = d.merge(t[["box", "month", "ohc"]], on=["box", "month"])
    d.to_parquet(os.path.join(OUT, f"dr_estimates{KTAG}.parquet"))
    return d


EST_LAB = dict(naive="sample mean", mean_only="mean field", krige="mean field + kriging", dr="mean field + DR kriging")
EST_COL = dict(naive="#2a78d6", mean_only="#eb6834", krige="#1baf7a", dr="#eda100")
BOX_LAB = dict(boxA="Box A", boxB="Box B", boxAB="Box A u B")
BOX_COL = dict(boxA="#2a78d6", boxB="#eb6834")
PERIODS = {"2010-2019": (2010, 2019), "2015-2019": (2015, 2019)}


def mean_field_box_months(years=YEARS):
    """Box-month means of the full-data mean field by the map-node quadrature (3 time nodes per month)."""
    nodes, full = map_nodes(), load_mean()[0]
    W = interp_matrix(nodes.lat.values, nodes.lon.values) @ np.nan_to_num(full)
    rows = []
    for m0 in pd.date_range(f"{years[0]}-01", f"{years[1]}-12", freq="MS"):
        tn = (m0 - EPOCH) / pd.Timedelta(days=1) + m0.days_in_month * np.asarray(T_NODES)
        mu = W @ tbasis(tn).mean(0)
        for b, g in nodes.groupby("box"):
            rows.append(dict(box=b, month=m0, mean_only=np.average(mu[g.index], weights=g.area)))
    return pd.DataFrame(rows)


def assemble_estimates(label="speed"):
    """Box-month truth with the sample mean and mean field over YEARS, and kriging and DR (2015-2019) from
    dr_estimates; box A u B is the area-weighted combination of the two boxes."""
    t = pd.read_parquet(os.path.join(BASE, "truth_monthly.parquet"))
    obs = pd.read_parquet(os.path.join(OUT, "obs.parquet"))
    nv = obs[obs.box != ""].assign(month=lambda d: d.time.dt.to_period("M").dt.to_timestamp()).groupby(
        ["box", "month"]).z.mean().rename("naive").reset_index()
    dr = pd.read_parquet(os.path.join(OUT, f"dr_estimates{KTAG}.parquet"))
    dr = dr[(dr.label == label) & (dr.box != "boxAB")][["box", "month", "krige", "dr"]]
    d = t[t.box != "boxAB"].merge(nv, on=["box", "month"], how="left").merge(
        mean_field_box_months(), on=["box", "month"]).merge(dr, on=["box", "month"], how="left")
    cols = ["ohc", *EST_LAB]
    ab = d.groupby("month").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.area_m2) for c in cols}))
    d = pd.concat([d, ab.reset_index().assign(box="boxAB")], ignore_index=True)[["box", "month", *cols]]
    d.to_parquet(os.path.join(OUT, f"estimates{KTAG}.parquet"))
    return d


def load_estimates():
    return pd.read_parquet(os.path.join(OUT, f"estimates{KTAG}.parquet"))


def _period_means(d):
    """Day-weighted means per box and year, and over each of PERIODS (NaN where any month is missing)."""
    d = d.assign(w=d.month.dt.days_in_month, year=d.month.dt.year)
    cols = ["ohc", *EST_LAB]
    wavg = lambda x: pd.Series({c: np.average(x[c], weights=x.w) for c in cols})
    rows = [d.assign(period=d.year.astype(str))] + [d[d.year.between(*y)].assign(period=k) for k, y in PERIODS.items()]
    return pd.concat(rows).groupby(["box", "period"], sort=False).apply(wavg)


def summary_table(box, d=None):
    """Truth and each estimator's bias (GJ/m2) and bias as % of truth, per year and period."""
    g = _period_means(load_estimates() if d is None else d).loc[box]
    out = pd.DataFrame({"truth (GJ/m2)": g.ohc * 1e-9})
    for c, lab in EST_LAB.items():
        out[f"{lab}: bias"] = (g[c] - g.ohc) * 1e-9
        out[f"{lab}: %"] = 100 * (g[c] - g.ohc) / g.ohc
    return out.rename_axis("year").reset_index()


def plot_bias(d=None, pct=True, figsize=(13, 8.5)):
    """Box-month bias (percent of truth or GJ/m2): one row per box, the sample mean in the left panel and
    the model-based estimators in the right, each panel on its own scale."""
    import matplotlib.pyplot as plt
    d = load_estimates() if d is None else d
    groups = (["naive"], [c for c in EST_LAB if c != "naive"])
    fig, ax = plt.subplots(len(BOX_LAB), 2, figsize=figsize, sharex=True)
    for row, b in zip(ax, BOX_LAB):
        x = d[d.box == b]
        for a, cols in zip(row, groups):
            a.axhline(0, color="#52514e", lw=0.8)
            for c in cols:
                v = (x[c] - x.ohc) / x.ohc * 100 if pct else (x[c] - x.ohc) * 1e-9
                a.plot(x.month, v, color=EST_COL[c], lw=1.4, label=EST_LAB[c])
            a.set_title(f"{BOX_LAB[b]}: {'sample mean' if cols == ['naive'] else 'model-based estimators'}",
                        loc="left", fontsize=10)
        row[0].set_ylabel("bias (% of truth)" if pct else "bias (GJ/m2)")
    ax[0, 1].legend(fontsize=8, frameon=False, loc="lower left")
    for a in ax[-1]:
        a.set_xlabel("month")
    fig.tight_layout()
    return fig


def plot_series(d=None, figsize=(11, 6.5)):
    """Monthly box-mean OHC (solid) and the estimated mean field (dashed) per box, and their difference."""
    import matplotlib.pyplot as plt
    d = load_estimates() if d is None else d
    fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for b, c in BOX_COL.items():
        x = d[d.box == b]
        a1.plot(x.month, x.ohc * 1e-9, color=c, lw=1.2, label=f"{BOX_LAB[b]} truth")
        a1.plot(x.month, x.mean_only * 1e-9, color=c, lw=1.6, ls="--", label=f"{BOX_LAB[b]} mean field")
        a2.plot(x.month, (x.ohc - x.mean_only) * 1e-9, color=c, lw=1.2, label=BOX_LAB[b])
    a1.set_ylabel("box-mean OHC 0-2000 m (GJ/m2)")
    a1.legend(fontsize=8, ncol=2, frameon=False)
    a2.axhline(0, color="#52514e", lw=0.8, ls=":")
    a2.set(ylabel="truth - mean field (GJ/m2)", xlabel="month")
    a2.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("truth", "obs", "synth", "mean", "lattice", "krige", "dr"))
    ap.add_argument("--month")
    a = ap.parse_args()
    if a.what == "truth":
        mo = build_truth()
        print(mo.assign(ohc=mo.ohc * 1e-9).groupby("box").ohc.describe().round(3))
    elif a.what == "obs":
        o = build_obs()
        print(len(o), o.float_id.nunique(), o.groupby([o.time.dt.year, "box"]).size().unstack().to_string())
    elif a.what == "synth":
        o, dr = build_synth()
        print(len(o), "obs;", len(dr), "drawn;", o[o.prof_id.isin(dr.prof_id)].groupby("box").size().to_dict())
    elif a.what == "lattice":
        d = anomaly_lattice()
        print(len(d), d.groupby("year").z.agg(["mean", "std"]).round(3).to_string())
    elif a.what == "krige":
        run_krige(a.month)
    elif a.what == "dr":
        d = run_dr()
        e = d.assign(**{c: (d[c] - d.ohc) * 1e-9 for c in ("naive", "mean_only", "krige", "dr")})
        print(e.groupby(["label", "box"])[["naive", "mean_only", "krige", "dr", "cover"]].mean().round(3).to_string())
    else:
        o = build_mean()
        r, rl = (o.z - o.mu) * 1e-9, (o.z - o.mu_lofo) * 1e-9
        print(f"residual sd (GJ/m2) full {r.std():.3f}, lofo {rl.std():.3f}")
        print(o.assign(r=r, rl=rl).groupby("box")[["r", "rl"]].agg(["mean", "std"]).round(3))
