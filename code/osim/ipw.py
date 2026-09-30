"""Inverse-intensity and doubly robust estimators of a box-year mean, and the testbed.

For box-year domain T with area-time measure mu (OHC mask x days), target
Zbar = (1/mu(T)) int_T Z dmu, profiles x_i in T with values Z_i and weights
v_i = 1/q_hat(x_i) (q_hat per m2 per day, intensity.py):

    naive  mean_i Z_i
    HT     sum_i v_i Z_i / mu(T)
    Hajek  sum_i v_i Z_i / sum_i v_i
    KS     int m_hat dP = sum_j a_j Z_j            (krige_estimate aggregated weights)
    AIPW   (1/K) sum_k [ int m_hat_{-k} dP + sum_{i in k} v_i (Z_i - m_hat_{-k}(x_i)) / sum_{i in k} v_i ]

with folds k of whole floats (cross-fitting; DML1 fold average). The testbed puts the
real eligible Argo positions on the GLORYS residual field, where Zbar is known.

    python code/osim/ipw.py testbed [--box boxB --year 2015]             # real positions
    python code/osim/ipw.py testbed --positions sim --rep 3              # one sim replicate
    python code/osim/ipw.py gather
    python code/osim/ipw.py chain [--sigma 0.1667,4]                      # all replicates
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import scipy.linalg as sla
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import argo_meta as am
from osim import continuous_sim as cs
from osim import intensity as qi
from osim import krige_estimate as ke
from osim import osim_config as cfg

FIELD = os.path.join(cfg.OUT_DIR, "residual_field_buffer.nc")
AREA = os.path.join(cfg.OUT_DIR, "cell_area_buffer.nc")
PARAMS = os.path.join(cfg.OUT_DIR, "krige_plugin_oracle.csv")
TESTBED_DIR = os.path.join(cfg.OUT_DIR, "ipw_testbed")
N_FOLDS = 5
TIME_BLOCK = 32   # matches the residual field's time chunking


def _blocks(t_index, nt, size=TIME_BLOCK):
    for b0 in range(0, nt, size):
        yield b0, min(b0 + size, nt), np.flatnonzero((t_index >= b0) & (t_index < b0 + size))


def sample_field(z, lat, lon, time):
    """Z at the containing native cell and day; NaN off the field's mask or record."""
    zt = z["time"].values.astype("datetime64[D]")
    it = np.searchsorted(zt, np.asarray(time, dtype="datetime64[D]"), side="right") - 1
    ila = cfg.containing_index(np.asarray(lat), z["latitude"].values)
    ilo = cfg.containing_index(np.asarray(lon), z["longitude"].values)
    ok = (it >= 0) & (it < len(zt)) & (ila >= 0) & (ilo >= 0)
    out = np.full(len(it), np.nan)
    it_ok = np.where(ok, it, -1)
    for b0, b1, sel in _blocks(it_ok, len(zt)):
        if sel.size:
            blk = z.isel(time=slice(b0, b1)).values
            out[sel] = blk[it[sel] - b0, ila[sel], ilo[sel]]
    return out


def native_box(lat, lon, z, boxes=("boxA", "boxB")):
    """Box whose closed rectangle holds the containing native cell's centre (the estimand's
    A_b), else None."""
    la_c, lo_c = z["latitude"].values, z["longitude"].values
    i, j = cfg.containing_index(np.asarray(lat), la_c), cfg.containing_index(np.asarray(lon), lo_c)
    ok = (i >= 0) & (j >= 0)
    cla, clo = np.where(ok, la_c[i], np.nan), np.where(ok, lo_c[j], np.nan)
    out = np.full(len(i), None, object)
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        out[(cla >= la0) & (cla <= la1) & (clo >= lo0) & (clo <= lo1)] = b
    return out


def box_truth(z, area, box, year):
    """Zbar and mu(T) (m2 day) for a box-year on the OHC mask."""
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    ii = (z["latitude"].values >= la0) & (z["latitude"].values <= la1)
    jj = (z["longitude"].values >= lo0) & (z["longitude"].values <= lo1)
    w = np.nan_to_num(area.values[np.ix_(ii, jj)])
    k = np.flatnonzero(pd.DatetimeIndex(z["time"].values).year == year)
    s = 0.0
    for b0 in range(k[0], k[-1] + 1, TIME_BLOCK):
        b1 = min(b0 + TIME_BLOCK, k[-1] + 1)
        blk = z.isel(time=slice(b0, b1), latitude=np.flatnonzero(ii), longitude=np.flatnonzero(jj)).values
        s += float(np.nansum(blk * w))
    return s / (w.sum() * len(k)), float(w.sum() * len(k))


def predict_points(cond_xyz, cond_z, pts_xyz, pts_month, params, jitter=ke.JITTER):
    """Simple-kriging predictions at points, blocked like ke.aggregated_weights: points
    sharing a parameter cell and calendar month share one conditioning set (the
    profiles within LAMBDA_G / LAMBDA_T of the cell centre and mid-month)."""
    pred = np.full(len(pts_xyz), np.nan)
    clat = np.floor((pts_xyz[:, 1] - 22) / ke.PARAM_DEG) * ke.PARAM_DEG + 22 + 1
    clon = np.floor((pts_xyz[:, 0] + 60) / ke.PARAM_DEG) * ke.PARAM_DEG - 60 + 1
    key = pd.DataFrame({"clat": clat, "clon": clon, "m": pts_month})
    for (la, lo, m), g in key.groupby(["clat", "clon", "m"], sort=False):
        idx = g.index.to_numpy()
        p = params.iloc[np.argmin((params.lat - la) ** 2 + (params.lon - lo) ** 2)]
        th = np.array([p.th_lon, p.th_lat, p.th_t])
        tau2 = max(float(p.nugget_ratio), jitter)
        day = (pd.Timestamp(m.start_time) + pd.Timedelta(days=14) - pd.Timestamp(ke.EPOCH)).days
        sel = np.flatnonzero((np.abs(cond_xyz[:, 0] - lo) <= ke.LAMBDA_G) &
                             (np.abs(cond_xyz[:, 1] - la) <= ke.LAMBDA_G) &
                             (np.abs(cond_xyz[:, 2] - day) <= ke.LAMBDA_T))
        if sel.size < 5:
            pred[idx] = 0.0
            continue
        X = cond_xyz[sel]
        R = ke._corr(X, X, th)
        R[np.diag_indices_from(R)] += tau2
        for k in range(6):
            try:
                u = sla.cho_solve(sla.cho_factor(R, lower=True), cond_z[sel])
                break
            except sla.LinAlgError:
                R[np.diag_indices_from(R)] += 10.0 ** (-6 + k)
        pred[idx] = ke._corr(pts_xyz[idx], X, th) @ u
    return pred


def real_profiles(t0="2014-07-01", t1="2019-12-31"):
    """Eligible real profiles over the kriging conditioning region and window."""
    e = qi.eligible(am.read_argo())
    (la0, la1), (lo0, lo1) = cfg.BOXES["buffer"]["lat_bounds"], cfg.BOXES["buffer"]["lon_bounds"]
    e = e[e["lat"].between(la0, la1) & e["lon"].between(lo0, lo1)
          & (e["date"] >= pd.Timestamp(t0, tz="UTC")) & (e["date"] <= pd.Timestamp(t1, tz="UTC"))]
    e = e.assign(time=e["date"].dt.tz_localize(None))
    e["day"] = (e["time"].values - ke.EPOCH) / np.timedelta64(1, "D")
    return e.reset_index(drop=True)


def folds(float_ids, k=N_FOLDS, seed=0):
    u = np.unique(float_ids)
    f = np.random.default_rng(seed).permutation(len(u)) % k
    return pd.Series(f, index=u).reindex(float_ids).to_numpy()


def estimate(prof, box, year, zbar, mu_T, qds, params, area, kinds=("rho_age", "nominal")):
    """All estimators for one box-year. `prof` carries lon, lat, day, time, z, fold."""
    inb = ((prof["box"] == box) & (prof["time"].dt.year == year)).to_numpy()
    I = prof[inb]
    xyz = prof[["lon", "lat", "day"]].to_numpy(float)
    z = prof["z"].to_numpy()
    grid = ke.prediction_grid(box, year, area)
    a, _, _ = ke.aggregated_weights(xyz, grid, params)
    out = {"box": box, "year": year, "zbar": zbar, "n": int(inb.sum()),
           "n_floats": int(I["float_id"].nunique()), "naive": float(I["z"].mean()),
           "ks": float(a @ z), "ks_mass": float(a.sum())}
    for kind in kinds:
        v = 1.0 / qi.at(qds, I["lat"].to_numpy(), I["lon"].to_numpy(), I["time"].to_numpy(), kind)
        ok = np.isfinite(v)
        out[f"ht_{kind}"] = float(np.sum(v[ok] * I["z"].to_numpy()[ok]) / mu_T)
        out[f"hajek_{kind}"] = float(np.sum(v[ok] * I["z"].to_numpy()[ok]) / v[ok].sum())
        out[f"ess_{kind}"] = float(v[ok].sum() ** 2 / (v[ok] ** 2).sum() / ok.sum())
        out[f"n_noq_{kind}"] = int((~ok).sum())

    v = 1.0 / qi.at(qds, I["lat"].to_numpy(), I["lon"].to_numpy(), I["time"].to_numpy(), kinds[0])
    Im = pd.PeriodIndex(I["time"], freq="M")
    fold_est, fold_int, fold_cor, fold_n = [], [], [], []
    for k in range(N_FOLDS):
        cond = (prof["fold"] != k).to_numpy()
        ak, _, _ = ke.aggregated_weights(xyz[cond], grid, params)
        held = (I["fold"] == k).to_numpy() & np.isfinite(v)
        if not held.any():
            continue
        m_held = predict_points(xyz[cond], z[cond], I[["lon", "lat", "day"]].to_numpy(float)[held],
                                Im[held], params)
        e = I["z"].to_numpy()[held] - m_held
        fold_int.append(float(ak @ z[cond]))
        fold_cor.append(float(np.sum(v[held] * e) / v[held].sum()))
        fold_n.append(int(held.sum()))
        fold_est.append(fold_int[-1] + fold_cor[-1])
    out["aipw"] = float(np.mean(fold_est))
    out["aipw_fold_sd"] = float(np.std(fold_est, ddof=1))
    out["fold_int"] = ";".join(f"{x:.4g}" for x in fold_int)
    out["fold_cor"] = ";".join(f"{x:.4g}" for x in fold_cor)
    out["fold_n"] = ";".join(map(str, fold_n))
    return out


def sim_profiles(rep, tag="_argo_resample", t0="2014-07-01", t1="2019-12-31"):
    """One replicate's simulated nominal profiles (deployed floats) over the conditioning
    region and window, in the same layout as real_profiles."""
    p = pd.read_parquet(cs.profile_path(tag, chunk=rep), columns=["float_id", "time", "lat", "lon", "source"])
    (la0, la1), (lo0, lo1) = cfg.BOXES["buffer"]["lat_bounds"], cfg.BOXES["buffer"]["lon_bounds"]
    p = p[(p["source"] == "deploy") & p["lat"].between(la0, la1) & p["lon"].between(lo0, lo1)]
    p["time"] = pd.to_datetime(p["time"])
    p = p[(p["time"] >= pd.Timestamp(t0)) & (p["time"] <= pd.Timestamp(t1))]
    p["day"] = (p["time"].values - ke.EPOCH) / np.timedelta64(1, "D")
    return p.reset_index(drop=True)


def testbed(boxes=("boxA", "boxB"), years=range(2015, 2020), positions="real", rep=None,
            deg=2.0, months=2, kernel=None, fine=(0.125, 2.5)):
    """Real (or one simulated replicate's) positions on the GLORYS residual field. Real
    positions are weighted by eligible-profile intensities; simulated nominal profiles by
    the nominal intensity built without their own replicate."""
    os.makedirs(TESTBED_DIR, exist_ok=True)
    z = xr.open_dataset(FIELD)["z"]
    area = xr.open_dataset(AREA)["cell_area"]
    params = pd.read_csv(PARAMS)
    excl = () if positions == "real" else (rep,)
    if kernel is not None:
        qds = qi.build_kernel(*kernel, exclude=excl, deg=fine[0], days=fine[1])
        qlabel = f"_k{kernel[0]:g}_{kernel[1]:g}"
    elif positions == "real":
        qds, qlabel = xr.open_dataset(qi.qhat_path(deg=deg, months=months)), ""
    else:
        qds, qlabel = qi.build(deg=deg, months=months, exclude=excl), ""
    if positions == "real":
        prof, kinds, label = real_profiles(), ("rho_age", "nominal"), "real" + qlabel
    else:
        prof, kinds, label = sim_profiles(rep), ("nominal",), f"sim{qlabel}_r{rep}"
    prof["z"] = sample_field(z, prof["lat"].to_numpy(), prof["lon"].to_numpy(), prof["time"].to_numpy())
    prof = prof[np.isfinite(prof["z"])].reset_index(drop=True)
    prof["fold"] = folds(prof["float_id"].to_numpy())
    prof["box"] = native_box(prof["lat"].to_numpy(), prof["lon"].to_numpy(), z)
    rows = []
    for box in boxes:
        for year in years:
            zbar, mu_T = box_truth(z, area, box, year)
            row = {"positions": label, **estimate(prof, box, year, zbar, mu_T, qds, params, area, kinds)}
            pd.DataFrame([row]).to_csv(os.path.join(TESTBED_DIR, f"testbed_{label}_{box}_{year}.csv"), index=False)
            print(f"{label} {box} {year}: bias naive {(row['naive'] - zbar) * 1e-9:+.2f}, "
                  f"hajek {(row[f'hajek_{kinds[0]}'] - zbar) * 1e-9:+.2f}, "
                  f"aipw {(row['aipw'] - zbar) * 1e-9:+.2f} GJ/m2", flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


SIGMA = (1 / 6, 4.0)


def union_truth(z, area, years=cfg.ANALYSIS_YEARS):
    """(zbar, mu_T) per (box, year) for boxA, boxB and their disjoint union boxAB."""
    t = pd.DataFrame([(b, y, *box_truth(z, area, b, y)) for b in ("boxA", "boxB") for y in years],
                     columns=["box", "year", "truth", "mu_T"])
    u = (t.assign(s=t["truth"] * t["mu_T"]).groupby("year")[["s", "mu_T"]].sum()
         .assign(truth=lambda d: d["s"] / d["mu_T"], box="boxAB").reset_index())
    return pd.concat([t, u[t.columns]], ignore_index=True)


def chain(sigma=SIGMA, tag="_argo_resample", years=cfg.ANALYSIS_YEARS, reps=None, qds=None, kind="nominal"):
    """Full GLORYS chain: each replicate's in-box nominal profiles read Z and are weighted by
    1/q-hat (default: the native-grid kernel from all replicates, in-sample). Returns naive,
    HT and Hajek per (rep, box, year) in J/m2, boxes A, B and A u B."""
    z = xr.open_dataset(FIELD)["z"]
    area = xr.open_dataset(AREA)["cell_area"]
    qds = qi.load_kernel(*sigma, tag, deg="native") if qds is None else qds
    files = sorted(glob.glob(cs.profile_path(tag, chunk="*")))
    rep_of = {f: int(f.rsplit("_c", 1)[1].split(".")[0]) for f in files}
    files = [f for f in files if reps is None or rep_of[f] in set(reps)]
    P = []
    for f in files:
        p = pd.read_parquet(f, columns=["time", "lat", "lon", "source"])
        p = p[(p["source"] == "deploy") & pd.to_datetime(p["time"]).dt.year.isin(years)]
        p = p.assign(rep=rep_of[f], box=native_box(p["lat"].to_numpy(), p["lon"].to_numpy(), z))
        P.append(p[p["box"].notna()])
    P = pd.concat(P, ignore_index=True)
    P["time"] = pd.to_datetime(P["time"])
    P["year"] = P["time"].dt.year
    P["z"] = sample_field(z, P["lat"].to_numpy(), P["lon"].to_numpy(), P["time"].to_numpy())
    P = P[np.isfinite(P["z"])]
    v = 1.0 / qi.at(qds, P["lat"].to_numpy(), P["lon"].to_numpy(), P["time"].to_numpy(), kind)
    fin = np.isfinite(v)
    P = P.assign(v=np.where(fin, v, 0.0), miss=~fin)
    P = pd.concat([P, P.assign(box="boxAB")], ignore_index=True)
    P = P.assign(vz=P["v"] * P["z"], v2=P["v"] ** 2)
    e = P.groupby(["rep", "box", "year"]).agg(n=("z", "size"), naive=("z", "mean"), sv=("v", "sum"),
                                              svz=("vz", "sum"), sv2=("v2", "sum"), missing=("miss", "sum"))
    e = e.reset_index().merge(union_truth(z, area, years), on=["box", "year"])
    e["ht"] = e["svz"] / e["mu_T"]
    e["hajek"] = e["svz"] / e["sv"]
    e["ess"] = e["sv"] ** 2 / e["sv2"] / (e["n"] - e["missing"])
    return e.drop(columns=["sv", "svz", "sv2"]).assign(sigma_deg=sigma[0], sigma_days=sigma[1])


def chain_path(sigma=SIGMA):
    return os.path.join(TESTBED_DIR, f"chain_kernel_s{sigma[0]:.4g}_{sigma[1]:g}_reps.parquet")


def gather():
    t = pd.concat((pd.read_csv(f).assign(positions=os.path.basename(f).split("_box")[0][8:])
                   for f in sorted(glob.glob(os.path.join(TESTBED_DIR, "testbed_*_box*.csv")))),
                  ignore_index=True)
    t["setting"] = t["positions"].str.replace(r"_r\d+$", "", regex=True)
    est = [c for c in ("naive", "ks", "ht_rho_age", "hajek_rho_age", "ht_nominal", "hajek_nominal", "aipw")
           if c in t]
    bias = t[["setting", "positions", "box", "year"]].assign(**{c: (t[c] - t["zbar"]) * 1e-9 for c in est})
    t.to_csv(os.path.join(TESTBED_DIR, "testbed_summary.csv"), index=False)
    for s_, g in bias.groupby("setting"):
        print(f"\n[{s_}] bias (estimate - truth), GJ/m2, mean over {g['positions'].nunique()} position set(s)")
        print(g.groupby(["box", "year"])[est].mean().round(2).to_string())
        print("mean |bias|:", g[est].abs().mean().round(2).to_dict())
        if g["positions"].nunique() > 1:
            print("sd across replicates (avg over box-years):",
                  g.groupby(["box", "year"])[est].std().mean().round(2).to_dict())
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tb = sub.add_parser("testbed")
    tb.add_argument("--box", default=None, help="default both boxes")
    tb.add_argument("--year", type=int, default=None, help="default 2015-2019")
    tb.add_argument("--positions", choices=("real", "sim"), default="real")
    tb.add_argument("--rep", type=int, default=None, help="replicate for --positions sim")
    tb.add_argument("--kernel", default=None, help="sigma_deg,sigma_days for kernel q-hat")
    sub.add_parser("gather")
    ch = sub.add_parser("chain")
    ch.add_argument("--sigma", default=None, help="sigma_deg,sigma_days (default 1/6 deg, 4 d)")
    a = ap.parse_args()
    if a.cmd == "chain":
        s = tuple(float(x) for x in a.sigma.split(",")) if a.sigma else SIGMA
        chain(s).to_parquet(chain_path(s), index=False)
        print("->", chain_path(s))
    elif a.cmd == "testbed":
        testbed((a.box,) if a.box else ("boxA", "boxB"), (a.year,) if a.year else range(2015, 2020),
                a.positions, a.rep,
                kernel=tuple(float(x) for x in a.kernel.split(",")) if a.kernel else None)
    else:
        gather()
