"""q-hat diagnostics for the North Atlantic simulation: kernel bandwidth, source (profiles vs alive daily positions),
split-half noise, two-fold likelihood cross-validation, implied sample-mean bias, positivity.

    python code/osim/qhat_diag.py            # all YEARS, writes OUT/metrics.csv, maps_{year}.npz, points_{year}.parquet
    python code/osim/qhat_diag.py --wide     # sigma_s 2-5 deg, to OUT_wide
    python code/osim/qhat_diag.py crosshalf  # per-replicate HT/Hajek with the other half's q-hat, to crosshalf.csv

q-hat is the edge-corrected Gaussian kernel intensity on a lat x lon x day grid, per replicate, of the in-domain points:
q(x) = sum_cells K(x - c) N_c / sum_cells K(x - c) |D cap c|, D = domain x [T0, T1], computed by separable
filtering of block-summed counts. Daily positions of alive floats estimate 10 x the profile intensity (uniform cycle
phase), so their q-hat is divided by CYCLE.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import costlog                      # noqa: E402
from osim import natl_sim as ns               # noqa: E402

OUT = os.path.join(ns.OUT, "qhat_diag")
SIG_S = (1 / 12, 1 / 6, 1 / 3, 1 / 2, 1.0)    # deg
SIG_T = (2.0, 5.0, 10.0, 30.0)                # days
YEARS = (2016, 2020, 2024)
R, HALF, TRUNC = 200, 100, 4.0
MAP_SIG = ((1 / 6, 2.0), (1 / 3, 10.0), (1 / 2, 10.0), (1.0, 30.0))
PAD = int(TRUNC * max(SIG_T)) + 1


def grid():
    """Native grid cropped to the domain plus the widest kernel's reach: rows, cols, support area (km2), domain area (km2)."""
    la, lo, m, a = ns.domain()
    sup = ns.support()[2] / 1e6
    p = int(np.ceil(TRUNC * max(SIG_S) / np.median(np.diff(la))))
    i, j = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    rs = slice(max(i[0] - p, 0), min(i[-1] + p + 1, len(la)))
    cs = slice(max(j[0] - p, 0), min(j[-1] + p + 1, len(lo)))
    return la[rs], lo[cs], sup[rs, cs], np.where(m, a / 1e6, 0.0)[rs, cs]


def points(lat, lon, A, years):
    """All replicates' in-domain profile and alive-daily positions within YEARS (+/- PAD days): day (since T0), row,
    col, half (0 for reps < HALF); ev marks the evaluation profiles."""
    t0s = {y: (pd.Timestamp(f"{y}-01-01") - ns.T0).days for y in years}
    keep_t = lambda d: np.any([(d >= t0s[y] - PAD) & (d < t0s[y] + 366 + PAD) for y in years], axis=0)
    dlat, dlon = np.median(np.diff(lat)), np.median(np.diff(lon))
    P, D = [], []
    for r in range(R):
        s = pd.read_parquet(ns.seeds_path(r), columns=["float_id", "release"]).merge(ns.fleet(r)[["float_id", "life"]])
        end = s.release + pd.to_timedelta(np.minimum(s.life.to_numpy(), 4e4), "D")
        for src, f in (("p", ns.profile_path(r)), ("d", ns.track_path(r))):
            x = pd.read_parquet(f, columns=["float_id", "time", "lat", "lon"] + (["inside", "cycle"] if src == "p" else []))
            if src == "d":
                x = x.merge(s[["float_id"]].assign(rel=s.release.values, end=end.values))
                x = x[(x.time >= x.rel) & (x.time <= x.end)]
            d = ((x.time - ns.T0) / pd.Timedelta(days=1)).to_numpy()
            ii = np.floor((x.lat.to_numpy() - lat[0]) / dlat + 0.5).astype(np.int32)
            jj = np.floor((x.lon.to_numpy() - lon[0]) / dlon + 0.5).astype(np.int32)
            ok = keep_t(d) & (ii >= 0) & (ii < len(lat)) & (jj >= 0) & (jj < len(lon))
            ok[ok] = x.inside.to_numpy()[ok] if src == "p" else A[ii[ok], jj[ok]] > 0  # q is an intensity on D only
            df = pd.DataFrame(dict(day=d[ok].astype(np.float32), i=ii[ok], j=jj[ok], half=np.int8(r >= HALF),
                                   rep=np.int16(r)))
            if src == "p":
                df["ev"] = x.inside.to_numpy()[ok]
                df["float_id"] = x.float_id.to_numpy()[ok]
                df["cycle"] = x.cycle.to_numpy()[ok].astype(np.int16)
                P.append(df)
            else:
                D.append(df.drop(columns="rep"))
        print(f"rep {r}: {sum(map(len, P))} profiles, {sum(map(len, D))} daily", flush=True)
    return pd.concat(P, ignore_index=True), pd.concat(D, ignore_index=True)


def truth(year, lat, lon):
    """Area-weighted daily total OHC (GJ/m2 x km2) on the crop, domain cells only: (days, rows, cols) float32."""
    la, lo, _, _ = ns.domain()
    r0, c0 = int(np.argmin(np.abs(la - lat[0]))), int(np.argmin(np.abs(lo - lon[0])))
    A = grid()[3]
    out = []
    for mo in range(1, 13):
        d = ns.ohc_month(year, mo)
        z = sum(d[k].values for k in ns.LAYER_NAMES)[:, r0:r0 + len(lat), c0:c0 + len(lon)] * 1e-9
        out.append(np.where(A > 0, np.nan_to_num(z) * A, 0.0).astype(np.float32))
    return np.concatenate(out)


def block(x, f, axis):
    """Sum consecutive blocks of f along axis (zero-padded to a multiple of f)."""
    n = -(-x.shape[axis] // f) * f
    x = np.pad(x, [(0, n - x.shape[axis]) if k == axis else (0, 0) for k in range(x.ndim)])
    return x.reshape(*x.shape[:axis], n // f, f, *x.shape[axis + 1:]).sum(axis + 1)


def counts(day, i, j, t0, ft, fs, shape):
    """Count grid (coarse days x rows x cols) of points."""
    k = np.floor((day - t0) / ft).astype(np.int64)
    ok = (k >= 0) & (k < shape[0])
    flat = (k[ok] * shape[1] + i[ok] // fs) * shape[2] + j[ok] // fs
    u, c = np.unique(flat, return_counts=True)
    g = np.zeros(int(np.prod(shape)), np.float32)
    g[u] = c
    return g.reshape(shape)


def run(years=YEARS):
    os.makedirs(OUT, exist_ok=True)
    lat, lon, _, A = grid()
    dx = float(np.median(np.diff(lat)))
    with costlog.step(ns.COSTS, "qhat_diag_points", n=R):
        P, D = points(lat, lon, A, years)
    rows = []
    for y in years:
        t0 = (pd.Timestamp(f"{y}-01-01") - ns.T0).days
        ny = pd.Timestamp(f"{y}-12-31").dayofyear
        with costlog.step(ns.COSTS, "qhat_diag_truth", rep=y):
            Z = truth(y, lat, lon)
        ev = P[P.ev & (P.day >= t0) & (P.day < t0 + ny)]
        qbar = ev.groupby("rep").size().mean() / (A.sum() * ny)
        maps, keep = {}, {}
        for ss in SIG_S:
            for st in SIG_T:
                with costlog.step(ns.COSTS, "qhat_diag_sigma", rep=f"{y}_s{ss:.4g}_t{st:g}"):
                    fs, ft = max(1, int(ss / dx / 2)), max(1, int(st / 2))
                    pad = -(-int(TRUNC * st + 1) // ft) * ft
                    ta = t0 - pad
                    nt = -(-(ny + 2 * pad) // ft)
                    days = ta + np.arange(nt * ft)
                    tsup = block(((days >= 0) & (days <= (ns.T1 - ns.T0).days)).astype(np.float32), ft, 0)
                    ssup = block(block(A.astype(np.float32), fs, 0), fs, 1)
                    shape = (nt, *ssup.shape)
                    sg = (st / ft, ss / dx / fs, ss / dx / fs)
                    den = (ndimage.gaussian_filter1d(tsup, sg[0], truncate=TRUNC, mode="constant")[:, None, None]
                           * ndimage.gaussian_filter(ssup, sg[1:], truncate=TRUNC, mode="constant")[None])
                    zi = np.arange(ny) + pad
                    Zc = np.zeros((nt * ft, *Z.shape[1:]), np.float32)
                    Zc[zi] = Z
                    Zc = block(block(block(Zc, ft, 0), fs, 1), fs, 2)
                    Ac = np.zeros(nt * ft, np.float32)
                    Ac[zi] = 1
                    Ac = block(Ac, ft, 0)[:, None, None] * block(block(A.astype(np.float32), fs, 0), fs, 1)[None]
                    theta = Zc.sum() / Ac.sum()
                    ke = (np.floor((ev.day.to_numpy() - ta) / ft).astype(np.int64), ev.i.to_numpy() // fs,
                          ev.j.to_numpy() // fs)
                    res = {}
                    for src, X, scale in (("profiles", P, 1.0), ("daily", D, float(ns.CYCLE))):
                        q = []
                        for h in (0, 1):
                            g = X[X.half == h]
                            c = counts(g.day.to_numpy(), g.i.to_numpy(), g.j.to_numpy(), ta, ft, fs, shape)
                            ndimage.gaussian_filter(c, sg, truncate=TRUNC, mode="constant", output=c)
                            q.append(np.where(den > 0, c / np.where(den > 0, den, 1) / (HALF * scale), 0).astype(np.float32))
                            del c
                        qa, qb = (x[ke] for x in q)
                        qf = 0.5 * (q[0] + q[1])
                        own = np.where(ev.half.to_numpy() == 0, qb, qa)
                        qs = q[0] + q[1]
                        wg = np.where(qs > 0, qf * Ac, 0)
                        bias = float((qf * Zc).sum() / (qf * Ac).sum() - theta)
                        w = 1 / np.where(own > 0, own, np.nan)
                        ess = ev.assign(w=w).groupby("rep").w.apply(lambda v: v.sum() ** 2 / (len(v) * (v ** 2).sum()))
                        rel = np.where(own > 0, own, np.nan) / qbar
                        res[src] = (own, float((qf * Ac).sum()))
                        rows.append(dict(year=y, sig_deg=ss, sig_days=st, source=src, theta=theta, bias_naive=bias,
                                         rel_se=float(np.sqrt((wg * ((q[0] - q[1]) / np.where(qs > 0, qs, 1)) ** 2).sum() / wg.sum())),
                                         lcv=float(np.mean(np.log(np.maximum(own, 1e-3 * qbar)))),
                                         n_zero_heldout=int((own == 0).sum()), n_ev=len(ev), qbar=qbar,
                                         q_q01=np.nanquantile(rel, .01), q_q10=np.nanquantile(rel, .1),
                                         q_q50=np.nanquantile(rel, .5), q_q90=np.nanquantile(rel, .9),
                                         q_q99=np.nanquantile(rel, .99), ess_frac=float(ess.mean())))
                        if src == "daily" and (ss, st) in MAP_SIG:
                            maps[f"s{ss:.4g}_t{st:g}"] = (qf[pad // ft: (pad + ny) // ft].mean(0) / qbar).astype(np.float32)
                            maps[f"lat_s{ss:.4g}"] = lat[::fs] + (fs - 1) / 2 * dx
                            maps[f"lon_s{ss:.4g}"] = lon[::fs] + (fs - 1) / 2 * dx
                    rows[-1]["phase_ratio"] = rows[-2]["phase_ratio"] = res["daily"][1] / res["profiles"][1]
                    keep[f"s{ss:.4g}_t{st:g}_daily"] = res["daily"][0].astype(np.float32)
                    keep[f"s{ss:.4g}_t{st:g}_profiles"] = res["profiles"][0].astype(np.float32)
                    print(pd.DataFrame(rows[-2:]).round(4).to_string(index=False), flush=True)
                    pd.DataFrame(rows).to_csv(os.path.join(OUT, "metrics.csv"), index=False)
        np.savez_compressed(os.path.join(OUT, f"maps_{y}.npz"), **maps)
        ev[["rep", "half", "float_id", "cycle", "day", "i", "j"]].assign(**keep).to_parquet(os.path.join(OUT, f"points_{y}.parquet"))


def crosshalf(dirs=(OUT, OUT + "_wide")):
    """Per replicate, year, layer and (sigma, source): naive mean, HT sum(z/q) / (|D| ny) and Hajek
    sum(z/q) / sum(1/q), each profile weighted by the other half's q-hat (GJ/m2); profiles with q-hat = 0 are dropped
    and counted."""
    L = list(ns.LAYER_NAMES)
    AD = grid()[3].sum()
    th = ns.truth().assign(theta=lambda x: x.theta * 1e-9).set_index(["year", "layer"]).theta
    rows = []
    for y in YEARS:
        ny = pd.Timestamp(f"{y}-12-31").dayofyear
        z = pd.concat([pd.read_parquet(ns.profile_path(r), columns=["rep", "float_id", "cycle", "time", "inside"] + L)
                       .query("inside").loc[lambda x: x.time.dt.year == y].drop(columns=["inside", "time"])
                       for r in range(R)])
        z[L] = z[L] * 1e-9
        z["total"] = z[L].sum(axis=1)
        for d in [d for d in dirs if os.path.exists(os.path.join(d, f"points_{y}.parquet"))]:
            p = pd.read_parquet(os.path.join(d, f"points_{y}.parquet")).merge(z, on=["rep", "float_id", "cycle"])
            for c in [c for c in p.columns if c.startswith("s") and c.endswith(("_daily", "_profiles"))]:
                sig, src = c.rsplit("_", 1)
                ok = p[c] > 0
                w = 1 / p[c].where(ok)
                for layer in [*L, "total"]:
                    g = p.assign(wz=w * p[layer], w=w, ok=ok).groupby("rep")
                    e = g.agg(sz=("wz", "sum"), sw=("w", "sum"), naive=(layer, "mean"), n=("ok", "size"), n_ok=("ok", "sum"))
                    rows.append(e.assign(ht=e.sz / (AD * ny), hajek=e.sz / e.sw, year=y, layer=layer, sigma=sig,
                                         source=src, theta=th[(y, layer)]).drop(columns=["sz", "sw"]).reset_index())
        print(y, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "crosshalf.csv"), index=False)
    return out


if __name__ == "__main__":
    if "crosshalf" in sys.argv:
        with costlog.step(ns.COSTS, "qhat_crosshalf", n=R):
            crosshalf()
    elif "--wide" in sys.argv:
        SIG_S, SIG_T, MAP_SIG, OUT = (2.0, 3.0, 5.0), (10.0, 30.0), ((2.0, 30.0), (5.0, 30.0)), OUT + "_wide"
        run()
    elif "--test" in sys.argv:   # 4 replicates, 2 bandwidths, one year
        R, HALF, SIG_S, SIG_T, MAP_SIG, OUT = 4, 2, (1 / 6, 1.0), (2.0, 30.0), ((1.0, 30.0),), OUT + "_test"
        run((2020,))
    else:
        run()
