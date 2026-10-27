"""Deployment-anchored age curve on real Argo: own-float-out kriging residuals by float age, unweighted and
reweighted to the deployment measure with the transition-matrix q-hat.

Target at every age cutoff a: the map error averaged over the deployment intensity d,
    T(a) = sum_{age_i < a} r_i w_i / sum w_i,   w_i = d(x_i) / q_{<a}(x_i),
with d the deployment law: q-hat of floats younger than 30 d averaged over time and Gaussian-smoothed in space
(realized launches are point sources, so the unsmoothed q-hat would put zero weight on almost every older profile).
The anchor is the first age band reweighted the same way; with A5 (exogenous deployments) it is unbiased.

    python code/tm/validate.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import argo_meta as am
from osim import intensity as qi
from tm import run
from tm import transitions as tr

DOMAINS = {"buffer": ("_real", run.ocfg.BOXES["buffer"]["lat_bounds"], run.ocfg.BOXES["buffer"]["lon_bounds"]),
           "natl": ("_real_natl", (0.0, 66.0), (-100.0, 0.0)),
           "natl_sla": ("_real_natl_sla", (0.0, 66.0), (-100.0, 0.0))}


def residuals(tag="_real"):
    """Own-float-out kriging residual r = z - mu_lofo - anom_lofo (GJ/m2) at every profile of the domain."""
    real = os.path.join(run.ocfg.OUT_DIR, f"glorys_dr{tag}")
    p = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(real, f"krige{tag}", "prof_*.parquet"))))
    t = pd.read_parquet(os.path.join(real, "obs.parquet"), columns=["prof_id", "time"])
    return p.merge(t, on="prof_id").assign(r=lambda d: (d.z - d.mu_lofo - d.anom_lofo) * 1e-9)


def attach(p, c, ds):
    """Float age at each profile and q-hat by age band at its cell and step."""
    L = tr.launches(run.load_profiles(c)).set_index("float_id")
    L = L.reindex(p.float_id)
    p = p.assign(age=(p.time - pd.to_datetime(L["date"].values)).dt.total_seconds() / 86400,
                 lat0=L["lat"].values, lon0=L["lon"].values)
    bands = [v for v in ds.data_vars if v.startswith("q_age")]
    q = np.column_stack([qi.at(ds, p.lat.values, p.lon.values, p.time.values, kind=v[2:]) for v in bands])
    dl = at_grid(ds, deploy_law(ds), p.lat.values, p.lon.values)
    return p.assign(dl=dl, **{f"q{j}": q[:, j] for j in range(len(bands))}), np.asarray(c.band_days)


def deploy_law(ds, sigma_deg=2.0, var="q_age0_30"):
    """Time-mean, NaN-aware Gaussian-smoothed young-float intensity on the q-hat grid."""
    from scipy.ndimage import gaussian_filter
    q = ds[var].mean("time").values
    m = np.isfinite(q)
    k = sigma_deg / float(np.diff(ds["lat_edge"].values[:2])[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(m, gaussian_filter(np.where(m, q, 0.0), k) / gaussian_filter(m.astype(float), k), np.nan)


def at_grid(ds, field, lat, lon):
    iy = np.searchsorted(ds["lat_edge"].values, lat, side="right") - 1
    ix = np.searchsorted(ds["lon_edge"].values, lon, side="right") - 1
    ok = (iy >= 0) & (iy < field.shape[0]) & (ix >= 0) & (ix < field.shape[1])
    out = np.full(len(lat), np.nan)
    out[ok] = field[iy[ok], ix[ok]]
    return out


def curve(p, edges):
    """Per cutoff band j (age < edges[j+1]): n, unweighted mean, reweighted (Hajek) mean; also per band alone."""
    nb = len(edges) - 1
    Q = np.cumsum(p[[f"q{j}" for j in range(nb)]].to_numpy(), axis=1)
    d = p["dl"].to_numpy()
    band = np.searchsorted(edges, p["age"].to_numpy(), side="right") - 1
    rows = []
    for j in range(nb):
        for kind, m, qj in (("cum", band <= j, Q[:, j]), ("band", band == j, p[f"q{j}"].to_numpy())):
            w = d[m] / qj[m]
            ok = np.isfinite(w) & (w > 0)
            r = p["r"].to_numpy()[m][ok]
            rows.append(dict(kind=kind, j=j, lo=edges[j if kind == "band" else 0], hi=edges[j + 1], n=int(m.sum()),
                             n_w=int(ok.sum()), unweighted=p["r"].to_numpy()[m].mean(),
                             weighted=np.sum(w[ok] * r) / np.sum(w[ok]), ess=w[ok].sum() ** 2 / (w[ok] ** 2).sum()))
    return pd.DataFrame(rows)


def strat_curve(p, edges, deg=4.0):
    """Post-stratified age curve: strata = deg x deg cells; each age band's stratum means (unweighted, and Hajek with
    1/q_band within the stratum) are combined with weights n0_s, the anchor-band profile count in the stratum,
    over strata where both bands have data. The anchor is then the young-profile mean on the same strata."""
    nb = len(edges) - 1
    band = np.searchsorted(edges, p["age"].to_numpy(), side="right") - 1
    s = stratum(p, deg)
    q = p[[f"q{j}" for j in range(nb)]].to_numpy()
    with np.errstate(divide="ignore"):
        w = 1.0 / q[np.arange(len(p)), band]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    d = pd.DataFrame({"s": s, "band": band, "r": p["r"].to_numpy(), "w": w, "wr": w * p["r"].to_numpy()})
    g = d.groupby(["band", "s"]).agg(n=("r", "size"), m=("r", "mean"), wr=("wr", "sum"), w=("w", "sum"))
    g["mw"] = (g.wr / g.w).where(g.w > 0)
    n0 = g.loc[0, "n"]
    rows = []
    for j in range(nb):
        gj = g.loc[j].join(n0.rename("n0"), how="inner")
        a = g.loc[0].loc[gj.index]
        wt = gj.n0 / gj.n0.sum()
        okq = gj.mw.notna() & a.mw.notna()
        wq = gj.n0[okq] / gj.n0[okq].sum()
        rows.append(dict(j=j, lo=edges[j], hi=edges[j + 1], strata=len(gj), n=int(gj.n.sum()),
                         anchor=(wt * a.m).sum(), unweighted=(wt * gj.m).sum(), strata_q=int(okq.sum()),
                         anchor_q=(wq * a.mw[okq]).sum(), qweighted=(wq * gj.mw[okq]).sum()))
    return pd.DataFrame(rows).assign(d_unw=lambda t: t.unweighted - t.anchor, d_q=lambda t: t.qweighted - t.anchor_q)


def strat_boot(p, edges, B=500, seed=0, **kw):
    rng = np.random.default_rng(seed)
    g = p.groupby("float_id").indices
    ids = np.array(list(g))
    r = pd.concat(strat_curve(p.iloc[np.concatenate([g[k] for k in rng.choice(ids, len(ids))])], edges, **kw)
                  for _ in range(B))
    return r.groupby("j")[["anchor", "unweighted", "d_unw", "d_q"]].std().add_prefix("se_")


def cell_units(p, deg=4.0, years=3, tau=30.0, old_min=None, y0=2004, unit="event", w=None):
    """Unit-by-cell-by-group mean residuals; cell = deg x deg box x `years`-year period, groups new (age < tau),
    local (older, launched in the same box) and advected (older, launched elsewhere). unit="event" pools floats
    launched in the same week and 2 deg box (co-deployed floats share residuals), else unit = float."""
    s = stratum(p, deg)
    old = p["age"].to_numpy() >= (tau if old_min is None else old_min)
    grp = np.where(p["age"].to_numpy() < tau, "new",
                   np.where(old & (stratum(p.assign(lat=p.lat0, lon=p.lon0), deg) == s), "local",
                            np.where(old, "advected", "gap")))
    ev = ((p["time"] - pd.to_timedelta(p["age"], "D")).to_numpy().astype("datetime64[W]").astype(np.int64) * 10**7
          + stratum(p.assign(lat=p.lat0, lon=p.lon0), 2.0))
    d = pd.DataFrame({"cell": s * 100 + (p["time"].dt.year.to_numpy() - y0) // years, "group": grp,
                      "unit": ev if unit == "event" else p["float_id"].to_numpy(), "r": p["r"].to_numpy(),
                      "w": 1.0 if w is None else np.asarray(w, float)})
    d = d[(d.group != "gap") & np.isfinite(d.w)].assign(wr=lambda x: x.w * x.r)
    g = d.groupby(["cell", "group", "unit"]).agg(wr=("wr", "sum"), w=("w", "sum"), size=("r", "size"))
    return g.assign(mean=g.wr / g.w)[["mean", "size", "w"]].reset_index()


def _contrast(u, lab, cell):
    """Per-cell difference of float-unit means (lab True minus False) with the cell's pooled unit variance."""
    k = pd.DataFrame({"cell": cell, "lab": lab, "m": u})
    g = k.groupby(["cell", "lab"]).m.agg(["mean", "size"]).unstack()
    v = k.groupby("cell").m.var()
    se = np.sqrt(v * (1 / g[("size", True)] + 1 / g[("size", False)]))
    return g[("mean", True)] - g[("mean", False)], se


def meta(d, se):
    """Inverse-variance pooled mean, Cochran Q and DerSimonian-Laird tau^2."""
    w = 1 / se ** 2
    m = np.sum(w * d) / np.sum(w)
    Q = np.sum(w * (d - m) ** 2)
    t2 = max(0.0, (Q - (len(d) - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    return dict(cells=len(d), fe=m, se_fe=np.sqrt(1 / np.sum(w)), Q=Q, df=len(d) - 1, tau2_sd=np.sqrt(t2),
                mean_d=np.mean(d))


def cell_test(u, a="advected", b="new", min_units=3, B=999, seed=0):
    """Cell contrasts a - b over cells holding both, with a within-cell label-permutation null for the pooled mean
    and for Q (float units exchangeable within a cell under no preferential sampling)."""
    u = u[u.group.isin([a, b])]
    n = u.groupby(["cell", "group"]).size().unstack(fill_value=0)
    keep = n.index[(n[a] > 0) & (n[b] > 0) & (n.sum(1) >= min_units)]
    u = u[u.cell.isin(keep)].reset_index(drop=True)
    lab = (u.group == a).to_numpy()
    d, se = _contrast(u["mean"].to_numpy(), lab, u.cell.to_numpy())
    ok = se > 0
    res = meta(d[ok], se[ok])
    rng = np.random.default_rng(seed)
    cell = u.cell.to_numpy()
    na = pd.Series(lab).groupby(cell).transform("sum").to_numpy()
    null = []
    for _ in range(B):
        rk = pd.Series(rng.random(len(u))).groupby(cell).rank(method="first").to_numpy()
        dp, sp = _contrast(u["mean"].to_numpy(), rk <= na, cell)
        null.append(meta(dp[ok], sp[ok]))
    null = pd.DataFrame(null)
    res.update(p_fe=np.mean(np.abs(null.fe) >= abs(res["fe"])), p_mean=np.mean(np.abs(null.mean_d) >= abs(res["mean_d"])),
               p_Q=np.mean(null.Q >= res["Q"]), Q_null=null.Q.median(), units_a=int(lab.sum()), units_b=int((~lab).sum()))
    return res, pd.DataFrame({"d": d, "se": se})


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def deployments(p, tau=30.0):
    """New-float units: floats launched in the same week and 2 deg box, profiles younger than tau; centre and
    float-weighted mean of every column in p."""
    n = p[p["age"] < tau]
    ev = (n["time"] - pd.to_timedelta(n["age"], "D")).dt.to_period("W").astype(str) + "|" + \
        pd.Series(stratum(n.assign(lat=n.lat0, lon=n.lon0), 2.0), index=n.index).astype(str)
    num = n.select_dtypes("number").columns.drop(["float_id"], errors="ignore")
    f = n.assign(ev=ev.values).groupby(["ev", "float_id"]).agg(**{c: (c, "mean") for c in num}, t=("time", "mean"))
    return f.groupby("ev").agg(**{c: (c, "mean") for c in f.columns}, n_float=("lat", "size"))


def nbhd(p, ev, R=200.0, T=90.0, tau=30.0, cols=("r",)):
    """Advected floats around each deployment: older profiles within R km and T days of the deployment centre from
    floats launched more than R km away; per-deployment means over floats (each float weighted once)."""
    o = p[p["age"] >= tau]
    o = o[np.isfinite(o[list(cols)]).all(1)]
    tree = cKDTree(_xyz(o.lat.values, o.lon.values))
    hits = tree.query_ball_point(_xyz(ev.lat.values, ev.lon.values), 2 * np.sin(R / 2 / 6371.0))
    e = np.repeat(np.arange(len(ev)), [len(h) for h in hits])
    k = np.concatenate([np.asarray(h, int) for h in hits])
    dt = np.abs((o.time.values[k] - ev.t.values[e]) / np.timedelta64(1, "D"))
    d0 = np.linalg.norm(_xyz(o.lat0.values[k], o.lon0.values[k]) - _xyz(ev.lat.values[e], ev.lon.values[e]), axis=1)
    m = (dt <= T) & (d0 > 2 * np.sin(R / 2 / 6371.0))
    h = pd.DataFrame({c: o[c].values[k[m]] for c in cols}).assign(e=e[m], float_id=o.float_id.values[k[m]])
    a = h.groupby(["e", "float_id"])[list(cols)].mean().groupby("e")
    a = a.mean().join(a.size().rename("n_adv")).join(a.var().add_suffix("_var"))
    return ev.reset_index().join(a.add_suffix("_adv").rename(columns={"n_adv_adv": "n_adv"}), how="inner")


def stratum(p, deg):
    return (np.floor(p["lat"] / deg).astype(int) * 1000 + np.floor(p["lon"] / deg).astype(int)).to_numpy()


def demean(Z, groups, w, tol=1e-10, it=500):
    """Weighted projection off several sets of fixed effects by alternating within-group demeaning."""
    Z = Z.copy()
    sw = [pd.Series(w).groupby(g).transform("sum").to_numpy()[:, None] for g in groups]
    for _ in range(it if len(groups) > 1 else 1):
        Z0 = Z.copy()
        for g, s in zip(groups, sw):
            Z -= pd.DataFrame(Z * w[:, None]).groupby(g).transform("sum").to_numpy() / s
        if np.abs(Z - Z0).max() < tol:
            break
    return Z


def trend(p, deg=4.0, w=None, year_fe=True, scale=30.0, absorb=(), cluster="float_id"):
    """Slope of r on log(1 + age/scale) with deg x deg stratum fixed effects, optional calendar-year effects and
    further absorbed fixed effects (columns of p), optional weights; CR1 standard errors clustered on `cluster`."""
    w = np.ones(len(p)) if w is None else np.asarray(w, float)
    groups = [stratum(p, deg)] + ([p["time"].dt.year.to_numpy()] if year_fe else []) + [p[a].to_numpy() for a in absorb]
    Z = demean(np.column_stack([p["r"].to_numpy(), np.log1p(p["age"].to_numpy() / scale)]), groups, w)
    y, x = Z[:, 0], Z[:, 1]
    A = 1.0 / np.sum(w * x * x)
    b = A * np.sum(w * x * y)
    u = pd.Series(x * w * (y - x * b)).groupby(p[cluster].to_numpy()).sum().to_numpy()
    G = len(u)
    se = np.sqrt(A * A * np.sum(u * u) * G / (G - 1))
    k = np.log1p(730 / scale)
    return dict(deg=deg, n=len(y), clusters=G, slope=b, se=se, t=b / se, d2y=b * k, se_d2y=se * k)


def bootstrap(p, edges, B=500, seed=0):
    """Float-cluster bootstrap SEs of the curve and of each point's difference from the anchor."""
    rng = np.random.default_rng(seed)
    g = p.groupby("float_id").indices
    ids = np.array(list(g))
    reps = []
    for b in range(B):
        idx = np.concatenate([g[k] for k in rng.choice(ids, len(ids))])
        cb = curve(p.iloc[idx], edges)
        a, aw = cb.loc[(cb.kind == "cum") & (cb.j == 0), ["unweighted", "weighted"]].iloc[0]
        reps.append(cb.assign(d_unw=cb.unweighted - a, d_w=cb.weighted - aw, b=b))
    r = pd.concat(reps)
    return r.groupby(["kind", "j"])[["unweighted", "weighted", "d_unw", "d_w"]].std().add_prefix("se_")


def main(c=run.Config(), years=(2010, 2019), interior=2.0, B=500, domain="buffer"):
    tag, (la0, la1), (lo0, lo1) = DOMAINS[domain]
    ds = run.load_qhat(c)[0]
    p, edges = attach(residuals(tag), c, ds)
    p = p[p.time.dt.year.between(*years) & p.lat.between(la0 + interior, la1 - interior)
          & p.lon.between(lo0 + interior, lo1 - interior) & (p.age >= 0)]
    cv = curve(p, edges)
    a, aw = cv.loc[(cv.kind == "cum") & (cv.j == 0), ["unweighted", "weighted"]].iloc[0]
    out = cv.assign(d_unw=cv.unweighted - a, d_w=cv.weighted - aw)
    out = out.join(bootstrap(p, edges, B), on=["kind", "j"]) if B else out
    return p, out


def trend_table(p, edges, degs=(2.0, 4.0, 8.0), trim=0.95):
    """Unweighted and 1/q_band weighted (clipped at the band's `trim` quantile, band mean 1) trends."""
    band = np.searchsorted(edges, p["age"].to_numpy(), side="right") - 1
    with np.errstate(divide="ignore"):
        w = 1.0 / p[[f"q{j}" for j in range(len(edges) - 1)]].to_numpy()[np.arange(len(p)), band]
    ok = np.isfinite(w) & (w > 0)
    w = pd.Series(np.where(ok, w, np.nan)).groupby(band)
    w = w.transform(lambda x: x.clip(upper=x.quantile(trim))).pipe(lambda x: x / x.groupby(band).transform("mean"))
    w = w.to_numpy()
    rows = [dict(weights=wn, year_fe=fe, **trend(p[m], deg, ww, fe))
            for deg in degs for fe in (False, True)
            for wn, m, ww in (("none", np.ones(len(p), bool), None), ("1/q", ok, w[ok]))]
    return pd.DataFrame(rows)


def float_meta(wmos, path=os.path.join(am.META_DIR, "float_table.csv")):
    """GDAC metadata table for the floats, downloading and caching any missing."""
    m = pd.read_csv(path, keep_default_na=False) if os.path.exists(path) else pd.DataFrame(columns=["float_id"])
    new = np.setdiff1d(wmos, m.float_id.astype(int))
    if len(new):
        am.download(new)
        m = pd.concat([m, am.float_table(new)], ignore_index=True)
        m.to_csv(path, index=False)
        m = pd.read_csv(path, keep_default_na=False)
    return m[m.float_id.isin(wmos)].set_index("float_id")


def add_meta(p):
    """Float metadata factors: platform family, pressure sensor maker, deployment cruise, project, launch year."""
    m = float_meta(p.float_id.unique())
    fam = {"APEX": "APEX", "S2A": "SOLO-II", "SOLO_W": "SOLO", "SOLO_D": "SOLO", "NOVA": "NOVA"}
    plat = m.platform_type.map(lambda v: fam.get(v, "NKE" if v.startswith(("ARVOR", "PROVOR")) else "other"))
    pres = m.pres_sensor.str.extract(r"^(DRUCK|KISTLER|SBE)", expand=False).fillna("other")
    m = m.assign(platform=plat, pressure=pres, cruise=(m.deployment_cruise_id.str.strip().replace("", np.nan).fillna(m.index.to_series().astype(str))
                                                     + "|" + m.deployment_platform))
    p = p.join(m[["platform", "pressure", "cruise", "project_name", "data_centre"]], on="float_id")
    return p.assign(vintage=(p.time - pd.to_timedelta(p.age, "D")).dt.year, month=p.time.dt.month)


def confounder_table(p, deg=4.0):
    specs = {"baseline": {}, "+platform": dict(absorb=["platform"]), "+pressure sensor": dict(absorb=["pressure"]),
             "+DAC": dict(absorb=["data_centre"]), "+project": dict(absorb=["project_name"]),
             "+launch year": dict(absorb=["vintage"]), "+launch year (no year)": dict(absorb=["vintage"], year_fe=False), "+month": dict(absorb=["month"]),
             "+all float factors": dict(absorb=["platform", "pressure", "project_name", "vintage", "month"]),
             "+cruise FE": dict(absorb=["cruise", "month"]), "cluster on cruise": dict(cluster="cruise"),
             "float FE (no year)": dict(absorb=["float_id"], year_fe=False),
             "float FE + month (no year)": dict(absorb=["float_id", "month"], year_fe=False)}
    rows = [dict(spec=k, **trend(p, deg, **kw)) for k, kw in specs.items()]
    for col in ("platform", "pressure"):
        for v, g in p.groupby(col):
            if g.float_id.nunique() >= 20:
                rows.append(dict(spec=f"{col}={v}", floats=g.float_id.nunique(), **trend(g, deg)))
    return pd.DataFrame(rows).set_index("spec")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="buffer", choices=list(DOMAINS))
    ap.add_argument("--years", type=int, nargs=2, default=(2010, 2019))
    a = ap.parse_args()
    p, out = main(B=0, domain=a.domain, years=tuple(a.years))
    edges = np.asarray(run.Config().band_days)
    print(f"{len(p)} profiles, {p.float_id.nunique()} floats")
    for deg in (2.0, 4.0, 8.0):
        sc = strat_curve(p, edges, deg).set_index("j").join(strat_boot(p, edges, 300, deg=deg))
        print(f"\n{deg:g} deg strata\n" + sc.round(3).to_string())
    print("\nlog-age trend, stratum FE, float-clustered SE\n" + trend_table(p, edges).round(3).to_string())
    print("\nconfounders (4 deg strata, year FE unless noted)\n" + confounder_table(add_meta(p)).round(3).to_string())
