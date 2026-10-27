"""Preferential-sampling detection and correction on real N Atlantic Argo (2004-2025); see
outlines/ps_detection_natl.md. Residuals r (baseline map, tag _real_natl) and r_sla (local-SLA mean field, tag
_real_natl_sla) must exist (code/osim/glorys_dr.py with GLORYS_DR_DOMAIN=natl).

    python code/tm/detect.py build      # profile table + SSH covariates + case-control samples (~10 min)
    python code/tm/detect.py report     # all tables (~10 min)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sat import ssh
from tm import ps_tests as ps
from tm import run
from tm import validate as v

OUT = os.path.join(run.ocfg.OUT_DIR, "..", "tm", "detect_natl")
SHIFTS = (0, -365, 365)
CLUSTERS = ("float_id", "b2x1", "b4x1", "b8x3")
REGS = ("Gulf Stream", "NAC", "Subtropical W", "Subpolar", "Subtropical E", "Tropical")


def _cc_path(sh):
    return os.path.join(OUT, f"cc_{sh}.npz")


def build():
    os.makedirs(OUT, exist_ok=True)
    p = v.main(B=0, domain="natl", years=(2004, 2025))[0].reset_index(drop=True)
    r2 = v.residuals(v.DOMAINS["natl_sla"][0])[["prof_id", "r"]].rename(columns={"r": "r_sla"})
    p = p.merge(r2, on="prof_id", how="left")
    a = ssh.at(p.lat.values, p.lon.values, p.time.values, var=("sla", "eke", "grad_sla"), half_deg=(0.5, 1.0, 2.0))
    p = p.assign(**{k: (x * 100 if k.startswith("sla") else x) for k, x in a.items()})
    p["beke"] = ps.box_eke(p)
    p["leke"] = np.log10(p.beke)
    p["region"] = ps.region(p.lat.to_numpy(), p.lon.to_numpy())
    yr = p.time.dt.year.to_numpy()
    for d, y in ((2, 1), (4, 1), (8, 3)):
        p[f"b{d}x{y}"] = v.stratum(p, float(d)) * 100 + (yr - 2004) // y
    for sh in SHIFTS:
        gc, gk = ps.sample(p, shift_days=sh)
        np.savez(_cc_path(sh), gc=gc.to_numpy(), gk=gk, cols=np.array(gc.columns))
    cc = load_cc()
    for c in ("sla_loc", "sla", "adt_rank"):
        a0, b1, b2 = (pref(cc[sh], c) for sh in SHIFTS)
        p[c + "_raw"], p[c + "_ex"] = a0, a0 - np.nanmean(np.column_stack([b1, b2]), 1)
    p.to_parquet(os.path.join(OUT, "profiles.parquet"))
    return p


def load():
    return pd.read_parquet(os.path.join(OUT, "profiles.parquet")), load_cc()


def load_cc():
    return {sh: dict(np.load(_cc_path(sh), allow_pickle=True)) for sh in SHIFTS}


def pref(c, var):
    """Case minus control mean of a covariate; adt_rank = share of the case's box (controls) with lower ADT."""
    cols = list(c["cols"])
    if var == "adt_rank":
        g, k = c["gc"][:, cols.index("adt")], c["gk"][:, :, cols.index("adt")]
        ok = np.isfinite(k)
        return np.where(ok.sum(1) >= 2, ((k < g[:, None]) & ok).sum(1) / np.maximum(ok.sum(1), 1), np.nan)
    return c["gc"][:, cols.index(var)] - np.nanmean(c["gk"][:, :, cols.index(var)], 1)


def cmean(q, c, w=None, cluster="b4x1"):
    """(Weighted) mean of q[c] with a CR SE clustered on `cluster`."""
    x = q[c].to_numpy(float)
    w = np.ones(len(q)) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w)
    m = np.sum(w[ok] * x[ok]) / w[ok].sum()
    u = pd.Series(w[ok] * (x[ok] - m)).groupby(q[cluster].to_numpy()[ok]).sum()
    return m, np.sqrt((u ** 2).sum()) / w[ok].sum()


def ols_slope(q, y, x, clusters=CLUSTERS):
    q = q[np.isfinite(q[y]) & np.isfinite(q[x])]
    X = np.column_stack([np.ones(len(q)), q[x] - q[x].mean()])
    A = np.linalg.inv(X.T @ X)
    b = A @ X.T @ q[y].to_numpy()
    e = q[y].to_numpy() - X @ b
    out = dict(n=len(q), slope=b[1])
    for c in clusters:
        S = pd.DataFrame(X * e[:, None]).groupby(q[c].to_numpy()).sum().to_numpy()
        out[f"z_{c}"] = b[1] / np.sqrt((A @ S.T @ S @ A)[1, 1] * len(S) / (len(S) - 1))
    return out


def anchor_curves(p):
    edges = np.asarray(run.Config().band_days)
    out = []
    for col in ("r", "r_sla"):
        q = p.dropna(subset=[col]).assign(r=lambda d: d[col])
        sc = v.strat_curve(q, edges, 4.0).set_index("j").join(v.strat_boot(q, edges, 200, deg=4.0))
        out.append(sc[["lo", "hi", "anchor", "se_anchor", "d_unw", "se_d_unw"]].assign(resid=col))
    return pd.concat(out)


def cell_table(p):
    rows = []
    for col in ("r", "r_sla"):
        q = p.dropna(subset=[col]).assign(r=lambda d: d[col])
        for deg in (2.0, 4.0, 8.0):
            u = v.cell_units(q, deg, 3, 30.0)
            for a in ("advected", "local"):
                r, _ = v.cell_test(u, a, "new", B=199)
                rows.append(dict(resid=col, deg=deg, contrast=f"{a}-new",
                                 **{k: r[k] for k in ("cells", "fe", "se_fe", "p_fe", "p_Q", "tau2_sd")}))
    return pd.DataFrame(rows)


def surface_tables(p):
    old = p.age >= 30
    yr = p.time.dt.year
    trend = pd.DataFrame([dict(y=y, sample=nm, **ols_slope(p[m], y, "leke"))
                          for y in ("sla_loc_ex", "sla_ex", "adt_rank_ex")
                          for nm, m in (("old", old), ("old 2004-14", old & (yr < 2015)),
                                        ("old 2015-25", old & (yr >= 2015)), ("new", ~old))])
    dec = p[old].assign(dec=pd.qcut(p.beke[old], 10, labels=False)).groupby("dec").apply(
        lambda q: pd.Series(dict(eke=q.beke.median(), **{f"{c}{s}": x for c in ("sla_loc_ex", "adt_rank_ex")
                                                         for s, x in zip(("", "_se"), cmean(q, c))})))
    reg = pd.DataFrame([dict(region=g, floats=nm, n=int(m.sum()),
                             **{f"{c}{s}": x for c in ("sla_loc_ex", "adt_rank_ex", "sla_ex")
                                for s, x in zip(("", "_se"), cmean(p[m], c, cluster="float_id"))})
                        for g in REGS for nm, m in (("old", old & (p.region == g)), ("new", ~old & (p.region == g)))
                        if m.sum() >= 300])
    return trend, dec, reg


def tilt_design(p, cc, sh):
    i = list(cc[sh]["cols"]).index("sla_loc")
    b = p.beke.to_numpy()
    return ps.design(cc[sh]["gc"][:, [i]], b, "step"), ps.design(cc[sh]["gk"][:, :, [i]], b, "step")


def fit_tilt(p, cc, m):
    """Step tilt (local SLA, and its interaction with box EKE >= 200), actual minus mean placebo (heuristic)."""
    fid = p.float_id.to_numpy()
    b = {}
    for sh in SHIFTS:
        Xc, Xk = tilt_design(p, cc, sh)
        b[sh] = ps.clogit(pd.DataFrame(Xc[m], columns=["s", "sx"]), Xk[m], fid[m], ["s", "sx"]).beta.to_numpy()
    return b[0] - (b[-365] + b[365]) / 2


def tilt_tables(p, cc):
    old = (p.age >= 30).to_numpy()
    hi = (p.beke >= 200).to_numpy()
    yr = p.time.dt.year.to_numpy()
    Xc, Xk = tilt_design(p, cc, 0)
    u = pd.Series(np.random.default_rng(1).random(p.float_id.nunique()), index=p.float_id.unique())
    h = u.reindex(p.float_id).to_numpy() < 0.5
    reg = p.region.to_numpy()
    folds = {"2004-14 -> 2015-25": (old & (yr < 2015), old & (yr >= 2015) & hi),
             "2015-25 -> 2004-14": (old & (yr >= 2015), old & (yr < 2015) & hi),
             **{f"leave out {g}": (old & (reg != g), old & (reg == g) & hi) for g in REGS[:3]},
             "floats A -> B": (old & h, old & ~h & hi), "floats B -> A": (old & ~h, old & h & hi),
             "in-sample": (old, old & hi)}
    rows = []
    for k, (tr, te) in folds.items():
        b = fit_tilt(p, cc, tr)
        w = ps.tilt_weights(b, Xc[te], Xk[te])
        q = p[te]
        row = dict(fold=k, beta_s=b[0], beta_sx=b[1], n_test=int(te.sum()))
        for nm, ww in (("unw", None), ("tilt", w)):
            row[f"pref_{nm}"], row[f"pref_{nm}_se"] = cmean(q, "sla_loc_ex", ww)
            for col in ("r", "r_sla"):
                row[f"{col}_{nm}"] = cmean(q, col, ww)[0]
        rows.append(row)
    xf = pd.DataFrame(rows).assign(shift_r=lambda d: d.r_tilt - d.r_unw, shift_r_sla=lambda d: d.r_sla_tilt - d.r_sla_unw)
    return xf


def regional_ohc(p):
    old = (p.age >= 30).to_numpy()
    dn = ps.density(p, 100, 15)
    rows = []
    for col in ("r", "r_sla"):
        q = p.assign(r=p[col])
        u = v.cell_units(q, 4.0, 3, 30.0)
        s = u.cell // 100
        a = np.round(s / 1000).astype(int)
        creg = ps.region(a * 4 + 2.0, (s - a * 1000) * 4 + 2.0)
        for g in REGS:
            r, _ = v.cell_test(u[creg == g], "advected", "new", B=199)
            m = (q.region == g).to_numpy()
            mk = ps.mark_test(q[m & old], np.log1p(dn[m & old]))
            rows.append(dict(resid=col, region=g, cells=r["cells"], adv_new=r["fe"], se=r["se_fe"], p=r["p_fe"],
                             old_mean=q.r[m & old].mean(), new_mean=q.r[m & ~old].mean(),
                             mark=mk["slope"], mark_se=mk["se"]))
    return pd.DataFrame(rows)


def report():
    pd.set_option("display.width", 250)
    p, cc = load()
    show = lambda t, d: print(f"\n== {t}\n{d.round(4).to_string()}", flush=True)
    show("residual sd", p[["r", "r_sla"]].std().to_frame("sd"))
    show("anchor and age bands (4 deg strata, float bootstrap)", anchor_curves(p))
    show("cell contrasts vs new floats (deg x 3 y cells, co-deployment units, permutation p)", cell_table(p))
    trend, dec, reg = surface_tables(p)
    show("surface preference (placebo-corrected) on log10 box EKE", trend)
    show("surface preference by box-EKE decile, older floats", dec)
    show("surface preference by region", reg)
    show("tilt cross-fitting (held-out high-EKE boxes)", tilt_tables(p, cc))
    show("regional OHC: advected-new, mean residuals, mark test", regional_ohc(p))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("build", "report"))
    a = ap.parse_args()
    build() if a.what == "build" else report()
