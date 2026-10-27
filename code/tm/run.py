"""Transition-matrix design model: grid, movement matrix P_move, survival s(a), propagated q-hat.

    python code/tm/run.py [--deg 0.5 --step 10 --tol 3 --prior pooled --alpha 1 --radius 2 --survival km] [--qhat]
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as ocfg
from tm import grid as tg
from tm import movement as mv
from tm import occupancy as oc
from tm import survival as sv
from tm import transitions as tr

OUT = os.path.join(ocfg.REPO_ROOT, "data", "tm")
CUTOFF = "2026-06-30"   # argo_ohc.csv build date; later-dated profiles are bad dates
BATHY = os.path.join(OUT, "global_static_bathy.nc")   # GLORYS12 deptho, 1/12 deg


@dataclasses.dataclass(frozen=True)
class Config:
    deg: float = 0.5
    lat: tuple = (-90.0, 90.0)
    lon: tuple = (-180.0, 180.0)
    min_depth: float | None = 100.0
    support_radius: int | None = 2
    step_days: float = 10.0
    tol_days: float = 3.0
    dedupe_days: float = 2.0
    prior: str = "pooled"
    alpha: float = 1.0
    radius: int = 2
    empty: str = "absorb"
    survival: str = "km"
    censor_days: float = 540.0
    max_age_days: float = 5000.0
    cutoff: str = CUTOFF
    t0: str = "1998-01-01"
    rate: str = "empirical"
    profile_set: str = "dedup"
    out_lat: tuple = (14.0, 55.0)
    out_lon: tuple = (-72.0, -32.0)
    out_t: tuple = ("2009-10-01", "2020-01-01")
    band_days: tuple = (0.0, 30.0, 90.0, 365.0, 730.0, np.inf)

    @property
    def pairs_tag(self):
        return f"{self.step_days:g}d_tol{self.tol_days:g}_dd{self.dedupe_days:g}"

    @property
    def grid_tag(self):
        sup = "" if self.support_radius is None else f"_s{self.support_radius}"
        ext = "" if (self.lat, self.lon) == ((-90.0, 90.0), (-180.0, 180.0)) else \
            f"_{self.lat[0]:g}_{self.lat[1]:g}_{self.lon[0]:g}_{self.lon[1]:g}"
        return f"{self.deg:g}deg_z{self.min_depth or 0:g}{sup}{ext}"

    @property
    def move_tag(self):
        return f"{self.grid_tag}_{self.pairs_tag}_{self.prior}_a{self.alpha:g}_r{self.radius}_{self.empty}"

    @property
    def surv_tag(self):
        return f"{self.survival}_{self.step_days:g}d_c{self.censor_days:g}"

    @property
    def f_tag(self):
        bands = "-".join(f"{b:g}" for b in self.band_days)
        return f"{self.move_tag}_{self.survival}_c{self.censor_days:g}_age{self.max_age_days:g}_{self.t0}_b{bands}_out{self.out_t[0]}_{self.out_t[1]}"

    @property
    def q_tag(self):
        return f"{self.f_tag}_{self.rate}-{self.profile_set}"


def _path(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def load_profiles(c):
    return tr.profiles(dedupe_days=c.dedupe_days, cutoff=c.cutoff)


def load_grid(c, d=None):
    """Optionally restricted to cells within support_radius cells of any Argo profile."""
    if c.support_radius is None:
        return tg.Grid(c.deg, c.lat, c.lon, c.min_depth, BATHY)
    d = load_profiles(c) if d is None else d
    return tg.Grid(c.deg, c.lat, c.lon, c.min_depth, BATHY, (d["lat"].to_numpy(), d["lon"].to_numpy()),
                   c.support_radius)


def load_pairs(c, d=None, force=False):
    p = _path(f"pairs_{c.pairs_tag}.parquet")
    if force or not os.path.exists(p):
        tr.pairs(load_profiles(c) if d is None else d, c.step_days, c.tol_days).to_parquet(p)
    return pd.read_parquet(p)


def load_movement(c, g=None, force=False):
    p = _path(f"P_{c.move_tag}.npz")
    if force or not os.path.exists(p):
        g = g or load_grid(c)
        C, n_off = mv.counts(g, load_pairs(c))
        P, info = mv.movement_matrix(g, C, c.prior, c.alpha, c.empty,
                                     **({"radius": c.radius} if c.prior != "none" else {}))
        sp.save_npz(p, P)
        sp.save_npz(p.replace("P_", "C_"), C)
        print(f"{C.sum():.0f} transitions in {g.n} ocean states ({n_off} dropped off-grid/land), "
              f"{info['empty'].sum()} rows with no data or prior ({c.empty})")
    return sp.load_npz(p).tocsr(), sp.load_npz(p.replace("P_", "C_")).tocsr()


def load_survival(c, d=None, force=False):
    p = _path(f"surv_{c.surv_tag}.npz")
    if force or not os.path.exists(p):
        life = tr.lifetimes(load_profiles(c) if d is None else d, c.cutoff, c.censor_days)
        s = sv.step_survival(sv.fit(life, c.survival), c.step_days, c.max_age_days)
        np.savez(p, s=s, duration=life["duration"].to_numpy(), event=life["event"].to_numpy())
    z = np.load(p)
    return z["s"], pd.DataFrame({"duration": z["duration"], "event": z["event"]})


def profile_set(c, d):
    if c.profile_set == "dedup":
        return d
    from osim import intensity
    e = intensity.eligible(d.drop(columns="day"))
    return e.assign(day=tr._days(e["date"]))


def _grid_time(c, d=None):
    d = load_profiles(c) if d is None else d
    g = load_grid(c, d)
    t_edge = oc.steps(*tr._days(pd.Series(pd.to_datetime([c.t0, c.cutoff], utc=True))), c.step_days)
    return d, g, t_edge


def launch_years(c):
    return list(range(pd.Timestamp(c.t0).year, pd.Timestamp(c.cutoff).year + 1))


def cohort_path(c, year):
    os.makedirs(_path(f"floats_{c.f_tag}"), exist_ok=True)
    return _path(f"floats_{c.f_tag}/{year}.npz")


def run_cohort(c, year, d=None, g=None, t_edge=None, P=None, s=None):
    """Expected live floats from launches in calendar `year`: in the output crop by age band, and per (year, state)."""
    if t_edge is None:
        d, g, t_edge = _grid_time(c, d)
    P = load_movement(c, g)[0] if P is None else P
    s = load_survival(c, d)[0] if s is None else s
    src, _ = oc.source(g, tr.launches(d), d, t_edge)
    yk, years, _ = oc.step_years(t_edge)
    ka, kb = np.searchsorted(t_edge, tr._days(pd.Series(pd.to_datetime([f"{year}-01-01", f"{year + 1}-01-01"],
                                                                       utc=True))))
    ka, kb = max(ka, 0), min(kb, len(t_edge) - 1)
    out_k = np.searchsorted(t_edge, tr._days(pd.Series(pd.to_datetime(list(c.out_t), utc=True)))) + np.array([-1, 0])
    out_k = out_k.clip(0, len(t_edge) - 1)
    st = oc.crop_states(g, c.out_lat, c.out_lon)[0]
    band = oc.band_of_age(c.band_days, c.step_days, len(s))
    F_out, Fy = oc.propagate(P.T.tocsr().astype(np.float32), s, src, ka, kb, len(t_edge) - 1, out_k, st, yk,
                             band, len(c.band_days) - 1)
    np.savez(cohort_path(c, year), F_out=F_out, Fy=Fy, out_k=out_k)


def load_qhat(c, force=False, d=None):
    """Sum the launch-year cohorts (running any missing ones here), convert floats to profiles with the
    profile rate, and write q-hat over out_lat x out_lon x out_t plus per-(year, state) counts."""
    p = _path(f"qhat_{c.q_tag}.nc")
    if force or not os.path.exists(p):
        d, g, t_edge = _grid_time(c, d)
        missing = [y for y in launch_years(c) if not os.path.exists(cohort_path(c, y))]
        if missing:
            P, s = load_movement(c, g)[0], load_survival(c, d)[0]
            for y in missing:
                run_cohort(c, y, d, g, t_edge, P, s)
        parts = [np.load(cohort_path(c, y)) for y in launch_years(c)]
        F_out, Fy, out_k = sum(z["F_out"] for z in parts), sum(z["Fy"] for z in parts), parts[0]["out_k"]
        yk, years, y_edge = oc.step_years(t_edge)
        L = tr.launches(d)
        ps = profile_set(c, d)
        c_y = oc.profile_rate(d, L, y_edge, c.step_days, c.rate, ps)
        ds = oc.qhat_dataset(g, F_out * c_y[yk[out_k[0]:out_k[1]]][:, None, None], c.band_days,
                             t_edge[out_k[0]:out_k[1] + 1], c.step_days, c.out_lat, c.out_lon)
        si = g.index(ps["lat"], ps["lon"])
        yi = np.searchsorted(y_edge, ps["day"].to_numpy(), side="right") - 1
        ok = (si >= 0) & (yi >= 0) & (yi < len(years))
        Oy = np.zeros(Fy.shape)
        np.add.at(Oy, (yi[ok], si[ok]), 1.0)
        ds.attrs.update(year0=int(years[0]))
        ds.to_netcdf(p)
        np.savez(p.replace(".nc", "_counts.npz"), E=Fy * c_y[:, None], O=Oy, year0=int(years[0]), c_y=c_y)
    z = np.load(p.replace(".nc", "_counts.npz"))
    return xr.open_dataset(p), z["E"], z["O"], int(z["year0"])


def calibration(c, g, Ey, Oy, year0, years=range(2010, 2020)):
    """Expected vs realized profiles per year in each box (cells by centre) and globally."""
    iy, ix = g.yx(np.arange(g.n))
    la, lo = g.lat_edge[iy] + c.deg / 2, g.lon_edge[ix] + c.deg / 2
    rows = []
    for y in years:
        r = {"year": y, "global_E": Ey[y - year0].sum(), "global_O": Oy[y - year0].sum()}
        for b in ("boxA", "boxB"):
            (a0, a1), (o0, o1) = ocfg.BOXES[b]["lat_bounds"], ocfg.BOXES[b]["lon_bounds"]
            m = (la >= a0) & (la <= a1) & (lo >= o0) & (lo <= o1)
            r.update({f"{b}_E": Ey[y - year0, m].sum(), f"{b}_O": Oy[y - year0, m].sum()})
        rows.append(r)
    t = pd.DataFrame(rows).set_index("year")
    for k in ("global", "boxA", "boxB"):
        t[f"{k}_E/O"] = t[f"{k}_E"] / t[f"{k}_O"]
    return t


def summary(c, g, C, s, life, box=((22.0, 45.0), (-60.0, -40.0))):
    n_b = np.asarray(C.sum(axis=1)).ravel()
    iy, ix = g.yx(np.arange(g.n))
    la, lo = g.lat_edge[iy] + c.deg / 2, g.lon_edge[ix] + c.deg / 2
    inb = (la >= box[0][0]) & (la <= box[0][1]) & (lo >= box[1][0]) & (lo <= box[1][1])
    q = lambda v: np.percentile(v, [10, 50, 90]).round(1).tolist()
    S = np.cumprod(np.r_[1.0, s])
    print(f"departures per row: global p10/50/90 {q(n_b)}, zero {np.mean(n_b == 0):.1%}; "
          f"boxes A+B p10/50/90 {q(n_b[inb])}, zero {np.mean(n_b[inb] == 0):.1%}")
    print(f"lifetimes: {len(life)} floats, {life['event'].mean():.1%} observed deaths; "
          f"S(1y)={S[int(365 / c.step_days)]:.2f} S(3y)={S[int(1096 / c.step_days)]:.2f} "
          f"S(5y)={S[int(1826 / c.step_days)]:.2f}")


def build(c, force=False, qhat=False):
    d = load_profiles(c)
    g = load_grid(c, d)
    load_pairs(c, d, force)
    P, C = load_movement(c, g, force)
    s, life = load_survival(c, d, force)
    summary(c, g, C, s, life)
    if qhat:
        ds, Ey, Oy, y0 = load_qhat(c, force, d)
        print(calibration(c, g, Ey, Oy, y0).round(2).to_string())
    return g, P, C, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--prior", default="pooled", choices=list(mv.PRIORS))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--min-depth", type=float, default=100.0, help="state mask; <= 0 uses the land mask only")
    ap.add_argument("--support-radius", type=int, default=2, help="cells within r of a profile; < 0 = no support mask")
    ap.add_argument("--empty", default="absorb", choices=("absorb", "stay"))
    ap.add_argument("--survival", default="km", choices=("km", "weibull", "exponential"))
    ap.add_argument("--rate", default="empirical", choices=("empirical", "one"))
    ap.add_argument("--profile-set", default="dedup", choices=("dedup", "eligible"))
    ap.add_argument("--qhat", action="store_true", help="also propagate and write q-hat")
    ap.add_argument("--cohort", type=int, help="propagate only launches in this calendar year (array task)")
    ap.add_argument("--cohort-index", type=int, help="index into the launch years (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = Config(deg=a.deg, step_days=a.step, tol_days=a.tol, prior=a.prior, alpha=a.alpha,
                 radius=a.radius, min_depth=a.min_depth if a.min_depth > 0 else None, empty=a.empty,
                 support_radius=a.support_radius if a.support_radius >= 0 else None,
                 survival=a.survival, rate=a.rate, profile_set=a.profile_set)
    if a.cohort is not None or a.cohort_index is not None:
        y = a.cohort if a.cohort is not None else launch_years(cfg)[a.cohort_index]
        if a.force or not os.path.exists(cohort_path(cfg, y)):
            run_cohort(cfg, y)
        print(f"cohort {y} -> {cohort_path(cfg, y)}")
    else:
        build(cfg, a.force, a.qhat)
