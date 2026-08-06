"""Lagrangian design statistics per velocity year, for testing D1b.

D1b asserts the cell-weight vectors are identically distributed across velocity
years, w^(l) =d= w^(l'). That is a statement about trajectories, so the Eulerian
diagnostics in velocity_diagnostics.py can only flag drivers -- this produces
the weight vectors themselves.

Deliberately a PROXY for the full park-and-profile OSSE: pure horizontal
advection at the park depth, no ascent/descent/surface drift and no mortality.
The nine-day park dominates displacement (osse.tex sec:simulation), so this
captures the occupancy structure D1b is about at a fraction of the cost. It is a
diagnostic, not the experiment.

Every year gets an identical deployment plan on identical days-of-year, so any
difference in the resulting weights is attributable to the velocity field.

Run from the repo root:
    python code/osse/design_invariance.py [--year 2015] [--reps 5] [--box boxB]
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg

VEL_DIR = os.path.join(cfg.GLORYS_DIR, "velocity_daily")
PARK_DEPTH = 1000.0
CYCLE_DAYS = 10          # Argo surfacing cadence: one profile per cycle
N_FLOATS = 50            # osse.tex: ~45-50 resident floats per box
# Seed beyond the box: floats advect in and out, and observed rms displacement
# is ~730 km (~6.6 deg) per year, so 4 deg out is genuinely reachable. It also
# widens the real-deployment pool from 104 to 153, which matters because
# replicate realisations are drawn from that finite pool.
SEED_MARGIN = 4.0
JITTER_KM = 5.0          # per-replicate seed perturbation


def load_deployment_pool(box=None, lat_bounds=(15, 50), lon_bounds=(-70, -30),
                         years=(2010, 2019), path=None):
    """Pooled empirical deployment distribution from the real Argo record.

    One row per float: its first profile's position and day-of-year, plus its
    observed lifespan. POOLED ACROSS YEARS on purpose -- drawing year-l designs
    from year-l's real deployments would break D1a/D1b, since deployment counts
    in this region swing by an order of magnitude between years for programmatic
    reasons. Pooling also protects D0: even if real positions or lifespans were
    influenced by the ocean state of their own year, severing the calendar year
    means the sampled plan cannot correlate with any particular Z_k.
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    df = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)

    g = df.sort_values("date").groupby("float_id")
    dep = g.first().reset_index()
    dep["lifespan_days"] = (g["date"].max() - g["date"].min()).dt.days.values
    dep["doy"] = dep["date"].dt.dayofyear - 1
    dep["year"] = dep["date"].dt.year

    if box is not None:
        la0, la1 = cfg.BOXES[box]["lat_bounds"]
        lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
        lat_bounds = (la0 - SEED_MARGIN, la1 + SEED_MARGIN)
        lon_bounds = (lo0 - SEED_MARGIN, lo1 + SEED_MARGIN)

    pool = dep[dep["lat"].between(*lat_bounds) & dep["lon"].between(*lon_bounds)
               & dep["year"].between(*years)]
    return pool[["lat", "lon", "doy", "lifespan_days"]].reset_index(drop=True)


def deployment_plan_argo(pool, n_floats=N_FLOATS, reps=5, year=0):
    """Independent draws from the pooled deployment distribution.

    One independent draw PER (l, d) pair, not per d: theory.tex Lemma 1 states
    omega_{l,d} iid ~ D, and the permutation test needs all MD designs
    exchangeable. Reusing a replicate's seeds across velocity years would pair
    the designs -- more powerful for isolating the velocity effect, but it makes
    designs sharing d dependent, so a global label permutation is no longer
    valid. The RNG is therefore keyed on (year, rep).

    Each draw is a without-replacement sample of n_floats deployments; position
    and lifespan travel together, preserving their observed joint structure.
    """
    lat, lon, rep, life = [], [], [], []
    for d in range(reps):
        r = np.random.default_rng([2000, int(year), d])
        idx = r.choice(len(pool), size=min(n_floats, len(pool)), replace=False)
        s = pool.iloc[idx]
        lat.append(s["lat"].to_numpy())
        lon.append(s["lon"].to_numpy())
        life.append(s["lifespan_days"].to_numpy())
        rep.append(np.full(len(s), d + 1))
    return (np.concatenate(lat), np.concatenate(lon),
            np.concatenate(rep), np.concatenate(life))


def deployment_plan(box, n_floats=N_FLOATS, reps=5, independent=True):
    """Replicate deployment realisations, identical across velocity years.

    ``independent=True`` draws each replicate's seeds afresh from the deployment
    distribution, matching theory.tex Lemma 1 (omega_{l,d} iid ~ D). This is
    what D1b is stated over, and what the permutation test relies on.

    ``independent=False`` reproduces osse.tex sec:simulation's ensemble recipe
    (perturb one plan by a few km). Those members are NOT iid draws -- they are
    near-duplicate seeds in the same flow -- so within-year dispersion collapses
    and any between-year test built on them is anticonservative. Kept only for
    comparison; do not use it to test D1b.

    The same seeds are reused across velocity years, so a difference in the
    resulting weights is attributable to the velocity field, not the plan.
    """
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]

    lat, lon, rep = [], [], []
    for d in range(reps):
        r = np.random.default_rng(1000 + d)
        if independent:
            la = r.uniform(la0 - SEED_MARGIN, la1 + SEED_MARGIN, n_floats)
            lo = r.uniform(lo0 - SEED_MARGIN, lo1 + SEED_MARGIN, n_floats)
        else:
            base = np.random.default_rng(0)
            la = base.uniform(la0 - SEED_MARGIN, la1 + SEED_MARGIN, n_floats)
            lo = base.uniform(lo0 - SEED_MARGIN, lo1 + SEED_MARGIN, n_floats)
            la = la + JITTER_KM / 111.0 * r.normal(size=n_floats)
            lo = lo + JITTER_KM / (111.0 * np.cos(np.deg2rad(la))) * r.normal(size=n_floats)
        lat.append(la)
        lon.append(lo)
        rep.append(np.full(n_floats, d + 1))
    return np.concatenate(lat), np.concatenate(lon), np.concatenate(rep)


def run_year(year, box, reps, out_zarr, runtime_days=360, independent=True, pool=None):
    """Advect the plan through one velocity year at the park depth.

    Floats leaving the velocity buffer are deleted rather than reflected: the
    box is not closed, and a float advected out of the domain has left the
    experiment. The per-year loss rate is itself a D1b diagnostic, so it is
    reported (frac_domain_exit) rather than silently absorbed.
    """
    from parcels import (AdvectionRK4, FieldSet, JITParticle, ParticleSet,
                         StatusCode, Variable)

    class ArgoFloat(JITParticle):
        age = Variable("age", dtype=np.float32, initial=0.0)
        lifespan = Variable("lifespan", dtype=np.float32, initial=0.0)

    def DeleteOutOfBounds(particle, fieldset, time):  # noqa: N802 (parcels kernel)
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    def Mortality(particle, fieldset, time):  # noqa: N802 (parcels kernel)
        particle.age += particle.dt
        if particle.lifespan > 0 and particle.age > particle.lifespan:
            particle.delete()

    files = sorted(glob.glob(os.path.join(VEL_DIR, f"velocity_daily_park_{year}_*.nc")))
    if len(files) != 12:
        raise FileNotFoundError(f"{year}: found {len(files)}/12 park velocity files")

    fieldset = FieldSet.from_netcdf(
        files,
        variables={"U": "uo", "V": "vo"},
        dimensions={"time": "time", "depth": "depth", "lat": "latitude", "lon": "longitude"},
        allow_time_extrapolation=False,
    )
    if pool is not None:
        lat, lon, rep, life = deployment_plan_argo(pool, reps=reps, year=year)
    else:
        lat, lon, rep = deployment_plan(box, reps=reps, independent=independent)
        life = np.zeros(len(lat))
    # All floats start on 1 Jan: the design must be a function of (omega, V_l)
    # alone, so a float cannot outlive its velocity year. Mortality still runs,
    # so floats whose observed lifespan is under a year retire mid-year. The
    # array is therefore at full strength from 1 Jan -- younger and more
    # clustered near deployment sites than the real standing stock.
    t0 = np.datetime64(f"{year}-01-01")
    pset = ParticleSet(fieldset=fieldset, pclass=ArgoFloat,
                       lon=lon, lat=lat, depth=np.full(len(lat), PARK_DEPTH),
                       time=np.full(len(lat), t0),
                       lifespan=life * 86400.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pset.execute([AdvectionRK4, DeleteOutOfBounds, Mortality],
                     runtime=np.timedelta64(runtime_days, "D"),
                     dt=np.timedelta64(1, "h"),
                     output_file=pset.ParticleFile(name=out_zarr,
                                                   outputdt=np.timedelta64(CYCLE_DAYS, "D")),
                     verbose_progress=False)
    return out_zarr, rep


def profiles_from_zarr(path, rep, year):
    """Trajectory output -> one profile position per float per cycle."""
    ds = xr.open_zarr(path)
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds["time"].values
    n_traj, n_obs = lat.shape
    doy = ((time - time[:, :1].astype("datetime64[Y]").astype("datetime64[ns]"))
           / np.timedelta64(1, "D")).astype("float")
    df = pd.DataFrame({
        "vel_year": year,
        "rep": np.repeat(rep[:n_traj], n_obs),
        "float_id": np.repeat(np.arange(n_traj), n_obs),
        "cycle": np.tile(np.arange(n_obs), n_traj),
        "lat": lat.ravel(),
        "lon": lon.ravel(),
        "doy": np.floor(doy.ravel()).astype("float"),
    })
    # Drop cycle 0: the deployment position, not a post-ascent surfacing.
    return df[df["cycle"] > 0].dropna(subset=["lat", "lon", "doy"])


def in_box(df, box):
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
    return df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)


def weights_and_summary(df, box, deg=cfg.COARSE_DEG, days=cfg.COARSE_DAYS):
    """Per-(year, rep) occupancy weights over box cells, plus scalar diagnostics."""
    df = df.copy()
    df["inside"] = in_box(df, box)
    df["cell_lat"] = np.floor(df["lat"] / deg) * deg + deg / 2
    df["cell_lon"] = np.floor(df["lon"] / deg) * deg + deg / 2
    df["doy_bin"] = np.minimum((df["doy"] // days).astype(int),
                               cfg.DAYS_PER_YEAR // days - 1)

    ins = df[df["inside"]]
    w = (ins.groupby(["vel_year", "rep", "cell_lat", "cell_lon", "doy_bin"])
            .size().rename("n").reset_index())
    w["w"] = w["n"] / w.groupby(["vel_year", "rep"])["n"].transform("sum")

    rows = []
    for (y, r), g in df.groupby(["vel_year", "rep"]):
        gi = g[g["inside"]]
        wr = w[(w["vel_year"] == y) & (w["rep"] == r)]["w"].to_numpy()
        last_in = gi.groupby("float_id")["cycle"].max()
        # Great-circle displacement from each float's first recorded position.
        gl = g.sort_values("cycle").groupby("float_id")
        d0 = gl.first()[["lat", "lon"]]
        dl = gl.last()[["lat", "lon"]]
        dy = (dl["lat"] - d0["lat"]) * 111.0
        dx = (dl["lon"] - d0["lon"]) * 111.0 * np.cos(np.deg2rad(d0["lat"]))
        final_cycle = g["cycle"].max()
        rows.append({
            "vel_year": y, "rep": r,
            "n_profiles_in_box": len(gi),
            "n_floats": g["float_id"].nunique(),
            # A float whose last record precedes the final cycle was deleted at
            # the domain edge.
            "frac_domain_exit": 1.0 - gl["cycle"].max().eq(final_cycle).mean(),
            "frac_inside": g["inside"].mean(),
            "residence_days": gi.groupby("float_id").size().mean() * CYCLE_DAYS,
            "retention_end": last_in.eq(g["cycle"].max()).mean() if len(last_in) else 0.0,
            "rms_disp_km": float(np.sqrt((dx**2 + dy**2).mean())),
            "n_cells_occupied": len(wr),
            "participation_ratio": float(1.0 / np.sum(wr**2)) if len(wr) else 0.0,
        })
    return w, pd.DataFrame(rows)


def main(years=None, reps=10, box="boxB", runtime_days=360, keep_zarr=False,
         independent=True, source="argo"):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)

    pool = None
    if source == "argo":
        pool = load_deployment_pool(box=box)
        print(f"deployment pool: {len(pool)} real Argo deployments "
              f"(pooled 2010-2019), median lifespan {pool['lifespan_days'].median():.0f} d")

    if years is None:
        years = sorted({int(os.path.basename(f).split("_")[-2])
                        for f in glob.glob(os.path.join(VEL_DIR, "velocity_daily_park_*.nc"))})
        years = [y for y in years
                 if len(glob.glob(os.path.join(VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]

    all_prof = []
    for y in years:
        zpath = os.path.join(scratch, f"traj_{box}_{y}.zarr")
        print(f"[{y}] advecting {N_FLOATS} floats x {reps} reps at {PARK_DEPTH:.0f} m ...")
        _, rep = run_year(y, box, reps, zpath, runtime_days, independent, pool)
        all_prof.append(profiles_from_zarr(zpath, rep, y))
        if not keep_zarr:
            import shutil
            shutil.rmtree(zpath, ignore_errors=True)

    prof = pd.concat(all_prof, ignore_index=True)
    w, summ = weights_and_summary(prof, box)

    w.to_csv(os.path.join(cfg.OUT_DIR, f"design_weights_{box}.csv"), index=False)
    summ.to_csv(os.path.join(cfg.OUT_DIR, f"design_summary_{box}.csv"), index=False)
    prof.to_csv(os.path.join(cfg.OUT_DIR, f"design_profiles_{box}.csv"), index=False)

    print("\nPer-year means (averaged over replicates):")
    print(summ.groupby("vel_year").mean(numeric_only=True).round(3).to_string())
    return w, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", help="comma-separated; default all complete years")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--source", default="argo", choices=("argo", "uniform"),
                    help="argo = pooled real deployment positions; uniform = synthetic")
    ap.add_argument("--jitter-ensemble", action="store_true",
                    help="osse.tex-style perturbed members (NOT iid; invalid for D1b)")
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--days", type=int, default=360)  # 36 cycles; stops before the last daily field
    ap.add_argument("--keep-zarr", action="store_true")
    a = ap.parse_args()
    main([int(y) for y in a.years.split(",")] if a.years else None,
         a.reps, a.box, a.days, a.keep_zarr, not a.jitter_ensemble, a.source)
