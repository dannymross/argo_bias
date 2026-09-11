"""Lagrangian design statistics per velocity year, for testing D1b.

D1b asserts the cell-weight vectors are identically distributed across velocity
years, w^(l) =d= w^(l'). That is a statement about trajectories, so the Eulerian
diagnostics in velocity_diagnostics.py can only flag drivers -- this produces
the weight vectors themselves.

Deliberately a PROXY for the full park-and-profile OSSE: pure horizontal
advection at the park depth, no ascent/descent/surface drift. The nine-day park
dominates displacement (osse.tex sec:simulation), so this captures the occupancy
structure D1b is about at a fraction of the cost. It is a diagnostic, not the
experiment.

Two deployment arms, differing only in deployment design (both draw lifespans
from the same empirical pool, and both key their RNG on (year, rep) so the MD
designs are iid as Lemma 1 requires):

  argo    -- positions drawn from the pooled real Argo deployment record
  uniform -- space-time uniform with respect to P: equal-area in latitude,
             uniform in day-of-year, seeded well beyond the box

Contrasting them separates advective sampling bias from deployment-design bias.

Run from the repo root:
    python code/osse/design_invariance.py [--years 2015] [--reps 5] [--box boxB]
                                          [--source argo|uniform]
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
# widens the finite real-deployment pool that replicate realisations draw from.
SEED_MARGIN = 4.0
# The uniform arm must not starve the box of inflow, so its margin exceeds the
# annual rms displacement (4 deg is only 0.6x of it in lat, 0.5x in lon).
SEED_MARGIN_UNIFORM = 8.0
N_FLOATS_UNIFORM = 160   # deployed across the year; the array ramps from empty,
                         # so this is ~N_FLOATS in mean standing stock
JITTER_KM = 5.0          # per-replicate seed perturbation

# Steady-state arm: deploy from SPIN_DAYS before the analysis year, on a release
# schedule calibrated (calibrate_release.py) to hold the in-box standing stock
# flat across [0, run_days). A CONSTANT rate does not work -- the empirical
# lifespan pool's median is multi-year, so floats keep re-entering for years and
# the stock is still ramping after ~900 d of spin-up. The spin-up runs on
# the PRECEDING years' velocity; cycling the analysis year's own flow would send
# floats through the same eddies twice and inflate exactly the aligned coupling
# the experiment is built to measure.
SPIN_DAYS = 540
N_FLOATS_STEADY = 307    # over SPIN_DAYS + run_days; ~48 in-box standing stock
                         # (calibrated on the clipped seeding region: every seed lands
                         #  in the velocity domain, so none are deleted on step one)


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


def deployment_plan_uniform(pool, box, n_floats=N_FLOATS_UNIFORM, reps=5, year=0,
                            margin=SEED_MARGIN_UNIFORM, seed_days=cfg.DAYS_PER_YEAR):
    """Space-time uniform deployment: the design that samples P before advection.

    Latitude is drawn uniform in sin(lat), not in degrees: P is the normalised
    area measure, and uniform-in-degrees tilts it by 1/cos(lat) (16% across box
    B), which would put a deployment component into the advective arm at t = 0.
    Day-of-year is uniform so the array is an age mixture rather than one cohort.
    Lifespans come from the same empirical pool as the real-deployment arm, so
    the two arms differ in deployment design alone and not in mortality.
    """
    (la0, la1), (lo0, lo1) = seed_bounds(box, margin)
    s0, s1 = np.sin(np.deg2rad([la0, la1]))
    life_pool = pool["lifespan_days"].to_numpy()

    out = []
    for d in range(reps):
        r = np.random.default_rng([3000, int(year), d])
        out.append((np.rad2deg(np.arcsin(r.uniform(s0, s1, n_floats))),
                    r.uniform(lo0, lo1, n_floats),
                    r.uniform(0.0, seed_days, n_floats),
                    np.full(n_floats, d + 1),
                    r.choice(life_pool, size=n_floats, replace=True)))
    return tuple(np.concatenate(x) for x in zip(*out))


def deployment_plan_steady(pool, box, n_floats=N_FLOATS_STEADY, reps=5, year=0,
                           margin=SEED_MARGIN_UNIFORM, spin_days=SPIN_DAYS,
                           run_days=360, rep_offset=0):
    """Uniform deployment at a constant rate, started before the analysis year.

    Identical to deployment_plan_uniform in space; the difference is that
    release day runs from -spin_days, so during [0, run_days) the standing stock
    is a stationary renewal process rather than an array ramping up from empty.
    That removes both the late-loaded day-of-year marginal and the collinearity
    between float age and day of year.
    """
    (la0, la1), (lo0, lo1) = seed_bounds(box, margin)
    s0, s1 = np.sin(np.deg2rad([la0, la1]))
    life_pool = pool["lifespan_days"].to_numpy()

    sched = release_schedule(box, spin_days, run_days)

    # rep_offset lets an array job split the replicates across tasks: the RNG is
    # keyed on (year, d), so without it every chunk would redraw the same designs.
    out = []
    for d in range(rep_offset, rep_offset + reps):
        r = np.random.default_rng([4000, int(year), d])
        out.append((np.rad2deg(np.arcsin(r.uniform(s0, s1, n_floats))),
                    r.uniform(lo0, lo1, n_floats),
                    sched(r, n_floats),
                    np.full(n_floats, d + 1),
                    r.choice(life_pool, size=n_floats, replace=True)))
    return tuple(np.concatenate(x) for x in zip(*out))


def release_schedule(box, spin_days, run_days, path=None):
    """Sampler for release day, from calibrate_release.py's flat-stock schedule.

    Falls back to a constant rate if no schedule has been calibrated, which
    leaves the standing stock ramping ~1.4x across the analysis year.
    """
    path = path or os.path.join(cfg.OUT_DIR, f"release_schedule_{box}.csv")
    if not os.path.exists(path):
        warnings.warn(f"{path} missing: constant-rate release, stock will ramp")
        return lambda r, n: r.uniform(-spin_days, run_days, n)

    tab = pd.read_csv(path)
    day, w = tab["release_day"].to_numpy(float), tab["weight"].to_numpy(float)
    if day.min() != -spin_days:
        raise ValueError(f"{path} spans [{day.min():.0f}, {day.max():.0f}] but "
                         f"spin_days={spin_days}: recalibrate")
    step = float(np.diff(day).min())
    cdf = np.cumsum(w / w.sum())
    return lambda r, n: (day[np.searchsorted(cdf, r.random(n)).clip(max=len(day) - 1)]
                         + r.uniform(0.0, step, n))


def park_files(year, spin_days=0):
    """Park-depth velocity files covering [Jan 1 of year - spin_days, year end].

    The spin-up may reach back more than one calendar year, so every intervening
    month must be present -- a gap in the file list is a gap in the FieldSet.
    """
    start = np.datetime64(f"{year}-01-01") - np.timedelta64(int(max(spin_days, 0)), "D")
    months = pd.period_range(str(start)[:7], f"{year}-12", freq="M")
    files = [os.path.join(VEL_DIR, f"velocity_daily_park_{m.year}_{m.month:02d}.nc")
             for m in months]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{year}: {len(missing)} park velocity files missing, "
                                f"first {os.path.basename(missing[0])}")
    return files


def velocity_extent(path=None, inset=0.25):
    """Seedable (lat, lon) bounds of the park velocity grid, from the files.

    Read rather than hard-coded: the buffer region lives in the download config,
    and a seed outside it is deleted on the first kernel step. The inset keeps
    seeds a few grid cells clear of the edge so they are not lost to
    interpolation at the boundary.
    """
    path = path or sorted(glob.glob(os.path.join(VEL_DIR, "velocity_daily_park_*.nc")))[0]
    with xr.open_dataset(path) as ds:
        return ((float(ds["latitude"].min()) + inset, float(ds["latitude"].max()) - inset),
                (float(ds["longitude"].min()) + inset, float(ds["longitude"].max()) - inset))


def seed_bounds(box, margin):
    """Box +/- margin, clipped to the velocity domain.

    Box B sits only 5 deg below the northern edge of the buffer, so an 8 deg
    margin puts 9.5% of equal-area seeds outside the domain, where they are
    deleted before contributing. Clipping is distribution-neutral -- those seeds
    never entered any design -- and simply stops wasting them.

    NOT masked to ocean, and box B's margin reaches the North American coast:
    3.4% of its seeds land on dry cells (0.1% for box A), sit motionless, and
    emit a position sample every cycle for the whole run. Harmless to Zhat,
    which evaluates only on the analysis mask, and the surviving seeds stay
    area-uniform over the ocean part of the rectangle -- it costs ~3% of box B's
    effective array. KNOWN, DEFERRED: rejecting off-mask seeds at draw time
    changes the deployment law, so it needs a full re-simulation and should be
    batched with the next change that also needs one.
    """
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
    (vla0, vla1), (vlo0, vlo1) = velocity_extent()
    return ((max(la0 - margin, vla0), min(la1 + margin, vla1)),
            (max(lo0 - margin, vlo0), min(lo1 + margin, vlo1)))


def park_files_available(year, spin_days=0):
    try:
        park_files(year, spin_days)
        return True
    except FileNotFoundError:
        return False


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


def run_year(year, box, reps, out_zarr, runtime_days=360, independent=True, pool=None,
             source="argo", n_floats_uniform=N_FLOATS_UNIFORM, spin_days=0, rep_offset=0):
    """Advect the plan through one velocity year at the park depth.

    Floats leaving the velocity buffer are deleted rather than reflected: the
    box is not closed, and a float advected out of the domain has left the
    experiment. The per-year loss rate is itself a D1b diagnostic, so it is
    reported (frac_early_stop) rather than silently absorbed.
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

    fieldset = FieldSet.from_netcdf(
        park_files(year, spin_days),
        variables={"U": "uo", "V": "vo"},
        dimensions={"time": "time", "depth": "depth", "lat": "latitude", "lon": "longitude"},
        allow_time_extrapolation=False,
    )
    if source == "steady":
        lat, lon, doy, rep, life = deployment_plan_steady(
            pool, box, n_floats=n_floats_uniform, reps=reps, year=year,
            spin_days=spin_days, run_days=runtime_days, rep_offset=rep_offset)
    elif source == "uniform":
        lat, lon, doy, rep, life = deployment_plan_uniform(
            pool, box, n_floats=n_floats_uniform, reps=reps, year=year,
            seed_days=runtime_days)
    elif pool is not None:
        lat, lon, rep, life = deployment_plan_argo(pool, reps=reps, year=year)
        doy = np.zeros(len(lat))
    else:
        lat, lon, rep = deployment_plan(box, reps=reps, independent=independent)
        doy, life = np.zeros(len(lat)), np.zeros(len(lat))
    # doy is measured from 1 January of the analysis year and is negative during
    # the spin-up, so the design is a function of (omega, V_{l-1}, V_l). For the
    # non-steady arms spin_days = 0 and releases stay inside the velocity year.
    t0 = np.datetime64(f"{year}-01-01", "s")
    pset = ParticleSet(fieldset=fieldset, pclass=ArgoFloat,
                       lon=lon, lat=lat, depth=np.full(len(lat), PARK_DEPTH),
                       time=t0 + (doy * 86400.0).astype("int64").astype("timedelta64[s]"),
                       lifespan=life * 86400.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pset.execute([AdvectionRK4, DeleteOutOfBounds, Mortality],
                     endtime=t0 + np.timedelta64(runtime_days * 86400, "s"),
                     dt=np.timedelta64(1, "h"),
                     output_file=pset.ParticleFile(name=out_zarr,
                                                   outputdt=np.timedelta64(CYCLE_DAYS, "D")),
                     verbose_progress=False)
    return out_zarr, rep, life


def monthly_stock(path, box, year, life, reps, run_days=360, spin_days=0):
    """Per-calendar-month float budget over the whole run, spin-up included.

    Must run before the trajectory record is discarded: profiles_from_zarr clips
    to the analysis window, so this is the only point at which the spin-up is
    observable. Counts are totals over replicates; n_reps is carried so callers
    can normalise.

    Retirement and domain exit are separable because death day is release +
    lifespan: a float that stops reporting before the run ends without reaching
    that day left the velocity domain. Both are keyed on the last profile, so
    they are mutually exclusive.
    """
    ds = xr.open_zarr(path)
    lat, lon, t = ds["lat"].values, ds["lon"].values, ds["time"].values
    pid = np.asarray(ds["trajectory"].values)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    ds.close()
    if pid.min() < 0 or pid.max() >= len(life):
        raise ValueError(f"{path}: trajectory ids outside the seeded ParticleSet")

    ok = np.isfinite(lat)
    first = ok.argmax(axis=1)
    released = t[np.arange(len(lat)), first]
    ok[np.arange(len(lat)), first] = False          # obs 0 is the deployment
    ym = pd.PeriodIndex(pd.to_datetime(t[ok]), freq="M")
    fid = np.repeat(np.arange(len(lat)), ok.sum(axis=1))

    (sla0, sla1), (slo0, slo1) = seed_bounds(box, SEED_MARGIN_UNIFORM)
    prof = pd.DataFrame({"ym": ym, "fid": fid, "lat": lat[ok], "lon": lon[ok]})
    prof["in_seed"] = prof["lat"].between(sla0, sla1) & prof["lon"].between(slo0, slo1)
    prof["in_box"] = in_box(prof, box)

    g = prof.groupby("ym", observed=True)
    tab = pd.DataFrame({"in_domain": g["fid"].nunique(),
                        "prof_seed": g["in_seed"].sum(),
                        "prof_box": g["in_box"].sum(),
                        "float_box": (prof[prof["in_box"]].groupby("ym", observed=True)
                                      ["fid"].nunique())})
    tab["float_box"] = tab["float_box"].fillna(0)

    span = prof.groupby("fid")["ym"].agg(["min", "max"])
    last_t = pd.Series(pd.to_datetime(t[ok])).groupby(fid).max()
    death = (pd.to_datetime(released[span.index.to_numpy()])
             + pd.to_timedelta(life[pid][span.index.to_numpy()], unit="D"))
    end = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=run_days - CYCLE_DAYS)
    gone = (last_t < end).to_numpy()
    retired = gone & (death <= last_t + pd.Timedelta(days=CYCLE_DAYS)).to_numpy()

    tab["deployed"] = pd.PeriodIndex(pd.to_datetime(released), freq="M").value_counts()
    tab["retired"] = span["max"][retired].value_counts()
    tab["advected_out"] = span["max"][gone & ~retired].value_counts()
    tab = tab.fillna(0.0).sort_index()

    left = tab["retired"] + tab["advected_out"]
    entered = span["min"].value_counts().reindex(tab.index).fillna(0.0)
    bal = (tab["in_domain"].diff() - (entered - left.shift())).abs().iloc[1:]
    if bal.max() > 1e-6:
        raise ValueError(f"stock budget off by {bal.max():.4f} floats/month")

    # Capped at the last profile actually written: the nominal grid runs one tick
    # past the end of the record, which would understate the final month's
    # per-epoch stock by counting an epoch that produced no profiles.
    ticks = pd.date_range(pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=spin_days),
                          min(pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=run_days),
                              pd.to_datetime(t[ok]).max()),
                          freq=f"{CYCLE_DAYS}D")
    tab["epochs"] = (pd.PeriodIndex(ticks, freq="M").value_counts()
                     .reindex(tab.index).fillna(0.0))
    tab["n_reps"] = reps
    return tab.reset_index().rename(columns={"index": "ym"})


def profiles_from_zarr(path, rep, year, run_days=360):
    """Trajectory output -> one profile position per float per cycle.

    Only profiles falling in [0, run_days) of the analysis year are returned;
    spin-up profiles exist solely to bring the array to equilibrium.
    """
    ds = xr.open_zarr(path)
    lat = ds["lat"].values
    lon = ds["lon"].values
    time = ds["time"].values
    n_traj, n_obs = lat.shape

    # Particles deleted before the first output are never written, so the zarr
    # rows are a subset of the seeded ParticleSet. Index the design labels by
    # particle id, not by position -- slicing rep[:n_traj] silently shifts every
    # label past the first deletion and drops whole replicates off the end.
    pid = np.asarray(ds["trajectory"].values)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    if pid.min() < 0 or pid.max() >= len(rep):
        raise ValueError(f"{path}: trajectory ids outside the seeded ParticleSet")

    # doy is days since 1 January of the analysis year: negative during spin-up.
    doy = (time - np.datetime64(f"{year}-01-01", "ns")) / np.timedelta64(1, "D")
    df = pd.DataFrame({
        "vel_year": year,
        "rep": np.repeat(rep[pid], n_obs),
        "float_id": np.repeat(pid, n_obs),
        "cycle": np.tile(np.arange(n_obs), n_traj),
        "lat": lat.ravel(),
        "lon": lon.ravel(),
        "doy": np.floor(np.asarray(doy, dtype=float).ravel()),
    })
    # Drop each float's FIRST record -- its release position, not a post-ascent
    # surfacing. Done before the analysis-window filter, so a float released
    # during spin-up contributes real profiles rather than losing one to this.
    df = df.dropna(subset=["lat", "lon", "doy"]).sort_values(["float_id", "cycle"])
    df = df[df.groupby("float_id").cumcount() > 0]
    return df[df["doy"].between(0, run_days - 1)]


def in_box(df, box):
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
    return df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)


def weights_and_summary(df, box, deg=cfg.COARSE_DEG, days=cfg.COARSE_DAYS,
                        run_days=360):
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
        # Great-circle displacement from each float's first recorded position.
        gl = g.sort_values("cycle").groupby("float_id")
        d0 = gl.first()[["lat", "lon"]]
        dl = gl.last()[["lat", "lon"]]
        dy = (dl["lat"] - d0["lat"]) * 111.0
        dx = (dl["lon"] - d0["lon"]) * 111.0 * np.cos(np.deg2rad(d0["lat"]))
        # Parcels packs every trajectory from its own obs 0, so "cycle" is the
        # float's age in cycles, not calendar position. Under staggered release
        # these must be judged per float against the run end, not a global cycle.
        last_cycle, last_doy = gl["cycle"].max(), gl["doy"].max()
        last_in = gi.groupby("float_id")["cycle"].max().reindex(last_cycle.index)
        rows.append({
            "vel_year": y, "rep": r,
            "n_profiles_in_box": len(gi),
            "n_floats": g["float_id"].nunique(),
            # Stopped recording more than a cycle before the run end: domain
            # exit or mortality, which the profile record cannot separate.
            "frac_early_stop": float((last_doy < run_days - CYCLE_DAYS).mean()),
            "frac_inside": g["inside"].mean(),
            # Distinct floats profiling in the box per cycle window: the standing
            # stock the deployment rate has to be calibrated against.
            "standing_stock": float(gi.groupby(gi["doy"] // CYCLE_DAYS)["float_id"]
                                    .nunique().reindex(range(run_days // CYCLE_DAYS),
                                                       fill_value=0).mean()),
            "residence_days": gi.groupby("float_id").size().mean() * CYCLE_DAYS,
            "retention_end": float(last_in.eq(last_cycle).mean()),
            "rms_disp_km": float(np.sqrt((dx**2 + dy**2).mean())),
            "n_cells_occupied": len(wr),
            "participation_ratio": float(1.0 / np.sum(wr**2)) if len(wr) else 0.0,
        })
    return w, pd.DataFrame(rows)


def main(years=None, reps=10, box="boxB", runtime_days=360, keep_zarr=False,
         independent=True, source="argo", n_floats_uniform=N_FLOATS_UNIFORM,
         tag="", spin_days=0, rep_offset=0):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)

    # The uniform arm needs the pool too -- for lifespans only, so that the two
    # arms share a mortality model and their contrast isolates deployment design.
    pool = None
    if source in ("argo", "uniform", "steady"):
        pool = load_deployment_pool(box=box)
        print(f"deployment pool: {len(pool)} real Argo deployments "
              f"(pooled 2010-2019), median lifespan {pool['lifespan_days'].median():.0f} d")

    if years is None:
        years = sorted({int(os.path.basename(f).split("_")[-2])
                        for f in glob.glob(os.path.join(VEL_DIR, "velocity_daily_park_*.nc"))})
        years = [y for y in years
                 if len(glob.glob(os.path.join(VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        if spin_days > 0:  # the spin-up runs on the preceding years' velocity
            years = [y for y in years if park_files_available(y, spin_days)]

    nf = n_floats_uniform if source in ("uniform", "steady") else N_FLOATS
    sfx = ("" if source == "argo" else f"_{source}") + tag

    all_prof, all_stock = [], []
    for y in years:
        zpath = os.path.join(scratch, f"traj_{box}{sfx}_{y}.zarr")
        print(f"[{y}] advecting {nf} floats x {reps} reps at {PARK_DEPTH:.0f} m "
              f"({spin_days} d spin-up) ...")
        _, rep, life = run_year(y, box, reps, zpath, runtime_days, independent, pool,
                                source, n_floats_uniform, spin_days, rep_offset)
        all_prof.append(profiles_from_zarr(zpath, rep, y, runtime_days))
        # Recorded here because the profile table is clipped to the analysis
        # window and the trajectory record is about to be deleted: without this
        # the spin-up is unrecoverable without re-simulating. Non-fatal on
        # purpose -- it is a diagnostic, and the advection above is the
        # expensive part of the job.
        try:
            all_stock.append(monthly_stock(zpath, box, y, life, reps,
                                           runtime_days, spin_days).assign(vel_year=y))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"[{y}] monthly_stock failed, continuing: {e}")
        if not keep_zarr:
            import shutil
            shutil.rmtree(zpath, ignore_errors=True)

    prof = pd.concat(all_prof, ignore_index=True)
    w, summ = weights_and_summary(prof, box, run_days=runtime_days)

    w.to_csv(os.path.join(cfg.OUT_DIR, f"design_weights_{box}{sfx}.csv"), index=False)
    summ.to_csv(os.path.join(cfg.OUT_DIR, f"design_summary_{box}{sfx}.csv"), index=False)
    prof.to_csv(os.path.join(cfg.OUT_DIR, f"design_profiles_{box}{sfx}.csv"), index=False)
    if all_stock:
        pd.concat(all_stock, ignore_index=True).to_csv(
            os.path.join(cfg.OUT_DIR, f"design_stock_{box}{sfx}.csv"), index=False)

    print("\nPer-year means (averaged over replicates):")
    print(summ.groupby("vel_year").mean(numeric_only=True).round(3).to_string())
    return w, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", help="comma-separated; default all complete years")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--source", default="argo", choices=("argo", "uniform", "steady"),
                    help="argo = pooled real deployment positions; uniform = synthetic "
                         "ramping from empty; steady = synthetic in renewal equilibrium")
    ap.add_argument("--jitter-ensemble", action="store_true",
                    help="osse.tex-style perturbed members (NOT iid; invalid for D1b)")
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--days", type=int, default=360)  # 36 cycles; stops before the last daily field
    ap.add_argument("--keep-zarr", action="store_true")
    ap.add_argument("--n-floats-uniform", type=int, default=N_FLOATS_UNIFORM)
    ap.add_argument("--tag", default="", help="output filename suffix (array jobs)")
    ap.add_argument("--rep-offset", type=int, default=0,
                    help="first replicate index; array tasks must not overlap")
    ap.add_argument("--spin-days", type=int, default=None,
                    help="pre-analysis-year deployment window; default SPIN_DAYS "
                         "for --source steady, 0 otherwise")
    a = ap.parse_args()
    spin = a.spin_days if a.spin_days is not None else (SPIN_DAYS if a.source == "steady" else 0)
    nf = a.n_floats_uniform
    if a.source == "steady" and nf == N_FLOATS_UNIFORM:
        nf = N_FLOATS_STEADY
    main([int(y) for y in a.years.split(",")] if a.years else None,
         a.reps, a.box, a.days, a.keep_zarr, not a.jitter_ensemble, a.source,
         nf, a.tag, spin, a.rep_offset)
