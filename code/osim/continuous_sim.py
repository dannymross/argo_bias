"""Continuous-seeding simulation: one advection per realization over the velocity
record, scored on every analysis year of BOTH boxes from the same deployment.

Floats are seeded uniformly over normalized spacetime -- area-uniform over the
deep (deptho >= park depth) part of a single region spanning both analysis boxes
plus a margin, uniform in release time -- and advected in one continuous run per
realization. The lead from the deployment start to the first scored year brings
the drifting array to renewal equilibrium (constant seed rate against an
empirical lifespan pool), so a stable in-box stock is a by-product of the
seeding, not a calibrated schedule. One deployment feeds both boxes: a profile is
scored against whichever box contains it. Each realization yields the realized
sampling bias

    B_k = Zhat_k - Zbar_k,   Zhat_k = mean over year-k in-box profiles of Z_k,

for every scored year and both boxes at once, flow and field the same calendar
year by construction -- the diagonal of the old velocity-year x field-year grid.
This replaced the per-box spin-up / release-schedule calibration of the old
steady arm, which has since been removed.

Run from the repo root:
    python code/osim/continuous_sim.py [--n-floats 1000] [--reps 1] [--rep-offset 0]
        [--start 2010-01-01] [--end 2019-12-26] [--score-years 2015,2016,2017,2018,2019]
    python code/osim/continuous_sim.py --gather --tag _pilot   # concat array chunks
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
from osim import osim_config as cfg
from osim import deployment as di

PARK_DEPTH = di.PARK_DEPTH
CYCLE_DAYS = di.CYCLE_DAYS
SEED_MARGIN = di.SEED_MARGIN_UNIFORM
BOXES = ("boxA", "boxB")  # both scored from one deployment
# Seed around the region the analysis READS OUT (the buffer), not the estimand.
# Seeding on boxes+8 deg covered only 0.87x the buffer's own area, so the buffer
# edge was populated by advective inflow alone -- a coverage deficit that is an
# artefact of the seeding rectangle and would bias the kriging mass m.
SEED_BOXES = ("buffer",)
# Covers the whole velocity domain. The older static_vel_bathy.nc spans only
# 15-50 N / -70..-30 W, so seed_region silently intersected with it and every run
# to date was seeded 3 deg short in the north and 1 deg in the south of what the
# margin asked for; seed_region now asserts coverage rather than clipping.
BATHY_PATH = os.path.join(cfg.GLORYS_DIR, "static", "static_veldom_bathy.nc")


PROFILE_EXT = ".parquet"


def profile_path(tag, chunk=None):
    sfx = f"_c{chunk}" if chunk is not None else ""
    return os.path.join(cfg.OUT_DIR, f"design_profiles_continuous{tag}{sfx}{PROFILE_EXT}")


def write_profiles(df, tag, chunk=None):
    """Profiles are ~3.3x larger now that the full footprint is kept, and every
    report re-reads them; Parquet is ~10x smaller and ~20x faster than CSV here.
    """
    df.to_parquet(profile_path(tag, chunk), index=False, compression="zstd")
    return profile_path(tag, chunk)


def read_profiles(tag, columns=None):
    """Parquet if present, else the legacy CSV so pre-existing arms still load."""
    p = profile_path(tag)
    if os.path.exists(p):
        return pd.read_parquet(p, columns=columns)
    csv = os.path.join(cfg.OUT_DIR, f"design_profiles_continuous{tag}.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(f"{p} (and no legacy {os.path.basename(csv)})")
    return pd.read_csv(csv, usecols=columns)


def seed_region(boxes=SEED_BOXES, margin=SEED_MARGIN, min_depth=PARK_DEPTH):
    """Deep native cells over (union of boxes + margin) cap velocity domain.

    The seed region is defined once from the bathymetry rather than as a
    rectangle sampled with rejection: a cell is eligible iff deptho >= min_depth
    (the float must reach park depth). Returns the sin-latitude and longitude
    edges of each eligible cell (for exact within-cell area-uniform draws) and
    normalised area weights.
    """
    la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes) - margin
    la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes) + margin
    lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes) - margin
    lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes) + margin
    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()

    with xr.open_dataset(BATHY_PATH) as ds:
        dep = ds["deptho"].sel(
            latitude=slice(max(la0, vla0), min(la1, vla1)),
            longitude=slice(max(lo0, vlo0), min(lo1, vlo1)))
        lat = dep["latitude"].values.astype(float)
        lon = dep["longitude"].values.astype(float)
        deep = np.isfinite(dep.values) & (dep.values >= min_depth)

    want = (max(la0, vla0), min(la1, vla1), max(lo0, vlo0), min(lo1, vlo1))
    got = (lat.min(), lat.max(), lon.min(), lon.max())
    if got[0] > want[0] + 0.5 or got[1] < want[1] - 0.5 or got[2] > want[2] + 0.5 or got[3] < want[3] - 0.5:
        raise ValueError(f"{os.path.basename(BATHY_PATH)} covers lat {got[0]:.1f}-{got[1]:.1f}, "
                         f"lon {got[2]:.1f}-{got[3]:.1f}, short of the requested seed window "
                         f"lat {want[0]:.1f}-{want[1]:.1f}, lon {want[2]:.1f}-{want[3]:.1f}")

    late, lone = cfg.cell_edges(lat), cfg.cell_edges(lon)
    jla, jlo = np.where(deep)
    slat0, slat1 = np.sin(np.deg2rad(late[jla])), np.sin(np.deg2rad(late[jla + 1]))
    area = slat1 - slat0  # spherical cell area is proportional to d(sin lat)
    return {"slat0": slat0, "slat1": slat1, "lon0": lone[jlo], "lon1": lone[jlo + 1],
            "w": area / area.sum(), "n_cells": int(deep.sum())}


def deployment_plan(n_floats, rep, t_start, t_end, boxes=SEED_BOXES, margin=SEED_MARGIN):
    """One realisation's seeds: area-uniform in space over the deep region,
    uniform in release time over [t_start, t_end), lifespans from the Argo pool.
    """
    pool = di.load_deployment_pool()
    region = seed_region(boxes, margin)
    r = np.random.default_rng([5000, int(rep)])

    c = r.choice(region["n_cells"], size=n_floats, p=region["w"])
    lat = np.rad2deg(np.arcsin(r.uniform(region["slat0"][c], region["slat1"][c])))
    lon = r.uniform(region["lon0"][c], region["lon1"][c])

    span = (np.datetime64(t_end) - np.datetime64(t_start)) / np.timedelta64(1, "s")
    rel = np.datetime64(t_start) + (r.uniform(0.0, span, n_floats)
                                    ).astype("int64").astype("timedelta64[s]")
    life = r.choice(pool["lifespan_days"].to_numpy(), size=n_floats, replace=True)
    return lat, lon, rel, life.astype(float)


def _window_files(t_start, t_end):
    months = pd.period_range(str(t_start)[:7], str(t_end)[:7], freq="M")
    files = [os.path.join(di.VEL_DIR, f"velocity_daily_park_{m.year}_{m.month:02d}.nc")
             for m in months]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} park velocity files missing over "
                                f"[{t_start}, {t_end}], first {os.path.basename(missing[0])}")
    return files


def run_continuous(rep, out_zarr, n_floats, t_start, t_end, margin=SEED_MARGIN, seeds=None):
    """Advect one realisation continuously over [t_start, t_end].

    `seeds`, if given, is a (lat, lon, release_datetime64, lifespan_days) tuple used
    verbatim (for the real-Argo deployment arms); otherwise the uniform space-time
    deployment_plan is drawn.
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
        _window_files(t_start, t_end),
        variables={"U": "uo", "V": "vo"},
        dimensions={"time": "time", "depth": "depth", "lat": "latitude", "lon": "longitude"},
        allow_time_extrapolation=False,
    )
    if seeds is None:
        lat, lon, rel, life = deployment_plan(n_floats, rep, t_start, t_end, margin=margin)
    else:
        lat, lon, rel, life = seeds
    pset = ParticleSet(fieldset=fieldset, pclass=ArgoFloat,
                       lon=lon, lat=lat, depth=np.full(len(lat), PARK_DEPTH),
                       time=rel.astype("datetime64[s]"), lifespan=life * 86400.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pset.execute([AdvectionRK4, DeleteOutOfBounds, Mortality],
                     endtime=np.datetime64(t_end, "s"),
                     dt=np.timedelta64(1, "h"),
                     output_file=pset.ParticleFile(name=out_zarr,
                                                   outputdt=np.timedelta64(CYCLE_DAYS, "D")),
                     verbose_progress=False)
    return out_zarr


def leap_dropped_doy(times):
    """(calendar year, 0-based day-of-year) on the Feb-29-dropped 365-day calendar.

    The residual field drops Feb 29 (osim_config.drop_leap_day), so a profile's
    doy must be indexed the same way. Feb 29 itself has no field slot and is
    flagged with doy = -1 for the caller to drop.
    """
    t = pd.DatetimeIndex(times)
    year = np.asarray(t.year)
    doy0 = np.asarray(t.dayofyear) - 1
    leap = np.asarray(t.is_leap_year)
    is_feb29 = leap & (np.asarray(t.month) == 2) & (np.asarray(t.day) == 29)
    doy0 = np.where(leap & (doy0 >= 60), doy0 - 1, doy0)  # shift days after Feb 29
    doy0 = np.where(is_feb29, -1, doy0)
    return year, doy0


def profiles_from_zarr(path, rep, boxes=BOXES):
    """Trajectory output -> profile positions tagged with (box, year, doy).

    Drops each float's release record and any Feb-29 profile, and keeps EVERY
    remaining position, in or out of a box; `box` is NA outside. Recording the
    whole footprint is what lets the analysis or conditioning region be widened
    later without re-advecting -- an earlier version kept only in-box profiles
    and discarded the trajectories, which cost a full re-run when the kriging
    needed profiles from outside the boxes. Consumers filter on `box` (or on
    lat/lon for an arbitrary region); scoring years are applied in the estimator.

    Note this future-proofs the *readout* region only, up to wherever the seeded
    array actually reaches -- widening beyond the seeding footprint still needs a
    re-run (see SEED_MARGIN).
    """
    ds = xr.open_zarr(path)
    lat, lon, time = ds["lat"].values, ds["lon"].values, ds["time"].values
    pid = np.asarray(ds["trajectory"].values)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    ds.close()
    n_traj, n_obs = lat.shape

    df = pd.DataFrame({
        "rep": int(rep),
        "float_id": np.repeat(pid, n_obs),
        "cycle": np.tile(np.arange(n_obs), n_traj),
        "lat": lat.ravel(), "lon": lon.ravel(), "time": time.ravel(),
    }).dropna(subset=["lat", "lon", "time"])

    df = df.sort_values(["float_id", "time"])
    df = df[df.groupby("float_id").cumcount() > 0]           # drop the release record
    df["year"], df["doy"] = leap_dropped_doy(df["time"].to_numpy())

    df["box"] = pd.NA
    for b in boxes:
        df.loc[di.in_box(df, b), "box"] = b
    # Feb-29 rows are kept with doy = -1: they have no residual-field slot, but the
    # exact `time` makes the position usable against any other discretization.
    # sample_residuals.gather drops them via its (idoy >= 0) guard.
    # Raw `time` is kept alongside the leap-dropped (year, doy): the estimator uses
    # only (year, doy, lat, lon), but the absolute timestamp lets these positions
    # be re-scored against a residual field on a different time discretization
    # without re-advecting.
    cols = ["rep", "float_id", "cycle", "time", "lat", "lon", "box", "year", "doy"]
    return df[cols].reset_index(drop=True)


# --------------------------- estimator --------------------------------------

def diagonal_estimates(prof, score_years=None):
    """Per-(realisation, box, year) Zhat_k over year-k in-box profiles, and B_k.

    Z is gathered for every analysis year at each position (one cell-day lookup)
    against that box's residual field, then only the profile's own calendar year
    is kept -- the diagonal the continuous seeding produces directly.
    """
    from osim import mean_mc as mm
    from osim import sample_residuals as sr

    out, ndropped = [], 0
    for box in sorted(prof["box"].dropna().unique()):
        sub = prof[prof["box"] == box].copy()
        z = sr.load_residual_field(box)
        years = [int(y) for y in z["year"].values]
        sampled, nd = sr.gather(sub, z)
        ndropped += nd
        col = {y: j for j, y in enumerate(years)}
        zdiag = sampled[[f"z_{y}" for y in years]].to_numpy()
        yr = sampled["year"].to_numpy()
        sampled["z_own"] = zdiag[np.arange(len(sampled)), [col[y] for y in yr]]
        if score_years is not None:
            sampled = sampled[sampled["year"].isin(score_years)]

        mm.set_box(box)
        zf, wet, w = mm.load_field(box)
        zbar = mm.domain_means(zf, wet, w)[0].set_index("year")["zbar"]

        g = sampled.groupby(["rep", "year"], sort=True)
        stock = (sampled.assign(cyc=sampled["doy"] // CYCLE_DAYS)
                 .groupby(["rep", "year", "cyc"])["float_id"].nunique()
                 .groupby(level=[0, 1]).mean())
        e = pd.DataFrame({"zhat": g["z_own"].mean(), "n": g.size(),
                          "n_floats": g["float_id"].nunique(),
                          "standing_stock": stock}).reset_index()
        e.insert(0, "box", box)
        e["zbar"] = e["year"].map(zbar)
        e["B"] = e["zhat"] - e["zbar"]
        out.append(e)
    return pd.concat(out, ignore_index=True), ndropped


def uniform_reference(est, seed=0):
    """U-full check: for each (box, rep, year) draw n iid points from P; E[B]=0.

    MUST be run once over the FULL ensemble, not per array chunk: a single rng
    walks all rows, so every rep gets an independent draw. Computing it per chunk
    (each seeded identically) makes the reps share their random stream, which
    underestimates the reference spread and biases its mean toward one draw's
    sampling error -- gather_chunks therefore recomputes it on the merged frame.
    """
    from osim import mean_mc as mm
    rng = np.random.default_rng(seed)
    ub = np.full(len(est), np.nan)
    for box in est["box"].unique():
        mm.set_box(box)
        zf, wet, w = mm.load_field(box)
        yidx = {y: j for j, y in enumerate(mm.YEARS)}
        zbar = mm.domain_means(zf, wet, w)[0].set_index("year")["zbar"]
        cw = np.cumsum(w)
        for i in est.index[est["box"] == box]:
            n, y = int(est.at[i, "n"]), int(est.at[i, "year"])
            cells = wet[np.searchsorted(cw, rng.random(n)).clip(max=len(w) - 1)]
            d = rng.integers(0, cfg.DAYS_PER_YEAR, n)
            ub[i] = zf[yidx[y], d, cells].mean() - zbar[y]
    est = est.copy()
    est["B_unif"] = ub
    return est


def gather_chunks(tag):
    """Concatenate array-chunk estimate/profile CSVs into one ensemble each."""
    from osim import mean_mc as mm
    ep = sorted(glob.glob(os.path.join(cfg.OUT_DIR, f"estimates_continuous{tag}_c*.csv")))
    if not ep:
        raise FileNotFoundError(f"estimates_continuous{tag}_c*.csv")
    est = pd.concat((pd.read_csv(f) for f in ep), ignore_index=True)
    dup = est.duplicated(["box", "rep", "year"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (box, rep, year) rows: chunk rep-offsets overlap")
    est = uniform_reference(est)  # recompute on the full ensemble; per-chunk values share a seed
    out = os.path.join(cfg.OUT_DIR, f"estimates_continuous{tag}.csv")
    est.to_csv(out, index=False)

    pp = sorted(glob.glob(profile_path(tag, chunk="*")))
    if pp:
        prof = pd.concat((pd.read_parquet(f) for f in pp), ignore_index=True)
        pout = write_profiles(prof, tag)
        print(f"{len(pp)} profile chunks, {len(prof):,} positions -> {pout}")

    _print_summary(est)
    print(f"\n{len(ep)} chunks, {est['rep'].nunique()} reps -> {out}")
    return est


def _print_summary(est):
    from osim import mean_mc as mm
    print("\nPer-year bias, averaged over realisations (GJ/m2):")
    has_u = "B_unif" in est.columns
    for box, g in est.groupby("box"):
        agg = dict(mean_B=("B", "mean"), sd_B=("B", "std"),
                   mean_stock=("standing_stock", "mean"),
                   mean_n=("n", "mean"), reps=("rep", "nunique"))
        if has_u:
            agg["mean_B_unif"] = ("B_unif", "mean")
        s = g.groupby("year").agg(**agg)
        s[["mean_B", "sd_B"] + (["mean_B_unif"] if has_u else [])] *= mm.J_TO_GJ
        print(f"\n[{box}]")
        print(s.round(3).to_string())


def main(n_floats=1000, reps=1, rep_offset=0, t_start=None, t_end=None,
         score_years=None, keep_zarr=False, tag=""):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)

    if t_start is None or t_end is None:
        yrs = sorted({int(os.path.basename(f).split("_")[-2])
                      for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
        yrs = [y for y in yrs
               if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        t_start = t_start or f"{yrs[0]}-01-01"
        t_end = t_end or f"{yrs[-1]}-12-26"  # a few days inside the last daily field

    sfx = f"_continuous{tag}"
    print(f"{reps} reps (offset {rep_offset}) x {n_floats} floats, continuous "
          f"{t_start}..{t_end}, score {score_years or 'all field years'}")
    all_prof = []
    for d in range(rep_offset, rep_offset + reps):
        zpath = os.path.join(scratch, f"traj{sfx}_r{d}.zarr")
        print(f"  rep {d}: advecting {n_floats} floats ...", flush=True)
        run_continuous(d, zpath, n_floats, t_start, t_end)
        all_prof.append(profiles_from_zarr(zpath, d))
        if not keep_zarr:
            import shutil
            shutil.rmtree(zpath, ignore_errors=True)

    prof = pd.concat(all_prof, ignore_index=True)
    write_profiles(prof, tag)

    est, ndropped = diagonal_estimates(prof, score_years)
    est = uniform_reference(est)
    est.to_csv(os.path.join(cfg.OUT_DIR, f"estimates{sfx}.csv"), index=False)
    print(f"{len(prof):,} in-box profiles, {ndropped:,} dropped (masked) -> estimates{sfx}.csv")
    _print_summary(est)
    return est


def run_illustration(out_csv=None, n_floats=100, rep=0, t_start=None, t_end=None, margin=SEED_MARGIN):
    """Run one realization keeping ALL positions (in-box and out), tagged by year.

    For the advection-illustration map only: a legible subset of floats from one
    realization's deployment law, full drift paths. Writes float_id, cycle, lat,
    lon, time, year, box (whichever box contains the point, else blank).
    """
    import shutil
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)
    if t_start is None or t_end is None:
        yrs = sorted({int(os.path.basename(f).split("_")[-2])
                      for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
        yrs = [y for y in yrs
               if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        t_start = t_start or f"{yrs[0]}-01-01"
        t_end = t_end or f"{yrs[-1]}-12-26"
    zpath = os.path.join(scratch, f"traj_illustration_r{rep}.zarr")
    shutil.rmtree(zpath, ignore_errors=True)
    print(f"illustration: {n_floats} floats, {t_start}..{t_end}", flush=True)
    run_continuous(rep, zpath, n_floats, t_start, t_end, margin)

    ds = xr.open_zarr(zpath)
    lat, lon, time = ds["lat"].values, ds["lon"].values, ds["time"].values
    pid = np.asarray(ds["trajectory"].values)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    ds.close()
    n_traj, n_obs = lat.shape
    df = pd.DataFrame({"float_id": np.repeat(pid, n_obs),
                       "cycle": np.tile(np.arange(n_obs), n_traj),
                       "lat": lat.ravel(), "lon": lon.ravel(), "time": time.ravel()}
                      ).dropna(subset=["lat", "lon", "time"])
    df = df.sort_values(["float_id", "time"])
    df = df[df.groupby("float_id").cumcount() > 0]           # drop the release record
    df["year"] = pd.DatetimeIndex(df["time"]).year
    df["box"] = pd.NA
    for b in BOXES:
        df.loc[di.in_box(df, b), "box"] = b
    out_csv = out_csv or os.path.join(cfg.OUT_DIR, "illustration_tracks.csv")
    df.to_csv(out_csv, index=False)
    shutil.rmtree(zpath, ignore_errors=True)
    print(f"{len(df):,} positions, {df['float_id'].nunique()} floats -> {out_csv}")
    return df


def run_actual_trajectories(out_csv=None, pool_csv=None, t_start="2010-01-01",
                            t_end=None, rep=0):
    """Advect the real Argo deployments verbatim, keeping the FULL drift path.

    Same seeds as the argo-actual arm (each cleaned float's first profile in the
    seed region), but every 10-day position is kept, in-box or out, so the
    GLORYS-park track can be compared against the same real float's actual profile
    positions (the trajectory-divergence attribution). Writes the tracks and the
    seed->real-float mapping (seed_id = Parcels trajectory index = pool row).
    """
    import shutil
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)
    if t_end is None:
        yrs = sorted({int(os.path.basename(f).split("_")[-2])
                      for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
        yrs = [y for y in yrs
               if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        t_end = f"{yrs[-1]}-12-26"

    pool = argo_deployment_pool(t_start=t_start, t_end=t_end).reset_index(drop=True)
    pool.insert(0, "seed_id", np.arange(len(pool)))
    pool_csv = pool_csv or os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv")
    pool.to_csv(pool_csv, index=False)
    rel = pool["date"].dt.tz_localize(None).to_numpy().astype("datetime64[s]")
    seeds = (pool["lat"].to_numpy(), pool["lon"].to_numpy(), rel, pool["lifespan"].to_numpy())

    zpath = os.path.join(scratch, f"traj_actual_full_r{rep}.zarr")
    shutil.rmtree(zpath, ignore_errors=True)
    print(f"actual-full: {len(pool)} real deployments, {t_start}..{t_end}", flush=True)
    run_continuous(rep, zpath, len(pool), t_start, t_end, seeds=seeds)

    ds = xr.open_zarr(zpath)
    lat, lon, time = ds["lat"].values, ds["lon"].values, ds["time"].values
    pid = np.asarray(ds["trajectory"].values)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    ds.close()
    n_traj, n_obs = lat.shape
    df = pd.DataFrame({"seed_id": np.repeat(pid, n_obs),
                       "cycle": np.tile(np.arange(n_obs), n_traj),
                       "lat": lat.ravel(), "lon": lon.ravel(), "time": time.ravel()}
                      ).dropna(subset=["lat", "lon", "time"])
    df = df.sort_values(["seed_id", "time"]).reset_index(drop=True)
    out_csv = out_csv or os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv")
    df.to_csv(out_csv, index=False)
    shutil.rmtree(zpath, ignore_errors=True)
    print(f"{len(df):,} positions, {df['seed_id'].nunique()} floats -> {out_csv}")
    return df


def argo_deployment_pool(cleaned=None, boxes=BOXES, margin=SEED_MARGIN,
                         t_start=None, t_end=None, deep=True):
    """Real Argo deployments (each cleaned float's first profile) in the seed region.

    Returns float_id, lat, lon, date (deploy datetime), lifespan (days = last-first
    cleaned profile). Restricted to first-profile inside (union box + margin) cap the
    velocity domain, deploy date in [t_start, t_end), and (deep) deptho >= park depth.
    """
    from osim import mean_mc as mm
    d = mm.clean_argo() if cleaned is None else cleaned
    g = d.sort_values("date").groupby("float_id")
    dep = g[["lat", "lon", "date"]].first()
    dep["lifespan"] = (g["date"].last() - dep["date"]).dt.total_seconds() / 86400.0

    la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes) - margin
    la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes) + margin
    lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes) - margin
    lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes) + margin
    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()
    dep = dep[dep["lat"].between(max(la0, vla0), min(la1, vla1))
              & dep["lon"].between(max(lo0, vlo0), min(lo1, vlo1))]
    if t_start is not None:
        dep = dep[dep["date"] >= pd.Timestamp(t_start, tz="UTC")]
    if t_end is not None:
        dep = dep[dep["date"] < pd.Timestamp(t_end, tz="UTC")]
    if deep:
        with xr.open_dataset(BATHY_PATH) as ds:
            dd = ds["deptho"]
            ii = cfg.containing_index(dep["lat"].to_numpy(), dd["latitude"].values)
            jj = cfg.containing_index(dep["lon"].to_numpy(), dd["longitude"].values)
            ok = (ii >= 0) & (jj >= 0)
            keep = np.zeros(len(dep), bool)
            keep[ok] = np.isfinite(dd.values[ii[ok], jj[ok]]) & (dd.values[ii[ok], jj[ok]] >= PARK_DEPTH)
        dep = dep[keep]
    return dep.reset_index()


def main_argo(mode="actual", reps=1, rep_offset=0, n=None, t_start="2010-01-01",
              t_end=None, score_years=None, tag="", keep_zarr=False):
    """Real-Argo deployment arm: advect floats seeded from the real deployment record
    through GLORYS. mode='actual' uses the deployments verbatim (one realization, the
    advection-fidelity check); mode='boot' resamples them with replacement at the real
    rate (n = pool size unless given), one draw per rep (RNG keyed on rep, jittered)."""
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = os.path.join(cfg.OUT_DIR, "_traj")
    os.makedirs(scratch, exist_ok=True)
    if t_end is None:
        yrs = sorted({int(os.path.basename(f).split("_")[-2])
                      for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
        yrs = [y for y in yrs
               if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        t_end = f"{yrs[-1]}-12-26"

    pool = argo_deployment_pool(t_start=t_start, t_end=t_end)
    rel = pool["date"].dt.tz_localize(None).to_numpy().astype("datetime64[s]")
    lat, lon, life = pool["lat"].to_numpy(), pool["lon"].to_numpy(), pool["lifespan"].to_numpy()
    sfx = f"_argo_{mode}{tag}"
    print(f"argo pool: {len(pool)} real deployments in seed region, {t_start}..{t_end}")

    all_prof = []
    for d in range(rep_offset, rep_offset + reps):
        if mode == "actual":
            seeds = (lat, lon, rel, life)
        else:
            r = np.random.default_rng([6000, int(d)])
            nn = n or len(pool)
            idx = r.integers(0, len(pool), nn)
            jl = r.normal(0.0, di.JITTER_KM / 111.0, nn)
            jo = r.normal(0.0, di.JITTER_KM / (111.0 * np.cos(np.deg2rad(lat[idx]))), nn)
            seeds = (lat[idx] + jl, lon[idx] + jo, rel[idx], life[idx])
        zpath = os.path.join(scratch, f"traj{sfx}_r{d}.zarr")
        print(f"  rep {d}: advecting {len(seeds[0])} floats ...", flush=True)
        run_continuous(d, zpath, len(seeds[0]), t_start, t_end, seeds=seeds)
        all_prof.append(profiles_from_zarr(zpath, d))
        if not keep_zarr:
            import shutil
            shutil.rmtree(zpath, ignore_errors=True)

    prof = pd.concat(all_prof, ignore_index=True)
    write_profiles(prof, sfx)
    est, ndropped = diagonal_estimates(prof, score_years)
    est.to_csv(os.path.join(cfg.OUT_DIR, f"estimates_continuous{sfx}.csv"), index=False)
    print(f"{len(prof):,} in-box profiles, {ndropped:,} dropped -> estimates_continuous{sfx}.csv")
    _print_summary(est)
    return est


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-floats", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--start", default=None, help="release/advection start, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="advection end, YYYY-MM-DD")
    ap.add_argument("--score-years", default=None,
                    help="comma-separated years to score; default all residual-field years")
    ap.add_argument("--keep-zarr", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gather", action="store_true", help="concat array chunks for --tag and exit")
    ap.add_argument("--illustrate", action="store_true",
                    help="run one realization keeping all positions -> illustration_tracks.csv")
    ap.add_argument("--argo", choices=("actual", "boot"), default=None,
                    help="real-Argo deployment arm: 'actual' (verbatim) or 'boot' (resampled)")
    ap.add_argument("--argo-traj", action="store_true",
                    help="advect real deployments verbatim keeping full drift paths "
                         "-> actual_full_tracks.csv (trajectory-divergence attribution)")
    a = ap.parse_args()
    sy = [int(y) for y in a.score_years.split(",")] if a.score_years else None
    if a.gather:
        gather_chunks(a.tag)
    elif a.argo_traj:
        run_actual_trajectories(t_start=a.start or "2010-01-01", t_end=a.end)
    elif a.illustrate:
        run_illustration(n_floats=a.n_floats, t_start=a.start, t_end=a.end)
    elif a.argo:
        main_argo(a.argo, a.reps, a.rep_offset, None,   # boot uses the real deployment rate
                  a.start or "2010-01-01", a.end, sy, a.tag, a.keep_zarr)
    else:
        main(a.n_floats, a.reps, a.rep_offset, a.start, a.end, sy, a.keep_zarr, a.tag)
