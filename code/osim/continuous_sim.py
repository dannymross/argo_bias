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
import shutil
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
TRAJ_KEEP_DIR = os.path.join(cfg.OUT_DIR, "_traj")


def scratch_dir():
    """Job-unique node-local directory for Parcels trajectory output.

    Parcels appends to the zarr every output step; with staggered releases those small
    writes take hours on the network filesystem and minutes on local disk.
    """
    job = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    d = os.path.join(os.environ.get("TMPDIR") or "/tmp", f"osim_{os.environ.get('USER', 'u')}_{job}")
    os.makedirs(d, exist_ok=True)
    return d


def keep_traj(zpath):
    os.makedirs(TRAJ_KEEP_DIR, exist_ok=True)
    shutil.copytree(zpath, os.path.join(TRAJ_KEEP_DIR, os.path.basename(zpath)), dirs_exist_ok=True)


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


def _window_files(t_start, t_end, vel_dir=None):
    months = pd.period_range(str(t_start)[:7], str(t_end)[:7], freq="M")
    files = [os.path.join(vel_dir or di.SIM_VEL_DIR, f"velocity_daily_park_{m.year}_{m.month:02d}.nc")
             for m in months]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} park velocity files missing over "
                                f"[{t_start}, {t_end}], first {os.path.basename(missing[0])}")
    return files


SENTINEL_LATLON = (30.0, -45.0)   # deep open ocean, far from the velocity-domain edge


def run_continuous(rep, out_zarr, n_floats, t_start, t_end, margin=SEED_MARGIN, seeds=None,
                   vel_dir=None, output_days=CYCLE_DAYS, sentinel=True, dt_hours=1):
    """Advect one realisation continuously over [t_start, t_end] with RK4 steps of dt_hours.

    `seeds`, if given, is a (lat, lon, release_datetime64, lifespan_days) tuple used
    verbatim (for the real-Argo deployment arms); otherwise the uniform space-time
    deployment_plan is drawn. `vel_dir` defaults to di.SIM_VEL_DIR; a directory with a
    depth dimension (the original two-level files) runs a 3D field instead.

    `sentinel` adds one immortal particle, pinned in place and flagged sentinel=1, released with
    the earliest seed (so the output-time grid, anchored at the first release, is unchanged). Parcels fast-forwards past time chunks with no active particle, which strands a
    float released inside such a gap outside the loaded field window (TimeExtrapolationError);
    one always-active particle prevents the skip. Readers drop it via the sentinel flag.
    """
    from parcels import (AdvectionRK4, FieldSet, JITParticle, ParticleSet,
                         StatusCode, Variable)

    class ArgoFloat(JITParticle):
        age = Variable("age", dtype=np.float32, initial=0.0)
        lifespan = Variable("lifespan", dtype=np.float32, initial=0.0)
        sentinel = Variable("sentinel", dtype=np.int32, initial=0, to_write="once")

    def Pin(particle, fieldset, time):  # noqa: N802 (parcels kernel)
        if particle.sentinel == 1:
            particle_dlon = 0  # noqa: F841
            particle_dlat = 0  # noqa: F841

    def DeleteOutOfBounds(particle, fieldset, time):  # noqa: N802 (parcels kernel)
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    def Mortality(particle, fieldset, time):  # noqa: N802 (parcels kernel)
        particle.age += particle.dt
        if particle.lifespan > 0 and particle.age > particle.lifespan:
            particle.delete()

    files = _window_files(t_start, t_end, vel_dir)
    with xr.open_dataset(files[0]) as ds:
        has_depth = "depth" in ds["uo"].dims
    dims = {"time": "time", "lat": "latitude", "lon": "longitude"}
    if has_depth:
        dims["depth"] = "depth"
    fieldset = FieldSet.from_netcdf(files, variables={"U": "uo", "V": "vo"}, dimensions=dims,
                                    allow_time_extrapolation=False)
    if seeds is None:
        lat, lon, rel, life = deployment_plan(n_floats, rep, t_start, t_end, margin=margin)
    else:
        lat, lon, rel, life = seeds
    rel, life = np.asarray(rel).astype("datetime64[s]"), np.asarray(life, float)
    flag = np.zeros(len(lat), np.int32)
    if sentinel:
        lat, lon = np.r_[lat, SENTINEL_LATLON[0]], np.r_[lon, SENTINEL_LATLON[1]]
        rel, life = np.r_[rel, rel.min()], np.r_[life, 0.0]   # keeps the output-time anchor
        flag = np.r_[flag, 1].astype(np.int32)
    pset = ParticleSet(fieldset=fieldset, pclass=ArgoFloat,
                       lon=lon, lat=lat, depth=np.full(len(lat), PARK_DEPTH) if has_depth else None,
                       time=rel, lifespan=life * 86400.0, sentinel=flag)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pset.execute([AdvectionRK4, Pin, DeleteOutOfBounds, Mortality],
                     endtime=np.datetime64(t_end, "s"),
                     dt=np.timedelta64(int(dt_hours), "h"),
                     output_file=pset.ParticleFile(name=out_zarr,
                                                   outputdt=np.timedelta64(output_days, "D")),
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


def _not_sentinel(ds):
    """Boolean mask over trajectories excluding run_continuous's pinned sentinel."""
    if "sentinel" not in ds:
        return np.ones(ds.sizes["trajectory"], bool)
    f = np.asarray(ds["sentinel"].values)
    return (f[:, 0] if f.ndim == 2 else f) != 1


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
    keep = _not_sentinel(ds)
    lat, lon, time = ds["lat"].values[keep], ds["lon"].values[keep], ds["time"].values[keep]
    pid = np.asarray(ds["trajectory"].values)[keep]
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
    return _tag_profiles(df, boxes)


def _tag_profiles(df, boxes=BOXES):
    df = df.copy()
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


def tracks_from_zarr(path, rep):
    """Every output position (daily in the resample arm) as rep, float_id, time, lat, lon."""
    with xr.open_zarr(path) as ds:
        keep = _not_sentinel(ds)
        lat, lon, time = ds["lat"].values[keep], ds["lon"].values[keep], ds["time"].values[keep]
        pid = np.asarray(ds["trajectory"].values)[keep]
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(np.int32)
    df = pd.DataFrame({"rep": np.int16(rep), "float_id": np.repeat(pid, lat.shape[1]),
                       "time": time.ravel(), "lat": lat.ravel().astype(np.float32),
                       "lon": lon.ravel().astype(np.float32)}).dropna(subset=["lat", "lon", "time"])
    return df.sort_values(["float_id", "time"]).reset_index(drop=True)


def profiles_from_tracks(tracks, release, rep, cycle_days=CYCLE_DAYS, boxes=BOXES):
    """Nominal profiles at release + k*cycle_days (k >= 1), linearly interpolated
    between output positions, up to each float's last position. `release` is indexed
    by float_id. Same columns as profiles_from_zarr."""
    t = tracks["time"].to_numpy().astype("datetime64[s]").astype("int64")
    fid = tracks["float_id"].to_numpy()
    b = np.flatnonzero(np.r_[True, fid[1:] != fid[:-1], True])
    step = int(cycle_days) * 86400
    out = []
    for i, j in zip(b[:-1], b[1:]):
        k = fid[i]
        r0 = np.asarray(release[k], dtype="datetime64[s]").astype("int64")
        tk = r0 + step * np.arange(1, (t[j - 1] - r0) // step + 1)
        tk = tk[tk >= t[i]]
        if tk.size:
            out.append(pd.DataFrame({
                "float_id": k, "cycle": np.rint((tk - r0) / step).astype(int),
                "time": tk.astype("datetime64[s]"),
                "lat": np.interp(tk, t[i:j], tracks["lat"].to_numpy()[i:j]),
                "lon": np.interp(tk, t[i:j], tracks["lon"].to_numpy()[i:j])}))
    df = pd.concat(out, ignore_index=True).assign(rep=int(rep))
    return _tag_profiles(df, boxes)


def profiles_from_outputs(tracks, release, rep, cycle_days=CYCLE_DAYS, boxes=BOXES):
    """Every output position taken as a profile, for runs written every `cycle_days`.
    Parcels writes on a global output grid, so profiles fall 0-cycle_days after release
    instead of exactly at release + k*cycle_days; with jittered releases the cycle phase
    is uniform either way, and the profile rate per float is the same."""
    r0 = np.asarray(release, dtype="datetime64[s]")[tracks["float_id"].to_numpy()]
    age = (tracks["time"].to_numpy().astype("datetime64[s]") - r0).astype(float)
    df = tracks.assign(cycle=np.maximum(np.rint(age / (cycle_days * 86400)), 1).astype(int),
                       rep=int(rep))
    return _tag_profiles(df[age > 0], boxes)


def tracks_path(tag):
    return os.path.join(cfg.OUT_DIR, f"design_tracks_continuous{tag}{PROFILE_EXT}")


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
    scratch = scratch_dir()

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
        try:
            run_continuous(d, zpath, n_floats, t_start, t_end)
            all_prof.append(profiles_from_zarr(zpath, d))
            if keep_zarr:
                keep_traj(zpath)
        finally:
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
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = scratch_dir()
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
    try:
        run_continuous(rep, zpath, n_floats, t_start, t_end, margin)
        with xr.open_zarr(zpath) as ds:
            keep = _not_sentinel(ds)
            lat, lon, time = ds["lat"].values[keep], ds["lon"].values[keep], ds["time"].values[keep]
            pid = np.asarray(ds["trajectory"].values)[keep]
    finally:
        shutil.rmtree(zpath, ignore_errors=True)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
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
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = scratch_dir()
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
    try:
        run_continuous(rep, zpath, len(pool), t_start, t_end, seeds=seeds)
        with xr.open_zarr(zpath) as ds:
            keep = _not_sentinel(ds)
            lat, lon, time = ds["lat"].values[keep], ds["lon"].values[keep], ds["time"].values[keep]
            pid = np.asarray(ds["trajectory"].values)[keep]
    finally:
        shutil.rmtree(zpath, ignore_errors=True)
    pid = (pid[:, 0] if pid.ndim == 2 else pid).astype(int)
    n_traj, n_obs = lat.shape
    df = pd.DataFrame({"seed_id": np.repeat(pid, n_obs),
                       "cycle": np.tile(np.arange(n_obs), n_traj),
                       "lat": lat.ravel(), "lon": lon.ravel(), "time": time.ravel()}
                      ).dropna(subset=["lat", "lon", "time"])
    df = df.sort_values(["seed_id", "time"]).reset_index(drop=True)
    out_csv = out_csv or os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv")
    df.to_csv(out_csv, index=False)
    print(f"{len(df):,} positions, {df['seed_id'].nunique()} floats -> {out_csv}")
    return df


def deep_mask(lat, lon, min_depth=PARK_DEPTH):
    """True where the native bathymetry cell is at least min_depth deep."""
    with xr.open_dataset(BATHY_PATH) as ds:
        dd = ds["deptho"]
        ii = cfg.containing_index(np.asarray(lat), dd["latitude"].values)
        jj = cfg.containing_index(np.asarray(lon), dd["longitude"].values)
        ok = (ii >= 0) & (jj >= 0)
        keep = np.zeros(len(ii), bool)
        v = dd.values[ii[ok], jj[ok]]
        keep[ok] = np.isfinite(v) & (v >= min_depth)
    return keep


def argo_deployment_pool(cleaned=None, boxes=BOXES, margin=SEED_MARGIN,
                         t_start=None, t_end=None, deep=True, domain=False):
    """Real Argo deployments (each cleaned float's first profile) in the seed region.

    Returns float_id, lat, lon, date (deploy datetime), lifespan (days = last-first
    cleaned profile). Restricted to first-profile inside (union box + margin) cap the
    velocity domain -- or the whole velocity domain if `domain` -- deploy date in
    [t_start, t_end), and (deep) deptho >= park depth. Selection uses only where and
    when a float was deployed, never where it went.
    """
    from osim import mean_mc as mm
    d = mm.clean_argo() if cleaned is None else cleaned
    g = d.sort_values("date").groupby("float_id")
    dep = g[["lat", "lon", "date"]].first()
    dep["lifespan"] = (g["date"].last() - dep["date"]).dt.total_seconds() / 86400.0

    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()
    if domain:
        la0, la1, lo0, lo1 = vla0, vla1, vlo0, vlo1
    else:
        la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes) - margin
        la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes) + margin
        lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes) - margin
        lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes) + margin
    dep = dep[dep["lat"].between(max(la0, vla0), min(la1, vla1))
              & dep["lon"].between(max(lo0, vlo0), min(lo1, vlo1))]
    if t_start is not None:
        dep = dep[dep["date"] >= pd.Timestamp(t_start, tz="UTC")]
    if t_end is not None:
        dep = dep[dep["date"] < pd.Timestamp(t_end, tz="UTC")]
    if deep:
        dep = dep[deep_mask(dep["lat"].to_numpy(), dep["lon"].to_numpy())]
    return dep.reset_index()


def argo_initial_state(cleaned=None, t0="2010-01-01", deep=True):
    """Cleaned floats alive at t0 (first profile before, last on/after), placed at
    their first profile on/after t0 inside the velocity domain, with the lifespan
    remaining from that profile. Conditions the run on the array state at t0."""
    from osim import mean_mc as mm
    d = (mm.clean_argo() if cleaned is None else cleaned).sort_values("date")
    t0 = pd.Timestamp(t0, tz="UTC")
    g = d.groupby("float_id")["date"]
    first, last = g.transform("min"), g.transform("max")
    after = d[(first < t0) & (last >= t0) & (d["date"] >= t0)]
    s = after.groupby("float_id").first()[["lat", "lon", "date"]]
    s["lifespan"] = (last.groupby(d["float_id"]).max().loc[s.index] - s["date"]).dt.total_seconds() / 86400.0
    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()
    s = s[s["lat"].between(vla0, vla1) & s["lon"].between(vlo0, vlo1)]
    if deep:
        s = s[deep_mask(s["lat"].to_numpy(), s["lon"].to_numpy())]
    return s.reset_index()


RESAMPLE_POOL = os.path.join(cfg.OUT_DIR, "argo_resample_pool.csv")


def write_resample_pool(t_start="2010-01-01", t_end=None, path=RESAMPLE_POOL,
                        max_km0=50.0, max_km_per_day=20.0, include_init=False):
    """Cache the resample pool: every float in argo_ohc.csv launched inside the velocity
    domain (deep water) in [t_start, t_end) as source='deploy' -- and, if include_init,
    floats launched earlier and still profiling at t_start as source='init'. By default
    the early years are a ramp-up instead: no real in-box profile after 2014 comes from a
    float launched before 2010. No float-level QC.

    Launch position/time come from the GDAC metadata (argo_meta.py) when consistent
    with the first profile -- at most 1 d after it and within max_km0 + max_km_per_day
    km per elapsed day -- else from the first profile (launch_src records which).
    Lifespan is operating life: launch to last profile of any depth; eligibility rules
    (depth, duplicates) are applied after advection, never here.
    """
    from osim import argo_meta as am
    t_end = t_end or _velocity_end()
    d = am.read_argo()
    f = d.groupby("float_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                  first=("date", "first"), last=("date", "last"),
                                  n_prof=("date", "size"))
    f = f.join(am.read_launch().set_index("float_id"), how="left")
    days = (f["first"] - f["launch_date"]).dt.total_seconds() / 86400.0
    la1, la2 = np.deg2rad(f["launch_lat"]), np.deg2rad(f["lat"])
    km = 6371.0 * np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2)
                                    * np.cos(np.deg2rad(f["lon"] - f["launch_lon"])), -1, 1))
    ok = (days >= -1) & (km <= max_km0 + max_km_per_day * days.clip(lower=0))
    f["launch_src"] = np.where(ok, "meta", "first_profile")
    f["dep_lat"] = np.where(ok, f["launch_lat"], f["lat"])
    f["dep_lon"] = np.where(ok, f["launch_lon"], f["lon"])
    f["dep_date"] = f["launch_date"].where(ok, f["first"])

    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()
    t0, t1 = pd.Timestamp(t_start, tz="UTC"), pd.Timestamp(t_end, tz="UTC")
    dep = f[f["dep_lat"].between(vla0, vla1) & f["dep_lon"].between(vlo0, vlo1)
            & (f["dep_date"] >= t0) & (f["dep_date"] < t1)]
    dep = dep[deep_mask(dep["dep_lat"].to_numpy(), dep["dep_lon"].to_numpy())]
    dep = pd.DataFrame({"float_id": dep.index, "lat": dep["dep_lat"].to_numpy(),
                        "lon": dep["dep_lon"].to_numpy(), "date": dep["dep_date"].to_numpy(),
                        "lifespan": ((dep["last"] - dep["dep_date"]).dt.total_seconds() / 86400).to_numpy(),
                        "launch_src": dep["launch_src"].to_numpy(), "n_prof": dep["n_prof"].to_numpy(),
                        "source": "deploy"})

    early = f.index[(f["dep_date"] < t0) & (f["last"] >= t0)]
    s = d[d["float_id"].isin(early) & (d["date"] >= t0)].groupby("float_id").first()
    s = s[s["lat"].between(vla0, vla1) & s["lon"].between(vlo0, vlo1)]
    s = s[deep_mask(s["lat"].to_numpy(), s["lon"].to_numpy())]
    ini = pd.DataFrame({"float_id": s.index, "lat": s["lat"].to_numpy(), "lon": s["lon"].to_numpy(),
                        "date": s["date"].to_numpy(),
                        "lifespan": ((f.loc[s.index, "last"] - s["date"]).dt.total_seconds() / 86400).to_numpy(),
                        "launch_src": "init_profile", "n_prof": f.loc[s.index, "n_prof"].to_numpy(),
                        "source": "init"})

    pool = pd.concat([dep, ini] if include_init else [dep], ignore_index=True)
    pool.insert(0, "pool_idx", np.arange(len(pool)))
    pool["year"] = pd.to_datetime(pool["date"], utc=True).dt.year
    pool.to_csv(path, index=False)
    print(f"{len(dep)} deployments {t_start}..{t_end} ({(dep['launch_src'] == 'meta').sum()} metadata "
          f"launches)" + (f" + {len(ini)} alive at {t_start}" if include_init else "") + f" -> {path}")
    print(dep.groupby(pd.to_datetime(dep["date"], utc=True).dt.year).size().to_string())
    return pool


def read_resample_pool(path=RESAMPLE_POOL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run continuous_sim.py --argo-pool first")
    pool = pd.read_csv(path)
    pool["date"] = pd.to_datetime(pool["date"], format="ISO8601", utc=True)
    return pool


def resample_seeds(pool, rep, t_start, t_end, jitter_deg=1 / 12, jitter_days=10.0):
    """Smoothed bootstrap of the deployment pattern for one replicate.

    Deployments are drawn with replacement within each deployment year, N_y per year,
    so every replicate keeps the real yearly counts; initial-state floats are seeded
    once. Every seed is jittered uniformly by +/- jitter_deg of great-circle distance
    (longitude scaled by 1/cos lat) and +/- jitter_days; jittered positions off the
    deep water are redrawn. Returns the Parcels seed tuple and a table whose row
    order is the Parcels trajectory index.
    """
    r = np.random.default_rng([7000, int(rep)])
    dep, ini = pool[pool["source"] == "deploy"], pool[pool["source"] == "init"]
    idx = np.concatenate([r.choice(g.index.to_numpy(), size=len(g), replace=True)
                          for _, g in dep.groupby("year")] + [ini.index.to_numpy()])
    tab = pool.loc[idx, ["pool_idx", "float_id", "source", "lat", "lon", "date", "lifespan"]]
    tab = tab.rename(columns={"float_id": "argo_id"}).reset_index(drop=True)

    (vla0, vla1), (vlo0, vlo1) = di.velocity_extent()
    lat0, lon0 = tab["lat"].to_numpy(), tab["lon"].to_numpy()
    lat, lon = lat0.copy(), lon0.copy()
    todo = np.ones(len(tab), bool)
    for _ in range(50):
        k = int(todo.sum())
        if not k:
            break
        lat[todo] = np.clip(lat0[todo] + r.uniform(-jitter_deg, jitter_deg, k), vla0, vla1)
        lon[todo] = np.clip(lon0[todo] + r.uniform(-jitter_deg, jitter_deg, k)
                            / np.cos(np.deg2rad(lat0[todo])), vlo0, vlo1)
        todo[todo] = ~deep_mask(lat[todo], lon[todo])
    lat[todo], lon[todo] = lat0[todo], lon0[todo]

    t = tab["date"].dt.tz_localize(None).to_numpy().astype("datetime64[s]")
    dt = (r.uniform(-jitter_days, jitter_days, len(tab)) * 86400).astype("int64").astype("timedelta64[s]")
    rel = np.clip(t + dt, np.datetime64(t_start, "s"), np.datetime64(t_end, "s") - np.timedelta64(1, "D"))
    tab = tab.assign(float_id=np.arange(len(tab)), seed_lat=lat, seed_lon=lon, release=rel)
    return (lat, lon, rel, tab["lifespan"].to_numpy(float)), tab


def _velocity_end():
    yrs = sorted({int(os.path.basename(f).split("_")[-2])
                  for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
    yrs = [y for y in yrs
           if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
    return f"{yrs[-1]}-12-26"


def main_argo(mode="actual", reps=1, rep_offset=0, n=None, t_start="2010-01-01",
              t_end=None, score_years=None, tag="", keep_zarr=False,
              jitter_deg=1 / 12, jitter_days=10.0, output_days=1):
    """Real-Argo deployment arm: advect floats seeded from the real deployment record
    through GLORYS. mode='actual' uses the deployments verbatim (one realization, the
    advection-fidelity check); mode='boot' resamples them with replacement at the real
    rate (n = pool size unless given), one draw per rep (RNG keyed on rep, jittered)."""
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    scratch = scratch_dir()
    if t_end is None:
        yrs = sorted({int(os.path.basename(f).split("_")[-2])
                      for f in glob.glob(os.path.join(di.VEL_DIR, "velocity_daily_park_*.nc"))})
        yrs = [y for y in yrs
               if len(glob.glob(os.path.join(di.VEL_DIR, f"velocity_daily_park_{y}_*.nc"))) == 12]
        t_end = f"{yrs[-1]}-12-26"

    if mode == "resample":
        pool = read_resample_pool()
        return _main_resample(pool, reps, rep_offset, t_start, t_end, score_years, tag,
                              keep_zarr, jitter_deg, jitter_days, scratch, output_days)

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
        try:
            run_continuous(d, zpath, len(seeds[0]), t_start, t_end, seeds=seeds)
            all_prof.append(profiles_from_zarr(zpath, d))
            if keep_zarr:
                keep_traj(zpath)
        finally:
            shutil.rmtree(zpath, ignore_errors=True)

    prof = pd.concat(all_prof, ignore_index=True)
    write_profiles(prof, sfx)
    est, ndropped = diagonal_estimates(prof, score_years)
    est.to_csv(os.path.join(cfg.OUT_DIR, f"estimates_continuous{sfx}.csv"), index=False)
    print(f"{len(prof):,} in-box profiles, {ndropped:,} dropped -> estimates_continuous{sfx}.csv")
    _print_summary(est)
    return est


def _main_resample(pool, reps, rep_offset, t_start, t_end, score_years, tag, keep_zarr,
                   jitter_deg, jitter_days, scratch, output_days=1):
    sfx = f"_argo_resample{tag}"
    ndep = int((pool["source"] == "deploy").sum())
    print(f"resample pool: {ndep} deployments + {len(pool) - ndep} initial-state floats, "
          f"{t_start}..{t_end}, jitter +/-{jitter_deg:.4f} deg, +/-{jitter_days:g} d")
    all_prof, all_tab = [], []
    for d in range(rep_offset, rep_offset + reps):
        seeds, tab = resample_seeds(pool, d, t_start, t_end, jitter_deg, jitter_days)
        all_tab.append(tab.assign(rep=d))
        zpath = os.path.join(scratch, f"traj{sfx}_r{d}.zarr")
        print(f"  rep {d}: advecting {len(tab)} floats ...", flush=True)
        try:
            run_continuous(d, zpath, len(tab), t_start, t_end, seeds=seeds, output_days=output_days)
            tracks = tracks_from_zarr(zpath, d)
            if keep_zarr:
                keep_traj(zpath)
        finally:
            shutil.rmtree(zpath, ignore_errors=True)
        if output_days == 1:
            tracks.to_parquet(tracks_path(f"{sfx}_r{d}" if reps > 1 else sfx), index=False,
                              compression="zstd")
            prof = profiles_from_tracks(tracks, tab["release"].to_numpy(), d)
        else:
            prof = profiles_from_outputs(tracks, tab["release"].to_numpy(), d, output_days)
        all_prof.append(prof.merge(tab[["float_id", "pool_idx", "argo_id", "source"]],
                                   on="float_id", how="left"))
    prof = pd.concat(all_prof, ignore_index=True)
    write_profiles(prof, sfx)
    pd.concat(all_tab).to_csv(os.path.join(cfg.OUT_DIR, f"seeds_continuous{sfx}.csv"), index=False)
    est, ndropped = diagonal_estimates(prof, score_years)
    est.to_csv(os.path.join(cfg.OUT_DIR, f"estimates_continuous{sfx}.csv"), index=False)
    print(f"{len(prof):,} profiles, {ndropped:,} dropped -> estimates_continuous{sfx}.csv")
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
    ap.add_argument("--argo", choices=("actual", "boot", "resample"), default=None,
                    help="real-Argo deployment arm: 'actual' (verbatim), 'boot' (old resampling), "
                         "'resample' (year-stratified smoothed bootstrap of the domain-wide pool)")
    ap.add_argument("--argo-pool", action="store_true",
                    help="write the cached resample pool (deployments + initial state) and exit")
    ap.add_argument("--jitter-deg", type=float, default=1 / 12)
    ap.add_argument("--jitter-days", type=float, default=10.0)
    ap.add_argument("--output-days", type=int, default=1,
                    help="resample arm: 1 = daily tracks + interpolated profiles; 10 = outputs are the profiles")
    ap.add_argument("--argo-traj", action="store_true",
                    help="advect real deployments verbatim keeping full drift paths "
                         "-> actual_full_tracks.csv (trajectory-divergence attribution)")
    ap.add_argument("--vel-dir", default=None,
                    help=f"velocity directory (default {os.path.relpath(di.SIM_VEL_DIR, cfg.REPO_ROOT)})")
    a = ap.parse_args()
    if a.vel_dir:
        di.SIM_VEL_DIR = os.path.abspath(a.vel_dir)
    sy = [int(y) for y in a.score_years.split(",")] if a.score_years else None
    if a.argo_pool:
        write_resample_pool(a.start or "2010-01-01", a.end)
    elif a.gather:
        gather_chunks(a.tag)
    elif a.argo_traj:
        run_actual_trajectories(t_start=a.start or "2010-01-01", t_end=a.end)
    elif a.illustrate:
        run_illustration(n_floats=a.n_floats, t_start=a.start, t_end=a.end)
    elif a.argo:
        main_argo(a.argo, a.reps, a.rep_offset, None,   # boot uses the real deployment rate
                  a.start or "2010-01-01", a.end, sy, a.tag, a.keep_zarr,
                  a.jitter_deg, a.jitter_days, a.output_days)
    else:
        main(a.n_floats, a.reps, a.rep_offset, a.start, a.end, sy, a.keep_zarr, a.tag)
