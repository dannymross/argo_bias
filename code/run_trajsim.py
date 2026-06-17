#!/usr/bin/env python
"""Batch runner for North Atlantic Argo trajectory simulations.

Parcels advects all particles in a single-threaded loop, but floats are
independent, so we parallelise by splitting the fleet into batches and running
one batch per process. The same script drives both back ends:

  * Local (M4 Pro):   one process per core via a process pool.
      python run_trajsim.py --nfloats 1000 --years 5 --local-cores 8

  * SLURM array:      one array task per batch (auto-detected from env).
      sbatch --array=0-31 submit_trajsim.slurm

Each batch rebuilds the same deterministic deployment plan, takes its own slice,
and writes an independent <prefix>_taskNN.zarr. Merge afterwards by opening the
batch outputs together (concat on the trajectory dimension).

Memory: the velocity field is built from a file path/glob so parcels loads it
lazily (deferred_load=True) — RAM stays bounded by the active time chunks, not
the field's total size, so multi-year fields are fine.
"""

# Pin math-library threads to 1 BEFORE importing numpy/parcels so that running
# N worker processes does not oversubscribe cores. Must happen before numpy.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import resource
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import trajsim


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _peak_memory_mb():
    """Peak RSS in MB. getrusage units differ: bytes on macOS, KB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


def _write_log(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Log written -> {path}")

# JSON config keys applied as run_trajsim defaults. Orchestration-only keys
# (experiment, python, local_cores, slurm) are consumed by run_experiment.sh,
# not here — notably local_cores is excluded so a config never silently forces
# a local pool run when a SLURM task imports it.
_CONFIG_KEYS = {
    "experiment", "velocity", "model", "nfloats", "deploy_time", "years",
    "days", "step_seconds", "record_seconds", "outdir", "prefix", "ntasks",
    "snap_deg",
}
_CONFIG_TUPLE_KEYS = {"top_left", "bottom_right"}


def load_config(path):
    """Load an experiment JSON file into a dict of argparse defaults."""
    with open(path) as f:
        raw = json.load(f)
    out = {k: raw[k] for k in _CONFIG_KEYS if raw.get(k) is not None}
    for k in _CONFIG_TUPLE_KEYS:
        if raw.get(k) is not None:
            out[k] = tuple(raw[k])
    return out


def parse_pair(s):
    """Parse 'lat,lon' into a (float, float) tuple."""
    a, b = s.split(",")
    return (float(a), float(b))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Config file. Values become defaults; any flag below overrides them.
    p.add_argument("--config",
                   help="Experiment JSON (see configs/nac_default.json). "
                        "Provides defaults for the options below.")

    # Velocity field
    p.add_argument("--velocity",
                   help="Path or glob to velocity NetCDF(s), e.g. "
                        "'data/velocity/*.nc'. Loaded lazily by parcels.")
    p.add_argument("--model", default="GLORYS12V1")

    # Deployment plan (Mode 1: bounding box + nfloats)
    p.add_argument("--nfloats", type=int, default=100)
    p.add_argument("--top-left", type=parse_pair, default=(42, -68),
                   help="NW corner 'lat,lon' (default 42,-68)")
    p.add_argument("--bottom-right", type=parse_pair, default=(30, -44),
                   help="SE corner 'lat,lon' (default 30,-44)")
    p.add_argument("--deploy-time", default="2020-01-01")
    p.add_argument("--snap-deg", type=float, default=None,
                   help="snap deployment positions to the nearest multiple of this "
                        "degree value (model cell centres; use 0.0833333 for GLORYS12)")

    # Simulation
    p.add_argument("--years", type=float, default=None,
                   help="Duration in years (overrides --days if set)")
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--step-seconds", type=int, default=300)
    p.add_argument("--record-seconds", type=int, default=3600)

    # Output
    p.add_argument("--outdir", default="data/virtualfleet/")
    p.add_argument("--prefix", default="traj")
    p.add_argument("--nc-save", action="store_true",
                   help="Also write a .nc copy of each batch")

    # Parallel layout
    p.add_argument("--ntasks", type=int, default=None,
                   help="Total number of batches. Auto-detected from SLURM if "
                        "unset; defaults to 1.")
    p.add_argument("--task-id", type=int, default=None,
                   help="0-based batch index to run. Auto-detected from SLURM "
                        "if unset.")
    p.add_argument("--local-cores", type=int, default=None,
                   help="Run ALL batches locally across this many processes "
                        "(ignores --task-id). Use on the M4, not on SLURM.")
    return p


def resolve_layout(args):
    """Resolve (ntasks, task_id) from CLI args or the SLURM environment."""
    ntasks = args.ntasks
    task_id = args.task_id

    if ntasks is None:
        ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task_id is None:
        task_id = 0
    return ntasks, task_id


def run_batch(args, ntasks, task_id):
    """Run one batch (task_id of ntasks), write a run log, return output path."""
    duration_days = args.years * 365.25 if args.years is not None else args.days

    plan = trajsim.deploy_float_grid(
        top_left=args.top_left,
        bottom_right=args.bottom_right,
        nfloats=args.nfloats,
        deploy_time=args.deploy_time,
        snap_deg=args.snap_deg,
    )
    nfloats_total = len(plan["lat"])

    start, count = trajsim.split_indices(nfloats_total, ntasks)[task_id]
    if count == 0:
        print(f"[task {task_id}] empty batch, nothing to do")
        return None
    batch = trajsim.select_floats(plan, start, count)
    print(f"[task {task_id}/{ntasks}] floats {start}..{start + count - 1} "
          f"({count} floats), {duration_days:.1f} days")

    field = trajsim.build_velocity_field(args.velocity, model=args.model)
    out_file = f"{args.prefix}_task{task_id:03d}.zarr"

    t0 = time.perf_counter()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    output = trajsim.run_simulation(
        plan=batch,
        velocity_field=field,
        duration_days=duration_days,
        step_seconds=args.step_seconds,
        record_seconds=args.record_seconds,
        out_dir=args.outdir,
        out_file=out_file,
        nc_save=args.nc_save,
    )

    wall_seconds = time.perf_counter() - t0
    peak_mb = _peak_memory_mb()

    log = {
        "experiment": getattr(args, "experiment", args.prefix),
        "config": getattr(args, "config", None),
        "started_at": started_at,
        "wall_seconds": round(wall_seconds, 2),
        "peak_memory_mb": round(peak_mb, 1),
        # --- parallel layout ---
        "task_id": task_id,
        "ntasks": ntasks,
        "cores_available": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        # --- simulation parameters ---
        "nfloats_total": nfloats_total,
        "nfloats_batch": count,
        "float_index_start": start,
        "float_index_end": start + count - 1,
        "deploy_time": args.deploy_time,
        "top_left": list(args.top_left),
        "bottom_right": list(args.bottom_right),
        "duration_days": duration_days,
        "step_seconds": args.step_seconds,
        "record_seconds": args.record_seconds,
        "velocity": args.velocity,
        "model": args.model,
        # --- output ---
        "output": output,
        # --- environment ---
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "parcels": _pkg_version("parcels"),
            "virtualargofleet": _pkg_version("virtualargofleet"),
            "xarray": _pkg_version("xarray"),
            "numpy": _pkg_version("numpy"),
        },
    }

    log_path = os.path.join(args.outdir, f"{args.prefix}_task{task_id:03d}.log.json")
    _write_log(log_path, log)
    return output


def _worker(args, ntasks, task_id):
    # Top-level (picklable) callable for the local process pool.
    return run_batch(args, ntasks, task_id)


def main(argv=None):
    parser = build_parser()
    # First pass: find --config, fold its values in as defaults so explicit
    # command-line flags still win. Second pass: the real parse.
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        parser.set_defaults(**load_config(pre.config))
    args = parser.parse_args(argv)

    if not args.velocity:
        sys.exit("--velocity is required (set it via --config or the flag)")

    # Carry the experiment name through to run_batch for the log, without
    # exposing it as a CLI flag (it's a config-only concept).
    if not hasattr(args, "experiment"):
        args.experiment = args.prefix

    if args.local_cores:
        # Run every batch locally; one process per batch, capped at local-cores.
        ntasks = args.ntasks or args.local_cores
        print(f"Local run: {ntasks} batches over {args.local_cores} processes")
        outputs = []
        with ProcessPoolExecutor(max_workers=args.local_cores) as ex:
            futures = {ex.submit(_worker, args, ntasks, t): t for t in range(ntasks)}
            for fut in as_completed(futures):
                t = futures[fut]
                out = fut.result()
                print(f"[task {t}] done -> {out}")
                outputs.append(out)
        print("All batches complete:")
        for o in sorted(filter(None, outputs)):
            print(f"  {o}")
        return

    ntasks, task_id = resolve_layout(args)
    if not (0 <= task_id < ntasks):
        sys.exit(f"task-id {task_id} out of range for ntasks {ntasks}")
    out = run_batch(args, ntasks, task_id)
    print(f"[task {task_id}] done -> {out}")


if __name__ == "__main__":
    main()
