# Argo trajectory simulation (North Atlantic)

Simulate the drift of Argo floats through a CMEMS GLORYS velocity field using
[VirtualFleet](https://github.com/euroargodev/VirtualFleet) /
[Parcels](https://docs.parcels-code.org). Designed to scale from a quick local
run on a laptop to 100–1000 floats over 1–10 years on an HPC, parallelised
across cores by splitting the fleet into independent batches.

## Files

| File | Purpose |
|------|---------|
| `trajsim.py` | Compute library: deploy floats, fetch/build the velocity field, run a simulation. No plotting or notebook side effects. |
| `run_trajsim.py` | CLI batch runner. Runs one float batch, or all batches locally across cores. Auto-detects SLURM array env. |
| `run_experiment.sh` | **Main entry point.** Reads a config and runs it locally or submits it to SLURM. |
| `submit_trajsim.slurm` | SLURM array job body (invoked by `run_experiment.sh`, not directly). |
| `lib_args.sh` | Shared helper that turns config variables into CLI flags (used by both run paths). |
| `configs/*.conf` | One file per experiment. Copy `nac_default.conf` and edit. |
| `trajplots.py` | Plotting / trajectory map helpers (local analysis only; not needed on HPC). |
| `archive/` | Superseded scratch scripts. |

## Environment

A conda env with `parcels`, `virtualargofleet`, `xarray`, `global-land-mask`
(and `copernicusmarine` for downloading). On this machine that's the
`virtualfleet` env:

```bash
conda activate virtualfleet
```

If you don't activate it, point the config's `PYTHON` at that env's interpreter.

## Quickstart

```bash
# 1. Get a velocity field (one file per month — see below)
python -c "import trajsim; trajsim.fetch_velocity_months('2020-01','2020-12')"

# 2. Copy and edit a config
cp configs/nac_default.conf configs/my_run.conf

# 3. Run it (auto-detects local vs SLURM)
./run_experiment.sh configs/my_run.conf
```

## Configuring an experiment

Each experiment is a small bash file in `configs/`. Copy `nac_default.conf`,
edit, and you have a new, reproducible run. The knobs:

| Variable | Meaning |
|----------|---------|
| `EXPERIMENT` | Name; used for job name, output prefix, log files. |
| `PYTHON` | Python command (`python`, or a full path to the env's interpreter). |
| `VELOCITY` | Path or glob to the velocity NetCDF(s), e.g. `data/velocity/*.nc`. |
| `MODEL` | VirtualFleet model id (default `GLORYS12V1`). |
| `NFLOATS` | Number of floats. |
| `TOP_LEFT` / `BOTTOM_RIGHT` | Deployment box corners, `"lat,lon"`. |
| `DEPLOY_TIME` | Deployment date (all floats). |
| `YEARS` | Duration in years. Set empty `""` to use `DAYS` instead. |
| `DAYS` | Duration in days (used when `YEARS=""`). |
| `STEP_SECONDS` | Integration time step (default 300). |
| `RECORD_SECONDS` | Output recording period (multiple of step; default 3600). |
| `NTASKS` | Number of float batches = SLURM array size. |
| `LOCAL_CORES` | Processes for a local run. |
| `OUTDIR` / `PREFIX` | Output location; batch files are `<PREFIX>_taskNNN.zarr`. |
| `SLURM_TIME` / `SLURM_MEM` / `SLURM_CPUS` | Per-task SLURM resources. |
| `SLURM_CONDA_ENV` | Env activated inside the SLURM job. |

Paths in the config resolve relative to the **project root** (the parent of
`code/`), so `data/velocity/*.nc` and `data/virtualfleet` work as written.

## Downloading the velocity field

Use **one file per month** so each download stays manageable and parcels can
stream months lazily along time (memory bounded by the active chunks, not the
whole multi-year field):

```python
import trajsim
paths, glob = trajsim.fetch_velocity_months(
    "2015-01", "2024-12",            # inclusive YYYY-MM bounds
    out_dir="data/velocity/",
    lon_bounds=(-78, 17), lat_bounds=(18, 80), depth_bounds=(0, 2000),
)
# Point your config's VELOCITY at `glob`, i.e. data/velocity/velocity_*.nc
```

Existing files are skipped, so an interrupted download resumes cleanly. A decade
of 1/12° North Atlantic daily 0–2000 m is on the order of hundreds of GB on
disk — stage it where you have room before launching big runs.

## Running

```bash
./run_experiment.sh configs/my_run.conf           # auto: SLURM if available, else local
./run_experiment.sh configs/my_run.conf local     # force local (M-series / laptop)
./run_experiment.sh configs/my_run.conf slurm      # force SLURM array submission
```

- **Local**: runs all `NTASKS` batches across `LOCAL_CORES` processes.
- **SLURM**: submits an array of `NTASKS` tasks; each task runs one batch.
  Resources come from the `SLURM_*` config values. Logs land in `logs/`.

You can also drive `run_trajsim.py` directly (`python run_trajsim.py --help`) if
you'd rather pass flags than use a config.

## Output and merging

Each batch writes an independent `<PREFIX>_taskNNN.zarr` in `OUTDIR`. Combine
them for analysis by concatenating on the trajectory dimension:

```python
import trajplots
ds = trajplots.open_trajectories("data/virtualfleet/my_run_task*.zarr")
trajplots.map_trajectories("data/virtualfleet/my_run_task*.zarr", save_path="figures/")
```

## Why batches instead of MPI

Parcels advects all particles in a single-threaded loop, but floats are
independent, so we parallelise by splitting the fleet into batches — one process
per core locally, one array task per batch on SLURM. Each batch rebuilds the
same deterministic deployment plan and takes its own slice.

Parcels' built-in MPI mode is aimed at very large particle counts (~10⁵–10⁶),
where it clusters particles spatially across ranks; at 100–1000 floats its setup
(`mpi4py` + a system MPI) and overhead aren't worth it, and the batch approach
gives the same particle-parallelism more simply. The memory concern is handled
separately by lazy loading (`FieldSet.from_netcdf(deferred_load=True)`, which
VirtualFleet uses when you pass a path/glob): parcels holds only a few
full-domain time-snapshots at once, so multi-year fields stay within node RAM.
**Always pass a file path/glob to the field — not an opened `xarray.Dataset`,
which forces the whole field into memory.**
