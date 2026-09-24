# `code/` — float-simulation pipeline

Everything here serves the sampling-bias simulation described in `outlines/`. The only
live report is `reports/osim.qmd`, which imports `osim/mean_mc.py`.

The pipeline runs in four stages; each writes to `data/osim/` and is resumable.

## 1. Data

| File | Purpose |
|------|---------|
| `ohc.py` | OHC depth integration. `profile_ohc` (0–z trapezoid) and `profile_ohc_layers` (PCHIP, four layers summing exactly to 0–2000 m). |
| `osim/download_glorys.py` | `copernicusmarine.subset` wrappers: resumable, one JSON sidecar per file. |
| `osim/download_ohc_layers.py` | **Entry point.** Streams GLORYS thetao month by month over the buffer, collapses to the four OHC layers, deletes the raw file. ~300 GB of thetao becomes ~7 GB. |
| `osim/download_velocity_full.py`, `osim/run_download.py` | Park-depth velocity over the advection domain; `configs/glorys.json` describes the regions. |

## 2. Field

| File | Purpose |
|------|---------|
| `osim/osim_config.py` | Analysis boxes, the conditioning `buffer` region, the trend/harmonic basis, grid helpers. |
| `osim/build_residual_field.py` | **Entry point.** Per-cell mean field (quadratic trend + annual/semiannual harmonics, fitted on the monthly record) and the residual field `Z = OHC - mhat`. The buffer field uses a real calendar (flat `time` axis); per-box fields use `(year, doy)`. |

## 3. Sampling

| File | Purpose |
|------|---------|
| `osim/continuous_sim.py` | **Entry point.** One continuous Parcels advection per realization at 1000 m park depth. Records *every* profile, in or out of a box, so the analysis region can be widened without re-advecting. Profiles are Parquet. |
| `osim/deployment.py` | Float-array constants and deployment helpers (park depth, cycle length, seed margin, `in_box`, velocity extent, the Argo lifespan pool). |
| `osim/sample_residuals.py` | Evaluates `Z` at profile positions (cell-day lookup; masked positions are dropped, never filled from a neighbour). |

## 4. Estimators

| File | Purpose |
|------|---------|
| `osim/mean_mc.py` | Estimand, sampled estimates and diagnostics. Computation only. |
| `osim/report.py` | The summary tables and figures `osim.qmd` calls. Presentation only. |
| `osim/krige_local.py`, `osim/krige_local_fit.py` | **Entry point.** Locally stationary (Kuusela) covariance MLE: one anisotropic exponential per window, years as iid replicates, exact block likelihood, analytic gradients. |
| `osim/krige.py`, `osim/krige_fit.R` | A single *global* stationary fit. Kept for the negative result — it returns ranges ~8x the empirical e-folds, which is the evidence for fitting locally. See `outlines/paper_outline.md` §4.6. |

## Running

`submit_*.slurm` are array jobs; each maps to one entry point above. Every stage
skips work whose output already exists, so a partial array is topped up by
resubmitting the missing indices.

```bash
sbatch --array=2010-2019 code/osim/submit_ohc_layers.slurm daily
sbatch code/osim/submit_residual_field.slurm buffer
sbatch --array=0-99%25 code/osim/submit_continuous_pilot.slurm 1000 2010-01-01 2015,2016,2017,2018,2019 "" _pilot_wide
sbatch --array=0-49    code/osim/submit_krige_local.slurm 50 oracle
```

R is invoked only as a subprocess, never imported, and always with
`cwd=REPO_ROOT` so `.Rprofile` activates the `renv` library.
