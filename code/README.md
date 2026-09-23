# `code/` — OSSE pipeline

Everything here serves the sampling-bias OSSE described in `outlines/`. The only
live report is `reports/osse.qmd`, which imports `osse/mean_mc.py`.

The pipeline runs in four stages; each writes to `data/osse/` and is resumable.

## 1. Data

| File | Purpose |
|------|---------|
| `ohc.py` | OHC depth integration. `profile_ohc` (0–z trapezoid) and `profile_ohc_layers` (PCHIP, four layers summing exactly to 0–2000 m). |
| `osse/download_glorys.py` | `copernicusmarine.subset` wrappers: resumable, one JSON sidecar per file. |
| `osse/download_ohc_layers.py` | **Entry point.** Streams GLORYS thetao month by month over the buffer, collapses to the four OHC layers, deletes the raw file. ~300 GB of thetao becomes ~7 GB. |
| `osse/download_velocity_full.py`, `osse/run_download.py` | Park-depth velocity over the advection domain; `configs/osse_glorys.json` describes the regions. |

## 2. Field

| File | Purpose |
|------|---------|
| `osse/osse_config.py` | Analysis boxes, the conditioning `buffer` region, the trend/harmonic basis, grid helpers. |
| `osse/build_residual_field.py` | **Entry point.** Per-cell mean field (quadratic trend + annual/semiannual harmonics, fitted on the monthly record) and the residual field `Z = OHC - mhat`. The buffer field uses a real calendar (flat `time` axis); per-box fields use `(year, doy)`. |

## 3. Sampling

| File | Purpose |
|------|---------|
| `osse/continuous_sim.py` | **Entry point.** One continuous Parcels advection per realization at 1000 m park depth. Records *every* profile, in or out of a box, so the analysis region can be widened without re-advecting. Profiles are Parquet. |
| `osse/deployment.py` | Float-array constants and deployment helpers (park depth, cycle length, seed margin, `in_box`, velocity extent, the Argo lifespan pool). |
| `osse/sample_residuals.py` | Evaluates `Z` at profile positions (cell-day lookup; masked positions are dropped, never filled from a neighbour). |

## 4. Estimators

| File | Purpose |
|------|---------|
| `osse/mean_mc.py` | Simple-mean estimator, the bias tables and all report plotting. This is what `osse.qmd` imports. |
| `osse/krige_local.py`, `osse/krige_local_fit.py` | **Entry point.** Locally stationary (Kuusela) covariance MLE: one anisotropic exponential per window, years as iid replicates, exact block likelihood, analytic gradients. |
| `osse/krige.py`, `osse/krige_fit.R` | A single *global* stationary fit. Kept for the negative result — it returns ranges ~8x the empirical e-folds, which is the evidence for fitting locally. See `outlines/paper_outline.md` §4.6. |

## Running

`submit_*.slurm` are array jobs; each maps to one entry point above. Every stage
skips work whose output already exists, so a partial array is topped up by
resubmitting the missing indices.

```bash
sbatch --array=2010-2019 code/osse/submit_ohc_layers.slurm daily
sbatch code/osse/submit_residual_field.slurm buffer
sbatch --array=0-99%25 code/osse/submit_continuous_pilot.slurm 1000 2010-01-01 2015,2016,2017,2018,2019 "" _pilot_wide
sbatch --array=0-49    code/osse/submit_krige_local.slurm 50 oracle
```

R is invoked only as a subprocess, never imported, and always with
`cwd=REPO_ROOT` so `.Rprofile` activates the `renv` library.
