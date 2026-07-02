# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo (Summer 2026) quantifying Argo float sampling bias in ocean heat content (OHC)
estimates. GLORYS12 reanalysis provides "truth" (complete spatiotemporal coverage); real and
VirtualFleet-simulated Argo floats sample it sparsely, and the study compares sparse-sampling
(and interpolated) OHC estimates against that truth. Results are published as a Quarto website
of analysis reports (`reports/*.qmd`) to GitHub Pages.

## Environments

Two separate environments, used for different parts of the pipeline:

- **`argo_bias`** (conda, `environment.yml`) — Python 3.11 + numpy/pandas/xarray/matplotlib/
  cartopy/parcels/copernicusmarine + `global_land_mask`, `VirtualFleet` (pip, from GitHub). This
  is the Jupyter kernel (`python3`) that all `reports/*.qmd` execute under, and what `code/*.py`
  (analysis modules) run in.
- **`virtualfleet`** (conda, HPC-only) — the default env trajectory-simulation SLURM jobs activate
  (see `code/run_experiment.sh` / `code/configs/*.json`'s `slurm.conda_env`). Used for
  `code/trajsim.py` / `code/run_trajsim.py` batch runs, not for reports.
- **R via `renv`** (`renv.lock`, `.Rprofile` auto-activates `renv/activate.R`) — `GpGp`,
  `data.table`, `fields`, `FNN` for the Gaussian-process interpolation scripts. On HPC (Roar
  Collab), set up/update with `./renv_setup.sh` (loads `module load r/4.5.0`, installs the R
  packages, then `renv::snapshot()`). R scripts are only ever invoked as subprocesses from Python
  (never imported/sourced directly) — see "Python ↔ R bridge" below.

`data/` is git-ignored and holds all raw/derived inputs (GLORYS velocity/temperature, RG
Argo Climatology, Argo profile CSVs, cached truth fields, GP intermediates — multi-GB to
multi-hundred-GB). It must exist locally/on HPC to execute report or analysis code; it does not
exist in CI (see below).

## Coding conventions

- **Concise, vectorized code.** Prefer numpy/xarray/pandas vectorized operations (Python) and
  `data.table`/vectorized ops (R) over explicit loops.
- **Build summary tables concisely on the first pass, not as later cleanup.** When a table needs a
  per-group breakdown plus an overall summary row (e.g. per-month stats + a trailing "Year"/"Total"
  row), compute both from the same `.agg([...])` call — grouped, then again on the ungrouped data —
  rather than hand-writing each statistic twice in a separate dict/row construction. See
  `make_gp_table` in `reports/nac_gp.qmd` for the pattern.
- **Write generalized functions.** New helpers (especially in `ohc.py`/`ohc_bias.py`/
  `ohc_climatology.py`) should flexibly handle argument variations (e.g. depth, resolution, value
  columns) rather than being one-off/single-callsite — see "cells table" note below.
- **Don't duplicate code within or across reports.** Reuse functions from `code/*.py`/`*.R` and
  cached datasets under `data/` (e.g. `data/ohc_truth/`, `data/gp_interp/`) instead of
  re-deriving the same computation in multiple reports or chunks.
- **Code comments and docstrings must be minimal — default to none.** Only add one when there's a
  genuinely non-obvious statistical/physical assumption, a gotcha, or a shape/contract a caller
  cannot get from the signature itself; when in doubt, leave it out. One short line/sentence is the
  norm; reach for two only when truly necessary, and never write multi-paragraph docstrings restating
  parameter names, spelling out call sites, or explaining what the code visibly does — the audience
  is experienced statisticians reading the code directly, not readers who need methodology narrated.
- **Code comments and code-chunk options (`fig-cap`, `code-summary`, etc.) are plain text only —
  no LaTeX and no Unicode math symbols** (e.g. write `+/-`, `sigma`, `deg` rather than `$\pm$`/
  `±`, `$\sigma$`/`σ`, `$^\circ$`/`°`). These contexts aren't reliably passed through Pandoc/
  MathJax rendering, so embedded LaTeX/Unicode can show up literally instead of rendering. LaTeX
  is fine (preferred) in actual report prose/markdown body text — see Reports below.
- **Git commit messages:** do not add a "Co-Authored-By: Claude" line.
- **Prefer standard-library/established-package implementations over custom ones** — e.g. use
  `scipy`/numpy/xarray (distance, interpolation, integration) or R's base/`stats`/established
  packages rather than hand-rolling algorithms those already provide correctly and efficiently.
- **For R data wrangling/processing, prioritize base R and `data.table`** over other frameworks
  (e.g. `dplyr`/tidyverse) to stay consistent with the R code already in `code/*.R`.
- **For R graphics, prioritize base R plotting** over `ggplot2`, unless `ggplot2` (or another
  package) gives a materially better result for the specific case — e.g. maps or other complex/
  layered graphics.

## Rendering reports (the main dev loop)

**Do not run `quarto render`/`quarto preview` and do not visually inspect rendered plots/reports
unless explicitly directed to** — rendering is expensive (multi-GB `data/` access, GP fits) and
the user typically reviews output themselves.

```bash
quarto render reports/nac_gp.qmd   # one report (run from repo root or reports/)
quarto render                      # whole site (per _quarto.yml's render list)
quarto preview                     # live-reloading dev server
```

There is no separate lint/test suite — "does it render, and does the output look right" is the
correctness check for report changes, when rendering is requested. `execute: freeze: auto` is set project-wide
(`_quarto.yml`), and Quarto's freeze cache lives in `_freeze/` (tracked in git, unlike the
`reports/_freeze/` path in `.gitignore` — that's a stale entry from before the freeze dir moved
to project root).

**CI has no access to `data/`.** `.github/workflows/publish.yml` runs `quarto render`/publish on
every push to `main` touching `reports/**`, using only the committed `_freeze/` execution cache —
it never re-executes a report's Python from scratch. `.gitignore` normally excludes `*.png`, but
carves out `!_freeze/**/figure-html/*.png` specifically so the frozen figures survive. **Practical
consequence: after editing a `.qmd` (or a `code/*.py`/`*.R` module a report depends on), always
`quarto render` it locally first and commit the resulting `_freeze/` + figure diffs alongside the
source change** — otherwise CI publishes stale output, or the freeze cache silently drifts from
the source.

`reports/pilot_simulation.qmd` is currently commented out of `_quarto.yml`'s render list (to keep
site preview fast) — re-enable that line if you need to render/publish it.

**Organize reports for efficient, iterable rendering:** split expensive one-time computation into
its own early chunk (cached to `data/` where reusable across reports — see `nac.qmd` publishing to
`data/ohc_truth/`), keep plotting/formatting chunks cheap and separate so they re-run fast on
their own, and lean on `execute: freeze: auto` rather than recomputing unchanged results.

## Python ↔ R bridge (GP interpolation)

The Gaussian-process interpolation (fit with `GpGp`, a Vecchia-approximated spatiotemporal
Matérn kriging model) is implemented in R for `GpGp`'s speed, but orchestrated from Python. The
pattern, in `code/ohc_climatology.py`:

1. Python writes profile observations and a prediction grid to CSV (`write_profile_csv`,
   `build_pred_grid`).
2. Python calls `Rscript code/ohc_gp_fit.R <profiles.csv> <fit_summary.csv> <model_cache.rds>`
   (via `run_gp_fit`, `subprocess.run(..., cwd=REPO_ROOT)`) — fits once, cached to `.rds`.
3. Python calls `Rscript code/ohc_gp_predict.R <profiles.csv> <pred_grid.csv> <model_cache.rds>
   <out.csv> [m|exact] [first|middle|last]` (via `run_gp_predict`) — reuses the cached fit to
   predict mean + standard error on a grid, one call per resolution. `m=exact` conditions on every
   observation (fine at ~35 points); a small integer `m` (e.g. 30) switches to a bounded-Vecchia
   neighbourhood so it scales to the ~3,600-point native 1/12° grid.
4. Python reads `out.csv` back (`load_gp_anomaly_field`) into an xarray field for plotting.

`gp_audit_fields.R` (via `run_gp_audit_fields`) independently cross-checks GpGp's kriging against
`fields::Krig` and a hand-rolled Cholesky solve. `ohc_levitus_interp.R` (via `run_levitus_interp`)
is a simpler, non-GP interpolation baseline (fixed-radius Gaussian weighting, Levitus-style).

**`cwd=REPO_ROOT` on every subprocess call is load-bearing**: R only auto-sources `.Rprofile`
(which activates the `renv` project library) from the exact working directory, not an ancestor —
and `reports/` has no `.Rprofile` of its own. File paths passed to the subprocess are made
absolute first since cwd is being redirected.

## Core Python modules (`code/`)

- **`ohc.py`** — OHC = `trapz(z, T * cp * rho)` integrated 0–700 m and 0–2000 m (J/m², advisor's
  seawater constants). `profile_ohc` does the core depth integration (vectorized over any leading
  shape); `grid_cells`/`coarsen_truth` floor-bin points/gridded truth onto 1°×1° cells
  (`cell = floor(x/deg)*deg + deg/2`). Truth and float estimates are **always coarsened through
  the same `grid_cells` path** so cell-definition quirks (no cos(lat) area weighting, unweighted
  means, per-depth NaN handling) apply identically to both sides and don't bias the truth-vs-float
  comparison — see `docs/ohc_gridding_assumptions.md` for the full list of binning assumptions
  before changing gridding logic.
- **`ohc_bias.py`** — bias metrics (`compute_bias`: float vs. truth, decomposed into sampling bias
  + grid bias) and nearly all plotting (`plot_monthly_cell_maps`, `plot_bias_se_violin`,
  `compute_coverage_cells`/`monthly_coverage_rate`/`plot_coverage_by_month`, region/resolution
  sweep plots). The common interchange format across this module is a **"cells table"**:
  columns `month, cell_lat, cell_lon, <value_cols...>` — new metrics should conform to this shape
  to reuse the existing plotting functions. Values are J/m² internally; `J_TO_GJ = 1e-9` converts
  for display.
- **`ohc_climatology.py`** — builds the RG Argo Climatology mean/anomaly field (Option A: seasonal
  12-month climatology from `ARGO_TEMPERATURE_MEAN` + `ARGO_TEMPERATURE_ANOMALY`, recommended;
  Option B: static 15-yr mean only — see the module docstring for why A is preferred), samples it
  onto arbitrary points/grids, and owns the R-subprocess orchestration described above.
- **`trajsim.py`** — VirtualFleet/Parcels compute library: float deployment plans, GLORYS velocity
  field fetch/build, running a simulation batch. No plotting/notebook side effects, so it stays
  lightweight on HPC compute nodes. `run_trajsim.py` is the CLI batch runner (auto-detects SLURM
  array env); `run_experiment.sh <config.json> [local|slurm]` is the actual entry point most runs
  should go through — it reads one JSON config (`code/configs/*.json`; see
  `code/configs/nac_gs_1yr_wide.json` for the shape: experiment name, python interpreter,
  velocity glob, deployment box/time, duration, batching, SLURM resources) and dispatches to
  local multi-core or a SLURM array job. **Note:** `code/README.md` documents an older
  `.conf`-file config format — the current format is JSON, read `run_experiment.sh` itself as the
  source of truth if the two disagree.
- **`trajplots.py`** — trajectory plotting only (matplotlib/cartopy); deliberately kept out of
  `trajsim.py` so simulation runs don't need those deps on HPC compute nodes.
- **`download_rg_monthly.py`**, **`download_gs_2021_2022.py`** — one-off/resumable data pulls
  (RG Argo Climatology; GLORYS12V1 velocity+temperature via `copernicusmarine`), run directly with
  `python code/download_*.py` from the repo root.

## Reports (`reports/`)

- **`nac.qmd`** — North Atlantic Current analysis region setup + OHC anomaly derivation against
  the RG climatology. Publishes the multi-year, analysis-domain-restricted truth field and
  truth-vs-climatology anomaly cells to `data/ohc_truth/` for other reports to reuse directly
  (skips re-running the expensive depth integration / RG reprocessing).
- **`nac_gp.qmd`** — picks up `nac.qmd`'s cached truth field. Fits one pooled spatiotemporal GP
  per depth (`matern_spheretime` covariance, all three years' real Argo profiles) and predicts OHC
  directly (not an anomaly) on both a coarse 1° grid (exact conditioning) and the native 1/12°
  grid (bounded Vecchia, `m=30`); compares predictions against truth via bias maps, SE maps, and
  ±1 SE coverage-rate diagnostics.
- **`pilot_simulation.qmd`** — earlier virtual-float pilot study; currently excluded from the
  site build (see above).

Reports share a small `_emit`/`_show_fig` helper pattern (defined in each report's first cell) for
building Quarto panel-tabsets (`::: {.panel-tabset}`) programmatically from Python loops over
depth × resolution × plot-kind — follow that pattern rather than hand-writing tabset markdown when
adding new tabbed figures.

**Report prose:** keep it concise — state the method/result, don't over-explain methodology the
reader (an experienced statistician) already knows. Use LaTeX (`$...$`) for math/notation rather
than Unicode symbols (e.g. `$\sigma^2$` not `σ²`, `$\pm$` not `±`).
