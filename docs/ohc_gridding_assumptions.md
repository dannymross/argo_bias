# Assumptions in the OHC 1-degree gridding (`code/ohc.py`)

This note documents the assumptions built into the functions that bin ocean heat
content (OHC) onto 1-degree cells for the sampling-bias study:
`_cell_centers`, `grid_cells`, and `coarsen_truth` in
[`code/ohc.py`](../code/ohc.py).

These functions take point or gridded OHC and reduce it to one value per
(month, 1-deg cell) for both the **truth** field (GLORYS, complete coverage) and
the **synthetic Argo** estimate (floats). The bias is then a cell-by-cell
difference between the two.

---

## 1. Cell definition (`_cell_centers`)

```python
np.floor(values / deg) * deg + deg / 2.0
```

- **Floor binning anchored to integer degrees.** A 1-deg cell's edges sit on
  whole degrees and its label is the half-degree centre — e.g. every point with
  `lat in [40, 41)` falls in cell `40.5`. This matches the standard 1x1 deg box
  convention used by the EN4 gridded product, so cells line up with the
  real-world reference data (`data/ohc_en4_gridded.rds`, `data/argo_ohc.csv`).
- **Degrees are the binning unit, not distance.** A 1-deg cell is ~111 km tall
  but only ~85 km wide at 40N (cos 40). Cells are **not equal-area**, and the
  cell mean is an **unweighted average** of the points inside the lon/lat box —
  the cos(lat) area shrinkage with latitude is ignored. Over the 33-48N study
  box this is a real (if modest) latitudinal area gradient. This is the
  conventional Argo-gridding choice, but it is an approximation.

## 2. Truth <-> float consistency (the key design assumption)

`coarsen_truth` deliberately flattens the gridded truth field to points and runs
it through the **same** `grid_cells` used for floats. Therefore:

- Truth and synthetic-float OHC land on **exactly the same cells**, and the bias
  is a clean cell-by-cell difference rather than a regridding artifact.
- Any approximation in the cell definition (e.g. no area weighting) applies
  **identically to both sides**, so it does not bias the truth-vs-float
  comparison itself.

## 3. Truth coarsening (`coarsen_truth`)

```python
monthly = truth_ds.resample(time="1MS").mean()
... dropna(subset=value_cols, how="all")
... grid_cells(df, value_cols, deg=deg)
```

- **Monthly mean is a simple equal-weight time average** of the daily fields
  (`resample("1MS").mean()`). Every day counts equally; a partial month at the
  start/end of the record would still be labelled and averaged as a month.
- **Unweighted spatial mean over native cells inside each 1-deg box.** GLORYS is
  on a regular ~1/12 deg grid, so all ~144 native cells in a 1-deg box receive
  equal weight — again no cos(lat) area weighting *within* the box.
- **`dropna(how="all")` drops only all-NaN rows.** A native point with a valid
  `ohc_700` but NaN `ohc_2000` (seafloor between 700 and 2000 m) is **kept**, so
  a cell's `ohc_700` and `ohc_2000` means can be computed over **different sets
  of native points**. A partially-deep coastal cell's `ohc_2000` therefore
  reflects only its deep-water fraction.

## 4. Float gridding (`grid_cells`)

```python
df.groupby(["month", "cell_lat", "cell_lon"]).agg(mean ... , size)
```

- **Each profile counts once, equally.** The cell mean is an unweighted average
  over whatever profiles fall in the box that month. A cell visited 6 times by
  one float and a cell visited once by each of 6 floats are treated identically.
  The reported `n` is the **raw profile count**, not the number of distinct
  floats.
- **A profile's single (lat, lon) assigns it wholly to one cell** — no spatial
  spreading or objective mapping. This mimics the simplest "bin-and-average"
  estimator (step 4 of the study methodology), not optimal interpolation.
- **Month assignment is by calendar month** of the profile date, with no
  smoothing across month boundaries.

---

## Implications for the study

The two assumptions most worth keeping in mind as the pilot scales up:

1. **No cos(lat) area weighting** in the within-cell mean or in any domain mean
   derived from these cells. Over a 15-deg latitude span this introduces a small
   systematic tilt in *absolute* domain means. (It does not bias the
   truth-vs-float comparison, since both use the same procedure.) Adding optional
   area weighting is straightforward if true area-average domain means are
   wanted.
2. **Per-depth NaN handling differs across cells**, so `ohc_2000` coverage is
   sparser than `ohc_700` near the shelf. This is correct physics, but it means
   the two depth layers are not sampled on identical cell sets.

Both are intentional simplifications that match the EN4/Argo gridding convention
for the pilot. Because truth and floats are gridded by the **identical**
procedure, neither assumption biases the sampling-bias estimate that is the
target of the study.
