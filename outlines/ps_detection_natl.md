# Detecting and removing preferential sampling in real North Atlantic Argo

Status: analysis complete through correction (Oct 2026); not yet in a report. Reproduce with
`code/tm/detect.py` (see bottom). All numbers are GJ/m² (0-2000 m OHC) unless noted.

## Question and identifying idea

Within a local neighbourhood $j$ (box × period), a newly deployed float samples the residual field
non-preferentially (A5: deployment exogenous to the anomaly); floats advected in later may sample preferentially.
Estimand per neighbourhood
$$\delta_j = E[r \mid \text{advected float in } j] - E[r \mid \text{new float in } j],$$
$r$ = own-float-out kriging residual $z - \hat\mu_{-f} - \hat a_{-f}$. For a linear estimator the older-float bias
is $B \approx \sum_j W_j^{\text{old}}\delta_j$, so heterogeneous $\delta_j$ can cancel in a pooled test but not in
$B$. The new-float mean residual itself estimates the map's bias at deployment sites (map-estimator bias); the
advected - new contrast estimates the sorting effect (raw-average bias).

Earlier framing (log-age slope, `validate.trend`) imposed growth of bias with age; superseded by the step /
neighbourhood contrast above (user's model: zero at deployment, appears once advected, region/time dependent).

## Data

- 185,202 deep profiles (1,834 floats), 0-66N, 100W-0, 2004-2025; residual maps: baseline local RG mean + local
  KS kriging (`data/osim/glorys_dr_real_natl/`), and with local SLA in the mean field
  (`data/osim/glorys_dr_real_natl_sla/`).
- DUACS L4 altimeter-only SLA, 10-day bins at 0.125 deg (`data/sat/ssh_natl_10d/`, `code/sat/ssh.py`);
  CNES-CLS MDT (`data/sat/mdt_natl.nc`). Local SLA anomaly = SLA minus its 2 deg boxcar mean (cm).

## Tests and results (baseline map)

| Test | Result |
|---|---|
| New-float anchor (age < 30 d), 4 deg strata | -0.035 +/- 0.065; age bands flat (+/- 0.08) |
| Anchor variants (< 10/15/30 d, first profile only) | -0.08 ... +0.03, all +/- 0.08 |
| Within-float age slope (float FE) | +0.03 +/- 0.08 at 2 y |
| Cell contrasts advected - new, 2/4/8 deg x 3 y, co-deployment units, within-cell permutation | pooled ~0 +/- 0.03; Q excess with tau 0.19/0.29/0.41 growing with cell size (map mismatch, not sorting) |
| Local - new (same-box launches) | ~0, no heterogeneity |
| Deployment neighbourhoods (R 50-400 km, T 30/90 d) | null, SE 0.10-0.19; neighbour LOFO residuals correlate ~ -0.2 |
| Case-control intensity: case vs 10 area-uniform controls in its 4 deg box, same 10-d bin; conditional logit; +/-365 d placebos | pooled local-SLA preference -0.03 +/- 0.03 cm (~ -0.01 GJ/m²); EKE and SLA² preference = geography (placebos equal) |
| Mark-location (r on log local density of other floats, box x period FE) | null at 50-200 km |
| **Pre-specified**: placebo-corrected local-SLA preference on log10 box EKE, older floats | **-0.22 cm/decade, z -2.5 ... -3.2** for clusters float, 2x1, 4x1, 8x3 (deg x yr); 2004-14 -0.41 (z ~ -3.1), 2015-25 -0.15 (z ~ -1.7); new floats +0.15 +/- 0.4 |
| Box-EKE deciles | flat to ~160 cm²/s², negative above ~200 (top decile -0.33 +/- 0.15 cm) |
| Regional (placebo-corrected, older floats) | Gulf Stream -0.39 +/- 0.22 cm, NAC -0.23 +/- 0.15, Subtrop W -0.13 +/- 0.07, quiet regions ~0 |
| ADT rank in box (cross-front position) | Gulf Stream -0.012 +/- 0.004: slightly toward low ADT (cold side / cyclonic; not separable) |

Conversion: residual on local SLA slope kappa ~ 0.39 (0.42 Gulf Stream) GJ/m² per cm, so the surface-visible bias is
~ -0.08 in high-EKE boxes (box EKE >= 200 cm²/s²) and -0.17 in the Gulf Stream, ~0 elsewhere.

## Correction: two routes

1. **SSH tilt** (design model): within-box intensity $\propto \exp\{\beta_1 s + \beta_2 s\,1(\text{EKE}_{box}\ge 200)\}$,
   $s$ = local SLA; $\beta$ = actual minus mean placebo conditional-logit fit (heuristic geography removal). Weights
   0.95-1.05. Cross-fitting on held-out high-EKE boxes: leave-out-Gulf-Stream fit removes the GS preference
   (-0.40 -> -0.01 cm); float halves give beta differing x2.3 (beta SE ~ 0.003); across periods it fails
   (2004-14 fit over-corrects 2015-25: -0.14 -> +0.31) - preference is non-stationary. In-sample OHC shift +0.085.
2. **Outcome model**: $\kappa(x)\,s(x,t)$ added to the local mean (`GLORYS_DR_SLA=1`), refit LOFO, covariances refit,
   264 months re-kriged (jobs 56478938/39/40, ~10 core-h, open). kappa median 0.46 (5-95%: 0.12-0.90).
   Residual sd 3.85 -> 2.89 (variance -44%); residual-on-SLA slope 0.39 -> -0.02 in every region.

| Older floats, mean residual | baseline | + SLA in map | baseline, tilted |
|---|---|---|---|
| High-EKE boxes | -0.081 | -0.003 | +0.004 |
| Gulf Stream | -0.24 | +0.04 | -0.07 |
| NAC | -0.05 | -0.02 | -0.03 |

The two routes agree (+0.078 vs +0.085 in high-EKE boxes); with SLA in the map the tilt moves residuals by <= 0.01
(the double-robustness check). With the SLA map: anchor -0.018 +/- 0.051, bands flat; pooled advected - new ~0,
tau 0.22 (4 deg) / 0.30 (8 deg).

## Open issues

1. Gulf Stream advected - new remains -0.85 +/- 0.44 (p 0.06) with the SLA map (35 cells, ~219 new-float profiles):
   subsurface sorting invisible to SLA, or noise.
2. Local - new at 4 deg with the SLA map: -0.041 +/- 0.018 (p 0.01; 8 deg -0.057 +/- 0.030). Mostly within-float,
   suggests age-related drift (sensor), not sorting; check by platform family / pressure sensor (`validate.add_meta`).
3. EKE trend of the OHC residual (+0.17 +/- 0.09, unchanged by SLA) compares 4 deg boxes within 8 deg regions =
   geographic map error; not a sorting test, drop.
4. Transition-matrix q-hat covers only the buffer crop (from Oct 2009); N Atlantic q-hat needs a cohort rerun.
5. Box A/B OHC levels and changes under the SLA map vs baseline vs tilt not yet computed.
6. Caveats: regions in `ps_tests.REGIONS` were drawn after seeing pooled results (the EKE test is the pre-specified
   one); placebo-difference beta is heuristic; float-clustered SEs ignore shared-eddy correlation (block SEs agree).

## Reproduce

```
# residual maps (already on disk): code/osim/glorys_dr.py with
#   GLORYS_DR_DOMAIN=natl GLORYS_DR_TAG=_real_natl[_sla] GLORYS_DR_KTAG=_real_natl[_sla] GLORYS_DR_PROF=all [GLORYS_DR_SLA=1]
python code/tm/detect.py build    # -> data/tm/detect_natl/{profiles.parquet, cc_*.npz}
python code/tm/detect.py report   # every table above
```

Code: `code/tm/validate.py` (residuals, age curve, `cell_units`/`cell_test`/`meta`, `deployments`/`nbhd`,
confounder FE), `code/tm/ps_tests.py` (case-control `sample`/`clogit`, `mark_test`, `density`, `REGIONS`,
`box_eke`, `design`/`tilt_weights`), `code/sat/ssh.py` (`at`, `local_anom`, `mdt_at`, `climatology`),
`code/osim/glorys_dr.py` (N Atlantic domain, SLA covariate).
