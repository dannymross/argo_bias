# Literature for the cross-fit DR / debiased-kriging estimator

Our estimator, briefly: the target is the area integral of a fixed field over a box (design frame).
We use a GP map with a GLS mean (optionally with log q-hat as a covariate). The map is corrected by
1/q-weighted leave-one-float-out residuals. q is the sampling intensity, predicted by advecting
particle clouds through GLORYS velocities rather than fitted to the floats' own measurements.

Keys refer to `paper/references.bib`. "Checked" means the bibliographic details were confirmed
online on 2026-10-07; summaries of the 2026 preprints come from abstracts plus a machine summary of
the HTML, so re-check theorem statements against the PDFs before quoting them.

## 1. Closest recent work (read these first)

- **shirota2026**, *Design-Based Prediction-Powered Inference for Spatial Data* (arXiv 2608.10356,
  Aug 2026). Checked.
  - *Setting:* a census mean over a fixed finite pixel population; the only randomness is the
    labelling design. The estimator is PPI/AIPW, the map mean plus a design-weighted rectifier.
  - *Propensities:* either known, or a logistic model fitted on all units (with the map as a
    covariate), with sandwich inference. No cross-fitting.
  - *Theorem 1:* if the propensity is wrong and the outcome model is right, a conditional
    finite-population gap of order sigma_u / sqrt(N_eff,v) remains. N_eff,v counts the residual
    "patches" seen by the centred weight-ratio field pi/pi-tilde. The gap is negligible for iid
    residuals. Under spatially coherent residuals it binds and does not shrink with n, so coverage
    can worsen as n grows.
  - *Scope:* selection on the residual itself (preferential sampling) is explicitly out of scope.
  - *Data:* a 48k-cell enumerated land-cover population, plus Estonian LUCAS data.
  - *For us:* this is the formal version of our misspecified-cloud results. With a wrong q, DR
    keeps a bias of about 0.01-0.02 that the replicates do not average away, because the field is
    fixed. We differ in four ways: a continuous domain with Campbell intensity, not a finite
    population; q from physics, not fitted; cluster (float) dependence plus LOFO cross-fitting;
    and sampling that is preferential through the flow. Cite it for the asymmetry; do not claim
    the asymmetry as ours.
- **salerno2026**, *Spatially Robust Inference with Predicted and Missing at Random Labels* (arXiv
  2603.11368, Mar 2026). Checked (abstract only).
  - *Method:* a DR estimator with cross-fit nuisances under MAR labelling and spatial dependence.
  - *Key point:* cross-fitting induces fold-level correlation that distorts spatial HAC variances,
    so they propose a jackknife spatial-HAC correction.
  - *Estimand:* a superpopulation one (Shirota contrasts himself with it).
  - *For us:* the closest precedent for "cross-fit DR plus spatial dependence". Its variance issue
    is the one we will meet when we build intervals from LOFO. Read the full paper next.
- **hsiao2026** (Hsiao & Waller, JABES 2026; already in the bib). ISIW estimates the sampling
  intensity, uses 1/lambda-hat as likelihood weights, then kriges. It is IPW inside the parameter
  estimates, not DR on the functional. They find predictive accuracy is weakly tied to parameter
  accuracy, which is consistent with our "the mean fit matters less than the correction".
- **lu2025ignorability** (Lu, Xu, Yang & Guan, arXiv 2511.03158). Gives conditions under which
  conventional geostatistical estimators that ignore preferential sampling stay unbiased and
  consistent. It cites Hsiao & Waller. We need to check whether its conditions cover the GLS+log q
  mean, i.e. whether conditioning on q makes sampling ignorable for the mean.

## 2. Same estimator form, known design (model-assisted / PPI)

- **sarndal1992** (in bib), **breidt2017** (checked): GREG/difference estimator, i.e. predictions
  summed over the population plus HT-weighted residuals. Ours has this form with an estimated q.
- **cordy1993** (in bib): Horvitz-Thompson for point sampling from a continuous universe, the
  finite-population counterpart of our Campbell identity.
- **brus1997** (checked): design-based vs geostatistical (kriging) strategies for soil means; a
  classic discussion paper on this tension.
- **cicchitelli2012** (checked; volume and pages not confirmed): a model-assisted spatial mean
  with penalized splines on the coordinates, compared with block kriging, in a design-based frame.
- **stahl2016** (checked): forestry and remote sensing, comparing model-assisted, model-based and
  hybrid estimation.
- **angelopoulos2023** (checked), **datta2025** (IPW version of PPI with known or estimated
  propensities), **kluger2025** (PPI under weighted, stratified or clustered labelled samples).
  These are the PPI/ML counterparts.
- **rizvi2026** (arXiv 2610.08715, NeurIPS 2026 workshop) and **burt2025** (NeurIPS 2025): PPI for
  spatial time series, and Lipschitz-based coverage under nonrandom spatial locations. They are
  adjacent rather than direct precedents.
- Also in the bib: **dumelle2022** (design-based vs model-based finite-population spatial
  inference).

## 3. Unknown propensity: AIPW, nonprobability samples, cross-fitting

- **robins1994** (in bib), **bang2005** (checked; DR plus the clever covariate behind our log q
  idea), **tan2006**.
- **chen2020** (checked) and **yang2019** (arXiv): DR estimation for nonprobability samples, where
  the propensity is estimated using a reference probability sample. This is the survey analogue of
  our problem; there the q-hat is fitted to data, ours comes from a physical model.
- **chernozhukov2018** and **chiang2022** (in bib): cross-fitting, and cluster-level cross-fitting.
  LOFO is cluster cross-fitting with floats as clusters.

## 4. Preferential sampling in geostatistics

- **diggle2010**, **gelfand2012**, **conn2017** (all in bib), **vedensky2023** (checked): the last
  maps preferential sampling onto informative sampling in survey statistics, which is the bridge
  our framing needs.
- **warton2013** (checked): puts "observer bias" covariates in a point-process model for species
  presence data. This is the ecology analogue of putting log q-hat in the mean.

## 5. Oceanography

- **davis1991**: *array bias*, i.e. pseudo-Eulerian averages of float data are biased by gradients
  in float density, and a correction can use float density and eddy diffusivity. This is the
  physical-oceanography precedent for "float density correlates with the field". Only the citation
  was checked (Deep-Sea Res. 38 Suppl. 1); the S531-S571 pages and the content are from memory and
  secondary sources, so read it.
- **kuusela2018**, **park2023** (in bib): local GP / debiased latent-GP mapping of Argo data. These
  are model-based, with no propensity correction.
- I found no IPW or DR estimator applied to Argo or OHC mapping in this search.

## Positioning

These ingredients appear separately in the literature; this search found no paper combining them:

1. A propensity from an external physical model of how the sampler moves (advected clouds), not
   from a known design or a model fitted to the sample.
2. Lagrangian samplers as clustered point processes, with LOFO cross-fitting and the exact
   propensity-mixed (q_f/q) map identity.
3. A continuous-domain (Campbell) design frame for a fixed field.
4. The application to OHC.

The double-robustness claim must be stated in Shirota's asymmetric form for a fixed field.

## Next reads

1. Shirota, Theorem 1 and its N_eff definition. Map it onto our misspecification scenarios: can we
   compute N_eff,v from the q-hat/q ratio fields?
2. Salerno et al.: the jackknife spatial-HAC correction, for LOFO intervals.
3. Lu et al. (2025): the ignorability conditions vs the GLS+log q mean.
4. Davis (1991): the array-bias correction, as a physical precedent for q.
