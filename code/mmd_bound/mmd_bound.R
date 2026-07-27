## Preferential-sampling bound via kernel mean embeddings (clean, standalone).
##
## Estimand: the area-time-average OHC over a lon-lat box for a TIME WINDOW,
## theta = int Y dQ, with Q the uniform area x time measure over the window.
## Bound:   |B| <= R * MMD[H, P, Q], with P the float sampling law and Q the
## reference. The design factor MMD is estimated from float POSITIONS only
## (float_id, lon, lat, date); OHC values enter only the amplitude R.
##
## A "window" is any [t_start, t_end); a single year and a multi-year average
## are just different windows. This file reuses no other project scripts.

library(data.table)

## ---- configuration -------------------------------------------------------

mmd_config <- function() list(
  data_path = "data/argo_ohc.csv",
  box   = list(lat = c(36, 40), lon = c(-68, -62)),
  years = 2020:2025,        # per-year windows to report
  avg_years = 2021:2025,    # the multi-year-average window (configurable)
  ell_s = 150,              # spatial correlation scale (km), declared, conservative
  ell_t = 45,               # temporal correlation scale (days)
  nu    = 0.5,              # Matern smoothness, fixed by fiat
  M_mc  = 2000,             # Monte Carlo draws from the reference measure Q
  ohc_col = "ohc_2000",
  seed  = 1
)

## ---- 1. data: load the box once, then subset to windows ------------------

load_argo_box <- function(path, box, ohc_col = "ohc_2000") {
  dt <- fread(path, select = c("float_id", "lat", "lon", "date", ohc_col))
  setnames(dt, ohc_col, "ohc")
  if (!inherits(dt$date, "POSIXct"))                    # fread may already parse ISO dates
    dt[, date := as.POSIXct(substr(date, 1, 19), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")]
  dt[lat >= box$lat[1] & lat <= box$lat[2] &
       lon >= box$lon[1] & lon <= box$lon[2] & is.finite(ohc),
     .(float_id, lon, lat, date, ohc)]
}

year_window <- function(years) {                        # [Jan 1 of first, Jan 1 after last)
  y <- range(years)
  list(t_start = as.POSIXct(sprintf("%d-01-01", y[1]),     tz = "UTC"),
       t_end   = as.POSIXct(sprintf("%d-01-01", y[2] + 1), tz = "UTC"),
       label   = if (y[1] == y[2]) as.character(y[1]) else sprintf("%d-%d", y[1], y[2]))
}

window_subset <- function(argo_box, w) {
  d <- argo_box[date >= w$t_start & date < w$t_end]
  d[, day := as.numeric(difftime(date, w$t_start, units = "days"))]
  d[]
}

## ---- 2. coordinates and the reference-measure sampler --------------------
## Equirectangular projection about the box centre gives kernel distances in km;
## the area measure on the sphere is uniform in (lon, sin lat), which sample_Q
## respects, so the Monte Carlo integrals against Q are area-weighted correctly.

to_physical <- function(lon, lat, day, box) {
  lat0 <- mean(box$lat); lon0 <- mean(box$lon)
  cbind(east  = (lon - lon0) * 111.32 * cos(lat0 * pi / 180),
        north = (lat - lat0) * 110.57,
        day   = day)
}

sample_Q <- function(M, box, w) {
  s <- sin(box$lat * pi / 180)
  ndays <- as.numeric(difftime(w$t_end, w$t_start, units = "days"))
  data.table(
    lon = runif(M, box$lon[1], box$lon[2]),
    lat = asin(runif(M, s[1], s[2])) * 180 / pi,   # uniform in sin(lat) = area measure
    day = runif(M, 0, ndays)
  )
}

## ---- 3. kernel -----------------------------------------------------------
## Stationary anisotropic Matern: Sigma = diag(ell_s^2, ell_s^2, ell_t^2), the
## constant-Sigma case of the Paciorek-Schervish construction (prefactor 1).

matern_cor <- function(r, nu = 0.5) {
  r <- pmax(r, 0)
  if (nu == 0.5) return(exp(-r))
  if (nu == 1.5) return((1 + sqrt(3) * r) * exp(-sqrt(3) * r))
  if (nu == 2.5) return((1 + sqrt(5) * r + 5 * r^2 / 3) * exp(-sqrt(5) * r))
  s <- sqrt(2 * nu) * r
  out <- (2^(1 - nu) / gamma(nu)) * s^nu * besselK(s, nu)
  out[r == 0] <- 1
  out
}

kernel_matrix <- function(X1, X2, ell_s, ell_t, nu = 0.5) {
  sc <- c(ell_s, ell_s, ell_t)
  A <- sweep(X1, 2, sc, "/"); B <- sweep(X2, 2, sc, "/")
  D2 <- outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B)
  matern_cor(sqrt(pmax(D2, 0)), nu)
}

## ---- 4-5. the estimator --------------------------------------------------
## Kobs = k(Z_i, Z_j); kbar_z(i) = int k(Z_i, .) dQ (Monte Carlo);
## kbb = int int k dQ dQ. Jackknife reuses these by subsetting.

compute_pieces <- function(argo, mc, cfg) {
  Xobs <- to_physical(argo$lon, argo$lat, argo$day, cfg$box)
  Xmc  <- to_physical(mc$lon,  mc$lat,  mc$day,  cfg$box)
  kfun <- if (!is.null(cfg$kfun)) cfg$kfun else               # non-stationary kernel hook
    function(A, B) kernel_matrix(A, B, cfg$ell_s, cfg$ell_t, cfg$nu)
  list(Kobs = kfun(Xobs, Xobs), kbar_z = rowMeans(kfun(Xobs, Xmc)),
       kbb = mean(kfun(Xmc, Xmc)), fid = argo$float_id)
}

mmd_from_pieces <- function(K, kbar, kbb, fid) {
  n <- length(fid)
  off_sum <- sum(K) - sum(diag(K))
  grp <- split(seq_len(n), fid)
  within <- sum(vapply(grp, function(ix) {
    Kf <- K[ix, ix, drop = FALSE]; sum(Kf) - sum(diag(Kf))
  }, numeric(1)))
  d <- lengths(grp)
  P_bf <- n * (n - 1) - sum(d * (d - 1))
  term1 <- (off_sum - within) / P_bf                     # between-float U-statistic
  mmd2 <- term1 - 2 * mean(kbar) + kbb
  list(mmd2 = mmd2, mmd = sqrt(max(mmd2, 0)),
       term1 = term1, term2 = 2 * mean(kbar), term3 = kbb,
       n = n, L = length(grp), P_bf = P_bf, d = d)
}

estimate_mmd <- function(argo, mc, cfg) {
  pc <- compute_pieces(argo, mc, cfg)
  mmd_from_pieces(pc$Kobs, pc$kbar_z, pc$kbb, pc$fid)
}

## Delete-one-float jackknife on MMD^2 (near the null MMD^2 can be slightly
## negative; the clamped MMD would give a degenerate SE). Upper limit on MMD is
## sqrt of the clamped upper limit on MMD^2.
jackknife_mmd <- function(argo, mc, cfg, z = 1.645) {
  pc <- compute_pieces(argo, mc, cfg)
  full <- mmd_from_pieces(pc$Kobs, pc$kbar_z, pc$kbb, pc$fid)
  floats <- unique(pc$fid); L <- length(floats)
  theta2 <- vapply(floats, function(f) {
    keep <- pc$fid != f
    mmd_from_pieces(pc$Kobs[keep, keep], pc$kbar_z[keep], pc$kbb, pc$fid[keep])$mmd2
  }, numeric(1))
  se2 <- sqrt((L - 1) / L * sum((theta2 - mean(theta2))^2))
  mmd2_upper <- full$mmd2 + z * se2
  list(mmd = full$mmd, mmd2 = full$mmd2, se2 = se2, mmd2_upper = mmd2_upper,
       upper = sqrt(max(mmd2_upper, 0)), terms = full,
       theta2 = setNames(theta2, floats), L = L)
}

## ---- 5b. iid comparison and hypothesis tests -----------------------------

## Two first-term variants from the precomputed pieces:
##  - iid  : all ordered pairs (Muandet eq. 3.32), the standard estimator that
##           treats every profile as an independent draw;
##  - bf   : our between-float U-statistic, which drops within-float pairs.
## Their difference is the trajectory-autocorrelation inflation; within_frac is
## the share of pair kernel-mass that sits within floats.
mmd_variants <- function(pc) {
  K <- pc$Kobs; fid <- pc$fid; n <- length(fid)
  off_sum <- sum(K) - sum(diag(K))
  grp <- split(seq_len(n), fid)
  within <- sum(vapply(grp, function(ix) {
    Kf <- K[ix, ix, drop = FALSE]; sum(Kf) - sum(diag(Kf))
  }, numeric(1)))
  d <- lengths(grp); P_bf <- n * (n - 1) - sum(d * (d - 1))
  ref <- -2 * mean(pc$kbar_z) + pc$kbb
  list(mmd2_iid = off_sum / (n * (n - 1)) + ref,
       mmd2_bf  = (off_sum - within) / P_bf + ref,
       n = n, L = length(grp), d = d, P_bf = P_bf,
       within_frac = within / off_sum)
}

## Large-deviation (McDiarmid) test of H0: P = Q. Distribution-free, one-sided,
## on the between-float MMD^2, with the float-level bounded differences of the
## deviation bound (Prop. 2.3, kappa_k = 1 for a correlation kernel). The p-value
## is the bound itself, hence conservative (Gretton et al. 2012).
mmd_ld_test <- function(pc, kappa = 1, alpha = 0.05) {
  v <- mmd_variants(pc); d <- v$d; N <- v$n; P_bf <- v$P_bf
  cl <- 4 * kappa * (d * (N - d) / P_bf + d / N)
  S <- sum(cl^2)
  stat <- v$mmd2_bf
  data.table(stat = stat, threshold = sqrt(log(1 / alpha) * S / 2),
             pvalue = if (stat > 0) min(exp(-2 * stat^2 / S), 1) else 1,
             reject = stat > sqrt(log(1 / alpha) * S / 2))
}

## Monte Carlo test of H0: P = Q using the known reference: each of the L floats
## draws its d_l positions iid from Q, giving a calibrated null distribution of
## the between-float MMD^2 and a p-value.
mmd_mc_test <- function(argo, mc, cfg, w, B = 199, alpha = 0.05) {
  pc <- compute_pieces(argo, mc, cfg)
  stat_obs <- mmd_variants(pc)$mmd2_bf
  d <- lengths(split(seq_along(pc$fid), pc$fid))
  N <- sum(d); fid_t <- rep(seq_along(d), d)
  Xmc <- to_physical(mc$lon, mc$lat, mc$day, cfg$box)
  null <- vapply(seq_len(B), function(b) {
    qs <- sample_Q(N, cfg$box, w)
    Xq <- to_physical(qs$lon, qs$lat, qs$day, cfg$box)
    Kobs <- kernel_matrix(Xq, Xq, cfg$ell_s, cfg$ell_t, cfg$nu)
    kbar <- rowMeans(kernel_matrix(Xq, Xmc, cfg$ell_s, cfg$ell_t, cfg$nu))
    mmd_from_pieces(Kobs, kbar, pc$kbb, fid_t)$mmd2
  }, numeric(1))
  pval <- (1 + sum(null >= stat_obs)) / (B + 1)
  list(stat = stat_obs, pvalue = pval, reject = pval < alpha, null = null, B = B)
}

## ---- 6. amplitude anchors (diagnostic; the mean contributes no bias) ------
## The field amplitude here is a VARIANCE-scale quantity: tau_hat estimates the
## marginal SD sigma_Y, which is the amplitude of the PROBABILISTIC bound
## sd(B) = tau * MMD (Cor. 2.1). Popoviciu caps a variance, so tau <= M/2. It is
## NOT the RKHS norm R = ||Y||_H of the deterministic bound (Popoviciu does not
## bound R; R needs the smoothness ceiling M * ell_min^-s). We therefore report
## the probabilistic sd(B) and, for the deterministic bound, a breakdown budget.

amplitude_anchors <- function(argo, to_GJ = 1e-9) {
  o <- argo$ohc * to_GJ
  list(tau_hat = sd(o), M = diff(range(o)), tau_max = diff(range(o)) / 2,
       mean = mean(o))
}

## Deterministic breakdown budget: the RKHS norm ||Y||_H at which |B| reaches a
## tolerance T (GJ/m2). Reported instead of R*MMD, since R is not estimable.
breakdown_budget <- function(mmd, T_vec = c(0.05, 0.1, 0.25, 0.5)) {
  data.table(tolerance_GJm2 = T_vec,
             R_star = if (mmd > 0) T_vec / mmd else rep(Inf, length(T_vec)))
}

## ---- 7. window drivers ---------------------------------------------------

## Per-window summary. The rigorous quantity is the design factor MMD. sd_bias is
## the probabilistic bound sd(B) = tau * MMD (data tau; Cor. 2.1); a magnitude
## bound multiplies it by a confidence factor (1.96 Gaussian; lambda with
## P<=1/lambda^2 Chebyshev). No deterministic R*MMD is reported (see note above).
estimate_window <- function(argo_box, cfg, w) {
  argo <- window_subset(argo_box, w)
  N <- nrow(argo); L <- uniqueN(argo$float_id)
  if (N < 3 || L < 2)
    return(data.table(window = w$label, N = N, L = L, mmd2 = NA_real_,
                      mmd_upper = NA_real_, tau = NA_real_, sd_bias = NA_real_,
                      prob95 = NA_real_))
  mc  <- sample_Q(cfg$M_mc, cfg$box, w)
  jk  <- jackknife_mmd(argo, mc, cfg)
  amp <- amplitude_anchors(argo)
  data.table(window = w$label, N = N, L = L, mmd2 = jk$mmd2,
             mmd_upper = jk$upper, tau = amp$tau_hat,
             sd_bias = amp$tau_hat * jk$upper,               # sd(B) = tau * MMD
             prob95  = 1.96 * amp$tau_hat * jk$upper)        # 95% Gaussian magnitude
}

## Build the windows: one per year plus the multi-year average.
build_windows <- function(cfg) {
  wins <- lapply(cfg$years, function(y) year_window(y))
  wins[[length(wins) + 1]] <- year_window(cfg$avg_years)
  wins
}

run_all_windows <- function(argo_box, cfg) {
  set.seed(cfg$seed)
  rbindlist(lapply(build_windows(cfg), function(w) estimate_window(argo_box, cfg, w)))
}

## iid-vs-between-float comparison and both hypothesis tests, per window.
run_tests <- function(argo_box, cfg, B = 199) {
  set.seed(cfg$seed)
  rbindlist(lapply(build_windows(cfg), function(w) {
    argo <- window_subset(argo_box, w)
    if (nrow(argo) < 3 || uniqueN(argo$float_id) < 2) return(NULL)
    mc <- sample_Q(cfg$M_mc, cfg$box, w)
    pc <- compute_pieces(argo, mc, cfg)
    v  <- mmd_variants(pc); ld <- mmd_ld_test(pc)
    mct <- mmd_mc_test(argo, mc, cfg, w, B = B)
    data.table(window = w$label, mmd2_iid = v$mmd2_iid, mmd2_bf = v$mmd2_bf,
               within_frac = v$within_frac, ld_pvalue = ld$pvalue,
               mc_pvalue = mct$pvalue)
  }))
}

## Scale sensitivity: MMD (point and upper) over a grid of ell_s, per window.
sweep_windows <- function(argo_box, cfg, ell_s_grid = c(50, 100, 150, 300)) {
  set.seed(cfg$seed)
  rbindlist(lapply(build_windows(cfg), function(w) {
    argo <- window_subset(argo_box, w)
    if (nrow(argo) < 3 || uniqueN(argo$float_id) < 2) return(NULL)
    mc <- sample_Q(cfg$M_mc, cfg$box, w)
    rbindlist(lapply(ell_s_grid, function(e) {
      cfg2 <- cfg; cfg2$ell_s <- e
      jk <- jackknife_mmd(argo, mc, cfg2)
      data.table(window = w$label, ell_s = e, mmd = jk$mmd, mmd_upper = jk$upper)
    }))
  }))
}
