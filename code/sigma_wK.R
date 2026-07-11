## =====================================================================
## sigma_wK.R -- Model-free estimation of the coarsened sampling
##   dispersion sigma^2_{w,K} with a delete-one-float jackknife CI.
##
##   hat{sigma^2_{w,K}} = K * sum_j n_j(n_j-1) / [n(n-1)] - 1
##
##   K = K_lon*K_lat*K_t equal-nu cells; n_j = profile count in cell j
##   (summed over ALL K cells, empties included); n = total.
##
##   All 364 profiles are used (no thinning): consecutive surfacings of a
##   float are ~10 days / tens-to-hundreds of km apart, so profiles are
##   treated as independent draws from the sampling measure pi, under
##   which the estimator above is unbiased (Test A).  Float is ALWAYS the
##   resampling/jackknife unit for the CI (Tests C,D), since the
##   independent replication in the data is the trajectory, not the
##   profile.  Two options are provided for robustness:
##     * between_float = TRUE : counts only BETWEEN-float coincidence
##       pairs, staying exactly unbiased even if within-float profiles
##       are dependent (Test D).  Makes no within-float independence
##       assumption.
##     * consecutive_cell_check(): diagnostic for whether the
##       independence assumption holds at the CELL scale in use.
## Base R only.  Inputs are float LOCATIONS (OHC value not used here).
## =====================================================================

## ---- 1. Equal-measure cell assignment -------------------------------
assign_cells <- function(lon, lat, t, box, K_lon, K_lat, K_t) {
  in_box <- lon >= box$lon[1] & lon <= box$lon[2] &
            lat >= box$lat[1] & lat <= box$lat[2] &
            t   >= box$t[1]   & t   <= box$t[2]
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  e_t   <- seq(box$t[1], box$t[2], length.out = K_t + 1)
  fl <- findInterval(lon,             e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  fb <- findInterval(sin(lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  ft <- findInterval(t,               e_t,   rightmost.closed = TRUE, all.inside = TRUE)
  cell <- (ft - 1L) * (K_lon * K_lat) + (fb - 1L) * K_lon + fl
  cell[!in_box] <- NA_integer_
  cell
}

## ---- 2. Estimators ---------------------------------------------------
## Standard diagonal-removed (unbiased if ALL profiles are iid ~ pi).
sigma_wK2_hat <- function(cell, K) {
  cell <- cell[!is.na(cell)]; n <- length(cell)
  if (n < 2L) return(NA_real_)
  nj <- tabulate(cell, nbins = K)
  K * sum(nj * (nj - 1)) / (n * (n - 1)) - 1
}
## Between-float-only diagonal removal: counts only coincidence pairs
## from DIFFERENT floats, normalized by the between-float pair total.
## Unbiased if BETWEEN-float profiles are iid ~ pi, with NO within-float
## independence assumption.  Uses:
##   sum_j n_j(n_j-1) = (between-float pairs) + sum_f (within-float pairs)
sigma_wK2_hat_bf <- function(cell, float, K) {
  ok <- !is.na(cell); cell <- cell[ok]; float <- float[ok]
  n <- length(cell); if (n < 2L) return(NA_real_)
  nj    <- tabulate(cell, nbins = K)
  S_all <- sum(nj * (nj - 1))                              # all ordered coincidences
  by_f  <- split(cell, float)
  S_wf  <- sum(vapply(by_f, function(cc) {                 # within-float coincidences
    m <- tabulate(match(cc, unique(cc))); sum(m * (m - 1)) }, numeric(1)))
  d_f   <- lengths(by_f)
  P_bf  <- n * (n - 1) - sum(d_f * (d_f - 1))              # ordered between-float pairs
  if (P_bf <= 0) return(NA_real_)
  K * (S_all - S_wf) / P_bf - 1
}
## Plug-in (biased high), diagnostic only.
sigma_wK2_plugin <- function(cell, K) {
  cell <- cell[!is.na(cell)]; n <- length(cell); nj <- tabulate(cell, nbins = K)
  K * sum((nj / n)^2) - 1
}
## Dispatcher used by the CI machinery.
sigma_wK2_core <- function(cell, float, K, between_float = FALSE)
  if (between_float) sigma_wK2_hat_bf(cell, float, K) else sigma_wK2_hat(cell, K)

## ---- 3. Optional thinning (off by default) --------------------------
thin_one_per_float_cell <- function(cell, float) {
  keep <- !duplicated(cbind(as.integer(factor(float)), cell))
  list(cell = cell[keep], float = float[keep], keep = keep)
}

## ---- 4. Point estimate ----------------------------------------------
estimate_sigma_wK <- function(argo, box, K_lon, K_lat, K_t,
                              thin = FALSE, between_float = FALSE) {
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  fl <- argo$float_id; drop <- is.na(cell)
  if (any(drop)) { cell <- cell[!drop]; fl <- fl[!drop] }
  if (thin) { th <- thin_one_per_float_cell(cell, fl); cell <- th$cell; fl <- th$float }
  K <- K_lon * K_lat * K_t
  s2 <- sigma_wK2_core(cell, fl, K, between_float)
  list(sigma2 = s2, sigma = sqrt(max(0, s2)),
       sigma2_std = sigma_wK2_hat(cell, K),                # both reported for comparison
       sigma2_bf  = sigma_wK2_hat_bf(cell, fl, K),
       sigma2_plugin = sigma_wK2_plugin(cell, K),
       n = length(cell), K = K, n_float = length(unique(fl)),
       n_dropped = sum(drop), cell = cell, float = fl)
}

## ---- 5. Uncertainty: delete-one-float jackknife ---------------------
## Float is the unit: each replicate deletes ALL profiles of one float.
## Works for both estimators.  (A with-replacement float bootstrap is
## biased for this U-statistic via a self-overlap tie artifact -- see
## boot_sigma_wK, kept as a diagnostic only.)
jackknife_sigma_wK <- function(argo, box, K_lon, K_lat, K_t,
                               thin = FALSE, between_float = FALSE) {
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  fl <- argo$float_id; drop <- is.na(cell)
  if (any(drop)) { cell <- cell[!drop]; fl <- fl[!drop] }
  if (thin) { th <- thin_one_per_float_cell(cell, fl); cell <- th$cell; fl <- th$float }
  K <- K_lon * K_lat * K_t
  floats <- unique(fl); F <- length(floats)
  full <- sigma_wK2_core(cell, fl, K, between_float)
  theta <- vapply(floats, function(f) {                    # delete whole float f
    keep <- fl != f
    sigma_wK2_core(cell[keep], fl[keep], K, between_float)
  }, numeric(1))
  var_jack <- (F - 1) / F * sum((theta - mean(theta))^2)
  list(sigma2_hat = full, se = sqrt(var_jack), n_float = F)
}

## Diagnostic only (NOT for CIs): with-replacement float bootstrap.
## Duplicated floats get fresh labels so between-float pairs are correct.
boot_sigma_wK <- function(argo, box, K_lon, K_lat, K_t, n_boot = 2000,
                          thin = FALSE, between_float = FALSE, seed = 1L) {
  set.seed(seed)
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  fl <- argo$float_id; drop <- is.na(cell)
  if (any(drop)) { cell <- cell[!drop]; fl <- fl[!drop] }
  if (thin) { th <- thin_one_per_float_cell(cell, fl); cell <- th$cell; fl <- th$float }
  K <- K_lon * K_lat * K_t
  floats <- unique(fl); nf <- length(floats)
  foot <- split(cell, factor(fl, levels = floats))
  vapply(seq_len(n_boot), function(b) {
    idx <- sample.int(nf, nf, replace = TRUE)
    cc  <- unlist(foot[idx], use.names = FALSE)
    ff  <- rep(seq_along(idx), lengths(foot[idx]))         # fresh labels per draw
    sigma_wK2_core(cc, ff, K, between_float)
  }, numeric(1))
}

## Point estimate + jackknife CI on both sigma^2 and sigma scales.
sigma_wK_with_ci <- function(argo, box, K_lon, K_lat, K_t,
                             thin = FALSE, between_float = FALSE, level = 0.95) {
  z  <- qnorm(1 - (1 - level) / 2)
  jk <- jackknife_sigma_wK(argo, box, K_lon, K_lat, K_t, thin, between_float)
  est <- estimate_sigma_wK(argo, box, K_lon, K_lat, K_t, thin, between_float)
  lo2 <- jk$sigma2_hat - z * jk$se; hi2 <- jk$sigma2_hat + z * jk$se
  data.frame(K = est$K, n = est$n, n_float = est$n_float,
             estimator = if (between_float) "between-float" else "standard",
             sigma2_hat = jk$sigma2_hat, se_sigma2 = jk$se,
             sigma2_lo = lo2, sigma2_hi = hi2,
             sigma_hat = sqrt(max(0, jk$sigma2_hat)),
             sigma_lo = sqrt(max(0, lo2)), sigma_hi = sqrt(max(0, hi2)),
             row.names = NULL)
}

## ---- 6. Refinement curve --------------------------------------------
grid_from_cellsize <- function(box, cell_km_vec, K_t) {
  Rk <- 6371; lat0 <- mean(box$lat)
  w_lon <- Rk * cos(lat0 * pi / 180) * diff(box$lon) * pi / 180
  w_lat <- Rk *                        diff(box$lat) * pi / 180
  do.call(rbind, lapply(cell_km_vec, function(ck)
    data.frame(cell_km = ck,
               K_lon = max(1L, round(w_lon / ck)),
               K_lat = max(1L, round(w_lat / ck)),
               K_t   = K_t)))
}
refinement_curve <- function(argo, box, grids, thin = FALSE,
                             between_float = FALSE, level = 0.95) {
  do.call(rbind, lapply(seq_len(nrow(grids)), function(i) {
    g <- grids[i, ]
    r <- sigma_wK_with_ci(argo, box, g$K_lon, g$K_lat, g$K_t, thin, between_float, level)
    cbind(cell_km = g$cell_km, K_lon = g$K_lon, K_lat = g$K_lat, K_t = g$K_t, r)
  }))
}

## ---- 7. Consecutive-cell check (independence-at-cell-scale test) -----
## For consecutive surfacings of the same float, what fraction land in
## the SAME (or adjacent) spatial cell?  Small fractions support treating
## profiles as independent draws at the cell scale in use.
gc_dist_km <- function(lon1, lat1, lon2, lat2) {
  R <- 6371; tr <- pi / 180
  dlat <- (lat2 - lat1) * tr; dlon <- (lon2 - lon1) * tr
  a <- sin(dlat/2)^2 + cos(lat1*tr) * cos(lat2*tr) * sin(dlon/2)^2
  2 * R * asin(pmin(1, sqrt(a)))
}
consecutive_cell_check <- function(argo, box, K_lon, K_lat, K_t) {
  o <- order(argo$float_id, argo$t); a <- argo[o, ]
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  e_t   <- seq(box$t[1], box$t[2], length.out = K_t + 1)
  bl <- findInterval(a$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  bb <- findInterval(sin(a$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  bt <- findInterval(a$t, e_t, rightmost.closed = TRUE, all.inside = TRUE)
  n  <- nrow(a); sf <- which(a$float_id[-1] == a$float_id[-n])
  dlon <- abs(bl[-1] - bl[-n])[sf]; dlat <- abs(bb[-1] - bb[-n])[sf]
  dtb  <- abs(bt[-1] - bt[-n])[sf]
  step <- gc_dist_km(a$lon[-n], a$lat[-n], a$lon[-1], a$lat[-1])[sf]
  data.frame(K_lon = K_lon, K_lat = K_lat, K_t = K_t,
             n_consec_pairs = length(sf), median_step_km = median(step),
             frac_same_cell = mean(dlon == 0 & dlat == 0 & dtb == 0),
             frac_same_spatial_cell = mean(dlon == 0 & dlat == 0),
             frac_adjacent_spatial = mean(dlon <= 1 & dlat <= 1),
             row.names = NULL)
}
consecutive_check_curve <- function(argo, box, grids)
  do.call(rbind, lapply(seq_len(nrow(grids)), function(i) {
    g <- grids[i, ]
    cbind(cell_km = g$cell_km, consecutive_cell_check(argo, box, g$K_lon, g$K_lat, g$K_t))
  }))

## ---- 8. Data cleaning: drop intra-cycle repeats (dt < min_gap days) --
## Greedily keeps the first profile of each float, then a profile only if
## it is >= min_gap days after the last KEPT profile of that float, so a
## run of sub-cycle repeats collapses to a single retained profile.
thin_min_gap <- function(argo, min_gap = 9) {
  o <- order(argo$float_id, argo$t); argo <- argo[o, ]
  keep <- ave(argo$t, argo$float_id, FUN = function(tt) {
    k <- logical(length(tt)); last <- -Inf
    for (i in seq_along(tt)) if (tt[i] - last >= min_gap) { k[i] <- TRUE; last <- tt[i] }
    as.numeric(k)
  })
  argo[keep == 1, ]
}

## ---- 9. Per-cell weight: observed / uniform-expected count ----------
## w_bar_j = K * n_j / n : 1 = sampled as under uniform coverage,
## >1 over-sampled (preferential), 0 = unsampled.  sigma_{w,K} is the
## sd of this map about 1, so the map and the scalar are two views.
cell_weights <- function(argo, box, K_lon, K_lat, K_t) {
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  cell <- cell[!is.na(cell)]; n <- length(cell); K <- K_lon * K_lat * K_t
  nj <- tabulate(cell, nbins = K)
  data.frame(cell = seq_len(K), n_j = nj, w_bar = K * nj / n)
}
## spatial marginal weight map (aggregated over time): [K_lon x K_lat]
cell_weights_spatial <- function(argo, box, K_lon, K_lat) {
  cw <- cell_weights(argo, box, K_lon, K_lat, K_t = 1L)
  list(w = matrix(cw$w_bar, K_lon, K_lat), n_j = matrix(cw$n_j, K_lon, K_lat))
}
plot_cell_weights <- function(argo, box, K_lon, K_lat, annotate = TRUE) {
  cw <- cell_weights_spatial(argo, box, K_lon, K_lat)
  el <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  lon_mid <- (el[-1] + el[-(K_lon+1)]) / 2
  sl <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  lat_mid <- asin((sl[-1] + sl[-(K_lat+1)]) / 2) * 180 / pi
  brks <- seq(0, 2, length.out = 21)                  # white centered at w=1
  cols <- colorRampPalette(c("navy","white","firebrick"))(20)
  image(lon_mid, lat_mid, pmin(cw$w, 2), col = cols, breaks = brks,
        xlab = "lon", ylab = "lat",
        main = expression("cell weight "*hat(bar(w))[j]*"   (1 = uniform, >1 preferential)"))
  contour(lon_mid, lat_mid, cw$w, levels = 1, add = TRUE, lwd = 2, lty = 2)
  if (annotate) text(rep(lon_mid, K_lat), rep(lat_mid, each = K_lon),
                     sprintf("%.1f", as.vector(cw$w)), cex = 0.55)
  invisible(cw)
}

## diverging palette centered at w=1 (navy<1<firebrick)
div_scale <- function(cap = 3, n = 10) list(
  breaks = c(seq(0, 1, length.out = n + 1), seq(1, cap, length.out = n + 1)[-1]),
  cols   = c(colorRampPalette(c("#2166ac","white"))(n),
             colorRampPalette(c("white","#b2182b"))(n)))

## per-time-bin spatial weight maps at the estimation resolution.
## Shows the K_lon x K_lat x K_t weights w_bar_j = K n_j / n (the exact
## object sigma_{w,K} aggregates) as K_t spatial slices on a shared scale.
plot_cell_weights_by_time <- function(argo, box, K_lon, K_lat, K_t,
                                      cap = NULL, annotate = TRUE) {
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  cell <- cell[!is.na(cell)]; n <- length(cell); K <- K_lon * K_lat * K_t
  w <- K * tabulate(cell, nbins = K) / n
  A <- array(w, dim = c(K_lon, K_lat, K_t))            # lon fastest, then lat, then time
  el <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  lon_mid <- (el[-1] + el[-(K_lon+1)]) / 2
  sl <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  lat_mid <- asin((sl[-1] + sl[-(K_lat+1)]) / 2) * 180 / pi
  et <- as.Date(seq(box$t[1], box$t[2], length.out = K_t + 1), origin = "1970-01-01")
  if (is.null(cap)) cap <- max(2, ceiling(max(w)))
  ds <- div_scale(cap)
  op <- par(mfrow = c(1, K_t), mar = c(4, 3.6, 3, 1), oma = c(0, 0, 2.4, 0)); on.exit(par(op))
  for (k in seq_len(K_t)) {
    image(lon_mid, lat_mid, pmin(A[,,k], cap), col = ds$cols, breaks = ds$breaks,
          xlab = "lon", ylab = if (k == 1) "lat" else "",
          main = sprintf("%s to %s", et[k], et[k + 1]))
    contour(lon_mid, lat_mid, A[,,k], levels = 1, add = TRUE, lwd = 1.5, lty = 2)
    if (annotate) text(rep(lon_mid, K_lat), rep(lat_mid, each = K_lon),
                       sprintf("%.1f", as.vector(A[,,k])), cex = 0.5)
  }
  mtext(bquote("cell weight "*hat(bar(w))[j]*
        " at "*.(round(6371*cos(mean(box$lat)*pi/180)*diff(box$lon)*pi/180/K_lon))*
        " km cells, by time bin   (1 = uniform, >1 preferential)"),
        outer = TRUE, cex = 1.0, font = 2)
  invisible(A)
}

## ---- 10. Weight map matching the UNBIASED sigma_w --------------------
## The raw map's dispersion is the biased plug-in.  Each cell value
## w_bar_j = K n_j/n is individually unbiased for the true weight, but
## the map's spread = plug-in = true + per-cell noise.  To show a map
## whose dispersion equals the unbiased estimate, shrink every cell
## toward 1 by s = sqrt(unbiased / plug-in).  (between_float=TRUE targets
## the between-float estimate; if it is <= 0 the map is all uniform.)
shrink_weights <- function(argo, box, K_lon, K_lat, K_t, between_float = FALSE) {
  cell <- assign_cells(argo$lon, argo$lat, argo$t, box, K_lon, K_lat, K_t)
  ok <- !is.na(cell); fl <- argo$float_id[ok]; cell <- cell[ok]
  n <- length(cell); K <- K_lon * K_lat * K_t
  w <- K * tabulate(cell, K) / n
  s2_plug <- sum((w - 1)^2) / K
  s2_unb  <- sigma_wK2_core(cell, fl, K, between_float)
  s <- if (s2_plug > 0) sqrt(max(0, s2_unb) / s2_plug) else 0
  list(w_raw = w, w_shrunk = 1 + s * (w - 1), s = s,
       sigma2_plugin = s2_plug, sigma2_unbiased = s2_unb, K_lon = K_lon, K_lat = K_lat, K_t = K_t)
}
plot_weights_raw_vs_shrunk <- function(argo, box, K_lon, K_lat, K_t, between_float = FALSE) {
  sh <- shrink_weights(argo, box, K_lon, K_lat, K_t, between_float)
  Araw <- array(sh$w_raw, c(K_lon, K_lat, K_t)); Ashr <- array(sh$w_shrunk, c(K_lon, K_lat, K_t))
  el <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1); lon_mid <- (el[-1]+el[-(K_lon+1)])/2
  sl <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  lat_mid <- asin((sl[-1]+sl[-(K_lat+1)])/2) * 180 / pi
  et <- as.Date(seq(box$t[1], box$t[2], length.out = K_t + 1), origin = "1970-01-01")
  cap <- max(2, ceiling(max(sh$w_raw))); ds <- div_scale(cap)
  panel <- function(A, k, lab) {
    image(lon_mid, lat_mid, pmin(A[,,k], cap), col = ds$cols, breaks = ds$breaks,
          xlab = "lon", ylab = if (k == 1) "lat" else "", main = lab)
    contour(lon_mid, lat_mid, A[,,k], levels = 1, add = TRUE, lwd = 1.3, lty = 2)
    text(rep(lon_mid, K_lat), rep(lat_mid, each = K_lon), sprintf("%.1f", as.vector(A[,,k])), cex = 0.5)
  }
  op <- par(mfrow = c(2, K_t), mar = c(4, 3.6, 3, 1), oma = c(0, 2.2, 2.4, 0)); on.exit(par(op))
  for (k in seq_len(K_t)) panel(Araw, k, sprintf("RAW  %s..%s", et[k], et[k+1]))
  for (k in seq_len(K_t)) panel(Ashr, k, sprintf("SHRUNK  s=%.2f", sh$s))
  mtext(sprintf("raw plug-in weights (top; disp %.2f)  vs  shrunk to unbiased disp %.2f (bottom)",
                sh$sigma2_plugin, max(0, sh$sigma2_unbiased)), outer = TRUE, cex = 1.0, font = 2)
  invisible(sh)
}

## ---- 11. Float-level permutation test of uniform sampling -----------
## H0: pi = nu (uniform sampling).  The textbook chi^2_{K-1} reference is
## invalid here (Argo profiles are cluster-dependent and cells sparse),
## so we build the null by TOROIDAL SHIFT of whole floats in nu-space
## (lon uniform, sin(lat) uniform, t uniform): each float keeps its
## internal shape but is placed independently and uniformly.  This
## preserves the within-float clustering under H0, so the permutation
## null carries the correct over-dispersion.  Statistics: Pearson X^2 and
## the between-float estimator (centered at 0 under H0).  One-sided:
## preferential sampling => excess coincidences => larger statistic.
permute_toroidal <- function(argo, box, shift_time = TRUE) {
  s1 <- sin(box$lat[1]*pi/180); s2 <- sin(box$lat[2]*pi/180)
  u_lon <- (argo$lon - box$lon[1]) / diff(box$lon)
  u_s   <- (sin(argo$lat*pi/180) - s1) / (s2 - s1)
  u_t   <- (argo$t - box$t[1]) / diff(box$t)
  fl <- argo$float_id
  for (f in unique(fl)) {
    i <- fl == f
    u_lon[i] <- (u_lon[i] + runif(1)) %% 1
    u_s[i]   <- (u_s[i]   + runif(1)) %% 1
    if (shift_time) u_t[i] <- (u_t[i] + runif(1)) %% 1
  }
  argo$lon <- box$lon[1] + u_lon * diff(box$lon)
  argo$lat <- asin(s1 + u_s * (s2 - s1)) * 180 / pi
  argo$t   <- box$t[1] + u_t * diff(box$t)
  argo
}
uniformity_test <- function(argo, box, K_lon, K_lat, K_t,
                            n_perm = 999, shift_time = TRUE, seed = 1) {
  set.seed(seed); K <- K_lon * K_lat * K_t
  stat <- function(a) {
    cell <- assign_cells(a$lon, a$lat, a$t, box, K_lon, K_lat, K_t)
    ok <- !is.na(cell); fl <- a$float_id[ok]; cell <- cell[ok]; n <- length(cell)
    nj <- tabulate(cell, K)
    c(X2 = sum((nj - n/K)^2 / (n/K)), Tbf = sigma_wK2_hat_bf(cell, fl, K))
  }
  obs  <- stat(argo)
  null <- t(vapply(seq_len(n_perm),
                   function(b) stat(permute_toroidal(argo, box, shift_time)), numeric(2)))
  data.frame(
    statistic = c("Pearson X2", "between-float sigma^2"),
    observed  = c(obs["X2"], obs["Tbf"]),
    null_mean = c(mean(null[,"X2"]), mean(null[,"Tbf"])),
    p_perm    = c((1 + sum(null[,"X2"]  >= obs["X2"]))  / (n_perm + 1),
                  (1 + sum(null[,"Tbf"] >= obs["Tbf"])) / (n_perm + 1)),
    p_naive_chisq = c(pchisq(obs["X2"], K - 1, lower.tail = FALSE), NA_real_),
    row.names = NULL)
}

## =====================================================================
## SELF-TEST  (set RUN_SELFTEST <- FALSE before source() to silence)
## =====================================================================
if (!exists("RUN_SELFTEST")) RUN_SELFTEST <- TRUE
if (RUN_SELFTEST) {
  cat("== Test A: unbiasedness under the multinomial (all-iid) model ==\n")
  set.seed(42); K <- 24; n <- 364
  w <- runif(K, 0.3, 3); pvec <- w / sum(w); true_s2 <- K * sum(pvec^2) - 1
  dv <- pv <- numeric(6000)
  for (r in seq_along(dv)) { ce <- sample.int(K, n, TRUE, pvec)
    dv[r] <- sigma_wK2_hat(ce, K); pv[r] <- sigma_wK2_plugin(ce, K) }
  cat(sprintf("  true=%.4f  mean(diag)=%.4f  mean(plug-in)=%.4f\n",
              true_s2, mean(dv), mean(pv)))

  cat("\n== Test D: between-float estimator stays unbiased under within-float clustering ==\n")
  ## each float lands in ONE cell (~pi) then puts ALL its profiles there:
  ## maximal within-float clustering.  Standard is biased high; between-float is not.
  set.seed(11); nfl <- 40
  st_std <- st_bf <- numeric(3000)
  for (r in seq_along(st_std)) {
    d_f <- sample(3:12, nfl, TRUE)                         # profiles per float
    home <- sample.int(K, nfl, TRUE, pvec)                 # one cell per float
    cell <- rep(home, d_f); fl <- rep(seq_len(nfl), d_f)
    st_std[r] <- sigma_wK2_hat(cell, K)
    st_bf[r]  <- sigma_wK2_hat_bf(cell, fl, K)
  }
  cat(sprintf("  true=%.4f  mean(standard)=%.4f (biased high)  mean(between-float)=%.4f\n",
              true_s2, mean(st_std), mean(st_bf)))

  cat("\n== Test C: full pipeline on synthetic floats (all profiles, jackknife CI) ==\n")
  set.seed(123)
  box <- list(lon = c(-68,-62), lat = c(36,40), t = c(0,1095))
  gen <- function(nfl = 40, per = 9) {
    hl <- box$lon[1] + diff(box$lon) * rbeta(nfl, 1.2, 2.6)
    hb <- box$lat[1] + diff(box$lat) * rbeta(nfl, 1.5, 1.5)
    do.call(rbind, lapply(seq_len(nfl), function(i) {
      x <- pmin(pmax(hl[i] + cumsum(rnorm(per,0,.30)), box$lon[1]), box$lon[2])
      y <- pmin(pmax(hb[i] + cumsum(rnorm(per,0,.25)), box$lat[1]), box$lat[2])
      data.frame(lon = x, lat = y, t = sort(runif(per,0,1095)), float_id = paste0("F",i))
    }))
  }
  argo <- gen()
  grids <- grid_from_cellsize(box, c(110, 90, 75), K_t = 3)
  print(round(refinement_curve(argo, box, grids)[, c("cell_km","K","n","sigma2_hat",
        "se_sigma2","sigma_hat","sigma_lo","sigma_hi")], 3))

  ## ---- real-data demonstration (runs if the clean csv is present) ----
  csv <- "argo_nac_clean.csv"
  if (file.exists(csv)) {
    a0 <- read.csv(csv)
    a  <- thin_min_gap(a0, min_gap = 9)                  # drop intra-cycle repeats
    box <- list(lon = c(-68,-62), lat = c(36,40), t = range(a$t))
    cat(sprintf("\n== REAL DATA: %d profiles -> %d after dropping dt<9d  (%d floats) ==\n",
                nrow(a0), nrow(a), length(unique(a$float_id))))
    grids <- grid_from_cellsize(box, c(150, 110, 90, 75), K_t = 3)
    cat("\n[standard] cleaned profiles, float jackknife CI:\n")
    print(round(refinement_curve(a, box, grids)[, c("cell_km","K","n","n_float",
          "sigma2_hat","se_sigma2","sigma_hat","sigma_lo","sigma_hi")], 3))
    cat("\n[between-float-only]:\n")
    print(round(refinement_curve(a, box, grids, between_float = TRUE)[, c("cell_km","K",
          "sigma2_hat","se_sigma2","sigma_hat","sigma_lo","sigma_hi")], 3))
    cat("\nconsecutive-cell check after cleaning:\n")
    print(round(consecutive_check_curve(a, box, grids), 3))
    cat("\nper-cell weight map (coarse 4x3 spatial, aggregated over time):\n")
    print(round(cell_weights_spatial(a, box, 4, 3)$w, 2))
    png("cell_weight_map.png", width = 640, height = 470, res = 110)
    par(mar = c(4,4,3,1)); plot_cell_weights(a, box, 4, 3); dev.off()
    cat("saved cell_weight_map.png\n")
  }
}
## =====================================================================
## USER PIPELINE (runs only in your environment: needs data.table + data)
## =====================================================================
if (file.exists("data/argo_ohc.csv") && requireNamespace("data.table", quietly = TRUE)) {
  library(data.table)
  sec_to_day <- function(d) d / (60 * 60 * 24)
  prep_argo_data <- function(file_path, lat_bounds, lon_bounds, years, dt_range = NULL) {
    argo <- fread(file_path)
    argo[, year := year(date)]
    argo[, t := sec_to_day(as.numeric(date))]
    setkey(argo, float_id, t)
    argo[, float_obs := 1:.N, float_id]
    argo <- argo[(lat %between% lat_bounds) & (lon %between% lon_bounds) & (year %in% years), ]
    argo[, float_obs_a := 1:.N, float_id]
    argo[, dt := c(NA, diff(t)), float_id]
    if (!is.null(dt_range)) argo <- argo[is.na(dt) | (dt %between% dt_range)]
    argo[, float_obs_at := 1:.N, float_id]
    argo[, c("float_id","float_obs","float_obs_a","float_obs_at","lon","lat","t","dt")]
  }
  LAT_BNDS <- c(36, 40); LON_BNDS <- c(-68, -62)
  T_BNDS <- as.numeric(as.POSIXct(c("2020-01-01 00:00:00","2022-12-31 23:59:59"), tz = "UTC"))
  YEARS  <- c(2020, 2021, 2022)
  argo <- prep_argo_data("data/argo_ohc.csv", LAT_BNDS, LON_BNDS, seq(1998,2026,1))
  box  <- list(lon = LON_BNDS, lat = LAT_BNDS, t = sec_to_day(T_BNDS))
  grids <- grid_from_cellsize(box, c(200, 110, 90, 50), K_t = 3)
  print(refinement_curve(argo, box, grids, thin = FALSE))
}
