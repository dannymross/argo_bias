# Levitus et al. (2012)-style objective-analysis interpolation of OHC anomalies.
#
# By default this script operates per calendar month -- the literal formula
# (r_n is purely spatial, no time term) does not pool across months the way
# the pooled spatio-temporal GP does. Pass "pooled" as a 5th argument to pool
# every month's profiles into one fixed set instead (reasonable precisely
# because these are de-seasonalized anomalies, not raw OHC): one static,
# time-invariant prediction per grid point, with no month dimension at all,
# using every profile from the whole year regardless of when it was taken.
#
# Usage (from the repo root):
#   Rscript code/ohc_levitus_interp.R <profiles.csv> <pred_grid.csv> <out.csv> <R_km> [pooled]
#
# profiles.csv  columns: month, lon, lat, ohc_700_anom, ohc_2000_anom
# pred_grid.csv columns: lon, lat            (static prediction grid, no month)
# out.csv (monthly, default) columns:
#   month, lon, lat, ohc_700_anom_pred, ohc_700_anom_se_a, ohc_700_anom_se_0,
#   ohc_700_anom_n_obs_in_radius, ohc_2000_anom_pred, ohc_2000_anom_se_a,
#   ohc_2000_anom_se_0, ohc_2000_anom_n_obs_in_radius, too_few_profiles
# out.csv (pooled) columns: same, minus `month` (one row per grid point).

suppressMessages(library(data.table))
suppressMessages(library(fields))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4 || length(args) > 5) {
  stop("usage: Rscript code/ohc_levitus_interp.R <profiles.csv> <pred_grid.csv> <out.csv> <R_km> [pooled]")
}
profiles_path <- args[1]
pred_grid_path <- args[2]
out_path <- args[3]
R_KM <- as.numeric(args[4])
POOLED <- length(args) == 5 && identical(tolower(args[5]), "pooled")

E <- 4.0
MIN_OBS_FOR_SE <- 2

# Gaussian-shaped weight w = exp(-E (r/R)^2) for r<=R, else 0.
levitus_weights <- function(r_km, R, e = E) {
  w <- exp(-e * (r_km / R)^2)
  ifelse(r_km <= R, w, 0)
}

# Levitus-style kernel-smoother prediction, vectorized over every prediction
# point at once (one (n_pred x n_obs) matrix of weights, reduced by row).
# Returns a list of length-n_pred vectors: mean, se_a, se_0, n_obs_in_radius.
# mean is NA wherever zero observations fall in the radius (or the single
# observation's own value if exactly one does -- a well-defined, if
# unweighted-by-anything-else, prediction); both SEs are NA below
# MIN_OBS_FOR_SE (sigma_0's sample std needs >=2 points).
levitus_predict <- function(lon_obs, lat_obs, q_obs, lon_pred, lat_pred, R) {
  n_obs <- length(lon_obs)
  n_pred <- length(lon_pred)

  # Great-circle distance (km), as one (n_pred x n_obs) matrix -- same
  # fields::rdist.earth call ohc_gp_predict.R's kriging_predict uses for its
  # own neighbour distances.
  r <- fields::rdist.earth(cbind(lon_pred, lat_pred), cbind(lon_obs, lat_obs), miles = FALSE)
  in_radius <- r <= R
  w <- levitus_weights(r, R = R)

  n_in <- rowSums(in_radius)
  W <- rowSums(w)
  Qmat <- matrix(q_obs, n_pred, n_obs, byrow = TRUE)
  lam <- w / W # recycles W (length n_pred) down columns -- divides each row by its own sum.
  mean_pred <- rowSums(w * Qmat) / W

  # sigma_0 (source doc eq. 6): sample std (N-1) of the *corrections*
  # C_n = w_n*Q_n/W = lambda_n*Q_n -- not of the raw Q_n -- across the N
  # ODSQs within each point's radius. See header note.
  Cmat <- Qmat * lam
  Cmat_masked <- ifelse(in_radius, Cmat, NA_real_)
  Cbar <- rowMeans(Cmat_masked, na.rm = TRUE)
  C_sq_dev <- ifelse(in_radius, (Cmat_masked - Cbar)^2, 0)
  sigma0 <- sqrt(rowSums(C_sq_dev) / (n_in - 1))

  sum_lam2 <- rowSums(lam^2)
  se_a <- sigma0 * sqrt(sum_lam2)
  se_0 <- sigma0

  enough_for_se <- n_in >= MIN_OBS_FOR_SE
  se_a[!enough_for_se] <- NA_real_
  se_0[!enough_for_se] <- NA_real_
  mean_pred[n_in < 1] <- NA_real_

  list(mean = mean_pred, se_a = se_a, se_0 = se_0, n_obs_in_radius = n_in)
}

profiles <- fread(profiles_path)
grid <- fread(pred_grid_path)
depth_cols <- intersect(c("ohc_700_anom", "ohc_2000_anom"), names(profiles))

if (POOLED) {
  # Every profile from the whole year, regardless of month -- one static
  # prediction per grid point (see the header note above).
  out_row <- list(lon = grid$lon, lat = grid$lat)
  any_too_few <- FALSE
  for (col in depth_cols) {
    y <- profiles[[col]]
    keep <- is.finite(y)
    n_obs <- sum(keep)
    any_too_few <- any_too_few || (n_obs < 2)
    res <- levitus_predict(profiles$lon[keep], profiles$lat[keep], y[keep], grid$lon, grid$lat, R = R_KM)
    out_row[[paste0(col, "_pred")]] <- res$mean
    out_row[[paste0(col, "_se_a")]] <- res$se_a
    out_row[[paste0(col, "_se_0")]] <- res$se_0
    out_row[[paste0(col, "_n_obs_in_radius")]] <- res$n_obs_in_radius
  }
  out_row$too_few_profiles <- any_too_few
  out <- as.data.table(out_row)
  cat(sprintf("pooled: predicted %d depth(s) from n=%d profiles (whole year)\n", length(depth_cols), nrow(profiles)))
} else {
  months <- sort(unique(profiles$month))
  rows <- vector("list", length(months))
  for (i in seq_along(months)) {
    mo <- months[i]
    sub <- profiles[month == mo]
    out_row <- list(month = mo, lon = grid$lon, lat = grid$lat)
    any_too_few <- FALSE
    for (col in depth_cols) {
      y <- sub[[col]]
      keep <- is.finite(y)
      n_obs <- sum(keep)
      any_too_few <- any_too_few || (n_obs < 2)
      res <- levitus_predict(sub$lon[keep], sub$lat[keep], y[keep], grid$lon, grid$lat, R = R_KM)
      out_row[[paste0(col, "_pred")]] <- res$mean
      out_row[[paste0(col, "_se_a")]] <- res$se_a
      out_row[[paste0(col, "_se_0")]] <- res$se_0
      out_row[[paste0(col, "_n_obs_in_radius")]] <- res$n_obs_in_radius
    }
    out_row$too_few_profiles <- any_too_few
    rows[[i]] <- as.data.table(out_row)
    cat(sprintf("month %s: predicted %d depth(s)\n", mo, length(depth_cols)))
  }
  out <- rbindlist(rows)
}

fwrite(out, out_path)
cat("wrote", nrow(out), "rows ->", out_path, "\n")
