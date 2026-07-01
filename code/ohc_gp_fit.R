# Fit a Vecchia-approximated Gaussian process (GpGp::fit_model) per depth to
# pooled OHC profiles. Prediction is a separate step -- see
# code/ohc_gp_predict.R -- so a single fit here can be reused across any
# number of prediction grids/resolutions without refitting.
#
# Usage (from the repo root):
#   Rscript code/ohc_gp_fit.R <profiles.csv> <fit_summary.csv> <model_cache.rds>
#
# profiles.csv     columns: date, lon, lat, ohc_700[_anom], ohc_2000[_anom]
# fit_summary.csv  columns: depth, parameter, estimate, std_error, z_stat,
#                            loglik, converged, n_obs
# model_cache.rds  list(fits, too_few_overall, depth_cols) -- read by
#                   ohc_gp_predict.R

suppressMessages(library(data.table))
suppressMessages(library(GpGp))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript code/ohc_gp_fit.R <profiles.csv> <fit_summary.csv> <model_cache.rds>")
}
profiles_path    <- args[1]
fit_summary_path <- args[2]
model_cache_path <- args[3]

FIXED_SMOOTHNESS <- 0.5

profiles <- fread(profiles_path)

# Matches either the anomaly-modeling columns (ohc_700_anom, ohc_2000_anom) or
# the raw-value columns (ohc_700, ohc_2000) written by write_profile_csv's
# suffix="" mode -- everything below is suffix-agnostic (just string-pastes
# onto whatever depth_cols resolves to), so no other change is needed to model
# OHC directly instead of its anomaly.
depth_cols <- intersect(c("ohc_700_anom", "ohc_2000_anom", "ohc_700", "ohc_2000"), names(profiles))

# Continuous day count (not day-of-year) -- day-of-year would wrap every 366
# days and treat e.g. Jan 2020 and Jan 2021 as temporally adjacent, capping
# the fitted temporal range at within-year scale. A continuous count lets the
# temporal_range parameter be estimated freely, including ranges spanning
# multiple years, when profiles cover more than one year.
EPOCH <- as.Date("2020-01-01")
profiles[, day_num := as.integer(as.Date(date) - EPOCH)]

fit_pooled <- function(y, lon, lat, day_num) {
  locs <- cbind(lon, lat, day_num)
  start_parms <- c(var(y), 0.01, 60, FIXED_SMOOTHNESS, 0.2)
  n <- length(y)
  m_seq <- unique(pmin(c(10L, 30L), n - 1L))
  fit_model(y, locs,
    X = NULL, covfun_name = "matern_spheretime", m_seq = m_seq,
    fixed_parms = 4, start_parms = start_parms, silent = TRUE
  )
}

# Tidy parameter table for one depth's fit: the mean (intercept) plus the four
# freely-estimated covariance parameters, each with its asymptotic standard
# error from fit$info (the Fisher information, on the log scale fit_model
# optimizes covariance parameters on -- propagated to natural units via the
# delta method: se(theta) = theta * se(log(theta))). Smoothness has no SE since
# it's fixed, not estimated.
#
# fit$info is occasionally near-exactly singular (reciprocal condition number
# as low as ~1e-45 observed) -- a real numerical artifact of GpGp's unseeded
# neighbour-ordering jitter on some fits, not a bug here. solve() fails outright
# in that case; since the asymptotic SE theory is meaningless right at a
# singularity anyway, NA is the honest answer, not a regularized guess, so we
# catch the error rather than letting it halt the whole script.
fit_summary_table <- function(fit, depth) {
  covparm_names <- c("variance", "spatial_range", "temporal_range", "smoothness", "nugget")
  free_idx <- c(1, 2, 3, 5) # smoothness (4) is fixed
  se_log_free <- tryCatch(
    sqrt(diag(solve(fit$info))),
    error = function(e) rep(NA_real_, length(free_idx))
  )
  se_nat <- rep(NA_real_, 5)
  se_nat[free_idx] <- fit$covparms[free_idx] * se_log_free # delta method

  out <- data.table(
    depth = depth,
    parameter = c("mean (intercept)", covparm_names),
    estimate = c(fit$betahat, fit$covparms),
    std_error = c(fit$sebeta, se_nat)
  )
  out[, z_stat := estimate / std_error]
  out[, loglik := fit$loglik]
  out[, converged := fit$conv]
  out[, n_obs := nrow(fit$locs)]
  out
}

fits <- list()
too_few_overall <- list()
for (col in depth_cols) {
  y <- profiles[[col]]
  keep <- is.finite(y)
  n_obs <- sum(keep)
  too_few_overall[[col]] <- n_obs < 5
  if (!too_few_overall[[col]]) {
    invisible(capture.output({
      fits[[col]] <- fit_pooled(
        y[keep], profiles$lon[keep], profiles$lat[keep], profiles$day_num[keep]
      )
    }))
    cat(sprintf(
      "%s: pooled fit n=%d, covparms=%s\n",
      col, n_obs, paste(signif(fits[[col]]$covparms, 4), collapse = ", ")
    ))
  } else {
    cat(sprintf("%s: only n=%d profiles -- too few\n", col, n_obs))
  }
}
saveRDS(list(fits = fits, too_few_overall = too_few_overall, depth_cols = depth_cols), model_cache_path)
cat("saved model cache to", model_cache_path, "\n")

fitted_cols <- depth_cols[!vapply(too_few_overall[depth_cols], isTRUE, logical(1))]
fit_summary <- rbindlist(lapply(
  fitted_cols, function(col) fit_summary_table(fits[[col]], depth = col)
))
fwrite(fit_summary, fit_summary_path)
cat("wrote", nrow(fit_summary), "rows ->", fit_summary_path, "\n")
