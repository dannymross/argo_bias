# Vecchia-approximated Gaussian-process interpolation of synthetic-Argo OHC
# anomalies onto the native analysis grid, via GpGp::fit_model.
#
# Input is per-profile OHC *anomalies* (profile OHC minus the RG climatological
# seasonal-mean OHC for that location/month) -- see code/ohc_climatology.py for
# how the anomaly is constructed. 
#
# Usage (from the repo root):
#   Rscript code/ohc_gp_interp.R <profiles.csv> <pred_grid.csv> <out.csv> <fit_summary.csv>
#
# profiles.csv  columns: date, lon, lat, ohc_700_anom, ohc_2000_anom
# pred_grid.csv columns: lon, lat            (static prediction grid, no month)
# out.csv       columns: month, lon, lat, ohc_700_anom_pred, ohc_700_anom_se,
#                         ohc_700_anom_pred_gpgp, ohc_2000_anom_pred,
#                         ohc_2000_anom_se, ohc_2000_anom_pred_gpgp,
#                         too_few_profiles
# fit_summary.csv columns: depth, parameter, estimate, std_error, z_stat

suppressMessages(library(data.table))
suppressMessages(library(GpGp))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: Rscript code/ohc_gp_interp.R <profiles.csv> <pred_grid.csv> <out.csv> <fit_summary.csv> [model_cache.rds] [first|middle|last]")
}
profiles_path <- args[1]
pred_grid_path <- args[2]
out_path <- args[3]
fit_summary_path <- args[4]
# Optional args 5+: detect by value so model_cache and month_day can appear in
# either order.
month_day        <- "middle"
model_cache_path <- NULL
for (opt in args[seq_len(max(0, length(args) - 4)) + 4]) {
  if (opt %in% c("first", "middle", "last")) month_day <- opt
  else model_cache_path <- opt
}

FIXED_SMOOTHNESS <- 0.5

profiles <- fread(profiles_path)
grid <- fread(pred_grid_path)

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
# multiple years, when profiles cover more than one year (as here, pooling
# 2020-2022).
EPOCH <- as.Date("2020-01-01")
profiles[, day_num := as.integer(as.Date(date) - EPOCH)]
profiles[, month_str := format(as.Date(date), "%Y-%m-01")]
months_first <- sort(unique(profiles$month_str))

pred_day <- function(first_of_month, position) {
  d <- as.Date(first_of_month)
  target <- switch(position,
    first  = d,
    middle = as.Date(format(d, "%Y-%m-15")),
    last   = as.Date(format(d + 32, "%Y-%m-01")) - 1
  )
  as.integer(target - EPOCH)
}
month_pred_day <- setNames(
  vapply(months_first, pred_day, integer(1), position = month_day),
  months_first
)

# Exact kriging mean + SE at locs_pred, every prediction point conditioned
# independently on *all* n_obs observations (pooled across the year -- see the
# header note). One vecchia_Linv call's row for a prediction point already
# encodes both the kriging weights and the conditional SD (see the header
# note), so neither the mean nor the SE needs Linv_mult/L_mult.
kriging_predict <- function(fit, locs_pred) {
  y_obs <- fit$y
  locs_obs <- as.matrix(fit$locs)
  beta <- fit$betahat
  covparms <- fit$covparms
  covfun_name <- fit$covfun_name
  n_obs <- nrow(locs_obs)
  n_pred <- nrow(locs_pred)

  locs_all <- rbind(locs_obs, locs_pred)
  inds2 <- (n_obs + 1):(n_obs + n_pred)

  # Full conditioning: every prediction row's neighbour set is *all* n_obs
  # observations, in any order -- with n_obs in the low hundreds (pooled across
  # the year) an exact GP is cheap, so there's no need to rank/select a nearest
  # subset at all. st_scale (fitted spatial/temporal ranges) only matters for
  # the obs-block placeholder rows below, whose values are never read.
  st_scale <- covparms[2:3]
  NNarray_obs <- find_ordered_nn(locs_obs, m = n_obs - 1, lonlat = TRUE, st_scale = st_scale)
  NNarray_all <- matrix(NA_integer_, n_obs + n_pred, n_obs + 1)
  NNarray_all[1:n_obs, 1:ncol(NNarray_obs)] <- NNarray_obs
  NNarray_all[inds2, 1] <- inds2
  NNarray_all[inds2, 2:(n_obs + 1)] <- matrix(rep(1:n_obs, each = n_pred), nrow = n_pred)

  Linv_all <- vecchia_Linv(covparms, covfun_name, locs_all, NNarray_all, n_obs + 1)

  diag_val <- Linv_all[inds2, 1]
  nbr_idx <- NNarray_all[inds2, 2:(n_obs + 1), drop = FALSE]
  # b = -Linv_off_diag / Linv_diag (the kriging weights), applied row-wise.
  b <- -Linv_all[inds2, 2:(n_obs + 1), drop = FALSE] / diag_val
  resid <- matrix(y_obs[nbr_idx] - beta, nrow = n_pred, ncol = n_obs)
  mean_out <- beta + rowSums(b * resid)
  list(mean = mean_out, se = sqrt(1 / diag_val^2))
}


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

X_pred <- matrix(1, nrow = nrow(grid), ncol = 1)

if (!is.null(model_cache_path) && file.exists(model_cache_path)) {
  cached <- readRDS(model_cache_path)
  fits <- cached$fits
  too_few_overall <- cached$too_few_overall
  cat("loaded cached model from", model_cache_path, "\n")
} else {
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
  if (!is.null(model_cache_path)) {
    saveRDS(list(fits = fits, too_few_overall = too_few_overall), model_cache_path)
    cat("saved model cache to", model_cache_path, "\n")
  }
}

fitted_cols <- depth_cols[!vapply(too_few_overall[depth_cols], isTRUE, logical(1))]
fit_summary <- rbindlist(lapply(
  fitted_cols, function(col) fit_summary_table(fits[[col]], depth = col)
))
fwrite(fit_summary, fit_summary_path)
cat("wrote", nrow(fit_summary), "rows ->", fit_summary_path, "\n")

rows <- vector("list", length(months_first))
for (i in seq_along(months_first)) {
  mo <- months_first[i]
  day_pred <- month_pred_day[[mo]]
  out_row <- list(month = mo, lon = grid$lon, lat = grid$lat)
  any_too_few <- FALSE
  for (col in depth_cols) {
    if (too_few_overall[[col]]) {
      pred <- rep(0, nrow(grid))
      pred_gpgp <- pred
      se <- rep(NA_real_, nrow(grid))
      any_too_few <- TRUE
    } else {
      fit <- fits[[col]]
      locs_pred <- cbind(grid$lon, grid$lat, rep(day_pred, nrow(grid)))
      # GpGp prints "Assuming columns 1 and 2 of locs are (longitude,latidue)
      # in degrees" straight to stdout from its C++ internals every time a
      # lonlat covariance is used -- not an R message/warning, so
      # suppressMessages() doesn't catch it. capture.output() does.
      invisible(capture.output({
        out <- kriging_predict(fit = fit, locs_pred = locs_pred)
        pred_gpgp <- as.numeric(GpGp::predictions(fit = fit, locs_pred = locs_pred, X_pred = X_pred))
      }))
      pred <- out$mean
      se <- out$se
    }
    out_row[[paste0(col, "_pred")]] <- pred
    out_row[[paste0(col, "_se")]] <- se
    out_row[[paste0(col, "_pred_gpgp")]] <- pred_gpgp
  }
  out_row$too_few_profiles <- any_too_few
  out_row$n_profiles <- sum(is.finite(profiles[month_str == mo][[depth_cols[1]]]))
  rows[[i]] <- as.data.table(out_row)
  # cat(sprintf(
  #  "month %s (prediction day=%d): predicted %d depth(s), n_profiles_this_month=%d\n",
  #  mo, day_pred, length(depth_cols), out_row$n_profiles
  # ))
}

out <- rbindlist(rows)
fwrite(out, out_path)
cat("wrote", nrow(out), "rows ->", out_path, "\n")
