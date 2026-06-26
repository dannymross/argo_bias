# Vecchia-approximated Gaussian-process interpolation of synthetic-Argo OHC
# anomalies onto the native analysis grid, via GpGp::fit_model.
#
# Input is per-profile OHC *anomalies* (profile OHC minus the RG climatological
# seasonal-mean OHC for that location/month) -- see code/ohc_climatology.py for
# how the anomaly is constructed. Because the seasonal cycle is already removed,
# different calendar months' anomalies are treated as (not necessarily
# independent) draws from one underlying spatio-temporal process, rather than
# fitting a separate spatial GP per month.
#
# One spatio-temporal GP per depth, pooling all ~150 profiles across the year
# (`matern_spheretime`, locs = (lon, lat, calendar_month)). This replaces an
# earlier per-month design: fitting matern_sphere independently on each
# month's 3-41 profiles let the (range, smoothness) MLE run away to wildly
# implausible values in roughly half the months (e.g. a fitted spatial range
# >100x the analysis box's own size, or =0) -- the classic small-sample
# range/smoothness confounding problem in Matern covariance estimation. The
# visible symptom was a near-uniform SE map for those months (degenerate range
# means the whole 4x6 deg box looks "equally correlated" to the fit, killing the
# expected close-to-data/far-from-data SE pattern). Pooling across the year
# fixes this by giving the covariance-shape estimate ~150 points instead of a
# handful; the temporal range comes out of the fit too (~0.13 calendar months in
# practice), rather than being assumed -- if different months turned out to
# carry real information about each other, this model would show it; here it
# mostly doesn't, validating the original "treat months independently" instinct
# while still estimating the spatial shape far more stably than per-month fits
# could.
#
# Mean and SE: GpGp::predictions()'s strategy -- combine obs+pred into one
# Vecchia ordering and let later prediction points condition on *earlier
# prediction points* -- is fine when there's enough data, but breaks down once
# data is sparse relative to the (much denser) prediction grid: prediction
# points end up conditioning almost entirely on each other, not on data,
# producing a salt-and-pepper SE (and, faintly, mean) artifact. `kriging_predict`
# avoids this by conditioning every prediction point *independently* on all
# n_obs observations (pooled across the year, so n_obs is a few hundred --
# cheap for an exact, non-approximated GP) -- exact ordinary kriging, computed
# via GpGp's Vecchia machinery (vecchia_Linv) purely as the computational
# engine. A single vecchia_Linv call's output row for a prediction point i is
# [1/sigma_i, -b_1/sigma_i, ..., -b_n/sigma_i], where b = K_obs^-1 k_0 are the
# kriging weights and sigma_i is the conditional SD -- so both the mean
# (beta + b^T(y_obs - beta)) and the SE (sigma_i) come straight off that one
# row, with no need for Linv_mult/L_mult. Validated against a brute-force
# dense-GP posterior mean/variance for matern_spheretime -- matches to ~1e-15.
#
# `<col>_pred_gpgp` is GpGp::predictions()'s own mean (same pooled fit), kept
# alongside as a reference/comparison column (not used elsewhere in this
# project).
#
# Smoothness is fixed at 1.5 (a conventional Matern choice) rather than
# estimated jointly: letting it float, the MLE doesn't converge at all (Fisher
# scoring still hadn't met its tolerance after 100+ iterations, and pushing
# further triggers a Bessel-function overflow as smoothness drifts toward
# infinity) -- smoothness and range are notoriously confounded, and unlike the
# spatial range, pooling more data across months didn't fix this one. Fixing
# it gives a properly converged fit (small gradient, well-conditioned Fisher
# information) for the other four parameters, which is what the "Fit summary"
# tables in the report are built from.
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
  stop("usage: Rscript code/ohc_gp_interp.R <profiles.csv> <pred_grid.csv> <out.csv> <fit_summary.csv> [model_cache.rds]")
}
profiles_path    <- args[1]
pred_grid_path   <- args[2]
out_path         <- args[3]
fit_summary_path <- args[4]
model_cache_path <- if (length(args) >= 5) args[5] else NULL

FIXED_SMOOTHNESS <- 1.5

profiles <- fread(profiles_path)
grid <- fread(pred_grid_path)

depth_cols <- intersect(c("ohc_700_anom", "ohc_2000_anom"), names(profiles))
# Day-of-year (1-366) as the temporal GP coordinate -- finer-grained than
# calendar month (1-12) and lets each profile's actual observation day inform
# the temporal covariance directly. The anomaly subtracted the climatological
# mean for the calendar month of each profile (see ohc_climatology.py), so
# the GP still sees de-seasonalized residuals.
profiles[, day_num := as.integer(format(as.Date(date), "%j"))]
profiles[, month_str := format(as.Date(date), "%Y-%m-01")]
months_first <- sort(unique(profiles$month_str))
# Last day of each calendar month as the prediction temporal coordinate so
# predictions land at the end of each month (matching the report's monthly maps).
last_day_yday <- function(first_of_month) {
  d <- as.Date(first_of_month)
  as.integer(format(as.Date(format(d + 32, "%Y-%m-01")) - 1, "%j"))
}
last_day_of <- setNames(vapply(months_first, last_day_yday, integer(1)), months_first)

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
  se_nat[free_idx] <- fit$covparms[free_idx] * se_log_free

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
  day_pred <- last_day_of[[mo]]
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
  cat(sprintf(
    "month %s (prediction day-of-year=%d): predicted %d depth(s), n_profiles_this_month=%d\n",
    mo, day_pred, length(depth_cols), out_row$n_profiles
  ))
}

out <- rbindlist(rows)
fwrite(out, out_path)
cat("wrote", nrow(out), "rows ->", out_path, "\n")
