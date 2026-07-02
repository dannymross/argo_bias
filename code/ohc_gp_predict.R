# Predict a Vecchia-approximated GP (fitted by ohc_gp_fit.R) onto a grid, one
# calendar month at a time, via a custom Vecchia kriging predictor giving both
# a mean and a standard error.
#
# Usage (from the repo root):
#   Rscript code/ohc_gp_predict.R <profiles.csv> <pred_grid.csv> <model_cache.rds> <out.csv> [m|exact] [first|middle|last]
#
# profiles.csv     columns: date, lon, lat, ohc_700[_anom], ohc_2000[_anom] --
#                   only used here for the list of observed months and each
#                   month's profile count, not refit (see ohc_gp_fit.R).
# pred_grid.csv    columns: lon, lat            (static prediction grid, no month)
# model_cache.rds  from ohc_gp_fit.R
# out.csv          columns: month, lon, lat, ohc_700[_anom]_pred, ohc_700[_anom]_se,
#                            ohc_2000[_anom]_pred, ohc_2000[_anom]_se,
#                            too_few_profiles, n_profiles
#
# The `m` argument controls exact vs Vecchia-approximate prediction:
# m="exact" (default) conditions every prediction point on *all* observations
# -- correct, but doesn't scale past a few hundred pooled profiles. Pass a
# small integer (e.g. 30, matching ohc_gp_fit.R's own m_seq) for a real
# Vecchia approximation instead: same find_ordered_nn/vecchia_Linv machinery
# with a bounded neighbour set (each prediction point's own via
# FNN::get.knnx), so it stays fast and still returns a valid SE -- unlike
# GpGp::predictions(), which has no SE at all.

suppressMessages(library(data.table))
suppressMessages(library(GpGp))
suppressMessages(library(FNN))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: Rscript code/ohc_gp_predict.R <profiles.csv> <pred_grid.csv> <model_cache.rds> <out.csv> [m|exact] [first|middle|last]")
}
profiles_path    <- args[1]
pred_grid_path   <- args[2]
model_cache_path <- args[3]
out_path         <- args[4]
# Optional args 5+: detect by value so m and month_day can appear in either order.
m_arg     <- "exact"
month_day <- "middle"
for (opt in args[seq_len(max(0, length(args) - 4)) + 4]) {
  if (opt %in% c("first", "middle", "last")) month_day <- opt
  else m_arg <- opt
}
exact <- identical(m_arg, "exact")

profiles <- fread(profiles_path)
grid <- fread(pred_grid_path)

cached <- readRDS(model_cache_path)
fits <- cached$fits
too_few_overall <- cached$too_few_overall
depth_cols <- cached$depth_cols

EPOCH <- as.Date("2020-01-01")
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

# GpGp matern_spheretime's internal metric: (lon,lat) -> 3D unit-sphere coords
# scaled by the fitted spatial range, day scaled by the fitted temporal range
# -- Euclidean distance in this space is what find_ordered_nn/vecchia_Linv use
# internally (see also gp_audit_fields.R's matern_st_corr, which reproduces
# this same transform independently for its own audit).
to_scaled_space <- function(locs, rs, rt) {
  lon_r <- locs[, 1] * pi / 180
  lat_r <- locs[, 2] * pi / 180
  s <- cbind(cos(lat_r) * cos(lon_r), cos(lat_r) * sin(lon_r), sin(lat_r)) / rs
  t <- locs[, 3] / rt
  cbind(s, t)
}

# Vecchia kriging mean + SE at locs_pred, conditioned on the observation
# neighbour structure NNarray_obs (hoisted once per depth, below). Two modes:
#   - exact: every point conditions on every observation -- the neighbour
#     indices don't depend on locs_pred, so both blocks are reused unchanged
#     across all months.
#   - bounded: each prediction point's own m nearest *observation* neighbours
#     come from FNN::get.knnx (query-vs-reference) rather than
#     find_ordered_nn on the combined obs+pred array, which would let a
#     prediction point neighbour other prediction points (no known y, so
#     indexing y_obs with them silently returns NA) whenever those are
#     closer than any real observation -- true for most points on a dense
#     regular grid.
kriging_predict <- function(fit, locs_pred, NNarray_obs, exact) {
  y_obs <- fit$y
  locs_obs <- as.matrix(fit$locs)
  beta <- fit$betahat
  covparms <- fit$covparms
  covfun_name <- fit$covfun_name
  n_obs <- nrow(locs_obs)
  n_pred <- nrow(locs_pred)

  locs_all <- rbind(locs_obs, locs_pred)
  inds2 <- (n_obs + 1):(n_obs + n_pred)

  if (exact) {
    NNarray_all <- matrix(NA_integer_, n_obs + n_pred, n_obs + 1)
    NNarray_all[1:n_obs, 1:ncol(NNarray_obs)] <- NNarray_obs
    NNarray_all[inds2, 1] <- inds2
    NNarray_all[inds2, 2:(n_obs + 1)] <- matrix(rep(1:n_obs, each = n_pred), nrow = n_pred)
  } else {
    m <- ncol(NNarray_obs) - 1
    obs_scaled  <- to_scaled_space(locs_obs, covparms[2], covparms[3])
    pred_scaled <- to_scaled_space(locs_pred, covparms[2], covparms[3])
    nn <- FNN::get.knnx(obs_scaled, pred_scaled, k = m)$nn.index

    NNarray_all <- matrix(NA_integer_, n_obs + n_pred, m + 1)
    NNarray_all[1:n_obs, ] <- NNarray_obs
    NNarray_all[inds2, 1] <- inds2
    NNarray_all[inds2, 2:(m + 1)] <- nn
  }

  Linv_all <- vecchia_Linv(covparms, covfun_name, locs_all, NNarray_all, n_obs + 1)

  diag_val <- Linv_all[inds2, 1]
  nbr_idx <- NNarray_all[inds2, 2:ncol(NNarray_all), drop = FALSE]
  # b = -Linv_off_diag / Linv_diag (the kriging weights), applied row-wise.
  b <- -Linv_all[inds2, 2:ncol(NNarray_all), drop = FALSE] / diag_val
  resid <- matrix(y_obs[nbr_idx] - beta, nrow = n_pred, ncol = ncol(nbr_idx))
  mean_out <- beta + rowSums(b * resid)
  list(mean = mean_out, se = sqrt(1 / diag_val^2))
}

# Hoist the observation-to-observation neighbour structure once per depth,
# before the month loop -- it's month-invariant, so recomputing it 36 times
# (the previous behaviour) was pure waste. Reused as the full neighbour
# structure in exact mode, and as the obs-block half in bounded mode (each
# prediction point's own neighbours still come from a fresh FNN::get.knnx
# call per month, since those depend on locs_pred).
NNarray_obs_by_col <- list()
for (col in depth_cols) {
  if (too_few_overall[[col]]) next
  fit <- fits[[col]]
  locs_obs <- as.matrix(fit$locs)
  n_obs <- nrow(locs_obs)
  st_scale <- fit$covparms[2:3]
  m <- if (exact) n_obs - 1 else as.integer(m_arg)
  # See kriging_predict's header note re: the GpGp C++ stdout print this
  # triggers for every lonlat covariance call, not just this one.
  invisible(capture.output({
    NNarray_obs_by_col[[col]] <- find_ordered_nn(locs_obs, m = m, lonlat = TRUE, st_scale = st_scale)
  }))
}

rows <- vector("list", length(months_first))
for (i in seq_along(months_first)) {
  mo <- months_first[i]
  day_pred <- month_pred_day[[mo]]
  out_row <- list(month = mo, lon = grid$lon, lat = grid$lat)
  any_too_few <- FALSE
  for (col in depth_cols) {
    if (too_few_overall[[col]]) {
      pred <- rep(0, nrow(grid))
      se <- rep(NA_real_, nrow(grid))
      any_too_few <- TRUE
    } else {
      fit <- fits[[col]]
      locs_pred <- cbind(grid$lon, grid$lat, rep(day_pred, nrow(grid)))
      invisible(capture.output({
        out <- kriging_predict(
          fit = fit, locs_pred = locs_pred,
          NNarray_obs = NNarray_obs_by_col[[col]], exact = exact
        )
      }))
      pred <- out$mean
      se <- out$se
    }
    out_row[[paste0(col, "_pred")]] <- pred
    out_row[[paste0(col, "_se")]] <- se
  }
  out_row$too_few_profiles <- any_too_few
  out_row$n_profiles <- sum(is.finite(profiles[month_str == mo][[depth_cols[1]]]))
  rows[[i]] <- as.data.table(out_row)
}

out <- rbindlist(rows)
fwrite(out, out_path)
cat("wrote", nrow(out), "rows ->", out_path, "\n")
