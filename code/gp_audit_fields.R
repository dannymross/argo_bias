#!/usr/bin/env Rscript
# code/gp_audit_fields.R
#
# Audits the custom kriging_predict predictions from ohc_gp_interp.R by
# independently reproducing them with fields::Krig and a hand-coded
# matern_spheretime correlation function that exactly matches GpGp's C++
# implementation.
#
# The GpGp fit parameters are read from the model_cache.rds written by
# ohc_gp_interp.R.  fields::Krig is called with:
#   - lambda     = GpGp nugget (relative nugget: tau^2 / sigma^2)
#   - cov.function = matern_spheretime correlation at GpGp's fitted ranges
#   - m = 1      (intercept-only drift, matching GpGp's X=NULL)
#   - scale.type = "none"  (never rescale lon/lat/time)
#
# A second comparison bypasses fields entirely: it rebuilds the kriging system
# from scratch using R's own Cholesky (chol/backsolve) with the same covariance
# function and GpGp's fitted parameters.  This "direct SK" column should agree
# with GpGp to within floating-point rounding.
#
# NOTE ON STANDARD ERRORS
# GpGp's kriging_predict holds beta fixed at betahat (plug-in simple kriging SE):
#   Var_SK(s0) = sigma2 * (1 - k0' K^{-1} k0)
# fields::predict.se includes the additional variance from estimating beta
# (ordinary kriging / MSPE):
#   Var_OK(s0) = Var_SK(s0) + sigma2 * (1 - 1' K^{-1} k0)^2 / (1' K^{-1} 1)
# Var_OK >= Var_SK always; the gap closes as observations densify.
# The "se_sk" column uses GpGp's sigma2 and the simple-kriging formula, so it
# is directly comparable to "se_gpgp".  The "se_fields" column uses sigma2
# estimated by GLS inside Krig (generally close to but not identical to GpGp's
# ML sigma2) and the ordinary-kriging formula.
#
# Usage:
#   Rscript code/gp_audit_fields.R \
#     <profiles.csv> <pred_grid.csv> <gpgp_out.csv> <model_cache.rds> \
#     [<audit_out.csv>]
#
# Output CSV columns:
#   month, lon, lat, depth,
#   pred_gpgp, pred_sk, pred_fields,          # predictions (J/m2)
#   se_gpgp,   se_sk,   se_fields,             # standard errors (J/m2)
#   pred_sk_diff, pred_fields_diff,            # delta vs pred_gpgp
#   se_sk_ratio,  se_fields_ratio              # ratio vs se_gpgp

suppressMessages({
  library(fields)
  library(data.table)
})

# ---- 0. Args ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4)
  stop(paste(
    "usage: Rscript code/gp_audit_fields.R",
    "<profiles.csv> <pred_grid.csv> <gpgp_out.csv> <model_cache.rds>",
    "[<audit_out.csv>]"
  ))

profiles_path    <- args[1]
pred_grid_path   <- args[2]
gpgp_out_path    <- args[3]
model_cache_path <- args[4]
audit_out_path   <- if (length(args) >= 5) args[5] else
  sub("(\\.csv)?$", "_fields_audit.csv", gpgp_out_path, perl = TRUE)

# ---- 1. Load ---------------------------------------------------------------
profiles  <- fread(profiles_path)
pred_grid <- fread(pred_grid_path)
gpgp_out  <- fread(gpgp_out_path)
cached    <- readRDS(model_cache_path)
fits      <- cached$fits

profiles[, day_num   := as.integer(format(as.Date(date), "%j"))]
profiles[, month_str := format(as.Date(date), "%Y-%m-01")]
months_first <- sort(unique(profiles$month_str))

cat(sprintf("Profiles: %d  |  grid cells: %d  |  months: %d\n",
            nrow(profiles), nrow(pred_grid), length(months_first)))
cat("Depths in cache:", paste(names(fits), collapse = ", "), "\n\n")

# Helper: day-of-year of prediction point within a month (must match
# the same position used in ohc_gp_interp.R -- defaults to "middle" / 15th)
pred_yday <- function(month_str, position = "middle") {
  d <- as.Date(month_str)
  target <- switch(position,
    first  = d,
    middle = as.Date(format(d, "%Y-%m-15")),
    last   = as.Date(format(d + 32, "%Y-%m-01")) - 1
  )
  as.integer(format(target, "%j"))
}

# ---- 2. matern_spheretime correlation function matching GpGp ---------------
#
# GpGp matern_spheretime (C++ source):
#   1. (lon_deg, lat_deg) -> 3D unit sphere:
#        x = cos(lat_rad)*cos(lon_rad),
#        y = cos(lat_rad)*sin(lon_rad),
#        z = sin(lat_rad)
#   2. 4D location: (x/rs, y/rs, z/rs, time/rt)
#   3. r = Euclidean distance in that scaled 4D space
#   4. Matern nu=3/2 correlation: (1 + r) * exp(-r)
#
# This function returns the n1 x n2 *correlation* matrix (C(0)=1, no sigma2,
# no nugget).  The nugget diagonal is added separately (as lambda*I inside
# fields::Krig, or explicitly in the direct Cholesky solve below).

matern_st_corr <- function(x1, x2 = NULL, rs, rt, C = NA, marginal = FALSE) {
  # fields::Krig covariance interface (mirrors Exp.cov.simple):
  #   - marginal=TRUE  -> return rep(1, nrow(x1))   (diagonal only, C(s,s)=1)
  #   - C supplied     -> return Corr(x1, x2) %*% C  (matrix-vector product)
  #   - otherwise      -> return Corr(x1, x2)         (full matrix)
  if (marginal) return(rep(1.0, nrow(x1)))
  if (is.vector(x1)) x1 <- matrix(x1, nrow = 1)
  if (is.vector(x2)) x2 <- matrix(x2, nrow = 1)
  lon1_r <- x1[, 1] * pi / 180;  lat1_r <- x1[, 2] * pi / 180
  lon2_r <- x2[, 1] * pi / 180;  lat2_r <- x2[, 2] * pi / 180
  # 3D unit-sphere coords, pre-scaled by the spatial range
  s1 <- cbind(cos(lat1_r) * cos(lon1_r),
              cos(lat1_r) * sin(lon1_r),
              sin(lat1_r)) / rs
  s2 <- cbind(cos(lat2_r) * cos(lon2_r),
              cos(lat2_r) * sin(lon2_r),
              sin(lat2_r)) / rs
  t1 <- x1[, 3] / rt
  t2 <- x2[, 3] / rt
  # Vectorised squared Euclidean distance in 4D scaled space
  D2s <- (outer(rowSums(s1^2), rep(1.0, nrow(x2)))
        + outer(rep(1.0, nrow(x1)), rowSums(s2^2))
        - 2 * tcrossprod(s1, s2))
  D2t <- outer(t1, t2, function(a, b) (a - b)^2)
  r   <- sqrt(pmax(D2s + D2t, 0))   # numerical safety for near-zero values
  K   <- (1 + r) * exp(-r)          # Matern nu=3/2 correlation
  if (is.na(C[1])) K else K %*% C
}

# ---- 3. Per-depth audit ----------------------------------------------------
all_rows <- list()

for (col in names(fits)) {
  fit <- fits[[col]]
  covparms    <- fit$covparms   # (sigma2, rs, rt, nu, nugget)
  sigma2_gpgp <- covparms[1]
  rs          <- covparms[2]   # spatial range (radians on unit sphere)
  rt          <- covparms[3]   # temporal range (days)
  nugget      <- covparms[5]   # relative nugget (tau^2 / sigma^2)
  beta_gpgp   <- fit$betahat

  cat(sprintf("=== %s ===\n", col))
  cat(sprintf("  GpGp: sigma2=%.4g  rs=%.4g rad (~%.0f km)  rt=%.3g days",
              sigma2_gpgp, rs, rs * 6371, rt))
  cat(sprintf("  nugget=%.4g  beta=%.4g\n", nugget, beta_gpgp))

  # Use the observations stored inside the fitted model -- these are the data
  # the model was actually fitted on, which may differ from the profiles CSV if
  # the CSV was regenerated after the cache was saved.
  y_obs    <- fit$y
  locs_obs <- as.matrix(fit$locs)   # columns: lon, lat, day_num
  n_obs    <- nrow(locs_obs)
  cat(sprintf("  n_obs = %d\n", n_obs))

  # ---- 3a. fields::Krig fit -----------------------------------------------
  # scale.type="none" prevents fields from rescaling the lon/lat/time columns
  # before passing them to our covariance function (which expects degrees and days).
  fit_f <- Krig(
    x            = locs_obs,
    Y            = y_obs,
    cov.function = matern_st_corr,
    cov.args     = list(rs = rs, rt = rt),
    lambda       = nugget,
    m            = 1,            # intercept only (matches GpGp X=NULL)
    scale.type   = "unscaled"
  )
  # fields v17 stores GLS variance in $rhohat, intercept coefficient in $d
  sigma2_fields <- fit_f$rhohat
  beta_fields   <- fit_f$d[1]
  cat(sprintf("  fields: sigma2_GLS=%.4g  (ratio vs GpGp: %.4f)  beta=%.4g  (delta=%.4g)\n",
              sigma2_fields, sigma2_fields / sigma2_gpgp,
              beta_fields, beta_fields - beta_gpgp))

  # ---- 3b. Pre-factor obs covariance for direct simple-kriging solve -------
  # K_obs = C_corr(obs, obs) + nugget*I  (sigma2 scales out of kriging weights)
  C_obs <- matern_st_corr(locs_obs, locs_obs, rs = rs, rt = rt)
  K_obs <- C_obs + nugget * diag(n_obs)
  chol_K <- chol(K_obs)           # upper triangular Cholesky

  # ---- 3c. Predict per month -----------------------------------------------
  month_rows <- list()
  for (mo in months_first) {
    yday_pred <- pred_yday(mo)
    locs_pred <- as.matrix(data.table(
      lon = pred_grid$lon,
      lat = pred_grid$lat,
      day = rep(yday_pred, nrow(pred_grid))
    ))
    n_pred <- nrow(locs_pred)

    # Cross-correlation: n_obs x n_pred
    C_cross <- matern_st_corr(locs_obs, locs_pred, rs = rs, rt = rt)

    # Kriging weights: b = K_obs^{-1} C_cross  (n_obs x n_pred)
    # Use base:: to avoid spam package masking backsolve/forwardsolve.
    b <- base::backsolve(chol_K, base::forwardsolve(t(chol_K), C_cross))

    # Simple kriging prediction (beta fixed at GpGp's betahat)
    resid_obs <- y_obs - beta_gpgp
    pred_sk   <- beta_gpgp + colSums(b * resid_obs)

    # Simple kriging SE (sigma2 fixed at GpGp's estimate)
    #   Var_SK(s0) = sigma2 * (C(s0,s0) - k0' K^{-1} k0)
    #             = sigma2 * (1 - sum_i b_i * C_cross_i)   since C(s0,s0) = 1
    c0_b  <- colSums(C_cross * b)           # k0' K^{-1} k0, one value per pred point
    se_sk <- sqrt(sigma2_gpgp * pmax(1 - c0_b, 0))

    # fields::Krig predictions (ordinary kriging: beta estimated, SE includes
    # beta estimation uncertainty)
    pred_fv <- as.numeric(predict(fit_f, x = locs_pred))
    se_fv   <- as.numeric(predictSE(fit_f, x = locs_pred))

    # GpGp output from the existing CSV
    gpgp_mo   <- gpgp_out[month == mo]
    pred_gpgp <- gpgp_mo[[paste0(col, "_pred")]]
    se_gpgp   <- gpgp_mo[[paste0(col, "_se")]]

    month_rows[[length(month_rows) + 1]] <- data.table(
      month              = mo,
      lon                = pred_grid$lon,
      lat                = pred_grid$lat,
      depth              = col,
      # GpGp custom kriging_predict (the primary audit target)
      pred_gpgp          = pred_gpgp,
      se_gpgp            = se_gpgp,
      # Direct simple kriging via R Cholesky + GpGp params
      pred_sk            = pred_sk,
      se_sk              = se_sk,
      pred_sk_diff       = pred_sk   - pred_gpgp,
      se_sk_ratio        = se_sk     / se_gpgp,
      # fields::Krig ordinary kriging (sigma2 from GLS, SE includes beta uncertainty)
      pred_fields        = pred_fv,
      se_fields          = se_fv,
      pred_fields_diff   = pred_fv   - pred_gpgp,
      se_fields_ratio    = se_fv     / se_gpgp,
      # sigma2 bookkeeping
      sigma2_gpgp        = sigma2_gpgp,
      sigma2_fields      = sigma2_fields
    )
  }

  depth_dt <- rbindlist(month_rows)
  all_rows[[col]] <- depth_dt

  # Summary statistics for this depth
  cat(sprintf("  Direct SK vs GpGp custom:\n"))
  cat(sprintf("    RMSE(pred): %.4g J/m2    max|diff|: %.4g J/m2\n",
              sqrt(mean(depth_dt$pred_sk_diff^2,     na.rm = TRUE)),
              max(abs(depth_dt$pred_sk_diff),         na.rm = TRUE)))
  cat(sprintf("    median(se_sk/se_gpgp): %.6f\n",
              median(depth_dt$se_sk_ratio, na.rm = TRUE)))
  cat(sprintf("  fields::Krig vs GpGp custom:\n"))
  cat(sprintf("    RMSE(pred): %.4g J/m2    max|diff|: %.4g J/m2\n",
              sqrt(mean(depth_dt$pred_fields_diff^2, na.rm = TRUE)),
              max(abs(depth_dt$pred_fields_diff),     na.rm = TRUE)))
  cat(sprintf("    median(se_fields/se_gpgp): %.6f\n",
              median(depth_dt$se_fields_ratio, na.rm = TRUE)))
  cat("\n")
}

out <- rbindlist(all_rows)
fwrite(out, audit_out_path)
cat(sprintf("Wrote %d rows -> %s\n", nrow(out), audit_out_path))
