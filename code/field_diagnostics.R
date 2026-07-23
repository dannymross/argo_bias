## =====================================================================
## field_diagnostics.R -- distributional-shape diagnostics on the OBSERVED
##   Argo OHC, to inform the assumptions that enter the preferential-sampling
##   bias bound (paper/sections/methods.tex): the declared anomaly-field
##   kurtosis kappa (Lemma kurt / Prop sigmaY-envelope), the within-cell
##   kappa_perp and sigma_Y_perp (eq kappaperp-pairs / sigmaYperp-pairs), the
##   design variance R = Var_pi(Y), and the feasibility gate Delta < 1.
##
##   The mean-field / control-variate deseasonalization is NOT used: no offset
##   is fit and no mean field is assumed known. Seasonality and the secular
##   trend are handled by the space-time PARTITION -- with enough time cells
##   they become between-cell variation sigma_Y* paired with the count-estimable
##   sigma_w*. The only "detrending" is subtracting each cell's own empirical
##   mean (within-cell residual), which is intrinsic to the partition.
##
##   Two anomaly views (per user): RAW observed OHC (conservative total sigma_Y
##   / kappa headline) and the WITHIN-CELL RESIDUAL (design-based eddy-scale
##   kappa_perp). GLORYS is the dense nu-representative cross-check for kappa
##   (affine-invariant, so a design-law float estimate and a nu estimate are
##   directly comparable).
##
##   CAVEAT: floats realize the sampling law pi, not the reference nu, so
##   kappa_hat and R_hat here are design-law quantities. kappa is affine-
##   invariant, which is what licenses reading its shape off the floats.
##
##   Base R (+ data.table for loading). Reuses code/sigma_wK.R (sigma_w* count
##   estimator + delete-one-float jackknife) and code/leakage_curve.R (loaders,
##   cell_index, variance splits, GLORYS cross-check). Source those with their
##   self-tests silenced.
## =====================================================================

## ---- source helpers (silence their self-test / driver blocks) -------
FD_CODE_DIR <- local({
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[1]))) else "code"
})
local({
  keep_selftest <- if (exists("RUN_SELFTEST")) RUN_SELFTEST else NULL
  RUN_SELFTEST <<- FALSE; RUN_GLORYS <<- FALSE; SOURCED_AS_LIB <<- TRUE
  source(file.path(FD_CODE_DIR, "sigma_wK.R"))
  source(file.path(FD_CODE_DIR, "leakage_curve.R"))
  if (is.null(keep_selftest)) rm(RUN_SELFTEST, envir = .GlobalEnv) else RUN_SELFTEST <<- keep_selftest
})

sec_to_day <- function(s) s / 86400
`%||%` <- function(a, b) if (is.null(a)) b else a   # leakage_curve also defines it

## ---- 1. Loader: keep float_id + OHC value (prep_argo_data drops the value) ----
## Parses the ISO date to both a Date and a day-since-epoch t (the unit
## sigma_wK.R::assign_cells expects), rescales J/m^2 -> GJ/m^2, filters to the
## box + years, and greedily drops intra-cycle repeats (dt < min_gap days).
load_argo_profiles <- function(path = file.path("data", "argo_ohc.csv"),
                               lat_bnds, lon_bnds, years,
                               value_col = c("ohc_700", "ohc_2000"),
                               min_gap = 9, scale = 1e-9) {
  value_col <- match.arg(value_col)
  dt <- data.table::fread(path, select = c("float_id", "lon", "lat", "date", value_col))
  posix <- if (inherits(dt$date, "POSIXct")) dt$date else
    as.POSIXct(sub("Z$", "", dt$date), format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC")
  dt[, date := as.Date(posix)]
  dt[, t := sec_to_day(as.numeric(posix))]
  dt[, y := get(value_col) * scale]
  dt <- dt[is.finite(y) & is.finite(t) &
           lat >= lat_bnds[1] & lat <= lat_bnds[2] &
           lon >= lon_bnds[1] & lon <= lon_bnds[2] &
           (data.table::year(date) %in% years)]
  data.table::setorder(dt, float_id, t)
  keep <- dt[, {                                     # greedy min-gap thinning per float
    k <- logical(.N); last <- -Inf
    for (i in seq_len(.N)) if (t[i] - last >= min_gap) { k[i] <- TRUE; last <- t[i] }
    k
  }, by = float_id]$V1
  a <- as.data.frame(dt[keep, .(float_id, lon, lat, date, t, ohc = y)])
  a
}

## ---- 2. Distributional shape -> informs kappa -----------------------
## Population (nu-style) central moments; kappa is the kurtosis (Gaussian = 3),
## the exact constant of Lemma kurt. Optional weights for the area-weighted
## dense-field (nu) cross-check.
field_moments <- function(y, w = NULL) {
  ok <- is.finite(y); y <- y[ok]
  if (is.null(w)) w <- rep(1, length(y)) else w <- w[ok]
  W <- sum(w); m <- sum(w * y) / W
  mu2 <- sum(w * (y - m)^2) / W
  mu3 <- sum(w * (y - m)^3) / W
  mu4 <- sum(w * (y - m)^4) / W
  data.frame(n = length(y), mean = m, var = mu2, sd = sqrt(mu2),
             skew = mu3 / mu2^1.5, kappa = mu4 / mu2^2,
             excess_kurt = mu4 / mu2^2 - 3)
}

kurtosis_verdict <- function(kappa) {
  if (!is.finite(kappa)) return("indeterminate")
  if (kappa < 2.5) "platykurtic (< Gaussian; seasonal/bimodal signature)"
  else if (kappa <= 3.5) "~Gaussian (kappa=3 defensible)"
  else if (kappa <= 6)   "mildly leptokurtic (Gaussian..Laplace); declare kappa above the estimate"
  else "heavy-tailed (>= Laplace); declare a large kappa"
}

## Normal QQ + density panel for one field; anchors label Gaussian/Laplace/exp.
qq_density_panel <- function(y, label, file = NULL, w = 2400, h = 1200, res = 300) {
  y <- y[is.finite(y)]; fm <- field_moments(y)
  draw <- function() {
    op <- par(mfrow = c(1, 2), mar = c(4.2, 4.2, 3, 1)); on.exit(par(op))
    qqnorm(y, main = sprintf("Normal QQ: %s", label), cex.main = .5, pch = 20, cex = 0.5,
           col = "grey30"); qqline(y, col = "firebrick", lwd = 2)
    d <- density(y)
    hist(y, breaks = "FD", freq = FALSE, col = "grey85", border = "grey60",
         main = sprintf("%s\nk=%.2f",label, fm$kappa), cex.main = .5,
         xlab = "OHC (GJ m^-2)")
    lines(d, col = "steelblue", lwd = 2)
    curve(dnorm(x, fm$mean, fm$sd), add = TRUE, col = "firebrick", lwd = 2, lty = 2)
    legend("topright", c("empirical", "Gaussian fit"), col = c("steelblue", "firebrick"),
           lwd = 2, lty = c(1, 2), bty = "n", cex = 0.85)
  }
  if (is.null(file)) draw() else { open_png(file, width = w, height = h, res = res); draw(); dev.off() }
  fm
}

## ---- 3. Within-cell residual + pair-based kappa_perp / sigma_Y_perp --
## Design-based anomaly: subtract each equal-measure cell's own mean. The pair
## estimators (eq kappaperp-pairs / sigmaYperp-pairs) avoid the small-count
## bias of cell-demeaning by using within-cell pair differences D:
##   sigma_Y_perp^2 = mean(D^2)/2 ,  kappa_perp = 2 mean(D^4)/mean(D^2)^2 - 3 .
## between_float = TRUE keeps only pairs from distinct floats (removes within-
## float trajectory autocorrelation), matching sigma_wK.R's between-float logic.
within_cell_residual <- function(dt, box, period, K_lon, K_lat, Kt) {
  cell <- cell_index(dt, box, period, K_lon, K_lat, Kt)
  dt$ohc - ave(dt$ohc, cell, FUN = mean)
}

within_pair_diffs <- function(y, cell, float = NULL, between_float = TRUE) {
  idx <- split(seq_along(y), cell)
  unlist(lapply(idx, function(ii) {
    if (length(ii) < 2L) return(numeric(0))
    cb <- utils::combn(ii, 2L)
    d <- y[cb[1, ]] - y[cb[2, ]]
    if (between_float && !is.null(float)) d <- d[float[cb[1, ]] != float[cb[2, ]]]
    d
  }), use.names = FALSE)
}

pair_shape <- function(D) {
  D <- D[is.finite(D)]; d2 <- mean(D^2); d4 <- mean(D^4)
  data.frame(n_pairs = length(D), sigmaYperp2 = d2 / 2, sigmaYperp = sqrt(d2 / 2),
             kappa_perp = 2 * d4 / d2^2 - 3)
}

## kappa_perp and sigma_Y_perp across a K_t bracket (fixed spatial grid) and,
## separately, across spatial cell sizes (fixed K_t) -- shows the eddy-scale
## shape stabilizing and the residual shrinking as cells shrink.
residual_shape_sweep <- function(dt, box, period, K_lon, K_lat, Kt_grid,
                                 between_float = TRUE) {
  fl <- dt$float_id
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    cell <- cell_index(dt, box, period, K_lon, K_lat, Kt)
    D <- within_pair_diffs(dt$ohc, cell, fl, between_float)
    res_cm <- within_cell_residual(dt, box, period, K_lon, K_lat, Kt)
    fm <- field_moments(res_cm)                        # cell-demeaned residual moments
    ps <- pair_shape(D)
    data.frame(K_lon = K_lon, K_lat = K_lat, K_t = Kt, K = K_lon * K_lat * Kt,
               n_pairs = ps$n_pairs,
               sigmaYperp_pair = ps$sigmaYperp, kappa_perp_pair = ps$kappa_perp,
               sigmaYperp_cellmean = fm$sd, kappa_perp_cellmean = fm$kappa,
               row.names = NULL)
  }))
}

## ---- 4. Seasonality + trend context (DIAGNOSTIC ONLY, not the estimator) ----
## Quantifies how large the seasonal cycle and the secular trend are relative to
## sigma_Y, and returns the trend over the window as the breakdown benchmark T.
## This is context for interpreting the raw-field variance; it is NOT fit into
## the bound (which handles season+trend through the partition).
season_trend_context <- function(dt) {
  d <- data.frame(ohc = dt$ohc,
                  month = factor(as.integer(format(dt$date, "%m")), levels = 1:12),
                  t_yr = as.numeric(dt$date - min(dt$date)) / 365.25)
  win_yr <- diff(range(d$t_yr))
  r2 <- function(f) summary(f)$r.squared
  f_s  <- lm(ohc ~ month, d)
  f_t  <- lm(ohc ~ t_yr, d)
  f_st <- lm(ohc ~ month + t_yr, d)
  slope <- unname(coef(f_t)["t_yr"])                    # GJ m^-2 per year
  meff  <- tapply(residuals(lm(ohc ~ 1, d)) + coef(lm(ohc ~ 1, d)), d$month, mean)
  seas_pp <- diff(range(predict(f_s, data.frame(month = factor(1:12, levels = 1:12)))))
  list(sd_total = sd(d$ohc),
       R2_season = r2(f_s), R2_trend = r2(f_t), R2_season_trend = r2(f_st),
       sd_resid_season_trend = sd(residuals(f_st)),
       seasonal_peaktopeak = unname(seas_pp),
       trend_slope_per_yr = slope,
       trend_over_window = slope * win_yr,             # benchmark T for the breakdown budget
       window_yr = win_yr)
}

## ---- 5. Feasibility gate Delta < 1 (ties kappa to achievable resolution) ----
## Delta(kappa) = sigma_w* sqrt(kappa-1) + sigma_w*^2 ; the envelope
## (Prop sigmaY-envelope) is finite only where Delta < 1. Per Rem delta-fails
## the check uses the UPPER jackknife limit of sigma_w*. Also reports the
## refinement ceiling kappa_max = 1 + ((1 - sigma_w*^2)/sigma_w*)^2.
feasibility_table <- function(argo, box, grids, kappa_vec = c(3, 6, 9),
                              level = 0.95, between_float = FALSE) {
  rc <- refinement_curve(argo, box, grids, between_float = between_float, level = level)
  out <- do.call(rbind, lapply(seq_len(nrow(rc)), function(i) {
    sw  <- max(0, rc$sigma_hat[i]); swU <- max(0, rc$sigma_hi[i])
    kmax <- if (swU > 0) 1 + ((1 - swU^2) / swU)^2 else Inf
    delta <- function(s, k) s * sqrt(pmax(0, k - 1)) + s^2
    d <- data.frame(cell_km = rc$cell_km[i], K = rc$K[i],
                    sigma_wstar = sw, sigma_wstar_hi = swU,
                    kappa_max = kmax)
    for (k in kappa_vec) {
      d[[sprintf("Delta_k%g", k)]]    <- delta(sw,  k)
      d[[sprintf("feasible_k%g", k)]] <- delta(swU, k) < 1   # conservative: use upper sigma_w*
    }
    d
  }))
  out
}

plot_feasibility <- function(argo, box, grids, kappa_grid = seq(1.5, 12, by = 0.25),
                             level = 0.95, file = NULL) {
  rc <- refinement_curve(argo, box, grids, between_float = FALSE, level = level)
  draw <- function() {
    op <- par(mar = c(4.4, 4.4, 3, 1)); on.exit(par(op))
    cols <- hcl.colors(nrow(rc), "Zissou 1")
    plot(NA, xlim = range(kappa_grid), ylim = c(0, 1.6),
         xlab = expression(kappa ~ "(declared anomaly-field kurtosis)"),
         ylab = expression(Delta == sigma[w]^"*" * sqrt(kappa - 1) + sigma[w]^{"*2"}),
         main = "Envelope feasibility", cex.main = .5)
    abline(h = 1, lty = 2, col = "grey40")
    for (i in seq_len(nrow(rc))) {
      swU <- max(0, rc$sigma_hi[i])
      lines(kappa_grid, swU * sqrt(pmax(0, kappa_grid - 1)) + swU^2, col = cols[i], lwd = 2)
    }
    legend("topleft", legend = sprintf("%.0f km (K=%d)", rc$cell_km, rc$K),
           col = cols, lwd = 2, bty = "n", cex = 0.8)
    axis_anchors <- c(Gaussian = 3, Laplace = 6, Exp = 9)
    abline(v = axis_anchors, lty = 3, col = "grey70")
    text(axis_anchors, 1.55, names(axis_anchors), cex = 0.7, col = "grey40")
  }
  if (is.null(file)) draw() else { open_png(file, width = 2400, height = 1200, res = 300); draw(); dev.off() }
  invisible(rc)
}

## ---- 6. GLORYS dense cross-check (guarded) ---------------------------
## Dense nu-representative kappa (area-weighted) for the raw field and the
## within-cell residual, plus sigma_Y factors from leakage_curve. Compared to
## the float (design-law) estimates; affine-invariance of kappa makes them
## directly comparable.
glorys_shape_crosscheck <- function(glorys, box, period, K_lon, K_lat, Kt) {
  g <- filter_domain(glorys, box, period)
  w <- area_weight(g$lat)
  fm_raw <- field_moments(g$ohc, w)
  cell <- cell_index(g, box, period, K_lon, K_lat, Kt)
  swy <- rowsum(w * g$ohc, cell); sw <- rowsum(w, cell)
  cm <- setNames(as.numeric(swy / sw), rownames(sw))
  fm_perp <- field_moments(g$ohc - cm[as.character(cell)], w)
  data.frame(source = "GLORYS(nu)", view = c("raw", "within_cell"),
             kappa = c(fm_raw$kappa, fm_perp$kappa),
             sd = c(fm_raw$sd, fm_perp$sd), row.names = NULL)
}

glorys_available <- function(dir = file.path("data", "glorys_ohc"), years = 2020:2022) {
  any(file.exists(file.path(dir, sprintf("glorys_ohc_%d.csv", years))))
}

## ---- 7. Driver ------------------------------------------------------
run_field_diagnostics <- function(argo_path = file.path("data", "argo_ohc.csv"),
                                   lat_bnds = c(36, 40), lon_bnds = c(-68, -62),
                                   years = 2020:2022,
                                   depths = c("700", "2000"),
                                   K_lon = 6, K_lat = 5,
                                   Kt_grid = c(1, 2, 3, 6, 12, 24, 36),
                                   cell_km = c(200, 150, 110, 90, 75),
                                   K_t_feas = 3,
                                   kappa_vec = c(3, 6, 9),
                                   out_dir = ".", make_plots = TRUE) {
  period <- as.Date(sprintf(c("%d-01-01", "%d-12-31"), c(min(years), max(years))))
  T_BNDS <- as.numeric(as.POSIXct(paste(c(min(years), max(years)),
                                        c("01-01 00:00:00", "12-31 23:59:59"), sep = "-"),
                                  tz = "UTC"))
  box_ll  <- list(lon = lon_bnds, lat = lat_bnds)                 # leakage/cell_index box
  box_sig <- list(lon = lon_bnds, lat = lat_bnds, t = sec_to_day(T_BNDS))  # sigma_wK box
  grids   <- grid_from_cellsize(box_sig, cell_km, K_t = K_t_feas)
  gl <- if (glorys_available(years = years))
    tryCatch(load_glorys_ohc(years = years, depth = depths[1]), error = function(e) NULL) else NULL

  res <- list(period = period, box = box_ll, K_lon = K_lon, K_lat = K_lat, per_depth = list())
  for (d in depths) {
    a <- load_argo_profiles(argo_path, lat_bnds, lon_bnds, years, value_col = paste0("ohc_", d))
    stopifnot(nrow(a) >= 2)

    moments_raw <- field_moments(a$ohc)
    res_shape   <- residual_shape_sweep(a, box_ll, period, K_lon, K_lat, Kt_grid)
    split_kt    <- argo_leakage(a, box_ll, period, K_lon, K_lat, Kt_grid, mu_truth = NA_real_)
    st          <- season_trend_context(a)
    feas        <- feasibility_table(a, box_sig, grids, kappa_vec = kappa_vec)

    ## within-cell residual at the feasibility operating K_t, for the QQ view
    resid_op <- within_cell_residual(a, box_ll, period, K_lon, K_lat, K_t_feas)

    gl_cross <- if (!is.null(gl))
      glorys_shape_crosscheck(gl, box_ll, period, K_lon, K_lat, K_t_feas) else NULL

    if (make_plots) {
      qq_density_panel(a$ohc, sprintf("raw OHC %sm", d),
                       file = file.path(out_dir, sprintf("qq_raw_%s.png", d)))
      qq_density_panel(resid_op, sprintf("within-cell residual %sm (K_t=%d)", d, K_t_feas),
                       file = file.path(out_dir, sprintf("qq_residual_%s.png", d)))
      plot_feasibility(a, box_sig, grids,
                       file = file.path(out_dir, sprintf("feasibility_%s.png", d)))
    }

    res$per_depth[[d]] <- list(
      n = nrow(a), n_float = length(unique(a$float_id)),
      moments_raw = moments_raw, residual_shape = res_shape, variance_split = split_kt,
      season_trend = st, feasibility = feas, glorys_crosscheck = gl_cross)
  }
  res
}

print_field_diagnostics <- function(res) {
  for (d in names(res$per_depth)) {
    x <- res$per_depth[[d]]
    cat(sprintf("\n================ OHC %sm : %d profiles, %d floats ================\n",
                d, x$n, x$n_float))
    cat("\n-- raw-field moments (design-law; conservative total sigma_Y/kappa) --\n")
    print(round(x$moments_raw, 3), row.names = FALSE)
    cat(sprintf("   verdict: %s\n", kurtosis_verdict(x$moments_raw$kappa)))
    cat("\n-- within-cell (eddy-scale) shape vs resolution (between-float pairs) --\n")
    print(round(x$residual_shape[, c("K_t", "K", "n_pairs", "sigmaYperp_pair",
          "kappa_perp_pair", "sigmaYperp_cellmean", "kappa_perp_cellmean")], 3),
          row.names = FALSE)
    cat("\n-- variance split R = sigma_Y*^2 + sigma_Y_perp^2 vs K_t (L = perp fraction;",
        "sigma_wstar here is the plug-in for context -- see feasibility table for unbiased) --\n")
    print(round(x$variance_split[, c("K_t", "K", "sigmaYstar2", "sigmaYperp2",
          "sigmaY2", "L", "sigma_wstar")], 4), row.names = FALSE)
    cat("\n-- season/trend context (DIAGNOSTIC ONLY, not the estimator) --\n")
    st <- x$season_trend
    cat(sprintf("   sd_total=%.3f  R2 season=%.2f trend=%.2f season+trend=%.2f  resid sd=%.3f\n",
                st$sd_total, st$R2_season, st$R2_trend, st$R2_season_trend, st$sd_resid_season_trend))
    cat(sprintf("   seasonal peak-to-peak=%.3f GJ  trend=%.4f GJ/yr  over window=%.3f GJ (benchmark T)\n",
                st$seasonal_peaktopeak, st$trend_slope_per_yr, st$trend_over_window))
    cat("\n-- feasibility Delta<1 by resolution (upper jackknife sigma_w*) --\n")
    print(round(x$feasibility, 3), row.names = FALSE)
    if (!is.null(x$glorys_crosscheck)) {
      cat("\n-- GLORYS(nu) vs float(pi) kappa cross-check (affine-invariant shape) --\n")
      print(round(x$glorys_crosscheck[, c("view", "kappa", "sd")], 3), row.names = FALSE)
    } else {
      cat("\n-- GLORYS cross-check skipped: data/glorys_ohc/glorys_ohc_YYYY.csv absent",
          "(generate via code/export_glorys_ohc.py) --\n")
    }
  }
}

## ---- 8. Tidy-CSV export for reports/field_diagnostics.qmd -----------
## Mirrors leakage_curve.R::write_*_tables: one CSV per diagnostic, `depth`
## column, consumed by code/field_diagnostics.py. samples.csv carries the raw
## and within-cell-residual OHC values for the (Python-drawn) QQ panels.
write_field_diag_tables <- function(out_dir,
                                    depths = c("700", "2000"),
                                    argo_path = file.path("data", "argo_ohc.csv"),
                                    lat_bnds = c(36, 40), lon_bnds = c(-68, -62),
                                    years = 2020:2022, K_lon = 6, K_lat = 5,
                                    Kt_grid = c(1, 2, 3, 6, 12, 24, 36),
                                    cell_km = c(200, 150, 110, 90, 75), K_t_feas = 3) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  period <- as.Date(sprintf(c("%d-01-01", "%d-12-31"), c(min(years), max(years))))
  T_BNDS <- as.numeric(as.POSIXct(paste(c(min(years), max(years)),
                                        c("01-01 00:00:00", "12-31 23:59:59"), sep = "-"),
                                  tz = "UTC"))
  box_ll  <- list(lon = lon_bnds, lat = lat_bnds)
  box_sig <- list(lon = lon_bnds, lat = lat_bnds, t = sec_to_day(T_BNDS))
  grids   <- grid_from_cellsize(box_sig, cell_km, K_t = K_t_feas)
  gl_ok   <- glorys_available(years = years)

  mom <- rsh <- vsp <- sea <- fea <- smp <- glc <- list()
  for (d in depths) {
    a <- load_argo_profiles(argo_path, lat_bnds, lon_bnds, years, value_col = paste0("ohc_", d))
    fm <- field_moments(a$ohc)
    mom[[d]] <- data.frame(depth = d, n = nrow(a), n_float = length(unique(a$float_id)),
                           mean = fm$mean, var = fm$var, sd = fm$sd, skew = fm$skew,
                           kappa = fm$kappa, excess_kurt = fm$excess_kurt)
    rsh[[d]] <- cbind(depth = d, residual_shape_sweep(a, box_ll, period, K_lon, K_lat, Kt_grid))
    vs <- argo_leakage(a, box_ll, period, K_lon, K_lat, Kt_grid, mu_truth = NA_real_)
    vsp[[d]] <- cbind(depth = d, vs[, c("K_t", "K", "sigmaYstar2", "sigmaYperp2",
                                        "sigmaY2", "L", "sigma_wstar")])
    st <- season_trend_context(a)
    sea[[d]] <- data.frame(depth = d, st[c("sd_total", "R2_season", "R2_trend",
      "R2_season_trend", "sd_resid_season_trend", "seasonal_peaktopeak",
      "trend_slope_per_yr", "trend_over_window", "window_yr")])
    rc <- refinement_curve(a, box_sig, grids, between_float = FALSE)
    swU <- pmax(0, rc$sigma_hi)
    fea[[d]] <- data.frame(depth = d, cell_km = rc$cell_km, K = rc$K,
                           sigma_wstar = pmax(0, rc$sigma_hat), sigma_wstar_hi = swU,
                           kappa_max = ifelse(swU > 0, 1 + ((1 - swU^2) / swU)^2, Inf))
    resid_op <- within_cell_residual(a, box_ll, period, K_lon, K_lat, K_t_feas)
    smp[[d]] <- rbind(data.frame(depth = d, view = "raw", value = a$ohc),
                      data.frame(depth = d, view = "residual", value = resid_op))
    if (gl_ok) {
      gld <- tryCatch(load_glorys_ohc(years = years, depth = d), error = function(e) NULL)
      if (!is.null(gld))
        glc[[d]] <- cbind(depth = d, glorys_shape_crosscheck(gld, box_ll, period,
                                                             K_lon, K_lat, K_t_feas))
    }
  }
  wr <- function(l, f) data.table::fwrite(do.call(rbind, l), file.path(out_dir, f))
  wr(mom, "moments.csv"); wr(rsh, "residual_shape.csv"); wr(vsp, "variance_split.csv")
  wr(sea, "season_trend.csv"); wr(fea, "feasibility.csv"); wr(smp, "samples.csv")
  if (length(glc)) wr(glc, "glorys_cross.csv")
  invisible(out_dir)
}

## =====================================================================
## CLI: Rscript code/field_diagnostics.R tables <out_dir> [depths]
## Otherwise (no args): self-test + real-data pipeline (FD_SELFTEST <- FALSE
## to skip both, e.g. when sourcing this file only to reuse its functions).
## =====================================================================
FD_ARGS <- commandArgs(trailingOnly = TRUE)
if (length(FD_ARGS) >= 2 && FD_ARGS[1] == "tables") {
  fd_depths <- if (length(FD_ARGS) >= 3) strsplit(FD_ARGS[3], ",")[[1]] else c("700", "2000")
  write_field_diag_tables(FD_ARGS[2], fd_depths)
} else if (length(FD_ARGS) >= 2 && FD_ARGS[1] == "plots") {
    run_field_diagnostics(make_plots = TRUE,out_dir = FD_ARGS[2])
} else {
if (!exists("FD_SELFTEST")) FD_SELFTEST <- FALSE
if (FD_SELFTEST) {

  cat("== Test: pair estimators recover a planted within-cell kappa ==\n")
  set.seed(1)
  ## one cell, iid Laplace draws (kappa = 6): between-pair estimator should hit 6.
  lap <- function(n) { u <- runif(n) - 0.5; -sign(u) * log(1 - 2 * abs(u)) }
  y <- lap(4000); cell <- rep(1L, length(y)); fl <- seq_along(y)   # all distinct floats
  D <- within_pair_diffs(y, cell, fl, between_float = TRUE)
  cat(sprintf("  planted kappa=6 (Laplace):  pair kappa_perp=%.2f   direct kappa=%.2f\n",
              pair_shape(D)$kappa_perp, field_moments(y)$kappa))

  cat("\n== Test: synthetic floats -- raw platykurtic (season+trend), residual leptokurtic ==\n")
  set.seed(11)
  nfl <- 40; per <- 12
  b_lon <- c(-68, -62); b_lat <- c(36, 40)
  d0 <- as.Date("2020-01-01"); win <- 1095
  rows <- do.call(rbind, lapply(seq_len(nfl), function(f) {
    hl <- runif(1, b_lon[1], b_lon[2]); hb <- runif(1, b_lat[1], b_lat[2])
    lon <- pmin(pmax(hl + cumsum(rnorm(per, 0, .25)), b_lon[1]), b_lon[2])
    lat <- pmin(pmax(hb + cumsum(rnorm(per, 0, .20)), b_lat[1]), b_lat[2])
    tt  <- sort(runif(per, 0, win))
    date <- d0 + tt
    front <- 70 + 5 * (38 - lat) + 2 * (-65 - lon)
    seas  <- 8 * cos(2 * pi * as.integer(format(date, "%j")) / 365 - 0.4)
    trend <- 0.6 * tt / 365.25
    eddy  <- 3 * lap(per)                                # leptokurtic sub-cell signal
    data.frame(float_id = f, lon = lon, lat = lat, date = date,
               t = sec_to_day(as.numeric(as.POSIXct(date, tz = "UTC"))),
               ohc = front + seas + trend + eddy + rnorm(per, 0, 0.3))
  }))
  box_ll <- list(lon = b_lon, lat = b_lat); period <- c(d0, d0 + win)
  cat(sprintf("  raw kappa=%.2f (%s)\n", field_moments(rows$ohc)$kappa,
              kurtosis_verdict(field_moments(rows$ohc)$kappa)))
  rs <- residual_shape_sweep(rows, box_ll, period, K_lon = 4, K_lat = 3,
                             Kt_grid = c(1, 3, 12, 36))
  print(round(rs[, c("K_t", "n_pairs", "sigmaYperp_pair", "kappa_perp_pair")], 2),
        row.names = FALSE)
  vs <- argo_leakage(rows, box_ll, period, 4, 3, c(1, 3, 12, 36), mu_truth = NA_real_)
  cat("  L (within-cell variance fraction) should fall as K_t rises:\n")
  print(round(vs[, c("K_t", "L", "sigmaYstar2", "sigmaYperp2")], 3), row.names = FALSE)

  ## ---- real-data pipeline (runs if argo_ohc.csv present) ----
  if (file.exists("data/argo_ohc.csv") && requireNamespace("data.table", quietly = TRUE)) {
    cat("\n\n############## REAL DATA: NAC box, 2020-2022 ##############\n")
    res <- run_field_diagnostics(make_plots = TRUE)
    print_field_diagnostics(res)
    cat("\nSaved qq_raw_*.png, qq_residual_*.png, feasibility_*.png\n")
  }
}
}
