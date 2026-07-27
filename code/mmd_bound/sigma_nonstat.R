## Non-stationary anisotropy Sigma(x) from Argo directional variograms, and the
## Paciorek-Schervish kernel that uses it. Argo-only (GLORYS-free); the fitted
## ranges shape the kernel geometry, are deflated conservatively, and never enter
## the amplitude or the guarantee. Companion to mmd_bound.R (kernel-agnostic core).

library(data.table)

## ---- 1. full-record Argo anomaly (deseasonalized + detrended) -------------
## The variogram must see the correlation STRUCTURE, not the mean front gradient
## or the seasonal cycle, so we remove a monthly climatology and a spatial plane.
## This is a calibration step; it never enters the bound.

load_argo_anomaly <- function(path, box, ohc_col = "ohc_2000") {
  dt <- fread(path, select = c("lat", "lon", "date", ohc_col))
  setnames(dt, ohc_col, "ohc")
  if (!inherits(dt$date, "POSIXct"))
    dt[, date := as.POSIXct(substr(date, 1, 19), format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")]
  dt <- dt[lat >= box$lat[1] & lat <= box$lat[2] &
             lon >= box$lon[1] & lon <= box$lon[2] & is.finite(ohc)]
  dt[, ohc := ohc * 1e-9]                                   # GJ/m2
  dt[, month := as.integer(format(date, "%m"))]
  dt[, ohc_ds := ohc - mean(ohc), by = month]              # remove seasonal cycle
  dt[, anom := residuals(lm(ohc_ds ~ lon + lat))]          # remove spatial plane (front gradient)
  dt[]
}

proj_km <- function(lon, lat, lon0, lat0)
  cbind(x = (lon - lon0) * 111.32 * cos(lat0 * pi / 180),
        y = (lat - lat0) * 110.57)

## ---- 2. directional empirical variogram -----------------------------------
## Binned semivariance gamma(h, direction) over spatial pairs. dt_bw is a
## CALIBRATION temporal bandwidth: only pairs within dt_bw days are used, so time
## decorrelation does not confound the SPATIAL fit. It is distinct from the
## kernel's temporal scale ell_t (Sec 3b / cfg$ell_t); keep dt_bw small.

directional_variogram <- function(xkm, ykm, z, day = NULL,
                                  dmax = 400, n_dbin = 16, n_abin = 8,
                                  dt_bw = 30) {
  DX <- outer(xkm, xkm, "-"); DY <- outer(ykm, ykm, "-")
  ut <- upper.tri(DX)
  dx <- DX[ut]; dy <- DY[ut]
  h  <- sqrt(dx^2 + dy^2)
  g  <- 0.5 * (outer(z, z, "-")[ut])^2
  keep <- h > 0 & h <= dmax
  if (!is.null(day)) keep <- keep & abs(outer(day, day, "-")[ut]) <= dt_bw
  dx <- dx[keep]; dy <- dy[keep]; h <- h[keep]; g <- g[keep]
  ang <- atan2(dy, dx) %% pi                               # fold to [0, pi)
  dbrk <- seq(0, dmax, length.out = n_dbin + 1)
  abrk <- seq(0, pi, length.out = n_abin + 1)
  d <- data.table(dbin = findInterval(h, dbrk, rightmost.closed = TRUE),
                  abin = findInterval(ang, abrk, rightmost.closed = TRUE),
                  h = h, g = g, ang = ang)
  d <- d[dbin >= 1 & abin >= 1]
  d[, .(gamma = mean(g), hbar = mean(h), angc = mean(ang), npair = .N),
    by = .(dbin, abin)]
}

## ---- 3. anisotropic exponential fit ---------------------------------------
## Fit gamma(h, ang) = s * (1 - exp(-d)), with d the anisotropic distance in a
## frame rotated by phi with principal ranges (r_maj, r_min). Weighted by pair
## count. Returns the kernel (anisotropy) matrix Sigma_s = R diag(r^2) R'.

rot <- function(phi) matrix(c(cos(phi), sin(phi), -sin(phi), cos(phi)), 2, 2)

fit_anisotropy <- function(dvar, sill0 = NULL) {
  if (is.null(sill0)) sill0 <- max(dvar$gamma)
  obj <- function(p) {
    phi <- p[1]; rmaj <- exp(p[2]); rmin <- exp(p[3]); s <- exp(p[4])
    dd <- dvar$hbar * sqrt((cos(dvar$angc - phi) / rmaj)^2 +
                           (sin(dvar$angc - phi) / rmin)^2)
    sum(dvar$npair * (dvar$gamma - s * (1 - exp(-dd)))^2)
  }
  best <- NULL
  for (phi0 in seq(0, pi - pi/6, by = pi/6)) {             # multistart over orientation
    op <- optim(c(phi0, log(150), log(80), log(sill0)), obj, method = "Nelder-Mead")
    if (is.null(best) || op$value < best$value) best <- op
  }
  phi <- best$par[1] %% pi
  rmaj <- exp(best$par[2]); rmin <- exp(best$par[3])
  if (rmin > rmaj) { tmp <- rmaj; rmaj <- rmin; rmin <- tmp; phi <- (phi + pi/2) %% pi }
  R <- rot(phi)
  list(phi = phi, r_maj = rmaj, r_min = rmin, sill = exp(best$par[4]),
       Sigma = R %*% diag(c(rmaj^2, rmin^2)) %*% t(R))
}

## ---- 3b. temporal variogram and range -------------------------------------
## Isolate temporal decorrelation from pairs close in space (space_max km) and
## fit an exponential to estimate the kernel temporal scale ell_t, instead of
## hard-coding it. (The design factor is non-monotone in ell_t just as in the
## spatial scale, so ell_t is also a swept axis, not a "conservative" value.)

temporal_variogram <- function(xkm, ykm, day, z, space_max = 60, dt_max = 365, n_bin = 20) {
  DS <- sqrt(outer(xkm, xkm, "-")^2 + outer(ykm, ykm, "-")^2)
  DT <- abs(outer(day, day, "-"))
  ut <- upper.tri(DS)
  ds <- DS[ut]; dtd <- DT[ut]; g <- 0.5 * (outer(z, z, "-")[ut])^2
  keep <- ds <= space_max & dtd > 0 & dtd <= dt_max
  dtd <- dtd[keep]; g <- g[keep]
  b <- findInterval(dtd, seq(0, dt_max, length.out = n_bin + 1), rightmost.closed = TRUE)
  data.table(bin = b, t = dtd, g = g)[b >= 1,
    .(gamma = mean(g), tbar = mean(t), npair = .N), by = bin][order(tbar)]
}

fit_temporal_range <- function(tvar, sill0 = NULL) {
  if (is.null(sill0)) sill0 <- max(tvar$gamma)
  obj <- function(p) sum(tvar$npair *
    (tvar$gamma - exp(p[2]) * (1 - exp(-tvar$tbar / exp(p[1]))))^2)
  op <- optim(c(log(45), log(sill0)), obj, method = "Nelder-Mead")
  list(ell_t = exp(op$par[1]), sill = exp(op$par[2]))
}

## ---- 4. Paciorek-Schervish kernel with spatially-varying Sigma -------------
## Sigma_fun(xkm, ykm) returns the 2x2 spatial anisotropy matrix at a location;
## the temporal axis is separable with scale ell_t. Constant Sigma reduces this
## to the stationary anisotropic Matern (prefactor 1).

sigma_components <- function(X, Sigma_fun) {
  S <- lapply(seq_len(nrow(X)), function(i) Sigma_fun(X[i, 1], X[i, 2]))
  a <- vapply(S, function(m) m[1, 1], 0.0)
  b <- vapply(S, function(m) m[1, 2], 0.0)
  c <- vapply(S, function(m) m[2, 2], 0.0)
  list(a = a, b = b, c = c, det = a * c - b^2)
}

ps_kernel_matrix <- function(X1, X2, S1, S2, ell_t, nu = 0.5) {
  a <- outer(S1$a, S2$a, "+") / 2
  b <- outer(S1$b, S2$b, "+") / 2
  cc <- outer(S1$c, S2$c, "+") / 2
  detA <- a * cc - b^2
  dx <- outer(X1[, 1], X2[, 1], "-"); dy <- outer(X1[, 2], X2[, 2], "-")
  Qs <- (dx^2 * cc - 2 * dx * dy * b + dy^2 * a) / detA        # Mahalanobis in avg metric
  Qt <- (outer(X1[, 3], X2[, 3], "-") / ell_t)^2
  pref <- outer(S1$det^0.25, S2$det^0.25, "*") / sqrt(detA)    # PS determinant prefactor
  pref * matern_cor(sqrt(pmax(Qs + Qt, 0)), nu)
}

## kfun closure for compute_pieces (set cfg$kfun to this). Precomputes Sigma
## components on the fly; X are (east_km, north_km, day) from to_physical.
make_ps_kfun <- function(Sigma_fun, ell_t, nu = 0.5) {
  function(Xa, Xb) {
    Sa <- sigma_components(Xa, Sigma_fun); Sb <- sigma_components(Xb, Sigma_fun)
    ps_kernel_matrix(Xa, Xb, Sa, Sb, ell_t, nu)
  }
}

make_sigma_constant <- function(Sigma) function(x, y) Sigma

## ---- 5. non-stationary Sigma(x): local fits at anchors, smooth interpolation

logm <- function(S) { e <- eigen(S, symmetric = TRUE); e$vectors %*% diag(log(e$values)) %*% t(e$vectors) }
expm_sym <- function(S) { e <- eigen(S, symmetric = TRUE); e$vectors %*% diag(exp(e$values)) %*% t(e$vectors) }

## Fit a local anisotropy at each anchor from Argo pairs within radius_km.
estimate_sigma_anchors <- function(an, box, anchors_lonlat, radius_km = 175, min_pts = 60) {
  lon0 <- mean(box$lon); lat0 <- mean(box$lat)
  xy  <- proj_km(an$lon, an$lat, lon0, lat0)
  day <- as.numeric(an$date) / 86400
  A   <- proj_km(anchors_lonlat[, 1], anchors_lonlat[, 2], lon0, lat0)
  out <- lapply(seq_len(nrow(A)), function(i) {
    sel <- sqrt((xy[, 1] - A[i, 1])^2 + (xy[, 2] - A[i, 2])^2) <= radius_km
    if (sum(sel) < min_pts) return(NULL)
    dv <- directional_variogram(xy[sel, 1], xy[sel, 2], an$anom[sel], day = day[sel], dmax = 300)
    list(anchor_km = A[i, ], anchor_lonlat = anchors_lonlat[i, ],
         n = sum(sel), fit = fit_anisotropy(dv))
  })
  out[!vapply(out, is.null, logical(1))]
}

## Smooth, SPD, bounded Sigma(x) by log-Euclidean inverse-distance interpolation
## of the anchor matrices. Anchors where data were too sparse are dropped; the
## bandwidth borrows from neighbours (a soft density floor). `scale` multiplies
## all ranges -- default 1 (variogram best estimate). It is NOT a conservative
## deflation: the design factor MMD is non-monotone in scale, so conservativeness
## comes from the scale SWEEP (sweep_sigma_scale), not from shortening ranges.
build_sigma_fun <- function(anchors, bw_km = 150, scale = 1, r_bounds = c(20, 200)) {
  Akm <- t(vapply(anchors, function(a) a$anchor_km, numeric(2)))
  Lg  <- lapply(anchors, function(a) logm(a$fit$Sigma))
  function(x, y) {
    w <- exp(-0.5 * ((Akm[, 1] - x)^2 + (Akm[, 2] - y)^2) / bw_km^2); w <- w / sum(w)
    Lbar <- Reduce(`+`, Map(function(Li, wi) Li * wi, Lg, w))
    e <- eigen(expm_sym(Lbar), symmetric = TRUE)
    r <- pmin(pmax(sqrt(pmax(e$values, 1e-6)) * scale, r_bounds[1]), r_bounds[2])
    e$vectors %*% diag(r^2) %*% t(e$vectors)
  }
}

## Scale sweep for the non-stationary kernel: multiply the Sigma(x) ranges by s
## and recompute MMD per window. The conservative design factor is the max over s
## (the design factor is non-monotone in scale). This replaces "deflation" as the
## conservative mechanism.
sweep_sigma_scale <- function(argo_box, cfg, sig_fun, scales = c(0.5, 1, 2, 4)) {
  set.seed(cfg$seed)
  rbindlist(lapply(build_windows(cfg), function(w) {
    argo <- window_subset(argo_box, w)
    if (nrow(argo) < 3 || uniqueN(argo$float_id) < 2) return(NULL)
    mc <- sample_Q(cfg$M_mc, cfg$box, w)
    rbindlist(lapply(scales, function(s) {
      cfg2 <- cfg
      cfg2$kfun <- make_ps_kfun(function(x, y) s^2 * sig_fun(x, y), cfg$ell_t, cfg$nu)
      data.table(window = w$label, scale = s,
                 mmd_upper = jackknife_mmd(argo, mc, cfg2)$upper)
    }))
  }))
}

