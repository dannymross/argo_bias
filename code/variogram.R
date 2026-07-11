## =====================================================================
## variogram.R -- OHC spatial (directional) and temporal decorrelation
##   scales from GLORYS, to set grid resolution and check whether the
##   within-cell residual can be made small at the achievable cell size.
##
##   Outputs, for BOTH estimands (raw OHC, deseasonalized OHC):
##     * directional spatial variograms (omni / zonal / meridional)
##     * decorrelation ranges via empirical 1/e crossing AND exp fit
##     * temporal decorrelation (e-folding) of the regional-mean series
##     * verdict: is the SHORT (meridional) range >= the sampling scale?
##       (the ordering that makes sigma_Y^perp small at the operating cell)
##
##   Parameterized on (box, analysis, clim); base R only.
##   INPUT: data.frame `glorys` with lon, lat, date (Date), ohc.
## =====================================================================

default_config <- function() list(
  box      = list(lon = c(-68, -62), lat = c(36, 40)),
  analysis = as.Date(c("2020-01-01", "2022-12-31")),
  clim     = as.Date(c("2020-01-01", "2022-12-31")),
  sampling_scale_km = 75,     # from float count: sqrt(|A| / n_floats)
  n_sub    = 400,             # points subsampled per snapshot
  n_bins   = 25,
  snap_stride = 9,            # use every k-th snapshot (speed)
  max_lag_days = 220,
  seed     = 1
)

## ------------------------- helpers ----------------------------------
EARTH_KM <- 6371
area_weight <- function(lat) cos(lat * pi / 180)

filter_domain <- function(dt, box, period) {
  keep <- dt$lon >= box$lon[1] & dt$lon <= box$lon[2] &
          dt$lat >= box$lat[1] & dt$lat <= box$lat[2] &
          dt$date >= period[1] & dt$date <= period[2]
  dt[keep, , drop = FALSE]
}
build_climatology <- function(dt) {
  ulon <- sort(unique(dt$lon)); ulat <- sort(unique(dt$lat))
  nlon <- length(ulon); nlat <- length(ulat)
  keyfun <- function(lon, lat, date) {
    (as.integer(format(date,"%m"))-1L)*nlon*nlat +
      (match(lat,ulat)-1L)*nlon + match(lon,ulon)
  }
  list(clim = tapply(dt$ohc, keyfun(dt$lon,dt$lat,dt$date), mean), keyfun = keyfun)
}
deseasonalize <- function(dt, clim) {
  dt$ohc <- dt$ohc - clim$clim[as.character(clim$keyfun(dt$lon,dt$lat,dt$date))]
  dt
}
collapse_space <- function(dt) {
  w <- area_weight(dt$lat); g <- as.character(dt$date)
  data.frame(date = as.Date(rownames(rowsum(w, g))),
             ybar = as.numeric(rowsum(w*dt$ohc, g) / rowsum(w, g)), row.names = NULL)
}
project_km <- function(lon, lat, lon0, lat0)
  list(x = EARTH_KM*cos(lat0*pi/180)*(lon-lon0)*pi/180,
       y = EARTH_KM*(lat-lat0)*pi/180)

## ---- directional spatial variogram, pooled over snapshots ----------
spatial_variogram <- function(dt, box, detrend = c("plane","mean"),
                              n_sub = 400, n_bins = 25, max_dist_km = NULL,
                              snap_stride = NULL, seed = 1) {
  detrend <- match.arg(detrend); set.seed(seed)
  lon0 <- mean(box$lon); lat0 <- mean(box$lat)
  if (is.null(max_dist_km)) {
    wx <- EARTH_KM*cos(lat0*pi/180)*diff(box$lon)*pi/180
    wy <- EARTH_KM*diff(box$lat)*pi/180
    max_dist_km <- 0.6*sqrt(wx^2 + wy^2)
  }
  breaks <- seq(0, max_dist_km, length.out = n_bins + 1)
  mids   <- (breaks[-1] + breaks[-(n_bins+1)]) / 2
  dirs   <- c("omni","zonal","merid")
  sum_g  <- cnt_g <- matrix(0, n_bins, 3, dimnames = list(NULL, dirs))
  sill_acc <- numeric(0)
  snaps <- sort(unique(dt$date))
  if (!is.null(snap_stride)) snaps <- snaps[seq(1, length(snaps), by = snap_stride)]
  for (dd in as.character(snaps)) {
    sub <- dt[as.character(dt$date) == dd, ]
    z <- sub$ohc; lon <- sub$lon; lat <- sub$lat
    z <- if (detrend == "mean") z - mean(z) else residuals(lm(z ~ lon + lat))
    sill_acc <- c(sill_acc, mean(z^2))
    if (length(z) > n_sub) { i <- sample.int(length(z), n_sub); z<-z[i]; lon<-lon[i]; lat<-lat[i] }
    p <- project_km(lon, lat, lon0, lat0)
    dx <- outer(p$x, p$x, "-"); dy <- outer(p$y, p$y, "-")
    d  <- sqrt(dx^2 + dy^2); g <- 0.5 * outer(z, z, "-")^2
    lt <- lower.tri(d)
    dv <- d[lt]; gv <- g[lt]
    ang <- atan2(abs(dy[lt]), abs(dx[lt])) * 180/pi         # 0 (E-W) .. 90 (N-S)
    keep <- dv > 0 & dv <= max_dist_km
    dv <- dv[keep]; gv <- gv[keep]; ang <- ang[keep]
    bin <- findInterval(dv, breaks, rightmost.closed = TRUE)
    for (k in 1:3) {
      sel <- switch(k, rep(TRUE, length(dv)), ang < 30, ang > 60)
      if (!any(sel)) next
      sg <- tapply(gv[sel], bin[sel], sum); cg <- tapply(gv[sel], bin[sel], length)
      bi <- as.integer(names(sg))
      sum_g[bi,k] <- sum_g[bi,k] + sg; cnt_g[bi,k] <- cnt_g[bi,k] + cg
    }
  }
  gamma <- sum_g / cnt_g
  data.frame(dist_km = mids,
             gamma_omni = gamma[,1],  n_omni = cnt_g[,1],
             gamma_zonal= gamma[,2],  n_zonal= cnt_g[,2],
             gamma_merid= gamma[,3],  n_merid= cnt_g[,3],
             sill = mean(sill_acc, na.rm = TRUE))
}

## ---- range extraction: empirical 1/e + exponential fit -------------
fit_range <- function(dist_km, gamma, npair, sill) {
  ok <- is.finite(gamma) & npair > 0
  h <- dist_km[ok]; g <- gamma[ok]; w <- npair[ok]
  thr <- (1 - 1/exp(1)) * sill
  emp <- NA_real_
  ab <- which(g >= thr)
  if (length(ab)) emp <- if (ab[1] == 1) h[1] else
    approx(g[c(ab[1]-1, ab[1])], h[c(ab[1]-1, ab[1])], xout = thr)$y
  a_fit <- NA_real_
  fit <- tryCatch(nls(g ~ nug + (ps - nug)*(1 - exp(-h/a)),
                      start = list(nug = 0, ps = sill, a = max(h)/3),
                      weights = w, algorithm = "port",
                      lower = c(0, 0, 1), upper = c(sill, 2*sill, max(h))),
                  error = function(e) NULL)
  if (!is.null(fit)) a_fit <- unname(coef(fit)["a"])
  c(range_1e = emp, range_exp = a_fit)
}

## ---- temporal decorrelation of the regional-mean series ------------
temporal_acf <- function(series, max_lag_days = 220) {
  y <- series$ybar
  step <- as.numeric(median(diff(series$date)))
  a <- acf(y, lag.max = ceiling(max_lag_days/step), plot = FALSE)$acf[,1,1]
  lags <- (0:(length(a)-1)) * step; thr <- 1/exp(1)
  bl <- which(a < thr)
  ef <- if (length(bl) && bl[1] > 1)
    approx(a[c(bl[1]-1, bl[1])], lags[c(bl[1]-1, bl[1])], xout = thr)$y else NA_real_
  list(lag_days = lags, acf = a, efold_days = ef)
}

## ------------------------- driver -----------------------------------
run_variogram_analysis <- function(glorys, config = default_config()) {
  box <- config$box
  ana  <- filter_domain(glorys, box, config$analysis)
  clim <- build_climatology(filter_domain(glorys, box, config$clim))
  ana_ds <- deseasonalize(ana, clim)
  vg_raw <- spatial_variogram(ana,    box, "plane", config$n_sub, config$n_bins,
                              snap_stride = config$snap_stride, seed = config$seed)
  vg_ds  <- spatial_variogram(ana_ds, box, "mean",  config$n_sub, config$n_bins,
                              snap_stride = config$snap_stride, seed = config$seed)
  rng <- function(vg, field) do.call(rbind, lapply(c("omni","zonal","merid"), function(dir) {
    r <- fit_range(vg$dist_km, vg[[paste0("gamma_",dir)]], vg[[paste0("n_",dir)]], vg$sill[1])
    data.frame(field = field, direction = dir, sill = round(vg$sill[1],3),
               range_1e_km = round(r["range_1e"],1), range_exp_km = round(r["range_exp"],1),
               row.names = NULL)
  }))
  ranges <- rbind(rng(vg_raw,"raw"), rng(vg_ds,"deseas"))
  tac <- temporal_acf(collapse_space(ana_ds), config$max_lag_days)
  list(vg_raw = vg_raw, vg_ds = vg_ds, ranges = ranges, temporal = tac, config = config)
}

report_verdict <- function(res) {
  ss <- res$config$sampling_scale_km
  cat(sprintf("\nSampling scale (float-count floor): %.0f km\n", ss))
  for (fld in c("raw","deseas")) {
    m <- res$ranges[res$ranges$field==fld & res$ranges$direction=="merid", ]
    short <- suppressWarnings(min(m$range_1e_km, m$range_exp_km, na.rm=TRUE))
    verdict <- if (!is.finite(short)) "indeterminate (no clean range)" else
      if (short >= ss) "OK: cells at sampling scale sit within a decorrelation length -> sigma_Y^perp small" else
      "CONCERN: short scale < sampling scale -> bias-relevant structure unresolved; residual not automatically negligible"
    cat(sprintf("  [%s] meridional (short) range ~ %.0f km  ->  %s\n", fld, short, verdict))
  }
  cat(sprintf("\nTemporal decorrelation (deseasonalized regional mean): e-fold ~ %.0f days\n",
              res$temporal$efold_days))
}

plot_variograms <- function(res) {
  op <- par(mfrow=c(1,2), mar=c(4.2,4.4,3,1)); on.exit(par(op))
  for (fld in c("raw","deseas")) {
    vg <- if (fld=="raw") res$vg_raw else res$vg_ds
    cols <- c(omni="grey40", zonal="firebrick", merid="steelblue")
    plot(vg$dist_km, vg$gamma_omni, type="n",
         ylim=c(0, max(vg[,c("gamma_omni","gamma_zonal","gamma_merid")],na.rm=TRUE)*1.05),
         xlab="separation (km)", ylab="semivariance",
         main=sprintf("%s OHC variogram", fld))
    abline(h=vg$sill[1], lty=3, col="grey60")
    abline(v=res$config$sampling_scale_km, lty=2, col="grey50")
    for (dir in c("omni","zonal","merid")) {
      lines(vg$dist_km, vg[[paste0("gamma_",dir)]], col=cols[dir], lwd=2, type="b", pch=20)
    }
    legend("bottomright", c("omni","zonal","meridional","sill","sampling scale"),
           col=c(cols,"grey60","grey50"), lwd=c(2,2,2,1,1), lty=c(1,1,1,3,2),
           pch=c(20,20,20,NA,NA), bty="n", cex=0.8)
  }
}

## ========================= SELF-TEST ================================
if (!exists("RUN_SELFTEST")) RUN_SELFTEST <- TRUE
if (RUN_SELFTEST) {
  set.seed(1)
  box <- list(lon = c(-68,-62), lat = c(36,40))
  lonv <- seq(box$lon[1], box$lon[2], by=1/12); nlon <- length(lonv)
  latv <- seq(box$lat[1], box$lat[2], by=1/12); nlat <- length(latv)
  grid <- expand.grid(lon=lonv, lat=latv, KEEP.OUT.ATTRS=FALSE)
  front <- 70 + 5*(38-grid$lat) + 2*((-65)-grid$lon)      # anisotropic front (stronger in lat)
  dates <- seq(as.Date("2020-01-01"), as.Date("2022-12-31"), by=3)
  ## anisotropic mesoscale via separable Gaussian-smoothed white noise
  gk <- function(s){ r<-max(1,ceiling(3*s)); k<-dnorm((-r):r,0,s); k/sum(k) }
  gsmooth <- function(M, sx, sy) {
    M <- apply(M, 2, function(col) as.numeric(stats::filter(col, gk(sx), circular=TRUE)))
    t(apply(M, 1, function(row) as.numeric(stats::filter(row, gk(sy), circular=TRUE))))
  }
  rows <- lapply(dates, function(d) {
    doy <- as.integer(format(d,"%j")); k <- as.numeric(d - as.Date("2020-01-01"))
    seas <- 8*cos(2*pi*doy/365 - 0.4)
    edd <- gsmooth(matrix(rnorm(nlon*nlat), nlon, nlat), sx=6, sy=3)  # ~54 km zonal, ~27 km merid
    edd <- 2.5 * as.vector(edd) / sd(edd)
    data.frame(lon=grid$lon, lat=grid$lat, date=d,
               ohc = front + seas + 0.6*k/365.25 + edd + rnorm(nrow(grid),0,0.3))
  })
  glorys <- do.call(rbind, rows)
  cat(sprintf("synthetic GLORYS: %d rows; eddy built with sx=6 (~54km) sy=3 (~27km) cells\n",
              nrow(glorys)))

  res <- run_variogram_analysis(glorys, default_config())
  cat("\n== directional decorrelation ranges (km) ==\n")
  print(res$ranges, row.names = FALSE)
  report_verdict(res)
  png("variograms.png", width=1100, height=470, res=110); plot_variograms(res); dev.off()
  cat("\nSaved variograms.png\n")
}
