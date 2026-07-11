## =====================================================================
## leakage_curve.R -- within-bin field-variance fraction L(K_t) to
##   inform the choice of K_t for sigma_w, under raw and deseasonalized
##   OHC estimands.
##
##   Decomposes sigma_Y^2 = sigma_{Y,Kt}^2 (between-bin) + (sigma_Y^perp)^2
##   (within-bin) on an equal-duration time partition, and reports
##   L(K_t) = (sigma_Y^perp / sigma_Y)^2.  Computed on the dense GLORYS
##   field (area-weighted by cos(lat), time-weighted by day), so it is
##   free of the float coincidence-starvation issues that limit sigma_w.
##
##   Two views:
##     time-only : collapse space first -> temporal leakage of the
##                 regional-mean series (isolates seasonality; sets K_t).
##     full      : full space x time equal-nu partition (the sigma_Y^perp
##                 that actually enters the residual term of the bound).
##
##   Parameterized on (box, analysis_period, clim_period): defaults to the
##   NAC box / 2020-2022, but works for any region/period.
##   Base R only (data.table optional for very large inputs).
##
##   INPUT: data.frame `glorys` with columns
##     lon (deg), lat (deg), date (Date), ohc (e.g. GJ m^-2), daily 1/12 deg.
## =====================================================================

## ---------------------------- config --------------------------------
default_config <- function() list(
  box      = list(lon = c(-68, -62), lat = c(36, 40)),   # NAC box
  analysis = as.Date(c("2020-01-01", "2022-12-31")),      # working window
  clim     = as.Date(c("2020-01-01", "2022-12-31")),      # climatology baseline
                                                          #  (dev: same; full run: use a
                                                          #   long fixed baseline, match Baugh)
  Kt_grid  = c(1, 2, 3, 4, 6, 9, 12, 18, 24, 36),
  K_lon    = 6, K_lat = 5,          # spatial grid for the full space x time view
  tol_L    = 0.05                   # leakage tolerance reference line
)

## ------------------------- helpers ----------------------------------
area_weight <- function(lat) cos(lat * pi / 180)

filter_domain <- function(dt, box, period) {
  keep <- dt$lon >= box$lon[1] & dt$lon <= box$lon[2] &
          dt$lat >= box$lat[1] & dt$lat <= box$lat[2] &
          dt$date >= period[1] & dt$date <= period[2]
  dt[keep, , drop = FALSE]
}

## Per-(cell, calendar-month) GLORYS climatology from the baseline data.
build_climatology <- function(dt) {
  ulon <- sort(unique(dt$lon)); ulat <- sort(unique(dt$lat))
  nlon <- length(ulon); nlat <- length(ulat)
  keyfun <- function(lon, lat, date) {
    li <- match(lon, ulon); bi <- match(lat, ulat)
    mo <- as.integer(format(date, "%m"))
    (mo - 1L) * nlon * nlat + (bi - 1L) * nlon + li
  }
  key <- keyfun(dt$lon, dt$lat, dt$date)
  clim <- tapply(dt$ohc, key, mean)             # named by integer key
  list(clim = clim, keyfun = keyfun)
}

deseasonalize <- function(dt, clim) {
  k <- clim$keyfun(dt$lon, dt$lat, dt$date)
  dt$ohc <- dt$ohc - clim$clim[as.character(k)]
  dt
}

## Area-weighted regional-mean OHC series (one value per date).
collapse_space <- function(dt) {
  w  <- area_weight(dt$lat)
  g  <- as.character(dt$date)
  num <- rowsum(w * dt$ohc, g); den <- rowsum(w, g)
  data.frame(date = as.Date(rownames(num)),
             ybar = as.numeric(num / den), row.names = NULL)
}

## ---- time-only leakage: decompose the regional-mean series ---------
time_leakage <- function(series, period, Kt_grid) {
  t    <- as.numeric(series$date - period[1])
  tmax <- as.numeric(period[2] - period[1])
  y    <- series$ybar; N <- length(y); mu <- mean(y)
  total <- mean((y - mu)^2)                     # equal day weights
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    edges <- seq(0, tmax, length.out = Kt + 1)
    bin   <- findInterval(t, edges, rightmost.closed = TRUE, all.inside = TRUE)
    ybar_b <- tapply(y, bin, mean)
    var_b  <- tapply(y, bin, function(v) mean((v - mean(v))^2))
    wb     <- tapply(y, bin, length) / N
    between <- sum(wb * (ybar_b - mu)^2)
    within  <- sum(wb * var_b)
    data.frame(K_t = Kt, bin_days = tmax / Kt, total = total,
               between = between, within = within,
               L = within / total, identity_gap = between + within - total)
  }))
}

## ---- full space x time leakage: equal-nu partition -----------------
full_leakage <- function(dt, box, period, K_lon, K_lat, Kt_grid) {
  w    <- area_weight(dt$lat); y <- dt$ohc
  t    <- as.numeric(dt$date - period[1]); tmax <- as.numeric(period[2] - period[1])
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  bl <- findInterval(dt$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  bb <- findInterval(sin(dt$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  sp <- (bb - 1L) * K_lon + bl
  W  <- sum(w); mu <- sum(w * y) / W
  total <- sum(w * (y - mu)^2) / W
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    e_t  <- seq(0, tmax, length.out = Kt + 1)
    bt   <- findInterval(t, e_t, rightmost.closed = TRUE, all.inside = TRUE)
    cell <- (bt - 1L) * (K_lon * K_lat) + sp
    sw   <- as.numeric(rowsum(w, cell))
    swy  <- as.numeric(rowsum(w * y, cell))
    swy2 <- as.numeric(rowsum(w * y * y, cell))
    cmean <- swy / sw; cvar <- pmax(0, swy2 / sw - cmean^2)
    wcell <- sw / W
    between <- sum(wcell * (cmean - mu)^2); within <- sum(wcell * cvar)
    data.frame(K_t = Kt, total = total, between = between, within = within,
               L = within / total, identity_gap = between + within - total)
  }))
}

## ------------------------- driver -----------------------------------
run_leakage_analysis <- function(glorys, config = default_config()) {
  box <- config$box
  ana <- filter_domain(glorys, box, config$analysis)
  stopifnot(nrow(ana) > 0)

  ## raw estimand
  s_raw   <- collapse_space(ana)
  t_raw   <- time_leakage(s_raw, config$analysis, config$Kt_grid); t_raw$field <- "raw"
  f_raw   <- full_leakage(ana, box, config$analysis, config$K_lon, config$K_lat, config$Kt_grid)
  f_raw$field <- "raw"

  ## deseasonalized estimand (GLORYS climatology, Argo-independent)
  clim    <- build_climatology(filter_domain(glorys, box, config$clim))
  ana_ds  <- deseasonalize(ana, clim)
  s_ds    <- collapse_space(ana_ds)
  t_ds    <- time_leakage(s_ds, config$analysis, config$Kt_grid); t_ds$field <- "deseas"
  f_ds    <- full_leakage(ana_ds, box, config$analysis, config$K_lon, config$K_lat, config$Kt_grid)
  f_ds$field <- "deseas"

  list(time = rbind(t_raw, t_ds), full = rbind(f_raw, f_ds),
       series = list(raw = s_raw, deseas = s_ds), clim = clim, config = config)
}

## ------------------------- plotting ---------------------------------
plot_leakage <- function(res, which = c("time", "full"), main = NULL) {
  which <- match.arg(which); d <- res[[which]]; cfg <- res$config
  r <- d[d$field == "raw", ]; s <- d[d$field == "deseas", ]
  r <- r[order(r$K_t), ]; s <- s[order(s$K_t), ]
  plot(r$K_t, r$L, type = "b", pch = 19, log = "x", ylim = c(0, max(d$L, cfg$tol_L)),
       xlab = expression(K[t] ~ "(time bins)"),
       ylab = expression((sigma[Y]^{minute} / sigma[Y])^2 ~ " within-bin fraction"),
       main = main %||% sprintf("Leakage curve (%s view)", which), col = "firebrick")
  lines(s$K_t, s$L, type = "b", pch = 17, col = "steelblue")
  abline(h = cfg$tol_L, lty = 3, col = "grey50")
  legend("topright", c("raw OHC", "deseasonalized", sprintf("tol = %.2f", cfg$tol_L)),
         col = c("firebrick", "steelblue", "grey50"), pch = c(19, 17, NA),
         lty = c(1, 1, 3), bty = "n")
}
`%||%` <- function(a, b) if (is.null(a)) b else a

## ========================= SELF-TEST ================================
if (!exists("RUN_SELFTEST")) RUN_SELFTEST <- TRUE
if (RUN_SELFTEST) {
  set.seed(1)
  ## synthetic daily GLORYS-like field on the NAC box (coarsened cadence
  ## for test speed): strong spatial front + seasonal cycle + trend + noise.
  box <- list(lon = c(-68, -62), lat = c(36, 40))
  lon <- seq(box$lon[1], box$lon[2], by = 1/12)
  lat <- seq(box$lat[1], box$lat[2], by = 1/12)
  dates <- seq(as.Date("2020-01-01"), as.Date("2022-12-31"), by = 3)   # 3-daily
  grid <- expand.grid(lon = lon, lat = lat, KEEP.OUT.ATTRS = FALSE)
  front <- 70 + 5*(38 - grid$lat) + 2*((-65) - grid$lon)               # dominant spatial
  ## slow mesoscale temporal wiggle (a couple of low-freq modes)
  meso <- function(d) 1.0*sin(2*pi*as.numeric(d)/(370*1.3) + 1) +
                      0.6*sin(2*pi*as.numeric(d)/(180) + 0.3)
  rows <- lapply(dates, function(d) {
    doy   <- as.integer(format(d, "%j"))
    yrfrac<- as.numeric(d - as.Date("2020-01-01")) / 365.25
    k     <- as.numeric(d - as.Date("2020-01-01"))
    seas  <- 8 * cos(2*pi*doy/365 - 0.4)          # regional seasonal cycle
    ## moving spatial mesoscale eddies (~130 km wavelength) -> genuine
    ## sub-cell spatial anomaly structure that survives deseasonalizing
    eddy  <- 3 * sin(2*pi*grid$lon/1.5 + k/40) * sin(2*pi*grid$lat/1.2 + k/55)
    ohc   <- front + seas + 0.6*yrfrac + meso(d) + eddy + rnorm(nrow(grid), 0, 0.4)
    data.frame(lon = grid$lon, lat = grid$lat, date = d, ohc = ohc)
  })
  glorys <- do.call(rbind, rows)
  cat(sprintf("synthetic GLORYS: %d rows (%d cells x %d dates)\n",
              nrow(glorys), nrow(grid), length(dates)))

  cfg <- default_config()
  res <- run_leakage_analysis(glorys, cfg)

  cat("\n== identity check (max |between+within-total|) ==\n")
  cat(sprintf("  time view: %.3e   full view: %.3e  (should be ~0)\n",
              max(abs(res$time$identity_gap)), max(abs(res$full$identity_gap))))

  cat("\n== time-only leakage L(K_t): fraction of REGIONAL-MEAN temporal variance ==\n")
  tt <- res$time
  tt$sd_within_GJ <- sqrt(tt$within)                       # absolute, GJ m^-2
  wide <- reshape(tt[, c("K_t","bin_days","field","L","sd_within_GJ")],
                  idvar = c("K_t","bin_days"), timevar = "field", direction = "wide")
  print(round(wide[order(wide$K_t), ], 3), row.names = FALSE)

  cat("\n== full space x time leakage L(K_t): fraction of FULL sigma_Y^2 (bound-relevant) ==\n")
  ff <- res$full
  ff$sd_within_GJ <- sqrt(ff$within)
  wf <- reshape(ff[, c("K_t","field","L","sd_within_GJ")],
                idvar = "K_t", timevar = "field", direction = "wide")
  print(round(wf[order(wf$K_t), ], 4), row.names = FALSE)
  cat(sprintf("\n  full-view sigma_Y (total, GJ):  raw=%.2f  deseas=%.2f\n",
              sqrt(res$full$total[res$full$field=="raw"][1]),
              sqrt(res$full$total[res$full$field=="deseas"][1])))

  cat("\n== L at K_t in {3,6,12} (time view) ==\n")
  for (kt in c(3,6,12)) {
    lr <- res$time$L[res$time$field=="raw"    & res$time$K_t==kt]
    ls <- res$time$L[res$time$field=="deseas" & res$time$K_t==kt]
    cat(sprintf("  K_t=%2d (%3.0f-day bins):  raw=%.3f  deseas=%.3f\n",
                kt, res$time$bin_days[res$time$K_t==kt][1], lr, ls))
  }

  png("leakage_time.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "time",
      main = "Within-bin variance fraction vs K_t (NAC box, 2020-2022)")
  dev.off()
  png("leakage_full.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "full",
      main = "Full space x time within-cell fraction vs K_t")
  dev.off()
  cat("\nSaved leakage_time.png, leakage_full.png\n")
}
