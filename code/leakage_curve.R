## =====================================================================
## leakage_curve.R -- field factors sigma_{Y^star}, sigma_{Y^perp} and the
##   within-cell residual diagnostic resid = sigma_{Y^perp}/sigma_Y on the
##   equal-measure partition P_K, to inform the choice of K_t, under raw and
##   deseasonalized OHC estimands.  Notation follows the methodology paper
##   (eq:sigmaYK-hat, eq:sigmaYperp-hat, eq:residual-diagnostic).
##
##   Pythagorean split sigma_Y^2 = sigma_{Y^star}^2 + sigma_{Y^perp}^2 with the
##   field on the native 1/12 deg grid, cell moments cos(lat) area-weighted, and
##   equal-measure (1/K) cells.  Computed on the dense GLORYS field, so it is
##   free of the float coincidence-starvation issues that limit sigma_w.
##
##   Two views:
##     time-only : collapse space first -> temporal decomposition of the
##                 regional-mean series (K_lon = K_lat = 1; sets K_t).
##     full      : full space x time equal-measure partition P_K (the
##                 sigma_{Y^perp} that enters the residual term of the bound).
##
##   Parameterized on (box, analysis_period, clim_period): defaults to the
##   NAC box / 2020-2022, but works for any region/period.
##   Base R + data.table (the latter only for loading/very large inputs).
##
##   INPUT: data.frame `glorys` with columns
##     lon (deg), lat (deg), date (Date), ohc (e.g. GJ m^-2), daily 1/12 deg.
##   Real GLORYS input is produced by code/export_glorys_ohc.py (native-grid
##   OHC -> data/glorys_ohc/glorys_ohc_YYYY.csv) and read by load_glorys_ohc().
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
  spatial_sizes_km = c(200,150,100,75,50,35,25,18,12),  # target cell sizes for the spatial sweep
  tol_L    = 0.05,                  # leakage tolerance reference line
  depth    = "700"                  # OHC integration limit for load_glorys_ohc ("700"|"2000")
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

## Load native-grid GLORYS OHC exported by code/export_glorys_ohc.py into the
## `glorys` data.frame run_leakage_analysis expects. `depth` picks which
## integration limit becomes the `ohc` column; rows with NA at that depth
## (column shallower than the seafloor) are dropped.
read_glorys_ohc <- function(dir, years, cols) {
  files <- file.path(dir, sprintf("glorys_ohc_%d.csv", years))
  files <- files[file.exists(files)]
  stopifnot(length(files) > 0)
  dt <- data.table::rbindlist(lapply(files, data.table::fread,
                                     select = c("lon", "lat", "date", cols)))
  dt[, date := as.Date(date)]
  as.data.frame(dt)
}

load_glorys_ohc <- function(dir = file.path("data", "glorys_ohc"),
                            years = 2020:2022, depth = c("700", "2000")) {
  depth <- match.arg(depth)
  dt <- read_glorys_ohc(dir, years, paste0("ohc_", depth))
  names(dt)[names(dt) == paste0("ohc_", depth)] <- "ohc"
  dt[!is.na(dt$ohc), , drop = FALSE]
}

## ---- temporal-only decomposition of the regional-mean series -------
## The K_lon = K_lat = 1 special case of the partition P_K: space is
## collapsed to the area-weighted regional mean Ybar(t) and only time is
## binned into K_t equal-duration bins. Returns the paper's field factors
## for that partition (sigmaYstar2 = resolved / between-bin, sigmaYperp2 =
## within-bin residual) and the residual diagnostic resid = sigmaYperp/sigmaY.
time_leakage <- function(series, period, Kt_grid) {
  t    <- as.numeric(series$date - period[1])
  tmax <- as.numeric(period[2] - period[1])
  y    <- series$ybar; N <- length(y); mu <- mean(y)
  sigmaY2 <- mean((y - mu)^2)                   # equal day weights
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    edges <- seq(0, tmax, length.out = Kt + 1)
    bin   <- findInterval(t, edges, rightmost.closed = TRUE, all.inside = TRUE)
    ybar_b <- tapply(y, bin, mean)
    var_b  <- tapply(y, bin, function(v) mean((v - mean(v))^2))
    wb     <- tapply(y, bin, length) / N
    sigmaYstar2 <- sum(wb * (ybar_b - mu)^2)
    sigmaYperp2 <- sum(wb * var_b)
    data.frame(K_t = Kt, bin_days = tmax / Kt, sigmaY2 = sigmaY2,
               sigmaYstar2 = sigmaYstar2, sigmaYperp2 = sigmaYperp2,
               resid = sqrt(sigmaYperp2 / sigmaY2), L = sigmaYperp2 / sigmaY2,
               identity_gap = sigmaYstar2 + sigmaYperp2 - sigmaY2)
  }))
}

## ---- full space x time field factors on the equal-measure partition P_K ----
## Implements the paper's eq:sigmaYK-hat / eq:sigmaYperp-hat: cells are the
## equal-measure lon x sin(lat) x time boxes (K = K_lon*K_lat*K_t), y is
## evaluated on the native 1/12 deg grid, cell moments ybar_j and Var(y|C_j)
## are cos(lat) area-weighted, and the two field factors weight cells equally
## (1/K, the nominal cell measure):
##   sigmaYstar2 = (1/K) sum_j (ybar_j - muhat)^2,  muhat = (1/K) sum_j ybar_j
##   sigmaYperp2 = (1/K) sum_j Var(y | C_j)
## sigmaY2 := sigmaYstar2 + sigmaYperp2 holds by construction; quad_gap reports
## its departure from the direct cos(lat)-weighted field variance, i.e. the
## error of treating the discrete cells as exactly equal-measure.
full_leakage <- function(dt, box, period, K_lon, K_lat, Kt_grid) {
  w    <- area_weight(dt$lat); y <- dt$ohc
  t    <- as.numeric(dt$date - period[1]); tmax <- as.numeric(period[2] - period[1])
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  bl <- findInterval(dt$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  bb <- findInterval(sin(dt$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  sp <- (bb - 1L) * K_lon + bl
  W  <- sum(w); mu_d <- sum(w * y) / W
  sigmaY2_direct <- sum(w * (y - mu_d)^2) / W   # cos(lat)-weighted field variance
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    e_t  <- seq(0, tmax, length.out = Kt + 1)
    bt   <- findInterval(t, e_t, rightmost.closed = TRUE, all.inside = TRUE)
    cell <- (bt - 1L) * (K_lon * K_lat) + sp
    sw   <- as.numeric(rowsum(w, cell))
    swy  <- as.numeric(rowsum(w * y, cell))
    swy2 <- as.numeric(rowsum(w * y * y, cell))
    ybar_j <- swy / sw                            # area-weighted cell mean
    var_j  <- pmax(0, swy2 / sw - ybar_j^2)       # area-weighted within-cell variance
    Kpop   <- length(sw)                          # populated equal-measure cells
    muhat  <- mean(ybar_j)                         # (1/K) sum_j ybar_j
    sigmaYstar2 <- mean((ybar_j - muhat)^2)
    sigmaYperp2 <- mean(var_j)
    sigmaY2 <- sigmaYstar2 + sigmaYperp2
    data.frame(K_t = Kt, K = Kpop, sigmaY2 = sigmaY2,
               sigmaYstar2 = sigmaYstar2, sigmaYperp2 = sigmaYperp2,
               resid = sqrt(sigmaYperp2 / sigmaY2), L = sigmaYperp2 / sigmaY2,
               identity_gap = sigmaY2 - sigmaYstar2 - sigmaYperp2,
               quad_gap = sigmaY2 - sigmaY2_direct)
  }))
}

## ---- spatial within-cell variance vs spatial cell size (per-snapshot) ----
## Isolates the SPATIAL residual sigma_{Y^perp} that fixes K_lon, K_lat, free
## of seasonality: within each daily snapshot we partition space into
## K_lon x K_lat equal-measure (lon x sin lat) cells, take the cos(lat)
## area-weighted within-cell spatial variance and the total spatial variance,
## then average over snapshots. A spatially-uniform seasonal cycle shifts a
## whole snapshot equally and cancels in that day's spatial variance, so no
## deseasonalizing is needed. L = sigmaYperp2/sigmaY2 is the fraction of
## spatial field variance below the cell scale (eq:residual-diagnostic, sq'd).
spatial_part_grid <- function(box, sizes_km = c(200,150,100,75,50,35,25,18,12)) {
  lat0 <- mean(box$lat)
  wx <- 111 * cos(lat0*pi/180) * diff(box$lon)   # zonal box width, km
  wy <- 111 * diff(box$lat)                       # meridional box width, km
  g <- data.frame(target_km = sizes_km,
                  K_lon = pmax(1, round(wx / sizes_km)),
                  K_lat = pmax(1, round(wy / sizes_km)))
  g <- g[!duplicated(g[c("K_lon","K_lat")]), ]
  g$dx_km <- wx / g$K_lon; g$dy_km <- wy / g$K_lat
  g$cell_km <- sqrt(g$dx_km * g$dy_km); g
}

spatial_leakage <- function(dt, box, part_grid) {
  w <- area_weight(dt$lat); y <- dt$ohc
  di <- match(dt$date, sort(unique(dt$date)))     # snapshot index 1..Ndays
  slon <- dt$lon; ssin <- sin(dt$lat*pi/180)
  dtot <- rowsum(cbind(w, w*y, w*y*y), di)         # per-snapshot totals
  day_totvar <- pmax(0, dtot[,3]/dtot[,1] - (dtot[,2]/dtot[,1])^2)
  sigmaY2 <- mean(day_totvar)                      # mean per-snapshot spatial variance
  do.call(rbind, Map(function(Kl, Kb, tkm, dx, dy, ckm) {
    ncell <- Kl * Kb
    e_lon <- seq(box$lon[1], box$lon[2], length.out = Kl + 1)
    e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = Kb + 1)
    bl <- findInterval(slon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
    bb <- findInterval(ssin, e_lat, rightmost.closed = TRUE, all.inside = TRUE)
    key <- (di - 1L) * ncell + (bb - 1L) * Kl + bl   # unique per (snapshot, cell)
    m  <- rowsum(cbind(w, w*y, w*y*y), key)
    cvar <- pmax(0, m[,3]/m[,1] - (m[,2]/m[,1])^2)   # within-cell spatial variance
    dof  <- (as.integer(rownames(m)) - 1L) %/% ncell  # snapshot of each (snapshot,cell)
    sigmaYperp2 <- mean(tapply(cvar, dof, mean))      # (1/K) over cells, then over days
    data.frame(target_km = tkm, K_lon = Kl, K_lat = Kb, n_cells = ncell,
               dx_km = dx, dy_km = dy, cell_km = ckm,
               sigmaY2 = sigmaY2, sigmaYperp2 = sigmaYperp2,
               resid = sqrt(sigmaYperp2 / sigmaY2), L = sigmaYperp2 / sigmaY2)
  }, part_grid$K_lon, part_grid$K_lat, part_grid$target_km,
     part_grid$dx_km, part_grid$dy_km, part_grid$cell_km))
}

## ============ observed Argo profiles vs GLORYS truth ================
## The Argo profiles realize the sampling measure pi (methodology sec. 2):
## occupancy n_j estimates pi_j = n_j/N, hence the between-cell sampling
## weights wbar_j = K*pi_j and their dispersion sigma_{w^star}. The naive
## profile average is the pi-mean mu_w; its departure from the GLORYS
## nu-mean mu is the realized preferential-sampling bias B_samp. Comparing
## the Argo within/between OHC factors (estimated from the sparse, clustered
## profiles) to the GLORYS ones (the dense-field truth) shows how far the
## observed sample distorts the field's variance structure.

read_argo_ohc <- function(path, cols) {
  dt <- data.table::fread(path, select = c("lon", "lat", "date", cols))
  dt[, date := as.Date(substr(date, 1, 10))]
  as.data.frame(dt)
}

## OHC in argo_ohc.csv is J m^-2; GLORYS tables are GJ m^-2, so rescale.
load_argo_ohc <- function(path = file.path("data", "argo_ohc.csv"),
                          depth = c("700", "2000")) {
  depth <- match.arg(depth)
  dt <- read_argo_ohc(path, paste0("ohc_", depth))
  dt$ohc <- dt[[paste0("ohc_", depth)]] / 1e9
  dt[!is.na(dt$ohc), c("lon", "lat", "date", "ohc")]
}

## Equal-measure lon x sin(lat) x time cell index (1..K_lon*K_lat*K_t) for a
## set of points, shared by the Argo estimators below.
cell_index <- function(dt, box, period, K_lon, K_lat, Kt) {
  t     <- as.numeric(dt$date - period[1]); tmax <- as.numeric(period[2] - period[1])
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  e_t   <- seq(0, tmax, length.out = Kt + 1)
  bl <- findInterval(dt$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  bb <- findInterval(sin(dt$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  bt <- findInterval(t, e_t, rightmost.closed = TRUE, all.inside = TRUE)
  (bt - 1L) * (K_lon * K_lat) + (bb - 1L) * K_lon + bl
}

## Argo within/between-cell OHC factors on the operating partition, per K_t.
## Profiles ARE the sampling measure, so cell means/variances are unweighted
## over profiles (no cos(lat): the sample already carries the sampling
## intensity). Field factors use equal cell weight 1/K_occ over occupied
## cells (all K occupied at the operating grid). sigma_{w^star} is taken over
## ALL K nominal cells, empty cells contributing (0-1)^2. mu_naive is the
## pi-mean (simple profile average); mu_grid the equal-measure cell-averaged
## estimator; both are compared to the GLORYS nu-mean mu_truth.
argo_leakage <- function(argo, box, period, K_lon, K_lat, Kt_grid, mu_truth) {
  y <- argo$ohc; N <- length(y); mu_naive <- mean(y)
  do.call(rbind, lapply(Kt_grid, function(Kt) {
    Kall <- K_lon * K_lat * Kt
    cell <- cell_index(argo, box, period, K_lon, K_lat, Kt)
    n_j    <- as.numeric(table(cell))
    ybar_j <- tapply(y, cell, mean)
    var_j  <- tapply(y, cell, function(v) mean((v - mean(v))^2))
    Kocc   <- length(ybar_j)
    muhat  <- mean(ybar_j)
    sigmaYstar2 <- mean((ybar_j - muhat)^2)
    sigmaYperp2 <- mean(var_j)
    sigma_wstar2 <- (sum((Kall * n_j / N - 1)^2) + (Kall - Kocc)) / Kall
    data.frame(K_t = Kt, K = Kall, n_prof = N, n_occ = Kocc,
               sigmaYstar2 = sigmaYstar2, sigmaYperp2 = sigmaYperp2,
               sigmaY2 = sigmaYstar2 + sigmaYperp2,
               resid = sqrt(sigmaYperp2 / (sigmaYstar2 + sigmaYperp2)),
               L = sigmaYperp2 / (sigmaYstar2 + sigmaYperp2),
               sigma_wstar = sqrt(sigma_wstar2),
               mu_truth = mu_truth, mu_naive = mu_naive, mu_grid = muhat,
               bias_naive = mu_naive - mu_truth, bias_grid = muhat - mu_truth)
  }))
}

## Per-cell Argo occupancy on the operating spatial grid (pooled over time):
## profile count n_j, sampling weight wbar_j = K*pi_j, and the sample cell
## mean/SD -- the concrete preferential-sampling map.
argo_occupancy <- function(argo, box, K_lon, K_lat) {
  e_lon <- seq(box$lon[1], box$lon[2], length.out = K_lon + 1)
  e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = K_lat + 1)
  bl <- findInterval(argo$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
  bb <- findInterval(sin(argo$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
  clon <- (e_lon[-1] + e_lon[-(K_lon+1)]) / 2
  clat <- asin((e_lat[-1] + e_lat[-(K_lat+1)]) / 2) * 180 / pi
  K <- K_lon * K_lat; N <- nrow(argo)
  grid <- expand.grid(bl = seq_len(K_lon), bb = seq_len(K_lat))
  cell <- (grid$bb - 1L) * K_lon + grid$bl
  key  <- (bb - 1L) * K_lon + bl
  n_j  <- as.integer(table(factor(key, levels = cell)))
  ybar <- tapply(argo$ohc, factor(key, levels = cell), mean)
  ysd  <- tapply(argo$ohc, factor(key, levels = cell), sd)
  data.frame(cell_lon = clon[grid$bl], cell_lat = clat[grid$bb],
             n = n_j, wbar = K * n_j / N,
             ohc_mean = as.numeric(ybar), ohc_sd = as.numeric(ysd))
}

## Spatial sampling floor: as cells shrink, how many can the ~N profiles keep
## populated (>=2, needed for a within-cell variance)? Pooled over time.
argo_occupancy_sweep <- function(argo, box, part_grid) {
  N <- nrow(argo)
  do.call(rbind, Map(function(Kl, Kb, tkm, ckm) {
    e_lon <- seq(box$lon[1], box$lon[2], length.out = Kl + 1)
    e_lat <- seq(sin(box$lat[1]*pi/180), sin(box$lat[2]*pi/180), length.out = Kb + 1)
    bl <- findInterval(argo$lon, e_lon, rightmost.closed = TRUE, all.inside = TRUE)
    bb <- findInterval(sin(argo$lat*pi/180), e_lat, rightmost.closed = TRUE, all.inside = TRUE)
    n_j <- as.numeric(table((bb - 1L) * Kl + bl))
    K <- Kl * Kb
    data.frame(target_km = tkm, K_lon = Kl, K_lat = Kb, n_cells = K, cell_km = ckm,
               n_prof = N, occ = length(n_j), occ_ge2 = sum(n_j >= 2),
               frac_ge2 = sum(n_j >= 2) / K, min_n = min(n_j), median_n = median(n_j),
               expected_per_cell = N / K)
  }, part_grid$K_lon, part_grid$K_lat, part_grid$target_km, part_grid$cell_km))
}

## GLORYS nu-mean (cos(lat)-and-time-uniform regional-mean OHC) over box+period.
glorys_truth_mean <- function(glorys, box, period) {
  d <- filter_domain(glorys, box, period)
  w <- area_weight(d$lat); sum(w * d$ohc) / sum(w)
}

## ------------------------- driver -----------------------------------
run_leakage_analysis <- function(glorys, config = default_config()) {
  box <- config$box
  ana <- filter_domain(glorys, box, config$analysis)
  stopifnot(nrow(ana) > 0)

  ## spatial resolution (raw, per-snapshot: seasonality-free by construction)
  sp <- spatial_leakage(ana, box, spatial_part_grid(box, config$spatial_sizes_km))
  sp$field <- "raw"

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

  list(time = rbind(t_raw, t_ds), full = rbind(f_raw, f_ds), spatial = sp,
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

## cairo-backed PNG so plots render on a headless node (no X11 display).
open_png <- function(file, ...) png(file,
  type = if (isTRUE(capabilities("cairo"))) "cairo" else "Xlib", ...)

## Run the leakage analysis for one or more depths on the real GLORYS field and
## write tidy tables (columns include `depth`, `field`; the time table also has
## `bin_days`) for the resolution.qmd report. Reads the CSVs once and swaps the
## `ohc` column per depth. Consumed by code/leakage.py.
write_leakage_tables <- function(out_time, out_full, out_spatial,
                                 depths = c("700", "2000"),
                                 config = default_config(),
                                 dir = file.path("data", "glorys_ohc")) {
  yrs  <- as.integer(format(config$analysis, "%Y"))
  base <- read_glorys_ohc(dir, yrs[1]:yrs[2], paste0("ohc_", depths))
  tt <- list(); ff <- list(); ss <- list()
  for (d in depths) {
    g <- base; g$ohc <- g[[paste0("ohc_", d)]]; g <- g[!is.na(g$ohc), , drop = FALSE]
    res <- run_leakage_analysis(g, config)
    res$time$depth <- d; ff_d <- res$full; ff_d$depth <- d; res$spatial$depth <- d
    tt[[d]] <- res$time; ff[[d]] <- ff_d; ss[[d]] <- res$spatial
  }
  dir.create(dirname(out_time), showWarnings = FALSE, recursive = TRUE)
  data.table::fwrite(do.call(rbind, tt), out_time)
  data.table::fwrite(do.call(rbind, ff), out_full)
  data.table::fwrite(do.call(rbind, ss), out_spatial)
}

## Observed-Argo-vs-GLORYS comparison tables for resolution.qmd: the per-K_t
## Argo within/between factors + realized bias, the operating-grid occupancy
## map, and the spatial sampling-floor sweep. Reads argo_ohc.csv and the
## GLORYS field once, per depth. Consumed by code/resolution.py.
write_argo_tables <- function(out_argo, out_occ, out_sweep,
                              depths = c("700", "2000"),
                              config = default_config(),
                              glorys_dir = file.path("data", "glorys_ohc"),
                              argo_path = file.path("data", "argo_ohc.csv")) {
  box <- config$box; yrs <- as.integer(format(config$analysis, "%Y"))
  gbase <- read_glorys_ohc(glorys_dir, yrs[1]:yrs[2], paste0("ohc_", depths))
  aa <- list(); oo <- list(); ss <- list()
  for (d in depths) {
    g <- gbase; g$ohc <- g[[paste0("ohc_", d)]]; g <- g[!is.na(g$ohc), , drop = FALSE]
    mu_truth <- glorys_truth_mean(g, box, config$analysis)
    argo <- filter_domain(load_argo_ohc(argo_path, d), box, config$analysis)
    a <- argo_leakage(argo, box, config$analysis, config$K_lon, config$K_lat,
                      config$Kt_grid, mu_truth);                a$depth <- d
    o <- argo_occupancy(argo, box, config$K_lon, config$K_lat);  o$depth <- d
    s <- argo_occupancy_sweep(argo, box, spatial_part_grid(box, config$spatial_sizes_km))
    s$depth <- d
    aa[[d]] <- a; oo[[d]] <- o; ss[[d]] <- s
  }
  dir.create(dirname(out_argo), showWarnings = FALSE, recursive = TRUE)
  data.table::fwrite(do.call(rbind, aa), out_argo)
  data.table::fwrite(do.call(rbind, oo), out_occ)
  data.table::fwrite(do.call(rbind, ss), out_sweep)
}

## CLI: Rscript code/leakage_curve.R <out_time> <out_full> <out_spatial> [depths]
##   or  Rscript code/leakage_curve.R argo <out_argo> <out_occ> <out_sweep> [depths]
## depths is a comma-separated subset of {700,2000} (default both). Skips the
## interactive self-test / RUN_GLORYS blocks below.
LEAKAGE_CLI_ARGS <- commandArgs(trailingOnly = TRUE)
if (length(LEAKAGE_CLI_ARGS) >= 4 && LEAKAGE_CLI_ARGS[1] == "argo") {
  depths <- if (length(LEAKAGE_CLI_ARGS) >= 5)
    strsplit(LEAKAGE_CLI_ARGS[5], ",")[[1]] else c("700", "2000")
  write_argo_tables(LEAKAGE_CLI_ARGS[2], LEAKAGE_CLI_ARGS[3],
                    LEAKAGE_CLI_ARGS[4], depths)
} else if (length(LEAKAGE_CLI_ARGS) >= 3) {
  depths <- if (length(LEAKAGE_CLI_ARGS) >= 4)
    strsplit(LEAKAGE_CLI_ARGS[4], ",")[[1]] else c("700", "2000")
  write_leakage_tables(LEAKAGE_CLI_ARGS[1], LEAKAGE_CLI_ARGS[2],
                       LEAKAGE_CLI_ARGS[3], depths)
}

## ===================== REAL GLORYS DRIVER ===========================
## Opt in with RUN_GLORYS <- TRUE before sourcing (skips the synthetic
## self-test). Reads data/glorys_ohc/ (see code/export_glorys_ohc.py).
if (!exists("RUN_GLORYS")) RUN_GLORYS <- FALSE
if (RUN_GLORYS) {
  cfg <- default_config()
  glorys <- load_glorys_ohc(years = as.integer(format(cfg$analysis, "%Y"))[1]:
                                    as.integer(format(cfg$analysis, "%Y"))[2],
                            depth = cfg$depth)
  cat(sprintf("GLORYS OHC (ohc_%s): %d rows, %d dates, box lon[%g,%g] lat[%g,%g]\n",
              cfg$depth, nrow(glorys), length(unique(glorys$date)),
              cfg$box$lon[1], cfg$box$lon[2], cfg$box$lat[1], cfg$box$lat[2]))
  res <- run_leakage_analysis(glorys, cfg)
  cat(sprintf("identity gap (max |between+within-total|): time %.2e  full %.2e\n",
              max(abs(res$time$identity_gap)), max(abs(res$full$identity_gap))))
  tt <- res$time[order(res$time$field, res$time$K_t),
                 c("field","K_t","bin_days","sigmaY2","sigmaYperp2","resid","L")]
  num <- sapply(tt, is.numeric); tt[num] <- round(tt[num], 4)
  print(tt, row.names = FALSE)
  open_png("leakage_time.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "time",
      main = "Within-bin variance fraction vs K_t (GLORYS, NAC box)")
  dev.off()
  open_png("leakage_full.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "full",
      main = "Full space x time within-cell fraction vs K_t (GLORYS)")
  dev.off()
  cat("Saved leakage_time.png, leakage_full.png\n")
}

## ========================= SELF-TEST ================================
if (!exists("RUN_SELFTEST")) RUN_SELFTEST <- !RUN_GLORYS && length(LEAKAGE_CLI_ARGS) < 3
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

  cat("\n== temporal-only residual (regional-mean series): resid = sigmaYperp/sigmaY ==\n")
  tt <- res$time
  tt$sd_perp_GJ <- sqrt(tt$sigmaYperp2)                    # absolute, GJ m^-2
  wide <- reshape(tt[, c("K_t","bin_days","field","resid","sd_perp_GJ")],
                  idvar = c("K_t","bin_days"), timevar = "field", direction = "wide")
  print(round(wide[order(wide$K_t), ], 3), row.names = FALSE)

  cat("\n== full space x time residual: resid = sigmaYperp/sigmaY (bound-relevant) ==\n")
  ff <- res$full
  ff$sd_perp_GJ <- sqrt(ff$sigmaYperp2)
  wf <- reshape(ff[, c("K_t","field","resid","sd_perp_GJ")],
                idvar = "K_t", timevar = "field", direction = "wide")
  print(round(wf[order(wf$K_t), ], 4), row.names = FALSE)
  cat(sprintf("\n  full-view sigma_Y (GJ):  raw=%.2f  deseas=%.2f\n",
              sqrt(res$full$sigmaY2[res$full$field=="raw"][1]),
              sqrt(res$full$sigmaY2[res$full$field=="deseas"][1])))

  cat("\n== L at K_t in {3,6,12} (time view) ==\n")
  for (kt in c(3,6,12)) {
    lr <- res$time$L[res$time$field=="raw"    & res$time$K_t==kt]
    ls <- res$time$L[res$time$field=="deseas" & res$time$K_t==kt]
    cat(sprintf("  K_t=%2d (%3.0f-day bins):  raw=%.3f  deseas=%.3f\n",
                kt, res$time$bin_days[res$time$K_t==kt][1], lr, ls))
  }

  open_png("leakage_time.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "time",
      main = "Within-bin variance fraction vs K_t (NAC box, 2020-2022)")
  dev.off()
  open_png("leakage_full.png", width = 720, height = 520, res = 110)
  par(mar = c(4.5, 4.8, 3, 1)); plot_leakage(res, "full",
      main = "Full space x time within-cell fraction vs K_t")
  dev.off()
  cat("\nSaved leakage_time.png, leakage_full.png\n")
}
