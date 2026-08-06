## =====================================================================
## surrogate_designs.R -- synthetic sampling designs with a known answer, used
##   to check that the permutation test is calibrated (null) and that it points
##   the right way (power), before real Lagrangian trajectories exist.
##
##   Both generators emit profile POSITIONS in the same schema as
##   code/osse/export_profiles.py will, so surrogate runs go through the real
##   pipeline -- native cell-day gather, day-of-year alignment, Zbar_k
##   subtraction -- rather than bypassing it. A bug anywhere downstream shows
##   up here as a failed calibration check.
##
##   Positions are proposed on the coarse residual field (residual_coarse_*.nc)
##   because generating preferential positions is a normalize-and-draw over the
##   whole domain, unlike the real path which only gathers at known points. This
##   affects only WHERE floats are placed; every position is then evaluated
##   against the native field.
## =====================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ncdf4)
})

## ------------------------- proposal field ---------------------------

## Reads residual_coarse_<tag>.nc into a flat cell table:
##   cell_lat, cell_lon, doy_bin, year, z, p (cell measure, sums to 1 per year)
read_coarse_field <- function(tag = "boxB", dir = "data/osse") {
  nc <- nc_open(file.path(dir, sprintf("residual_coarse_%s.nc", tag)))
  on.exit(nc_close(nc))

  z <- ncvar_get(nc, "z")                 # (cell_lon, cell_lat, doy_bin, year)
  area <- ncvar_get(nc, "cell_area")      # (cell_lon, cell_lat)
  ndays <- ncvar_get(nc, "n_days")        # (doy_bin)
  lat <- ncvar_get(nc, "cell_lat")
  lon <- ncvar_get(nc, "cell_lon")
  doy_bin <- ncvar_get(nc, "doy_bin")
  year <- ncvar_get(nc, "year")

  # ncvar_get returns (cell_lon, cell_lat, doy_bin, year) and as.vector() varies
  # the FIRST dimension fastest, so the grid must be ordered with cell_lon
  # fastest and year slowest to line up without any permutation.
  # Both vectors are built out here: inside a data.table j-expression, column
  # names shadow local variables, so length(year) would be nrow(grid).
  nlon <- length(lon); nlat <- length(lat); nbin <- length(doy_bin); nyr <- length(year)
  p_vec <- rep(as.vector(area), times = nbin * nyr) *
           rep(rep(ndays, each = nlon * nlat), times = nyr)
  z_vec <- as.vector(z)

  grid <- CJ(year = year, doy_bin = doy_bin, cell_lat = lat, cell_lon = lon, sorted = FALSE)
  setorder(grid, year, doy_bin, cell_lat, cell_lon)
  stopifnot(nrow(grid) == length(z_vec), nrow(grid) == length(p_vec))
  set(grid, j = "z", value = z_vec)
  set(grid, j = "p", value = p_vec)

  # Pin the flattening: a few cells checked straight against the raw array.
  chk <- sample.int(nrow(grid), 20L)
  stopifnot(all(mapply(function(i) {
    g <- grid[i]
    a <- z[match(g$cell_lon, lon), match(g$cell_lat, lat),
           match(g$doy_bin, doy_bin), match(g$year, year)]
    isTRUE(all.equal(a, g$z)) || (is.na(a) && is.na(g$z))
  }, chk)))

  grid[!is.finite(z), p := 0]
  grid[, p := p / sum(p), by = year]
  setattr(grid, "n_days", ndays)
  grid[]
}

## Day-of-year bin -> a uniformly drawn day within that bin, so positions carry
## a native doy. Bin widths come from the file's n_days rather than a nominal
## width: 365 does not divide evenly, so the final bin is wider and a fixed
## width would never sample its last few days.
bin_to_doy <- function(doy_bin, n_days) {
  start <- c(0L, cumsum(as.integer(n_days))[-length(n_days)])
  width <- as.integer(n_days)[doy_bin + 1L]
  start[doy_bin + 1L] + floor(runif(length(doy_bin)) * width)
}

## Jitter a coarse cell centre to a position inside it, so the native gather
## sees a spread of fine cells rather than one repeated point.
jitter_position <- function(cell_lat, cell_lon, deg = 1) {
  list(lat = cell_lat + runif(length(cell_lat), -deg / 2, deg / 2),
       lon = cell_lon + runif(length(cell_lon), -deg / 2, deg / 2))
}

## --------------------------- generators -----------------------------

## Draw `n` positions for one design from cell weights `w` over `cells`.
draw_positions <- function(cells, w, n, n_days, deg = 1) {
  idx <- sample.int(nrow(cells), n, replace = TRUE, prob = w)
  sel <- cells[idx]
  pos <- jitter_position(sel$cell_lat, sel$cell_lon, deg)
  data.table(lat = pos$lat, lon = pos$lon,
             doy = bin_to_doy(sel$doy_bin, n_days))
}

## Exogenous designs: positions drawn from the cell measure alone, independent
## of Z. The permutation test must return uniform p-values on these.
sim_designs_null <- function(field, M, D, n_profiles, n_days, deg = 1) {
  vel_years <- sort(unique(field$year))[seq_len(M)]
  out <- vector("list", M * D)
  i <- 0L
  for (l in seq_len(M)) {
    cells <- field[year == vel_years[l]]
    for (d in seq_len(D)) {
      i <- i + 1L
      out[[i]] <- cbind(vel_year = vel_years[l], rep = d,
                        draw_positions(cells, cells$p, n_profiles, n_days, deg))
    }
  }
  rbindlist(out)
}

## Preferential designs: intensity proportional to p_j * exp(beta * Z_lj), i.e.
## floats in velocity year l prefer cells where year l's residual field is
## warm. beta = 0 recovers the null generator exactly.
sim_designs_pref <- function(field, M, D, n_profiles, n_days, beta, deg = 1) {
  vel_years <- sort(unique(field$year))[seq_len(M)]
  out <- vector("list", M * D)
  i <- 0L
  for (l in seq_len(M)) {
    cells <- field[year == vel_years[l]]
    # Standardise Z so beta is in units of field SD and comparable across years.
    zs <- (cells$z - mean(cells$z, na.rm = TRUE)) / sd(cells$z, na.rm = TRUE)
    w <- cells$p * exp(beta * zs)
    w[!is.finite(w)] <- 0
    for (d in seq_len(D)) {
      i <- i + 1L
      out[[i]] <- cbind(vel_year = vel_years[l], rep = d,
                        draw_positions(cells, w, n_profiles, n_days, deg))
    }
  }
  rbindlist(out)
}
