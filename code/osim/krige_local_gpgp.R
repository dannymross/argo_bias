# Per-window Vecchia MLE of Kuusela's locally stationary anisotropic exponential space-time covariance,
# GpGp::exponential_scaledim on (lon, lat, t). Years are independent replicates: each year's times are
# shifted by GAP days, so no cross-year correlation or Vecchia neighbour enters.
#
#   Rscript code/osim/krige_local_gpgp.R <points.csv> <centres.csv> <out.csv> <years> [m]
#
# points.csv  lon, lat, day (since 2010-01-01), year, z   (z mean-removed)
# centres.csv lat, lon, month
# GpGp's nugget is the ratio tau^2 (nugget variance = phi * tau^2).

suppressMessages({library(data.table); library(GpGp)})
a <- commandArgs(trailingOnly = TRUE)
pts <- fread(a[1])
cen <- fread(a[2])
years <- as.integer(strsplit(a[4], ",")[[1]])
m <- if (length(a) >= 5) as.integer(a[5]) else 30L
LG <- 10; LT <- 45; GAP <- 1e5; MIN_BLOCK <- 50
NM <- c("phi", "th_lon", "th_lat", "th_t", "nugget_ratio")

fit1 <- function(w) {
  s <- pts[lat %between% c(w$lat - LG, w$lat + LG) & lon %between% c(w$lon - LG, w$lon + LG) & year %in% years]
  tstar <- as.numeric(as.Date(sprintf("%d-%02d-15", years, w$month)) - as.Date("2010-01-01"))
  s[, dt := day - tstar[match(year, years)]]
  s <- s[abs(dt) <= LT]
  s <- s[year %in% s[, .N, by = year][N > MIN_BLOCK, year]]
  nb <- uniqueN(s$year)
  if (nb < 2) return(NULL)
  sy <- sd(s$z)
  locs <- cbind(s$lon, s$lat, s$dt + GAP * match(s$year, years))
  t0 <- proc.time()[["elapsed"]]
  f <- tryCatch(fit_model(s$z / sy, locs, X = matrix(1, nrow(s), 1), covfun_name = "exponential_scaledim",
                          start_parms = c(1, 1.3, 1, 20, 0.05), m_seq = c(10L, m), silent = TRUE),
                error = function(e) NULL)
  if (is.null(f)) return(data.table(lat = w$lat, lon = w$lon, month = w$month, n_obs = nrow(s), n_blocks = nb))
  se <- f$covparms * tryCatch(sqrt(diag(solve(f$info))), error = function(e) rep(NA_real_, 5))
  p <- f$covparms; p[1] <- p[1] * sy^2; se[1] <- se[1] * sy^2
  data.table(lat = w$lat, lon = w$lon, month = w$month, n_obs = nrow(s), n_blocks = nb,
             t(setNames(p, NM)), t(setNames(se, paste0("se_", NM))),
             loglik = f$loglik - nrow(s) * log(sy), mean = f$betahat * sy, conv = f$conv,
             secs = proc.time()[["elapsed"]] - t0)
}

out <- rbindlist(lapply(seq_len(nrow(cen)), function(i) {
  r <- fit1(cen[i])
  if (i %% 10 == 0) cat(sprintf("%d/%d windows\n", i, nrow(cen)))
  r
}), fill = TRUE)
fwrite(out, a[3])
cat(sprintf("wrote %d windows -> %s (median %.1f s/window)\n", nrow(out), a[3], median(out$secs, na.rm = TRUE)))
