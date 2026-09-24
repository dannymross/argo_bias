# Vecchia MLE of a stationary anisotropic exponential space-time covariance for
# the simple-kriging OHC estimator.
#
#   k(x,y) = phi * exp(-sqrt((dlon/th_lon)^2 + (dlat/th_lat)^2 + (dt/th_t)^2))
#
# which is exactly GpGp::exponential_scaledim on locs = (lon, lat, day), with
# covparms (variance, range_lon, range_lat, range_t, nugget). NOTE GpGp's nugget
# is the RATIO tau^2: the variance added to the diagonal is phi * tau^2.
#
#   Rscript code/osim/krige_fit.R <sample.csv> <fit.csv> <cache.rds> [m_seq]
#
# sample.csv  columns: lon, lat, day, z
# fit.csv     columns: parameter, estimate, std_error, z_stat, loglik, converged, n_obs

suppressMessages(library(data.table))
suppressMessages(library(GpGp))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: krige_fit.R <sample.csv> <fit.csv> <cache.rds> [m_seq]")
m_seq <- if (length(args) >= 4) as.integer(strsplit(args[4], ",")[[1]]) else c(10L, 30L)
covfun <- if (length(args) >= 5) args[5] else "exponential_scaledim"

d <- fread(args[1])
y <- d$z
locs <- as.matrix(d[, .(lon, lat, day)])

# Scaled so the optimizer sees an O(1) response; undone on the variance below.
sy <- sd(y)
# exponential_scaledim: (variance, range_lon, range_lat, range_t, nugget)
# matern_scaledim:      (variance, range_lon, range_lat, range_t, smoothness, nugget)
start <- if (covfun == "matern_scaledim") {
  c(1.0, 1.3, 1.0, 20.0, 1.0, 0.05)
} else {
  c(1.0, 1.3, 1.0, 20.0, 0.1)
}

cat(sprintf("n = %d, %d sites x %d days\n", nrow(d),
            uniqueN(d[, .(lon, lat)]), uniqueN(d$day)))
fit <- fit_model(y / sy, locs, X = as.matrix(rep(1, nrow(d))),
                 covfun_name = covfun,
                 m_seq = m_seq, start_parms = start, silent = FALSE)

nm <- if (covfun == "matern_scaledim") {
  c("variance", "range_lon_deg", "range_lat_deg", "range_time_days", "smoothness", "nugget_ratio")
} else {
  c("variance", "range_lon_deg", "range_lat_deg", "range_time_days", "nugget_ratio")
}
# Asymptotic SEs from the Fisher information are on the log scale; delta method
# to natural units. fit$info can be near-singular (a GpGp ordering artefact), and
# NA is the honest answer there.
se_log <- tryCatch(sqrt(diag(solve(fit$info))), error = function(e) rep(NA_real_, length(nm)))
se <- fit$covparms * se_log

est <- fit$covparms
est[1] <- est[1] * sy^2                      # undo the response scaling
se[1] <- se[1] * sy^2

out <- data.table(parameter = c(nm, "mean (intercept)"),
                  estimate = c(est, fit$betahat * sy),
                  std_error = c(se, fit$sebeta * sy))
out[, z_stat := estimate / std_error]
out[, loglik := fit$loglik]
out[, converged := fit$conv]
out[, n_obs := nrow(locs)]

saveRDS(list(fit = fit, sd_y = sy, covparms = est, names = nm), args[3])
fwrite(out, args[2])
cat("wrote", args[2], "and", args[3], "\n")
