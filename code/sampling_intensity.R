library(data.table)
library(ggplot2)
library(lubridate)

EARTH_RADIUS_KM <- 6371.0

sec_to_day <- function(d) d / (60 * 60 * 24)

# Core Matérn space-time kernel
matern_st <- function(x1, x2 = NULL, nu = 0.5, rs, rt, C = NA, marginal = FALSE) {
  if (marginal) return(rep(1.0, nrow(x1)))
  if (is.vector(x1)) x1 <- matrix(x1, nrow = 1)
  if (is.vector(x2)) x2 <- matrix(x2, nrow = 1)
  
  lon1_r <- x1[, 1] * pi / 180;  lat1_r <- x1[, 2] * pi / 180
  lon2_r <- x2[, 1] * pi / 180;  lat2_r <- x2[, 2] * pi / 180
  
  s1 <- cbind(cos(lat1_r) * cos(lon1_r), cos(lat1_r) * sin(lon1_r), sin(lat1_r)) / rs
  s2 <- cbind(cos(lat2_r) * cos(lon2_r), cos(lat2_r) * sin(lon2_r), sin(lat2_r)) / rs
  
  t1 <- x1[, 3] / rt
  t2 <- x2[, 3] / rt
  
  D2s <- (outer(rowSums(s1^2), rep(1.0, nrow(x2)))
        + outer(rep(1.0, nrow(x1)), rowSums(s2^2))
        - 2 * tcrossprod(s1, s2))
  D2t <- outer(t1, t2, function(a, b) (a - b)^2)
  
  r <- sqrt(pmax(D2s + D2t, 0))  
  
  if (nu == 0.5) K <- exp(-r)
  if (nu == 1.5) K <- (1 + r) * exp(-r)
  
  if (length(C) == 1 && is.na(C)) K else K %*% C
}


# Load, filter, and format Argo data
prep_argo_data <- function(file_path, lat_bounds, lon_bounds, years) {
  argo <- fread(file_path)
  argo[, year := year(date)]
  
  A <- argo[
    between(lat, lat_bounds[1], lat_bounds[2]) &
    between(lon, lon_bounds[1], lon_bounds[2]) &
    (year %in% years), 
  ]
  A[, t := sec_to_day(as.numeric(date))]
  
  list(
    data = A,
    coords = as.matrix(A[, .(lon, lat, t)])
  )
}

# Generate uniform space-time samples across the bounding box
generate_mc_samples <- function(M, lat_bounds, lon_bounds, t_bounds) {
  sin_lat <- runif(M, sin(lat_bounds[1] * pi / 180), sin(lat_bounds[2] * pi / 180))
  
  mc <- data.table(
    lon = runif(M, lon_bounds[1], lon_bounds[2]),
    lat = asin(sin_lat) * 180 / pi,
    t   = sec_to_day(runif(M, t_bounds[1], t_bounds[2]))
  )
  
  list(
    data = mc,
    coords = as.matrix(mc[, .(lon, lat, t)])
  )
}

# ------------------------------------------------------------
# 3. Modeling and Prediction Functions
# ------------------------------------------------------------

#' Fit the sampling density model and compute the normalizing constant
fit_sampling_density <- function(obs_coords, mc_coords, rs, rt, nu = 0.5) {
  
  # Raw Intensity at observed locations
  K_obs <- matern_st(x1 = obs_coords, x2 = obs_coords, nu = nu, rs = rs, rt = rt)
  pi_obs_raw <- rowMeans(K_obs)
  
  # Raw Intensity at Monte Carlo locations
  K_mc <- matern_st(x1 = mc_coords, x2 = obs_coords, nu = nu, rs = rs, rt = rt)
  pi_mc_raw <- rowMeans(K_mc)
  
  # Normalizing constant Z
  Z <- mean(pi_mc_raw)
  
  # Return the fitted model object
  list(
    Z = Z,
    rs = rs,
    rt = rt,
    nu = nu,
    pi_obs_norm = pi_obs_raw / Z,
    pi_mc_norm = pi_mc_raw / Z
  )
}

calc_sigma_w <- function(mc_w) {
  # mc_w represents w(x) evaluated at the uniformly drawn MC points
  var_hat <- mean((mc_w - 1)^2)
  sqrt(var_hat)
}

predict_density <- function(new_coords, obs_coords, model_fit) {
  if (is.vector(new_coords)) new_coords <- matrix(new_coords, nrow = 1)
  
  K_new <- matern_st(
    x1 = new_coords,
    x2 = obs_coords, 
    nu = model_fit$nu,
    rs = model_fit$rs,
    rt = model_fit$rt
  )
  
  rowMeans(K_new) / model_fit$Z
}

#' Predict w(x) on a common lon/lat grid at each of a set of target times,
#' returning one long-format table (month_label identifies each time slice).
#' Using a single grid across months, rather than one grid per plot, is what
#' lets the monthly panels later share one continuous fill scale.
build_monthly_density_grid <- function(target_dates, obs_coords, model_fit, lon_bounds, lat_bounds, grid_res = 100) {
  lon_seq <- seq(lon_bounds[1], lon_bounds[2], length.out = grid_res)
  lat_seq <- seq(lat_bounds[1], lat_bounds[2], length.out = grid_res)
  base_grid <- as.data.table(expand.grid(lon = lon_seq, lat = lat_seq))

  rbindlist(lapply(target_dates, function(d) {
    g <- copy(base_grid)
    g[, t := sec_to_day(as.numeric(as.POSIXct(paste(d, "12:00:00"), tz = "UTC")))]
    g[, w_hat := predict_density(as.matrix(g[, .(lon, lat, t)]), obs_coords, model_fit)]
    g[, month_label := format(d, "%B %Y")]
    g
  }))
}

plot_monthly_density_grid <- function(grid_data, obs_data, month_levels, sigma_w, ncol = 4) {
  grid_data <- copy(grid_data)
  obs_data <- copy(obs_data)
  grid_data[, month_label := factor(month_label, levels = month_levels)]
  obs_data[, month_label := factor(month_label, levels = month_levels)]

  ggplot() +
    geom_raster(data = grid_data, aes(x = lon, y = lat, fill = w_hat), interpolate = TRUE) +
    scale_fill_gradient2(
      midpoint = 1, low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
      name = "w(x)", limits = c(0, NA)
    ) +
    geom_point(data = obs_data, aes(x = lon, y = lat),
               color = "black", size = 0.6, alpha = 0.6, shape = 16) +
    facet_wrap(~ month_label, ncol = ncol) +
    labs(
      title = "Space-Time Sampling Intensity",
      subtitle = bquote(sigma[w] ~ ":" ~ .(round(sigma_w, 4))),
      x = "", y = ""
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      strip.text = element_text(face = "bold", size = 10),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 6)
    )
}

## run
set.seed(6737)
LAT_BNDS <- c(36, 40)
LON_BNDS <- c(-68, -62)
T_BNDS   <- as.numeric(as.POSIXct(c("2020-01-01 00:00:00", "2022-12-31 23:59:59"), tz = "UTC"))
YEARS    <- c(2020, 2021, 2022)

RS <- 10 / EARTH_RADIUS_KM
RT <- 365
NU <- 0.5
M  <- 10000

argo_obj <- prep_argo_data("data/argo_ohc.csv", LAT_BNDS, LON_BNDS, YEARS)
mc_obj   <- generate_mc_samples(M, LAT_BNDS, LON_BNDS, T_BNDS)

model <- fit_sampling_density(
  obs_coords = argo_obj$coords, 
  mc_coords  = mc_obj$coords, 
  rs = RS, 
  rt = RT, 
  nu = NU
)

hat_sigma_w <- calc_sigma_w(model$pi_mc_norm)
print(hat_sigma_w)

# Set the year you want to visualize
target_year <- 2021
GRID_RES <- 40

month_dates <- make_date(target_year, 1:12, 15)
month_labels <- format(month_dates, "%B %Y")

grid_data <- build_monthly_density_grid(
  target_dates = month_dates,
  obs_coords   = argo_obj$coords,
  model_fit    = model,
  lon_bounds   = LON_BNDS,
  lat_bounds   = LAT_BNDS,
  grid_res     = GRID_RES
)

obs_data <- rbindlist(lapply(seq_along(month_dates), function(m) {
  argo_obj$data[year(date) == target_year & month(date) == m][, month_label := month_labels[m]]
}))

combined_grid <- plot_monthly_density_grid(
  grid_data    = grid_data,
  obs_data     = obs_data,
  month_levels = month_labels,
  sigma_w      = hat_sigma_w
)

print(combined_grid)

