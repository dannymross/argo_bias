root <- getwd()
library(tidyverse)
library(pbapply)

# library(BayesianOHC)
# --- presumed functions from BayesianOHC
deg2rad <- function(deg) {
  deg * (pi / 180)
}

trapz <- function(x, y) {
  n <- length(x)
  sum(diff(x) * (y[-1] + y[-n]) / 2)
}
# ---

test_mode <- F
#' 01' is for january:
month_id <- 01
sw_sh <<- 3989.411
sw_density <<- 1028.319

raw_data_filename <- paste("/data/argo_object_",
  formatC(month_id, width = 2, flag = "0"), ".RData",
  sep = ""
)

if (file.exists(paste0(root, raw_data_filename))) {
  load(paste0(root, raw_data_filename), verb = T)
} else {
  print("No file")
}

max_depth <- unlist(lapply(mat_object$profPresAggr, function(x) max(x[[1]])))
lon_unshifted <- mat_object$profLongAggr[1, ]
lon_shifted <- ifelse(lon_unshifted > 180, lon_unshifted - 360, lon_unshifted)

profile_df <- data.frame(
  lat_degrees = mat_object$profLatAggr[1, ], lon_degrees = lon_shifted,
  lat_rad = deg2rad(mat_object$profLatAggr[1, ]), lon_rad = deg2rad(lon_shifted),
  max_depth = max_depth, years = mat_object$years,
  days = mat_object$days, months = mat_object$months, hours = mat_object$hours,
  minutes = mat_object$minutes, seconds = mat_object$seconds,
  time = mat_object$days * 24 * 60 * 60 +
    mat_object$hours * 60 * 60 +
    mat_object$minutes * 60 +
    mat_object$seconds,
  float_id = mat_object$profFloatIDAggr[1, ]
)

mm <- length(mat_object$days)

calculate_vhc <- function(ii) {
  obs_pressures <- mat_object$profPresAggr[ii][[1]][[1]]
  obs_pressures_clean <- obs_pressures[obs_pressures > 0 & obs_pressures < 2000]
  if (length(obs_pressures_clean) == 0) {
    return(data.frame("vhc_obs" = NA, "obs_pressures" = NA, "obs_temps" = NA))
  }
  obs_pressures_endpoints <- c(0, obs_pressures_clean, 2000)
  obs_temps <- mat_object$profTempAggr[ii][[1]][[1]][obs_pressures > 0 & obs_pressures < 2000]
  obs_temps_endpoints <- c(obs_temps[1], obs_temps, obs_temps[length(obs_temps)])
  vhc_obs <- trapz(
    obs_pressures_endpoints,
    obs_temps_endpoints * sw_sh * sw_density
  )
  return(data.frame(
    "vhc_obs" = vhc_obs,
    "obs_pressures" = I(list(obs_pressures_endpoints)),
    "obs_temps" = I(list(obs_temps_endpoints))
  ))
}

vhc_df <- do.call(rbind, pblapply(1:mm, calculate_vhc))
profile_df <- cbind(profile_df, vhc_df)

# scale vhc_obs to giga-joules and time to days since Jan 1
argo_data_unordered <- profile_df %>%
  dplyr::filter(max_depth > 1900) %>%
  mutate(vhc_obs = vhc_obs / 10^9) %>%
  mutate(time = (time - min(time)) / (60^2 * 24)) %>%
  mutate(z = vhc_obs) %>%
  dplyr::filter(., !duplicated(.[, c("lat_degrees", "lon_degrees", "float_id")]))

# argo_data_ordered <- order_yeardata(argo_data_unordered)

if (!test_mode) {
  save(argo_data_unordered, file = paste0(root, "/data/argo_data_january.RData"))
}

######################
### new processing ###
######################

library(data.table)
library(geosphere)

argo <- data.table(argo_data_unordered)
argo[, isodatetime := ISOdatetime(years, months, days, hours, minutes, seconds, tz = "UTC")]
setkey(argo, float_id, isodatetime)

argo[, float_n := .N, float_id]
argo[, float_i := 1:.N, float_id]
argo[, float_r := float_n > 1]

argo[(float_r), dt_s := as.numeric(difftime(isodatetime, lag(isodatetime), units = "secs")), float_id]

vincenty_dx <- function(lon1, lat1, lon2, lat2) {
  geosphere::distVincentyEllipsoid(
    p1 = cbind(lon1, lat1),
    p2 = cbind(lon2, lat2)
  )
}

argo[(float_r), dx_m := vincenty_dx(lag(lon_degrees), lag(lat_degrees), lon_degrees, lat_degrees), float_id]
argo[(float_r), dlat_m := vincenty_dx(lag(lon_degrees), lag(lat_degrees), lag(lon_degrees), lat_degrees), float_id]
argo[(float_r), dlon_m := vincenty_dx(lag(lon_degrees), lag(lat_degrees), lon_degrees, lag(lat_degrees)), float_id]

argo[, u_ms := dlon_m / dt_s] # eastward velocity m/s
argo[, v_ms := dlat_m / dt_s] # northward velocity m/s
argo[, speed_ms := dx_m / dt_s]
argo[, theta := atan2(u_ms, v_ms)] # bearing radians (N,E,S,W) = (0,pi/2,pi,-pi/2)

cols <- c("float_id", "float_r", "float_n", "float_i", "isodatetime")
setcolorder(argo, cols)


## add ocean region
# mregions2 downloads shapefiles from <https://www.marineregions.org/sources.php#goas>
library(mregions2)
library(sf)
sf_use_s2(FALSE)

goas_file <- paste0(root, "/data/goas.rds")

if (file.exists(goas_file)) {
  goas <- readRDS(goas_file)
} else {
  goas <- mrp_get("goas")
  saveRDS(goas, goas_file)
}

latlon <- c("lon_degrees", "lat_degrees")
float_loc <- argo[, .(float_id, float_i, lon_degrees, lat_degrees)]
float_loc_sf <- st_as_sf(float_loc, coords = latlon, crs = 4326) # WGS84 globe

float_loc_j <- st_join(float_loc_sf, goas["name"])
float_loc_j <- as.data.table(float_loc_j)[, geometry := NULL]
setnames(float_loc_j, c("name"), c("ocean"))
float_loc_j[, ocean := tolower(ocean)]

setkey(float_loc_j, float_id, float_i)
setkey(float_loc, float_id, float_i)

float_loc <- float_loc[float_loc_j]
float_loc[, (latlon) := NULL]

setkey(argo, float_id, float_i)
setkey(float_loc, float_id, float_i)

argo <- argo[float_loc]

# save preprocessed data
if (!test_mode) {
  save(argo, file = paste0(root, "/data/argo_velo_data_january.RData"))
}
