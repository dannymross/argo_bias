root <- getwd()
library(tidyverse)
library(data.table)
library(pbapply)
library(geosphere)
library(GpGp)

load(paste0(root, "/data/argo_velo_data_january.RData"))

# BEGIN old geocoding ----
library(sf)
sf_use_s2(FALSE)
library(rnaturalearth)

float_loc <- A[, .(float_id, float_i, lon_degrees, lat_degrees)]
float_loc_sf <- st_as_sf(float_loc, coords = c("lon_degrees", "lat_degrees"), crs = 4326) # WGS84 globe

# medium scale regions (1:50m)
oceans_50 <- ne_download(
  scale = 50,
  type = "geography_marine_polys",
  category = "physical",
  returnclass = "sf"
)
# large scale regions (1:110m)
oceans_110 <- ne_download(
  scale = 110,
  type = "geography_marine_polys",
  category = "physical",
  returnclass = "sf"
)

float_loc_j <- st_join(float_loc_sf, oceans_50["name"])
float_loc_j <- st_join(float_loc_j, oceans_110["name"])
float_loc_j <- as.data.table(float_loc_j)[, geometry := NULL]
cols <- c("region_s", "region_l")
setnames(float_loc_j, c("name.x", "name.y"), cols)
float_loc_j[, (cols) := lapply(.SD, tolower), .SDcols = cols]

setkey(float_loc_j, float_id, float_i)
setkey(float_loc, float_id, float_i)

float_loc <- float_loc[float_loc_j]

float_loc[region_s == "great barrier reef", region_s := "coral sea"]
float_loc <- unique(float_loc)

# END old geocoding ----
