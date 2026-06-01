# WGS84 globe

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


float_loc[
  , .(
    .N,
    lat_range = paste0(round(range(lat_degrees), 2), collapse = ","),
    lon_range = paste0(round(range(lon_degrees), 2), collapse = ",")
  ),
  .(region_s)
][order(-N)]

sample_locs <- float_loc[, .SD[sample(.N, min(1000, .N))], by = region_s]

map_pts <- function(df, x = "lon_degrees", y = "lat_degrees", color = "region_s", legend.rows = 1) {
  ggplot() +
    borders("world", fill = "gray80", colour = "gray80") +
    geom_point(
      data = df,
      aes(
        x = .data[[x]],
        y = .data[[y]],
        color = .data[[color]]
      ), size = .5
    ) +
    coord_sf() +
    theme_minimal() +
    theme(legend.position = "bottom") +
    # legend.title = element_blank()) +
    guides(colour = guide_legend(nrow = legend.rows))
}

map_pts(sample_locs, color = "region_s", legend.rows = 5)
