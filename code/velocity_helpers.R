library(data.table)

add_velocity_enu <- function(
  df,
  float_id = "float_id",
  lat = "lat",
  lon = "lon",
  date = "date"
) {
  if (!is.data.table(df)) setDT(df)
  unames <- c(float_id, lat, lon, date)
  mnames <- c("float_id", "lat", "lon", "date")
  setnames(df, unames, mnames)

  setorderv(df, c(float_id, date))

  df[, c("lat0", "lon0", "date0") := lapply(.SD, shift), .SDcols = c("lat", "lon", "date"), by = float_id]

  df[, dt_s := as.numeric(difftime(date, date0), units = "secs")]
  df[, dx_m := geosphere::distGeo(cbind(lon0, lat0), cbind(lon, lat))]
  df[, brng := geosphere::bearing(cbind(lon0, lat0), cbind(lon, lat))]

  df[, theta := brng * pi / 180]
  df[, speed_ms := dx_m / dt_s]
  df[, u_ms := speed_ms * sin(theta)]
  df[, v_ms := speed_ms * cos(theta)]

  df[, c("lon0", "lat0", "date0") := NULL]
  setnames(df, mnames, unames)
  return(invisible(df))
}

add_dtemp <- function(
  df,
  ohc_cols = NULL,
  float_id = "float_id",
  date = "date"
) {
  setorderv(df, c(float_id, date))
  if (is.null(ohc_cols)) {
    ohc_cols <- grep("ohc", names(df), value = T)
  }
  df[, (ohc_cols) := lapply(.SD, \(x) x / 1e9), .SDcols = ohc_cols]
  df[, paste0("d", ohc_cols) := lapply(.SD, \(x) x - lag(x)), .SDcols = ohc_cols, by = float_id]
  return(invisible(df))
}
