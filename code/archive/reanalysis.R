library(tidyverse)
library(data.table)

# EN4 OHC
ohc_en4 <- readRDS("data/ohc_en4_gridded.rds")
setDT(ohc_en4)
ohc_cols <- grep("ohc", names(ohc_en4), value = T)
ohc_en4[, (ohc_cols) := lapply(.SD, \(x) x / 1e9), .SDcols = ohc_cols]

BOUNDS <- c(36, 40, -68, -62)
ohc_en4_20 <- ohc_en4[year == 2020 & between(lat, BOUNDS[1], BOUNDS[2]) & between(lon, BOUNDS[3], BOUNDS[4])]
