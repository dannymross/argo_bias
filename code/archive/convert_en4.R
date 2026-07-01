# Convert the EN4 gridded-OHC RDS to a CSV subset for the Python pipeline.
#
# data/ohc_en4_gridded.rds is a global monthly 1-degree data.table of
# integrated OHC (J/m^2) for 0-700 m and 0-2000 m. We subset to the Northwest
# Atlantic (covering the pilot's advection domain plus margin) and write a CSV
# that the Quarto report / analysis code can read with pandas.
#
# Run from the repo root:
#   Rscript code/convert_en4.R

suppressMessages(library(data.table))

d <- readRDS("data/ohc_en4_gridded.rds")
setDT(d)

sub <- d[lat >= 25 & lat <= 55 & lon >= -80 & lon <= -40,
         .(lon, lat, year, month, date, ohc_700, ohc_2000, cell_area_m2)]

out <- "data/ohc_en4_gridded_nwatlantic.csv"
fwrite(sub, out)
cat("wrote", nrow(sub), "rows ->", out, "\n")
cat("date range:", as.character(min(sub$date)), "->", as.character(max(sub$date)), "\n")
