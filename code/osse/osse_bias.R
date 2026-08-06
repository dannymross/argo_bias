## =====================================================================
## osse_bias.R -- build the bias matrix B from sampled residuals.
##
##   At native resolution the sampling bias collapses to a sample mean minus a
##   domain mean (theory.tex, with {c_j} taken as the GLORYS grid):
##
##     B_k^(l,d) = sum_j (w_j - p_j) Z_kj
##               = (1/n) sum_i Z_k(x_i^(l,d))  -  Zbar_k,
##
##   so no weight vector over the ~10^7 native cells is ever materialised.
##
##   INPUT: sampled_residuals_<tag>.csv from code/osse/sample_residuals.py --
##     columns vel_year, rep, lat, lon, doy, z_<year> for each analysis year;
##     and domain_means_<tag>.csv -- columns year, zbar, sd_Z.
## =====================================================================

suppressPackageStartupMessages(library(data.table))

## Returns list(B = MD x m matrix, labels = velocity-year index per design,
## years = analysis years, designs = the (vel_year, rep) key).
bias_matrix <- function(sampled, domain_means) {
  sampled <- as.data.table(sampled)
  domain_means <- as.data.table(domain_means)

  zcols <- grep("^z_[0-9]{4}$", names(sampled), value = TRUE)
  years <- as.integer(sub("^z_", "", zcols))
  stopifnot(length(zcols) > 0L, setequal(years, domain_means$year))

  agg <- sampled[, c(.(n = .N), lapply(.SD, mean)), by = .(vel_year, rep), .SDcols = zcols]
  setorder(agg, vel_year, rep)

  zbar <- domain_means[match(years, year), zbar]
  B <- sweep(as.matrix(agg[, ..zcols]), 2, zbar, "-")
  dimnames(B) <- list(NULL, as.character(years))

  vel_years <- sort(unique(agg$vel_year))
  list(B = B,
       labels = match(agg$vel_year, vel_years),
       years = years,
       vel_years = vel_years,
       n = agg$n,
       designs = agg[, .(vel_year, rep)])
}

read_bias_inputs <- function(tag = "boxB", dir = "data/osse") {
  list(
    sampled = fread(file.path(dir, sprintf("sampled_residuals_%s.csv", tag))),
    domain_means = fread(file.path(dir, sprintf("domain_means_%s.csv", tag)))
  )
}
