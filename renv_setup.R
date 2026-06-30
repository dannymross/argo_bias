options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

renv::init(bioconductor = FALSE)

renv::install(c(
  "tidyverse",
  "data.table",
  "patchwork",
  "geosphere",
  "fields",
  "GpGp",
  "latex2exp"
))

renv::snapshot()
