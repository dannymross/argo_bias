options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

renv::init(bioconductor = FALSE)

renv::install(c(
  "data.table",
  "fields",
  "GpGp"
))

renv::snapshot()
