## =====================================================================
## run_perm_test.R -- driver for the preferential-sampling permutation test.
##
##   Rscript code/osse/run_perm_test.R <mode> [tag] [nruns]
##     null  : calibration study -- exogenous designs, p-values must be U(0,1)
##     power : rejection rate over a sweep of preferential strength beta
##     real  : one test on an existing sampled_residuals_<tag>.csv
##
##   The null and power modes generate every replicate's positions first and
##   make ONE call to sample_residuals.py, rather than a subprocess per
##   replicate. (Elsewhere in this repo Python drives R; here R owns the
##   statistics and Python owns the field, so the call goes the other way.)
##
##   Run from the repo root.
## =====================================================================

suppressPackageStartupMessages(library(data.table))

here <- function(...) file.path("code", "osse", ...)
source(here("perm_test.R"))
source(here("osse_bias.R"))
source(here("surrogate_designs.R"))

config <- function() list(
  tag = "boxB",
  dir = "data/osse",
  python = "/storage/work/dmr6737/.conda/envs/argo_bias/bin/python",
  M = 5L,             # velocity years
  D = 5L,             # deployment replicates per velocity year
  n_profiles = 500L,  # profiles per design
  coarse_deg = 1,
  N_perm = 2000L,     # permutation replicates per test
  n_runs = 200L,      # independent calibration/power runs
  betas = c(0, 0.01, 0.02, 0.04, 0.08, 0.16),
  seed = 1L,          # position generation; perm_test seeds off the run id
  keep_intermediates = FALSE
)

## ---------------------- python gather bridge ------------------------

## Surrogate intermediates run to ~1 GB per sweep and are fully regenerable, so
## they are removed once read back unless cfg$keep_intermediates is set.
gather_residuals <- function(positions, cfg, label) {
  pos_csv <- file.path(cfg$dir, sprintf("positions_%s_%s.csv", cfg$tag, label))
  out_csv <- file.path(cfg$dir, sprintf("sampled_residuals_%s_%s.csv", cfg$tag, label))
  fwrite(positions, pos_csv)
  status <- system2(cfg$python, c(here("sample_residuals.py"), pos_csv, out_csv, cfg$tag))
  if (status != 0L) stop("sample_residuals.py failed (status ", status, ")")
  sampled <- fread(out_csv)
  if (!isTRUE(cfg$keep_intermediates)) unlink(c(pos_csv, out_csv))
  sampled
}

## One permutation test per run id, on a table already carrying `run`.
## Keyed once so the per-run subset is a binary search rather than a full scan.
test_by_run <- function(sampled, domain_means, cfg) {
  setkey(sampled, run)
  rbindlist(lapply(sampled[, unique(run)], function(r) {
    bm <- bias_matrix(sampled[.(r)], domain_means)
    fit <- perm_test(bm$B, bm$labels, M = cfg$M, N = cfg$N_perm, seed = r)
    data.table(run = r, p_pool = fit$p_pool, Tbar = fit$Tbar_obs,
               p_min = min(fit$p_k), mean_absB = mean(abs(bm$B)))
  }))
}

with_run <- function(gen, n_runs, ...) {
  rbindlist(lapply(seq_len(n_runs), function(r) cbind(run = r, gen(...))))
}

## ------------------------------ modes -------------------------------

mode_null <- function(cfg) {
  set.seed(cfg$seed)
  field <- read_coarse_field(cfg$tag, cfg$dir)
  dm <- fread(file.path(cfg$dir, sprintf("domain_means_%s.csv", cfg$tag)))

  pos <- with_run(sim_designs_null, cfg$n_runs, field = field, M = cfg$M, D = cfg$D,
                  n_profiles = cfg$n_profiles, n_days = attr(field, "n_days"),
                  deg = cfg$coarse_deg)
  sampled <- gather_residuals(pos, cfg, "null")
  res <- test_by_run(sampled, dm, cfg)

  ks <- ks.test(res$p_pool, "punif")
  cat(sprintf("\nNULL CALIBRATION (%d runs, M=%d D=%d n=%d)\n",
              cfg$n_runs, cfg$M, cfg$D, cfg$n_profiles))
  cat(sprintf("  reject rate at 0.05 : %.3f  (nominal 0.05)\n", mean(res$p_pool <= 0.05)))
  cat(sprintf("  reject rate at 0.10 : %.3f  (nominal 0.10)\n", mean(res$p_pool <= 0.10)))
  cat(sprintf("  KS vs U(0,1)        : D = %.3f, p = %.3f\n", ks$statistic, ks$p.value))
  fwrite(res, file.path(cfg$dir, sprintf("perm_test_null_%s.csv", cfg$tag)))
  res
}

mode_power <- function(cfg) {
  set.seed(cfg$seed)
  field <- read_coarse_field(cfg$tag, cfg$dir)
  dm <- fread(file.path(cfg$dir, sprintf("domain_means_%s.csv", cfg$tag)))

  pos <- rbindlist(lapply(cfg$betas, function(b) {
    cbind(beta = b, with_run(sim_designs_pref, cfg$n_runs, field = field, M = cfg$M,
                             D = cfg$D, n_profiles = cfg$n_profiles,
                             n_days = attr(field, "n_days"), beta = b,
                             deg = cfg$coarse_deg))
  }))
  sampled <- gather_residuals(pos, cfg, "power")

  res <- rbindlist(lapply(cfg$betas, function(b) {
    cbind(beta = b, test_by_run(sampled[beta == b], dm, cfg))
  }))
  tab <- res[, .(reject_05 = mean(p_pool <= 0.05),
                 reject_10 = mean(p_pool <= 0.10),
                 median_Tbar = median(Tbar)), by = beta]
  cat(sprintf("\nPOWER SWEEP (%d runs per beta)\n", cfg$n_runs))
  print(tab, row.names = FALSE)
  fwrite(res, file.path(cfg$dir, sprintf("perm_test_power_%s.csv", cfg$tag)))
  res
}

mode_real <- function(cfg) {
  inp <- read_bias_inputs(cfg$tag, cfg$dir)
  bm <- bias_matrix(inp$sampled, inp$domain_means)
  fit <- perm_test(bm$B, bm$labels, M = cfg$M, N = cfg$N_perm)
  out <- perm_summary(fit, years = bm$years)
  cat(sprintf("\nPERMUTATION TEST (%s): %d designs, %d analysis years, N=%d\n",
              cfg$tag, nrow(bm$B), ncol(bm$B), fit$N))
  print(out, row.names = FALSE)
  fwrite(out, file.path(cfg$dir, sprintf("perm_test_real_%s.csv", cfg$tag)))
  out
}

## ------------------------------ main --------------------------------

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- config()
  if (length(args) >= 2L) cfg$tag <- args[2]
  if (length(args) >= 3L) cfg$n_runs <- as.integer(args[3])
  switch(args[1],
         null = mode_null(cfg),
         power = mode_power(cfg),
         real = mode_real(cfg),
         stop("mode must be one of: null, power, real"))
}
