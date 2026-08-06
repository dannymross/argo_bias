## =====================================================================
## design_invariance.R -- test assumption D1b (design invariance):
##   w^(l) =d= w^(l') across velocity years l.
##
##   Violations are not all equally damaging, and the tests are ordered by how
##   much they matter:
##
##   (1) TREND IN YEAR LAG -- the dangerous one. T_k contrasts the matched
##       design (l = k) against distant ones (|l - k| >= 2). If designs drift
##       systematically with year, then designs near k resemble year k more than
##       distant designs do, and that contrast is nonzero with no preferential
##       sampling present. A trend mimics the alternative. Tested by regressing
##       the between-design distance on |l - l'| and permuting year labels.
##
##   (2) ANY YEAR EFFECT -- PERMANOVA pseudo-F on the weight vectors. Breaks
##       exchangeability too, but random year-to-year variation does not align
##       with the |l - k| structure, so it inflates variance rather than
##       manufacturing signal.
##
##   (3) SCALAR SUMMARIES -- residence time, retention, dispersion, occupancy
##       concentration: which physical channel any year effect runs through.
##
##   Weights are compared marginally over day-of-year by default: with ~1800
##   profiles spread over ~7200 space-time cells the full vector is dominated by
##   Poisson noise, while the ~200 spatial cells estimate the design's spatial
##   distribution stably -- and a jet shift moves exactly that.
##
##   INPUT: design_weights_<box>.csv, design_summary_<box>.csv
##          from code/osse/design_invariance.py
##
##   Rscript code/osse/design_invariance.R [box] [nperm]
## =====================================================================

suppressPackageStartupMessages(library(data.table))

## ------------------------- weight matrix ----------------------------

## (MD x C) matrix of design weights, renormalised over the retained cells.
weight_matrix <- function(w, space_only = TRUE) {
  w <- as.data.table(w)
  if (space_only) {
    w <- w[, .(w = sum(w)), by = .(vel_year, rep, cell_lat, cell_lon)]
  }
  w[, cell := if (space_only) paste(cell_lat, cell_lon) else paste(cell_lat, cell_lon, doy_bin)]
  m <- dcast(w, vel_year + rep ~ cell, value.var = "w", fill = 0)
  key <- m[, .(vel_year, rep)]
  W <- as.matrix(m[, !c("vel_year", "rep")])
  W <- W / rowSums(W)
  list(W = W, year = key$vel_year, rep = key$rep)
}

## Hellinger distance: Euclidean on sqrt(w), the natural metric for
## probability vectors and insensitive to the many near-empty cells.
hellinger <- function(W) as.matrix(dist(sqrt(W))) / sqrt(2)

## ---------------------- (1) distance vs year lag ---------------------

## Slope of pairwise distance on |l - l'|. Zero under D1b.
lag_slope <- function(Dm, year) {
  lag <- abs(outer(year, year, "-"))
  ut <- upper.tri(lag)
  stats::coef(stats::lm(Dm[ut] ~ lag[ut]))[2]
}

## --------------------- (2) PERMANOVA pseudo-F ------------------------

## Between-group vs within-group dispersion on a distance matrix
## (Anderson 2001): F = [(SS_T - SS_W)/(g-1)] / [SS_W/(N-g)].
pseudo_F <- function(Dm, grp) {
  N <- length(grp); g <- length(unique(grp))
  D2 <- Dm^2
  ss_t <- sum(D2[upper.tri(D2)]) / N
  ss_w <- sum(vapply(unique(grp), function(k) {
    idx <- which(grp == k)
    sum(D2[idx, idx][upper.tri(D2[idx, idx])]) / length(idx)
  }, numeric(1)))
  ((ss_t - ss_w) / (g - 1)) / (ss_w / (N - g))
}

## ------------------------------ driver -------------------------------

test_invariance <- function(w, summ, nperm = 4999L, space_only = TRUE, seed = 1L) {
  set.seed(seed)
  wm <- weight_matrix(w, space_only)
  Dm <- hellinger(wm$W)
  year <- wm$year

  obs_slope <- lag_slope(Dm, year)
  obs_F <- pseudo_F(Dm, year)

  perm <- replicate(nperm, {
    p <- sample(year)
    c(lag_slope(Dm, p), pseudo_F(Dm, p))
  })
  p_slope <- (1 + sum(abs(perm[1, ]) >= abs(obs_slope))) / (1 + nperm)
  p_F <- (1 + sum(perm[2, ] >= obs_F)) / (1 + nperm)

  cat(sprintf("\nD1b TESTS  (%d designs, %d years, %d cells, %s)\n",
              nrow(wm$W), length(unique(year)), ncol(wm$W),
              if (space_only) "space-only weights" else "space x doy weights"))
  ## Only a POSITIVE slope is the hazard: designs growing more dissimilar with
  ## year lag is what makes the matched-vs-distant contrast nonzero under H0.
  ## A negative slope is benign, so flag on sign as well as significance.
  cat(sprintf("  (1) distance ~ |l-l'| slope : %+.5f per year   p = %.4f  %s\n",
              obs_slope, p_slope,
              if (p_slope < 0.05 && obs_slope > 0) "<-- drift mimics the alternative"
              else if (p_slope < 0.05) "(negative: benign)" else ""))
  cat(sprintf("  (2) PERMANOVA pseudo-F      : %8.3f            p = %.4f  %s\n",
              obs_F, p_F, if (p_F < 0.05) "<-- year effect present" else ""))

  ## (3) which channel: trend and year effect in each scalar summary
  summ <- as.data.table(summ)
  vars <- c("frac_inside", "residence_days", "retention_end", "rms_disp_km",
            "n_cells_occupied", "participation_ratio", "n_profiles_in_box")
  vars <- intersect(vars, names(summ))
  tab <- rbindlist(lapply(vars, function(v) {
    x <- summ[[v]]; y <- summ$vel_year
    r <- suppressWarnings(cor(y, x, method = "spearman"))
    obs <- abs(cor(y, x))
    pp <- mean(replicate(nperm, abs(cor(sample(y), x))) >= obs)
    data.table(stat = v, mean = mean(x), cv = sd(x) / abs(mean(x)),
               spearman_year = r, p_trend = (1 + pp * nperm) / (1 + nperm))
  }))
  cat("\n  (3) scalar summaries vs velocity year:\n")
  print(tab[, lapply(.SD, function(z) if (is.numeric(z)) round(z, 4) else z)], row.names = FALSE)

  invisible(list(slope = obs_slope, p_slope = p_slope, F = obs_F, p_F = p_F, summary = tab))
}

## ---------------- (4) location vs dispersion decomposition ------------
##
## pseudo-F responds to BOTH a shift in group centroids and a difference in
## group spread, and the two have different consequences:
##
##   location   (D1a: E[w^(l)] = wbar for all l) -- violation BIASES T_k, since
##              E[T_k] = sum_j (E[w_j^(k)] - mean_{l in C_k} E[w_j^(l)]) Z_kj
##              is exactly zero under D1a whatever the field is.
##   dispersion (D1b only) -- violation inflates Var(T_k); the permutation null
##              absorbs it.
##
## Hellinger distance is Euclidean on sqrt(w)/sqrt(2), so centroids are exact
## and the split needs no approximation.
test_location_dispersion <- function(w, nperm = 4999L, space_only = TRUE, seed = 1L) {
  set.seed(seed)
  wm <- weight_matrix(w, space_only)
  X <- sqrt(wm$W) / sqrt(2)
  year <- wm$year; yrs <- sort(unique(year))

  centroids <- function(g) t(sapply(yrs, function(l) colMeans(X[g == l, , drop = FALSE])))
  cg <- colMeans(X)
  ss_a <- function(g, cc) sum(vapply(seq_along(yrs), function(i)
            sum(g == yrs[i]) * sum((cc[i, ] - cg)^2), numeric(1)))
  disp <- function(g, cc) vapply(seq_along(yrs), function(i)
            mean(sqrt(rowSums(sweep(X[g == yrs[i], , drop = FALSE], 2, cc[i, ])^2))), numeric(1))

  cl <- centroids(year)
  obs <- c(loc = ss_a(year, cl), disp = sd(disp(year, cl)))
  perm <- replicate(nperm, {
    p <- sample(year); cp <- centroids(p)
    c(ss_a(p, cp), sd(disp(p, cp)))
  })
  data.table(
    component = c("location (D1a)", "dispersion"),
    statistic = c("SS_A", "sd of within-year dispersion"),
    value = obs,
    p_value = (1 + rowSums(perm >= obs)) / (1 + nperm),
    biases_Tk = c(TRUE, FALSE)
  )
}

## ------------------- (5) placebo-field separation ---------------------
##
## sum_j Delta_kj Z_kj IS what T_k estimates, so on its own it cannot say
## whether a nonzero value is design drift or genuine preferential sampling.
## Applying the same Delta_k to OTHER years' fields separates them: under pure
## drift the matched value is unremarkable against the placebo spread; under
## preferential sampling the matched value stands out.
placebo_table <- function(w, field, analysis_years, space_only = FALSE) {
  wm <- weight_matrix(w, space_only)
  W <- wm$W; year <- wm$year; key <- colnames(W)   # weight space: B is linear in w
  field <- as.data.table(field)
  field[, cell := paste(cell_lat, cell_lon, doy_bin)]

  rbindlist(lapply(analysis_years, function(k) {
    Ck <- setdiff(sort(unique(year)), c(k - 1, k, k + 1))
    delta <- colMeans(W[year == k, , drop = FALSE]) -
             colMeans(W[year %in% Ck, , drop = FALSE])
    rbindlist(lapply(analysis_years, function(kp) {
      z <- field[year == kp][match(key, cell), z]
      ok <- is.finite(z)
      data.table(design_year = k, field_year = kp, matched = k == kp,
                 n_comparison_years = length(Ck),
                 value_GJ = sum(delta[ok] * z[ok]) / 1e9)
    }))
  }))
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  box <- if (length(args) >= 1L) args[1] else "boxB"
  nperm <- if (length(args) >= 2L) as.integer(args[2]) else 4999L
  variant <- if (length(args) >= 3L) args[3] else "argo"
  sfx <- if (variant == "argo") "" else paste0("_", variant)
  out <- function(x) sprintf("data/osse/d1b_%s_%s_%s.csv", x, box, variant)

  w <- fread(sprintf("data/osse/design_weights_%s%s.csv", box, sfx))
  s <- fread(sprintf("data/osse/design_summary_%s%s.csv", box, sfx))

  tests <- rbindlist(lapply(c(TRUE, FALSE), function(so) {
    r <- test_invariance(w, s, nperm, space_only = so)
    data.table(representation = if (so) "space-only" else "space x doy",
               lag_slope = r$slope, p_lag_slope = r$p_slope,
               pseudo_F = r$F, p_pseudo_F = r$p_F)
  }))
  fwrite(tests, out("tests"))
  fwrite(test_invariance(w, s, nperm, space_only = TRUE)$summary, out("summary"))
  fwrite(rbindlist(lapply(c(TRUE, FALSE), function(so)
    cbind(representation = if (so) "space-only" else "space x doy",
          test_location_dispersion(w, nperm, so))), fill = TRUE), out("decomp"))

  if (file.exists("code/osse/surrogate_designs.R")) {
    source("code/osse/surrogate_designs.R")
    fld <- read_coarse_field(box)
    fwrite(placebo_table(w, fld, 2015:2019), out("placebo"))
  }
  cat(sprintf("\nwrote data/osse/d1b_{tests,summary,decomp,placebo}_%s_%s.csv\n", box, variant))
}
