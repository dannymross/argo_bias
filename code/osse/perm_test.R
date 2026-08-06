## =====================================================================
## perm_test.R -- permutation test of H0: w^(k) independent of Z_k
##   (no preferential sampling), following paper/sections/theory.tex.
##
##   Consumes a single matrix B (MD x m): rows are sampling designs
##   (velocity year l, replicate d), columns are analysis years k, with
##   B[(l,d), k] the sampling bias incurred when floats advected by year l's
##   velocity sample year k's residual OHC field.
##
##   Under H0 the MD designs are iid (theory.tex Lemma 1), so the velocity-year
##   labels are exchangeable. A replicate relabels the rows of B and recomputes
##
##     T_k(lambda) = mean{B[.,k] : label == k} - mean{B[.,k] : label in C_k},
##     C_k = {l : |l - k| >= 2},
##
##   which is centered at 0 under H0. One relabelling is applied across all m
##   analysis years per replicate, preserving the dependence among {T_k}.
##
##   NOTE: theory.tex eq:Tk normalises the comparison mean by (|C_k| - 1) D,
##   but k is not in C_k, so that sum has |C_k| D terms. We use |C_k| D, which
##   is what makes T_k a difference of means. The test is valid either way
##   (labels are exchangeable regardless of scaling); only the centering and
##   the interpretation of T_k as a bias difference need the correction.
##
##   Base R only -- no GLORYS specifics enter here.
## =====================================================================

## --------------------------- statistic ------------------------------

## Comparison set C_k = {l : |l - k| >= 2}, as a logical M x m matrix
## (rows = velocity-year index l, cols = analysis-year index k).
comparison_sets <- function(M, m) {
  outer(seq_len(M), seq_len(m), function(l, k) abs(l - k) >= 2)
}

## T_k for one labelling. `labels` gives each design's velocity-year index.
##
## A permutation only relabels designs, so the group sizes n_l are invariant
## across replicates and are computed once by the caller.
stat_T <- function(B, labels, M, Ck = NULL, n_l = NULL) {
  m <- ncol(B)
  if (is.null(Ck)) Ck <- comparison_sets(M, m)
  if (is.null(n_l)) n_l <- as.vector(table(factor(labels, levels = seq_len(M))))
  G <- rowsum(B, group = labels, reorder = TRUE)   # M x m group sums

  matched <- G[cbind(seq_len(m), seq_len(m))] / n_l[seq_len(m)]
  comp <- vapply(seq_len(m), function(k) {
    sel <- Ck[, k]
    sum(G[sel, k]) / sum(n_l[sel])
  }, numeric(1))
  matched - comp
}

## --------------------------- the test -------------------------------

## B      : MD x m bias matrix
## labels : true velocity-year index per design (length MD)
## N      : number of permutation replicates
##
## Returns the observed T, the permutation distribution, the pooled p-value on
## Tbar, and max-T p-values per year (familywise error controlled, no further
## correction).
perm_test <- function(B, labels, M = max(labels), N = 1e4, seed = 1L) {
  set.seed(seed)
  m <- ncol(B)
  Ck <- comparison_sets(M, m)

  n_l <- as.vector(table(factor(labels, levels = seq_len(M))))
  stopifnot(length(unique(n_l)) == 1L)   # balanced design; D per velocity year
  D <- n_l[1]
  T_obs <- stat_T(B, labels, M, Ck, n_l)

  # All N replicates in one pass. Relabelling designs is the same as permuting
  # rows of B, so draw N permutations, stack them, and take every replicate's
  # group sums with a single rowsum. G[l, r, k] after the reshape.
  MD <- nrow(B)
  perm <- replicate(N, sample.int(MD))            # MD x N
  G <- rowsum(B[as.vector(perm), , drop = FALSE],
              group = rep(seq_len(N * M), each = D), reorder = FALSE)
  G <- array(G, dim = c(M, N, m))

  T_perm <- vapply(seq_len(m), function(k) {
    sel <- Ck[, k]
    G[k, , k] / D - colSums(G[sel, , k, drop = FALSE]) / (sum(sel) * D)
  }, numeric(N))

  Tbar_obs <- mean(T_obs)
  Tbar_perm <- rowMeans(T_perm)
  p_pool <- (1 + sum(Tbar_perm >= Tbar_obs)) / (1 + N)

  sigma_k <- apply(T_perm, 2, sd)
  Lambda <- apply(sweep(T_perm, 2, sigma_k, "/"), 1, max)
  p_k <- vapply(seq_len(m), function(k) {
    (1 + sum(Lambda >= T_obs[k] / sigma_k[k])) / (1 + N)
  }, numeric(1))

  list(T_obs = T_obs, T_perm = T_perm, sigma_k = sigma_k,
       Tbar_obs = Tbar_obs, p_pool = p_pool, p_k = p_k, N = N, M = M)
}

## Compact one-row-per-year summary plus the pooled row.
perm_summary <- function(fit, years = seq_len(length(fit$T_obs))) {
  data.frame(
    year = c(years, NA),
    stat = c(fit$T_obs, fit$Tbar_obs),
    sigma = c(fit$sigma_k, sd(rowMeans(fit$T_perm))),
    p = c(fit$p_k, fit$p_pool),
    which = c(rep("year", length(years)), "pooled")
  )
}
