## =====================================================================
## plot_refinement.R -- visualize sigma^2_{w,K} refinement & sensitivity
## Base R graphics only. Source sigma_wK.R first.
## =====================================================================

## Run sigma_wK_with_ci over an arbitrary grid table (preserving any
## label columns such as `axis`/`x`), returning one row per grid.
run_grids <- function(argo, box, grids, thin = FALSE, level = 0.95) {
  do.call(rbind, lapply(seq_len(nrow(grids)), function(i) {
    g <- grids[i, , drop = FALSE]
    r <- sigma_wK_with_ci(argo, box, g$K_lon, g$K_lat, g$K_t, thin, level)
    cbind(g, r[, setdiff(names(r), names(g)), drop = FALSE], row.names = NULL)
  }))
}

## Build grids that vary ONE axis at a time about a baseline.
make_sensitivity_grids <- function(base, K_lon_vals, K_lat_vals, K_t_vals) {
  mk <- function(axis, vals) {
    g <- data.frame(K_lon = base["K_lon"], K_lat = base["K_lat"],
                    K_t = base["K_t"], row.names = NULL)[rep(1, length(vals)), ]
    g[[axis]] <- vals; g$axis <- axis; g$x <- vals
    rownames(g) <- NULL; g
  }
  rbind(mk("K_lon", K_lon_vals), mk("K_lat", K_lat_vals), mk("K_t", K_t_vals))
}

## Core one-panel plot: point estimate + CI ribbon along `xvar`.
plot_sigma_curve <- function(curve, xvar = "cell_km",
                             scale = c("sigma2", "sigma"),
                             xlab = xvar, main = NULL, col = "steelblue",
                             ylim = NULL) {
  scale <- match.arg(scale)
  o <- order(curve[[xvar]]); curve <- curve[o, ]; x <- curve[[xvar]]
  if (scale == "sigma2") {
    y <- curve$sigma2_hat; lo <- curve$sigma2_lo; hi <- curve$sigma2_hi
    ylab <- expression(hat(sigma)[list(w, K)]^2)
  } else {
    y <- curve$sigma_hat; lo <- curve$sigma_lo; hi <- curve$sigma_hi
    ylab <- expression(hat(sigma)[list(w, K)])
  }
  if (is.null(ylim)) ylim <- range(c(lo, hi, 0), na.rm = TRUE)
  plot(x, y, type = "n", ylim = ylim, xlab = xlab, ylab = ylab, main = main)
  abline(h = 0, col = "grey80")
  arrows(x, lo, x, hi, angle = 90, code = 3, length = 0.04, col = col, lwd = 1.5)
  points(x, y, pch = 19, col = col)
  ## annotate typical count per occupied cell as a data-adequacy cue
  #text(x, hi, sprintf("n=%d", curve$n), pos = 3, cex = 0.6, col = "grey40", xpd = NA)
}

## Three-panel sensitivity: vary K_lon, K_lat, K_t independently.
## All three panels share a common y-axis range (computed across all curves).
plot_sensitivity <- function(sens, base, scale = c("sigma2", "sigma")) {
  scale <- match.arg(scale)
  op <- par(mfrow = c(1, 3), mar = c(4, 4.2, 3, 1), oma = c(0, 0, 2, 0))
  on.exit(par(op))
  labs <- list(
    K_lon = bquote(paste("vary ", K[lon], "  (", K[lat] == .(base[["K_lat"]]),
                         ", ", K[t] == .(base[["K_t"]]), ")")),
    K_lat = bquote(paste("vary ", K[lat], "  (", K[lon] == .(base[["K_lon"]]),
                         ", ", K[t] == .(base[["K_t"]]), ")")),
    K_t   = bquote(paste("vary ", K[t], "  (", K[lon] == .(base[["K_lon"]]),
                         ", ", K[lat] == .(base[["K_lat"]]), ")")))
  if (scale == "sigma2") {
    ylim <- range(c(sens$sigma2_lo, sens$sigma2_hi, 0), na.rm = TRUE)
  } else {
    ylim <- range(c(sens$sigma_lo, sens$sigma_hi, 0), na.rm = TRUE)
  }
  for (a in c("K_lon", "K_lat", "K_t"))
    plot_sigma_curve(sens[sens$axis == a, ], xvar = "x", scale = scale,
                     xlab = a, main = labs[[a]], ylim = ylim)
  mtext(sprintf("Sensitivity of %s to grid resolution (95%% jackknife CI)",
                if (scale == "sigma2") "sigma^2_{w,K}" else "sigma_{w,K}"),
        outer = TRUE, cex = 1.05, font = 2)
}

## Build a full 2D grid over TWO axes (axis1 x axis2), holding the third
## K fixed at base[[other]]. vals1/vals2 should be given in increasing
## order -- that order is what gets reshaped into the surface matrix.
make_surface_grid <- function(base, axis1, vals1, axis2, vals2) {
  other <- setdiff(c("K_lon", "K_lat", "K_t"), c(axis1, axis2))
  g <- expand.grid(setNames(list(vals1, vals2), c(axis1, axis2)))
  g[[other]] <- base[[other]]
  g[, c("K_lon", "K_lat", "K_t")]
}

axis_label <- function(a)
  switch(a, K_lon = quote(K[lon]), K_lat = quote(K[lat]), K_t = quote(K[t]))

## Two-axis sensitivity surface: sigma_wK (z) as a function of two of the
## K's, with the third held fixed at base[[other]]. `sens` must come from
## run_grids() on a grid built by make_surface_grid() with the SAME
## axis1/vals1/axis2/vals2. type = "persp" gives a 3D surface (base R
## graphics only); "contour"/"filled.contour"/"image" give 2D fallbacks
## if the 3D view is hard to read.
plot_sigma_surface <- function(sens, base, axis1, vals1, axis2, vals2,
                               scale = c("sigma2", "sigma", "sigma2_hi", "sigma_hi"),
                               type = c("persp", "contour", "filled.contour", "image"),
                               col = "steelblue", theta = 35, phi = 25, ...) {
  scale <- match.arg(scale); type <- match.arg(type)
  zcol <- switch(scale, sigma2 = "sigma2_hat", sigma = "sigma_hat",
                sigma2_hi = "sigma2_hi", sigma_hi = "sigma_hi")
  z <- matrix(sens[[zcol]], nrow = length(vals1), ncol = length(vals2))
  other <- setdiff(c("K_lon", "K_lat", "K_t"), c(axis1, axis2))
  scale_lab <- switch(scale,
    sigma2    = bquote(hat(sigma)[list(w, K)]^2),
    sigma     = bquote(hat(sigma)[list(w, K)]),
    sigma2_hi = bquote(sigma[list(w, K)]^2 ~ "(95% CI upper)"),
    sigma_hi  = bquote(sigma[list(w, K)] ~ "(95% CI upper)"))
  main <- bquote(paste(.(scale_lab), ":  ", .(axis_label(axis1)), " vs ", .(axis_label(axis2)),
                       "  (", .(axis_label(other)) == .(base[[other]]), ")"))
  zlab <- scale_lab
  switch(type,
    persp = persp(vals1, vals2, z, xlab = axis1, ylab = axis2, zlab = zlab,
                  theta = theta, phi = phi, expand = 0.6, col = col,
                  shade = 0.4, ticktype = "detailed", main = main, ...),
    contour = contour(vals1, vals2, z, xlab = axis1, ylab = axis2,
                      main = main, ...),
    filled.contour = filled.contour(vals1, vals2, z, xlab = axis1, ylab = axis2,
                                    main = main,
                                    color.palette = function(n) hcl.colors(n, "YlOrRd", rev = TRUE),
                                    ...),
    image = {
      image(vals1, vals2, z, xlab = axis1, ylab = axis2, main = main,
            col = hcl.colors(64, "YlOrRd", rev = TRUE), ...)
      contour(vals1, vals2, z, add = TRUE)
    })
  invisible(z)
}

## --------------------------- demo / test ----------------------------
grids_iso <- grid_from_cellsize(box, cell_km_vec = c(200,110,90,50,10,7), K_t = 3)
curve_iso <- run_grids(argo, box, grids_iso)
par(mar = c(4.2, 4.5, 3, 1))
#plot_sigma_curve(curve_iso, xvar = "cell_km", scale = "sigma", xlab = "cell size (km)", main = "sigma_w vs cell size")
base <- c(K_lon = 6, K_lat = 4, K_t = 3)
sens <- make_sensitivity_grids(base, K_lon_vals = 6:72, K_lat_vals = 4:48, K_t_vals = 1:72)
sens_curve <- run_grids(argo, box, sens)
#plot_sensitivity(sens_curve, base, scale = "sigma")

vals_lon <- 1:12
vals_lat <- 1:8
base <- c(K_lon = 6, K_lat = 4, K_t = 1)
surf_grid <- make_surface_grid(base, "K_lon", vals_lon, "K_lat", vals_lat)
surf <- run_grids(argo, box, surf_grid)
plot_sigma_surface(surf, base, "K_lon", vals_lon, "K_lat", vals_lat, scale = "sigma_hi", type = "filled.contour")

vals_lon <- 6:72
vals_lat <- 4:48
base <- c(K_lon = 6, K_lat = 4, K_t = 12)
surf_grid <- make_surface_grid(base, "K_lon", vals_lon, "K_lat", vals_lat)
surf <- run_grids(argo, box, surf_grid)
png("plots/sigma_hi_Kt12.png", width = 8, height = 6, units = "in", res = 300)
plot_sigma_surface(surf, base, "K_lon", vals_lon, "K_lat", vals_lat, scale = "sigma_hi", type = "filled.contour")
dev.off()

vals_lon <- 6:72
vals_lat <- 4:48
base <- c(K_lon = 6, K_lat = 4, K_t = 3)
surf_grid <- make_surface_grid(base, "K_lon", vals_lon, "K_lat", vals_lat)
surf <- run_grids(argo, box, surf_grid)
png("plots/sigma_hi_Kt3.png", width = 8, height = 6, units = "in", res = 300)
plot_sigma_surface(surf, base, "K_lon", vals_lon, "K_lat", vals_lat, scale = "sigma_hi", type = "filled.contour")
dev.off()
