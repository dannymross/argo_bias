# 1. Setup a 1D grid (a cross-section through our ridges)
n_pts <- 100
x <- seq(-1, 1, length.out = n_pts)

# Two ridges in 1D
y_target <- exp(-15 * (x - 0.5)^2) + exp(-15 * (x + 0.5)^2)
D <- as.matrix(dist(x))

# 2. Kernel and Fitting Functions (Matérn 1/2)
matern12 <- function(d, rho) exp(-d / rho)

fit_1d <- function(rho, D, y, nugget = 1e-5) {
  K <- matern12(D, rho)
  K_reg <- K + diag(nugget, nrow(K))
  
  alpha <- solve(K_reg, y)
  
  # Calculate the SCALED kernels: sweep() multiplies each column of K by its respective alpha
  scaled_kernels <- sweep(K, 2, alpha, "*") 
  
  # The final fitted function is the sum of the scaled kernels
  y_fit <- K %*% alpha
  norm_sq <- as.numeric(t(alpha) %*% K %*% alpha)

  return(list(K = K, alpha = alpha, scaled = scaled_kernels, y_fit = y_fit, norm_sq = norm_sq))
}

# 3. Fit both scenarios
res_large <- fit_1d(2, D, y_target)
res_small <- fit_1d(0.05, D, y_target)

# 4. Visualization Setup
# We will only plot every 3rd kernel so the lines don't completely paint over each other
idx <- seq(1, n_pts, by = 3)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))

# === TOP ROW: LARGE LENGTH SCALE (rho = 1.5) ===

# 1. Unscaled Kernels
plot(x, y_target, type="n", ylim=c(0, 1.2), main="Unscaled Kernels (rho = 2)", ylab="y")
for(i in idx) lines(x, res_large$K[,i], col=rgb(0.2, 0.5, 0.8, 0.5), lwd=2)
lines(x, y_target, col="black", lwd=3) # Plot the final sum

# 2. Scaled Kernels (The Destructive Interference)
title_large <- sprintf("Scaled Kernels (Norm: %1.0f)", res_large$norm_sq)
plot(x, y_target, type="n", main=title_large, ylab="y", ylim = c(0,1.2))
for(i in idx) lines(x, res_large$scaled[,i], col=rgb(0.8, 0.2, 0.2, 0.4), lwd=2)
lines(x, y_target, col="black", lwd=3) # Plot the final sum
lines(x, res_large$y_fit, col="red", lwd=1) # Plot the final sum

# === BOTTOM ROW: SMALL LENGTH SCALE (rho = 0.1) ===
# 3. Unscaled Kernels
plot(x, y_target, type="n", main="Unscaled Kernels (rho = 0.05)", ylab="y", ylim = c(0,1.2))
for(i in idx) lines(x, res_small$K[,i], col=rgb(0.2, 0.5, 0.8, 0.5), lwd=2)
lines(x, y_target, col="black", lwd=3) # Plot the final sum

# 4. Scaled Kernels (Natural Alignment)
title_small <- sprintf("Scaled Kernels (Norm: %1.2f)", res_small$norm_sq)
plot(x, y_target, type="n", main=title_small, ylab="y", ylim = c(0,1.2))
for(i in idx) lines(x, res_small$scaled[,i], col=rgb(0.8, 0.2, 0.2, 0.4), lwd=2)
lines(x, y_target, col="black", lwd=3) # Plot the final sum
lines(x, res_small$y_fit, col="red", lwd=1) # Plot the final sum

par(mfrow=c(1,1))


## Load the fields library for the legend scale
#library(fields)
#
## 1. Setup the spatial grid
#n_pts <- 25
#x <- seq(-1, 1, length.out = n_pts)
#grid <- expand.grid(x1 = x, x2 = x)
#
## Define a target surface with TWO diagonal ridges using Gaussian functions
#y_target <- exp(-15 * (grid$x1 + grid$x2 - 0.5)^2) + 
#            exp(-15 * (grid$x1 + grid$x2 + 0.5)^2)
#            
#D <- as.matrix(dist(grid))
#
## 2. Define the Matern 1/2 Kernel (Exponential)
#matern12 <- function(d, rho) {
#  exp(-d / rho)
#}
#
## 3. Core fitting function (Now returning residuals)
#fit_rkhs <- function(rho, D, y, nugget = 1e-5) {
#  K <- matern12(D, rho)
#  
#  # Add nugget for numerical stability against near-singular matrices
#  K_reg <- K + diag(nugget, nrow(K))
#  
#  alpha <- solve(K_reg, y)
#  y_fit <- K %*% alpha
#  
#  # Calculate the residual (Target - Fit)
#  residual <- y - y_fit
#  
#  norm_sq <- as.numeric(t(alpha) %*% K %*% alpha)
#  
#  return(list(norm_sq = norm_sq, alpha = alpha, 
#              y_fit = y_fit, residual = residual))
#}
#
## 4. Visualization Functions
#plot_surface <- function(z_vector, title) {
#  z_matrix <- matrix(z_vector, nrow = n_pts, ncol = n_pts)
#  persp(x, x, z_matrix, theta = 35, phi = 30, expand = 0.6,
#        col = "lightblue", shade = 0.5,
#        zlab = "z", main = title)
#}
#
#plot_alphas <- function(alpha_vector, title) {
#  a_matrix <- matrix(abs(alpha_vector), nrow = n_pts, ncol = n_pts)
#  image.plot(x, x, a_matrix, 
#             col = hcl.colors(50, "Inferno"), main = title,
#             xlab = "x1", ylab = "x2", legend.shrink = 0.8)
#}
#
#plot_residual <- function(res_vector, title) {
#  r_matrix <- matrix(res_vector, nrow = n_pts, ncol = n_pts)
#  # Use a diverging Blue-Red palette for residuals centered at 0
#  image.plot(x, x, r_matrix, 
#             col = hcl.colors(50, "Blue-Red 3"), main = title,
#             xlab = "x1", ylab = "x2", legend.shrink = 0.8)
#}
#
## 5. Run the experiment
#rho_large <- 1.5   
#res_large <- fit_rkhs(rho_large, D, y_target)
#
#rho_small <- 0.1   
#res_small <- fit_rkhs(rho_small, D, y_target)
#
## 6. Print numerical results
#cat("=== RKHS Norm Comparison ===\n")
#cat(sprintf("Norm (Large rho = 1.5): %12.2f\n", res_large$norm_sq))
#cat(sprintf("Norm (Small rho = 0.1): %12.2f\n", res_small$norm_sq))
#
## 7. Render Plots in a 2x4 Grid
#par(mfrow = c(2, 4))
#
## Top row: Large length scale 
#plot_surface(y_target, "Target (Two Ridges)")
#plot_surface(res_large$y_fit, "Fit (rho = 1.5)")
#plot_alphas(res_large$alpha, "|Alpha| Weights")
#plot_residual(res_large$residual, "Residual (Target - Fit)")
#
## Bottom row: Small length scale 
#plot_surface(y_target, "Target (Two Ridges)")
#plot_surface(res_small$y_fit, "Fit (rho = 0.1)")
#plot_alphas(res_small$alpha, "|Alpha| Weights")
#plot_residual(res_small$residual, "Residual (Target - Fit)")
#
#par(mfrow = c(1, 1))
