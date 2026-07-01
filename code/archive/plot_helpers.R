library(tidyverse)
library(data.table)
library(patchwork)
library(latex2exp)

float_obs_hist <- function(df, ...) {
  obs_p_float <- df[, .(obs = .N), float_id]
  print(summary(obs_p_float$obs))
  hist(obs_p_float$obs, ...)
}

plot_velocity_heatmap <- function(df, z, title = z, binwidth = 0.01, midpoint = 0, ocean = F) {
  p <- ggplot(df, aes(x = u_ms, y = v_ms, z = .data[[z]])) +
    stat_summary_2d(fun = mean, binwidth = binwidth) +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = midpoint, name = z
    ) +
    labs(x = "u (m/s)", y = "v (m/s)", title = title) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.background = element_rect(fill = "grey80", colour = NA)
    ) +
    coord_equal()

  if (ocean) {
    p <- p + facet_wrap(~ocean)
  }
  p
}

map_sum_pts <- function(df, x = "lon", y = "lat", z = "u_ms",
                        xlim = NULL, ylim = NULL,
                        binwidth = c(1, 1),
                        voption = "viridis") {
  ggplot(df, aes(x = .data[[x]], y = .data[[y]], z = .data[[z]])) +
    borders("world", fill = "gray80", colour = "gray80") +
    stat_summary_2d(aes(fill = after_stat(value)), fun = mean, binwidth = binwidth) +
    scale_fill_viridis_c(option = voption, name = z) +
    coord_sf(xlim = xlim, ylim = ylim) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.background = element_rect(fill = "grey90", colour = NA)
    ) +
    guides(fill = guide_colorbar())
}


map_trace <- function(
  df, x = "lon", y = "lat", z = "u_ms",
  id = "float_id", date = "date",
  xlim = NULL, ylim = NULL, linewidth = 0.25, voption = "viridis"
) {
  setorderv(df, c(id, date))
  ggplot(df, aes(
    x = .data[[x]], y = .data[[y]],
    group = .data[[id]], color = .data[[z]]
  )) +
    borders("world", fill = "gray80", colour = "gray80") +
    geom_path(linewidth = linewidth) +
    scale_color_viridis_c(option = voption) +
    coord_sf(xlim = xlim, ylim = ylim) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.background = element_rect(fill = "grey90", colour = NA)
    ) +
    guides(color = guide_colorbar())
}


map_grid <- function(df, f, z, voptions = NULL, title = NULL, subtitle = NULL, ncol = NULL, ...) {
  if (is.null(voptions)) voptions <- rep("viridis", length(z))
  if (length(voptions) == 1) voptions <- rep(voptions, length(z))

  plots <- Map(
    \(zvar, vopt)
    f(df, z = zvar, voption = vopt, ...),
    z, voptions
  )

  wrap_plots(plots, ncol = ncol) +
    plot_annotation(title = title, subtitle = subtitle)
}
