library(tidyverse)
library(data.table)
library(patchwork)
library(latex2exp)
library(gganimate)

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

map_sum_pts_anim <- function(df, x = "lon", y = "lat", z = "u_ms", time = "year", xlim = NULL, ylim = NULL,
                             binwidth = c(1, 1), voption = "viridis", fps = 2, filename = NULL) {
  p <- map_sum_pts(df, x, y, z, xlim, ylim, binwidth, voption) +
    transition_states(.data[[time]], transition_length = 0, state_length = 1) +
    labs(title = "{closest_state}")

  a <- animate(
    p,
    nframes = length(unique(df[[time]])),
    fps = fps,
    detail = 1,
    width = 900,
    height = 500,
    renderer = av_renderer()
  )

  if (!is.null(filename)) {
    anim_save(filename, a)
  }

  # return(a)
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

map_trace_anim <- function(
  df,
  x = "lon", y = "lat", z = "u_ms",
  id = "float_id",
  obs = "float_obs_n",
  date = "date",
  title = title,
  xlim = NULL, ylim = NULL,
  linewidth = 0.25,
  voption = "viridis",
  wake_length = 0.05,
  fps = 5,
  filename = NULL
) {
  setorderv(df, c(id, date))

  df[, `:=`(
    xstart = shift(.SD[[x]], type = "lag"),
    ystart = shift(.SD[[y]], type = "lag")
  ), by = id]

  df[is.na(xstart), xstart := .SD[[x]]]
  df[is.na(ystart), ystart := .SD[[y]]]

  # df[, n_floats := uniqueN(float_id), obs]

  p <- ggplot(df) +
    labs(
      title = title,
      subtitle = paste(obs, "{closest_state}")
    ) +
    borders("world", fill = "gray80", colour = "gray80") +
    geom_segment(
      aes(
        xend = .data[[x]],
        yend = .data[[y]],
        x = xstart,
        y = ystart,
        color = .data[[z]],
        group = interaction(.data[[id]], .data[[obs]])
      ),
      linewidth = linewidth
    ) +
    geom_point(
      aes(
        x = .data[[x]],
        y = .data[[y]],
        color = .data[[z]],
        group = .data[[id]]
      ),
      size = 0.9
    ) +
    scale_color_viridis_c(option = voption) +
    coord_sf(xlim = xlim, ylim = ylim) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.background = element_rect(fill = "grey90", colour = NA)
    ) +
    guides(color = guide_colorbar()) +
    transition_states(.data[[obs]], transition_length = 0, state_length = 1) +
    # transition_manual(float_obs_n) +
    shadow_wake(wake_length = wake_length, wrap = FALSE, alpha = FALSE) +
    ease_aes("linear")

  a <- animate(
    p,
    nframes = max(df$float_obs_n),
    fps = fps,
    detail = 1,
    width = 900,
    height = 500,
    renderer = av_renderer()
  )

  if (!is.null(filename)) {
    anim_save(filename, a)
  }

  # return(a)
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
