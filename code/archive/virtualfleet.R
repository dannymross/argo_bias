library(tidyverse)
library(data.table)
library(tidync)

tds <- tidync("data/virtualfleet/trajectory_85179626.nc") %>%
  hyper_tibble() %>%
  setDT()

tds[, obs := as.numeric(obs)]
tds[, time_d := time / (60^2 * 24)]

# Cycle phase (init_descend = 0, drift = 1, profile_descend = 2, profile_ascend = 3, transmit = 4)

tds[, dt_s := c(NA, diff(cycle_age))]
tds[cycle_phase == 1]
tds[trajectory == 198 & obs >= 230 & obs <= 300]
x <- tds[trajectory == 199]
plot(x$time_d, x$z, type = "l")


# x1[cycle_phase==1, .N, z]
x1[, .SD[c(1, .N)], .(cycle_number, cycle_phase)]

plot_profiles <- function(tds) {
  cols <- rainbow(length(unique(tds$trajectory)))
  plot(NULL,
    xlim = range(tds$time_d),
    ylim = rev(range(tds$z, na.rm = TRUE)),
    xlab = "time_d",
    ylab = "z"
  )

  for (i in seq_along(unique(tds$trajectory))) {
    tr <- unique(tds$trajectory)[i]
    x <- tds[trajectory == tr]

    lines(x$time_d, x$z, col = cols[i])
  }
}

plot_profiles(tds)
