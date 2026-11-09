"""Shared configuration and grid helpers for the float-simulation study."""

import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLORYS_DIR = os.path.join(REPO_ROOT, "data", "glorys_osim")
OUT_DIR = os.path.join(REPO_ROOT, "data", "osim")

BOXES = {
    # Contiguous at 34 N. Bounds are inclusive at both ends, and the native grid
    # has a row exactly at 34.0, so box A stops just below it -- otherwise that row
    # falls in both boxes and is double counted in the A u B estimand. Box B owns
    # the 34-35 N transition strip, the most energetic part of the old gap.
    "boxB": {"lat_bounds": (34.0, 45.0), "lon_bounds": (-60.0, -40.0)},
    "boxA": {"lat_bounds": (22.0, 33.99), "lon_bounds": (-60.0, -40.0)},
    # The conditioning region: both boxes plus ~10 deg, trimmed to the deep basin.
    # Kriging near a box edge needs profiles from outside it, so the residual field
    # is built over this and the boxes are slices of it.
    "buffer": {"lat_bounds": (14.0, 55.0), "lon_bounds": (-72.0, -32.0)},
}

# Depth-layer OHC store written by download_ohc_layers.py; the four layers sum to
# the 0-2000 m integral.
LAYER_DIR = "ohc_layers"
STATIC_PREFIX = "static_buffer/buffer"

# Deep North Atlantic store written by download_natl.py. Raw files stay on scratch.
NATL_BOX = {"lat_bounds": (0.0, 70.0), "lon_bounds": (-90.0, 5.0)}
NATL_DIR = os.path.join(GLORYS_DIR, "natl")
NATL_RAW = "/scratch/dmr6737/glorys_natl"

ANALYSIS_YEARS = (2015, 2016, 2017, 2018, 2019)
FIT_START_YEAR = 1995
FIT_END_YEAR = 2019
ZMAX = 2000  # integration depth; int so ohc.truth_ohc_field names the var "ohc_2000"
MIN_DEPTH = 2000.0  # bathymetry mask: retain deptho > MIN_DEPTH

# Every year is forced to 365 days by dropping Feb 29, so a day-of-year index is
# comparable across years (the design's day-of-year alignment requirement).
DAYS_PER_YEAR = 365

COARSE_DEG = 1.0
COARSE_DAYS = 10


def box_slice(ds, box):
    la0, la1 = BOXES[box]["lat_bounds"]
    lo0, lo1 = BOXES[box]["lon_bounds"]
    return ds.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))


def cell_edges(centres):
    """Midpoint edges of a 1-D coordinate, extended half a step at each end."""
    c = np.asarray(centres, dtype=float)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def containing_index(values, centres):
    """Index of the grid cell containing each value; -1 outside the grid.

    Uses midpoint edges rather than arithmetic on a nominal spacing, so it is
    exact even though the GLORYS coordinates are stored as float32.
    """
    edges = cell_edges(centres)
    idx = np.searchsorted(edges, np.asarray(values, dtype=float), side="right") - 1
    idx[(idx < 0) | (idx >= len(centres))] = -1
    return idx


def drop_leap_day(ds, time_dim="time"):
    t = ds[time_dim].dt
    return ds.sel({time_dim: ~((t.month == 2) & (t.day == 29))})


def decimal_year(times):
    """Calendar year plus fraction of year elapsed.

    The integer part carries the trend; the fractional part drives the
    harmonics, so sin/cos(2*pi*t) is automatically phase-locked to the calendar.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    year = times.astype("datetime64[Y]").astype(int) + 1970
    y0 = times.astype("datetime64[Y]")
    y1 = y0 + np.timedelta64(1, "Y")
    elapsed = (times - y0.astype("datetime64[ns]")) / np.timedelta64(1, "D")
    length = (y1.astype("datetime64[ns]") - y0.astype("datetime64[ns]")) / np.timedelta64(1, "D")
    return year + elapsed / length


def _default_t_ref():
    return 0.5 * (FIT_START_YEAR + FIT_END_YEAR + 1)


def design_matrix(t, quadratic=True, t_ref=None):
    """[1, t, t^2, annual sin/cos, semiannual sin/cos] with the trend centred.

    Pointwise basis: use for evaluating the fitted mean at an instant (daily).
    Centring only affects conditioning; fitted values are unchanged.
    """
    t = np.asarray(t, dtype=float)
    if t_ref is None:
        t_ref = _default_t_ref()
    tc = t - t_ref
    cols = [np.ones_like(tc), tc]
    if quadratic:
        cols.append(tc**2)
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    return np.column_stack(cols), t_ref


def design_matrix_averaged(t0, t1, quadratic=True, t_ref=None):
    """Exact average of each basis function over [t0, t1).

    GLORYS monthly files hold a month *mean* but are stamped at the first of the
    month. Fitting them against the pointwise basis therefore imposes a ~15-day
    phase error on the harmonics (and attenuates their amplitude, since the mean
    of a sinusoid over a month is not the sinusoid at a point). Integrating the
    basis over the averaging window removes both exactly rather than
    approximately.
    """
    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    if t_ref is None:
        t_ref = _default_t_ref()
    a, b, dt = t0 - t_ref, t1 - t_ref, t1 - t0

    cols = [np.ones_like(a), 0.5 * (a + b)]
    if quadratic:
        cols.append((b**3 - a**3) / (3.0 * dt))
    for k in (1, 2):
        w = 2 * np.pi * k
        cols += [(np.cos(w * t0) - np.cos(w * t1)) / (w * dt),
                 (np.sin(w * t1) - np.sin(w * t0)) / (w * dt)]
    return np.column_stack(cols), t_ref
