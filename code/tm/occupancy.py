"""Age-structured propagation of launches through P_move with survival s(a) -> expected floats, profiles, q-hat.

    n_{t+1,a+1} = s(a) n_{t,a} P_move,   n_{t,0} = launches in step t
    E[profiles in cell b, step t] = c_y(t) sum_a n_{t,a}(b),   q-hat = E / (state area x step length)

Linear in the launches, so launch cohorts propagate independently and their expected floats add.
"""

import numpy as np
import pandas as pd
import xarray as xr

from tm import transitions as tr


def steps(day0, day1, step_days):
    return np.arange(day0, day1 + step_days, step_days)


def step_years(t_edge):
    """Calendar year of each step's midpoint and the year edges (days) spanning them."""
    mid = (tr.EPOCH + pd.to_timedelta((t_edge[:-1] + t_edge[1:]) / 2, "D")).year.to_numpy()
    years = np.arange(mid.min(), mid.max() + 1)
    y_edge = tr._days(pd.Series(pd.to_datetime([f"{y}-01-01" for y in np.r_[years, years[-1] + 1]], utc=True)))
    return mid - years[0], years, y_edge


def source(grid, launches, d, t_edge):
    """Launch counts per (step, state); launches outside the state space fall back to the first profile."""
    s = grid.index(launches["lat"], launches["lon"])
    f = d.groupby("float_id")[["lat", "lon"]].first().reindex(launches["float_id"])
    s = np.where(s >= 0, s, grid.index(f["lat"], f["lon"]))
    k = np.searchsorted(t_edge, launches["day"].to_numpy(), side="right") - 1
    ok = (s >= 0) & (k >= 0) & (k < len(t_edge) - 1)
    src = pd.DataFrame({"k": k[ok], "s": s[ok]}).value_counts().reset_index(name="n")
    return src, int((~ok).sum())


def profile_rate(d, launches, y_edge, step_days, kind="empirical", profile_set=None):
    """Profiles per live float-step for each calendar year. 'one': 1. 'empirical': profiles in
    `profile_set` (default d) over float-steps between launch and last profile."""
    ny = len(y_edge) - 1
    if kind == "one":
        return np.ones(ny)
    p = d if profile_set is None else profile_set
    life = launches.set_index("float_id")["day"].to_frame("a").join(d.groupby("float_id")["day"].max().rename("b"))
    alive = np.clip(np.minimum(life["b"].to_numpy()[:, None], y_edge[1:]) -
                    np.maximum(life["a"].to_numpy()[:, None], y_edge[:-1]), 0, None).sum(0)
    yi = np.searchsorted(y_edge, p["day"].to_numpy(), side="right") - 1
    n = np.bincount(yi[(yi >= 0) & (yi < ny)], minlength=ny)
    return np.divide(n, alive / step_days, out=np.ones(ny), where=alive > 0)


def band_of_age(band_days, step_days, n_ages):
    """Age band index of each age step: band j = [band_days[j], band_days[j+1]) days."""
    return np.searchsorted(np.asarray(band_days, float), np.arange(n_ages) * step_days, side="right") - 1


def propagate(PT, s, src, ka, kb, n_steps, out_k, out_states, yk, band, n_bands):
    """Launch steps [ka, kb) as one column per launch step (age of column j at step k is k - ka - j).
    Returns expected floats in out_states for steps out_k[0]:out_k[1] by age band, and per (year, state)."""
    A = len(s)
    W = kb - ka
    j = np.arange(W)
    N = np.zeros((PT.shape[0], W), np.float32)
    launch = {k: (g["s"].to_numpy(), g["k"].to_numpy() - ka, g["n"].to_numpy(np.float32))
              for k, g in src[(src["k"] >= ka) & (src["k"] < kb)].groupby("k")}
    F_out = np.zeros((out_k[1] - out_k[0], len(out_states), n_bands), np.float32)
    Fy = np.zeros((yk.max() + 1, PT.shape[0]), np.float32)
    for k in range(ka, min(kb + A, n_steps)):
        age = k - ka - j
        if k > ka:
            prev = age - 1
            N = (PT @ N) * np.where((prev >= 0) & (prev < A), s[np.clip(prev, 0, A - 1)], 0.0).astype(np.float32)
        if k in launch:
            st, col, n = launch[k]
            np.add.at(N, (st, col), n)
        Fy[yk[k]] += N.sum(axis=1)
        if out_k[0] <= k < out_k[1]:
            live = (age >= 0) & (age < A)
            M = np.zeros((W, n_bands), np.float32)
            M[j[live], band[age[live]]] = 1.0
            F_out[k - out_k[0]] = N[out_states] @ M
    return F_out, Fy


def crop_states(grid, lat, lon):
    """States whose cell lies in the lat/lon crop, their (row, col) within it, and the crop's edges."""
    i0, i1 = np.searchsorted(grid.lat_edge, lat)
    j0, j1 = np.searchsorted(grid.lon_edge, lon)
    iy, ix = grid.yx(np.arange(grid.n))
    m = (iy >= i0) & (iy < i1) & (ix >= j0) & (ix < j1)
    return np.flatnonzero(m), iy[m] - i0, ix[m] - j0, grid.lat_edge[i0:i1 + 1], grid.lon_edge[j0:j1 + 1]


def qhat_dataset(grid, E_bands, band_days, t_edge, step_days, lat, lon):
    """q-hat per m2 per day (expected profiles / state area / day) in the layout osim.intensity.at reads:
    q_all and q_age<lo>_<hi> (age band [lo, hi) days since launch). E_bands: (time, crop states, band)."""
    st, ry, rx, la, lo = crop_states(grid, lat, lon)
    q = E_bands / (grid.area_km2.ravel()[grid.cells[st]] * 1e6 * step_days)[None, :, None]
    field = lambda v: _place(v, ry, rx, len(la) - 1, len(lo) - 1)
    dims = ("time", "lat", "lon")
    vars_ = {"q_all": (dims, field(q.sum(-1)))}
    for b, (a0, a1) in enumerate(zip(band_days[:-1], band_days[1:])):
        vars_[f"q_age{a0:g}_{a1:g}"] = (dims, field(q[..., b]))
    time = (tr.EPOCH + pd.to_timedelta(t_edge, "D")).tz_convert(None).to_numpy()
    return xr.Dataset(vars_, coords={"time_edge": ("time_e", time), "lat_edge": ("lat_e", la),
                                     "lon_edge": ("lon_e", lo)})


def _place(v, ry, rx, ny, nx):
    out = np.full((v.shape[0], ny, nx), np.nan, np.float32)
    out[:, ry, rx] = v
    return out
