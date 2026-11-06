"""Fixed gyre masks: the largest closed contour of mean sea surface height around a gyre's extremum.

A level c is closed if the water-connected component of {h >= c} (high) or {h <= c} (low) holding the seed never
reaches the open edge of the domain; coasts close contours. The spill level is the outermost closed c.
"""
import numpy as np, scipy.ndimage as ni

STG = dict(sign=1, seed=dict(lat=(25, 40), lon=(-75, -45)))
SPG = dict(sign=-1, seed=dict(lat=(50, 62), lon=(-58, -25)))


def _edge(shape):
    e = np.zeros(shape, bool)
    e[0], e[-1], e[:, 0], e[:, -1] = True, True, True, True
    return e


def _seed(h, lat, lon, seed, sign):
    b = (lat[:, None] >= seed["lat"][0]) & (lat[:, None] <= seed["lat"][1]) \
        & (lon[None] >= seed["lon"][0]) & (lon[None] <= seed["lon"][1])
    return np.unravel_index(np.nanargmax(np.where(b, sign * h, np.nan)), h.shape)


def component(h, c, seed_ij, sign):
    """Water-connected component (4-neighbour) of the level set holding the seed."""
    lab = ni.label(np.nan_to_num(sign * h, nan=-np.inf) >= sign * c)[0]
    return lab == lab[seed_ij]


def spill(h, lat, lon, sign, seed, step=5e-4):
    """Outermost closed level and its mask; levels descend from the extremum toward the open boundary."""
    ij, edge = _seed(h, lat, lon, seed, sign), _edge(h.shape)
    levels = np.arange(h[ij], h[ij] - sign * np.nanmax(np.abs(h - h[ij])), -sign * step)
    lo, hi = 0, len(levels) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if not (component(h, levels[mid], ij, sign) & edge).any() else (lo, mid)
    return levels[lo], component(h, levels[lo], ij, sign), component(h, levels[hi], ij, sign)


JET = dict(lat=(33, 46), lon=(-75, -35))
VEL = dict(lat=(11, 66), lon=(-87, -12))
OHC = dict(lat=(14, 55), lon=(-72, -32))
BARRIER_S, TROUGH_WEST, MED_LON = (5, 28), -61.0, -5.6
MARGINAL = [(-100, 5), (-100, 31), (-81.5, 31), (-80, 25.5), (-78.5, 23), (-74, 20), (-68.5, 18.5), (-64.5, 18.5),
            (-61, 16), (-61, 5)]


def cell_area(lat, lon):
    r = 6371.0088
    return (r * np.deg2rad(abs(lat[1] - lat[0]))) * (r * np.deg2rad(abs(lon[1] - lon[0]))) \
        * np.cos(np.deg2rad(lat))[:, None] * np.ones((len(lat), len(lon)))


def speed(h, lat, lon):
    """Mean geostrophic surface speed (m/s), undefined within 5 deg of the equator."""
    r = 6.371e6
    dy, dx = r * np.deg2rad(lat[1] - lat[0]), r * np.deg2rad(lon[1] - lon[0]) * np.cos(np.deg2rad(lat))[:, None]
    f = 2 * 7.292e-5 * np.sin(np.deg2rad(lat))[:, None]
    s = 9.81 / np.abs(f) * np.hypot(np.gradient(h, axis=1) / dx, np.gradient(h, axis=0) / dy)
    s[np.abs(lat) < 5] = np.nan
    return s


def jet_level(h, lat, lon):
    """Speed-weighted median of h over the Gulf Stream - North Atlantic Current corridor."""
    b = (lat[:, None] >= JET["lat"][0]) & (lat[:, None] <= JET["lat"][1]) \
        & (lon[None] >= JET["lon"][0]) & (lon[None] <= JET["lon"][1])
    w = speed(h, lat, lon) * cell_area(lat, lon)
    k = b & np.isfinite(h) & np.isfinite(w)
    o = np.argsort(h[k])
    return float(h[k][o][np.searchsorted(np.cumsum(w[k][o]), .5 * w[k].sum())])


def trough(h, lat, lon):
    """Latitude of the minimum of h in the 5-28N band by longitude, median-smoothed over 3 deg: the NEC/NECC divide."""
    b = (lat >= BARRIER_S[0]) & (lat <= BARRIER_S[1])
    s = np.where(np.isfinite(h[b]), h[b], np.inf)
    t = lat[b][s.argmin(0)].astype(float)
    t[~np.isfinite(h[b]).any(0)] = np.nan
    n = int(round(3 / abs(lon[1] - lon[0]))) | 1
    return ni.generic_filter(t, np.nanmedian, size=n, mode="nearest")


def marginal(lat, lon):
    from matplotlib.path import Path
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return Path(MARGINAL).contains_points(np.column_stack([LO.ravel(), LA.ravel()])).reshape(LA.shape)


def stg_jet(h, lat, lon, c=None, seed=STG["seed"]):
    """Subtropical gyre: water with h above the jet level, north of the tropical trough, outside the marginal seas."""
    c = jet_level(h, lat, lon) if c is None else c
    t = np.where(lon >= TROUGH_WEST, trough(h, lat, lon), -np.inf)
    keep = (lat[:, None] >= t[None]) & ~marginal(lat, lon) & (lon[None] <= MED_LON)
    return c, component(np.where(keep, h, np.nan), c, _seed(h, lat, lon, seed, 1), 1)


def spg_closed(h, lat, lon, seed=SPG["seed"]):
    """Subpolar gyre: spill-level low with the Nordic Seas cut at the velocity domain's 66N edge."""
    b = lat <= VEL["lat"][1]
    c, m, nxt = spill(h[b], lat[b], lon, -1, seed)
    out = np.zeros(h.shape, bool)
    out[b] = m
    return c, out


def plot_masks(ds, boxes, figsize=(10, 6.6)):
    """Subtropical and subpolar masks over mean SSH, with the plain closed contour, boxes and velocity domain."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    la, lo, h = ds.latitude.values, ds.longitude.values, ds.h.values
    frame = lambda b, **k: ax.plot(np.array(b["lon"])[[0, 1, 1, 0, 0]], np.array(b["lat"])[[0, 0, 1, 1, 0]], **k)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor("#d9d9d9")
    ax.grid(False)
    ax.contourf(lo, la, ds.stg.values.astype(float), [.5, 1.5], colors=["#d1495b"], alpha=.35)
    ax.contourf(lo, la, ds.spg.values.astype(float), [.5, 1.5], colors=["#1f77c9"], alpha=.35)
    ax.contour(lo, la, h, np.arange(-1.2, 1.5, .1), colors="#666666", linewidths=.35)
    ax.contour(lo, la, ds.stg_plain.values.astype(float), [.5], colors="#222222", linewidths=1.2, linestyles="-.")
    for k, b in boxes.items():
        frame(b, color="#222222", lw=1)
        ax.text(b["lon"][0] + .6, b["lat"][1] - 2.4, f"Box {k[-1]}", fontsize=9, color="#222222")
    frame(VEL, color="#0b4f8a", lw=1.4, ls="--")
    frame(OHC, color="#2a7f62", lw=1.4, ls="--")
    ax.set(xlim=(-100, 5), ylim=(0, 70), xlabel="longitude (deg)", ylabel="latitude (deg)", aspect=1.22)
    ax.legend(handles=[Patch(color="#d1495b", alpha=.35, label="subtropical mask"), Patch(color="#1f77c9", alpha=.35, label="subpolar mask"),
                       Line2D([], [], color="#222222", ls="-.", label="plain closed contour"),
                       Line2D([], [], color="#222222", label="boxes A and B"),
                       Line2D([], [], color="#0b4f8a", ls="--", label="velocity domain"),
                       Line2D([], [], color="#2a7f62", ls="--", label="OHC field domain"),
                       Line2D([], [], color="#666666", lw=.5, label="mean SSH, 0.1 m interval")],
              loc="lower left", fontsize=8, framealpha=.95)
    return fig


def inside(b, lat, lon):
    return (lat[:, None] >= b["lat"][0]) & (lat[:, None] <= b["lat"][1]) & (lon[None] >= b["lon"][0]) & (lon[None] <= b["lon"][1])


def data_coverage(ds):
    """Share of each mask's area with OHC and velocity data, velocity only, and neither."""
    la, lo = ds.latitude.values, ds.longitude.values
    a, v, o = cell_area(la, lo), inside(VEL, la, lo), inside(OHC, la, lo)
    return {k: {"both": a[(ds[k].values > 0) & v & o].sum() / a[ds[k].values > 0].sum(),
                "velocity": a[(ds[k].values > 0) & v & ~o].sum() / a[ds[k].values > 0].sum(),
                "neither": a[(ds[k].values > 0) & ~v & ~o].sum() / a[ds[k].values > 0].sum(),
                "area": a[ds[k].values > 0].sum()} for k in ("stg", "spg")}


def plot_coverage(ds, figsize=(12, 6)):
    """Each gyre mask coloured by the data held there, over mean SSH contours."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    la, lo, h = ds.latitude.values, ds.longitude.values, ds.h.values
    v, o = inside(VEL, la, lo), inside(OHC, la, lo)
    col = ["#d1495b", "#e0a030", "#17a673"]
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
    for a, k, name in zip(ax, ("stg", "spg"), ("Subtropical mask", "Subpolar mask")):
        c = np.where(ds[k].values > 0, np.where(v & o, 2.0, np.where(v, 1.0, 0.0)), np.nan)
        a.set_facecolor("#d9d9d9")
        a.grid(False)
        a.pcolormesh(lo[::3], la[::3], c[::3, ::3], cmap=ListedColormap(col), vmin=-.5, vmax=2.5, shading="auto", rasterized=True)
        a.contour(lo, la, h, np.arange(-1.2, 1.5, .1), colors="#666666", linewidths=.3)
        for b, cl in ((VEL, "#0b4f8a"), (OHC, "#2a7f62")):
            a.plot(np.array(b["lon"])[[0, 1, 1, 0, 0]], np.array(b["lat"])[[0, 0, 1, 1, 0]], color=cl, lw=1.4, ls="--")
        a.set(xlim=(-100, 5), ylim=(0, 70), xlabel="longitude (deg)", title=name, aspect=1.22)
    ax[0].set_ylabel("latitude (deg)")
    fig.legend(handles=[Patch(color=col[2], label="OHC and velocity"), Patch(color=col[1], label="velocity only"),
                        Patch(color=col[0], label="no data")], loc="outside lower center", ncol=3, fontsize=9)
    return fig
