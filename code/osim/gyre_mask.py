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
PLACES = [(-98, 40, "N. America"), (-58, 4, "S. America"), (-10, 16, "Africa"), (-2, 48, "Europe"), (-41, 68, "Greenland"),
          (-3, 37.5, "Med.")]
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


def _geo(ax, extent=(-100, 5, 0, 70)):
    """Land, coast, borders, place names and degree ticks on a PlateCarree axis."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#e6e1d6", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#333333", lw=.5, zorder=4)
    ax.set_xticks(np.arange(-100, extent[1] + 1, 20), crs=pc)
    ax.set_yticks(np.arange(np.ceil(extent[2] / 10) * 10, extent[3] + 1, 10), crs=pc)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=8)
    for x, y, n in PLACES:
        ax.text(x, y, n, transform=pc, fontsize=8, color="#555555", ha="center", va="center", zorder=5)
    return pc


def plot_masks(ds, boxes, figsize=(10.5, 7.4)):
    """Subtropical and subpolar masks over mean SSH, with the plain closed contour, boxes and data domains."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    la, lo, h = ds.latitude.values, ds.longitude.values, ds.h.values
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()), constrained_layout=True)
    pc = _geo(ax)
    frame = lambda b, **k: ax.plot(np.array(b["lon"])[[0, 1, 1, 0, 0]], np.array(b["lat"])[[0, 0, 1, 1, 0]], transform=pc, zorder=6, **k)
    ax.contourf(lo, la, ds.stg.values.astype(float), [.5, 1.5], colors=["#d1495b"], alpha=.35, transform=pc)
    ax.contourf(lo, la, ds.spg.values.astype(float), [.5, 1.5], colors=["#1f77c9"], alpha=.35, transform=pc)
    ax.contour(lo, la, h, np.arange(-1.2, 1.5, .1), colors="#666666", linewidths=.35, transform=pc)
    ax.contour(lo, la, ds.stg_plain.values.astype(float), [.5], colors="#222222", linewidths=1.2, linestyles="-.", transform=pc)
    for k, b in boxes.items():
        frame(b, color="#222222", lw=1)
        ax.text(b["lon"][0] + .6, b["lat"][1] - 2.4, f"Box {k[-1]}", fontsize=9, color="#222222", transform=pc, zorder=6)
    frame(VEL, color="#0b4f8a", lw=1.4, ls="--")
    frame(OHC, color="#2a7f62", lw=1.4, ls="--")
    ax.legend(handles=[Patch(color="#d1495b", alpha=.35, label="subtropical mask"), Patch(color="#1f77c9", alpha=.35, label="subpolar mask"),
                       Line2D([], [], color="#222222", ls="-.", label="plain closed contour"),
                       Line2D([], [], color="#222222", label="boxes A and B"),
                       Line2D([], [], color="#0b4f8a", ls="--", label="velocity domain"),
                       Line2D([], [], color="#2a7f62", ls="--", label="OHC field domain"),
                       Line2D([], [], color="#666666", lw=.5, label="mean SSH, 0.1 m interval")],
              loc="lower left", fontsize=8, framealpha=.95).set_zorder(7)
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


def plot_coverage(ds, figsize=(13, 6.2)):
    """Each gyre mask coloured by the data held there, over mean SSH contours."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    la, lo, h = ds.latitude.values, ds.longitude.values, ds.h.values
    v, o = inside(VEL, la, lo), inside(OHC, la, lo)
    col = ["#d1495b", "#e0a030", "#17a673"]
    fig, ax = plt.subplots(1, 2, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()), constrained_layout=True)
    for a, k, name in zip(ax, ("stg", "spg"), ("Subtropical mask", "Subpolar mask")):
        pc = _geo(a)
        c = np.where(ds[k].values > 0, np.where(v & o, 2.0, np.where(v, 1.0, 0.0)), np.nan)
        a.pcolormesh(lo[::3], la[::3], c[::3, ::3], cmap=ListedColormap(col), vmin=-.5, vmax=2.5, shading="auto", rasterized=True, transform=pc)
        a.contour(lo, la, h, np.arange(-1.2, 1.5, .1), colors="#666666", linewidths=.3, transform=pc)
        for b, cl in ((VEL, "#0b4f8a"), (OHC, "#2a7f62")):
            a.plot(np.array(b["lon"])[[0, 1, 1, 0, 0]], np.array(b["lat"])[[0, 0, 1, 1, 0]], color=cl, lw=1.4, ls="--", transform=pc, zorder=6)
        a.set_title(name, fontsize=11)
    fig.legend(handles=[Patch(color=col[2], label="OHC and velocity"), Patch(color=col[1], label="velocity only"),
                        Patch(color=col[0], label="no data")], loc="outside lower center", ncol=3, fontsize=9)
    return fig


def deep_domain(depth, lat, lon, z=2000.0, south=10.0, seed=(30.0, -50.0)):
    """Water at least z deep, north of south, connected through such water to the seed (lat, lon)."""
    d = np.nan_to_num(depth) >= z
    d[lat < south] = False
    lab = ni.label(d)[0]
    return lab == lab[np.argmin(np.abs(lat - seed[0])), np.argmin(np.abs(lon - seed[1]))]


def plot_domain(depth, lat, lon, z=2000.0, south=10.0, figsize=(11, 7.6)):
    """Bathymetry with depth contours and the outline of the connected deep domain."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    m = deep_domain(depth, lat, lon, z, south)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()), constrained_layout=True)
    pc = _geo(ax)
    f = ax.contourf(lon, lat, depth, np.arange(0, 6001, 500), cmap="Blues", extend="max", transform=pc)
    ax.contour(lon, lat, depth, [1000, 3000, 4000, 5000], colors="#ffffff", linewidths=.3, alpha=.7, transform=pc)
    ax.contour(lon, lat, depth, [z], colors="#222222", linewidths=.6, transform=pc)
    ax.contour(lon, lat, m.astype(float), [.5], colors="#d1495b", linewidths=1.6, transform=pc)
    fig.colorbar(f, ax=ax, shrink=.8, pad=.02, label="depth (m)")
    ax.legend(handles=[Line2D([], [], color="#d1495b", lw=1.6, label=f"domain: connected water at least {z:.0f} m deep, north of {south:.0f}N"),
                       Line2D([], [], color="#222222", lw=.6, label=f"{z:.0f} m depth contour"),
                       Line2D([], [], color="#9db8cf", lw=.6, label="1000, 3000, 4000, 5000 m contours")],
              loc="lower left", fontsize=8, framealpha=.95).set_zorder(7)
    return fig


def _xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def water_paths(depth, lat, lon, mask):
    """Shortest through-water distance (km) from every ocean cell to the mask's border cells on the 8-neighbour grid,
    without cutting land corners, and the grid index of the border cell each path reaches (-1 on land)."""
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    wet = np.nan_to_num(depth) > 0
    n, m = wet.shape
    idx = np.arange(n * m).reshape(n, m)
    dy, dx = 6371.0 * np.deg2rad(lat[1] - lat[0]), 6371.0 * np.deg2rad(lon[1] - lon[0]) * np.cos(np.deg2rad(lat))
    I, J, W = [], [], []
    for a, b, ok, w in (((slice(None), slice(None, -1)), (slice(None), slice(1, None)), None, dx[:, None] * np.ones((n, m - 1))),
                        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), None, dy * np.ones((n - 1, m))),
                        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), wet[:-1, 1:] & wet[1:, :-1],
                         np.hypot(dy, dx[:-1, None]) * np.ones((n - 1, m - 1))),
                        ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), wet[:-1, :-1] & wet[1:, 1:],
                         np.hypot(dy, dx[:-1, None]) * np.ones((n - 1, m - 1)))):
        k = wet[a] & wet[b] if ok is None else wet[a] & wet[b] & ok
        I.append(idx[a][k]); J.append(idx[b][k]); W.append(w[k])
    I, J, W = map(np.concatenate, (I, J, W))
    G = sparse.csr_matrix((np.r_[W, W], (np.r_[I, J], np.r_[J, I])), shape=(n * m, n * m))
    src = idx[mask & ~ni.binary_erosion(mask)]
    km, _, s = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True)
    return km.reshape(n, m), s.reshape(n, m)


def local_reach(km, src, lat, lon, mask, rlat, rlon, pct, window=300.0):
    """Extended regions enclosing pct% of records. Each border cell's reach is a quantile of the through-water distances
    of outside records whose path reaches the border within window km of it (0 where none did); one quantile per level
    is solved so the region holds exactly that share of all records. Returns {pct: mask}, the quantiles and the inside share."""
    from scipy.spatial import cKDTree
    i, j = np.searchsorted(lat, rlat).clip(0, len(lat) - 1), np.searchsorted(lon, rlon).clip(0, len(lon) - 1)
    wet = np.where(np.isfinite(km) & (src >= 0))
    dry = ~(np.isfinite(km[i, j]) & (src[i, j] >= 0))
    if dry.any():
        t = cKDTree(_xyz(lat[wet[0]], lon[wet[1]]))
        p = t.query(_xyz(rlat[dry], rlon[dry]))[1]
        i, j = i.copy(), j.copy()
        i[dry], j[dry] = wet[0][p], wet[1][p]
    inside = mask[i, j]
    dr, sr = km[i, j][~inside], src[i, j][~inside]
    b = np.unique(src[src >= 0])
    bi, bj = np.unravel_index(b, km.shape)
    bx = _xyz(lat[bi], lon[bj])
    ri, rj = np.unravel_index(sr, km.shape)
    near = cKDTree(_xyz(lat[ri], lon[rj])).query_ball_point(bx, 2 * np.sin(window / 2 / 6371.0))
    sd = [np.sort(dr[k]) for k in near]
    pos = np.searchsorted(b, sr)

    def reach(q):
        return np.array([x[int(np.ceil(q * len(x))) - 1] if len(x) else 0.0 for x in sd])

    out, Q = {}, {}
    for p in pct:
        target = (p / 100 - inside.mean()) / (1 - inside.mean())
        lo, hi = 0.0, 1.0
        for _ in range(22):
            q = (lo + hi) / 2
            lo, hi = (q, hi) if (dr <= reach(q)[pos]).mean() < target else (lo, q)
        full = np.zeros(km.shape)
        ok = src >= 0
        full[ok] = reach(hi)[np.searchsorted(b, src[ok])]
        out[p], Q[p] = (ok & (km <= full)) | mask, hi
    return out, Q, inside.mean()


def plot_launches(ds, launches, inside, extent=(-100, 25, -8, 82), figsize=(11, 8.6)):
    """Launch positions, domain border and the extended regions enclosing given shares of the floats' records."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    la, lo = ds.latitude.values, ds.longitude.values
    pct = [float(v.split("_")[1]) for v in ds.data_vars if v.startswith("ext_")]
    col = ["#7fb6e6", "#3f94d8", "#1f77c9", "#0b4f8a"]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()), constrained_layout=True)
    pc = _geo(ax, extent)
    h = []
    for p, c in zip(pct, col):
        ax.contour(lo, la, ds[f"ext_{p:g}"].values.astype(float), [.5], colors=c, linewidths=1.0, transform=pc, zorder=2)
        h.append(Line2D([], [], color=c, lw=1.2, label=f"{p:g}% of profiles"))
    ax.contour(lo, la, ds.domain.values.astype(float), [.5], colors="#d1495b", linewidths=.7, transform=pc, zorder=6)
    ax.scatter(launches.lon, launches.lat, s=5, color="#222222", alpha=.55, linewidths=0, transform=pc, zorder=7)
    ax.legend(handles=[Line2D([], [], color="#d1495b", lw=.7, label=f"domain border ({inside:.0%} of profiles inside)"),
                       Line2D([], [], color="#222222", marker="o", ms=3, ls="", alpha=.7, label=f"launch positions ({len(launches):,} floats)"), *h],
              loc="lower left", fontsize=8, framealpha=.95).set_zorder(8)
    return fig


def plot_download(ds, box, level=99, extent=(-100, 25, -8, 82), figsize=(11, 8.6)):
    """Suggested velocity download box against the domain and one extended region; box is dict(lat=, lon=)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    la, lo = ds.latitude.values, ds.longitude.values
    (s, n), (w, e) = box["lat"], box["lon"]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()), constrained_layout=True)
    pc = _geo(ax, extent)
    ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec="#17a673", lw=2, transform=pc, zorder=5))
    ax.contour(lo, la, ds[f"ext_{level:g}"].values.astype(float), [.5], colors="#1f77c9", linewidths=1.0, transform=pc, zorder=4)
    ax.contour(lo, la, ds.domain.values.astype(float), [.5], colors="#d1495b", linewidths=.7, transform=pc, zorder=6)
    ax.legend(handles=[Line2D([], [], color="#17a673", lw=2, label=f"suggested velocity download ({s}-{n}N, {abs(w)}W-{e}E)"),
                       Line2D([], [], color="#1f77c9", lw=1.2, label=f"{level:g}% of profiles"),
                       Line2D([], [], color="#d1495b", lw=.7, label="domain border")],
              loc="lower left", fontsize=8, framealpha=.95).set_zorder(8)
    return fig
