"""Regular lat-lon grids whose states are the cells with nonzero ocean area."""

import numpy as np
import scipy.sparse as sp
from global_land_mask import globe

R_EARTH_KM = 6371.0088


def _deep(path, min_depth):
    """Point test deptho >= min_depth on a regular lat-lon bathymetry grid (nearest node; NaN = land)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        z = ds["deptho"].values
        la, lo = ds["latitude"].values.astype(float), ds["longitude"].values.astype(float)
    dla, dlo = (la[-1] - la[0]) / (len(la) - 1), (lo[-1] - lo[0]) / (len(lo) - 1)
    ok = np.nan_to_num(z, nan=0.0) >= min_depth

    def f(lat, lon):
        iy = np.rint((lat - la[0]) / dla).astype(int)
        ix = np.rint((lon - lo[0]) / dlo).astype(int) % len(lo)
        inside = (iy >= 0) & (iy < len(la))
        return inside & ok[np.clip(iy, 0, len(la) - 1), ix]
    return f


class Grid:
    """States are cells with some valid area: ocean (land mask), or bottom depth >= min_depth (bathymetry)."""

    def __init__(self, deg=0.5, lat=(-90.0, 90.0), lon=(-180.0, 180.0), min_depth=None, bathy=None,
                 support=None, support_radius=2, sub=10):
        self.deg = deg
        self.lat_edge = np.linspace(*lat, int(round((lat[1] - lat[0]) / deg)) + 1)
        self.lon_edge = np.linspace(*lon, int(round((lon[1] - lon[0]) / deg)) + 1)
        self.ny, self.nx = len(self.lat_edge) - 1, len(self.lon_edge) - 1
        self.periodic = bool(np.isclose(lon[1] - lon[0], 360.0))
        self._valid = globe.is_ocean if min_depth is None else _deep(bathy, min_depth)
        self.ocean_frac = self._ocean_frac(sub)
        if support is not None:
            self.ocean_frac *= self._support(*support, support_radius)
        band = np.diff(np.sin(np.deg2rad(self.lat_edge)))[:, None] * np.deg2rad(np.diff(self.lon_edge))[None, :]
        self.area_km2 = R_EARTH_KM**2 * band * self.ocean_frac
        self.cells = np.flatnonzero(self.ocean_frac.ravel() > 0)
        self.state = np.full(self.ny * self.nx, -1)
        self.state[self.cells] = np.arange(len(self.cells))

    @property
    def n(self):
        return len(self.cells)

    def _ocean_frac(self, sub):
        u = (np.arange(sub) + 0.5) / sub
        lon = (self.lon_edge[:-1, None] + u[None, :] * self.deg).ravel()
        frac = np.empty((self.ny, self.nx))
        for i, la0 in enumerate(self.lat_edge[:-1]):
            la, lo = np.meshgrid(la0 + u * self.deg, np.where(lon >= 180, lon - 360, lon), indexing="ij")
            frac[i] = self._valid(la, lo).reshape(sub, self.nx, sub).mean(axis=(0, 2))
        return frac

    def _support(self, lat, lon, radius):
        """Cells within `radius` cells of any of the given points (the sampled domain)."""
        from scipy.ndimage import binary_dilation
        lon = (np.asarray(lon, float) - self.lon_edge[0]) % 360.0 + self.lon_edge[0] if self.periodic else lon
        h = np.histogram2d(lat, lon, [self.lat_edge, self.lon_edge])[0] > 0
        if radius == 0:
            return h
        pad = ((radius, radius), (radius, radius) if self.periodic else (0, 0))
        h = np.pad(h, pad, mode="constant")
        if self.periodic:
            h[:, :radius], h[:, -radius:] = h[:, -2 * radius:-radius], h[:, radius:2 * radius]
        h = binary_dilation(h, np.ones((2 * radius + 1,) * 2, bool))
        return h[radius:-radius, radius:-radius] if self.periodic else h[radius:-radius]

    def index(self, lat, lon):
        """State index of each point; -1 off the grid or in a cell with no ocean area."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        if self.periodic:
            lon = (lon - self.lon_edge[0]) % 360.0 + self.lon_edge[0]
        iy = np.minimum(np.searchsorted(self.lat_edge, lat, side="right") - 1, self.ny - 1)
        ix = np.minimum(np.searchsorted(self.lon_edge, lon, side="right") - 1, self.nx - 1)
        ok = (iy >= 0) & (ix >= 0) & (lat <= self.lat_edge[-1]) & (lon <= self.lon_edge[-1])
        out = np.full(lat.shape, -1)
        out[ok] = self.state[iy[ok] * self.nx + ix[ok]]
        return out

    def yx(self, s):
        return np.divmod(self.cells[s], self.nx)

    def shift(self, s, dy, dx):
        """State reached from s by a (dy, dx) cell offset; -1 off the grid or on land."""
        iy, ix = self.yx(s)
        jy, jx = iy + dy, ix + dx
        if self.periodic:
            jx = jx % self.nx
        ok = (jy >= 0) & (jy < self.ny) & (jx >= 0) & (jx < self.nx)
        out = np.full(np.shape(s), -1)
        out[ok] = self.state[jy[ok] * self.nx + jx[ok]]
        return out

    def offsets(self, a, b):
        """Cell offsets (dy, dx) from state a to state b, dx wrapped to the short way round."""
        (ay, ax), (by, bx) = self.yx(a), self.yx(b)
        dx = bx - ax
        if self.periodic:
            dx = (dx + self.nx // 2) % self.nx - self.nx // 2
        return by - ay, dx

    def window(self, radius):
        """0/1 adjacency of ocean states within a (2r+1)^2 cell window; not water-connectivity."""
        s = np.arange(self.n)
        r, c = [], []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                j = self.shift(s, dy, dx)
                r.append(s[j >= 0])
                c.append(j[j >= 0])
        r, c = np.concatenate(r), np.concatenate(c)
        return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(self.n, self.n))
