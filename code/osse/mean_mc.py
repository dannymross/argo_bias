"""Monte-Carlo sampling distributions of the residual-field mean.

The estimand is the domain mean Zbar_k = int Z_k dP, with P the normalised
area x time measure on box B's wet native cells. A design {x_i} of size n gives
Zhat = mean_i Z_k(x_i) and bias B = Zhat - Zbar_k.

Three ensembles are compared, each advective design paired with synthetic
designs of exactly its own n so that only the point *locations* differ:

  advective  the advected uniform deployment, from
             sampled_residuals_<tag>_<box>.csv
  U-full     n iid draws, cell ~ area, doy ~ Uniform{0..364}; E[B] = 0
  U-doy      n iid draws over cells, holding the advective design's own doys,
             which isolates the bias attributable to *when* the array samples

E[B] for U-doy has a closed form -- mean_i Zbar_k(doy_i) - Zbar_k, where
Zbar_k(d) is the spatial mean on day d -- so it is computed exactly rather than
simulated. Values are J/m^2 throughout; J_TO_GJ converts for display.

    python code/osse/mean_mc.py [--box boxB]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg

J_TO_GJ = 1e-9


def _field_years(box="boxB"):
    """Analysis years actually present in the residual field, not the config.

    The field is what every downstream column name is keyed to, so rebuilding it
    over a wider window must not require editing cfg.ANALYSIS_YEARS in lockstep.
    """
    p = os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc")
    if os.path.exists(p):
        with xr.open_dataset(p) as ds:
            return [int(y) for y in ds["year"].values]
    return list(cfg.ANALYSIS_YEARS)


YEARS = _field_years()
ZCOLS = [f"z_{y}" for y in YEARS]


def set_box(box):
    """Rebind the analysis-year list to `box`'s residual field.

    Every z_/zhat_/temporal_/deff_ column name is keyed to YEARS, so a run on a
    box other than boxB must switch it before anything else -- otherwise the
    columns are silently keyed to boxB's window.
    """
    global YEARS, ZCOLS
    YEARS = _field_years(box)
    ZCOLS = [f"z_{y}" for y in YEARS]
    return YEARS


def area_weights(box="boxB"):
    """Flat index of the analysis mask and the P weights on it."""
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    a = a.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return wet, a[wet] / a[wet].sum()


def load_field(box="boxB", tag=None):
    """Flattened residual field, the wet-cell index, and the P weights on it."""
    z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{tag or box}.nc"))["z"]
    wet, w = area_weights(box)
    return z.values.reshape(z.sizes["year"], z.sizes["doy"], -1), wet, w


def domain_means(zf, wet, w):
    """Zbar_k and sd_P(Z_k), plus the spatial mean on each day of year.

    Both moments are taken under P, so sd_Z is area-weighted; the cached
    build_residual_field value is unweighted and differs by well under 1%.
    """
    by_doy = np.empty((zf.shape[0], zf.shape[1]))
    m2 = np.empty_like(by_doy)
    for i in range(zf.shape[0]):
        zw = zf[i][:, wet].astype(np.float64)
        by_doy[i] = zw @ w
        m2[i] = (zw * zw) @ w
    zbar = by_doy.mean(axis=1)
    return pd.DataFrame({"year": YEARS, "zbar": zbar,
                         "sd_Z": np.sqrt(m2.mean(axis=1) - zbar**2)}), by_doy


def sampled_path(box, tag):
    """Where a gathered arm lives. The box is part of the name, not optional.

    Both boxes run the same arms, so a box-free name lets a box A gather
    silently clobber box B's file -- which is what happened once, and cost a
    re-gather to notice and undo.
    """
    return os.path.join(cfg.OUT_DIR, f"sampled_residuals_{tag}_{box}.csv")


def gather_arm(box="boxB", arm="steady", out_tag=None):
    """Concatenate an arm's per-chunk profile CSVs and evaluate Z at each point.

    Chunked array jobs write design_profiles_<box>_<arm>_y<YYYY>_c<c>.csv; the
    rep labels are already disjoint across chunks (--rep-offset), so a plain
    concatenation is the full design ensemble.
    """
    from osse import sample_residuals as sr
    pat = os.path.join(cfg.OUT_DIR, f"design_profiles_{box}_{arm}_y*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(pat)
    prof = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    dup = prof.duplicated(["vel_year", "rep", "float_id", "cycle"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (vel_year, rep, float_id, cycle) rows: "
                         "chunk rep-offsets overlap")
    sampled, ndropped = sr.gather(prof, sr.load_residual_field(box))
    out = sampled_path(box, out_tag or arm)
    sampled.to_csv(out, index=False)
    print(f"{len(files)} files, {len(sampled):,} profiles kept, "
          f"{ndropped:,} dropped (masked/outside) -> {out}")
    return sampled


def thin_floats(df, keep, n_deployed, seed=17):
    """Drop each replicate's floats down to `keep` of the `n_deployed` seeded.

    Exact rather than approximate. deployment_plan_steady draws every float's
    seed position, release day and lifespan iid, and Parcels advects particles
    independently, so a uniform subset of k of a replicate's seeded floats is
    itself an iid k-float design from the same law -- identical in distribution
    to a fresh k-float run, not merely similar. Thinning is applied to the
    *seeded* ids, not to the floats that happen to reach the box, or floats in
    retentive years would be preferentially kept.
    """
    if keep >= n_deployed:
        raise ValueError(f"keep={keep} not below n_deployed={n_deployed}")
    rng = np.random.default_rng(seed)
    slot = df["float_id"].to_numpy() % n_deployed
    reps = df["rep"].to_numpy()
    # One independent draw per (vel_year, rep): hash the pair into a row of the
    # keep-mask so the same float slot is not kept in every design.
    yr = pd.factorize(df["vel_year"], sort=True)[0]
    ndes = (yr.max() + 1) * (reps.max() + 1)
    mask = np.zeros((ndes, n_deployed), dtype=bool)
    mask[:, :keep] = True
    for i in range(ndes):
        rng.shuffle(mask[i])
    return df[mask[yr * (reps.max() + 1) + reps, slot]]


def advective_estimates(box="boxB", tag="uniform_mc", path=None, keep=None,
                        n_deployed=None, seed=17):
    """Per-design Zhat, sample size, and the float-clustered design effect.

    The design effect is sum_f (cluster sum of centred z)^2 / sum_i (centred
    z)^2: consecutive cycles of one float are near-replicates, so it is the
    factor by which the effective sample size falls short of n.

    `keep` thins each replicate to that many of the `n_deployed` seeded floats.
    """
    path = path or sampled_path(box, tag)
    df = pd.read_csv(path, usecols=["vel_year", "rep", "float_id", "doy"] + ZCOLS)
    if keep is not None:
        df = thin_floats(df, keep, n_deployed, seed)
    key = ["vel_year", "rep"]
    g = df.groupby(key, sort=True)

    out = g[ZCOLS].mean().rename(columns=lambda c: c.replace("z_", "zhat_"))
    out["n"] = g.size()
    out["n_floats"] = g["float_id"].nunique()

    ctr = df[ZCOLS] - g[ZCOLS].transform("mean")
    ss = ctr.pow(2).groupby([df[k] for k in key]).sum()
    cl = ctr.groupby([df[k] for k in key] + [df["float_id"]]).sum()
    ss_cl = cl.pow(2).groupby(level=[0, 1]).sum()
    deff = (ss_cl / ss.where(ss > 0)).rename(columns=lambda c: c.replace("z_", "deff_"))

    return out.join(deff).reset_index(), df[key + ["doy"]]


def temporal_component(doys, by_doy, zbar):
    """Exact E[B] for a design that keeps its own doys but samples space from P."""
    key = ["vel_year", "rep"]
    per_doy = pd.DataFrame(by_doy.T[doys["doy"].to_numpy().astype(int)],
                           columns=[f"temporal_{y}" for y in YEARS],
                           index=doys.index)
    m = per_doy.groupby([doys[k] for k in key]).mean()
    return (m - zbar).reset_index()


def uniform_estimates(zf, wet, w, n_vec, doys=None, seed=0):
    """iid draws from P (or from P in space at fixed doys), n_vec points each."""
    rng = np.random.default_rng(seed)
    n_vec = np.asarray(n_vec, dtype=np.int64)
    total = int(n_vec.sum())
    cells = wet[np.searchsorted(np.cumsum(w), rng.random(total)).clip(max=len(w) - 1)]
    d = (rng.integers(0, cfg.DAYS_PER_YEAR, total) if doys is None
         else np.asarray(doys, dtype=int))
    vals = zf[:, d, cells]
    off = np.concatenate([[0], np.cumsum(n_vec)[:-1]])
    return (np.add.reduceat(vals, off, axis=1) / n_vec).T


def bias_grid(adv, unif, zbar):
    """Long cells-table summary of every (analysis year, velocity year) cell."""
    zb = pd.Series(zbar["zbar"].to_numpy(), index=YEARS)
    rows = []
    for k in YEARS:
        for l, ga in adv.groupby("vel_year"):
            gu = unif[unif["vel_year"] == l]
            b = ga[f"zhat_{k}"] - zb[k]
            rows.append({
                "analysis_year": k, "vel_year": l, "aligned": k == l,
                "n_designs": len(ga), "mean_n": ga["n"].mean(),
                "mean_B": b.mean(), "se_B": b.sem(), "sd_B": b.std(ddof=1),
                "mean_temporal": ga[f"temporal_{k}"].mean(),
                "mean_spatial": (b - ga[f"temporal_{k}"]).mean(),
                "deff": ga[f"deff_{k}"].mean(),
                "unif_mean_B": (gu[f"zhat_{k}"] - zb[k]).mean(),
                "unif_sd_B": (gu[f"zhat_{k}"] - zb[k]).std(ddof=1),
            })
    return pd.DataFrame(rows)


# --------------------------- domain and field ----------------------------

def _file_extent(pattern):
    """(lat0, lat1, lon0, lon1) of the first GLORYS file matching a glob."""
    path = sorted(glob.glob(os.path.join(cfg.GLORYS_DIR, pattern)))[0]
    with xr.open_dataset(path) as ds:
        return (float(ds["latitude"].min()), float(ds["latitude"].max()),
                float(ds["longitude"].min()), float(ds["longitude"].max()))


def domain_extents(boxes=("boxA", "boxB")):
    """Analysis boxes and the two downloaded regions, read from the files.

    Read rather than taken from the download config so the drawn extents are the
    data actually on disk.
    """
    out = {"velocity": _file_extent("velocity_daily/velocity_daily_park_*.nc"),
           "thetao": _file_extent("thetao_daily/thetao_daily_*.nc")}
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        out[b] = (la0, la1, lo0, lo1)
    return out


def daily_spread(box, path=None):
    """sd_P(Z) on each (year, doy) of the residual field; cached.

    The spatial dispersion under the same measure P the estimand integrates
    against, so it scores a day by how much spatial structure a design sampling
    that day could possibly be biased by.
    """
    path = path or os.path.join(cfg.OUT_DIR, f"daily_spread_{box}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    zf, wet, w = load_field(box)
    years = _field_years(box)
    rows = []
    for i, y in enumerate(years):
        zw = zf[i][:, wet].astype(np.float64)
        m1 = zw @ w
        rows.append(pd.DataFrame({"year": y, "doy": np.arange(zw.shape[0]),
                                  "mean": m1,
                                  "sd": np.sqrt((zw * zw) @ w - m1**2)}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def quantile_days(box, qs=(0.05, 0.5, 0.95)):
    """The (year, doy) whose spatial sd is nearest each quantile of sd_P(Z)."""
    s = daily_spread(box)
    sd = s["sd"].to_numpy()
    out = []
    for q in qs:
        i = int(np.argmin(np.abs(sd - np.quantile(sd, q))))
        out.append({"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]),
                    "sd": sd[i]})
    return pd.DataFrame(out)


VEL_COMPS = {"uo": "eastward", "vo": "northward"}
M_S_TO_CM_S = 100.0


def _velocity_year(box, year):
    """Park-depth (u, v) over one year on the leap-stripped 365-day calendar.

    Interpolated to PARK_DEPTH between the two archived levels, which is what
    Parcels does to the same files when it advects the floats.
    """
    from osse import design_invariance as di
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    files = [os.path.join(cfg.GLORYS_DIR, "velocity_daily",
                          f"velocity_daily_park_{year}_{m:02d}.nc") for m in range(1, 13)]
    with xr.open_mfdataset(files, combine="by_coords") as ds:
        d = (ds[["uo", "vo"]]
             .sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
             .interp(depth=di.PARK_DEPTH).load())
    t = d["time"].dt
    return d.isel(time=~((t.month == 2) & (t.day == 29)).values)


def daily_velocity_spread(box, path=None):
    """Spatial mean and sd of each velocity component under P, per (year, doy).

    Taken under the same P and on the same mask as the residual field -- the
    velocity grid is cell-for-cell the analysis grid, and park-depth velocity is
    defined everywhere the OHC integral is -- so these days are directly
    comparable to the ones daily_spread scores.
    """
    path = path or os.path.join(cfg.OUT_DIR, f"daily_vel_spread_{box}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    wet, w = area_weights(box)
    rows = []
    for y in _field_years(box):
        d = _velocity_year(box, y)
        r = {"year": y, "doy": np.arange(d.sizes["time"])}
        for c in VEL_COMPS:
            x = d[c].values.reshape(d.sizes["time"], -1)[:, wet].astype(np.float64)
            m1 = x @ w
            r[f"mean_{c}"] = m1 * M_S_TO_CM_S
            r[f"sd_{c}"] = np.sqrt(np.clip((x * x) @ w - m1**2, 0, None)) * M_S_TO_CM_S
        rows.append(pd.DataFrame(r))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def velocity_quantile_days(box, comp="uo", qs=(0.05, 0.5, 0.95)):
    """The (year, doy) whose spatial sd of `comp` is nearest each quantile."""
    s = daily_velocity_spread(box)
    sd = s[f"sd_{comp}"].to_numpy()
    return pd.DataFrame([
        {"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]), "sd": sd[i]}
        for q in qs
        for i in [int(np.argmin(np.abs(sd - np.quantile(sd, q))))]])


def plot_velocity_quantile_days(comp="uo", boxes=("boxB", "boxA"), qs=(0.05, 0.5, 0.95),
                                figsize=(12.5, 8.2), clip=0.995):
    """Single-day park-depth velocity at quantiles of each box's daily spread.

    One row per box on its own scale, as in plot_quantile_days: the boxes differ
    several-fold in current strength and a shared scale would flatten Box A.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig, axes, caxes = _map_grid(len(boxes), len(qs), figsize)
    for r, box in enumerate(boxes):
        sel = velocity_quantile_days(box, comp, qs)
        days = []
        for y, g in sel.groupby("year", sort=False):
            d = _velocity_year(box, int(y))[comp]
            days += [(i, d.isel(time=int(dd)) * M_S_TO_CM_S)
                     for i, dd in zip(g.index, g["doy"])]
        days = [d for _, d in sorted(days, key=lambda p: p[0])]
        hi = max(float(np.nanquantile(np.abs(d.values), clip)) for d in days)
        norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
        for c, (ax, day, (_, s)) in enumerate(zip(axes[r], days, sel.iterrows())):
            im = _draw_map(ax, day, norm, "RdBu_r", _box_extent(box),
                           f"{box}  {int(s.q * 100)}th pctile day\n"
                           f"{int(s.year)} doy {int(s.doy) + 1}, "
                           f"sd {s.sd:.1f} cm/s", r == len(boxes) - 1, c == 0, scale=1.0)
        fig.colorbar(im, cax=caxes[r],
                     label=f"{VEL_COMPS[comp]} velocity at 1000 m (cm/s)")
    return fig


def mean_field_months(box, year, months=(1, 4, 7, 10)):
    """Fitted mean field evaluated at mid-month, as a (month, lat, lon) array.

    The basis is smooth in time -- day-to-day change is ~0.03 GJ/m^2 against a
    ~4 GJ/m^2 seasonal range -- so month is the finest unit at which it varies
    visibly.
    """
    from osse import build_residual_field as brf
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        shape = (ds.sizes["latitude"], ds.sizes["longitude"])
        beta = ds["beta"].values.reshape(ds.sizes["coef"], -1)
        t = np.array([np.datetime64(f"{year}-{m:02d}-15") for m in months],
                     dtype="datetime64[ns]")
        m = brf.evaluate_mean_field(beta, t, shape, quadratic=True,
                                    t_ref=ds.attrs["t_ref"])
        return xr.DataArray(m, dims=("month", "latitude", "longitude"),
                            coords={"month": list(months),
                                    "latitude": ds["latitude"], "longitude": ds["longitude"]})


def _map_grid(nrow, ncol, figsize, cbar=True, left=0.05, right=0.885, bottom=0.05,
              top=0.90, hgap=0.015, vgap=0.11):
    """GeoAxes grid at explicit positions, optionally a colorbar slot per row.

    Positions are set by hand rather than by a layout engine: cartopy gridliners
    raise inside tight_layout, and constrained layout resolves row-spanning
    colorbars inconsistently, which silently collapses the map axes.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    right = right if cbar else 0.985
    fig = plt.figure(figsize=figsize)
    w = (right - left - (ncol - 1) * hgap) / ncol
    h = (top - bottom - (nrow - 1) * vgap) / nrow
    axes, caxes = [], []
    for r in range(nrow):
        y = top - (r + 1) * h - r * vgap
        axes.append([fig.add_axes([left + c * (w + hgap), y, w, h], projection=pc)
                     for c in range(ncol)])
        if cbar:
            caxes.append(fig.add_axes([right + 0.012, y + 0.04 * h, 0.014, h * 0.92]))
    return fig, axes, caxes


def _doy_date(year, doy):
    """Calendar date of a day index on the 365-day leap-stripped calendar."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return days[~((days.month == 2) & (days.day == 29))][int(doy)]


def ohc_decomposition(box, year, doy):
    """OHC, fitted mean field, and residual on one day.

    OHC is reconstructed as mhat + Z rather than re-integrated from thetao: the
    residual is defined as OHC - mhat on the same mask, so the sum recovers the
    GLORYS 0-2000 m OHC exactly, with no re-read of the daily temperature.
    """
    from osse import build_residual_field as brf
    t = _doy_date(year, doy)
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc")) as ds:
        z = ds["z"].sel(year=int(year)).isel(doy=int(doy)).load()
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        m = brf.evaluate_mean_field(
            ds["beta"].values.reshape(ds.sizes["coef"], -1),
            np.array([t], dtype="datetime64[ns]"),
            (ds.sizes["latitude"], ds.sizes["longitude"]),
            quadratic=True, t_ref=ds.attrs["t_ref"])[0]
    mhat = xr.DataArray(m, coords=z.coords, dims=z.dims).where(np.isfinite(z))
    return (mhat + z).rename("ohc"), mhat.rename("mhat"), z, t


def plot_ohc_decomposition(boxes=("boxB", "boxA"), q=0.5, figsize=(9.5, 11.0),
                           clip=0.995):
    """OHC = mean field + residual on one representative day, per box.

    Rows are the three fields; the first two share one sequential scale so the
    residual's size relative to the OHC it came from is readable directly.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    parts, dates = {}, {}
    for b in boxes:
        d = quantile_days(b, (q,)).iloc[0]
        *fields, t = ohc_decomposition(b, int(d.year), int(d.doy))
        parts[b], dates[b] = fields, t

    flat = np.concatenate([np.ravel(p[i].values) for b in boxes for i in (0, 1)
                           for p in [parts[b]]])
    lo, hi = (float(np.nanquantile(flat, x)) * J_TO_GJ for x in (0.005, 0.995))
    res = max(float(np.nanquantile(np.abs(parts[b][2].values), clip)) for b in boxes) * J_TO_GJ
    norms = [Normalize(lo, hi), Normalize(lo, hi),
             TwoSlopeNorm(vmin=-res, vcenter=0.0, vmax=res)]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    labels = ["OHC 0-2000 m (GJ/m2)", "mean field (GJ/m2)", "residual Z (GJ/m2)"]
    names = ["OHC (observed)", "fitted mean field", "residual Z = OHC - mean"]

    fig, axes, caxes = _map_grid(3, len(boxes), figsize, top=0.93, vgap=0.075)
    for r in range(3):
        for c, b in enumerate(boxes):
            im = _draw_map(axes[r][c], parts[b][r], norms[r], cmaps[r], _box_extent(b),
                           f"{b}  {names[r]}\n{dates[b]:%Y-%m-%d}",
                           r == 2, c == 0)
        fig.colorbar(im, cax=caxes[r], label=labels[r])
    return fig


def _geo_ticks(ax, extent, bottom_labels, left_labels, step=5.0):
    """Lat/lon ticks via the ordinary matplotlib tick machinery.

    Deliberately NOT ax.gridlines(draw_labels=True): the cartopy gridliner makes
    GeoAxes.get_tightbbox raise on a NaN label position, matplotlib swallows
    that, and the figure's tight bbox then collapses onto whatever axes did
    succeed -- so a bbox_inches="tight" save (what the Jupyter inline backend
    does, hence Quarto) drops every map and keeps only the colorbars.
    """
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    lo0, lo1, la0, la1 = extent
    ax.set_xticks(np.arange(np.ceil(lo0 / step) * step, lo1 + 1e-9, step),
                  crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(la0 / step) * step, la1 + 1e-9, step),
                  crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=7, labelbottom=bottom_labels, labelleft=left_labels)
    ax.grid(lw=0.25, color="0.6", alpha=0.4)


def _draw_map(ax, da, norm, cmap, extent, title, bottom_labels, left_labels,
              scale=J_TO_GJ):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.4)
    # shading="nearest" (not "auto") and antialiased=False keep every native
    # 1/12 deg cell a hard-edged quad; "auto" happens to resolve to the same
    # thing here, but only because the coordinate vectors match the data shape.
    im = ax.pcolormesh(da["longitude"], da["latitude"], da.values * scale,
                       cmap=cmap, norm=norm, transform=pc, shading="nearest",
                       antialiased=False, linewidth=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    _geo_ticks(ax, extent, bottom_labels, left_labels)
    return im


def _box_extent(box, pad=0.5):
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    return [lo0 - pad, lo1 + pad, la0 - pad, la1 + pad]


def plot_quantile_days(boxes=("boxB", "boxA"), qs=(0.05, 0.5, 0.95),
                       figsize=(12.5, 8.2), clip=0.995):
    """Single-day residual fields at quantiles of each box's daily spatial spread.

    One row per box on its own colour scale: the boxes differ several-fold in
    amplitude, and a shared scale would flatten the within-box structure this
    figure exists to show.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig, axes, caxes = _map_grid(len(boxes), len(qs), figsize)
    for r, box in enumerate(boxes):
        sel = quantile_days(box, qs)
        with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{box}.nc")) as ds:
            days = [ds["z"].sel(year=int(d.year)).isel(doy=int(d.doy)).load()
                    for _, d in sel.iterrows()]
        hi = max(float(np.nanquantile(np.abs(d.values), clip)) for d in days) * J_TO_GJ
        norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
        for c, (ax, day, (_, d)) in enumerate(zip(axes[r], days, sel.iterrows())):
            im = _draw_map(ax, day, norm, "RdBu_r", _box_extent(box),
                           f"{box}  {int(d.q * 100)}th pctile day\n"
                           f"{int(d.year)} doy {int(d.doy) + 1}, "
                           f"sd {d.sd * J_TO_GJ:.2f} GJ/m2",
                           r == len(boxes) - 1, c == 0)
        fig.colorbar(im, cax=caxes[r], label="Z (GJ/m2)")
    return fig


def deployment_sample(box, year=2015, reps=3, spin_days=None, n_floats=None):
    """Seed positions of the first few steady-arm replicates.

    Regenerated from the same (year, rep)-keyed RNG the production run used, so
    these are the actual deployments, not a fresh draw resembling them.
    """
    from osse import design_invariance as di
    pool = di.load_deployment_pool(box=box)
    lat, lon, day, rep, _ = di.deployment_plan_steady(
        pool, box, n_floats=n_floats or di.N_FLOATS_STEADY, reps=reps, year=year,
        spin_days=spin_days or di.SPIN_DAYS)
    return pd.DataFrame({"lat": lat, "lon": lon, "release_day": day, "rep": rep})


def trajectory_sample(box, year=2015, rep=1, arm="steady", chunk=0):
    """One replicate's profile positions, ordered along each float's track."""
    path = os.path.join(cfg.OUT_DIR, f"design_profiles_{box}_{arm}_y{year}_c{chunk}.csv")
    df = pd.read_csv(path, usecols=["rep", "float_id", "cycle", "lat", "lon", "doy"])
    return df[df["rep"] == rep].sort_values(["float_id", "cycle"])


def argo_region_summary(box="boxB", years=None, path=None, cache=None):
    """Real Argo occupancy of the box, per analysis year; cached.

    Standing stock is the mean number of distinct floats reporting within a
    10-day window, which is what compares against the virtual array: real floats
    cycle on their own schedules, so a straight profile count would confound
    array size with cycling cadence.
    """
    years = years or _field_years(box)
    cache = cache or os.path.join(cfg.OUT_DIR, f"argo_region_{box}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    from osse import design_invariance as di

    d = pd.read_csv(path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv"),
                    usecols=["float_id", "lat", "lon", "date", "ohc_2000"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)
          & d["ohc_2000"].notna()]
    d["year"] = d["date"].dt.year
    d = d[d["year"].isin(years)]
    d["bin"] = (d["date"].dt.dayofyear - 1) // di.CYCLE_DAYS

    stock = (d.groupby(["year", "bin"])["float_id"].nunique()
              .groupby("year").mean())
    out = (d.groupby("year")
             .agg(argo_profiles=("float_id", "size"), argo_floats=("float_id", "nunique"))
             .join(stock.rename("argo_stock")).reset_index())
    out.to_csv(cache, index=False)
    return out


def truth_table(boxes=("boxB", "boxA")):
    """Domain mean and dispersion of the residual field, both boxes side by side."""
    out = None
    for b in boxes:
        t = pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{b}.csv"))
        t = pd.DataFrame({"year": t["year"],
                          f"Zbar {b}": t["zbar"] * J_TO_GJ,
                          f"sd_P(Z) {b}": t["sd_Z"] * J_TO_GJ,
                          f"Zbar/sd {b}": t["zbar"] / t["sd_Z"]})
        out = t if out is None else out.merge(t, on="year")
    return out


def nsize_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Retained sample size per advective design, both boxes."""
    rows = []
    for b in boxes:
        adv = pd.read_csv(os.path.join(cfg.OUT_DIR, f"mean_mc_advective_{b}{arm}.csv"))
        for lab, s in (("profiles", adv["n"]), ("distinct floats", adv["n_floats"])):
            rows.append({"box": b, "quantity": f"{lab} per design", "min": s.min(),
                         "5%": s.quantile(.05), "median": s.median(),
                         "95%": s.quantile(.95), "max": s.max()})
    return pd.DataFrame(rows)


def argo_density_table(boxes=("boxB", "boxA")):
    """Simulated against real standing stock, both boxes, by analysis year.

    Only the stock and its ratio are carried: profile counts confound array size
    with cycling cadence, and the ratio is what says whether the two boxes are
    over-dense by the same factor and so remain comparable to each other.
    """
    out = None
    for b in boxes:
        t = argo_comparison_table(box=b)
        t = t[t["year"] != "mean"][["year", "Argo stock", "sim stock", "stock ratio"]]
        t.columns = ["year", f"Argo {b}", f"sim {b}", f"ratio {b}"]
        out = t if out is None else out.merge(t, on="year")
    out.loc["mean"] = ["mean", *out.iloc[:, 1:].mean()]
    return out


def argo_comparison_table(box="boxB", arm="_steady", years=None):
    """Virtual array against the real Argo record, same box and years.

    Virtual standing stock is in-box profiles per 10-day epoch: every virtual
    float profiles on every epoch, so n / n_epochs is the number in the box.
    """
    from osse import design_invariance as di
    real = argo_region_summary(box, years).set_index("year")
    adv = pd.read_csv(os.path.join(cfg.OUT_DIR, f"mean_mc_advective_{box}{arm}.csv"))
    n_ep = cfg.DAYS_PER_YEAR // di.CYCLE_DAYS
    sim = adv.groupby("vel_year").agg(sim_profiles=("n", "mean"),
                                      sim_floats=("n_floats", "mean"))
    sim["sim_stock"] = sim["sim_profiles"] / n_ep

    t = real.join(sim).reset_index().rename(columns={"index": "year"})
    t["stock ratio"] = t["sim_stock"] / t["argo_stock"]
    # Per-year means, not sums: distinct float counts would double-count floats
    # present in several years, and the simulated columns are already per-design
    # means over an independent 750-design ensemble in each year.
    t.loc["mean"] = ["mean", *t.iloc[:, 1:].mean()]
    return t.rename(columns={
        "year": "year", "argo_profiles": "Argo profiles", "argo_floats": "Argo floats",
        "argo_stock": "Argo stock", "sim_profiles": "sim profiles",
        "sim_floats": "sim floats", "sim_stock": "sim stock"})


def _doy_months(year):
    """Calendar month of each index on the 365-day leap-stripped calendar."""
    d = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return d[~((d.month == 2) & (d.day == 29))].month.to_numpy()


def stock_table(box="boxB", year=2015, tag="_spinup", reps=6, spin_days=None,
                n_floats=None, run_days=360, path=None):
    """Per-replicate float-stock budget by calendar month, spin-up included.

    Counting lives in design_invariance.monthly_stock, which production now runs
    before it discards the trajectory record; this only regenerates the lifespan
    vector needed to attribute departures and formats the result for display.
    """
    from osse import design_invariance as di

    pool = di.load_deployment_pool(box=box)
    *_, life = di.deployment_plan_steady(
        pool, box, n_floats=n_floats or di.N_FLOATS_STEADY, reps=reps, year=year,
        spin_days=spin_days or di.SPIN_DAYS, run_days=run_days)
    raw = di.monthly_stock(
        path or os.path.join(cfg.OUT_DIR, "_traj", f"traj_{box}_steady{tag}_{year}.zarr"),
        box, year, life, reps, run_days, spin_days or di.SPIN_DAYS)

    per = ["deployed", "in_domain", "retired", "advected_out", "prof_seed", "prof_box"]
    tab = raw.set_index("ym")[per] / reps
    tab["epochs"] = raw.set_index("ym")["epochs"]
    # Per-epoch, not per-month: a float profiles once an epoch, so these are
    # standing stocks, and the in-box one is what tbl-argo's "sim stock"
    # compares against.
    for src in ("seed", "box"):
        tab[f"stock_{src}"] = tab[f"prof_{src}"] / tab["epochs"]
    tab.index = tab.index.astype(str)
    return tab.rename(columns={
        "deployed": "new floats deployed", "in_domain": "floats in domain",
        "retired": "floats retired", "advected_out": "floats advected out",
        "epochs": "10-day epochs", "prof_seed": "profiles in seeding region",
        "stock_seed": "floats in seeding region",
        "prof_box": "profiles in analysis region",
        "stock_box": "floats in analysis region"})[
        ["new floats deployed", "floats in domain", "floats retired",
         "floats advected out", "10-day epochs", "profiles in seeding region",
         "floats in seeding region", "profiles in analysis region",
         "floats in analysis region"]]


def plot_deployment_advection(box="boxB", year=2015, reps=3, traj_rep=1,
                              figsize=(13.5, 6.2), max_tracks=60,
                              n_floats=None, spin_days=None):
    """Where floats are seeded, and where the flow then takes them.

    Left: seed positions of a few replicates over the clipped seeding margin.
    Right: one replicate's tracks through the analysis year, with a marker at
    every profile. Only the box interior contributes to Zhat.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from osse import design_invariance as di

    seeds = deployment_sample(box, year, reps, spin_days, n_floats)
    trk = trajectory_sample(box, year, traj_rep)
    (sla0, sla1), (slo0, slo1) = di.seed_bounds(box, di.SEED_MARGIN_UNIFORM)
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    ext = [min(slo0, trk.lon.min()) - 1, max(slo1, trk.lon.max()) + 1,
           min(sla0, trk.lat.min()) - 1, max(sla1, trk.lat.max()) + 1]

    fig, axes, _ = _map_grid(1, 2, figsize, cbar=False, top=0.90, bottom=0.08)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree()
    for j, ax in enumerate(axes[0]):
        ax.set_extent(ext, crs=pc)
        ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.4)
        ax.add_patch(Rectangle((slo0, sla0), slo1 - slo0, sla1 - sla0, transform=pc,
                               fill=False, edgecolor="tab:blue", lw=1.4, ls="--",
                               zorder=3, label="seeding margin"))
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                               fill=False, edgecolor="crimson", lw=2.0, zorder=4,
                               label=f"{box} (analysis domain)"))
        _geo_ticks(ax, ext, True, j == 0, step=10.0)

    for r, g in seeds.groupby("rep"):
        axes[0][0].plot(g["lon"], g["lat"], ".", ms=3.0, alpha=0.7, transform=pc,
                        label=f"replicate {int(r)}")
    axes[0][0].set_title(f"deployment: {reps} replicates x {len(seeds)//reps} floats",
                         fontsize=10)
    axes[0][0].legend(loc="upper left", fontsize=7, framealpha=0.9, markerscale=2.5)

    ids = trk["float_id"].unique()[:max_tracks]
    cols = plt.cm.turbo(np.linspace(0.05, 0.95, len(ids)))
    for k, fid in enumerate(ids):
        g = trk[trk["float_id"] == fid]
        axes[0][1].plot(g["lon"], g["lat"], "-", lw=0.7, color=cols[k], alpha=0.75,
                        transform=pc)
        axes[0][1].plot(g["lon"], g["lat"], ".", ms=1.6, color=cols[k], alpha=0.9,
                        transform=pc)
    axes[0][1].set_title(f"advection, {year}: {len(ids)} of {trk['float_id'].nunique()} "
                         f"tracks, dots are profiles", fontsize=10)
    return fig


def plot_mean_field_months(boxes=("boxB", "boxA"), year=2015, months=(1, 4, 7, 10),
                           figsize=(14.5, 8.2)):
    """Fitted mean field at mid-month, one row per box on its own scale."""
    from matplotlib.colors import Normalize
    names = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}

    fig, axes, caxes = _map_grid(len(boxes), len(months), figsize)
    for r, box in enumerate(boxes):
        mf = mean_field_months(box, year, months)
        lo, hi = (float(np.nanquantile(mf.values, q)) * J_TO_GJ for q in (0.01, 0.99))
        norm = Normalize(vmin=lo, vmax=hi)
        for c, ax in enumerate(axes[r]):
            im = _draw_map(ax, mf.isel(month=c), norm, "viridis", _box_extent(box),
                           f"{box}  {names.get(months[c], months[c])} {year}",
                           r == len(boxes) - 1, c == 0)
        fig.colorbar(im, cax=caxes[r], label="mean field (GJ/m2)")
    return fig


def plot_domain_map(boxes=("boxA", "boxB"), figsize=(7.5, 6.4)):
    """Analysis boxes inside the velocity and thetao download regions."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    e = domain_extents(boxes)
    pc = ccrs.PlateCarree()
    # tight_layout collapses the GeoAxes boundary that the gridliner draws
    # against, so layout is left to constrained mode here and below.
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": pc},
                           layout="constrained")
    v = e["velocity"]
    ax.set_extent([v[2] - 1.5, v[3] + 1.5, v[0] - 1.5, v[1] + 1.5], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.5)

    style = {"velocity": ("tab:blue", "velocity (advection domain)"),
             "thetao": ("tab:green", "thetao (OHC truth + mean field)"),
             "boxA": ("tab:orange", "Box A (subtropical gyre)"),
             "boxB": ("crimson", "Box B (NAC / Gulf Stream)")}
    for name in ("velocity", "thetao") + tuple(boxes):
        la0, la1, lo0, lo1 = e[name]
        color, label = style[name]
        fill = name in boxes
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                               fill=fill, facecolor=color if fill else "none",
                               alpha=0.18 if fill else 1.0, edgecolor=color,
                               lw=2.0, ls="-" if fill else "--", zorder=2, label=label))
    _geo_ticks(ax, [v[2] - 1.5, v[3] + 1.5, v[0] - 1.5, v[1] + 1.5], True, True, step=10.0)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return fig


def bathymetry(tag="static_vel"):
    """GLORYS deptho over the velocity domain, on the native advection grid."""
    with xr.open_dataset(os.path.join(cfg.GLORYS_DIR, "static", f"{tag}_bathy.nc")) as ds:
        return ds["deptho"].load()


def depth_mask_summary(boxes=("boxB", "boxA"), threshold=2000.0):
    """Box area retained by the depth cut alone and by the full analysis mask.

    The two differ: a cell can be deeper than the threshold and still be dropped
    because the OHC integral is undefined at some level within it.
    """
    st = xr.open_dataset(os.path.join(cfg.GLORYS_DIR, "static", "static_coords.nc"))
    dep = bathymetry()
    rows = []
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        sl = dict(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
        A = np.squeeze((st["e1t"].sel(**sl) * st["e2t"].sel(**sl)).values)
        d = dep.sel(**sl).values
        with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{b}.nc")) as ds:
            am = ds["cell_area"].values
        tot = np.nansum(A)
        rows.append({"box": b, "cells": d.size,
                     "% area deeper than cut": 100 * np.nansum(np.where(d > threshold, A, 0)) / tot,
                     "% area in analysis mask": 100 * np.nansum(am) / tot,
                     "median depth (m)": float(np.nanmedian(d))})
    return pd.DataFrame(rows)


def plot_bathymetry(boxes=("boxA", "boxB"), threshold=2000.0, figsize=(13.5, 6.2),
                    cap=6000.0):
    """Ocean depth over the advection domain, and what the depth cut removes.

    Left: deptho, capped for display so the shelf break is resolved rather than
    compressed against the abyssal plain. Right: the cells the cut excludes.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.patches import Rectangle, Patch

    dep = bathymetry()
    e = domain_extents(boxes)
    v = e["velocity"]
    ext = [v[2], v[3], v[0], v[1]]
    pc = ccrs.PlateCarree()

    fig, axes, caxes = _map_grid(1, 2, figsize, top=0.92, bottom=0.08, right=0.90)
    shallow = dep.where(dep <= threshold)
    for j, ax in enumerate(axes[0]):
        ax.set_extent(ext, crs=pc)
        ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=3)
        ax.add_feature(cfeature.COASTLINE, lw=0.4, zorder=4)
        if j == 0:
            im = ax.pcolormesh(dep["longitude"], dep["latitude"],
                               np.minimum(dep.values, cap) * 1e-3, cmap="Blues",
                               norm=Normalize(0.0, cap * 1e-3), transform=pc,
                               shading="nearest", antialiased=False, linewidth=0,
                               rasterized=True)
            ax.set_title(f"GLORYS deptho (capped at {cap * 1e-3:.0f} km)", fontsize=9)
        else:
            # NaN, not 0, off the painted set: a one-colour ListedColormap maps
            # every finite value to its single colour, so a 0/1 indicator paints
            # the whole panel. Only NaN is left unpainted.
            for m, col in ((np.isfinite(dep.values), "0.93"),
                           (np.isfinite(shallow.values), "tab:red")):
                ax.pcolormesh(dep["longitude"], dep["latitude"],
                              np.where(m, 1.0, np.nan), cmap=ListedColormap([col]),
                              transform=pc, shading="nearest", antialiased=False,
                              linewidth=0, rasterized=True, zorder=2)
            ax.set_title(f"ocean shallower than {threshold:.0f} m (excluded)", fontsize=9)
            ax.legend(handles=[Patch(facecolor="tab:red", label=f"< {threshold:.0f} m"),
                               Patch(facecolor="0.93", label="analysis-eligible")],
                      loc="lower left", fontsize=7, framealpha=0.9)
        for b in boxes:
            la0, la1, lo0, lo1 = e[b]
            ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                                   fill=False, edgecolor="k", lw=1.8, zorder=5))
            ax.text(lo0 + 0.4, la1 - 1.4, b, transform=pc, fontsize=8, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
        _geo_ticks(ax, ext, True, j == 0, step=10.0)
    fig.colorbar(im, cax=caxes[0], label="depth (km)")
    return fig


# ------------------------------- plotting --------------------------------

def _hist(ax, x, **kw):
    kw.setdefault("bins", 24)
    kw.setdefault("color", "0.35")
    ax.hist(np.asarray(x) * J_TO_GJ, density=True, **kw)


def bias_frame(est, zbar):
    """Per-design bias B = Zhat - Zbar_k, one column per analysis year."""
    zb = pd.Series(zbar["zbar"].to_numpy(), index=YEARS)
    return pd.DataFrame({k: est[f"zhat_{k}"].to_numpy() - zb[k] for k in YEARS},
                        index=est.index)


def load_arm(box="boxB", arm="_steady"):
    """Every cached ensemble for one box, with YEARS rebound to its field."""
    set_box(box)
    r = [pd.read_csv(os.path.join(cfg.OUT_DIR, f"mean_mc_{k}_{box}{arm}.csv"))
         for k in ("advective", "uniform", "grid")]
    return (*r, pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{box}.csv")))


def ref_table(unif, adv, zbar):
    """Uniform ensemble against E[B] = 0 and sd(B) = sigma_k sqrt(mean 1/n)."""
    nv = adv["n"].to_numpy()
    B = bias_frame(unif, zbar)
    se = B.std(ddof=1) / np.sqrt(len(B))
    return pd.DataFrame({
        "year": YEARS,
        "mean B (GJ/m2)": B.mean().to_numpy() * J_TO_GJ,
        "MC se": se.to_numpy() * J_TO_GJ,
        "mean B / se": (B.mean() / se).to_numpy(),
        "sd of B": B.std(ddof=1).to_numpy() * J_TO_GJ,
        "iid prediction": zbar["sd_Z"].to_numpy() * np.sqrt(np.mean(1.0 / nv)) * J_TO_GJ,
    })


def diagonal_table(adv, zbar):
    """Aligned-cell bias distribution: flow and field the same calendar year.

    A direct Monte-Carlo summary of the realized designs, so no comparison
    across flow years and no invariance assumption enters. The MC se describes
    the mean; the deployment sd and the quantiles describe the distribution.
    """
    B = bias_frame(adv, zbar)
    sd = zbar.set_index("year")["sd_Z"]
    v = adv["vel_year"].to_numpy()
    rows = []
    for k in YEARS:
        b = B.loc[v == k, k].to_numpy()
        rows.append({"year": k, "designs": b.size,
                     "mean B": b.mean() * J_TO_GJ,
                     "MC se": b.std(ddof=1) / np.sqrt(b.size) * J_TO_GJ,
                     "sd over deployments": b.std(ddof=1) * J_TO_GJ,
                     "5%": np.percentile(b, 5) * J_TO_GJ,
                     "95%": np.percentile(b, 95) * J_TO_GJ,
                     "P(B < 0)": float((b < 0).mean()),
                     "mean B / sd_k": b.mean() / sd[k]})
    return pd.DataFrame(rows)


def aligned_table(grid):
    """Aligned cell against the nine unaligned flows, per analysis year.

    "Implied between-flow sd" nets each unaligned entry's own Monte-Carlo error
    out of the observed spread across flows; it is the scale the diagonal
    contrast should be read against, not the MC se.
    """
    a = grid[grid["aligned"]].sort_values("analysis_year")
    o = grid[~grid["aligned"]].groupby("analysis_year")
    sd_un, se_un = o["mean_B"].std(ddof=1).to_numpy(), o["se_B"].mean().to_numpy()
    col = grid.groupby("analysis_year")["mean_B"].mean().to_numpy()
    return pd.DataFrame({
        "year": a["analysis_year"].to_numpy(),
        "aligned mean B": a["mean_B"].to_numpy() * J_TO_GJ,
        "aligned MC se": a["se_B"].to_numpy() * J_TO_GJ,
        "unaligned mean": o["mean_B"].mean().to_numpy() * J_TO_GJ,
        "unaligned sd (9 flows)": sd_un * J_TO_GJ,
        "mean MC se of those 9": se_un * J_TO_GJ,
        "implied between-flow sd": np.sqrt(np.clip(sd_un**2 - se_un**2, 0, None)) * J_TO_GJ,
        "aligned - column mean": (a["mean_B"].to_numpy() - col) * J_TO_GJ,
    })


def deff_table(adv, unif):
    """Float-clustered design effect, and the advective/uniform width ratio."""
    d = adv[[f"deff_{y}" for y in YEARS]].median()
    ga = [adv.groupby("vel_year")[f"zhat_{k}"].std(ddof=1).mean() for k in YEARS]
    gu = [unif.groupby("vel_year")[f"zhat_{k}"].std(ddof=1).mean() for k in YEARS]
    return pd.DataFrame({
        "year": YEARS,
        "median design effect": [d[f"deff_{k}"] for k in YEARS],
        "sd B advective": np.array(ga) * J_TO_GJ,
        "sd B uniform": np.array(gu) * J_TO_GJ,
        "ratio": np.array(ga) / np.array(gu),
    })


def decomp_table(grid):
    """Temporal/spatial split of the aligned cells."""
    a = grid[grid["aligned"]].sort_values("analysis_year")
    return pd.DataFrame({
        "year": a["analysis_year"].to_numpy(),
        "total B": a["mean_B"].to_numpy() * J_TO_GJ,
        "temporal": a["mean_temporal"].to_numpy() * J_TO_GJ,
        "spatial | time": a["mean_spatial"].to_numpy() * J_TO_GJ,
        "MC se": a["se_B"].to_numpy() * J_TO_GJ,
    })


def variance_audit(boxes=("boxB", "boxA"), arm="_steady", n_perm=4000, D=750):
    """Every variance identity the report relies on, observed against predicted.

    Each row is a ratio that should be 1. Kept as a rendered check rather than a
    one-off script so the identities are re-verified whenever the ensembles move.
    """
    rows = []
    for b in boxes:
        adv, unif, grid, zbar = load_arm(b, arm)
        Y = list(YEARS)
        Ba, Bu = bias_frame(adv, zbar), bias_frame(unif, zbar)
        n, sig = adv["n"].to_numpy(), zbar["sd_Z"].to_numpy()

        zf, wet, w = load_field(b)
        dm, by_doy = domain_means(zf, wet, w)
        sig_P = dm["sd_Z"].to_numpy()          # under P; `sig` is the cached unweighted one
        sv = np.empty_like(by_doy)
        for i in range(zf.shape[0]):
            zw = zf[i][:, wet].astype(np.float64)
            sv[i] = (zw * zw) @ w - (zw @ w) ** 2
        idy = (sv.mean(axis=1) + by_doy.var(axis=1)) / sig_P**2

        sa = np.array([adv.groupby("vel_year")[f"zhat_{k}"].std(ddof=1).mean() for k in Y])
        su = np.array([unif.groupby("vel_year")[f"zhat_{k}"].std(ddof=1).mean() for k in Y])
        dz = adv[[f"deff_{k}" for k in Y]].median().to_numpy()

        an = design_invariance_test(b, arm)
        c = an["flows in C_k"].to_numpy()
        tab, _ = permutation_test(b, arm, n_perm=n_perm)
        o, _ = null_sensitivity(b, arm, n_perm=n_perm)
        va, ve = (an["between-flow sd"].to_numpy() / J_TO_GJ) ** 2, \
                 (an["within-flow sd"].to_numpy() / J_TO_GJ) ** 2

        for name, r in (
            ("sigma_k^2 = E_d[spatial var] + Var_d[Zbar(d)]", idy),
            ("reported sigma_k / sigma_k under P", sig / sig_P),
            ("sd(B) uniform = sigma_k sqrt(E[1/n])",
             Bu.std(ddof=1).to_numpy() / (sig_P * np.sqrt(np.mean(1 / n)))),
            ("sd(B) advective / uniform = sqrt(deff)", sa / su / np.sqrt(dz)),
            ("perm sd of T_k = sqrt((1+1/|C_k|) S^2/D)",
             tab["perm sd"].to_numpy() / J_TO_GJ / np.sqrt((1 + 1 / c) * Ba.var(ddof=1).to_numpy() / D)),
            ("null inflation = sqrt((D va + ve)/(va + ve))",
             o["sd inflation"].to_numpy() / np.sqrt((D * va + ve) / (va + ve))),
        ):
            rows.append({"box": b, "identity": name, "min ratio": r.min(),
                         "median ratio": float(np.median(r)), "max ratio": r.max()})
    return pd.DataFrame(rows)


def density_sensitivity(boxes=("boxB", "boxA"), arms=(("_steady", "full"),
                                                      ("_thin", "Argo-density"))):
    """Array size, resolution and the aligned contrast, full against thinned."""
    rows = []
    for b in boxes:
        for arm, lab in arms:
            adv, _, grid, zbar = load_arm(b, arm)
            al = aligned_table(grid)
            c = al["aligned - column mean"].to_numpy()
            sd = zbar.set_index("year")["sd_Z"].reindex(al["year"]).to_numpy() * J_TO_GJ
            se = c.std(ddof=1) / np.sqrt(len(c))
            rows.append({
                "box": b, "array": lab,
                "floats/design": round(adv["n_floats"].mean(), 1),
                "profiles/design": round(adv["n"].mean()),
                "median deff": float(adv[[f"deff_{y}" for y in YEARS]].median().median()),
                "MC se/cell": grid["se_B"].mean() * J_TO_GJ,
                "between-flow sd": design_invariance_test(b, arm)["between-flow sd"].mean(),
                "contrast": c.mean(),
                "contrast/sd_k": (c / sd).mean(),
                "negative years": int((c < 0).sum()),
                "contrast/se": c.mean() / se,
            })
    return pd.DataFrame(rows)


def contrast_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Aligned-minus-column contrast per box, raw and standardized by sigma_k.

    The boxes differ several-fold in field amplitude, so only the sigma_k-scaled
    column compares across them; the raw one is kept because it is what every
    per-box table above reports.

    The "se" column changes meaning on the pooled row -- per-year entries are
    Monte-Carlo (deployment) error with the flow held fixed, the pooled entry is
    the across-year standard error of the mean, which is the wider and the
    relevant scale for the pooled statement. Stated in the caption.
    """
    out = None
    for b in boxes:
        _, _, grid, zbar = load_arm(b, arm)
        a = grid[grid["aligned"]].sort_values("analysis_year")
        col = grid.groupby("analysis_year")["mean_B"].mean().to_numpy()
        sd = zbar.set_index("year")["sd_Z"].reindex(a["analysis_year"]).to_numpy()
        c = a["mean_B"].to_numpy() - col
        t = pd.DataFrame({"year": a["analysis_year"].to_numpy(),
                          f"contrast {b}": c * J_TO_GJ,
                          f"se {b}": a["se_B"].to_numpy() * J_TO_GJ,
                          f"contrast/sd {b}": c / sd})
        out = t if out is None else out.merge(t, on="year")
    n = len(out)
    out.loc["pooled"] = ["pooled", *(out.iloc[:, 1:].mean())]
    for b in boxes:
        out.loc["pooled", f"se {b}"] = (
            out[f"contrast {b}"].iloc[:n].std(ddof=1) / np.sqrt(n))
    return out


def contrast_decomp_table(boxes=("boxB", "boxA"), arm="_steady"):
    """The contrast split the same way the bias itself is, both boxes.

    The temporal term is near-constant down a column, so it cancels in the
    contrast; its column sd is carried so that is a measurement, not a claim.
    """
    out = None
    for b in boxes:
        _, _, grid, _ = load_arm(b, arm)
        a = grid[grid["aligned"]].sort_values("analysis_year")
        t = pd.DataFrame({"year": a["analysis_year"].to_numpy()})
        for c, lab in (("mean_B", "contrast"), ("mean_temporal", "temporal"),
                       ("mean_spatial", "spatial")):
            col = grid.groupby("analysis_year")[c].mean().to_numpy()
            t[f"{lab} {b}"] = (a[c].to_numpy() - col) * J_TO_GJ
        t[f"col sd temporal {b}"] = (grid.groupby("analysis_year")["mean_temporal"]
                                     .std(ddof=1).to_numpy()) * J_TO_GJ
        out = t if out is None else out.merge(t, on="year")
    out.loc["pooled"] = ["pooled", *out.iloc[:, 1:].mean()]
    return out


def alignment_estimates(box="boxB", arm="_steady", part=None, gap=1):
    """Theta_k (alignment.tex eq:align-estimand), the cell-mean matrix, and Cov({Theta_k}).

    gap=1 is the estimand itself, whose reference is every unaligned flow; gap=2
    restricts the reference to |l - k| >= 2 and is the memory sensitivity only.
    `part` selects the channel contrasted: None the bias, "temporal" its
    closed-form doy-marginal component, "spatial" the remainder. The covariance
    is exact under independence of the deployment draws across (l, d), which the
    OSSE satisfies -- each velocity year is seeded from its own draw.
    """
    adv, _, _, zbar = load_arm(box, arm)
    Y = list(YEARS)
    B = bias_frame(adv, zbar).to_numpy()
    T = adv[[f"temporal_{k}" for k in Y]].to_numpy()
    Z = {None: B, "temporal": T, "spatial": B - T}[part]
    v, C = adv["vel_year"].to_numpy(), _contrast_matrix(Y, gap)
    G = np.array([[Z[v == l, k].mean() for l in Y] for k in range(len(Y))])
    V = sum(np.cov(Z[v == l] * C[:, i], rowvar=False, ddof=1) / (v == l).sum()
            for i, l in enumerate(Y))
    return (C * G).sum(1), G, V, zbar


def alignment_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Theta_k with the eq:var-general standard error, raw and standardized by sigma_k."""
    out = None
    for b in boxes:
        th, _, V, zbar = alignment_estimates(b, arm)
        se = np.sqrt(np.diag(V))
        sd = zbar.set_index("year")["sd_Z"].reindex(YEARS).to_numpy()
        t = pd.DataFrame({"year": list(YEARS), f"Theta {b}": th * J_TO_GJ,
                          f"se {b}": se * J_TO_GJ, f"t {b}": th / se,
                          f"Theta/sd_k {b}": th / sd})
        out = t if out is None else out.merge(t, on="year")
    return out


def alignment_pooled(boxes=("boxB", "boxA"), arm="_steady"):
    """Pooled Theta with its exact Monte-Carlo se, plus the year-to-year spread.

    "between-year sd" nets the Monte-Carlo variance out of the observed spread of
    Theta_k across years, so it is structure in the realized record rather than
    error. The drift is GLS on the same covariance and describes these ten years;
    it is not an inference to a wider population of years.
    """
    rows = []
    for b in boxes:
        th, _, V, _ = alignment_estimates(b, arm)
        M = len(th)
        se2 = np.diag(V)
        x = np.asarray(YEARS, float) - np.mean(YEARS)
        Vi = np.linalg.inv(V)
        drift, dse = (x @ Vi @ th) / (x @ Vi @ x), 1.0 / np.sqrt(x @ Vi @ x)
        rows.append({"box": b, "pooled Theta": th.mean() * J_TO_GJ,
                     "MC se": np.sqrt(np.ones(M) @ V @ np.ones(M)) / M * J_TO_GJ,
                     "t": th.mean() / (np.sqrt(np.ones(M) @ V @ np.ones(M)) / M),
                     "across-year sd": th.std(ddof=1) * J_TO_GJ,
                     "MC part": np.sqrt(se2.mean()) * J_TO_GJ,
                     "between-year sd": np.sqrt(max(th.var(ddof=1) - se2.mean(), 0)) * J_TO_GJ,
                     "drift per yr": drift * J_TO_GJ, "drift se": dse * J_TO_GJ,
                     "drift t": drift / dse,
                     "W/m2 per decade": drift * 10.0 / 3.1536e8})
    return pd.DataFrame(rows)


def alignment_decomp_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Theta split the way the bias is, into doy-marginal and spatial-given-time."""
    out = None
    for b in boxes:
        t = pd.DataFrame({"year": list(YEARS)})
        for p, lab in ((None, "Theta"), ("temporal", "temporal"), ("spatial", "spatial")):
            t[f"{lab} {b}"] = alignment_estimates(b, arm, part=p)[0] * J_TO_GJ
        out = t if out is None else out.merge(t, on="year")
    out.loc["pooled"] = ["pooled", *out.iloc[:, 1:].mean()]
    return out


def alignment_variance_table(boxes=("boxB", "boxA"), arm="_steady"):
    """What the homoscedastic reduction eq:var-homo costs against the general eq:var-general.

    sigma^2_kl is the deployment variance of the single-design bias in cell (k,l);
    the reduction assumes it constant in l.
    """
    rows = []
    for b in boxes:
        adv, _, _, zbar = load_arm(b, arm)
        Y = list(YEARS)
        B, v = bias_frame(adv, zbar).to_numpy(), adv["vel_year"].to_numpy()
        S = np.array([[B[v == l, k].var(ddof=1) for l in Y] for k in range(len(Y))])
        D = np.array([(v == l).sum() for l in Y])
        M = len(Y)
        gen = np.sqrt([S[k, k] / D[k] + (np.delete(S[k], k) / np.delete(D, k)).sum()
                       / (M - 1) ** 2 for k in range(M)])
        hom = np.sqrt(np.diag(S) / D * (1 + 1 / (M - 1)))
        rows.append({"box": b,
                     "max/min sigma^2_kl over l": float(np.median(S.max(1) / S.min(1))),
                     "se general": gen.mean() * J_TO_GJ,
                     "se homosc": hom.mean() * J_TO_GJ,
                     "homosc/general": float(np.mean(hom / gen))})
    return pd.DataFrame(rows)


def alignment_contamination_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Whether ocean memory contaminates the reference, and what it costs Theta.

    Flows one year off share eddy structure with Z_k, so lag-1 cells sitting in
    the reference pull it toward the aligned cell and attenuate Theta. The first
    block measures the deviation those cells carry against lag >= 2; the second
    re-estimates Theta with the reference restricted to |l - k| >= 2, which is a
    different estimand and is reported only as the implied attenuation.
    """
    rows = []
    for b in boxes:
        g = lag_cells(b, arm)
        u = g[g["lag"] > 0]
        s = u.groupby(u["lag"] == 1)["dev"].agg(["size", "mean", "sem"])
        d = s.loc[True, "mean"] - s.loc[False, "mean"]
        dse = float(np.hypot(s.loc[True, "sem"], s.loc[False, "sem"]))
        th = [alignment_estimates(b, arm, gap=q)[0].mean() for q in (1, 2)]
        rows.append({"box": b, "lag 1 cells": int(s.loc[True, "size"]),
                     "lag 1 dev": s.loc[True, "mean"] * J_TO_GJ,
                     "lag >=2 dev": s.loc[False, "mean"] * J_TO_GJ,
                     "difference": d * J_TO_GJ, "se": dse * J_TO_GJ, "t": d / dse,
                     "pooled Theta": th[0] * J_TO_GJ,
                     "pooled gap 2": th[1] * J_TO_GJ,
                     "attenuation": th[0] / th[1]})
    return pd.DataFrame(rows)


def alignment_audit_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Every identity the unaligned-reference construction leans on. Each ratio should be 1."""
    rows = []
    for b in boxes:
        th, _, V, _ = alignment_estimates(b, arm)
        M = len(th)
        _, _, grid, _ = load_arm(b, arm)
        a = grid[grid["aligned"]].sort_values("analysis_year")
        gm = a["mean_B"].to_numpy() - grid.groupby("analysis_year")["mean_B"].mean().to_numpy()
        tt = alignment_estimates(b, arm, part="temporal")[0]
        ss = alignment_estimates(b, arm, part="spatial")[0]
        adv, _, _, zbar = load_arm(b, arm)
        B, v = bias_frame(adv, zbar).to_numpy(), adv["vel_year"].to_numpy()
        C = _contrast_matrix(list(YEARS), 1)
        ind = np.array([sum(C[k, i] ** 2 * B[v == l, k].var(ddof=1) / (v == l).sum()
                            for i, l in enumerate(YEARS)) for k in range(M)])
        U = [B[v == l] @ C[:, i] / M for i, l in enumerate(YEARS)]
        pool = np.array([sum(u.var(ddof=1) / u.shape[0] for u in U)])
        for name, r in (("grand-mean contrast = (M-1)/M Theta_k", gm / ((M - 1) / M * th)),
                        ("Theta_k = temporal + spatial", th / (tt + ss)),
                        ("Var eq:var-general = diag of the full covariance", ind / np.diag(V)),
                        ("pooled Var = per-draw unit variance", pool / (np.ones(M) @ V @ np.ones(M) / M**2))):
            rows.append({"box": b, "identity": name, "min ratio": r.min(),
                         "median ratio": float(np.median(r)), "max ratio": r.max()})
    return pd.DataFrame(rows)


def _contrast_matrix(years, gap=2):
    """Rows of theory.tex eq (Tk): +1 on the aligned flow, -1/|C_k| on C_k.

    C_k = {l : |l - k| >= gap} is the comparison set; gap=2 is the paper's, which
    drops the adjacent velocity years because D2 is not claimed for them.
    """
    y = np.asarray(years)
    A = np.zeros((len(y), len(y)))
    for i, k in enumerate(y):
        c = np.flatnonzero(np.abs(y - k) >= gap)
        if not len(c):
            raise ValueError(f"empty comparison set for {k} at gap={gap}")
        A[i, i], A[i, c] = 1.0, -1.0 / len(c)
    return A


def permutation_test(box="boxB", arm="_steady", n_perm=10000, gap=2, seed=7):
    """Velocity-label permutation test of H0: w^(k) independent of Z_k.

    Valid only under theory.tex D0/D1a/D1b/D2, which make the MD designs iid
    conditional on the fields and so the velocity-year labels exchangeable. The
    null spread it generates is deployment (Monte-Carlo) variability alone --
    which is exactly why it is anti-conservative when D1b fails.

    Returns (per-year table, pooled dict). Values in GJ/m^2.
    """
    adv, _, _, zbar = load_arm(box, arm)
    years = list(YEARS)
    B = bias_frame(adv, zbar).to_numpy()
    sizes = adv.groupby("vel_year").size()
    if sizes.nunique() != 1:
        raise ValueError(f"unbalanced design counts: {sizes.to_dict()}")
    M, D, m = len(years), int(sizes.iloc[0]), len(years)
    order = np.argsort(adv["vel_year"].to_numpy(), kind="stable")
    A = _contrast_matrix(years, gap)

    G = B[order].reshape(M, D, m).mean(axis=1)
    T_obs = (A * G.T).sum(axis=1)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(B))
    Tp = np.empty((n_perm, m))
    for r in range(n_perm):
        rng.shuffle(idx)
        Tp[r] = (A * B[idx].reshape(M, D, m).mean(axis=1).T).sum(axis=1)

    sd = Tp.std(axis=0, ddof=1)
    z_obs, zp = T_obs / sd, np.abs(Tp) / sd
    lam = zp.max(axis=1)
    tb_obs, tb = T_obs.mean(), Tp.mean(axis=1)

    tab = pd.DataFrame({
        "year": years,
        "T_k (GJ/m2)": T_obs * J_TO_GJ,
        "perm sd": sd * J_TO_GJ,
        "T_k / perm sd": z_obs,
        "p_k (FWER, 2-sided)": (1 + (lam[:, None] >= np.abs(z_obs)).sum(axis=0)) / (1 + n_perm),
    })
    pooled = {
        "Tbar (GJ/m2)": tb_obs * J_TO_GJ,
        "perm sd of Tbar": tb.std(ddof=1) * J_TO_GJ,
        "Tbar / perm sd": tb_obs / tb.std(ddof=1),
        "p two-sided": (1 + (np.abs(tb) >= abs(tb_obs)).sum()) / (1 + n_perm),
        "p lower-tail": (1 + (tb <= tb_obs).sum()) / (1 + n_perm),
        "p upper-tail": (1 + (tb >= tb_obs).sum()) / (1 + n_perm),
        "n_perm": n_perm, "gap": gap, "D": D, "M": M,
    }
    return tab, pooled


def design_invariance_test(box="boxB", arm="_steady", gap=2):
    """One-way test of D1a/D1b on the comparison-set flows only.

    Under D1a/D1b (plus D2, which is what licenses using C_k) the designs in
    C_k are iid for a fixed analysis year, so their per-velocity-year means of
    B_k differ only by deployment error. The aligned flow is excluded, so this
    does not presuppose H0 and is not circular with permutation_test.
    """
    from scipy import stats

    adv, _, _, zbar = load_arm(box, arm)
    years = np.asarray(YEARS)
    B = bias_frame(adv, zbar)
    lab = adv["vel_year"].to_numpy()
    rows = []
    for i, k in enumerate(years):
        c = years[np.abs(years - k) >= gap]
        b, g = B[k].to_numpy(), lab
        keep = np.isin(g, c)
        b, g = b[keep], g[keep]
        grp = [b[g == l] for l in c]
        n = np.array([len(x) for x in grp])
        mu = np.array([x.mean() for x in grp])
        ms_b = (n * (mu - b.mean()) ** 2).sum() / (len(c) - 1)
        ms_w = sum(((x - x.mean()) ** 2).sum() for x in grp) / (len(b) - len(c))
        f = ms_b / ms_w
        rows.append({
            "year": int(k), "flows in C_k": len(c),
            "F": f, "df1": len(c) - 1, "df2": len(b) - len(c),
            "p": float(stats.f.sf(f, len(c) - 1, len(b) - len(c))),
            "within-flow sd": np.sqrt(ms_w) * J_TO_GJ,
            "between-flow sd": np.sqrt(max(0.0, (ms_b - ms_w) / n.mean())) * J_TO_GJ,
        })
    t = pd.DataFrame(rows)
    t["sd ratio"] = t["between-flow sd"] / (t["within-flow sd"] / np.sqrt(750.0))
    return t


def null_sensitivity(box="boxB", arm="_steady", gap=2, n_perm=10000, seed=7):
    """What the permutation p-values become once the flow-year variance is put back.

    The permutation null resamples designs, so its spread is sigma_MC^2/D only.
    If instead each velocity year carries its own effect with variance
    sigma_flow^2 -- which is what D1b denies and design_invariance_test measures
    -- then Var(T_k) = (sigma_flow^2 + sigma_MC^2/D)(1 + 1/|C_k|). This is a
    normal-approximation sensitivity, not a valid test: with one flow
    realization per year there is no exchangeable unit left to resample.
    """
    from scipy import stats

    tab, pooled = permutation_test(box, arm, n_perm, gap, seed)
    d = design_invariance_test(box, arm, gap)
    nc = d["flows in C_k"].to_numpy()
    var = (d["between-flow sd"] ** 2 + (d["within-flow sd"] ** 2) / 750.0) * (1 + 1 / nc)
    out = tab.copy()
    out["flow-inflated sd"] = np.sqrt(var)
    out["sd inflation"] = out["flow-inflated sd"] / out["perm sd"]
    out["T_k / inflated sd"] = out["T_k (GJ/m2)"] / out["flow-inflated sd"]
    out["p_k (normal, Bonferroni)"] = np.minimum(
        1.0, len(out) * 2 * stats.norm.sf(np.abs(out["T_k / inflated sd"])))
    t = tab["T_k (GJ/m2)"].to_numpy()
    n_neg = int((t < 0).sum())
    pooled = {**pooled,
              "sign test n_neg": n_neg, "sign test m": len(t),
              "sign test p (2-sided)": float(
                  2 * min(stats.binom.cdf(min(n_neg, len(t) - n_neg), len(t), 0.5), 0.5))}
    return out, pooled


def lag_cells(box="boxB", arm="_steady"):
    """Grid cells carrying |l - k| and each cell's deviation from its own row mean."""
    _, _, grid, zbar = load_arm(box, arm)
    g = grid.copy()
    g["lag"] = (g["vel_year"] - g["analysis_year"]).abs()
    g["dev"] = g["mean_B"] - g.groupby("analysis_year")["mean_B"].transform("mean")
    g["dev_sd"] = g["dev"] / g["analysis_year"].map(zbar.set_index("year")["sd_Z"])
    return g


def lag_profile(box="boxB", arm="_steady"):
    """Cell mean minus its row mean, averaged by |l - k|.

    D2 asserts the comparison flows carry no dependence on Z_k. If the deviation
    at lag 1 and lag 2 is of the same sign as lag 0 rather than flat, the
    comparison set is contaminated and the contrast is attenuated, not clean.
    """
    g = lag_cells(box, arm)
    out = g.groupby("lag").agg(cells=("dev", "size"),
                               mean_dev=("dev", "mean"),
                               se=("dev", "sem"),
                               mean_dev_sd=("dev_sd", "mean")).reset_index()
    out["mean_dev"] *= J_TO_GJ
    out["se"] *= J_TO_GJ
    return out


def lag_table(boxes=("boxB", "boxA"), arm="_steady"):
    """lag_profile for several boxes side by side, keyed on the shared lag."""
    out = None
    for b in boxes:
        t = lag_profile(b, arm)[["lag", "cells", "mean_dev", "se", "mean_dev_sd"]].rename(
            columns={"mean_dev": f"mean dev {b}", "se": f"se {b}",
                     "mean_dev_sd": f"dev/sd {b}"})
        out = t if out is None else out.merge(t.drop(columns="cells"), on="lag")
    return out


def plot_reference_hists(unif, zbar, figsize=(13, 3.0)):
    """Uniform-sampling bias, one panel per analysis year. Unbiased => centred on 0."""
    import matplotlib.pyplot as plt
    B = bias_frame(unif, zbar)
    fig, axes = plt.subplots(1, len(YEARS), figsize=figsize, sharey=True, sharex=True)
    for ax, k in zip(axes, YEARS):
        _hist(ax, B[k])
        ax.axvline(0.0, color="crimson", lw=1.6)
        ax.axvline(B[k].mean() * J_TO_GJ, color="k", lw=1.4, ls=":")
        ax.set_title(str(k), fontsize=10)
        ax.set_xlabel("B (GJ/m2)")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_sample_sizes(adv, figsize=(11, 3.4)):
    """Profiles and distinct floats retained in the box, by velocity year."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)
    years = sorted(adv["vel_year"].unique())
    for ax, col, lab in ((a1, "n", "profiles per design"),
                         (a2, "n_floats", "distinct floats per design")):
        ax.boxplot([adv.loc[adv["vel_year"] == y, col] for y in years],
                   tick_labels=[str(y) for y in years], showfliers=False)
        ax.set_ylabel(lab)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.tight_layout()
    return fig


def plot_aligned_hists(adv, unif, zbar, figsize=(13, 3.0)):
    """Aligned-cell bias distribution, one panel per analysis year.

    The diagonal of plot_advective_grid on its own, on axis limits set by the
    aligned cells alone rather than by the whole grid.
    """
    import matplotlib.pyplot as plt
    Ba, Bu = bias_frame(adv, zbar), bias_frame(unif, zbar)
    va, vu = adv["vel_year"].to_numpy(), unif["vel_year"].to_numpy()
    lo, hi = np.percentile(np.concatenate(
        [Ba.loc[va == k, k].to_numpy() for k in YEARS]) * J_TO_GJ, [0.5, 99.5])
    pad = 0.08 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 30)

    fig, axes = plt.subplots(1, len(YEARS), figsize=figsize, sharex=True, sharey=True)
    for ax, k in zip(axes, YEARS):
        ax.hist(Ba.loc[va == k, k] * J_TO_GJ, bins=bins, density=True, color="0.35")
        ax.hist(Bu.loc[vu == k, k] * J_TO_GJ, bins=bins, density=True,
                histtype="step", color="tab:blue", lw=1.2, ls="--")
        ax.axvline(0.0, color="crimson", lw=1.6)
        ax.axvline(Ba.loc[va == k, k].mean() * J_TO_GJ, color="k", lw=1.4, ls=":")
        ax.set_title(str(k), fontsize=10)
        ax.set_xlabel("B (GJ/m2)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_advective_grid(adv, unif, zbar, figsize=(18, 9.5)):
    """Bias B by analysis year (rows) x velocity year (columns).

    Plotting B rather than Zhat puts every panel on one scale centred at zero,
    so rows are directly comparable; the truth line is 0 everywhere. Aligned
    cells (velocity year == analysis year) are outlined in red; the dashed curve
    is the n-matched uniform reference for the same cell.
    """
    import matplotlib.pyplot as plt
    Ba, Bu = bias_frame(adv, zbar), bias_frame(unif, zbar)
    vy = sorted(adv["vel_year"].unique())
    lo, hi = np.percentile(np.concatenate([Ba.to_numpy().ravel(),
                                           Bu.to_numpy().ravel()]) * J_TO_GJ, [1.0, 99.0])
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 32)

    fig, axes = plt.subplots(len(YEARS), len(vy), figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, k in enumerate(YEARS):
        for j, l in enumerate(vy):
            ax = axes[i, j]
            m = adv["vel_year"].to_numpy() == l
            mu = unif["vel_year"].to_numpy() == l
            ax.hist(Ba.loc[m, k] * J_TO_GJ, bins=bins, density=True, color="0.35")
            ax.hist(Bu.loc[mu, k] * J_TO_GJ, bins=bins, density=True, histtype="step",
                    color="tab:blue", lw=1.2, ls="--")
            ax.axvline(0.0, color="crimson", lw=1.2)
            # The mean shift is ~0.1-0.3 GJ/m2 against a per-panel sd of ~0.65, so
            # it is under two bin widths: mark it explicitly or it is invisible.
            ax.axvline(Ba.loc[m, k].mean() * J_TO_GJ, color="k", lw=1.4, ls=":")
            ax.set_xlim(bins[0], bins[-1])
            ax.tick_params(labelsize=6)
            if k == l:
                for s in ax.spines.values():
                    s.set(color="crimson", linewidth=2.0)
            if i == 0:
                ax.set_title(str(l), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Z {k}", fontsize=9)
            if i == len(YEARS) - 1:
                ax.set_xlabel("B (GJ/m2)", fontsize=7)
    fig.suptitle("rows: analysis year of the field   columns: velocity year of the flow"
                 "   (B = Zhat - Zbar, red line at 0)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_bias_heatmap(grid, figsize=(7.6, 6.4)):
    """Mean B per cell in GJ/m2, annotated with its Monte-Carlo standard error."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    m = grid.pivot(index="vel_year", columns="analysis_year", values="mean_B") * J_TO_GJ
    s = grid.pivot(index="vel_year", columns="analysis_year", values="se_B") * J_TO_GJ
    hi = float(np.abs(m.to_numpy()).max())
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi))
    ax.set_xticks(range(m.shape[1]), [str(c) for c in m.columns])
    ax.set_yticks(range(m.shape[0]), [str(r) for r in m.index])
    ax.set_xlabel("analysis year (field)"); ax.set_ylabel("velocity year (flow)")
    for i, l in enumerate(m.index):
        for j, k in enumerate(m.columns):
            ax.text(j, i, f"{m.iat[i, j]:+.3f}\n({s.iat[i, j]:.3f})", ha="center",
                    va="center", fontsize=7,
                    fontweight="bold" if l == k else "normal")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean B (GJ/m2)")
    fig.tight_layout()
    return fig


def plot_decomposition(grid, figsize=(13, 3.4)):
    """Total bias split into its temporal-coverage and spatial-given-time parts."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(YEARS), figsize=figsize, sharey=True)
    for ax, k in zip(axes, YEARS):
        g = grid[grid["analysis_year"] == k].sort_values("vel_year")
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["mean_temporal"] * J_TO_GJ, 0.4, color="tab:orange",
               label="temporal")
        ax.bar(x + 0.2, g["mean_spatial"] * J_TO_GJ, 0.4, color="tab:blue",
               label="spatial | time")
        ax.plot(x, g["mean_B"] * J_TO_GJ, "k.", ms=7, label="total")
        for xi, al in zip(x, g["aligned"]):
            if al:
                ax.axvspan(xi - 0.5, xi + 0.5, color="crimson", alpha=0.12)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x, [str(v) for v in g["vel_year"]], rotation=90, fontsize=7)
        ax.set_title(str(k), fontsize=10)
    axes[0].set_ylabel("GJ/m2")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_alignment_years(boxes=("boxB", "boxA"), arm="_steady", figsize=(11, 3.6)):
    """Theta_k by year with +/-1 and +/-2 se, the pooled level, and the GLS drift."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        th, _, V, _ = alignment_estimates(b, arm)
        y, se = np.asarray(YEARS, float), np.sqrt(np.diag(V))
        x, Vi = y - y.mean(), np.linalg.inv(V)
        drift = (x @ Vi @ th) / (x @ Vi @ x)
        pse = np.sqrt(np.ones(len(th)) @ V @ np.ones(len(th))) / len(th)
        ax.axhline(0, color="crimson", lw=1.0)
        ax.fill_between(y, (th.mean() - pse) * J_TO_GJ, (th.mean() + pse) * J_TO_GJ,
                        color="0.6", alpha=0.3, label="pooled +/- 1 se")
        ax.plot(y, (th.mean() + drift * x) * J_TO_GJ, "--", color="tab:orange",
                lw=1.2, label="GLS drift")
        ax.errorbar(y, th * J_TO_GJ, yerr=2 * se * J_TO_GJ, fmt="none", ecolor="0.5", lw=1)
        ax.errorbar(y, th * J_TO_GJ, yerr=se * J_TO_GJ, fmt="o", color="k", ms=4, lw=1.8)
        ax.set_title(b, fontsize=10)
        ax.set_xticks(y, [str(int(v)) for v in y], rotation=90, fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("Theta_k (GJ/m2)")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--tag", default="uniform_mc",
                    help="read sampled_residuals_<tag>_<box>.csv")
    ap.add_argument("--arm", default="", help="output filename suffix, e.g. _steady")
    ap.add_argument("--gather", metavar="ARM",
                    help="first build sampled_residuals_<tag>_<box>.csv from this arm's "
                         "design_profiles_<box>_<ARM>_y*.csv")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--keep", type=int, help="thin each replicate to this many floats")
    ap.add_argument("--n-deployed", type=int, help="floats seeded per replicate; "
                                                   "required with --keep")
    a = ap.parse_args()
    set_box(a.box)

    if a.gather:
        gather_arm(a.box, a.gather, a.tag)

    zf, wet, w = load_field(a.box)
    zbar, by_doy = domain_means(zf, wet, w)
    cached = pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{a.box}.csv"))
    assert np.allclose(zbar["zbar"], cached["zbar"], rtol=1e-5, atol=1e3), "Zbar mismatch"
    assert np.allclose(zbar["sd_Z"], cached["sd_Z"], rtol=0.02), "sd_Z mismatch"

    adv, doys = advective_estimates(a.box, a.tag, keep=a.keep,
                                    n_deployed=a.n_deployed, seed=a.seed)
    adv = adv.merge(temporal_component(doys, by_doy, zbar["zbar"].to_numpy()),
                    on=["vel_year", "rep"])

    n_vec = adv["n"].to_numpy()
    zu = uniform_estimates(zf, wet, w, n_vec, seed=a.seed)
    unif = adv[["vel_year", "rep", "n"]].copy()
    unif[[f"zhat_{y}" for y in YEARS]] = zu

    grid = bias_grid(adv, unif, zbar)
    for name, df in (("advective", adv), ("uniform", unif), ("grid", grid)):
        df.to_csv(os.path.join(cfg.OUT_DIR,
                  f"mean_mc_{name}_{a.box}{a.arm}.csv"), index=False)

    zb = zbar["zbar"].to_numpy()
    bu = zu - zb
    print(f"{len(adv)} designs, n {n_vec.min()}-{n_vec.max()} (median {int(np.median(n_vec))})")
    print("\nU-full check: mean B +/- MC se, and sd vs sigma*sqrt(mean 1/n)   [GJ/m2]")
    for j, k in enumerate(YEARS):
        pred = zbar["sd_Z"][j] * np.sqrt(np.mean(1.0 / n_vec))
        print(f"  {k}: {bu[:, j].mean()*J_TO_GJ:+.4f} +/- "
              f"{bu[:, j].std(ddof=1)/np.sqrt(len(bu))*J_TO_GJ:.4f}   "
              f"sd {bu[:, j].std(ddof=1)*J_TO_GJ:.3f} vs {pred*J_TO_GJ:.3f}")
    d = grid[grid["aligned"]]
    print("\naligned cells: mean B (MC se), temporal share   [GJ/m2]")
    for _, r in d.iterrows():
        print(f"  {int(r.analysis_year)}: {r.mean_B*J_TO_GJ:+.4f} ({r.se_B*J_TO_GJ:.4f})"
              f"   temporal {r.mean_temporal*J_TO_GJ:+.4f}")
    print(f"\nwrote mean_mc_{{advective,uniform,grid}}_{a.box}{a.arm}.csv")
