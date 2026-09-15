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
                                figsize=(13.5, 5.8), clip=0.995):
    """Single-day park-depth velocity at quantiles of the combined daily spread.

    Both boxes on one both-box base map per column, one shared diverging scale.
    """
    from matplotlib.colors import TwoSlopeNorm

    sel = combined_velocity_quantile_days(comp, qs, boxes)
    cols = {}
    for y, g in sel.groupby("year", sort=False):
        perbox = {b: _velocity_year(b, int(y))[comp] for b in boxes}
        for idx, dd in zip(g.index, g["doy"]):
            cols[idx] = [perbox[b].isel(time=int(dd)) * M_S_TO_CM_S for b in boxes]
    cols = [cols[i] for i in sel.index]
    hi = max(float(np.nanquantile(np.abs(da.values), clip))
             for das in cols for da in das)
    norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(qs), figsize)
    for c, (das, (_, s)) in enumerate(zip(cols, sel.iterrows())):
        im = _draw_domain_map(axes[0][c], das, norm, "RdBu_r", ext,
                              f"{int(s.q * 100)}th pctile day\n"
                              f"{int(s.year)} doy {int(s.doy) + 1}, "
                              f"sd {s.sd:.1f} cm/s", True, c == 0, scale=1.0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label=f"{VEL_COMPS[comp]} velocity at 1000 m (cm/s)")
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


def plot_ohc_decomposition(boxes=("boxB", "boxA"), q=0.5, figsize=(13.5, 6.4),
                           clip=0.995):
    """OHC = mean field + residual on one shared day, both boxes on one map per field.

    The day is the combined-domain q-quantile of daily spatial spread, so both
    boxes are shown on the same date; OHC and mean field share one sequential
    scale (the residual's size is then readable against the OHC it came from),
    the residual its own diverging scale.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import Normalize, TwoSlopeNorm

    d = combined_quantile_days((q,), boxes).iloc[0]
    parts = {b: ohc_decomposition(b, int(d.year), int(d.doy)) for b in boxes}
    date = parts[boxes[0]][3]

    flat = np.concatenate([np.ravel(parts[b][i].values) for b in boxes for i in (0, 1)])
    lo, hi = (float(np.nanquantile(flat, x)) * J_TO_GJ for x in (0.005, 0.995))
    res = max(float(np.nanquantile(np.abs(parts[b][2].values), clip)) for b in boxes) * J_TO_GJ
    norms = [Normalize(lo, hi), Normalize(lo, hi),
             TwoSlopeNorm(vmin=-res, vcenter=0.0, vmax=res)]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    labels = ["OHC 0-2000 m (GJ/m2)", "mean field (GJ/m2)", "residual Z (GJ/m2)"]
    names = ["OHC (observed)", "fitted mean field", "residual Z = OHC - mean"]

    ext = _both_box_extent(boxes)
    pc = ccrs.PlateCarree()
    left, right, top, bottom, hgap = 0.04, 0.99, 0.90, 0.14, 0.03
    w = (right - left - 2 * hgap) / 3
    fig = plt.figure(figsize=figsize)
    for c in range(3):
        x = left + c * (w + hgap)
        ax = fig.add_axes([x, bottom, w, top - bottom], projection=pc)
        im = _draw_domain_map(ax, [parts[b][c] for b in boxes], norms[c], cmaps[c], ext,
                              f"{names[c]}\n{date:%Y-%m-%d}", True, c == 0, boxes=boxes)
        cax = fig.add_axes([x, bottom - 0.08, w, 0.02])
        fig.colorbar(im, cax=cax, orientation="horizontal", label=labels[c])
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


def _raw_area(box):
    """Wet-cell index and unnormalized cell areas (m^2) on the analysis mask."""
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    a = a.values.ravel()
    wet = np.flatnonzero(np.isfinite(a))
    return wet, a[wet]


def _both_box_extent(boxes=("boxB", "boxA"), pad=1.0):
    """Map extent covering every box, padded, for a shared both-box view."""
    la0 = min(cfg.BOXES[b]["lat_bounds"][0] for b in boxes)
    la1 = max(cfg.BOXES[b]["lat_bounds"][1] for b in boxes)
    lo0 = min(cfg.BOXES[b]["lon_bounds"][0] for b in boxes)
    lo1 = max(cfg.BOXES[b]["lon_bounds"][1] for b in boxes)
    return [lo0 - pad, lo1 + pad, la0 - pad, la1 + pad]


def _draw_domain_map(ax, das, norm, cmap, extent, title, bottom_labels, left_labels,
                     scale=J_TO_GJ, boxes=("boxB", "boxA")):
    """Draw one or more per-box field patches on a shared both-box base map.

    Each DataArray in `das` carries its own lat/lon, so the disjoint boxes land
    in place on one map covering both; box outlines are added for orientation.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle
    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="0.88", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.4)
    im = None
    for da in das:
        im = ax.pcolormesh(da["longitude"], da["latitude"], da.values * scale,
                           cmap=cmap, norm=norm, transform=pc, shading="nearest",
                           antialiased=False, linewidth=0, rasterized=True)
    for b in boxes:
        (la0, la1), (lo0, lo1) = cfg.BOXES[b]["lat_bounds"], cfg.BOXES[b]["lon_bounds"]
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, transform=pc,
                               fill=False, edgecolor="0.1", lw=0.8, zorder=4))
    ax.set_title(title, fontsize=9)
    _geo_ticks(ax, extent, bottom_labels, left_labels)
    return im


def combined_daily_spread(boxes=("boxB", "boxA"), path=None):
    """sd_P(Z) per (year, doy) over the union of both boxes under one area measure."""
    path = path or os.path.join(cfg.OUT_DIR, "daily_spread_combined.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    fields = {}
    for b in boxes:
        z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{b}.nc"))["z"]
        zf = z.values.reshape(z.sizes["year"], z.sizes["doy"], -1)
        wet, area = _raw_area(b)
        fields[b] = (zf, wet, area, [int(y) for y in z["year"].values])
    years = sorted(set.intersection(*[set(f[3]) for f in fields.values()]))
    w = np.concatenate([fields[b][2] for b in boxes])
    w = w / w.sum()
    ndoy = fields[boxes[0]][0].shape[1]
    rows = []
    for y in years:
        vals = np.concatenate(
            [fields[b][0][fields[b][3].index(y)][:, fields[b][1]].astype(np.float64)
             for b in boxes], axis=1)
        m1 = vals @ w
        rows.append(pd.DataFrame({"year": y, "doy": np.arange(ndoy), "mean": m1,
                                  "sd": np.sqrt(np.clip((vals * vals) @ w - m1**2, 0, None))}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def combined_quantile_days(qs=(0.05, 0.5, 0.95), boxes=("boxB", "boxA")):
    """The (year, doy) whose combined spatial sd is nearest each quantile."""
    s = combined_daily_spread(boxes)
    sd = s["sd"].to_numpy()
    return pd.DataFrame([
        {"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]), "sd": sd[i]}
        for q in qs for i in [int(np.argmin(np.abs(sd - np.quantile(sd, q))))]])


def combined_velocity_daily_spread(comp="uo", boxes=("boxB", "boxA"), path=None):
    """Spatial sd of a velocity component per (year, doy) over both boxes (cm/s)."""
    path = path or os.path.join(cfg.OUT_DIR, f"daily_vel_spread_combined_{comp}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    areas = {b: _raw_area(b) for b in boxes}
    w = np.concatenate([areas[b][1] for b in boxes])
    w = w / w.sum()
    years = sorted(set.intersection(*[set(_field_years(b)) for b in boxes]))
    rows = []
    for y in years:
        cols, ndoy = [], None
        for b in boxes:
            d = _velocity_year(b, y)[comp]
            ndoy = d.sizes["time"]
            cols.append(d.values.reshape(ndoy, -1)[:, areas[b][0]].astype(np.float64))
        vals = np.concatenate(cols, axis=1) * M_S_TO_CM_S
        m1 = vals @ w
        rows.append(pd.DataFrame({"year": y, "doy": np.arange(ndoy), "mean": m1,
                                  "sd": np.sqrt(np.clip((vals * vals) @ w - m1**2, 0, None))}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def combined_velocity_quantile_days(comp="uo", qs=(0.05, 0.5, 0.95),
                                    boxes=("boxB", "boxA")):
    """The (year, doy) whose combined velocity spatial sd is nearest each quantile."""
    s = combined_velocity_daily_spread(comp, boxes)
    sd = s["sd"].to_numpy()
    return pd.DataFrame([
        {"q": q, "year": int(s["year"][i]), "doy": int(s["doy"][i]), "sd": sd[i]}
        for q in qs for i in [int(np.argmin(np.abs(sd - np.quantile(sd, q))))]])


def plot_quantile_days(boxes=("boxB", "boxA"), qs=(0.05, 0.5, 0.95),
                       figsize=(13.5, 5.8), clip=0.995):
    """Single-day residual fields at quantiles of the combined daily spread.

    Both boxes on one both-box base map per column, one shared diverging scale, so
    the western-boundary and subtropical fields contrast directly; the quantile
    days are chosen on the union-domain spatial dispersion.
    """
    from matplotlib.colors import TwoSlopeNorm

    sel = combined_quantile_days(qs, boxes)
    opened = {b: xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{b}.nc"))
              for b in boxes}
    try:
        cols = [[opened[b]["z"].sel(year=int(d.year)).isel(doy=int(d.doy)).load()
                 for b in boxes] for _, d in sel.iterrows()]
    finally:
        for ds in opened.values():
            ds.close()
    hi = max(float(np.nanquantile(np.abs(da.values), clip))
             for das in cols for da in das) * J_TO_GJ
    norm = TwoSlopeNorm(vmin=-hi, vcenter=0.0, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(qs), figsize)
    for c, (das, (_, d)) in enumerate(zip(cols, sel.iterrows())):
        im = _draw_domain_map(axes[0][c], das, norm, "RdBu_r", ext,
                              f"{int(d.q * 100)}th pctile day\n"
                              f"{int(d.year)} doy {int(d.doy) + 1}, "
                              f"sd {d.sd * J_TO_GJ:.2f} GJ/m2",
                              True, c == 0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="Z (GJ/m2)")
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
                          f"sd_k {b}": t["sd_Z"] * J_TO_GJ,
                          f"Zbar/sd_k {b}": t["zbar"] / t["sd_Z"]})
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
                           figsize=(14.5, 5.2)):
    """Fitted mean field at mid-month, both boxes on one map per month, shared scale."""
    from matplotlib.colors import Normalize
    names = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}

    mfs = {b: mean_field_months(b, year, months) for b in boxes}
    flat = np.concatenate([np.ravel(mfs[b].values) for b in boxes])
    lo, hi = (float(np.nanquantile(flat, q)) * J_TO_GJ for q in (0.01, 0.99))
    norm = Normalize(vmin=lo, vmax=hi)
    ext = _both_box_extent(boxes)

    fig, axes, caxes = _map_grid(1, len(months), figsize)
    for c, ax in enumerate(axes[0]):
        im = _draw_domain_map(ax, [mfs[b].isel(month=c) for b in boxes], norm,
                              "viridis", ext, f"{names.get(months[c], months[c])} {year}",
                              True, c == 0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="mean field (GJ/m2)")
    return fig


def ohc_domain_series(box):
    """Daily domain-mean OHC and fitted mean field over `box`, 2010-2019 (J/m^2).

    Returns (dates, total, mhat) on the 365-day leap-dropped calendar. The mean
    field domain mean is phi(t)^T beta-bar with beta-bar the area-weighted mean
    coefficients; total = mhat + the area-weighted daily residual (domain_means),
    so total - mhat is the domain-mean anomaly.
    """
    zf, wet, w = load_field(box)
    _, by_doy = domain_means(zf, wet, w)
    years = _field_years(box)
    with xr.open_dataset(os.path.join(cfg.OUT_DIR, f"mean_field_coef_{box}.nc")) as ds:
        beta = ds["beta"].values.reshape(ds.sizes["coef"], -1)
        t_ref = float(ds.attrs["t_ref"])
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"].values.ravel()
    m = np.isfinite(a)
    betabar = (beta[:, m] * (a[m] / a[m].sum())).sum(axis=1)

    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    X, _ = cfg.design_matrix(cfg.decimal_year(dates.values), quadratic=True, t_ref=t_ref)
    mhat = X @ betabar
    return dates, mhat + by_doy.reshape(-1), mhat


def plot_ohc_timeseries(boxes=("boxA", "boxB"), figsize=(11, 6.5)):
    """Two stacked panels sharing the date axis: (top) domain-mean total OHC
    (solid) and fitted mean field (dashed) per box; (bottom) the domain-mean
    anomaly (total - mean field) per box. Daily over 2010-2019, GJ/m^2."""
    import matplotlib.pyplot as plt
    color = {"boxA": "tab:orange", "boxB": "tab:blue"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for b in boxes:
        dates, total, mhat = ohc_domain_series(b)
        c = color.get(b)
        a1.plot(dates, total * J_TO_GJ, color=c, lw=0.7, alpha=0.9, label=f"{b} total OHC")
        a1.plot(dates, mhat * J_TO_GJ, color=c, lw=1.4, ls="--", label=f"{b} mean field")
        a2.plot(dates, (total - mhat) * J_TO_GJ, color=c, lw=0.8, label=b)
    a1.set_ylabel("domain-mean OHC 0-2000 m (GJ/m2)")
    a1.legend(fontsize=8, ncol=2)
    a2.axhline(0.0, color="0.4", lw=1.0, ls=":")
    a2.set_ylabel("anomaly (GJ/m2)")
    a2.set_xlabel("date")
    a2.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def _domain_base(boxes=("boxA", "boxB"), figsize=(7.5, 6.4)):
    """Base analysis-domain map: land, the velocity/thetao regions, and the boxes.

    Returns (fig, ax, pc, extents); the caller adds the legend and any overlay.
    """
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
    ext = [v[2] - 1.5, v[3] + 1.5, v[0] - 1.5, v[1] + 1.5]
    ax.set_extent(ext, crs=pc)
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
    _geo_ticks(ax, ext, True, True, step=10.0)
    return fig, ax, pc, e


def plot_domain_map(boxes=("boxA", "boxB"), figsize=(7.5, 6.4)):
    """Analysis boxes inside the velocity and thetao download regions."""
    fig, ax, pc, e = _domain_base(boxes, figsize)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return fig


def _trajectory_map(tracks, boxes=("boxA", "boxB"), max_floats=60, figsize=(8.5, 7.2),
                    order="time", years=None, subset_seed=None):
    """Float drift tracks + profile points on the domain base map, coloured by the
    calendar year of each profile. `tracks` needs float_id, lat, lon, year and an
    `order` column; `years` fixes the colour scale so two maps stay comparable."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.collections import LineCollection

    tracks = tracks.dropna(subset=["lat", "lon"]).sort_values(["float_id", order])
    ids = tracks["float_id"].unique()
    if max_floats is not None and len(ids) > max_floats:
        ids = (np.random.default_rng(subset_seed).choice(ids, max_floats, replace=False)
               if subset_seed is not None else np.sort(ids)[:max_floats])
        tracks = tracks[tracks["float_id"].isin(ids)]
    yr = years or sorted(int(y) for y in tracks["year"].unique())
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)

    fig, ax, pc, e = _domain_base(boxes, figsize)
    for _, g in tracks.groupby("float_id"):
        p = g[["lon", "lat"]].to_numpy()
        if len(p) < 2:
            continue
        segs = np.stack([p[:-1], p[1:]], axis=1)
        ax.add_collection(LineCollection(segs, colors=cmap(norm(g["year"].to_numpy()[:-1])),
                                         lw=0.35, alpha=0.5, transform=pc, zorder=1.5))
    ax.scatter(tracks["lon"], tracks["lat"], c=tracks["year"], cmap=cmap, norm=norm,
               s=1.5, alpha=0.75, transform=pc, zorder=3)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        ticks=yr, shrink=0.8, pad=0.02)
    cbar.set_label("profile year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def plot_deployment_advection_map(tracks=None, boxes=("boxA", "boxB"),
                                  max_floats=60, figsize=(8.5, 7.2)):
    """One simulated realization's advection: drift tracks + profile points by year."""
    if tracks is None:
        tracks = pd.read_csv(os.path.join(cfg.OUT_DIR, "illustration_tracks.csv"))
    return _trajectory_map(tracks, boxes, max_floats, figsize, order="time",
                           years=list(range(2010, 2020)))


def plot_argo_trajectory_map(boxes=("boxA", "boxB"), years=(2010, 2019),
                             buffer=((15, 50), (-70, -30)), max_floats=60,
                             figsize=(8.5, 7.2), path=None):
    """Real Argo trajectories over the advection domain, same base map and year
    colouring as the simulated map, for side-by-side comparison. Positions are the
    profile (surface) locations in data/argo_ohc.csv; a random subset of the floats
    present in the domain is drawn for legibility."""
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d["year"] = d["date"].dt.year
    (la0, la1), (lo0, lo1) = buffer
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)
          & d["year"].between(*years)].rename(columns={"date": "time"})
    return _trajectory_map(d, boxes, max_floats, figsize, order="time",
                           years=list(range(years[0], years[1] + 1)), subset_seed=0)


# Six real Argo floats deployed 27-28 Sep 2015 in a tight cluster in the NW corner
# of Box B (lat 43.4-44.3 N, lon -55.8 to -57.8 W) -- one deployment leg on the
# Grand Banks / Gulf Stream-Labrador confluence, used for the per-float bias study.
NW_BOXB_2015 = (6902632, 6902634, 6902635, 6902633, 4902099, 4902100)


def _cluster_colors(float_ids):
    import matplotlib.pyplot as plt
    cyc = plt.get_cmap("tab10")
    return {int(f): cyc(i % 10) for i, f in enumerate(float_ids)}


def _delta_frame(d, box):
    """Attach per-profile delta = Z(x_i) - Zbar_k to positions with float_id, lat,
    lon, year, doy, age, pnum. Restricts to `box` support and the field's years."""
    from osse import sample_residuals as sr
    set_box(box)
    (la0, la1) = cfg.BOXES[box]["lat_bounds"]; (lo0, lo1) = cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)]
    z = sr.load_residual_field(box)
    zyears = [int(y) for y in z["year"].values]
    d = d[d["year"].isin(zyears)]
    sampled, _ = sr.gather(d, z)
    col = {y: j for j, y in enumerate(zyears)}
    zc = sampled[[f"z_{y}" for y in zyears]].to_numpy()
    sampled["z_own"] = zc[np.arange(len(sampled)), [col[y] for y in sampled["year"].to_numpy()]]
    zf, wet, w = load_field(box)
    dm = domain_means(zf, wet, w)[0].set_index("year")
    sampled["delta"] = sampled["z_own"] - sampled["year"].map(dm["zbar"])
    sampled["delta_std"] = sampled["delta"] / sampled["year"].map(dm["sd_Z"])
    return sampled[["float_id", "year", "doy", "lat", "lon", "age", "pnum",
                    "delta", "delta_std"]].reset_index(drop=True)


def float_bias_profile(float_ids, box="boxB", source="real", cleaned=None):
    """Per-profile single-profile bias delta = Z(x_i) - Zbar_k for named floats.

    source='real': QC-cleaned real Argo profiles (pnum = ordinal in the float's full
    cleaned record). source='sim': GLORYS 1000 m park-advection sim-twin positions
    (argo-actual arm, matched via the deployment pool; pnum = 10-day cycle). Both are
    restricted to `box` (the field's support), so out-of-box profiles drop out.
    """
    ids = [int(f) for f in float_ids]
    if source == "real":
        d = (clean_argo() if cleaned is None else cleaned).copy()
        d = d[d["float_id"].isin(ids)].sort_values(["float_id", "date"])
        d["age"] = (d["date"] - d.groupby("float_id")["date"].transform("min")).dt.total_seconds() / 86400.0
        d["pnum"] = d.groupby("float_id").cumcount() + 1
    elif source == "sim":
        from osse.continuous_sim import leap_dropped_doy
        pool = pd.read_csv(os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"))
        f2s = pool.set_index("seed_id")["float_id"]
        tr = pd.read_csv(os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
        tr = tr[tr["seed_id"].isin(pool.loc[pool["float_id"].isin(ids), "seed_id"])].copy()
        tr["float_id"] = tr["seed_id"].map(f2s).astype(int)
        tr = tr.sort_values(["float_id", "time"])
        tr["age"] = (tr["time"] - tr.groupby("float_id")["time"].transform("min")).dt.total_seconds() / 86400.0
        yr, doy = leap_dropped_doy(tr["time"].to_numpy())
        tr["year"], tr["doy"], tr["pnum"] = yr, doy, tr["cycle"].astype(int)
        d = tr[(tr["cycle"] > 0) & (tr["doy"] >= 0)]     # drop release record and Feb-29
    else:
        raise ValueError(f"source must be 'real' or 'sim', got {source!r}")
    return _delta_frame(d, box)


def plot_float_bias_profile(float_ids=NW_BOXB_2015, box="boxB", x="pnum",
                            source="real", figsize=(11, 5.0)):
    """Single-profile bias delta vs profile (or age) for each named float + the
    cluster mean, in GJ/m2. Only in-box profiles carry a defined bias."""
    import matplotlib.pyplot as plt
    s = float_bias_profile(float_ids, box, source)
    cols = _cluster_colors(float_ids)
    fig, ax = plt.subplots(figsize=figsize)
    for fid, g in s.groupby("float_id"):
        g = g.sort_values(x)
        ax.plot(g[x], g["delta"] * J_TO_GJ, "-o", ms=3, lw=1.0, color=cols[int(fid)],
                alpha=0.85, label=f"{int(fid)} (n={len(g)})")
    xb = np.linspace(s[x].min(), s[x].max(), 9)
    s = s.assign(_b=pd.cut(s[x], xb))
    m = s.groupby("_b", observed=True).agg(xc=(x, "mean"), mu=("delta", "mean"))
    ax.plot(m["xc"], m["mu"] * J_TO_GJ, "k-", lw=2.5, zorder=5, label="cluster mean")
    ax.axhline(0, color="0.4", lw=0.8)
    xlab = {"pnum": "profile number (cleaned record, ~10-day cycles)",
            "age": "days since deployment"}[x]
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"single-profile bias  $Z(x_i)-\bar Z_k$  (GJ/m$^2$)")
    src = {"real": "real Argo positions", "sim": "GLORYS park-advection positions"}[source]
    ax.set_title(f"Per-float sampling bias over life, {box} ({src}, deployed 27-28 Sep 2015)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _cluster_track_map(paths, float_ids, boxes=("boxA", "boxB"), figsize=(8.5, 7.2),
                       title=""):
    """Draw one colour-per-float trajectory set on the domain base map, with a black
    star at each deployment (first point) and an open circle at the last point."""
    fig, ax, pc, e = _domain_base(boxes, figsize)
    cols = _cluster_colors(float_ids)
    for fid in float_ids:
        g = paths.get(int(fid))
        if g is None or len(g) == 0:
            continue
        c = cols[int(fid)]
        ax.plot(g["lon"], g["lat"], "-", color=c, lw=0.9, alpha=0.9, transform=pc,
                zorder=3, label=str(int(fid)))
        ax.plot(g["lon"].iloc[0], g["lat"].iloc[0], "*", color="k", mfc=c, ms=13,
                mew=0.8, transform=pc, zorder=5)
        ax.plot(g["lon"].iloc[-1], g["lat"].iloc[-1], "o", mfc="none", mec=c, mew=1.4,
                ms=7, transform=pc, zorder=4)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, title="float")
    return fig


def plot_cluster_real_tracks(float_ids=NW_BOXB_2015, boxes=("boxA", "boxB"),
                             figsize=(8.5, 7.2), path=None):
    """Real Argo (raw) trajectories of the named floats, one colour each."""
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d[d["float_id"].isin([int(f) for f in float_ids])].dropna(subset=["lat", "lon"])
    paths = {int(f): g.sort_values("date") for f, g in d.groupby("float_id")}
    return _cluster_track_map(paths, float_ids, boxes, figsize,
                              "Real Argo trajectories (deployment star, last-fix circle)")


def plot_cluster_sim_tracks(float_ids=NW_BOXB_2015, boxes=("boxA", "boxB"),
                            figsize=(8.5, 7.2)):
    """GLORYS park-advection sim-twin trajectories of the named floats (argo-actual
    arm), matched to float_id through the deployment pool, one colour each."""
    pool = pd.read_csv(os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"))
    tr = pd.read_csv(os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"),
                     parse_dates=["time"])
    sid = pool.set_index("float_id")["seed_id"]
    paths = {}
    for fid in float_ids:
        s = int(sid.get(int(fid), -1))
        g = tr[tr["seed_id"] == s].sort_values("time")
        paths[int(fid)] = g.dropna(subset=["lat", "lon"])
    return _cluster_track_map(paths, float_ids, boxes, figsize,
                              "GLORYS park-advection sim-twin trajectories")


def raw_deployments(years=(2010, 2019), domain="velocity", boxes=("boxA", "boxB"), path=None):
    """First profile (deployment) of every float in argo_ohc.csv, NOT QC-cleaned.

    Restricted to deployment years in [years[0], years[1]] and to `domain`:
    'velocity' the advection domain, 'analysis' the union of the analysis boxes.
    Deployment is the earliest profile on record, so a float first observed before
    the file window is not resolved as deployed in-window -- see the report note.
    Returns float_id, lat, lon, date, year.
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d.dropna(subset=["lat", "lon", "date"])
    dep = d.loc[d.groupby("float_id")["date"].idxmin()].copy()
    dep["year"] = dep["date"].dt.year
    e = domain_extents(boxes)
    if domain == "velocity":
        la0, la1, lo0, lo1 = e["velocity"]
        m = dep["lat"].between(la0, la1) & dep["lon"].between(lo0, lo1)
    else:
        m = np.zeros(len(dep), bool)
        for b in boxes:
            la0, la1, lo0, lo1 = e[b]
            m = m | (dep["lat"].between(la0, la1) & dep["lon"].between(lo0, lo1)).to_numpy()
    dep = dep[m & dep["year"].between(years[0], years[1])]
    return dep[["float_id", "lat", "lon", "date", "year"]].reset_index(drop=True)


def plot_deployment_positions_map(years=(2010, 2019), domain="velocity",
                                  boxes=("boxA", "boxB"), figsize=(8.5, 7.2)):
    """Raw (not QC-cleaned) float deployment positions in the advection domain,
    coloured by deployment year, on the shared domain base map."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    dep = raw_deployments(years, domain, boxes)
    yr = list(range(years[0], years[1] + 1))
    cmap = plt.get_cmap("turbo", len(yr))
    norm = mpl.colors.BoundaryNorm(np.arange(yr[0], yr[-1] + 2) - 0.5, cmap.N)
    fig, ax, pc, e = _domain_base(boxes, figsize)
    ax.scatter(dep["lon"], dep["lat"], c=dep["year"], cmap=cmap, norm=norm, s=30,
               alpha=0.9, edgecolor="k", linewidth=0.3, transform=pc, zorder=3)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        ticks=yr, shrink=0.8, pad=0.02)
    cbar.set_label("deployment year")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    return fig


def plot_trajectory_twins(pairs=((295, 6901117), (68, 4901217)), tracks=None,
                          pool=None, cleaned=None, boxes=("boxA", "boxB"),
                          figsize=(8.5, 7.2)):
    """Real Argo float vs its GLORYS-park sim twin for a few long-lived floats,
    to show the two tracks drift apart from a shared deployment. Each pair shares a
    colour; real is solid, sim twin dashed; the shared deployment is a black star,
    the real end a filled circle and the sim end an open square."""
    import matplotlib.pyplot as plt
    tr = tracks if tracks is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
    pl = pool if pool is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"), parse_dates=["date"])
    real = clean_argo() if cleaned is None else cleaned
    to_naive = lambda s: pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)
    real = real.assign(date=to_naive(real["date"]))
    pl = pl.assign(date=to_naive(pl["date"]))
    tr = tr.assign(time=pd.to_datetime(tr["time"], format="ISO8601").astype("datetime64[ns]"))
    realg = {int(f): g.sort_values("date") for f, g in real.groupby("float_id")}
    simg = {int(s): g.sort_values("time") for s, g in tr.groupby("seed_id")}
    dep = {int(r.seed_id): r for r in pl.itertuples()}

    fig, ax, pc, e = _domain_base(boxes, figsize)
    # (real, sim twin) colour per float: matched hue families so the pairing reads,
    # distinct shades so real vs twin stay apart where the tracks overlap.
    pair_colors = [("#08519c", "#41b6c4"), ("#6a1b9a", "#ec7fc3"),
                   ("#1b7837", "#a6d96a"), ("#8c510a", "#dfc27d")]
    for k, (sid, fid) in enumerate(pairs):
        cr, cs = pair_colors[k % len(pair_colors)]
        d = dep[int(sid)]
        rt = realg[int(fid)]; rt = rt[rt["date"] >= d.date]
        st = simg[int(sid)]
        yrs = (rt["date"].max() - d.date).days / 365.25
        ax.plot(rt["lon"], rt["lat"], "-", color=cr, lw=1.0, transform=pc, zorder=3,
                label=f"real {fid} ({yrs:.1f} yr)")
        ax.plot(st["lon"], st["lat"], "--", color=cs, lw=1.0, transform=pc,
                zorder=3, label=f"sim twin {fid}")
        ax.plot(rt["lon"].iloc[-1], rt["lat"].iloc[-1], "o", color=cr, ms=6,
                transform=pc, zorder=4)
        ax.plot(st["lon"].iloc[-1], st["lat"].iloc[-1], "s", mfc="none", mec=cs,
                mew=1.4, ms=7, transform=pc, zorder=4)
        ax.plot(d.lon, d.lat, "*", color="k", ms=13, mec="w", mew=0.6,
                transform=pc, zorder=5, label="deployment" if k == 0 else None)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
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


def _reference_sd(box="boxB", arm="_steady"):
    """s_k in J/m2, and how far the reference errors of different years are related.

    s_k is the spread of the single-flow bias across the M-1 unaligned flows, net of
    each cell's own Monte-Carlo error. The scalar returned with it is the mean
    pairwise correlation, across analysis years, of the within-row deviations that
    generate s_k: near zero means each year's reference misses independently.
    """
    _, _, grid, _ = load_arm(box, arm)
    s = aligned_table(grid)["implied between-flow sd"].to_numpy() / J_TO_GJ
    G, M = alignment_estimates(box, arm)[1], len(s)
    off = ~np.eye(M, dtype=bool)
    E = G - np.array([G[k, off[k]].mean() for k in range(M)])[:, None]
    r = [np.corrcoef(E[k, off[k] & off[j]], E[j, off[k] & off[j]])[0, 1]
         for k in range(M) for j in range(k + 1, M)]
    return s, float(np.mean(r))


def alignment_reference_noise_table(boxes=("boxB", "boxA"), arm="_steady"):
    """What reading Theta_k as the preferential bias costs beyond Monte-Carlo error.

    The reference averages M-1 unaligned flows, so even with every one of them
    independent of Z_k it sits s_k/sqrt(M-1) away from the population footprint.
    That is a superpopulation quantity -- conditional on the realized record the gap
    is a fixed unknown, not a variance -- and it is absent from the eq:var-general se.
    """
    out = None
    for b in boxes:
        th, _, V, _ = alignment_estimates(b, arm)
        s, _ = _reference_sd(b, arm)
        f = s / np.sqrt(len(th) - 1)
        c = np.hypot(np.sqrt(np.diag(V)), f)
        t = pd.DataFrame({"year": list(YEARS), f"Theta {b}": th * J_TO_GJ,
                          f"MC se {b}": np.sqrt(np.diag(V)) * J_TO_GJ,
                          f"s_k {b}": s * J_TO_GJ, f"ref floor {b}": f * J_TO_GJ,
                          f"combined se {b}": c * J_TO_GJ, f"t {b}": th / c})
        out = t if out is None else out.merge(t, on="year")
    return out


def alignment_reference_pooling_table(boxes=("boxB", "boxA"), arm="_steady"):
    """Whether the reference error pools away, and what it costs the pooled level and drift."""
    rows = []
    for b in boxes:
        th, _, V, _ = alignment_estimates(b, arm)
        s, rho = _reference_sd(b, arm)
        M = len(th)
        x = np.asarray(YEARS, float) - np.mean(YEARS)
        r = {}
        for lab, C in (("mc", V), ("ref", V + np.diag(s ** 2 / (M - 1)))):
            Ci = np.linalg.inv(C)
            r[lab] = (np.sqrt(np.ones(M) @ C @ np.ones(M)) / M,
                      (x @ Ci @ th) / (x @ Ci @ x), 1.0 / np.sqrt(x @ Ci @ x))
        rows.append({"box": b, "mean corr across years": rho,
                     "pooled se MC": r["mc"][0] * J_TO_GJ,
                     "pooled se +ref": r["ref"][0] * J_TO_GJ,
                     "pooled t MC": th.mean() / r["mc"][0],
                     "pooled t +ref": th.mean() / r["ref"][0],
                     "drift t MC": r["mc"][1] / r["mc"][2],
                     "drift t +ref": r["ref"][1] / r["ref"][2],
                     "W/m2 per decade +ref": r["ref"][1] * 10.0 / 3.1536e8})
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


# ----------------------- continuous-seeding ensemble --------------------------
# One row per (box, rep, year) in estimates_continuous<tag>.csv: B, B_unif,
# standing_stock, n, n_floats, zbar. These read that frame directly and take the
# box as an argument, so none depend on the global YEARS.

def load_continuous(tag="_pilot"):
    return pd.read_csv(os.path.join(cfg.OUT_DIR, f"estimates_continuous{tag}.csv"))


def _sd_Z(box):
    return pd.read_csv(os.path.join(cfg.OUT_DIR, f"domain_means_{box}.csv")).set_index("year")["sd_Z"]


def continuous_bias_table(est, box, ref=None):
    """Per-year advective bias distribution, with the uniform-null mean alongside.

    The `uniform mean B` column is the mean of the large `ref` ensemble when given,
    else the aligned per-rep B_unif in `est`.
    """
    sd = _sd_Z(box)
    rows = []
    for k, gk in est[est["box"] == box].groupby("year"):
        b = gk["B"].to_numpy()
        um = (ref[(ref["box"] == box) & (ref["year"] == k)]["B_unif"].mean()
              if ref is not None else gk["B_unif"].mean())
        rows.append({"year": int(k), "reps": b.size,
                     "mean B": b.mean() * J_TO_GJ,
                     "MC se": b.std(ddof=1) / np.sqrt(b.size) * J_TO_GJ,
                     "sd over deployments": b.std(ddof=1) * J_TO_GJ,
                     "5%": np.percentile(b, 5) * J_TO_GJ,
                     "95%": np.percentile(b, 95) * J_TO_GJ,
                     "P(B < 0)": float((b < 0).mean()),
                     "sd_k": sd[int(k)] * J_TO_GJ,
                     "mean B / sd_k": b.mean() / sd[int(k)],
                     "uniform mean B": um * J_TO_GJ})
    return pd.DataFrame(rows)


def continuous_stock_table(est, boxes=("boxB", "boxA")):
    """Virtual against real Argo standing stock, both boxes, by analysis year.

    Standing stock is the mean number of distinct floats reporting in a 10-day
    window -- the quantity that compares across the two, since real and virtual
    floats cycle on different schedules. Real stock is from the Argo profile
    record restricted to the box; virtual stock is the ensemble-mean in-box stock.
    """
    out = None
    for b in boxes:
        sim = est[est["box"] == b].groupby("year")["standing_stock"].mean()
        real = argo_region_summary(b).set_index("year")["argo_stock"].reindex(sim.index)
        t = pd.DataFrame({"year": sim.index.astype(int),
                          f"Argo {b}": real.to_numpy(),
                          f"virtual {b}": sim.to_numpy(),
                          f"ratio {b}": sim.to_numpy() / real.to_numpy()})
        out = t if out is None else out.merge(t, on="year")
    out.loc["mean"] = ["mean", *out.iloc[:, 1:].mean()]
    return out


def continuous_nsize_table(est, boxes=("boxB", "boxA")):
    """Retained sample size per design (profiles, distinct floats), both boxes."""
    rows = []
    for b in boxes:
        g = est[est["box"] == b]
        for lab, s in (("profiles", g["n"]), ("distinct floats", g["n_floats"])):
            rows.append({"box": b, "quantity": f"{lab} per design",
                         "min": s.min(), "5%": s.quantile(.05), "median": s.median(),
                         "95%": s.quantile(.95), "max": s.max()})
    return pd.DataFrame(rows)


def uniform_reference_ensemble(est, R=2000, seed=0):
    """Large n-matched uniform null: R iid draws from P per (box, year), each at the
    median advective sample size for that cell. E[B]=0, so the ensemble mean is a
    precise zero reference (its 95% CI covers zero) and its 5-95% spread is the
    single-deployment scatter of a uniform array. Returns box, year, B_unif (R rows
    per group). Decoupled from the 100 advective reps, which are limited by the
    advection cost; the null is not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for box in sorted(est["box"].unique()):
        zf, wet, w = load_field(box)
        yidx = {y: j for j, y in enumerate(_field_years(box))}
        zbar = domain_means(zf, wet, w)[0].set_index("year")["zbar"]
        cw = np.cumsum(w)
        g = est[est["box"] == box]
        for y in sorted(int(v) for v in g["year"].unique()):
            n = int(np.median(g.loc[g["year"] == y, "n"].to_numpy()))
            cells = wet[np.searchsorted(cw, rng.random(R * n)).clip(max=len(w) - 1)].reshape(R, n)
            d = rng.integers(0, cfg.DAYS_PER_YEAR, (R, n))
            b = zf[yidx[y]][d, cells].mean(axis=1) - zbar[y]
            rows += [(box, y, float(v)) for v in b]
    return pd.DataFrame(rows, columns=["box", "year", "B_unif"])


def clean_argo(path=None, min_spacing=10, max_gap=90, min_span=90):
    """QC-cleaned real Argo profile record (float_id, lat, lon, date, year, doy).

    Applied to each float's FULL global record: (1) thin to >= min_spacing days
    between kept profiles (drops near-duplicate surface fixes); (2) drop the whole
    float if any consecutive gap exceeds max_gap days OR its total span is under
    min_span days. doy is on the 365-day leap-dropped calendar (Feb 29 dropped).
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    d = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    d["date"] = pd.to_datetime(d["date"], format="ISO8601", utc=True)
    d = d.dropna(subset=["lat", "lon", "date"]).sort_values(["float_id", "date"])

    # Greedy thin: keep a profile only if >= min_spacing days since the last KEPT one
    # (per float), so runs of sub-cycle fixes are thinned to ~min_spacing rather than
    # collapsed to a single point.
    fid = pd.factorize(d["float_id"].to_numpy())[0]
    day = (d["date"] - d["date"].min()).dt.total_seconds().to_numpy() / 86400.0
    keep = np.zeros(len(d), dtype=bool)
    last = -1e18
    for i in range(len(d)):
        if i and fid[i] != fid[i - 1]:
            last = -10 ** 9
        if day[i] - last >= min_spacing:
            keep[i] = True
            last = day[i]
    d = d[keep]
    gap = d.groupby("float_id")["date"].diff().dt.total_seconds() / 86400.0
    g = d.groupby("float_id")
    max_gap_f = gap.groupby(d["float_id"]).transform("max")      # NaN for single-profile floats
    span_f = (g["date"].transform("max") - g["date"].transform("min")).dt.total_seconds() / 86400.0
    d = d[(max_gap_f <= max_gap) & (span_f >= min_span)]         # NaN comparisons -> False -> dropped

    dt = d["date"].dt
    doy0 = (dt.dayofyear - 1).to_numpy()
    leap = dt.is_leap_year.to_numpy()
    d = d.assign(year=dt.year.to_numpy(),
                 doy=np.where(leap & (doy0 >= 60), doy0 - 1, doy0))
    d.loc[leap & (dt.month.to_numpy() == 2) & (dt.day.to_numpy() == 29), "doy"] = -1
    return d[d["doy"] >= 0]


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between arrays of lat/lon points."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def trajectory_divergence(tol_days=6, max_days=None, tracks=None, pool=None, cleaned=None):
    """Separation between each real Argo float and its GLORYS-park sim twin.

    The argo-actual full-trajectory run advects each real deployment verbatim; this
    matches every real profile (after deployment) to the nearest sim position in
    time (within tol_days) and returns the great-circle separation vs days since
    deployment. Isolates real-vs-reanalysis drift: a park-only sim that tracked
    real floats would stay small; one that diverges to the array's own scale shows
    the residual bias gap is advection error, which a dive cycle cannot fix.
    """
    tr = tracks if tracks is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "actual_full_tracks.csv"), parse_dates=["time"])
    pl = pool if pool is not None else pd.read_csv(
        os.path.join(cfg.OUT_DIR, "argo_actual_pool.csv"), parse_dates=["date"])
    real = clean_argo() if cleaned is None else cleaned
    to_naive = lambda s: pd.to_datetime(s, utc=True, format="ISO8601").dt.tz_localize(None)
    real = real.assign(date=to_naive(real["date"]))
    pl = pl.assign(date=to_naive(pl["date"]))
    tr = tr.assign(time=pd.to_datetime(tr["time"], format="ISO8601").astype("datetime64[ns]"))
    realg = {int(f): g.sort_values("date") for f, g in real.groupby("float_id")}
    simg = {int(s): g.sort_values("time") for s, g in tr.groupby("seed_id")}

    rows = []
    for pr in pl.itertuples():
        st = simg.get(int(pr.seed_id))
        rt = realg.get(int(pr.float_id))
        if st is None or rt is None:
            continue
        rt = rt[rt["date"] >= pr.date]
        if len(rt) < 2 or len(st) < 2:
            continue
        s_t = st["time"].to_numpy()
        r_t = rt["date"].to_numpy()
        j = np.clip(np.searchsorted(s_t, r_t), 1, len(s_t) - 1)
        pick = np.where(np.abs(r_t - s_t[j - 1]) <= np.abs(s_t[j] - r_t), j - 1, j)
        gap = np.abs((s_t[pick] - r_t) / np.timedelta64(1, "D"))
        ok = gap <= tol_days
        if not ok.any():
            continue
        rows.append(pd.DataFrame({
            "seed_id": int(pr.seed_id), "float_id": int(pr.float_id),
            "elapsed_days": (r_t[ok] - np.datetime64(pr.date)) / np.timedelta64(1, "D"),
            "sep_km": _haversine_km(rt["lat"].to_numpy()[ok], rt["lon"].to_numpy()[ok],
                                    st["lat"].to_numpy()[pick][ok],
                                    st["lon"].to_numpy()[pick][ok]),
            "rlat": rt["lat"].to_numpy()[ok], "rlon": rt["lon"].to_numpy()[ok]}))
    div = pd.concat(rows, ignore_index=True)
    return div[div["elapsed_days"] <= max_days] if max_days else div


def _decorrelation_km(div, seed=0, n=6000):
    """Median great-circle distance between random pairs of the array's profile
    positions -- the separation two independent floats reach, i.e. the no-skill floor."""
    pts = div[["rlat", "rlon"]].drop_duplicates().to_numpy()
    r = np.random.default_rng(seed)
    i, j = r.integers(0, len(pts), n), r.integers(0, len(pts), n)
    return float(np.median(_haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1])))


def trajectory_divergence_summary(div=None, marks=(10, 30, 90, 180, 365, 730)):
    """Median (IQR) sim-vs-real separation and n at a few elapsed-time marks."""
    div = trajectory_divergence() if div is None else div
    rows = []
    for m in marks:
        w = div[(div["elapsed_days"] > m - 15) & (div["elapsed_days"] <= m + 15)]["sep_km"]
        if len(w):
            rows.append({"days since deployment": m, "median sep (km)": w.median(),
                         "IQR (km)": w.quantile(0.75) - w.quantile(0.25), "n": len(w)})
    out = pd.DataFrame(rows)
    out.attrs["decorrelation_km"] = _decorrelation_km(div)
    return out


def plot_trajectory_divergence(div=None, bin_days=20, figsize=(9.5, 5.2)):
    """Sim-vs-real separation vs days since deployment: median + IQR, with the
    decorrelated-floats floor. Rising to that floor = park advection loses the float."""
    import matplotlib.pyplot as plt
    div = trajectory_divergence() if div is None else div
    b = (np.floor(div["elapsed_days"] / bin_days) * bin_days + bin_days / 2)
    g = div.assign(b=b).groupby("b")["sep_km"]
    med, q1, q3, n = g.median(), g.quantile(0.25), g.quantile(0.75), g.size()
    x = med.index.to_numpy()[n.to_numpy() >= 20]
    med, q1, q3 = med.loc[x], q1.loc[x], q3.loc[x]
    sat = _decorrelation_km(div)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, q1, q3, alpha=0.22, color="tab:blue", label="IQR")
    ax.plot(x, med, color="tab:blue", lw=1.8, label="median separation")
    ax.axhline(sat, ls="--", color="0.4",
               label=f"decorrelated floats (~{sat:.0f} km)")
    ax.set_xlabel("days since deployment")
    ax.set_ylabel("real vs GLORYS-park separation (km)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def argo_bias_by_year(box, cleaned=None, path=None):
    """Realized bias of the REAL Argo array sampling the GLORYS anomaly field:
    B_k = mean over that box's year-k Argo profiles of Z_k, minus Zbar_k (J/m^2).

    Same estimator as the simulated arm, on the QC-cleaned Argo profile positions
    (surface locations; see clean_argo). Returns a Series year -> B_argo for the
    years present in both the Argo record and the residual field.
    """
    from osse import sample_residuals as sr
    d = clean_argo(path) if cleaned is None else cleaned
    (la0, la1), (lo0, lo1) = cfg.BOXES[box]["lat_bounds"], cfg.BOXES[box]["lon_bounds"]
    d = d[d["lat"].between(la0, la1) & d["lon"].between(lo0, lo1)]

    z = sr.load_residual_field(box)
    years = [int(y) for y in z["year"].values]
    sampled, _ = sr.gather(d, z)
    sampled = sampled[sampled["year"].isin(years)].copy()
    col = {y: j for j, y in enumerate(years)}
    zc = sampled[[f"z_{y}" for y in years]].to_numpy()
    sampled["z_own"] = zc[np.arange(len(sampled)),
                          [col[y] for y in sampled["year"].to_numpy()]]
    zf, wet, w = load_field(box)
    zbar = domain_means(zf, wet, w)[0].set_index("year")["zbar"]
    ghat = sampled.groupby("year")["z_own"].mean()
    return (ghat - zbar).dropna()


def continuous_pilot_size_table(est, boxes=("boxB", "boxA")):
    """Per box/year pilot sample sizes: mean in-box standing stock and distinct
    floats per realization, mean profile observations per realization, and the
    profile total over all realizations."""
    rows = []
    for b in boxes:
        g = est[est["box"] == b]
        for y in sorted(int(v) for v in g["year"].unique()):
            gy = g[g["year"] == y]
            rows.append({"box": b, "year": y,
                         "in-box stock": gy["standing_stock"].mean(),
                         "floats / realization": gy["n_floats"].mean(),
                         "profiles / realization": gy["n"].mean(),
                         "profiles, all reps": int(gy["n"].sum())})
    return pd.DataFrame(rows)


def continuous_deff_table(tag="_pilot", boxes=("boxB", "boxA"), ref=None):
    """Float-clustered design effect and advective/uniform width ratio, per box/year.

    The design effect re-gathers the saved profile positions against the residual
    field; the width ratio is the advective across-realization sd(B) against the
    n-matched uniform sd(B) (from the large `ref` ensemble when given, else the
    aligned per-rep B_unif).
    """
    from osse import sample_residuals as sr
    est = load_continuous(tag)
    prof = pd.read_csv(os.path.join(cfg.OUT_DIR, f"design_profiles_continuous{tag}.csv"),
                       usecols=["rep", "float_id", "box", "year", "lat", "lon", "doy"])
    rows = []
    for b in boxes:
        z = sr.load_residual_field(b)
        years = [int(y) for y in z["year"].values]
        col = {y: j for j, y in enumerate(years)}
        sampled, _ = sr.gather(prof[prof["box"] == b], z)
        zc = sampled[[f"z_{y}" for y in years]].to_numpy()
        sampled["z_own"] = zc[np.arange(len(sampled)),
                              [col[y] for y in sampled["year"].to_numpy()]]
        e = est[est["box"] == b]
        for k in sorted(e["year"].unique()):
            deffs = []
            for _, gr in sampled[sampled["year"] == k].groupby("rep"):
                z0 = gr["z_own"].to_numpy() - gr["z_own"].mean()
                ss = float((z0 ** 2).sum())
                if ss <= 0:
                    continue
                cl = pd.Series(z0).groupby(gr["float_id"].to_numpy()).sum().to_numpy()
                deffs.append(float((cl ** 2).sum()) / ss)
            ek = e[e["year"] == k]
            su = (ref[(ref["box"] == b) & (ref["year"] == k)]["B_unif"].std(ddof=1)
                  if ref is not None else ek["B_unif"].std(ddof=1))
            rows.append({"box": b, "year": int(k),
                         "median design effect": float(np.median(deffs)),
                         "sd B advective": ek["B"].std(ddof=1) * J_TO_GJ,
                         "sd B uniform": su * J_TO_GJ,
                         "ratio": ek["B"].std(ddof=1) / su})
    return pd.DataFrame(rows)


def plot_continuous_bias(est, box, ref=None, argo=None, figsize=(13, 3.0)):
    """Bias distribution over realizations, one panel per year; uniform null overlaid
    (from the large `ref` ensemble when given). If `argo` (a Series year -> B in
    J/m^2 from argo_bias_by_year) is given, the realized real-Argo bias is drawn as
    a green reference line, and the x-axis is widened to include it."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    g = est[est["box"] == box]
    gu = ref[ref["box"] == box] if ref is not None else g
    years = sorted(int(y) for y in g["year"].unique())
    allB = g["B"].to_numpy() * J_TO_GJ
    lo, hi = np.percentile(allB, [0.5, 99.5])
    pad = 0.08 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 28)
    xlo, xhi = lo - pad, hi + pad
    if argo is not None:
        av = argo * J_TO_GJ
        xlo, xhi = min(xlo, av.min() - pad), max(xhi, av.max() + pad)
    fig, axes = plt.subplots(1, len(years), figsize=figsize, sharex=True, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), years):
        gk = g[g["year"] == k]
        ax.hist(gk["B"] * J_TO_GJ, bins=bins, density=True, color="0.35")
        ax.hist(gu[gu["year"] == k]["B_unif"] * J_TO_GJ, bins=bins, density=True,
                histtype="step", color="tab:blue", lw=1.2, ls="--")
        ax.axvline(0.0, color="crimson", lw=1.6)
        ax.axvline(gk["B"].mean() * J_TO_GJ, color="k", lw=1.4, ls=":")
        if argo is not None and k in argo.index:
            ax.axvline(argo[k] * J_TO_GJ, color="tab:green", lw=1.8)
        ax.set_xlim(xlo, xhi)
        ax.set_title(str(k), fontsize=10)
        ax.set_xlabel("B (GJ/m2)", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("density")
    if argo is not None:
        np.atleast_1d(axes)[0].legend(
            handles=[Line2D([0], [0], color="tab:green", lw=1.8, label="real Argo"),
                     Line2D([0], [0], color="tab:blue", lw=1.2, ls="--", label="uniform"),
                     Line2D([0], [0], color="crimson", lw=1.6, label="zero")],
            fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_bias_intervals(est, ref, boxes=("boxB", "boxA"), figsize=(11, 4.0)):
    """Headline bias chart: per year, dodged intervals for the advective bias (from
    `est`, 100 realizations) and the uniform null (from `ref`, the large n-matched
    ensemble), one panel per box (own y-axis).

    For each (box, year, source): point = ensemble mean; capped bar = 95% CI
    (mean +/- 1.96 MC se), so an interval clearing zero flags a mean distinguishable
    from zero at the 5% level; thin whisker = 5-95% (the scatter of a single
    deployment). Red line at 0 is the target.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    sources = [(est, "B", "0.35", "advective", -0.15),
               (ref, "B_unif", "tab:blue", "uniform (P)", 0.15)]
    for ax, box in zip(np.atleast_1d(axes), boxes):
        years = sorted(int(y) for y in est.loc[est["box"] == box, "year"].unique())
        for src, col, color, _lab, dx in sources:
            gs = src[src["box"] == box]
            for y in years:
                b = gs.loc[gs["year"] == y, col].to_numpy() * J_TO_GJ
                m, se = b.mean(), b.std(ddof=1) / np.sqrt(b.size)
                q05, q95 = np.percentile(b, [5, 95])
                x = y + dx
                ax.plot([x, x], [q05, q95], color=color, lw=1.0, alpha=0.6, zorder=2)
                ax.errorbar(x, m, yerr=1.96 * se, fmt="o", color=color, ms=4.5,
                            elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=4)
        ax.axhline(0.0, color="crimson", lw=1.4, zorder=1)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_title(box, fontsize=10)
        ax.set_xlabel("analysis year", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("sampling bias B (GJ/m2)")
    handles = [Line2D([0], [0], color=c, lw=1.4, marker="o", label=l)
               for _, _, c, l, _ in sources]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_design_effect(d, boxes=("boxB", "boxA"), figsize=(10, 3.6)):
    """Chart of the design-effect table `d` (from continuous_deff_table): the
    effect by year (left) and the advective/uniform width ratio against the
    sqrt(deff) prediction (right), both boxes."""
    import matplotlib.pyplot as plt
    color = {"boxB": "tab:blue", "boxA": "tab:orange"}
    years = sorted(int(y) for y in d["year"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for b in boxes:
        g = d[d["box"] == b].sort_values("year")
        c = color.get(b)
        a1.plot(g["year"], g["median design effect"], "o-", color=c, label=b)
        a2.plot(g["year"], g["ratio"], "o-", color=c, label=b)
        a2.plot(g["year"], np.sqrt(g["median design effect"]), "--", color=c, alpha=0.55)
    a1.axhline(1.0, color="crimson", lw=1.2, ls=":")
    a1.set_ylabel("design effect  n / n_eff")
    a1.set_title("Design effect", fontsize=10)
    a1.set_ylim(bottom=0)
    a1.legend(fontsize=8)
    a2.axhline(1.0, color="crimson", lw=1.2, ls=":")
    a2.set_ylabel("sd ratio  advective / uniform")
    a2.set_title("Width ratio (solid) vs sqrt(deff) (dashed)", fontsize=10)
    a2.set_ylim(bottom=0)
    for ax in (a1, a2):
        ax.set_xlabel("analysis year")
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
    fig.tight_layout()
    return fig


def argo_comparison_table(uniform, boot, actual, real, boxes=("boxB", "boxA")):
    """Bias by box/year across the three deployment laws (GJ/m2): the uniform-seeded
    ensemble mean, the real-deployment bootstrap ensemble (mean and sd over reps),
    the single actual-deployment realization, and the real Argo array."""
    rows = []
    for b in boxes:
        u = uniform[uniform["box"] == b].groupby("year")["B"].mean()
        bt = boot[boot["box"] == b].groupby("year")["B"]
        ac = actual[actual["box"] == b].set_index("year")["B"]
        rr = real[b]
        for y in sorted(int(v) for v in u.index):
            rows.append({"box": b, "year": y,
                         "uniform": u[y] * J_TO_GJ,
                         "argo-boot mean": bt.mean().get(y, np.nan) * J_TO_GJ,
                         "argo-boot sd": bt.std().get(y, np.nan) * J_TO_GJ,
                         "argo-actual": ac.get(y, np.nan) * J_TO_GJ,
                         "real Argo": rr.get(y, np.nan) * J_TO_GJ})
    return pd.DataFrame(rows)


def plot_argo_comparison(uniform, boot, actual, real, boxes=("boxB", "boxA"), figsize=(11, 4.2)):
    """Deployment-law comparison, one panel per box: dodged 95% CI + 5-95% intervals
    for the uniform ensemble and the real-deployment bootstrap, with the single
    actual-deployment realization and the real Argo array overlaid as markers."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ens = [(uniform, "0.45", -0.18), (boot, "tab:purple", 0.18)]
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    for ax, box in zip(np.atleast_1d(axes), boxes):
        years = sorted(int(y) for y in uniform.loc[uniform["box"] == box, "year"].unique())
        for src, color, dx in ens:
            g = src[src["box"] == box]
            for y in years:
                b = g.loc[g["year"] == y, "B"].to_numpy() * J_TO_GJ
                if not len(b):
                    continue
                m, se = b.mean(), b.std(ddof=1) / np.sqrt(len(b))
                q05, q95 = np.percentile(b, [5, 95])
                ax.plot([y + dx] * 2, [q05, q95], color=color, lw=1.0, alpha=0.5, zorder=2)
                ax.errorbar(y + dx, m, yerr=1.96 * se, fmt="o", color=color, ms=4,
                            elinewidth=1.4, capsize=3, zorder=4)
        ga = actual[actual["box"] == box].set_index("year")["B"] * J_TO_GJ
        rr = real[box] * J_TO_GJ
        yy = [y for y in years if y in ga.index]
        ax.plot(yy, [ga[y] for y in yy], "D", mfc="none", mec="tab:orange", ms=6, zorder=5)
        yr = [y for y in years if y in rr.index]
        ax.plot(yr, [rr[y] for y in yr], "X", color="tab:green", ms=8, zorder=6)
        ax.axhline(0.0, color="crimson", lw=1.4, zorder=1)
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=8)
        ax.set_title(box, fontsize=10)
        ax.set_xlabel("analysis year", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("sampling bias B (GJ/m2)")
    handles = [Line2D([0], [0], color="0.45", lw=1.4, marker="o", label="uniform sim"),
               Line2D([0], [0], color="tab:purple", lw=1.4, marker="o", label="argo-bootstrap"),
               Line2D([0], [0], color="tab:orange", lw=0, marker="D", mfc="none", label="argo-actual (1 real.)"),
               Line2D([0], [0], color="tab:green", lw=0, marker="X", label="real Argo")]
    np.atleast_1d(axes)[0].legend(handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_continuous_stock(est, boxes=("boxB", "boxA"), figsize=(11, 3.6)):
    """Virtual (mean +/- sd over realizations) against real Argo standing stock."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(boxes), figsize=figsize)
    for ax, b in zip(np.atleast_1d(axes), boxes):
        g = est[est["box"] == b]
        yrs = sorted(int(y) for y in g["year"].unique())
        m = g.groupby("year")["standing_stock"].mean().reindex(yrs)
        s = g.groupby("year")["standing_stock"].std().reindex(yrs)
        real = argo_region_summary(b).set_index("year")["argo_stock"].reindex(yrs)
        ax.errorbar(yrs, m.to_numpy(), yerr=s.to_numpy(), fmt="o-", color="0.2",
                    capsize=3, label="virtual (mean +/- sd)")
        ax.plot(yrs, real.to_numpy(), "s--", color="tab:red", label="real Argo")
        ax.set_title(b, fontsize=10)
        ax.set_xlabel("year")
        ax.set_ylim(bottom=0)
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel("in-box stock (floats per 10-day epoch)")
    a0.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _profile_weights(box, tag):
    """Native-cell-tagged profiles, each carrying weight 1/(n R) of its (year, rep).

    Profiles are binned to the containing 1/12 deg cell and restricted to the wet
    analysis mask -- the same support the estimator's w and the field live on, so n
    is the on-mask count per realization. Summing wt over any set of (year, cell or
    doy) bins gives E[w] = mean_rep (fraction of that rep's profiles in the bin),
    since a bin a rep misses contributes 0 to the mean. Returns (prof, lat, lon, p)
    with p = A_c / sum(A) the native uniform area weight (NaN off-mask).
    """
    a = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"cell_area_{box}.nc"))["cell_area"]
    lat, lon, A = a["latitude"].values, a["longitude"].values, a.values
    prof = pd.read_csv(os.path.join(cfg.OUT_DIR, f"design_profiles_continuous{tag}.csv"),
                       usecols=["rep", "box", "year", "lat", "lon", "doy"])
    prof = prof[prof["box"] == box].copy()
    ii = cfg.containing_index(prof["lat"].to_numpy(), lat)
    jj = cfg.containing_index(prof["lon"].to_numpy(), lon)
    wet = (ii >= 0) & (jj >= 0)
    wet[wet] = np.isfinite(A[ii[wet], jj[wet]])
    prof = prof.loc[wet].assign(ilat=ii[wet], ilon=jj[wet])
    n = prof.groupby(["year", "rep"])["lat"].transform("size")
    R = prof.groupby("year")["rep"].transform("nunique")
    prof["wt"] = 1.0 / (n * R)
    return prof, lat, lon, A / np.nansum(A)


def sampling_weight_space(box, tag="_pilot"):
    """E[w] spatial marginal and the uniform area reference p, on the NATIVE grid.

    No coarsening: the 1/12 deg cells the residual field and the inner product
    B = <Z, w - p> live on. Returns (Ew, lat, lon, p): Ew a Series indexed by
    (year, ilat, ilon) summing to 1 per year; p the (nlat, nlon) area weight.
    """
    prof, lat, lon, p = _profile_weights(box, tag)
    Ew = prof.groupby(["year", "ilat", "ilon"])["wt"].sum()
    return Ew, lat, lon, p


def sampling_weight_time(box, tag="_pilot"):
    """E[w] day-of-year marginal (native daily) and the uniform reference 1/365.

    Returns (Ew, p): Ew indexed by (year, doy) summing to 1 per year; p = 1/365.
    """
    prof, *_ = _profile_weights(box, tag)
    Ew = prof.groupby(["year", "doy"])["wt"].sum()
    return Ew, 1.0 / cfg.DAYS_PER_YEAR


def bias_decomposition_table(tag="_pilot", boxes=("boxB", "boxA")):
    """Space / time / interaction split of the measured bias B = <Z, w - p>.

    B_space = <w^sp - p^sp, Zc> pairs the spatial weight departure with the
    time-mean field Zc (time held uniform); B_time = <w^t - 1/365, Zd> pairs the
    day-of-year weight departure with the P-spatial-mean field Zd (space held
    uniform); the interaction is the remainder B - B_space - B_time. All exact and
    estimable from E[w]; the total is the ensemble-mean measured bias. GJ/m2.
    """
    est = load_continuous(tag)
    rows = []
    for b in boxes:
        Ew, lat, lon, p = sampling_weight_space(b, tag)
        wet = np.isfinite(p)
        pflat = p[wet]
        nlon = p.shape[1]
        wi, wj = np.where(wet)
        widx = wi * nlon + wj
        z = xr.open_dataset(os.path.join(cfg.OUT_DIR, f"residual_field_{b}.nc"))["z"]
        Ewt, pt = sampling_weight_time(b, tag)
        meas = est[est["box"] == b].groupby("year")["B"].mean()
        for y in sorted(int(v) for v in meas.index):
            w = Ew.xs(y, level="year")
            wsp = np.zeros(len(pflat))
            f = (w.index.get_level_values("ilat").to_numpy() * nlon
                 + w.index.get_level_values("ilon").to_numpy())
            wsp[np.searchsorted(widx, f)] = w.to_numpy()
            zy = z.sel(year=y).values
            Zc = zy.mean(axis=0)[wi, wj]
            b_space = float((wsp - pflat) @ Zc)
            Zday = (zy[:, wi, wj] * pflat).sum(axis=1)
            wt = Ewt.xs(y, level="year").reindex(range(cfg.DAYS_PER_YEAR), fill_value=0.0).to_numpy()
            b_time = float(np.sum((wt - pt) * Zday))
            m = float(meas[y])
            rows.append({"box": b, "year": int(y),
                         "B (measured)": m * J_TO_GJ,
                         "B space": b_space * J_TO_GJ,
                         "B time": b_time * J_TO_GJ,
                         "interaction": (m - b_space - b_time) * J_TO_GJ})
    return pd.DataFrame(rows)


def weight_dispersion_table(tag="_pilot", boxes=("boxB", "boxA")):
    """Spatial weight dispersion sd_P(w/p) against the field dispersion sigma_k.

    sd_P(w/p) is the spread of the (spatial-marginal) sampling density ratio -- the
    multiplier in the Cauchy-Schwarz cap |B| <= sd_P(r) sigma_k. Reported per box,
    year-averaged for the weight (it barely varies by year) and per year for
    sigma_k, both with the product as a distribution-free scale for the bias.
    """
    rows = []
    for b in boxes:
        ratio = sampling_weight_ratio(b, tag)
        Ew, lat, lon, p = sampling_weight_space(b, tag)
        pflat = p[np.isfinite(p)]
        sd = _sd_Z(b) * J_TO_GJ
        for y in sorted(ratio):
            r = ratio[y]
            sdr = float(np.sqrt(np.sum(pflat * r * r) - 1.0))
            rows.append({"box": b, "year": y, "sd_P(w/p)": sdr,
                         "sigma_k (GJ/m2)": float(sd[y]),
                         "cap = sd_P(w/p) sigma_k": sdr * float(sd[y])})
    return pd.DataFrame(rows)


def plot_sampling_weight_space(box, tag="_pilot", years=None, figsize=None):
    """Over-/under-sampling map, log2(E[w]/uniform), on the native 1/12 deg grid.

    One panel per year, shared symmetric colour scale from the 99th percentile of
    |log2 ratio|. Empty native cells (very few at 100 realizations) are left blank.
    """
    import matplotlib.pyplot as plt
    Ew, lat, lon, p = sampling_weight_space(box, tag)
    years = years or sorted(int(y) for y in Ew.index.get_level_values("year").unique())
    logp = np.where(p > 0, np.log2(np.where(p > 0, p, np.nan)), np.nan)

    grids = []
    for y in years:
        w = Ew.xs(y, level="year")
        i = w.index.get_level_values("ilat").to_numpy()
        j = w.index.get_level_values("ilon").to_numpy()
        v = w.to_numpy()
        G = np.full(p.shape, np.nan)
        pos = v > 0
        G[i[pos], j[pos]] = np.log2(v[pos]) - logp[i[pos], j[pos]]
        grids.append(G)
    finite = np.concatenate([g[np.isfinite(g)] for g in grids])
    vmax = float(np.percentile(np.abs(finite), 99))

    figsize = figsize or (2.5 * len(years) + 1.6, 3.2)
    fig, axes = plt.subplots(1, len(years), figsize=figsize, sharey=True)
    for ax, y, G in zip(np.atleast_1d(axes), years, grids):
        pm = ax.pcolormesh(lon, lat, G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.set_title(str(y), fontsize=10)
        ax.set_xlabel("lon", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("lat")
    fig.colorbar(pm, ax=list(np.atleast_1d(axes)), shrink=0.85,
                 label="log2( E[w] / uniform )")
    return fig


def plot_sampling_weight_space_combined(boxes=("boxB", "boxA"), tag="_pilot",
                                        years=None, figsize=None):
    """Over-/under-sampling map log2(E[w]/uniform), both boxes on one both-box base
    map per year and one shared symmetric scale, so the two arrays' spatial
    preference contrasts directly. Native 1/12 deg grid; red over-, blue under-sampled.
    """
    from matplotlib.colors import TwoSlopeNorm

    data = {}
    for b in boxes:
        Ew, lat, lon, p = sampling_weight_space(b, tag)
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, np.nan)), np.nan)
        yrs = years or sorted(int(y) for y in Ew.index.get_level_values("year").unique())
        per = {}
        for y in yrs:
            w = Ew.xs(y, level="year")
            i = w.index.get_level_values("ilat").to_numpy()
            j = w.index.get_level_values("ilon").to_numpy()
            v = w.to_numpy()
            G = np.full(p.shape, np.nan)
            pos = v > 0
            G[i[pos], j[pos]] = np.log2(v[pos]) - logp[i[pos], j[pos]]
            per[y] = xr.DataArray(G, dims=("latitude", "longitude"),
                                  coords={"latitude": lat, "longitude": lon})
        data[b] = per
    years = years or sorted(next(iter(data.values())).keys())
    finite = np.concatenate([data[b][y].values[np.isfinite(data[b][y].values)]
                             for b in boxes for y in years])
    vmax = float(np.percentile(np.abs(finite), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    ext = _both_box_extent(boxes)

    figsize = figsize or (3.0 * len(years) + 1.5, 4.6)
    fig, axes, caxes = _map_grid(1, len(years), figsize)
    for c, y in enumerate(years):
        im = _draw_domain_map(axes[0][c], [data[b][y] for b in boxes], norm, "RdBu_r",
                              ext, str(y), True, c == 0, scale=1.0, boxes=boxes)
    fig.colorbar(im, cax=caxes[0], label="log2( E[w] / uniform )")
    return fig


def sampling_weight_ratio(box, tag="_pilot"):
    """Per-year array of E[w]_c / p_c over ALL wet native cells (empty cells -> 0).

    The over-/under-sampling factor per cell: 1 is the uniform weight, >1
    over-sampled, <1 under-sampled. Returns {year: ndarray over wet cells}.
    """
    Ew, lat, lon, p = sampling_weight_space(box, tag)
    wi, wj = np.where(np.isfinite(p))
    wflat, pw = wi * p.shape[1] + wj, p[wi, wj]
    out = {}
    for y in sorted(int(y) for y in Ew.index.get_level_values("year").unique()):
        w = Ew.xs(y, level="year")
        f = (w.index.get_level_values("ilat").to_numpy() * p.shape[1]
             + w.index.get_level_values("ilon").to_numpy())
        r = np.zeros(len(wflat))
        pos = np.searchsorted(wflat, f)
        r[pos] = w.to_numpy() / pw[pos]
        out[y] = r
    return out


def plot_sampling_weight_hist(box, tag="_pilot", years=None, figsize=None):
    """Distribution of the per-cell sampling weight E[w]/uniform, one panel per year.

    A histogram over native wet cells; the red line at 1 is the uniform weight.
    Concentration near 1 means near-uniform sampling; a right tail is over-sampled
    cells, mass below 1 under-sampled. All panels share bins set from the pooled
    99.5th percentile.
    """
    import matplotlib.pyplot as plt
    ratio = sampling_weight_ratio(box, tag)
    years = years or sorted(ratio)
    hi = np.max(np.concatenate([ratio[y] for y in years]))
    bins = np.linspace(0, hi, 40)
    fig, axes = plt.subplots(1, len(years), figsize=figsize or (2.4 * len(years) + 1.0, 3.0),
                             sharex=True, sharey=True)
    for ax, y in zip(np.atleast_1d(axes), years):
        r = ratio[y]
        ax.hist(r, bins=bins, density=True, color="0.4")
        ax.axvline(1.0, color="crimson", lw=1.6)
        ax.axvline(r.mean(), color="k", lw=1.2, ls=":")
        ax.set_title(str(y), fontsize=10)
        ax.set_xlabel("E[w] / uniform", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("density (native cells)")
    fig.tight_layout()
    return fig


def plot_sampling_weight_time(box, tag="_pilot", years=None, figsize=(8.0, 3.4)):
    """Seasonal sampling weight E[w^t] / uniform vs day-of-year (daily), per year."""
    import matplotlib.pyplot as plt
    Ew, p = sampling_weight_time(box, tag)
    years = years or sorted(int(y) for y in Ew.index.get_level_values("year").unique())
    doy = np.arange(cfg.DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=figsize)
    for y in years:
        e = Ew.xs(y, level="year").reindex(doy, fill_value=0.0).to_numpy()
        ax.plot(doy, e / p, lw=1.0, label=str(y))
    ax.axhline(1.0, color="crimson", lw=1.4, ls="--")
    ax.set_xlabel("day of year")
    ax.set_ylabel("E[w] / uniform")
    ax.set_title(f"{box}: seasonal sampling weight", fontsize=10)
    ax.legend(fontsize=7, ncol=5)
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
