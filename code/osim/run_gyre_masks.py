"""Fixed gyre masks from mean GLORYS zos: writes data/osim/gyre_masks.nc, gyre_masks.txt and gyre_masks.png."""
import glob, os, sys
import numpy as np, pandas as pd, xarray as xr, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from osim import gyre_mask as gm, osim_config as cfg

Y0, Y1 = 1993, 2009
OUT = os.path.join(cfg.OUT_DIR, "gyre_masks")
zos = xr.open_mfdataset(sorted(glob.glob(os.path.join(cfg.GLORYS_DIR, "zos_monthly", "zos_monthly_natl_*.nc")))).zos.load()
zos = zos.sortby("latitude")
la, lo = zos.latitude.values.astype(float), zos.longitude.values.astype(float)
A = gm.cell_area(la, lo)
BOX = {k: dict(lat=v["lat_bounds"], lon=v["lon_bounds"]) for k, v in cfg.BOXES.items() if k in ("boxA", "boxB")}
L = [f"years on disk {int(zos.time.dt.year.min())}-{int(zos.time.dt.year.max())}, {zos.sizes['time']} of 396 months, grid {zos.shape}"]


def inside(b, la=la, lo=lo):
    return (la[:, None] >= b["lat"][0]) & (la[:, None] <= b["lat"][1]) & (lo[None] >= b["lon"][0]) & (lo[None] <= b["lon"][1])


def desc(m):
    i, j = np.where(m)
    return (f"area {A[m].sum() / 1e6:.2f}e6 km2, lat {la[i.min()]:.1f}..{la[i.max()]:.1f}, lon {lo[j.min()]:.1f}..{lo[j.max()]:.1f}, "
            f"in velocity domain {A[m & inside(gm.VEL)].sum() / A[m].sum():.0%}, "
            + ", ".join(f"{k} in mask {A[m & inside(b) & np.isfinite(h)].sum() / A[inside(b) & np.isfinite(h)].sum():.0%}" for k, b in BOX.items()))


def row(group, name, gyre, level, m, ref=None, lat=None, lon=None, area=None):
    la_, lo_, a_ = (la, lo, A) if lat is None else (lat, lon, area)
    i, j = np.where(m)
    return dict(group=group, name=name, gyre=gyre, level=level, area_km2=a_[m].sum(), lat_min=la_[i.min()], lat_max=la_[i.max()],
                lon_min=lo_[j.min()], lon_max=lo_[j.max()], frac_vel=a_[m & inside(gm.VEL, la_, lo_)].sum() / a_[m].sum(),
                box_a=a_[m & inside(BOX["boxA"], la_, lo_)].sum() / A[inside(BOX["boxA"]) & np.isfinite(h)].sum(),
                box_b=a_[m & inside(BOX["boxB"], la_, lo_)].sum() / A[inside(BOX["boxB"]) & np.isfinite(h)].sum(),
                iou=np.nan if ref is None else iou(m, ref))


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def period(a, b):
    return zos.sel(time=slice(str(a), str(b))).mean("time").values.astype(float)


h = period(Y0, Y1)
c_s, plain, leak = gm.spill(h, la, lo, 1, gm.STG["seed"])
c_j, stg = gm.stg_jet(h, la, lo)
c_p, spg = gm.spg_closed(h, la, lo)
L += [f"PRIMARY period {Y0}-{Y1}", f"STG plain spill level {c_s:.3f}: {desc(plain)}", f"STG jet rule level {c_j:.3f}: {desc(stg)}",
      f"SPG closed level {c_p:.3f}: {desc(spg)}", f"overlap STG/SPG cells {(stg & spg).sum()}"]
R = [row("rule", "plain closed contour", "STG", c_s, plain), row("rule", "jet and trough", "STG", c_j, stg),
     row("rule", "closed low", "SPG", c_p, spg)]
for d in (-.10, -.05, .05, .10):
    m = gm.stg_jet(h, la, lo, c_j + d)[1]
    R.append(row("level", f"{d:+.2f}", "STG", c_j + d, m, stg))
    L.append(f"STG level {d:+.2f}: {desc(m)}; IoU with primary {iou(m, stg):.2f}")
for a, b in ((1993, 2003), (2004, 2014), (2015, 2025), (1993, 2025)):
    if a <= int(zos.time.dt.year.max()) and b <= int(zos.time.dt.year.max()):
        hp = period(a, b)
        cp, m = gm.stg_jet(hp, la, lo)
        cq, q = gm.spg_closed(hp, la, lo)
        R += [row("period", f"{a}-{b}", "STG", cp, m, stg), row("period", f"{a}-{b}", "SPG", cq, q, spg)]
        L.append(f"period {a}-{b}: STG level {cp:.3f} IoU {iou(m, stg):.2f}; SPG level {cq:.3f} IoU {iou(q, spg):.2f}")
md = xr.open_dataset(os.path.join(os.path.dirname(cfg.GLORYS_DIR), "sat", "mdt_natl.nc")).mdt.isel(time=0).sortby("latitude")
hm = md.interp(latitude=xr.DataArray(la[la <= float(md.latitude.max())], dims="latitude"),
               longitude=xr.DataArray(lo[lo <= float(md.longitude.max())], dims="longitude"), method="nearest").values.astype(float)
lam, lom = la[la <= float(md.latitude.max())], lo[lo <= float(md.longitude.max())]
cm, mm = gm.stg_jet(hm, lam, lom)
R.append(row("mdt", "CNES-CLS22 MDT", "STG", cm, mm, stg[:len(lam), :][:, :len(lom)], lam, lom, A[:len(lam), :][:, :len(lom)]))
L.append(f"MDT STG level {cm:.3f}: IoU with GLORYS mask {iou(mm, stg[:len(lam), :][:, :len(lom)]):.2f}")
ds = xr.Dataset(dict(h=(("latitude", "longitude"), h), stg=(("latitude", "longitude"), stg), spg=(("latitude", "longitude"), spg),
                     stg_plain=(("latitude", "longitude"), plain)), coords=dict(latitude=la, longitude=lo),
                attrs=dict(period=f"{Y0}-{Y1}", stg_level=c_j, spg_level=c_p, plain_level=c_s))
ds.assign({k: ds[k].astype("i1") for k in ("stg", "spg", "stg_plain")}).to_netcdf(OUT + ".nc")

gm.plot_masks(ds, BOX).savefig(OUT + ".png", dpi=170)
pd.DataFrame(R).to_csv(OUT + ".csv", index=False)
open(OUT + ".txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
