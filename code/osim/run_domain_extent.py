"""Extended regions of the deep domain enclosing shares of the lifetime profiles of floats that profiled it."""
import os, sys, time
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from osim import gyre_mask as gm, osim_config as cfg

PCT = (90, 95, 99, 99.9)
D = os.path.dirname(cfg.GLORYS_DIR)
t0 = time.time()
b = xr.open_dataset(os.path.join(cfg.GLORYS_DIR, "static_natl", "deptho_wide.nc")).sortby("latitude")
la, lo, z = b.latitude.values.astype(float), b.longitude.values.astype(float), b.deptho.squeeze().values
m = gm.deep_domain(z, la, lo)
print("domain km2", gm.cell_area(la, lo)[m].sum() / 1e6, flush=True)
km, src = gm.water_paths(z, la, lo, m)
print("paths", round(time.time() - t0), "s", flush=True)
ids = pd.read_csv(os.path.join(D, "argo_meta", "launch_domain_clean.csv")).float_id
r = pd.read_csv(os.path.join(D, "argo_ohc.csv"), usecols=["float_id", "lat", "lon"]).dropna()
r = r[r.float_id.isin(ids)]
out, Q, ins = gm.local_reach(km, src, la, lo, m, r.lat.to_numpy(), r.lon.to_numpy(), PCT)
ds = xr.Dataset({"domain": (("latitude", "longitude"), m.astype("i1")), "km": (("latitude", "longitude"), km.astype("f4")),
                 **{f"ext_{p:g}": (("latitude", "longitude"), out[p].astype("i1")) for p in PCT}},
                coords=dict(latitude=la, longitude=lo), attrs=dict(inside=ins, n_records=len(r), window_km=300.0))
ds.to_netcdf(os.path.join(cfg.OUT_DIR, "domain_extent.nc"))
A = gm.cell_area(la, lo)
frame = gm.inside(dict(lat=(0, 70), lon=(-90, 5)), la, lo)
for p in PCT:
    e = out[p]; x = e & ~m; i, j = np.where(e)
    print(f"{p}: q {Q[p]:.3f}; added {A[x].sum() / 1e6:.2f}e6 km2; lat {la[i.min()]:.1f}..{la[i.max()]:.1f}, lon {lo[j.min()]:.1f}..{lo[j.max()]:.1f}; "
          f"share of added in 0-70,90W-5E {A[x & frame].sum() / max(A[x].sum(), 1):.3f}; max km {km[x].max() if x.any() else 0:.0f}", flush=True)
print("inside", round(ins, 4), "records", len(r), "total s", round(time.time() - t0))
