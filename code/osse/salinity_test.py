"""Does 0-2000 m OHC need salinity, or is constant rho*cp adequate? (osse.tex sec:glorys)

Downloads a short window of daily thetao + so over one box, then computes OHC two
ways per column and reports the percent difference:
  baseline : constant CP*RHO * potential-temperature   (code/ohc.py convention)
  in-situ  : TEOS-10, integrand = rho(SA,CT,p) * cp0 * CT

If the box-mean |difference| is ~1% as osse.tex predicts, the full pull omits so.

    python code/osse/salinity_test.py [config.json] [--stride D]

Requires gsw (TEOS-10). Downloads to <out_root>/_salinity_test/.
"""

import argparse
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_glorys as dg  # noqa: E402

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
CP = 3989.411  # J / (kg K)
CP0 = 3991.86795711963
RHO = 1028.319  # kg / m^3

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _integrate_column(integrand, depth, zmax):
    """Depth-integrate a pre-multiplied integrand [W... , z] over [0, zmax] (J/m2).

    Same quadrature as ohc.profile_ohc (surface slab + native trapezoid + linear
    interpolation to zmax) but on the integrand directly, so baseline and in-situ
    share identical quadrature and only the integrand differs.
    """
    depth = np.asarray(depth, float)
    in_range = depth <= zmax
    d_in = depth[in_range]
    f_in = integrand[..., in_range]

    integ = _trapz(f_in, d_in, axis=-1)
    if d_in[0] > 0:
        integ = integ + f_in[..., 0] * d_in[0]
    n_in = int(in_range.sum())
    if n_in < depth.size and d_in[-1] < zmax:
        d_lo, d_hi = d_in[-1], depth[n_in]
        f_lo, f_hi = f_in[..., -1], integrand[..., n_in]
        f_z = f_lo + (f_hi - f_lo) * (zmax - d_lo) / (d_hi - d_lo)
        integ = integ + 0.5 * (f_lo + f_z) * (zmax - d_lo)
    return integ


def run(config_path, stride):
    import gsw

    with open(config_path) as f:
        cfg = json.load(f)
    st = cfg["salinity_test"]
    out_dir = os.path.join(cfg["out_root"], "_salinity_test")

    _, glob = dg.fetch_monthly(
        out_dir, "saltest", cfg["groups"][1]["dataset_id"], st["variables"],
        st["region"]["lon_bounds"], st["region"]["lat_bounds"],
        st["start_month"], st["end_month"], depth_bounds=st["depth"])

    ds = xr.open_mfdataset(glob, combine="by_coords").isel(time=slice(None, None, stride))
    ds = ds.transpose("time", "latitude", "longitude", "depth")
    pt = ds["thetao"].values          # potential temperature (degC)
    sp = ds["so"].values              # practical salinity
    z = ds["depth"].values            # m, positive down
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # Pressure (dbar) from depth and latitude; broadcast to the field shape.
    p = gsw.p_from_z(-z[None, None, None, :], lat[None, :, None, None])
    p = np.broadcast_to(p, pt.shape)
    lon_b = np.broadcast_to(lon[None, None, :, None], pt.shape)
    lat_b = np.broadcast_to(lat[None, :, None, None], pt.shape)

    sa = gsw.SA_from_SP(sp, p, lon_b, lat_b)
    ct = gsw.CT_from_pt(sa, pt)
    rho = gsw.rho(sa, ct, p)

    integ_base = CP * RHO * pt
    integ_insitu = rho * CP0 * ct

    zmax = 2000.0
    ohc_base = _integrate_column(integ_base, z, zmax)
    ohc_insitu = _integrate_column(integ_insitu, z, zmax)

    diff = ohc_insitu - ohc_base
    pct = 100.0 * diff / ohc_base
    valid = np.isfinite(pct)
    pct_v = pct[valid]

    print("\n" + "=" * 66)
    print(f"SALINITY / in-situ density test  ({st['start_month']}..{st['end_month']}, "
          f"stride {stride}, {valid.sum()} columns)")
    print(f"box lon {st['region']['lon_bounds']} lat {st['region']['lat_bounds']}")
    print("=" * 66)
    print(f"  baseline OHC (const rho*cp) mean : {np.nanmean(ohc_base):.4e} J/m2")
    print(f"  in-situ  OHC (TEOS-10)      mean : {np.nanmean(ohc_insitu):.4e} J/m2")
    print(f"  percent difference (in-situ vs baseline):")
    print(f"    mean   {pct_v.mean():+.3f} %")
    print(f"    median {np.median(pct_v):+.3f} %")
    print(f"    p5..p95 [{np.percentile(pct_v, 5):+.3f}, {np.percentile(pct_v, 95):+.3f}] %")
    print(f"    max|.| {np.abs(pct_v).max():.3f} %")
    print("=" * 66)
    print("  -> if mean |diff| ~1% and roughly spatially uniform, omit `so` and")
    print("     use constant rho*cp for the first-pass OHC field.")
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "configs", "osse_glorys.json"))
    ap.add_argument("--stride", type=int, default=7, help="time subsample stride (days)")
    args = ap.parse_args()
    run(args.config, args.stride)
