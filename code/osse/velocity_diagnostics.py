"""Per-year velocity-field statistics, for assessing design invariance (D1b).

The permutation test assumes the cell-weight vectors are identically
distributed across velocity years, w^(l) =d= w^(l'). GLORYS assimilates Argo and
the array densified through the 2000s, so the reanalysis mesoscale field may
itself trend with the observing system -- which would break D1b rather than
merely weaken power. These are the Eulerian drivers of the sampling design:
if the mean flow, eddy energy, or eddy persistence trend across years, the
trajectories cannot be exchangeable.

Moments are accumulated month by month (sum, sum of squares, lagged products)
so a year never has to be held in memory at once.

Run from the repo root:
    python code/osse/velocity_diagnostics.py [--band park|surf] [--box boxB]
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

VEL_DIR = os.path.join(cfg.GLORYS_DIR, "velocity_daily")
PARK_DEPTH = 1000.0
LAGS = (1, 2, 5, 10, 20)


def _open_month(path, band, box):
    ds = cfg.box_slice(xr.open_dataset(path), box)
    if band == "park":
        # Linear interpolation to the nominal park depth, rather than picking
        # one of the two bracketing levels.
        ds = ds.interp(depth=PARK_DEPTH)
    else:
        ds = ds.isel(depth=0)
    return ds["uo"].values.astype(np.float64), ds["vo"].values.astype(np.float64)


def year_moments(year, band, box):
    """Accumulate first/second moments and lagged products over one year."""
    files = sorted(glob.glob(os.path.join(VEL_DIR, f"velocity_daily_{band}_{year}_*.nc")))
    if len(files) != 12:
        return None

    s1 = s2 = None
    lag_sum = {k: None for k in LAGS}
    lag_n = {k: 0 for k in LAGS}
    buf = []  # trailing days, for lagged products across month boundaries
    n = 0

    for f in files:
        u, v = _open_month(f, band, box)
        stack = np.stack([u, v])                      # (2, t, lat, lon)
        s1 = stack.sum(1) if s1 is None else s1 + stack.sum(1)
        s2 = (stack**2).sum(1) if s2 is None else s2 + (stack**2).sum(1)
        n += stack.shape[1]

        seq = buf + [stack[:, t] for t in range(stack.shape[1])]
        for k in LAGS:
            if len(seq) > k:
                a = np.stack(seq[:-k]); b = np.stack(seq[k:])
                prod = (a * b).sum(0)
                lag_sum[k] = prod if lag_sum[k] is None else lag_sum[k] + prod
                lag_n[k] += a.shape[0]
        buf = seq[-max(LAGS):]

    mean = s1 / n
    var = s2 / n - mean**2                            # (2, lat, lon), per component
    eke = 0.5 * var.sum(0)
    mke = 0.5 * (mean**2).sum(0)

    # Lagged autocovariance of the eddy part, pooled over u and v.
    acf = {}
    for k in LAGS:
        cov = lag_sum[k] / lag_n[k] - mean**2         # E[x_t x_{t+k}] - mean^2
        acf[k] = float(np.nanmean(cov.sum(0)) / np.nanmean(var.sum(0)))

    return {"mean": mean, "eke": eke, "mke": mke, "n": n, "acf": acf}


def integral_timescale(acf):
    """Trapezoidal integral of the Eulerian ACF out to its first zero crossing."""
    lags = np.array([0] + list(LAGS), dtype=float)
    r = np.array([1.0] + [acf[k] for k in LAGS])
    neg = np.where(r <= 0)[0]
    if len(neg):
        cut = neg[0]
        if cut == 0:
            return 0.0
        # linear crossing between the last positive and first non-positive lag
        x0, x1, y0, y1 = lags[cut - 1], lags[cut], r[cut - 1], r[cut]
        zero = x0 + y0 * (x1 - x0) / (y0 - y1)
        lags, r = np.append(lags[:cut], zero), np.append(r[:cut], 0.0)
    return float(np.trapezoid(r, lags))


def jet_axis(mean, lat, lon):
    """Latitude of maximum mean speed at each longitude (Gulf Stream axis proxy)."""
    speed = np.hypot(mean[0], mean[1])
    with np.errstate(invalid="ignore"):
        idx = np.nanargmax(np.where(np.isfinite(speed), speed, -np.inf), axis=0)
    return lat[idx]


def main(band="park", box="boxB", years=None):
    lat = lon = None
    rows, maps = [], {}
    years = years or sorted({int(os.path.basename(f).split("_")[-2])
                             for f in glob.glob(os.path.join(VEL_DIR, f"velocity_daily_{band}_*.nc"))})

    for y in years:
        m = year_moments(y, band, box)
        if m is None:
            print(f"  {y}: incomplete (skipped)")
            continue
        ds = cfg.box_slice(xr.open_dataset(
            sorted(glob.glob(os.path.join(VEL_DIR, f"velocity_daily_{band}_{y}_*.nc")))[0]), box)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        w = np.cos(np.deg2rad(lat))[:, None] * np.isfinite(m["eke"])

        axis = jet_axis(m["mean"], lat, lon)
        rows.append({
            "year": y, "band": band, "box": box, "n_days": m["n"],
            "EKE": np.nansum(m["eke"] * w) / w.sum(),
            "MKE": np.nansum(m["mke"] * w) / w.sum(),
            "EKE_over_MKE": np.nansum(m["eke"] * w) / np.nansum(m["mke"] * w),
            "mean_speed": np.nansum(np.hypot(*m["mean"]) * w) / w.sum(),
            "eke_p90": float(np.nanpercentile(m["eke"], 90)),
            "T_int_days": integral_timescale(m["acf"]),
            "jet_lat_mean": float(np.nanmean(axis)),
            **{f"acf_lag{k}": m["acf"][k] for k in LAGS},
        })
        maps[y] = m["eke"]
        print(f"  {y}: EKE={rows[-1]['EKE']:.4f} m2/s2  T_int={rows[-1]['T_int_days']:.2f} d  "
              f"jet_lat={rows[-1]['jet_lat_mean']:.2f}")

    if not rows:
        raise SystemExit("no complete years found")

    df = pd.DataFrame(rows)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(cfg.OUT_DIR, f"velocity_stats_{band}_{box}.csv"), index=False)

    eke = xr.DataArray(np.stack([maps[y] for y in df["year"]]),
                       dims=("year", "latitude", "longitude"),
                       coords={"year": df["year"].values, "latitude": lat, "longitude": lon},
                       name="eke")
    eke.attrs.update(units="m2 s-2", long_name=f"eddy kinetic energy, {band} band")
    eke.to_dataset().to_netcdf(os.path.join(cfg.OUT_DIR, f"eke_maps_{band}_{box}.nc"))

    # Spatial pattern stability: correlation of each year's EKE map with the rest.
    flat = eke.values.reshape(len(df), -1)
    ok = np.isfinite(flat).all(0)
    C = np.corrcoef(flat[:, ok])
    off = C[np.triu_indices(len(df), 1)]
    print(f"\nEKE spatial-pattern correlation across years: "
          f"mean {off.mean():.3f}, min {off.min():.3f}")
    print(f"EKE across years: mean {df['EKE'].mean():.4f}, "
          f"cv {df['EKE'].std()/df['EKE'].mean():.3f}, range "
          f"[{df['EKE'].min():.4f}, {df['EKE'].max():.4f}]")
    if len(df) >= 4:
        r = np.corrcoef(df["year"], df["EKE"])[0, 1]
        print(f"EKE vs year (trend, the D1b threat): r = {r:+.3f}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="park", choices=("park", "surf"))
    ap.add_argument("--box", default="boxB")
    ap.add_argument("--years", help="comma-separated subset")
    a = ap.parse_args()
    main(a.band, a.box, [int(y) for y in a.years.split(",")] if a.years else None)
