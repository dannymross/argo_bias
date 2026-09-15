"""Full-footprint GLORYS12V1 park-depth velocity download for the Lagrangian OSSE.

Daily uo/vo at the two levels bracketing the 1000 m park depth (900-1100 m ->
902 m, 1062 m), over the full region the real box-sampling floats roam across
2010-2019 -- so simulated floats are not deleted for leaving the velocity domain
(the old 15-50 N, 70-30 W domain lost 112 of 232 such floats). Files are one per
month, deflate-compressed int16 (native packing preserved; OceanParcels reads
them directly, no decompression step).

Region (from the 2010-2019 footprint of every float profiling Box A/B in
2015-2019, rounded outward): lat 11-66 N, lon 87-12 W.

Run from the repo root, under the argo_bias env:
    # one-month timing test to a scratch dir (does not touch the live files)
    python code/osse/download_velocity_full.py --test --test-dir <dir>
    # full pull (writes velocity_daily_park_YYYY_MM.nc in the live dir)
    python code/osse/download_velocity_full.py --start 2010-01 --end 2019-12
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osse import osse_config as cfg
from osse import download_glorys as dg

FULL_LON = (-87.0, -12.0)
FULL_LAT = (11.0, 66.0)
PARK_DEPTH_BOUNDS = (900.0, 1100.0)
OUT_DIR = os.path.join(cfg.GLORYS_DIR, "velocity_daily")
PREFIX = "velocity_daily_park"


def download(start_month, end_month, out_dir=OUT_DIR, prefix=PREFIX,
             compression_level=4, force=False):
    t0 = time.time()
    paths, glob_pattern = dg.fetch_monthly(
        out_dir=out_dir,
        file_prefix=prefix,
        dataset_id=dg.DAILY_ID,
        variables=("uo", "vo"),
        lon_bounds=FULL_LON,
        lat_bounds=FULL_LAT,
        start_month=start_month,
        end_month=end_month,
        depth_bounds=PARK_DEPTH_BOUNDS,
        compression_level=compression_level,
        force=force,
    )
    dt = time.time() - t0
    got = [p for p in paths if os.path.exists(p)]
    total_mb = sum(os.path.getsize(p) for p in got) / 1e6
    n = len(got)
    print(f"\n{n} files, {total_mb:.1f} MB, {dt:.1f} s "
          f"({dt / max(n, 1):.1f} s/file, {total_mb / max(n, 1):.1f} MB/file)")
    if n:
        print(f"extrapolated 120 months: {total_mb / n * 120 / 1000:.1f} GB, "
              f"{dt / n * 120 / 60:.1f} min")
    return paths, glob_pattern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01")
    ap.add_argument("--end", default="2019-12")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--compression-level", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="one-month (2015-01) timing test to --test-dir")
    ap.add_argument("--test-dir", default=None)
    a = ap.parse_args()
    if a.test:
        out = a.test_dir or os.path.join(cfg.GLORYS_DIR, "_velocity_full_test")
        download("2015-01", "2015-01", out_dir=out, prefix=a.prefix,
                 compression_level=a.compression_level, force=a.force)
    else:
        download(a.start, a.end, out_dir=a.out_dir, prefix=a.prefix,
                 compression_level=a.compression_level, force=a.force)


if __name__ == "__main__":
    main()
