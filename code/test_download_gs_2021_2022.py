"""Time a single-month download of the 2021-2022 box (lat 33-45, lon -74,-49)
before kicking off the full 24-month run in download_gs_2021_2022.py.

Downloads just 2021-01 into the same data/velocity_gs_2021_2022/ output dir,
so this isn't wasted work: the full script will see the file already exists
and skip re-downloading it.

Run from the repo root:
    python code/test_download_gs_2021_2022.py
"""

import time

from trajsim import fetch_velocity_data

LON_BOUNDS = (-74, -49)
LAT_BOUNDS = (33, 45)
DEPTH_BOUNDS = (0, 2300)
OUT_DIR = "data/velocity_gs_2021_2022/"

if __name__ == "__main__":
    start = time.time()
    path = fetch_velocity_data(
        out_dir=OUT_DIR,
        out_file="velocity_2021_01.nc",
        lon_bounds=LON_BOUNDS,
        lat_bounds=LAT_BOUNDS,
        start_date="2021-01-01",
        end_date="2021-01-31",
        depth_bounds=DEPTH_BOUNDS,
        variables=("uo", "vo", "thetao"),
        force_download=True,
    )
    elapsed = time.time() - start
    print(f"Downloaded {path} in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
