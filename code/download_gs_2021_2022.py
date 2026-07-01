"""Download GLORYS12V1 velocity + temperature for 2021-2022 over a narrower
Gulf Stream box (lat 33-45) for the next analysis phase.

Same lon span, dataset, variables, and depth range as data/velocity_gs_wide/
(lon -74 to -49, cmems_mod_glo_phy_my_0.083deg_P1D-m, uo/vo/thetao, 0-2300 m),
but a tighter latitude band (33-45N vs 30-48N) and a later time period.
One file per month, resumable: re-running skips months already on disk.

Run from the repo root:
    python code/download_gs_2021_2022.py
"""

from trajsim import fetch_velocity_months

LON_BOUNDS = (-74, -49)
LAT_BOUNDS = (33, 45)
DEPTH_BOUNDS = (0, 2300)  # brackets 2000 m via GLORYS levels 1941.9 / 2225.1 m

START_MONTH = "2021-01"
END_MONTH = "2022-12"
OUT_DIR = "data/velocity_gs_2021_2022/"

if __name__ == "__main__":
    paths, glob_pattern = fetch_velocity_months(
        start_month=START_MONTH,
        end_month=END_MONTH,
        out_dir=OUT_DIR,
        lon_bounds=LON_BOUNDS,
        lat_bounds=LAT_BOUNDS,
        depth_bounds=DEPTH_BOUNDS,
        variables=("uo", "vo", "thetao"),
    )
    print(f"Done: {len(paths)} files -> {glob_pattern}")
