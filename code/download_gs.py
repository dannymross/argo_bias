"""Download the GLORYS12 Gulf Stream / NAC pilot box for the OHC bias study.

A ~15deg x 15deg box over the Gulf Stream just after it separates from the US
coast (the strong current seen in reports/anim/argo_nac_float_dohco_anim.mp4).
Floats are deployed in the central ~5deg so they have room to advect for ~a
year before exiting the truth domain.

One file per month, resumable: re-running skips months already on disk. The box
is ~2.8 GB for the full year (uo, vo, thetao to 2300 m), small enough to iterate
on locally.

Run from the repo root:
    python code/download_gs.py
"""

from trajsim import fetch_velocity_months

# Truth / velocity domain (15 x 15 deg over the separated Gulf Stream).
LON_BOUNDS = (-74, -59)
LAT_BOUNDS = (33, 48)
DEPTH_BOUNDS = (0, 2300)  # brackets 2000 m via GLORYS levels 1941.9 / 2225.1 m

START_MONTH = "2020-01"
END_MONTH = "2020-12"
OUT_DIR = "data/velocity_gs/"


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
