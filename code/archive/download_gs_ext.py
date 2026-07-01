"""Augment the GLORYS truth box with downstream velocity strips.

Floats deployed in the Gulf Stream advect east/south and exit the original
33-48N, 74-59W advection domain within the year (they are then deleted by
Parcels). This extends the *advection* domain downstream to lat 30-48N,
lon -74 to -49W (east +10, south +3) so floats stay in-bounds longer.

To avoid re-downloading the existing center (data/velocity_gs, ~2.7 GB), only
the new frame strips are downloaded into data/velocity_gs_ext/:

    east strip : lat 30-48,  lon -59..-49   (full new latitude band, east side)
    south strip: lat 30-33,  lon -74..-59   (below the center, center lon band)

They are then stitched with the existing center, one file per month, into
data/velocity_gs_wide/. Strips are requested with a small overlap and the
duplicate row/column is dropped after concatenation, so the merge is robust to
the exact native grid offset. The truth/analysis region is a separate choice;
this only grows the field the floats are advected through.

Run from the repo root:
    python code/download_gs_ext.py
"""

import glob
import os

import xarray as xr

from trajsim import fetch_velocity_months

CENTER_DIR = "data/velocity_gs"
EXT_DIR = "data/velocity_gs_ext"
WIDE_DIR = "data/velocity_gs_wide"
START_MONTH, END_MONTH = "2020-01", "2020-12"


def download_strips():
    """Download only the new east and south strips (resumable)."""
    # East strip: full new latitude band, longitudes east of the center.
    # Slight westward overlap (-59.05) guarantees no gap at the seam.
    fetch_velocity_months(
        START_MONTH, END_MONTH, out_dir=EXT_DIR, file_prefix="velocity_east",
        lat_bounds=(30, 48), lon_bounds=(-59.05, -49),
    )
    # South strip: below the center, over the center's longitude band.
    fetch_velocity_months(
        START_MONTH, END_MONTH, out_dir=EXT_DIR, file_prefix="velocity_south",
        lat_bounds=(30, 33.05), lon_bounds=(-74, -59),
    )


def merge_months():
    """Stitch center + south + east into one wide file per month (resumable)."""
    os.makedirs(WIDE_DIR, exist_ok=True)
    for center_path in sorted(glob.glob(f"{CENTER_DIR}/velocity_*.nc")):
        tag = os.path.basename(center_path)[len("velocity_"):-len(".nc")]  # YYYY_MM
        out = f"{WIDE_DIR}/velocity_{tag}.nc"
        if os.path.exists(out):
            print(f"skip (exists): {out}")
            continue
        center = xr.open_dataset(center_path)
        south = xr.open_dataset(f"{EXT_DIR}/velocity_south_{tag}.nc")
        east = xr.open_dataset(f"{EXT_DIR}/velocity_east_{tag}.nc")

        west = (xr.concat([south, center], "latitude")
                .sortby("latitude").drop_duplicates("latitude"))
        full = (xr.concat([west, east], "longitude")
                .sortby("longitude").drop_duplicates("longitude"))
        full.to_netcdf(out)
        for ds in (center, south, east, west, full):
            ds.close()
        print(f"merged -> {out}")


if __name__ == "__main__":
    download_strips()
    merge_months()
    files = sorted(glob.glob(f"{WIDE_DIR}/velocity_*.nc"))
    print(f"done: {len(files)} wide files in {WIDE_DIR}/")
