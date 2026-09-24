"""Evaluate the OHC residual field at profile positions, for every analysis year.

Strictly a cell-day lookup: Z is a simple function on the GLORYS grid, so
evaluating it at a profile means reading the cell that *contains* the profile.
No interpolation, in space or time -- interpolating would evaluate a different
field than the one B = sum_j (w_j - p_j) Z_j is defined over.

The same gather serves surrogate designs and real trajectories, so the
null-calibration check exercises the path the real analysis uses.

Run from the repo root:
    python code/osim/sample_residuals.py <positions.csv> <out.csv> [box]
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg

POSITION_COLS = ["vel_year", "rep", "lat", "lon", "doy"]


def load_residual_field(box="boxB", tag=None):
    path = os.path.join(cfg.OUT_DIR, f"residual_field_{tag or box}.nc")
    return xr.open_dataset(path)["z"].load()


def gather(positions, z):
    """Look up Z_k at each position for every analysis year k.

    ``positions`` needs lat, lon, doy; ``z`` is (year, doy, latitude, longitude).
    Positions whose containing cell is masked are dropped, not filled from a
    neighbour -- a nearest-wet-cell fallback would bias B toward whatever sits
    beside the mask.
    """
    ilat = cfg.containing_index(positions["lat"].to_numpy(), z["latitude"].values)
    ilon = cfg.containing_index(positions["lon"].to_numpy(), z["longitude"].values)
    idoy = positions["doy"].to_numpy().astype(int)

    inside = (ilat >= 0) & (ilon >= 0) & (idoy >= 0) & (idoy < cfg.DAYS_PER_YEAR)
    vals = np.full((len(positions), z.sizes["year"]), np.nan)
    zi = z.values
    vals[inside] = zi[:, idoy[inside], ilat[inside], ilon[inside]].T

    out = positions.copy()
    for j, year in enumerate(z["year"].values):
        out[f"z_{year}"] = vals[:, j]
    zcols = [f"z_{y}" for y in z["year"].values]
    keep = out[zcols].notna().all(axis=1)
    return out[keep].reset_index(drop=True), int((~keep).sum())


def main(positions_csv, out_csv, box="boxB", tag=None):
    positions = pd.read_csv(positions_csv)
    missing = [c for c in POSITION_COLS if c not in positions.columns]
    if missing:
        raise ValueError(f"{positions_csv} missing required columns: {missing}")

    z = load_residual_field(box, tag)
    sampled, ndropped = gather(positions, z)
    sampled.to_csv(out_csv, index=False)
    print(f"{len(sampled):,} profiles kept, {ndropped:,} dropped (masked/outside) -> {out_csv}")
    return sampled


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["boxB"]))
