"""Float-array constants and deployment helpers shared by the float simulation.

What the simulated array is: floats parked at PARK_DEPTH, reporting one profile
every CYCLE_DAYS, seeded over the conditioning region plus SEED_MARGIN_UNIFORM
and advected in the park-depth velocity field under VEL_DIR.

Split out of the old design_invariance.py, which also carried the steady and
uniform arms that the continuous-seeding arm replaced.
"""

import glob
import os
import sys

import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osim import osim_config as cfg

VEL_DIR = os.path.join(cfg.GLORYS_DIR, "velocity_daily")
PARK_DEPTH = 1000.0
CYCLE_DAYS = 10          # Argo surfacing cadence: one profile per cycle
SEED_MARGIN_UNIFORM = 8.0  # seeded this far beyond the region being read out
JITTER_KM = 5.0          # per-replicate seed perturbation


def load_deployment_pool(lat_bounds=(15, 50), lon_bounds=(-70, -30),
                         years=(2010, 2019), path=None):
    """Pooled empirical deployment distribution from the real Argo record.

    One row per float: its first profile's position and day-of-year, plus its
    observed lifespan. POOLED ACROSS YEARS on purpose -- drawing year-l designs
    from year-l's real deployments would let the sampled plan correlate with that
    year's ocean state, and deployment counts in this region swing by an order of
    magnitude between years for programmatic reasons. Only `lifespan_days` is
    used by the continuous arm, which seeds positions uniformly rather than from
    this pool.
    """
    path = path or os.path.join(os.path.dirname(cfg.GLORYS_DIR), "argo_ohc.csv")
    df = pd.read_csv(path, usecols=["float_id", "lat", "lon", "date"])
    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)

    g = df.sort_values("date").groupby("float_id")
    dep = g.first().reset_index()
    dep["lifespan_days"] = (g["date"].max() - g["date"].min()).dt.days.values
    dep["doy"] = dep["date"].dt.dayofyear - 1
    dep["year"] = dep["date"].dt.year

    pool = dep[dep["lat"].between(*lat_bounds) & dep["lon"].between(*lon_bounds)
               & dep["year"].between(*years)]
    return pool[["lat", "lon", "doy", "lifespan_days"]].reset_index(drop=True)


def velocity_extent(path=None, inset=0.25):
    """Seedable (lat, lon) bounds of the park velocity grid, read from the files.

    Read rather than hard-coded: a seed outside the velocity domain is deleted on
    the first kernel step. The inset keeps seeds a few grid cells clear of the
    edge so they are not lost to boundary interpolation.
    """
    path = path or sorted(glob.glob(os.path.join(VEL_DIR, "velocity_daily_park_*.nc")))[0]
    with xr.open_dataset(path) as ds:
        return ((float(ds["latitude"].min()) + inset, float(ds["latitude"].max()) - inset),
                (float(ds["longitude"].min()) + inset, float(ds["longitude"].max()) - inset))


def in_box(df, box):
    """Boolean mask: rows whose (lat, lon) fall inside an analysis box."""
    la0, la1 = cfg.BOXES[box]["lat_bounds"]
    lo0, lo1 = cfg.BOXES[box]["lon_bounds"]
    return df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)
