import numpy as np
import xarray as xr
import os
import glob
import copernicusmarine
from datetime import timedelta
import parcels
from virtualargofleet import Velocity, FloatConfiguration, VirtualFleet
from virtualargofleet.utilities import simu2index, simu2csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# **** SETUP ****
# ----- DEPLOYMENT PLAN
# Number of floats we want to simulate:
nfloats = 10

# Define space/time locations of deployments:
lat = np.linspace(30, 38, nfloats)
lon = np.full_like(lat, -60)
tim = np.array(["2020-01-01" for i in range(nfloats)], dtype="datetime64")

# Define the deployment plan as a dictionary:
my_plan = {"lat": lat, "lon": lon, "time": tim}

# ----- VELOCITY FIELD
# download current velocities from copernicus marine service
# velocity field
# https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/download?dataset=cmems_mod_glo_phy_my_0.083deg_P1D-m_202311
out_path = os.path.expanduser("~/data/")
out_file = "nac_currents_jan2020.nc"

get_velo_data = copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=-78,
    maximum_longitude=17,
    minimum_latitude=18,
    maximum_latitude=80,
    start_datetime="2020-01-01",
    end_datetime="2020-01-31",
    minimum_depth=0,
    maximum_depth=2000,
    output_filename=out_file,
    output_directory=out_path,
)

path = os.path.join(os.path.expanduser(out_path), out_file)

ds = xr.open_dataset(path)
VELfield = Velocity(model="GLORYS12V1", src=ds)

# ----- MISSION PARAMETERS
cfg = FloatConfiguration("default")

# **** SIMULATION ****
VFleet = VirtualFleet(plan=my_plan, fieldset=VELfield, mission=cfg)
folder = os.path.expanduser("~/data/virtualfleet")
file = "trajectory_output.zarr"
out = os.path.join(folder, file)
VFleet.simulate(duration=timedelta(days=30), output_folder=folder, output_file=file)

# **** SIMULATION ANALYSIS ****
ds = xr.open_zarr(out)


# quick trajectory plot
def map_trajectories(
    ds,
    lon="lon",
    lat="lat",
    figsize=(14, 7),
    linewidth=0.4,
    title="VirtualFleet Trajectories",
    xlim=None,
    ylim=None,
):

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # basemap
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    # trajectories
    ax.plot(
        ds[lon].values.T,
        ds[lat].values.T,
        linewidth=linewidth,
        transform=ccrs.PlateCarree(),
    )

    # limits
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(title)

    return fig, ax


fig, ax = map_trajectories(ds, xlim=(-100, 30), ylim=(20, 80))
plt.savefig("reports/img/virtualfleet_trajectories.png")
