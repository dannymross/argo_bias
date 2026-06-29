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
from global_land_mask import globe
import uuid
from pyproj import Geod


# ---- SIM FUNCTIONS
def deploy_float_grid(
    top_left,
    bottom_right=None,
    nfloats=None,
    spacing_deg=None,
    deploy_time="2020-01-01",
    ocean_only=True,
):
    """
    Deploy floats on a regular lat/lon grid within a bounding box.

    Mode 1 — top_left + bottom_right + nfloats:
        Aspect-ratio-preserving even grid across the full box.
    Mode 2 — top_left + bottom_right + spacing_deg:
        Fixed degree spacing edge-to-edge across the full box.
    Mode 3 — top_left + spacing_deg [+ nfloats]:
        Fixed degree spacing expanding to North Atlantic default bounds.
        If nfloats is also given, the grid is built first then capped to
        the first nfloats ocean points (row-major: W→E, N→S).

    Parameters
    ----------
    top_left : tuple of (float, float)
        (lat, lon) of the NW corner; always the first grid point.
    bottom_right : tuple of (float, float), optional
        (lat, lon) of the SE corner.
    nfloats : int, optional
        Cap on total floats deployed.
    spacing_deg : float, optional
        Distance between floats in degrees.
    deploy_time : str
        ISO-8601 deployment date for all floats.
    ocean_only : bool
        Drop grid points that fall on land (default True).

    Returns
    -------
    dict with keys "lat", "lon", "time".
    """
    lat_max, lon_min = top_left

    # Resolve SE corner
    if bottom_right is not None:
        lat_min, lon_max = bottom_right
    elif spacing_deg is not None:
        # Expand to full North Atlantic default bounds
        lat_min, lon_max = 25.0, 30.0
    else:
        raise ValueError("Provide either bottom_right, spacing_deg, or both.")

    # Build 1-D grid arrays
    if spacing_deg is not None:
        lat_pts = np.arange(lat_max, lat_min - 1e-9, -spacing_deg)
        lon_pts = np.arange(lon_min, lon_max + 1e-9, spacing_deg)
    else:
        # nfloats only: split proportional to aspect ratio
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        aspect = lon_span / lat_span
        n_lat = max(1, round((nfloats / aspect) ** 0.5))
        n_lon = max(1, round(n_lat * aspect))
        lat_pts = np.linspace(lat_max, lat_min, n_lat)
        lon_pts = np.linspace(lon_min, lon_max, n_lon)

    lon_grid, lat_grid = np.meshgrid(lon_pts, lat_pts)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()

    # Land-sea mask
    if ocean_only:
        is_ocean = globe.is_ocean(lat_flat, lon_flat)
        lat_flat = lat_flat[is_ocean]
        lon_flat = lon_flat[is_ocean]

    # Cap to nfloats when spacing_deg drove the grid size
    if nfloats is not None and spacing_deg is not None:
        lat_flat = lat_flat[:nfloats]
        lon_flat = lon_flat[:nfloats]

    nfloats_actual = len(lat_flat)
    tim = np.array([deploy_time] * nfloats_actual, dtype="datetime64")

    print(f"{nfloats_actual} floats deployed")
    return {"lat": lat_flat, "lon": lon_flat, "time": tim}


def fetch_velocity_data(
    out_dir="~/data/",
    out_file="velocity.nc",
    lon_bounds=(-78, 17),
    lat_bounds=(18, 80),
    start_date="2020-01-01",
    end_date="2020-01-31",
    depth_bounds=(0, 2000),
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=("uo", "vo"),
    force_download=False,
):
    """
    Download velocity data from Copernicus Marine and return an xarray Dataset.

    Parameters
    ----------
    out_dir : str
        Directory to save the NetCDF file (~ expanded automatically).
    out_file : str
        Output filename (should end in .nc).
    lon_bounds : tuple of (float, float)
        (min_lon, max_lon).
    lat_bounds : tuple of (float, float)
        (min_lat, max_lat).
    start_date, end_date : str
        ISO-8601 date strings for the temporal subset.
    depth_bounds : tuple of (float, float)
        (min_depth, max_depth) in metres.
    dataset_id : str
        Copernicus Marine dataset ID.
    variables : tuple of str
        Variable names to download (default: zonal + meridional velocity).
    force_download : bool
        If False (default), skip the download if the file already exists.

    Returns
    -------
    xarray.Dataset
    """
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_file)

    if not force_download and os.path.exists(path):
        print(f"Using cached file: {path}")
    else:
        print(f"Downloading velocity data → {path}")
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=list(variables),
            minimum_longitude=lon_bounds[0],
            maximum_longitude=lon_bounds[1],
            minimum_latitude=lat_bounds[0],
            maximum_latitude=lat_bounds[1],
            start_datetime=start_date,
            end_datetime=end_date,
            minimum_depth=depth_bounds[0],
            maximum_depth=depth_bounds[1],
            output_filename=out_file,
            output_directory=out_dir,
        )

    return xr.open_dataset(path)


def build_velocity_field(ds, model="GLORYS12V1"):
    """
    Wrap an xarray Dataset in a VirtualArgoFleet Velocity fieldset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing velocity variables (as returned by fetch_velocity_data).
    model : str
        Model identifier passed to Velocity (default: "GLORYS12V1").

    Returns
    -------
    Velocity
    """
    return Velocity(model=model, src=ds)


def run_simulation(
    plan,
    velocity_field,
    duration_days=30,
    out_dir="data/virtualfleet/",
    out_file=None,
    cfg=None,
    nc_save=False,
):
    """
    Run a VirtualArgoFleet simulation.

    Parameters
    ----------
    plan : dict
        Deployment plan with keys "lat", "lon", "time".
    velocity_field : Velocity
        Fieldset built by build_velocity_field().
    duration_days : int or float
        Simulation duration in days.
    out_dir : str
        Directory to write the trajectory output.
    out_file : str
        Output filename (e.g. "trajectory_output.zarr").
    cfg : FloatConfiguration, optional
        Mission configuration. Defaults to FloatConfiguration("default").

    Returns
    -------
    str
        Full path to the trajectory output file.
    """
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if cfg is None:
        cfg = FloatConfiguration("default")

    if out_file is None:
        rand = uuid.uuid4().hex[:8]
        out_file = f"trajectory_{rand}.zarr"

    fleet = VirtualFleet(plan=plan, fieldset=velocity_field, mission=cfg)

    print(
        f"Running simulation: {duration_days} days → {os.path.join(out_dir, out_file)}"
    )
    fleet.simulate(
        duration=timedelta(days=duration_days),
        step=datetime.timedelta(seconds=300),
        record=datetime.timedelta(seconds=3600),
        output=True,
        output_folder=out_dir,
        output_file=out_file,
    )

    output = os.path.join(out_dir, out_file)

    if nc_save:
        nc_path = os.path.splitext(output.rstrip("/"))[0] + ".nc"
        xr.open_zarr(output).to_netcdf(nc_path)

    return output


# ---- PLOTS
def map_trajectories(
    output,
    lon="lon",
    lat="lat",
    figsize=(14, 7),
    linewidth=0.6,
    title="VirtualFleet Trajectories",
    xlim=None,
    ylim=None,
    save_path=None,
    dpi=300,
    show=True,
):
    ds = xr.open_zarr(output)
    filename = os.path.basename(output)
    stem = os.path.splitext(filename)[0]

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # basemap
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.3)

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
    plt.tight_layout()

    # save figure
    if save_path is not None:
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)

        fig_file = os.path.join(save_path, f"{stem}.png")

        fig.savefig(
            fig_file,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved figure → {fig_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_deployment_plan(
    plan, title="Float Deployment Plan", figsize=(12, 8), margin=3
):
    lats = plan["lat"]
    lons = plan["lon"]

    # Infer map extent with a small margin
    margin = 3
    extent = [
        lons.min() - margin,
        lons.max() + margin,
        lats.min() - margin,
        lats.max() + margin,
    ]

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f5")
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d0", edgecolor="grey", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="dimgrey")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="grey", linestyle=":")
    ax.gridlines(
        draw_labels=True, linewidth=0.4, color="grey", alpha=0.6, linestyle="--"
    )

    # Float locations
    ax.scatter(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        s=8,
        color="black",
        alpha=0.8,
        zorder=5,
        label=f"{len(lats)} floats",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.show()


# ---- VELOCITY
# def add_velocity_enu(ds):
#    """
#    Calculate ENU velocity components for each float trajectory using
#    WGS84 ellipsoidal geometry
#
#    Parameters
#    ----------
#    ds : xarray.Dataset
#        Trajectory dataset with dims (trajectory, obs) and variables
#        lat, lon, time.
#
#    Returns
#    -------
#    xarray.Dataset
#        Original dataset plus:
#            dt_s     – time step (s)
#            dx_m     – ellipsoidal distance (m)
#            bearing  – forward bearing (deg, 0=N 90=E)
#            speed_ms – scalar speed (m/s)
#            u_ms     – eastward velocity (m/s)
#            v_ms     – northward velocity (m/s)
#        First obs of each trajectory is NaN.
#    """
#    lat = ds["lat"].values.astype(float)  # (trajectory, obs)
#    lon = ds["lon"].values.astype(float)
#    time = ds["time"].values  # datetime64[ns]
#
#    # Lagged arrays (previous observation)
#    lat0 = np.full_like(lat, np.nan)
#    lat0[:, 1:] = lat[:, :-1]
#    lon0 = np.full_like(lon, np.nan)
#    lon0[:, 1:] = lon[:, :-1]
#    time0 = np.empty_like(time)
#    time0[:] = np.datetime64("NaT")
#    time0[:, 1:] = time[:, :-1]
#
#    # dt in seconds
#    dt_s = (time - time0).astype("float64") / 1e9
#
#    # Flatten for pyproj, mask NaN pairs
#    shape = lat.shape
#    lat0_f, lon0_f = lat0.ravel(), lon0.ravel()
#    lat_f, lon_f = lat.ravel(), lon.ravel()
#
#    valid = np.isfinite(lat0_f) & np.isfinite(lon0_f)
#    brng_f = np.full(lat_f.shape, np.nan)
#    dx_f = np.full(lat_f.shape, np.nan)
#
#    # pyproj.Geod.inv: (lon1, lat1, lon2, lat2) → (fwd_az, back_az, dist_m)
#    fwd_az, _, dist_m = Geod(ellps="WGS84").inv(
#        lon0_f[valid],
#        lat0_f[valid],
#        lon_f[valid],
#        lat_f[valid],
#    )
#    brng_f[valid] = fwd_az  # degrees, 0=N 90=E — matches geosphere::bearing
#    dx_f[valid] = dist_m  # WGS84 ellipsoidal distance — matches geosphere::distGeo
#
#    brng = brng_f.reshape(shape)
#    dx_m = dx_f.reshape(shape)
#
#    # ENU decomposition
#    theta = np.radians(brng)
#    speed = dx_m / dt_s
#    u_ms = speed * np.sin(theta)
#    v_ms = speed * np.cos(theta)
#
#    dims = ("trajectory", "obs")
#    ds = ds.assign(
#        dt_s=(dims, dt_s.astype("float32")),
#        dx_m=(dims, dx_m.astype("float32")),
#        bearing=(dims, brng.astype("float32")),
#        speed_ms=(dims, speed.astype("float32")),
#        u_ms=(dims, u_ms.astype("float32")),
#        v_ms=(dims, v_ms.astype("float32")),
#    )
#
#    attrs = {
#        "dt_s": ("s", "Time step"),
#        "dx_m": ("m", "WGS84 ellipsoidal distance"),
#        "bearing": ("deg", "Forward bearing (0=N, 90=E)"),
#        "speed_ms": ("m/s", "Scalar speed"),
#        "u_ms": ("m/s", "Eastward velocity"),
#        "v_ms": ("m/s", "Northward velocity"),
#    }
#    for var, (units, long_name) in attrs.items():
#        ds[var].attrs = {"units": units, "long_name": long_name}
#
#    return ds


# ---- SIMULATION
ds = fetch_velocity_data(
    out_dir=os.getcwd() + "/data/velocity/",
    out_file="nac_currents_jan2020.nc",
    lon_bounds=(-78, 17),
    lat_bounds=(18, 80),
    start_date="2020-01-01",
    end_date="2020-01-31",
)
VELfield = build_velocity_field(ds)
nac_plan = deploy_float_grid(top_left=(42, -68), bottom_right=(30, -44), nfloats=100)
plot_deployment_plan(nac_plan)
output = run_simulation(plan=nac_plan, velocity_field=VELfield, duration_days=60)

# ---- TRAJECTORY ANALYSIS
tds = xr.open_zarr(output)
zarr_path = output  # '/Users/.../trajectory_cfa5583d.zarr/'
nc_path = os.path.splitext(output.rstrip("/"))[0] + ".nc"

xr.open_zarr(output).to_netcdf(nc_path)

# map_trajectories(output, xlim=(-70, -42), ylim=(20, 50), save_path="figures/")
map_trajectories(output, xlim=(-80, -20), ylim=(30, 60), save_path="figures/")
map_trajectories(output, save_path="figures/")


# Example
ds_with_vel = add_velocity_enu(tds)
print(ds_with_vel)
