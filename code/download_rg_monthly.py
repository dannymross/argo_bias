"""Download the RG Argo Climatology: 2020 monthly anomalies + the long-term mean.

Source: https://sio-argo.ucsd.edu/RG_Climatology.html
Files:
    RG_ArgoClim_<YYYYMM>_2019.nc.gz       -- one per month, the 2020 anomaly extension.
    RG_ArgoClim_Temperature_2019.nc.gz    -- the 2004-2018 climatological mean
                                              (~695 MB compressed; one-time download).
Both land in data/rg_climatology/.

Run from the repo root:
    python code/download_rg_monthly.py
"""

import gzip
import os
import shutil
import urllib.request

BASE_URL = "https://sio-argo.ucsd.edu/RG"
OUT_DIR = "data/rg_climatology"
YEAR = 2020
MEAN_FNAME = "RG_ArgoClim_Temperature_2019.nc.gz"


def _download_and_extract(fname: str) -> None:
    gz_path = os.path.join(OUT_DIR, fname)
    nc_path = gz_path[: -len(".gz")]

    if os.path.exists(nc_path):
        print(f"skip (exists): {nc_path}")
        return

    url = f"{BASE_URL}/{fname}"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in, open(nc_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"extracted -> {nc_path}")


def download_month(year: int, month: int) -> None:
    _download_and_extract(f"RG_ArgoClim_{year}{month:02d}_2019.nc.gz")


def download_mean() -> None:
    """Fetch the 2004-2018 climatological mean temperature/salinity file.

    Large (~695 MB compressed, ~1+ GB extracted) -- run this deliberately, not as
    part of routine setup.
    """
    _download_and_extract(MEAN_FNAME)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for month in range(1, 13):
        download_month(YEAR, month)
    download_mean()
    print(f"done: files in {OUT_DIR}/")
