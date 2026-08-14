"""Full GLORYS OSSE download driver, config-driven and resumable.

Runs the real multi-GB pull per code/osse/configs/osse_glorys.json. Each group
(static / daily thetao / velocity bands / daily zos / monthly) downloads to
<out_root>/<group>/. Existing files are skipped, so a timed-out or requeued job
resumes cleanly. RUN scope_download.py FIRST and confirm the budget.

    python code/osse/run_download.py [config.json] [--only group1,group2]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_glorys as dg  # noqa: E402


def _period(g, cfg, start=None, end=None):
    """Effective period: global, overridden per group, then by CLI.

    Groups can span different windows -- the velocity record is extended back
    beyond the analysis years to enlarge the comparison sets C_k of the
    permutation test, while thetao/zos stay on the analysis window.
    """
    period = {**cfg["period"], **g.get("period", {})}
    if start:
        period["daily_start"] = period["monthly_start"] = start
    if end:
        period["daily_end"] = period["monthly_end"] = end
    return period


def _download_group(g, cfg, start=None, end=None):
    region = cfg["regions"][g["region"]]
    lon, lat = region["lon_bounds"], region["lat_bounds"]
    depth = cfg[g["depth"]] if "depth" in g else None
    period = _period(g, cfg, start, end)
    out_dir = os.path.join(cfg["out_root"], g["name"])
    name = g["name"]

    if g["cadence"] == "static":
        dg.fetch_static(out_dir, lon, lat, g["parts"], depth_bounds=depth,
                        file_prefix=g.get("file_prefix", "static"))
    elif g["cadence"] == "daily":
        dg.fetch_monthly(out_dir, name, g["dataset_id"], g["variables"], lon, lat,
                         period["daily_start"], period["daily_end"], depth_bounds=depth)
    elif g["cadence"] == "velocity_bands":
        bands = {k: v for k, v in cfg["depth_bands"].items() if k != "note"}
        dg.fetch_velocity_bands(out_dir, name, lon, lat, period["daily_start"],
                                period["daily_end"], bands, variables=g["variables"],
                                dataset_id=g["dataset_id"])
    elif g["cadence"] == "monthly":
        dg.fetch_monthly(out_dir, name, g["dataset_id"], g["variables"], lon, lat,
                         period["monthly_start"], period["monthly_end"], depth_bounds=depth)
    else:
        raise ValueError(f"unknown cadence {g['cadence']!r} for group {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "configs", "osse_glorys.json"))
    ap.add_argument("--only", help="comma-separated group names to download")
    ap.add_argument("--start", help="override period start, YYYY-MM (e.g. one array task's year)")
    ap.add_argument("--end", help="override period end, YYYY-MM")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    only = set(args.only.split(",")) if args.only else None
    for g in cfg["groups"]:
        if only and g["name"] not in only:
            continue
        p = _period(g, cfg, args.start, args.end)
        print(f"\n=== group: {g['name']} ({p['daily_start']}..{p['daily_end']}) ===")
        _download_group(g, cfg, args.start, args.end)
    print("\ndone.")
