"""Scope GLORYS OSSE download size + wall-time on small slices BEFORE the full pull.

Downloads a few days (daily groups) / months (monthly group) / the static grid,
per code/osse/configs/osse_glorys.json, into <out_root>/_scope/ (safe to rm -rf),
times each, and prints a table with the extrapolated full-period size per group
and a running total against the storage budget.

    python code/osse/scope_download.py [config.json] [--days N] [--months M]

Reads nothing large and triggers no full download; this is the gate the user
reads before committing to the multi-GB pull.
"""

import argparse
import calendar
import datetime as _dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_glorys as dg  # noqa: E402

GB = 1024 ** 3
MB = 1024 ** 2


def _daily_days(period):
    sy, sm = (int(x) for x in period["daily_start"].split("-"))
    ey, em = (int(x) for x in period["daily_end"].split("-"))
    start = _dt.date(sy, sm, 1)
    end = _dt.date(ey, em, calendar.monthrange(ey, em)[1])
    return (end - start).days + 1


def _monthly_months(period):
    sy, sm = (int(x) for x in period["monthly_start"].split("-"))
    ey, em = (int(x) for x in period["monthly_end"].split("-"))
    return (ey - sy) * 12 + (em - sm) + 1


def _slice_window(period, ndays):
    """A short daily window inside the analysis period (mid-year, mid-month)."""
    sy = int(period["daily_start"].split("-")[0])
    start = _dt.date(sy, 6, 1)
    end = start + _dt.timedelta(days=ndays - 1)
    return start.isoformat(), end.isoformat()


def _size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def scope(config_path, ndays, nmonths):
    with open(config_path) as f:
        cfg = json.load(f)

    regions = cfg["regions"]
    period = cfg["period"]
    out_dir = os.path.join(cfg["out_root"], "_scope")
    total_days = _daily_days(period)
    total_months = _monthly_months(period)
    s_start, s_end = _slice_window(period, ndays)
    m_start = period["monthly_start"]
    sy, sm = (int(x) for x in m_start.split("-"))
    m_end = f"{sy + (sm - 1 + nmonths - 1) // 12:04d}-{(sm - 1 + nmonths - 1) % 12 + 1:02d}"

    rows = []  # (group, slice_desc, slice_bytes, wall_s, full_gb)

    for g in cfg["groups"]:
        region = regions[g["region"]]
        lon, lat = region["lon_bounds"], region["lat_bounds"]
        depth = cfg[g["depth"]] if "depth" in g else None
        name = g["name"]

        t0 = time.perf_counter()
        if g["cadence"] == "static":
            paths = dg.fetch_static(out_dir, lon, lat, g["parts"], depth_bounds=depth,
                                    file_prefix=f"scope_{name}", force=True)
            wall = time.perf_counter() - t0
            b = sum(_size(p) for p in paths.values())
            rows.append((name, f"{len(paths)} parts", b, wall, b / GB))

        elif g["cadence"] == "daily":
            path = dg.fetch_subset(
                out_dir, f"scope_{name}.nc", g["dataset_id"], g["variables"],
                lon, lat, depth_bounds=depth, start_date=s_start, end_date=s_end,
                force=True)
            wall = time.perf_counter() - t0
            b = _size(path)
            rows.append((name, f"{ndays} d", b, wall, b / ndays * total_days / GB))

        elif g["cadence"] == "velocity_bands":
            bands = {k: v for k, v in cfg["depth_bands"].items() if k != "note"}
            b_tot, full_gb = 0, 0.0
            for label, db in bands.items():
                path = dg.fetch_subset(
                    out_dir, f"scope_{name}_{label}.nc", g["dataset_id"],
                    g["variables"], lon, lat, depth_bounds=db,
                    start_date=s_start, end_date=s_end, force=True)
                bb = _size(path)
                b_tot += bb
                full_gb += bb / ndays * total_days / GB
            wall = time.perf_counter() - t0
            rows.append((name, f"{ndays} d x {len(bands)} bands", b_tot, wall, full_gb))

        elif g["cadence"] == "monthly":
            paths, _ = dg.fetch_monthly(
                out_dir, f"scope_{name}", g["dataset_id"], g["variables"],
                lon, lat, m_start, m_end, depth_bounds=depth, force=True)
            wall = time.perf_counter() - t0
            b = sum(_size(p) for p in paths)
            rows.append((name, f"{nmonths} mo", b, wall,
                         b / nmonths * total_months / GB))

    _report(rows, cfg.get("budget_gb"), total_days, total_months)


def _report(rows, budget_gb, total_days, total_months):
    print("\n" + "=" * 78)
    print(f"SCOPING (full period: {total_days} daily days, {total_months} monthly months)")
    print("=" * 78)
    hdr = f"{'group':<22}{'slice':<16}{'slice MB':>10}{'wall s':>9}{'MB/s':>8}{'full GB':>10}"
    print(hdr)
    print("-" * 78)
    total_full = 0.0
    for name, desc, b, wall, full_gb in rows:
        mbps = (b / MB / wall) if wall > 0 else float("nan")
        print(f"{name:<22}{desc:<16}{b / MB:>10.1f}{wall:>9.1f}{mbps:>8.1f}{full_gb:>10.2f}")
        total_full += full_gb
    print("-" * 78)
    print(f"{'TOTAL full-period estimate':<48}{total_full:>28.2f} GB")
    if budget_gb:
        margin = budget_gb - total_full
        flag = "OK" if margin > 0 else "OVER BUDGET"
        print(f"{'budget':<48}{budget_gb:>25.0f} GB  [{flag}, {margin:+.1f} GB]")
    print("=" * 78)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "configs", "osse_glorys.json"))
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--months", type=int, default=3)
    args = ap.parse_args()
    scope(args.config, args.days, args.months)
