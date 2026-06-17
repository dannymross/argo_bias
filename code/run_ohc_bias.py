"""End-to-end OHC sampling-bias analysis driver.

Chains the pipeline for a finished simulation:

  1. open the GLORYS temperature field (truth) and the trajectory zarr;
  2. integrate truth OHC over the box and bin to 1-deg monthly cells;
  3. sample synthetic-Argo profiles at float surfacing points, integrate OHC,
     bin to the same cells;
  4. compute representation + coverage bias and write tables/plots.

Example (smoke test against the existing full-NA Jan/Feb files):

    python code/run_ohc_bias.py \
        --traj data/virtualfleet/nac_gs_smoke_task000.zarr \
        --velocity "data/velocity/velocity_2020_*.nc" \
        --lat-bounds 33 48 --lon-bounds -74 -59 \
        --prefix nac_gs_smoke
"""

import argparse
import os

import pandas as pd
import xarray as xr

import ohc
import ohc_bias
from trajplots import open_trajectories


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj", required=True, help="trajectory zarr path or glob")
    p.add_argument("--velocity", required=True, help="GLORYS netcdf path or glob (needs thetao)")
    p.add_argument("--lat-bounds", type=float, nargs=2, default=(33, 48))
    p.add_argument("--lon-bounds", type=float, nargs=2, default=(-74, -59))
    p.add_argument("--deg", type=float, default=1.0, help="analysis cell size (degrees)")
    p.add_argument("--degs", type=float, nargs="+", default=None,
                   help="if set, sweep these cell sizes and emit a bias-vs-resolution "
                        "curve instead of a single-resolution analysis")
    p.add_argument("--unweighted-ref", action="store_true",
                   help="use the unweighted native-cell truth mean as the reference "
                        "instead of the default cos(lat) area-weighted domain mean")
    p.add_argument("--outdir", default="data/ohc_bias")
    p.add_argument("--figdir", default="figures")
    p.add_argument("--prefix", default="ohc_bias")
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    print(f"opening velocity field: {args.velocity}")
    theta_ds = xr.open_mfdataset(args.velocity) if "*" in args.velocity \
        else xr.open_dataset(args.velocity)
    theta_ds = theta_ds.sel(
        latitude=slice(args.lat_bounds[0], args.lat_bounds[1]),
        longitude=slice(args.lon_bounds[0], args.lon_bounds[1]),
    )

    # 1. Truth OHC field (native grid) -- the expensive step, computed once.
    print("computing truth OHC field ...")
    truth_field = ohc.truth_ohc_field(theta_ds).compute()

    # 2. Synthetic-Argo OHC at float profiles -- also computed once.
    print("sampling synthetic-Argo profiles ...")
    traj = open_trajectories(args.traj)
    sim = ohc.float_ohc(traj, theta_ds)
    sim_path = os.path.join(args.outdir, f"{args.prefix}_sim_argo_ohc.csv")
    sim.to_csv(sim_path, index=False)
    print(f"  {len(sim)} synthetic profiles -> {sim_path}")

    weighted_ref = not args.unweighted_ref
    ref = ohc.truth_domain_mean(truth_field, weighted=weighted_ref)
    print(f"truth reference: {'cos(lat) area-weighted' if weighted_ref else 'unweighted'} "
          f"domain mean")

    # Cell-size sweep: re-grid truth and floats cheaply at each resolution.
    if args.degs:
        print(f"sweeping cell sizes: {args.degs}")
        sweep = ohc_bias.sweep_resolution(truth_field, sim, args.degs,
                                          weighted_reference=weighted_ref)
        sweep_path = os.path.join(args.outdir, f"{args.prefix}_sweep.csv")
        sweep.to_csv(sweep_path, index=False)
        print("\n=== bias vs cell size (GJ/m2, time-averaged) ===")
        print(sweep.to_string(index=False))
        for v in ("ohc_700", "ohc_2000"):
            ohc_bias.plot_bias_vs_resolution(
                sweep, value_col=v,
                out_path=os.path.join(args.figdir, f"{args.prefix}_{v}_sweep.png"),
            )
        print("done.")
        return

    # Single-resolution analysis at args.deg.
    truth_cells = ohc.coarsen_truth(truth_field, deg=args.deg)
    print(f"  truth cells: {len(truth_cells)} (month x cell)")
    float_cells = ohc.grid_cells(sim, ["ohc_700", "ohc_2000"], deg=args.deg)

    # 3. Bias.
    print("computing bias ...")
    res = ohc_bias.compute_bias(float_cells, truth_cells, true_domain_mean=ref)
    summary = ohc_bias.bias_summary(res["domain"])

    res["domain"].to_csv(os.path.join(args.outdir, f"{args.prefix}_domain.csv"), index=False)
    res["cells"].to_csv(os.path.join(args.outdir, f"{args.prefix}_cells.csv"), index=False)
    summary.to_csv(os.path.join(args.outdir, f"{args.prefix}_summary.csv"), index=False)

    print("\n=== bias summary (GJ/m2, time-averaged) ===")
    print(summary.to_string(index=False))

    # 4. Plots.
    for v in ("ohc_700", "ohc_2000"):
        ohc_bias.plot_domain_timeseries(
            res["domain"], value_col=v,
            out_path=os.path.join(args.figdir, f"{args.prefix}_{v}_timeseries.png"),
        )
    ohc_bias.plot_bias_map(
        res["cells"], value_col="ohc_2000",
        out_path=os.path.join(args.figdir, f"{args.prefix}_ohc_2000_biasmap.png"),
    )
    print("done.")


if __name__ == "__main__":
    main()
