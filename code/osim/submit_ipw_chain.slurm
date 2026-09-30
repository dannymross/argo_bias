#!/bin/bash
# Full GLORYS chain on all _argo_resample replicates (ipw.chain): native-grid kernel q-hat
# (1/6 deg, 4 d; cached via intensity.load_kernel) weights each replicate's in-box
# nominal profiles in-sample. Then the in-sample check: 50 replicates re-weighted by q-hat
# built from the other 950.
#
#   sbatch code/osim/submit_ipw_chain.slurm
#
#SBATCH --job-name=osim_ipw_chain
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/osim_ipw_chain_%j.out
#SBATCH --error=logs/osim_ipw_chain_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

"$PY" code/osim/ipw.py chain
"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, "code")
import numpy as np, pandas as pd
from osim import intensity as qi, ipw
full = pd.read_parquet(ipw.chain_path())
test = np.random.default_rng(1).choice(sorted(full["rep"].unique()), 50, replace=False)
qds = qi.build_kernel(*ipw.SIGMA, exclude=test, deg="native")
out = ipw.chain(reps=test, qds=qds)
m = full[full["rep"].isin(test)].merge(out, on=["rep", "box", "year"], suffixes=("_in", "_out"))
d = pd.DataFrame({c: (m[f"{c}_in"] - m[f"{c}_out"]) * 1e-9 for c in ("ht", "hajek")}).assign(box=m["box"], year=m["year"])
s = d.groupby(["box", "year"]).agg(["mean", "sem"])
print("in-sample minus held-out (950) q-hat, GJ/m2, 50 reps:\n", s.round(4).to_string())
s.to_csv(os.path.join(ipw.TESTBED_DIR, "chain_insample_check.csv"))
EOF
