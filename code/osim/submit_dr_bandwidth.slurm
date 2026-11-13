#!/bin/bash
# DR bandwidth sweep: LORO q-hat per kernel sd for r10-109, cross-fit DR re-estimated from the cached maps.
#SBATCH --job-name=dr_bw
#SBATCH --account=statsresearch_sc_default
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dr_bw_%j.out
#SBATCH --error=logs/dr_bw_%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
D=data/osim/natl_sim
rm -rf $D/q_s1_t30_prev && cp -r $D/q_s1_t30 $D/q_s1_t30_prev
OK=""
for s in 1 0.5 2 3 5 0.3333; do for t in 30 2 5 10 90; do
  $PY code/osim/natl_sim.py qpoints --reps 10,109 --sig $s,$t && $PY code/osim/natl_sim.py estimate --reps 10,109 --sig $s,$t \
    && OK="$OK $s,$t" || echo "FAILED sig $s,$t"
done; done
$PY -c "
import sys; sys.path.insert(0, 'code')
from osim import natl_sim as ns, natl_results as nr
nr.dr_bandwidth([tuple(float(v) for v in x.split(',')) for x in '$OK'.split()]).to_csv(ns.OUT + '/dr_bandwidth.csv')"
$PY -c "
import glob, os, numpy as np, pandas as pd
d = '$D/'
dev = max(np.abs(pd.read_parquet(f).q.to_numpy() / pd.read_parquet(d + 'q_s1_t30/' + os.path.basename(f)).q.to_numpy() - 1).max()
          for f in glob.glob(d + 'q_s1_t30_prev/r*.parquet'))
print('production q-hat max rel change:', dev)"
