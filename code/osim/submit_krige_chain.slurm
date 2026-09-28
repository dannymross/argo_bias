#!/bin/bash
# Gather the covariance refit, then submit the kriging-estimator array.
# Runs only after the refit array succeeds, so it cannot pick up stale plug-ins.
#   sbatch --dependency=afterok:<refit_jobid> code/osim/submit_krige_chain.slurm
#SBATCH --job-name=krige_chain
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/krige_chain_%j.out
#SBATCH --error=logs/krige_chain_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
"$PY" code/osim/krige_local.py --gather
test -s data/osim/krige_plugin_oracle.csv
sbatch --array=0-24 code/osim/submit_krige_estimate.slurm 25 _pilot_wide
