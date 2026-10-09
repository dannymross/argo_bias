#!/bin/bash
# Anchored GLORYS clouds (code/osim/clouds.py), one anchor month per array task. KIND=fit runs the
# 8 x 8 (kx, ky) grid for the box anchors; KIND=prod runs the arms in prod_arms.csv for every anchor.
#
#   python code/osim/clouds.py profiles                                   # once
#   sbatch --array=0-60 code/osim/submit_clouds.slurm fit
#   python code/osim/clouds.py kfit                                       # after the fit array
#   sbatch --array=0-60 --mem=6G --time=01:00:00 code/osim/submit_clouds.slurm fit2
#   python code/osim/clouds.py kfit2                                      # writes prod_arms.csv
#   sbatch --array=0-60 code/osim/submit_clouds.slurm prod data/osim/clouds/prod_arms.csv
#
#SBATCH --job-name=osim_clouds
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/osim_clouds_%A_%a.out
#SBATCH --error=logs/osim_clouds_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
KIND="${1:-fit}"
ARMS="${2:-}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

MONTHS=($("$PY" code/osim/clouds.py months --month "$KIND"))
MONTH="${MONTHS[$SLURM_ARRAY_TASK_ID]}"
echo "=== clouds $KIND $MONTH on $(hostname) ==="
if [ -s "data/osim/clouds/$KIND/${KIND}_${MONTH}_s0.parquet" ]; then echo "done already"; exit 0; fi
"$PY" code/osim/clouds.py "$KIND" --month "$MONTH" ${ARMS:+--kfit "$ARMS"}
