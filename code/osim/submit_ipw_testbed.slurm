#!/bin/bash
# Estimator testbed on the GLORYS residual field (code/osim/ipw.py). Each array task runs
# all ten box-years (boxes A, B x 2015-2019) for one position set: simulated replicate
# SLURM_ARRAY_TASK_ID (q-hat rebuilt without that replicate).
#
#   sbatch --array=0-9 code/osim/submit_ipw_testbed.slurm
#   python code/osim/ipw.py gather
#
#SBATCH --job-name=osim_ipw_testbed
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/osim_ipw_testbed_%A_%a.out
#SBATCH --error=logs/osim_ipw_testbed_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
"$PY" code/osim/ipw.py testbed --positions "${POSITIONS:-sim}" --rep "${SLURM_ARRAY_TASK_ID:-0}" ${KERNEL:+--kernel "$KERNEL"}
