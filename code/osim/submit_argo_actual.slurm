#!/bin/bash
# Actual real-Argo deployment arm (code/osim/continuous_sim.py --argo actual):
# advect the real deployment pool (~365 floats in the seed region, 2010-2019)
# VERBATIM (real positions/dates/lifespans) once through GLORYS park velocity,
# scored on 2015-2019 in both boxes -> estimates_continuous_argo_actual.csv.
# One realisation, so a single (non-array) job: light on the cluster.
#
#   sbatch code/osim/submit_argo_actual.slurm
#
#SBATCH --job-name=osim_argo_actual
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/osim_argo_actual_%j.out
#SBATCH --error=logs/osim_argo_actual_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

START="${1:-2010-01-01}"
SCORE="${2:-2015,2016,2017,2018,2019}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== argo-actual: verbatim real deployments, from $START, score $SCORE ==="
"$PY" code/osim/continuous_sim.py \
    --argo actual --reps 1 --start "$START" --score-years "$SCORE"
