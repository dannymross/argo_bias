#!/bin/bash
# Bootstrap real-Argo deployment arm (code/osim/continuous_sim.py --argo boot): each
# array task resamples the real deployment pool (~359 floats in the seed region,
# 2010-2019) with replacement at the real rate, jitters positions, and advects once
# through GLORYS. One rep per task (RNG keyed on rep index = SLURM_ARRAY_TASK_ID);
# scored on 2015-2019 in both boxes. Compare its bias distribution to the uniform
# pilot to isolate the deployment-geometry effect.
#
#   sbatch --array=0-99 code/osim/submit_argo_boot.slurm
#   # then, once all tasks finish:
#   python code/osim/continuous_sim.py --gather --tag _argo_boot
#
#SBATCH --job-name=osim_argo_boot
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/osim_argo_boot_%A_%a.out
#SBATCH --error=logs/osim_argo_boot_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# node-local Parcels output (continuous_sim.scratch_dir); removed even on failure
trap 'rm -rf "${TMPDIR:-/tmp}/osim_${USER}_${SLURM_JOB_ID}"' EXIT

START="${1:-2010-01-01}"
SCORE="${2:-2015,2016,2017,2018,2019}"
REP="${SLURM_ARRAY_TASK_ID:-0}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== argo-boot rep $REP: resampled real deployments, from $START, score $SCORE ==="
"$PY" code/osim/continuous_sim.py \
    --argo boot --reps 1 --rep-offset "$REP" \
    --start "$START" --score-years "$SCORE" --tag "_c${REP}"
