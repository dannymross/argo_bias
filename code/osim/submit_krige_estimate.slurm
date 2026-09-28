#!/bin/bash
# Simple-kriging estimate and its sampling bias, one array task per rep subset.
#   sbatch --array=0-24 code/osim/submit_krige_estimate.slurm 25 _pilot_wide
#SBATCH --job-name=krige_est
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/krige_est_%A_%a.out
#SBATCH --error=logs/krige_est_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NTASK="${1:-25}"; TAG="${2:-_pilot_wide}"
T="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osim/krige_estimate.py \
    "data/osim/krige_est${TAG}_t${T}.csv" "$T" "$NTASK" --tag "$TAG"
