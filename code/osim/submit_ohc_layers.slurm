#!/bin/bash
# Download GLORYS thetao over the analysis buffer and collapse it to the four OHC
# depth layers, one array task per YEAR (the array index IS the year). Raw thetao
# is deleted after each month, so the job is disk-cheap; it is network-bound.
#
#   sbatch --array=2010-2019 code/osim/submit_ohc_layers.slurm daily
#   sbatch --array=1993-2019 code/osim/submit_ohc_layers.slurm monthly
#
# Resumable: a month whose output exists is skipped, so a re-submit fills gaps.
#
#SBATCH --job-name=ohc_layers
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ohc_layers_%A_%a.out
#SBATCH --error=logs/ohc_layers_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CADENCE="${1:?usage: sbatch --array=<years> submit_ohc_layers.slurm <daily|monthly>}"
YEAR="${SLURM_ARRAY_TASK_ID:?this script must be run as an array job}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

# Per-task scratch so concurrent tasks cannot collide on a raw filename.
RAW="data/glorys_osim/_raw_tmp/${SLURM_ARRAY_JOB_ID}_${YEAR}"
mkdir -p "$RAW"
trap 'rm -rf "$RAW"' EXIT

echo "=== $CADENCE $YEAR ==="
"$PY" code/osim/download_ohc_layers.py "$CADENCE" \
      --start "${YEAR}-01" --end "${YEAR}-12" --raw-dir "$RAW"
