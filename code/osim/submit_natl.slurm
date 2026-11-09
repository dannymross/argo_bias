#!/bin/bash
# GLORYS for the deep North Atlantic: one array task per year (the array index IS the
# year). Network-bound; resumable, so a re-submit fills gaps. Pilot first, it writes
# the static grid the array reads.
#
#   sbatch --array=2015 --time=03:00:00 code/osim/submit_natl.slurm 2015-01 2015-01    # pilot
#   sbatch --array=2010-2025%16 code/osim/submit_natl.slurm
#
#SBATCH --job-name=natl_dl
#SBATCH --partition=basic
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --output=logs/natl_dl_%A_%a.out
#SBATCH --error=logs/natl_dl_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

YEAR="${SLURM_ARRAY_TASK_ID:?this script must be run as an array job}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

"$PY" code/osim/download_natl.py --start "${1:-$YEAR-01}" --end "${2:-$YEAR-12}" --check
