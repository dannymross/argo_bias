#!/bin/bash
# Full-footprint park-depth velocity download (daily uo/vo, 900-1100 m) over
# lat 11-66 N, lon 87-12 W for 2010-2019 -- the region every float that profiles
# Box A/B in 2015-2019 roams across, so simulated floats are not deleted for
# leaving the velocity domain. One array task per year (2010-2019); each task
# pulls its 12 months, compressed (int16 + deflate), into the live velocity_daily
# dir. Resumable: existing months are skipped, so a requeued/timed-out task
# resumes. Clear the OLD velocity_daily_park_*.nc first (different extent, same
# name) or the skip logic leaves a mixed-extent dir.
#
#   mkdir -p logs
#   sbatch code/osim/submit_velocity_full.slurm                 # 2010-2019
#   sbatch --array=0 code/osim/submit_velocity_full.slurm       # 2010 only
#
# ~58 MB/month, ~67 s/month => ~0.7 GB and ~13 min per year-task; ~7 GB total.
#
#SBATCH --job-name=osim_vel_full
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/osim_vel_full_%A_%a.out
#SBATCH --error=logs/osim_vel_full_%A_%a.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
source activate argo_bias
cd "$PROJECT_ROOT"

START_YEAR="${START_YEAR:-2010}"
YEAR=$((START_YEAR + SLURM_ARRAY_TASK_ID))

echo "array task ${SLURM_ARRAY_TASK_ID}: full-footprint velocity ${YEAR}"
python code/osim/download_velocity_full.py \
  --start "${YEAR}-01" \
  --end "${YEAR}-12"
