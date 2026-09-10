#!/bin/bash
# Backfill daily uo/vo (park + surf bands) over the extended velocity window in
# code/osse/configs/osse_glorys.json -- one array task per year.
#
# The velocity record is extended back to 2005 so floats seeded uniformly over
# 2005-2019 spacetime reach renewal equilibrium by the first analysis year
# (2010) in one continuous advection; only the velocity record needs extending,
# since thetao/zos (which define the analysis years) stay on their own window.
#
#   mkdir -p logs
#   sbatch code/osse/submit_velocity_backfill.slurm            # 2005-2019
#   sbatch --array=0-2 code/osse/submit_velocity_backfill.slurm  # 2005-2007 only
#
# Resumable: existing files are skipped, so already-downloaded years finish
# immediately and a timed-out or requeued task picks up where it stopped.
# ~1.4 GB/yr at ~0.43 MB/s sustained => ~1 h per task.
#
#SBATCH --job-name=osse_vel_backfill
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-14%5
#SBATCH --output=logs/osse_vel_backfill_%A_%a.out
#SBATCH --error=logs/osse_vel_backfill_%A_%a.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
source activate argo_bias
cd "$PROJECT_ROOT"

START_YEAR="${START_YEAR:-2005}"
YEAR=$((START_YEAR + SLURM_ARRAY_TASK_ID))

echo "array task ${SLURM_ARRAY_TASK_ID}: velocity_daily ${YEAR}"
python code/osse/run_download.py \
  --only velocity_daily \
  --start "${YEAR}-01" \
  --end "${YEAR}-12"
