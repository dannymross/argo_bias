#!/bin/bash
# SLURM job for the full 2021-2022 GLORYS download (24 monthly files).
#
# This is a single serial download (network/IO bound, not compute-heavy), so
# it's submitted directly rather than through run_experiment.sh's array-job
# pattern. Resumable: if it times out or is requeued, re-submitting picks up
# where it left off (fetch_velocity_months skips months already on disk).
#
# Before submitting: run code/test_download_gs_2021_2022.py interactively to
# time one month, then set --time below to roughly (per-month time x 24),
# plus margin.
#
#   mkdir -p logs
#   sbatch code/submit_download_gs_2021_2022.slurm

#SBATCH --job-name=download_gs_2021_2022
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_gs_2021_2022_%j.out
#SBATCH --error=logs/download_gs_2021_2022_%j.err

set -euo pipefail

# SLURM copies this script into a spool dir on the compute node before
# running it, so locating ourselves via BASH_SOURCE/dirname doesn't work.
# SLURM_SUBMIT_DIR is set to wherever `sbatch` was invoked from (the project
# root, per the submission instructions above) and is reliable instead.
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="$PROJECT_ROOT/code"

source activate argo_bias

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/download_gs_2021_2022.py"
