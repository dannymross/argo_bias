#!/bin/bash
# Gyre masks from monthly zos; run after submit_zos_monthly.slurm.
#
#   sbatch --dependency=afterok:<zos job> code/osim/submit_gyre_masks.slurm
#
#SBATCH --job-name=gyre_masks
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gyre_masks_%j.out
#SBATCH --error=logs/gyre_masks_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PYTHONPATH=code /storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osim/run_gyre_masks.py
