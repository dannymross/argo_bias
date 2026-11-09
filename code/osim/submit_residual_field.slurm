#!/bin/bash
# Mean field + residual field for one region from the OHC layer store.
#   sbatch code/osim/submit_residual_field.slurm buffer
#   sbatch --mem=24G code/osim/submit_residual_field.slurm natl --mean-only
#SBATCH --job-name=resid_field
#SBATCH --partition=basic
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/resid_field_%j.out
#SBATCH --error=logs/resid_field_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${1:?usage: sbatch submit_residual_field.slurm <region> [build_residual_field.py options]}"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osim/build_residual_field.py "$@"
