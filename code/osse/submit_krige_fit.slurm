#!/bin/bash
# Global space-time covariance MLE for the simple-kriging estimator.
#   sbatch code/osse/submit_krige_fit.slurm [coarse] [n_blocks]
#SBATCH --job-name=krige_fit
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/krige_fit_%j.out
#SBATCH --error=logs/krige_fit_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load r/4.5.0 2>/dev/null || true
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osse/krige.py \
    --coarse "${1:-12,12,20}" --n-blocks "${2:-3}" --covfun "${3:-exponential_scaledim}"
