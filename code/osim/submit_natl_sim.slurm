#!/bin/bash
# One natl_sim.py step per array task; the array index is the replicate.
#   sbatch --array=0-1 code/osim/submit_natl_sim.slurm advect
#SBATCH --job-name=natl_sim
#SBATCH --partition=basic
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/natl_sim_%x_%A_%a.out
#SBATCH --error=logs/natl_sim_%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osim/natl_sim.py "$@"
