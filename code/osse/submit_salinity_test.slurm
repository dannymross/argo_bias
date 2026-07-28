#!/bin/bash
# Constant-vs-variable-density OHC test: decide whether the full pull needs `so`.
# Downloads a short thetao+so window over one box and prints the percent difference.
#
#   mkdir -p logs
#   sbatch code/osse/submit_salinity_test.slurm
#
#SBATCH --job-name=osse_saltest
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/osse_saltest_%j.out
#SBATCH --error=logs/osse_saltest_%j.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
source activate argo_bias
cd "$PROJECT_ROOT"
python code/osse/salinity_test.py
