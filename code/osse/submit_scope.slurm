#!/bin/bash
# Scope the GLORYS OSSE download size + wall-time on small slices (RUN FIRST).
# Short job; reads code/osse/configs/osse_glorys.json, prints an estimate table.
#
#   mkdir -p logs
#   sbatch code/osse/submit_scope.slurm
#
#SBATCH --job-name=osse_scope
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/osse_scope_%j.out
#SBATCH --error=logs/osse_scope_%j.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
source activate argo_bias
cd "$PROJECT_ROOT"
python code/osse/scope_download.py
