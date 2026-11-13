#!/bin/bash
# Production DR with per-layer q-hat kernels (QSIG_LAYER) for r10-109, then the per-cell spatial diagnostics.
#SBATCH --job-name=dr_layers
#SBATCH --account=statsresearch_sc_default
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dr_layers_%j.out
#SBATCH --error=logs/dr_layers_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
$PY code/osim/natl_sim.py estimate_layers --reps 10,109
$PY code/osim/natl_sim.py spatial_diag --reps 10,109
