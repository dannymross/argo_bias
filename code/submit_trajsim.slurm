#!/bin/bash
# SLURM array job for Argo trajectory simulations.
#
# Don't sbatch this directly — submit through run_experiment.sh, which sets
# --array size and resources from your config and exports CONFIG to this script:
#
#   ./run_experiment.sh configs/nac_default.json slurm
#
# Each array task runs one float batch; run_trajsim.py reads
# SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT to pick its slice.
# Resource directives (--array, --time, --mem, ...) are supplied by
# run_experiment.sh on the sbatch command line — no #SBATCH lines here.

set -euo pipefail

: "${CONFIG:?CONFIG not set — submit via run_experiment.sh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # code/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the environment (adjust to your cluster's module/conda setup).
# module load anaconda3
source activate "${SLURM_CONDA_ENV:-virtualfleet}"

# Keep math libs single-threaded; the array provides the parallelism.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$PROJECT_ROOT"

# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT are auto-detected by run_trajsim.
"${PYTHON:-python}" "$SCRIPT_DIR/run_trajsim.py" --config "$CONFIG"
