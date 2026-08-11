#!/bin/bash
# Monte-Carlo the design distribution for the space-time uniform arm: many
# independent seedings per velocity year, one array task per year.
#
# Replicates are just labels on one ParticleSet, so R seedings of n floats cost
# the same as one run of R*n floats -- and because the seeds are iid and the
# flow is common, they are exactly R iid draws of omega through V_l.
#
#   mkdir -p logs
#   sbatch code/osse/submit_uniform_mc.slurm            # 200 reps, boxB
#   sbatch code/osse/submit_uniform_mc.slurm 100 boxA
#
#SBATCH --job-name=osse_unif_mc
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=2010-2019
#SBATCH --output=logs/osse_unif_mc_%A_%a.out
#SBATCH --error=logs/osse_unif_mc_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

REPS="${1:-200}"
BOX="${2:-boxB}"
YEAR="$SLURM_ARRAY_TASK_ID"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== uniform-arm MC: year $YEAR, $REPS seedings, box $BOX ==="
"$PY" code/osse/design_invariance.py \
    --source uniform --years "$YEAR" --reps "$REPS" --box "$BOX" \
    --tag "_y${YEAR}"
