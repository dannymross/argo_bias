#!/bin/bash
# Evaluate the residual field at every profile of one arm, then rebuild that
# arm's Monte-Carlo summaries. Memory-bound rather than CPU-bound: the whole
# profile table is concatenated in memory before the gather.
#
# Output is sampled_residuals_<TAG>_<BOX>.csv: the box is always in the name, so
# the two boxes' gathers of the same arm cannot overwrite each other.
#
#   sbatch code/osse/submit_gather_arm.slurm steady _steady steady boxB
#   sbatch code/osse/submit_gather_arm.slurm steady _steady steady boxA
#   sbatch code/osse/submit_gather_arm.slurm uniform "" uniform_mc boxB
#
#SBATCH --job-name=osse_gather_arm
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Sized from measured peak RSS: 18.6 GB for box B's ~45M concatenated profile
# rows, so ~27 GB for box A's ~66M. Scale this with the row count, not upward
# "for safety" -- the request, not the usage, is what is charged and queued on.
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/osse_gather_arm_%j.out
#SBATCH --error=logs/osse_gather_arm_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARM="${1:-steady}"
SUFFIX="${2:-_$ARM}"
TAG="${3:-$ARM}"
BOX="${4:-boxB}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== gathering arm '$ARM' ($BOX) -> sampled_residuals_${TAG}_${BOX}.csv ==="
"$PY" code/osse/mean_mc.py --box "$BOX" --gather "$ARM" --tag "$TAG" --arm "$SUFFIX"

echo "=== done ==="
