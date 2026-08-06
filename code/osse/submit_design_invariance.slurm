#!/bin/bash
# Lagrangian design-invariance (D1b) diagnostics over every complete velocity
# year, followed by the exchangeability tests.
#
# Run AFTER submit_velocity_backfill.slurm has landed 2010-2014: the whole point
# is that 5 years gives little power against a trend in the design distribution,
# and the Eulerian diagnostics show EKE rising and the jet migrating south
# monotonically across 2015-2019.
#
#   mkdir -p logs
#   sbatch code/osse/submit_design_invariance.slurm            # all complete years
#   sbatch code/osse/submit_design_invariance.slurm 20 boxA    # reps, box
#
#SBATCH --job-name=osse_d1b
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/osse_d1b_%j.out
#SBATCH --error=logs/osse_d1b_%j.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

REPS="${1:-10}"
BOX="${2:-boxB}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== Eulerian velocity statistics ==="
for band in park surf; do
  "$PY" code/osse/velocity_diagnostics.py --band "$band" --box "$BOX"
done

echo
echo "=== Lagrangian design statistics (reps=$REPS) ==="
"$PY" code/osse/design_invariance.py --reps "$REPS" --box "$BOX"

echo
echo "=== D1b exchangeability tests ==="
module load r/4.5.0 netcdf/4.9.2
Rscript code/osse/design_invariance.R "$BOX" 9999
