#!/bin/bash
# Single-year pilot of the steady-state arm. Serves both calibration roles:
# with no release_schedule_<box>.csv on disk the release falls back to a constant
# rate, which is exactly the run calibrate_release.py needs to estimate g(u);
# once a schedule exists the same job re-runs as the flat-stock verification.
# Keeps the zarr either way.
#
# SPIN sets how far g(u) can be traced: a float released at -SPIN reaches age
# SPIN + days. Calibrating a new box wants SPIN past the lifespan median, so the
# decay of g is physical rather than a truncation artefact.
#
#   sbatch code/osse/submit_steady_pilot.slurm                          # 2015, 4 reps, boxB
#   sbatch code/osse/submit_steady_pilot.slurm 2015 6 boxA 1080 307 _calib
#
#SBATCH --job-name=osse_steady_pilot
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/osse_steady_pilot_%j.out
#SBATCH --error=logs/osse_steady_pilot_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

YEAR="${1:-2015}"
REPS="${2:-4}"
BOX="${3:-boxB}"
SPIN="${4:-540}"
NFLOATS="${5:-307}"
TAG="${6:-_verify}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

SCHED="data/osse/release_schedule_${BOX}.csv"
if [ -f "$SCHED" ]; then
    echo "=== $SCHED present: this pilot VERIFIES the calibrated schedule ==="
else
    echo "=== $SCHED absent: constant-rate release, this pilot CALIBRATES g(u) ==="
fi

echo "=== steady-arm pilot: year $YEAR, $REPS reps, box $BOX,"
echo "    spin ${SPIN} d, $NFLOATS floats/rep, tag '$TAG' ==="
"$PY" code/osse/design_invariance.py \
    --source steady --years "$YEAR" --reps "$REPS" --box "$BOX" \
    --spin-days "$SPIN" --n-floats-uniform "$NFLOATS" \
    --tag "$TAG" --keep-zarr
