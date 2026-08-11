#!/bin/bash
# Monte-Carlo the design distribution for the steady-state uniform arm: floats
# deployed at a constant rate from SPIN days before the analysis year, so the
# in-box standing stock is flat across [0, 360) instead of ramping from empty.
#
# The spin-up runs on the PRECEDING years' velocity, so the earliest analysis
# year is bounded by the velocity archive (2010-2019 -> 2012-2019 at SPIN=540).
#
# Replicates are split across chunks because the RNG is keyed on (year, rep):
# --rep-offset is what keeps chunks from redrawing identical designs.
#
#   mkdir -p logs
#   sbatch code/osse/submit_steady_mc.slurm              # 5 chunks x 150 reps
#   sbatch code/osse/submit_steady_mc.slurm 100 boxB     # 5 chunks x 100 reps
#
#SBATCH --job-name=osse_steady_mc
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/osse_steady_mc_%A_%a.out
#SBATCH --error=logs/osse_steady_mc_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

REPS="${1:-150}"
BOX="${2:-boxB}"
SPIN="${3:-540}"
NFLOATS="${4:-450}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

YEARS=(2012 2013 2014 2015 2016 2017 2018 2019)
NCHUNK=5
YEAR="${YEARS[$((SLURM_ARRAY_TASK_ID / NCHUNK))]}"
CHUNK=$((SLURM_ARRAY_TASK_ID % NCHUNK))
OFFSET=$((CHUNK * REPS))

echo "=== steady-arm MC: year $YEAR, chunk $CHUNK, reps $REPS (offset $OFFSET),"
echo "    box $BOX, spin ${SPIN} d, $NFLOATS floats/rep ==="
"$PY" code/osse/design_invariance.py \
    --source steady --years "$YEAR" --reps "$REPS" --box "$BOX" \
    --spin-days "$SPIN" --n-floats-uniform "$NFLOATS" \
    --rep-offset "$OFFSET" --tag "_y${YEAR}_c${CHUNK}"
