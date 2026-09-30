#!/bin/bash
# Year-stratified smoothed bootstrap of the real Argo deployment pattern
# (code/osim/continuous_sim.py --argo resample), the ensemble behind q-hat. Each array
# task is one replicate (RNG keyed on SLURM_ARRAY_TASK_ID): the 994 domain-wide 2010-2019
# metadata launches resampled with replacement within year (2010-2014 = ramp-up), jittered
# +/-1/12 deg and +/-10 d, advected 2010-2019 through the converted park-depth field,
# daily positions kept. Output per task: design_profiles_continuous_argo_resample
# <PREFIX>_c<rep>.parquet (profiles tagged with pool_idx / argo_id / source).
#
#   python code/osim/continuous_sim.py --argo-pool          # once: cache the pool
#   sbatch --array=0-99 code/osim/submit_argo_resample.slurm
#   sbatch --array=0-49 code/osim/submit_argo_resample.slurm _wide 1.0 30   # bandwidth run
#   sbatch --array=... --mem=10G --time=01:00:00 code/osim/submit_argo_resample.slurm "" 0.08333333333333333 10 10  # 10-day output
#   python code/osim/continuous_sim.py --gather --tag _argo_resample     # any time
#
#SBATCH --job-name=osim_argo_resample
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/osim_argo_resample_%A_%a.out
#SBATCH --error=logs/osim_argo_resample_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# node-local Parcels output (continuous_sim.scratch_dir); removed even on failure
trap 'rm -rf "${TMPDIR:-/tmp}/osim_${USER}_${SLURM_JOB_ID}"' EXIT

PREFIX="${1:-}"
JDEG="${2:-0.08333333333333333}"
JDAYS="${3:-10}"
OUTDAYS="${4:-1}"      # 1 = daily tracks; 10 = 10-day outputs (~6x cheaper)
REP="${SLURM_ARRAY_TASK_ID:-0}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== argo-resample rep $REP (prefix '${PREFIX}', jitter ${JDEG} deg, ${JDAYS} d) on $(hostname) ==="
"$PY" code/osim/continuous_sim.py \
    --argo resample --reps 1 --rep-offset "$REP" \
    --jitter-deg "$JDEG" --jitter-days "$JDAYS" --output-days "$OUTDAYS" --tag "${PREFIX}_c${REP}"
