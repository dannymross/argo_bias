#!/bin/bash
# Convert the park velocity to a 2D float32 field at PARK_DEPTH, one chunk per day
# (code/osim/convert_velocity.py), then verify every month against the source.
# ~5 s/month -> ~10 min for 2010-2019. Skips months already converted.
#
#   sbatch code/osim/submit_velocity_convert.slurm                   # all years, zlib 1
#   sbatch code/osim/submit_velocity_convert.slurm 2015 0 data/glorys_osim/velocity_daily_1000_c0
#
#SBATCH --job-name=osim_vel_convert
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/osim_vel_convert_%j.out
#SBATCH --error=logs/osim_vel_convert_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

YEARS="${1:-}"
LEVEL="${2:-1}"
OUT="${3:-data/glorys_osim/velocity_daily_1000}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
ARGS=(--complevel "$LEVEL" --out-dir "$OUT")
[ -n "$YEARS" ] && ARGS+=(--years "$YEARS")

"$PY" code/osim/convert_velocity.py "${ARGS[@]}"
"$PY" code/osim/convert_velocity.py --verify "${ARGS[@]}"
