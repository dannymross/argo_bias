#!/bin/bash
# One advection benchmark (code/osim/bench_velocity.py): 1190 floats, all released
# 2015-01-01, advected one year through the given velocity directory. Submit one job
# per variant with the same resources, then compare:
#
#   sbatch code/osim/submit_velocity_bench.slurm orig data/glorys_osim/velocity_daily
#   sbatch code/osim/submit_velocity_bench.slurm c1000          # default: deployment.SIM_VEL_DIR
#   python code/osim/bench_velocity.py compare orig c1000
#   # production-like: 10-yr window, releases spread over it, extra args passed through
#   sbatch code/osim/submit_velocity_bench.slurm p_orig data/glorys_osim/velocity_daily --start 2010-01-01 --days 3647 --spread
#
#SBATCH --job-name=osim_vel_bench
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/osim_vel_bench_%j.out
#SBATCH --error=logs/osim_vel_bench_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

LABEL="$1"
VEL="${2:-}"
shift $(( $# < 2 ? $# : 2 ))
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
ARGS=(run --label "$LABEL")
[ -n "$VEL" ] && ARGS+=(--vel-dir "$VEL")
ARGS+=("$@")
echo "node $(hostname), TMPDIR=${TMPDIR:-unset}"; df -h "${TMPDIR:-/tmp}" | tail -1

"$PY" code/osim/bench_velocity.py "${ARGS[@]}"
