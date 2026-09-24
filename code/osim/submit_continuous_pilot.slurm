#!/bin/bash
# Pilot of the continuous-seeding arm (code/osim/continuous_sim.py): one advection
# per realisation over the velocity window, scored on BOTH boxes from the same
# deployment. Each array task is one independent realisation (RNG keyed on the
# rep index = SLURM_ARRAY_TASK_ID), so the ensemble is built by fanning tasks out
# across the array -- ~100 realisations finish in about the wall time of one.
#
# Default: deploy from 2010, score 2015-2019 (2010-2014 is the equilibrium lead),
# which needs only the 2010-2019 velocity already on disk. Each rep is ~3 h at
# 1000 floats over a 10 yr window.
#
#   sbatch --array=0-99 code/osim/submit_continuous_pilot.slurm            # 100 reps, defaults
#   sbatch --array=0-99 code/osim/submit_continuous_pilot.slurm 1400 2005-01-01 2010,...,2019
#   # then, once all tasks finish:
#   python code/osim/continuous_sim.py --gather --tag _pilot
#
#SBATCH --job-name=osim_continuous
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/osim_continuous_%A_%a.out
#SBATCH --error=logs/osim_continuous_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

NFLOATS="${1:-1000}"
START="${2:-2010-01-01}"
SCORE="${3:-2015,2016,2017,2018,2019}"
END="${4:-}"                 # blank => a few days inside the last daily field
TAG="${5:-_pilot}"
REP="${SLURM_ARRAY_TASK_ID:-0}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

ARGS=(--n-floats "$NFLOATS" --reps 1 --rep-offset "$REP"
      --start "$START" --score-years "$SCORE" --tag "${TAG}_c${REP}")
[ -n "$END" ] && ARGS+=(--end "$END")

echo "=== continuous rep $REP: $NFLOATS floats, from $START, score $SCORE ==="
"$PY" code/osim/continuous_sim.py "${ARGS[@]}"
