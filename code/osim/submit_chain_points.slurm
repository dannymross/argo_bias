#!/bin/bash
# Chain inputs (code/osim/chain.py): in-box profiles of replicates r = task mod 20 with Z and cohort q-hats.
#
#   sbatch --array=0-19 code/osim/submit_chain_points.slurm
#   sbatch --export=ALL,WHAT=nodes code/osim/submit_chain_points.slurm      # node q-hats (one task)
#   sbatch --export=ALL,WHAT=align code/osim/submit_chain_points.slurm      # Cov(Z, w-hat), after points
#
#SBATCH --job-name=chain_points
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chain_points_%A_%a.out
#SBATCH --error=logs/chain_points_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
WHAT="${WHAT:-points}"
if [ "$WHAT" = points ]; then
  "$PY" code/osim/chain.py points --chunk "${SLURM_ARRAY_TASK_ID:-0}" --nchunk 20
else
  "$PY" code/osim/chain.py "$WHAT"
fi
