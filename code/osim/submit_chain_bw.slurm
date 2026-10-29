#!/bin/bash
# q-hat bandwidth sensitivity of the chain (code/osim/chain.py): task = bandwidth * 20 + chunk, eval reps 0-199.
#
#   sbatch --array=0-159 code/osim/submit_chain_bw.slurm                       # points, then DR estimates
#   sbatch --array=0-7 --export=ALL,WHAT=nodes code/osim/submit_chain_bw.slurm   # node q-hats, task = bandwidth
#
#SBATCH --job-name=chain_bw
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chain_bw_%A_%a.out
#SBATCH --error=logs/chain_bw_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
BW=(0.08333,2 0.08333,4 0.08333,8 0.16666666666666666,2 0.16666666666666666,8 0.3333,2 0.3333,4 0.3333,8)
WHAT="${WHAT:-points}"
if [ "$WHAT" = nodes ]; then
  CHAIN_SIGMA="${BW[$SLURM_ARRAY_TASK_ID]}" "$PY" code/osim/chain.py nodes
  exit
fi
export CHAIN_SIGMA="${BW[$((SLURM_ARRAY_TASK_ID / 20))]}"
CHUNK=$((SLURM_ARRAY_TASK_ID % 20))
"$PY" code/osim/chain.py points --chunk "$CHUNK" --nchunk 20 --maxrep 200
