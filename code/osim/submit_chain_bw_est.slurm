#!/bin/bash
# DR estimates under each sensitivity bandwidth (after submit_chain_bw.slurm): task = bandwidth, reps 0-199.
#
#   sbatch --array=0-7 --dependency=afterok:<points>:<nodes> code/osim/submit_chain_bw_est.slurm
#
#SBATCH --job-name=chain_bw_est
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chain_bw_est_%A_%a.out
#SBATCH --error=logs/chain_bw_est_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
BW=(0.08333,2 0.08333,4 0.08333,8 0.16666666666666666,2 0.16666666666666666,8 0.3333,2 0.3333,4 0.3333,8)
export CHAIN_SIGMA="${BW[$SLURM_ARRAY_TASK_ID]}"
for r in $(seq 0 199); do "$PY" code/osim/chain.py estimate --rep "$r" > /dev/null; done
