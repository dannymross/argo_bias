#!/bin/bash
# Map + cross-fit DR for one replicate per task (code/osim/chain.py rep); needs the points and node q-hats.
#
#   sbatch --array=0-19 code/osim/submit_chain.slurm
#
#SBATCH --job-name=chain_rep
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chain_rep_%A_%a.out
#SBATCH --error=logs/chain_rep_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
"$PY" code/osim/chain.py rep --rep "$SLURM_ARRAY_TASK_ID"
