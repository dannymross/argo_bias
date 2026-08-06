#!/bin/bash
# Full GLORYS OSSE download (multi-GB). DO NOT submit until submit_scope.slurm
# has cleared the storage budget. Serial, network/IO-bound; resumable, so a
# timeout/requeue re-runs and skips files already on disk.
#
#   mkdir -p logs
#   sbatch code/osse/submit_download.slurm                    # all groups
#   sbatch code/osse/submit_download.slurm velocity_daily     # one group
#
# Set --time from the scope MB/s x total size, plus margin.
#
#SBATCH --job-name=osse_download
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/osse_download_%j.out
#SBATCH --error=logs/osse_download_%j.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
source activate argo_bias
cd "$PROJECT_ROOT"

ONLY_ARG=""
[ -n "${1:-}" ] && ONLY_ARG="--only $1"
python code/osse/run_download.py $ONLY_ARG
