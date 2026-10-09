#!/bin/bash
# Locally stationary covariance MLE, one array task per contiguous run of windows.
# Contiguous (not strided) so the warm start inside each task sees grid neighbours.
#   sbatch --array=0-49 code/osim/submit_krige_local.slurm
#   python code/osim/krige_local.py --gather      # concat + median over tau*
#   sbatch --array=0-49 --account=open code/osim/submit_krige_local.slurm 50 argomean data/osim/glorys_dr/krige_local_lattice_argomean.csv
#SBATCH --job-name=krige_local
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/krige_local_%A_%a.out
#SBATCH --error=logs/krige_local_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NTASK="${1:-50}"
TAG="${2:-oracle}"
T="${SLURM_ARRAY_TASK_ID:?must run as an array job}"
LATTICE="${3:-}"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osim/krige_local_fit.py \
    "data/osim/krige_local_${TAG}_t${T}.csv" "$T" "$NTASK" ${LATTICE:+--lattice "$LATTICE"}
