#!/bin/bash
# Monthly GLORYS zos 1993-2025 over the North Atlantic for the gyre masks; skips finished years.
#
#   sbatch code/osim/submit_zos_monthly.slurm
#
#SBATCH --job-name=zos_monthly
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/zos_monthly_%j.out
#SBATCH --error=logs/zos_monthly_%j.err
set -uo pipefail
export PATH=/storage/work/dmr6737/.conda/envs/argo_bias/bin:$PATH
bash data/glorys_osim/zos_monthly/fetch.sh
