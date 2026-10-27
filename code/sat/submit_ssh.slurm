#!/bin/bash
# DUACS L4 SSH -> 10-day native-resolution aggregates (code/sat/ssh.py), one 360-day chunk per array task.
#   N=$(python code/sat/ssh.py --n-chunks); sbatch --array=0-$((N-1))%6 code/sat/submit_ssh.slurm
#SBATCH --job-name=ssh_10d
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ssh_10d_%A_%a.out
#SBATCH --error=logs/ssh_10d_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/sat/ssh.py --chunk "$SLURM_ARRAY_TASK_ID"
