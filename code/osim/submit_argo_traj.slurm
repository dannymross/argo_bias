#!/bin/bash
# Full-trajectory actual-deployment advection (code/osim/continuous_sim.py --argo-traj):
# advect each real Argo deployment (verbatim seed position/date/lifespan) through
# GLORYS park velocity, 2010-2019, KEEPING every 10-day position (in-box and out).
# Writes actual_full_tracks.csv + argo_actual_pool.csv for the trajectory-divergence
# attribution (mean_mc.trajectory_divergence): real vs GLORYS-park separation over time.
#
#   sbatch code/osim/submit_argo_traj.slurm
#
#SBATCH --job-name=osim_argo_traj
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/osim_argo_traj_%j.out
#SBATCH --error=logs/osim_argo_traj_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# node-local Parcels output (continuous_sim.scratch_dir); removed even on failure
trap 'rm -rf "${TMPDIR:-/tmp}/osim_${USER}_${SLURM_JOB_ID}"' EXIT

START="${1:-2010-01-01}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

echo "=== argo-traj: full-path advection of real deployments from $START ==="
"$PY" code/osim/continuous_sim.py --argo-traj --start "$START"
