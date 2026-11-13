#!/bin/bash
# Commit and push the glorys_chain report after its render job succeeds.
#SBATCH --job-name=commit_push
#SBATCH --account=statsresearch_sc_default
#SBATCH --partition=sla-prio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/commit_push_%j.out
#SBATCH --error=logs/commit_push_%j.err

source ~/.bashrc
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
git add code/osim/natl_sim.py code/osim/natl_results.py code/osim/submit_qmaps_render.slurm \
  code/osim/submit_commit_push.slurm reports/glorys_chain.qmd _freeze/reports/glorys_chain
git commit -m "glorys_chain: single Bandwidth section with 2020 q-hat maps at six kernels (DR kernels marked), Hajek tied to the DR kernels, reweighting-effect tab in the yearly maps"
git push origin main
