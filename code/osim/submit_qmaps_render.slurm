#!/bin/bash
# Annual 2020 q-hat maps at the report's kernel sds (all 200 reps), then render glorys_chain.
#SBATCH --job-name=qmaps_render
#SBATCH --account=statsresearch_sc_default
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/qmaps_render_%j.out
#SBATCH --error=logs/qmaps_render_%j.err

source ~/.bashrc
conda activate argo_bias
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
python code/osim/natl_sim.py qmaps --sig 0.3333,5 --sig 0.5,5 --sig 1,5 --sig 1,30 --sig 2,30 --sig 5,30
quarto render reports/glorys_chain.qmd
