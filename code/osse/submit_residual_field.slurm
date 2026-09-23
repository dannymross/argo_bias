#!/bin/bash
# Mean field + residual field for one region from the OHC layer store.
#   sbatch code/osse/submit_residual_field.slurm buffer
#SBATCH --job-name=resid_field
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/resid_field_%j.out
#SBATCH --error=logs/resid_field_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
REGION="${1:?usage: sbatch submit_residual_field.slurm <region>}"
/storage/work/dmr6737/.conda/envs/argo_bias/bin/python code/osse/build_residual_field.py \
    "$REGION" --years 2010,2011,2012,2013,2014,2015,2016,2017,2018,2019
