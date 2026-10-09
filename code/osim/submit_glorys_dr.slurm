#!/bin/bash
# Leave-one-float-out kriging of the GLORYS-OHC testbed (code/osim/glorys_dr.py), one target month
# (MONTH0 + task id, default MONTH0=2015-01) per array task.
#
#   python code/osim/glorys_dr.py truth && python code/osim/glorys_dr.py obs && python code/osim/glorys_dr.py mean
#   sbatch --array=0-59 code/osim/submit_glorys_dr.slurm
#   sbatch --export=ALL,MONTH0=2010-01,GLORYS_DR_KTAG=_argomean --array=0-59 code/osim/submit_glorys_dr.slurm
#
#SBATCH --job-name=glorys_dr_krige
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/glorys_dr_krige_%A_%a.out
#SBATCH --error=logs/glorys_dr_krige_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
MONTH0="${MONTH0:-2015-01}"
MONTH=$("$PY" -c "import pandas as pd; print((pd.Period('$MONTH0', 'M') + $SLURM_ARRAY_TASK_ID).strftime('%Y-%m'))")
echo "=== krige $MONTH on $(hostname) ==="
"$PY" code/osim/glorys_dr.py krige --month "$MONTH"
