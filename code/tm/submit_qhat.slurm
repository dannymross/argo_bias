#!/bin/bash
# Transition-matrix q-hat, one launch-year cohort per array task (cohorts add linearly), then a merge
# that sums them and writes q-hat for each profile set. Build P_move and survival first (run.py without
# --qhat) so tasks only read the caches. Extra run.py flags pass through to both stages.
#
#   python code/tm/run.py [flags]                                   # caches P_move, survival
#   N=$(python -c "import sys; sys.path.insert(0,'code'); from tm import run; print(len(run.launch_years(run.Config())) - 1)")
#   jid=$(sbatch --parsable --array=0-$N code/tm/submit_qhat.slurm cohort [flags])
#   sbatch --dependency=afterok:$jid code/tm/submit_qhat.slurm merge [flags]
#
#SBATCH --job-name=tm_qhat
#SBATCH --partition=standard
#SBATCH --account=open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=logs/tm_qhat_%A_%a.out
#SBATCH --error=logs/tm_qhat_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python
STAGE="$1"; shift

if [ "$STAGE" = cohort ]; then
    "$PY" code/tm/run.py --cohort-index "$SLURM_ARRAY_TASK_ID" "$@"
else
    for ps in dedup eligible; do
        "$PY" code/tm/run.py --qhat --profile-set "$ps" "$@"
    done
fi
