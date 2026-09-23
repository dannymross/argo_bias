#!/bin/bash
# SUPERSEDED by submit_ohc_layers.slurm. The raw thetao_daily store this backfills
# has been deleted -- nothing reads it any more (build_residual_field.py now reads
# data/glorys_osse/ohc_layers/). Running this would re-download ~27 GB for nothing.
# Rebuild the residual field over a wider analysis window, then re-gather and
# re-summarise every arm that depends on it. Queue this AFTER the download that
# supplies the extra daily thetao:
#
#   sbatch --dependency=afterok:<download_jobid> code/osse/submit_extend_years.slurm
#   sbatch code/osse/submit_extend_years.slurm 2010,2011,...,2019 boxB
#
# The mean-field fit itself is unchanged -- it is fitted from the 1993-2019
# monthly record, which is already on disk; only the daily evaluation window
# grows. mean_mc.py reads its year list from the rebuilt field, so no source
# edit is needed at cutover.
#
#SBATCH --job-name=osse_extend_years
#SBATCH --partition=standard
#SBATCH --account=statsresearch_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/osse_extend_years_%j.out
#SBATCH --error=logs/osse_extend_years_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

YEARS="${1:-2010,2011,2012,2013,2014,2015,2016,2017,2018,2019}"
BOX="${2:-boxB}"
PY=/storage/work/dmr6737/.conda/envs/argo_bias/bin/python

# Refuse to start on a partial download: a missing month would silently shorten
# the field rather than fail, and every downstream z_<year> column would be wrong.
echo "=== checking daily thetao coverage for $YEARS ==="
"$PY" - "$YEARS" <<'EOF'
import os, sys, glob
years = [int(y) for y in sys.argv[1].split(",")]
d = "data/glorys_osse/thetao_daily"
missing = [f"{y}-{m:02d}" for y in years for m in range(1, 13)
           if not os.path.exists(os.path.join(d, f"thetao_daily_{y}_{m:02d}.nc"))]
if missing:
    raise SystemExit(f"ABORT: {len(missing)} daily thetao months missing, first {missing[0]}")
print(f"all {12*len(years)} months present")
EOF

echo "=== rebuilding residual field: $BOX, years $YEARS ==="
"$PY" code/osse/build_residual_field.py "$BOX" --years "$YEARS"

# A box whose arms have not been advected yet has nothing to re-gather; that is
# the normal state when standing up a new box, not an error.
if compgen -G "data/osse/design_profiles_${BOX}_uniform*.csv" > /dev/null; then
    echo "=== re-gathering the ramp-up uniform arm against the new field ==="
    "$PY" code/osse/mean_mc.py --box "$BOX" --gather uniform --tag uniform_mc

    echo "=== re-running the Monte-Carlo summaries ==="
    "$PY" code/osse/mean_mc.py --box "$BOX"
    echo "=== done; reports/mean_estimation_mc.qmd needs a re-render ==="
else
    echo "=== no uniform arm for $BOX yet: field built, nothing to re-gather ==="
fi
