#!/bin/bash
# Configure-and-run driver for Argo trajectory simulations.
#
#   ./run_experiment.sh <config.json> [local|slurm]
#
# Reads an experiment JSON (see configs/nac_default.json), then either runs all
# float batches locally across local_cores processes, or submits a SLURM array
# job (one task per batch). Mode defaults to 'slurm' if sbatch is on PATH,
# otherwise 'local'.
#
# Examples:
#   ./run_experiment.sh configs/nac_default.json           # auto-detect
#   ./run_experiment.sh configs/nac_default.json local     # force local
#   ./run_experiment.sh configs/gulf_stream.json slurm     # force SLURM

set -euo pipefail

CONFIG="${1:?usage: run_experiment.sh <config.json> [local|slurm]}"
MODE="${2:-auto}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # code/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }
# Absolute path so a SLURM compute node can find it too.
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# Extract shell-side orchestration values from the JSON via Python.
# All sim parameters are passed via --config to run_trajsim.py directly.
_json() { python3 -c "import json,sys; c=json.load(open('$CONFIG')); print(c$1)" 2>/dev/null; }

EXPERIMENT="$(_json "['experiment']")"
PYTHON="$(_json ".get('python','python')")"
NTASKS="$(_json "['ntasks']")"
LOCAL_CORES="$(_json "['local_cores']")"
SLURM_TIME="$(_json ".get('slurm',{}).get('time','12:00:00')")"
SLURM_MEM="$(_json ".get('slurm',{}).get('mem','16G')")"
SLURM_CPUS="$(_json ".get('slurm',{}).get('cpus',1)")"
SLURM_CONDA_ENV="$(_json ".get('slurm',{}).get('conda_env','virtualfleet')")"

if [ "$MODE" = "auto" ]; then
    if command -v sbatch >/dev/null 2>&1; then MODE=slurm; else MODE=local; fi
fi

# Relative paths in the config (data/..., output dirs) resolve from the root.
cd "$PROJECT_ROOT"

case "$MODE" in
    local)
        echo "[$EXPERIMENT] local run: $NTASKS batches over $LOCAL_CORES cores"
        "$PYTHON" "$SCRIPT_DIR/run_trajsim.py" \
            --config "$CONFIG" \
            --ntasks "$NTASKS" \
            --local-cores "$LOCAL_CORES"
        ;;
    slurm)
        echo "[$EXPERIMENT] submitting SLURM array 0-$((NTASKS - 1))"
        mkdir -p "$PROJECT_ROOT/logs"
        sbatch \
            --job-name="$EXPERIMENT" \
            --array="0-$((NTASKS - 1))" \
            --time="$SLURM_TIME" \
            --mem="$SLURM_MEM" \
            --cpus-per-task="$SLURM_CPUS" \
            --output="$PROJECT_ROOT/logs/${EXPERIMENT}_%A_%a.out" \
            --error="$PROJECT_ROOT/logs/${EXPERIMENT}_%A_%a.err" \
            --export=ALL,CONFIG="$CONFIG",PYTHON="$PYTHON",SLURM_CONDA_ENV="$SLURM_CONDA_ENV" \
            "$SCRIPT_DIR/submit_trajsim.slurm"
        ;;
    *)
        echo "unknown mode: $MODE (use 'local' or 'slurm')" >&2
        exit 1
        ;;
esac
