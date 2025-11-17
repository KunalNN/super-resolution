#!/usr/bin/env bash

# Launch Experiment 2: adaptive depth per scale based on the architectural design table.
# Each sbatch submission reuses train_adaptive_simple.sbatch and configures the encoder
# depth to the per-scale target while keeping batch sizes within 2080 Ti limits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_EXPERIMENT_ROOT="$REPO_ROOT/experiments/experiment_2_adaptive_depth"

SCRATCH_ROOT="${SEG_SCRATCH_ROOT:-/scratch/$USER/Segmenation}"
SCRATCH_EXPERIMENT_ROOT="$SCRATCH_ROOT/experiment_2_adaptive_depth"
LOG_BASE="$SCRATCH_EXPERIMENT_ROOT/logs"
MODEL_BASE="$SCRATCH_EXPERIMENT_ROOT/models"
META_BASE="$REPO_EXPERIMENT_ROOT/metadata"
PAIRS_MANIFEST="$REPO_ROOT/manifests/isic2017_train_val_pairs.json"
GLOBAL_EXTRA_ARGS="${GLOBAL_EXTRA_ARGS:-}"
PROTOCOL="${PROTOCOL:-A}"
export PROTOCOL

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$SCRATCH_ROOT" ]]; then
  echo "[error] Scratch root not found: $SCRATCH_ROOT" >&2
  echo "        Set SEG_SCRATCH_ROOT to the desired scratch location before running." >&2
  exit 1
fi

mkdir -p "$LOG_BASE" "$MODEL_BASE" "$META_BASE" "$(dirname "$PAIRS_MANIFEST")"

# Design table: target depth per scale with conservative batch sizes for a 2080 Ti.
SCALES=(
  0.20
  0.30
  0.40
  0.50
  0.60
  0.70
  0.80
)

declare -A DEPTH_FOR_SCALE=(
  [0.20]=1
  [0.30]=2
  [0.40]=3
  [0.50]=3
  [0.60]=4
  [0.70]=5
  [0.80]=5
)

declare -A BATCH_SIZE_FOR_SCALE=(
  [0.20]=8
  [0.30]=8
  [0.40]=6
  [0.50]=4
  [0.60]=3
  [0.70]=2
  [0.80]=1
)

echo "Submitting Experiment 2 runs (adaptive depth per scale)"
for scale in "${SCALES[@]}"; do
  depth="${DEPTH_FOR_SCALE[$scale]:-3}"
  batch_size="${BATCH_SIZE_FOR_SCALE[$scale]:-2}"
  run_name="exp2_adaptive_depth_scale${scale}"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_suffix="${run_name}_${timestamp}"
  log_dir="$LOG_BASE/$run_suffix"
  model_dir="$MODEL_BASE/$run_suffix"
  mkdir -p "$log_dir" "$model_dir"

  export SCALE="$scale"
  export BATCH_SIZE="$batch_size"
  export DEPTH="$depth"
  export LOG_DIR="$log_dir"
  export MODEL_DIR="$model_dir"
  export RUN_NAME="$run_name"
  export PAIRS_MANIFEST="$PAIRS_MANIFEST"
  export EXTRA_ARGS="$GLOBAL_EXTRA_ARGS"

  {
    echo "scale=${scale}"
    echo "batch_size=${batch_size}"
    echo "depth=${depth}"
    echo "run_name=${run_name}"
    echo "log_dir=${log_dir}"
    echo "model_dir=${model_dir}"
    echo "pairs_manifest=${PAIRS_MANIFEST}"
    echo "submitted=$(date --iso-8601=seconds)"
  } > "$META_BASE/${run_suffix}.txt"

  echo "  -> scale=${scale}, depth=${depth}, batch_size=${batch_size}, run_name=${run_name}"
  sbatch "$SBATCH_SCRIPT"
done

echo "All Experiment 2 jobs submitted. Use 'squeue -u $USER' to monitor them."
