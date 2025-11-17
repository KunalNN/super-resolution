#!/usr/bin/env bash

# Launch Experiment 1: fixed depth (4 levels) while sweeping the requested
# reconstruction scale. Each sbatch submission reuses train_adaptive_simple.sbatch
# and pins the encoder depth to four levels by exporting DEPTH=4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/train_adaptive_simple.sbatch"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_EXPERIMENT_ROOT="$REPO_ROOT/experiments/experiment_1_constant_depth_4"
RUN_ROOT="${EXPERIMENT_RUN_ROOT:-$REPO_EXPERIMENT_ROOT/runs}"
LOG_BASE="$RUN_ROOT/logs"
MODEL_BASE="$RUN_ROOT/models"
META_BASE="$REPO_EXPERIMENT_ROOT/metadata"
TRAINING_CSV="$REPO_EXPERIMENT_ROOT/training_runs.csv"
EVAL_CSV="$REPO_EXPERIMENT_ROOT/evaluation_runs.csv"
PAIRS_MANIFEST="$REPO_ROOT/manifests/isic2017_train_val_pairs.json"
GLOBAL_EXTRA_ARGS="${GLOBAL_EXTRA_ARGS:-}"
PROTOCOL="${PROTOCOL:-A}"
export PROTOCOL

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "[error] Expected sbatch script not found at $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_BASE" "$MODEL_BASE" "$META_BASE" "$(dirname "$PAIRS_MANIFEST")"
mkdir -p "$(dirname "$TRAINING_CSV")" "$(dirname "$EVAL_CSV")"

ensure_csv_header() {
  local csv_path="$1"
  local header="$2"
  if [[ ! -f "$csv_path" ]]; then
    echo "$header" > "$csv_path"
  fi
}

ensure_csv_header "$TRAINING_CSV" "submitted_at,job_id,scale,batch_size,depth,run_name,log_dir,model_dir"
ensure_csv_header "$EVAL_CSV" "submitted_at,job_id,run_name,scale,batch_size,depth,log_dir,model_dir,config_path,status"

# Scale sweep defined in Table 1 (Experiment 1).
SCALES=(
  0.20
  0.30
  0.40
  0.50
  0.60
  0.70
  0.80
  0.90
)

# Batch sizes chosen conservatively so the largest scales fit on a 2080 Ti.
declare -A BATCH_SIZE_FOR_SCALE=(
  [0.20]=8
  [0.30]=8
  [0.40]=8
  [0.50]=6
  [0.60]=4
  [0.70]=2
  [0.80]=1
  [0.90]=1
)

echo "Submitting Experiment 1 runs (depth=4, varying scale)"
for scale in "${SCALES[@]}"; do
  batch_size="${BATCH_SIZE_FOR_SCALE[$scale]:-4}"
  run_name="exp1_depth4_scale${scale}"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  run_suffix="${run_name}_${timestamp}"
  log_dir="$LOG_BASE/$run_suffix"
  model_dir="$MODEL_BASE/$run_suffix"
  mkdir -p "$log_dir" "$model_dir"

  export SCALE="$scale"
  export BATCH_SIZE="$batch_size"
  export DEPTH="4"
  export LOG_DIR="$log_dir"
  export MODEL_DIR="$model_dir"
  export RUN_NAME="$run_name"
  export PAIRS_MANIFEST="$PAIRS_MANIFEST"
  export EXTRA_ARGS="$GLOBAL_EXTRA_ARGS"


  {
    echo "scale=${scale}"
    echo "batch_size=${batch_size}"
    echo "depth=4"
    echo "run_name=${run_name}"
    echo "log_dir=${log_dir}"
    echo "model_dir=${model_dir}"
    echo "pairs_manifest=${PAIRS_MANIFEST}"
    echo "submitted=$(date --iso-8601=seconds)"
  } > "$META_BASE/${run_suffix}.txt"

  echo "  -> scale=${scale}, batch_size=${batch_size}, run_name=${run_name}, log_dir=${log_dir}, model_dir=${model_dir}"
  submit_output="$(sbatch "$SBATCH_SCRIPT")"
  echo "$submit_output"
  job_id="$(awk '{print $4}' <<<"$submit_output")"

  if [[ -n "$job_id" ]]; then
    submission_iso="$(date --iso-8601=seconds)"
    {
      printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$submission_iso" "$job_id" "$scale" "$batch_size" "$DEPTH" "$run_name" "$log_dir" "$model_dir"
    } >> "$TRAINING_CSV"

    config_path="$log_dir/$run_name/config.json"
    {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$submission_iso" "$job_id" "$run_name" "$scale" "$batch_size" "$DEPTH" "$log_dir" "$model_dir" "$config_path" "pending"
    } >> "$EVAL_CSV"
  fi
done

echo "All Experiment 1 jobs submitted. Use 'squeue -u $USER' to monitor them."
