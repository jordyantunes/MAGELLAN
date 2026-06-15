#!/usr/bin/env bash
set -euo pipefail

SEED=0
RESUME=""
CONFIG="local_gpu_config_magellan"

usage() {
  echo "Usage: $0 [--seed N] [--resume CHECKPOINT_PATH] [--dev]"
  echo "  --seed N              RNG seed (default: 0)"
  echo "  --resume PATH         Resume from checkpoint, e.g. outputs/magellan/10000"
  echo "  --dev                 Use dev config (RTX 5080, reduced VRAM usage, short run)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)    SEED="$2";   shift 2 ;;
    --resume)  RESUME="$2"; shift 2 ;;
    --dev)     CONFIG="local_gpu_config_magellan_dev"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

EXTRA_ARGS=()
if [[ -n "$RESUME" ]]; then
  EXTRA_ARGS+=("rl_script_args.loading_path=/app/${RESUME#./}")
fi

docker run --rm \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e MLFLOW_TRACKING_URI=sqlite:////app/outputs/mlflow.db \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "magellan_hf_cache:/app/.cache/huggingface" \
  -v "/LLMs:/LLMs:ro" \
  magellan:latest \
  python -m lamorel_launcher.launch \
    --config-path /app/configs/little_zoo/ \
    "--config-name" "${CONFIG}" \
    rl_script_args.path=/app/magellan/main.py \
    rl_script_args.output_dir=/app/outputs/magellan \
    "rl_script_args.seed=${SEED}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
