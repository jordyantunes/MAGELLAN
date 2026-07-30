#!/usr/bin/env bash
# Interactive launcher for a Dockerized MAGELLAN training run.
# Prompts for config / seed / output dir / MLflow run name; pressing Enter
# accepts the [default] shown. Pass --dry-run to print the launch command
# without starting anything.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=${1:-}

mapfile -t CONFIGS < <(ls configs/little_zoo/ | grep '^local_gpu_config_.*\.yaml$' | sed 's/\.yaml$//')
DEFAULT_CONFIG=local_gpu_config_magellan_4090

echo "Available configs (configs/little_zoo/):"
for i in "${!CONFIGS[@]}"; do
  marker=""
  [ "${CONFIGS[$i]}" = "$DEFAULT_CONFIG" ] && marker="  (default)"
  printf "  %2d) %s%s\n" "$((i + 1))" "${CONFIGS[$i]}" "$marker"
done
echo

read -rp "Config number or name [${DEFAULT_CONFIG}]: " CHOICE
if [ -z "$CHOICE" ]; then
  CONFIG_NAME=$DEFAULT_CONFIG
elif [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#CONFIGS[@]}" ]; then
  CONFIG_NAME=${CONFIGS[$((CHOICE - 1))]}
else
  CONFIG_NAME=$CHOICE
fi
if [ ! -f "configs/little_zoo/${CONFIG_NAME}.yaml" ]; then
  echo "error: configs/little_zoo/${CONFIG_NAME}.yaml not found" >&2
  exit 1
fi

read -rp "Seed [0]: " SEED
SEED=${SEED:-0}

read -rp "Output dir under outputs/ [magellan]: " OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-magellan}

# Empty RUN_NAME -> main.py falls back to "seed{seed}"
read -rp "MLflow run name [seed${SEED}]: " RUN_NAME
RUN_NAME=${RUN_NAME:-}

export CONFIG_NAME SEED OUTPUT_DIR RUN_NAME

echo
echo "Launching:"
echo "  config:   ${CONFIG_NAME}"
echo "  seed:     ${SEED}"
echo "  output:   outputs/${OUTPUT_DIR}"
echo "  run name: ${RUN_NAME:-seed${SEED}}"
echo

ONE_LINER="CONFIG_NAME=$CONFIG_NAME SEED=$SEED OUTPUT_DIR=$OUTPUT_DIR RUN_NAME=$RUN_NAME docker compose -f docker-compose.local.yml up -d --build"

if [ "$DRY_RUN" = "--dry-run" ]; then
  echo "[dry-run] $ONE_LINER"
  exit 0
fi

if docker compose -f docker-compose.local.yml ps --status running --services 2>/dev/null | grep -qx magellan; then
  read -rp "A magellan container is already running. Stop it and continue? [y/N]: " STOP
  if [ "${STOP,,}" = "y" ]; then
    docker compose -f docker-compose.local.yml down
  else
    echo "Aborted."
    exit 1
  fi
fi

docker compose -f docker-compose.local.yml up -d --build
echo
echo "Started. Follow logs with:  docker compose -f docker-compose.local.yml logs -f magellan"
echo
echo "To launch this same experiment non-interactively:"
echo "  $ONE_LINER"
