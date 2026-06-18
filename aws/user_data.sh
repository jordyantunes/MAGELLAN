#!/bin/bash
# Runs on the EC2 instance at first boot (injected as user-data by launch_spot.py).
# Placeholders like __BUCKET__ are replaced by launch_spot.py before injection.

set -euo pipefail
exec > >(tee /var/log/magellan-init.log | logger -t magellan-init) 2>&1

BUCKET="__BUCKET__"
S3_PREFIX="__S3_PREFIX__"
RUN_NAME="__RUN_NAME__"
DOCKER_IMAGE="__DOCKER_IMAGE__"
AWS_REGION="__AWS_REGION__"
SAMPLER="__SAMPLER__"
SEED="__SEED__"
SYNC_INTERVAL="__SYNC_INTERVAL__"

S3_RUN_PATH="s3://${BUCKET}/${S3_PREFIX}/${RUN_NAME}"
LOCAL_OUTPUTS="/home/ec2-user/outputs"
LOCAL_HF_CACHE="/home/ec2-user/hf_cache"

mkdir -p "$LOCAL_OUTPUTS" "$LOCAL_HF_CACHE"

# ---------------------------------------------------------------------------
# 1. Authenticate Docker with ECR (skip if using a public image)
# ---------------------------------------------------------------------------
if echo "$DOCKER_IMAGE" | grep -q "\.ecr\."; then
    aws ecr get-login-password --region "$AWS_REGION" \
        | docker login --username AWS --password-stdin \
          "$(echo "$DOCKER_IMAGE" | cut -d'/' -f1)"
fi

docker pull "$DOCKER_IMAGE"

# Install MLflow for the UI server (lightweight, host-side only)
pip install --quiet mlflow

# ---------------------------------------------------------------------------
# 2. Sync latest checkpoint from S3 (if any)
# ---------------------------------------------------------------------------
LOADING_PATH_ARG=""

# List checkpoint directories (numeric names) and find the latest
LATEST=$(aws s3 ls "${S3_RUN_PATH}/" 2>/dev/null \
    | awk '{print $2}' \
    | tr -d '/' \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -1 || true)

if [ -n "$LATEST" ]; then
    echo "Resuming from checkpoint ${LATEST}"
    aws s3 sync "${S3_RUN_PATH}/${LATEST}/" "${LOCAL_OUTPUTS}/${RUN_NAME}/${LATEST}/"
    LOADING_PATH_ARG="rl_script_args.loading_path=/app/outputs/${RUN_NAME}/${LATEST}"
else
    echo "No checkpoint found — starting fresh"
fi

# ---------------------------------------------------------------------------
# 3. Map sampler name to Hydra config name
# ---------------------------------------------------------------------------
case "$SAMPLER" in
    random)    CONFIG_NAME="local_gpu_config_random"    ;;
    magellan)  CONFIG_NAME="aws_g5_magellan"             ;;
    online)    CONFIG_NAME="local_gpu_config_online"    ;;
    ek_online) CONFIG_NAME="local_gpu_config_ek_online" ;;
    *)
        echo "Unknown sampler: $SAMPLER"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Spot interruption handler — syncs outputs before the instance is reclaimed
# ---------------------------------------------------------------------------
sync_to_s3() {
    aws s3 sync "${LOCAL_OUTPUTS}/${RUN_NAME}/" "${S3_RUN_PATH}/"
    # mlflow.db lives one level above the run dir — sync it to the bucket root
    [ -f "${LOCAL_OUTPUTS}/mlflow.db" ] && \
        aws s3 cp "${LOCAL_OUTPUTS}/mlflow.db" "s3://${BUCKET}/${S3_PREFIX}/mlflow.db" || true
}

handle_interruption() {
    echo "Spot interruption notice received — syncing outputs to S3"
    sync_to_s3
    echo "Sync complete. Instance will be terminated shortly."
}

poll_interruption() {
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            http://169.254.169.254/latest/meta-data/spot/termination-time \
            --max-time 2 || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            handle_interruption
            break
        fi
        sleep 5
    done
}

poll_interruption &
POLL_PID=$!

# ---------------------------------------------------------------------------
# 5. Periodic checkpoint sync to S3 while training runs
# ---------------------------------------------------------------------------
if [ "$SYNC_INTERVAL" -gt 0 ]; then
    periodic_sync() {
        while true; do
            sleep "$SYNC_INTERVAL"
            echo "Periodic sync to ${S3_RUN_PATH}/"
            sync_to_s3 || true
        done
    }
    periodic_sync &
    SYNC_PID=$!
fi

# ---------------------------------------------------------------------------
# 6. Run training
# ---------------------------------------------------------------------------
DOCKER_CMD=(
    docker run --rm --gpus all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e MLFLOW_TRACKING_URI="sqlite:////app/outputs/mlflow.db"
    -v "${LOCAL_OUTPUTS}:/app/outputs"
    -v "${LOCAL_HF_CACHE}:/app/.cache/huggingface"
    "$DOCKER_IMAGE"
    python -m lamorel_launcher.launch
    --config-path /app/configs/little_zoo/
    --config-name "$CONFIG_NAME"
    rl_script_args.path=/app/magellan/main.py
    "rl_script_args.output_dir=/app/outputs/${RUN_NAME}"
    "rl_script_args.seed=${SEED}"
)

if [ -n "$LOADING_PATH_ARG" ]; then
    DOCKER_CMD+=("$LOADING_PATH_ARG")
fi

# ---------------------------------------------------------------------------
# 6b. Start MLflow UI server (port 5000, accessible via the instance's public IP)
# ---------------------------------------------------------------------------
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:///${LOCAL_OUTPUTS}/mlflow.db" \
    --no-serve-artifacts \
    &
MLFLOW_PID=$!
echo "MLflow UI started (PID ${MLFLOW_PID}) — http://<instance-ip>:5000"

echo "Starting training: ${DOCKER_CMD[*]}"
"${DOCKER_CMD[@]}" && EXIT_CODE=0 || EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 7. Final sync on completion
# ---------------------------------------------------------------------------
kill "$POLL_PID" 2>/dev/null || true
[ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null || true
kill "$MLFLOW_PID" 2>/dev/null || true

echo "Training finished (exit code ${EXIT_CODE}) — final sync to S3"
sync_to_s3
echo "Done."
