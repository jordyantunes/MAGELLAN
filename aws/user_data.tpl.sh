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

INSTANCE_ID=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id)

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
docker pull ghcr.io/mlflow/mlflow:latest

# ---------------------------------------------------------------------------
# 2. CloudWatch agent — ship init log and enable Docker awslogs driver
# ---------------------------------------------------------------------------
# Install agent if not already present (DL AMIs usually include it)
if ! command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
    dnf install -y amazon-cloudwatch-agent
fi

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << CWEOF
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/magellan-init.log",
            "log_group_name": "/magellan/init",
            "log_stream_name": "${RUN_NAME}/${INSTANCE_ID}",
            "retention_in_days": 30
          }
        ]
      }
    }
  }
}
CWEOF

amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# ---------------------------------------------------------------------------
# 3. Sync latest checkpoint from S3 (if any)
# ---------------------------------------------------------------------------
LOADING_PATH_ARG=""

# Checkpoints are saved as <run_name>/<timestamp>/<episode>/
# Find the latest timestamp subdir, then the latest numeric episode inside it.
LATEST_TS=$(aws s3 ls "${S3_RUN_PATH}/" 2>/dev/null \
    | awk '{print $2}' \
    | tr -d '/' \
    | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}$' \
    | sort \
    | tail -1 || true)

LATEST_EP=""
if [ -n "$LATEST_TS" ]; then
    LATEST_EP=$(aws s3 ls "${S3_RUN_PATH}/${LATEST_TS}/" 2>/dev/null \
        | awk '{print $2}' \
        | tr -d '/' \
        | grep -E '^[0-9]+$' \
        | sort -n \
        | tail -1 || true)
fi

if [ -n "$LATEST_TS" ] && [ -n "$LATEST_EP" ]; then
    echo "Resuming from checkpoint ${LATEST_TS}/${LATEST_EP}"
    aws s3 sync "${S3_RUN_PATH}/${LATEST_TS}/${LATEST_EP}/" \
        "${LOCAL_OUTPUTS}/${RUN_NAME}/${LATEST_TS}/${LATEST_EP}/"
    LOADING_PATH_ARG="rl_script_args.loading_path=/app/outputs/${RUN_NAME}/${LATEST_TS}/${LATEST_EP}"
else
    echo "No checkpoint found — starting fresh"
fi

# ---------------------------------------------------------------------------
# 4. Map sampler name to Hydra config name
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
# 4b. Self-cancellation helper — called after training ends (clean or crash).
#     Looks up the fleet ID from this instance's tags and cancels it so the
#     fleet does not automatically launch a replacement.
#     On a spot interruption the instance is hard-killed before this runs,
#     so the fleet correctly relaunches in that case.
# ---------------------------------------------------------------------------
cancel_fleet() {
    SELF_ID=$(curl -s --max-time 2 \
        http://169.254.169.254/latest/meta-data/instance-id || true)
    SELF_REGION=$(curl -s --max-time 2 \
        http://169.254.169.254/latest/meta-data/placement/region || true)

    if [ -z "$SELF_ID" ] || [ -z "$SELF_REGION" ]; then
        echo "Could not reach instance metadata — skipping fleet cancellation"
        return
    fi

    FLEET_ID=$(aws ec2 describe-instances \
        --instance-ids "$SELF_ID" \
        --region "$SELF_REGION" \
        --query "Reservations[0].Instances[0].Tags[?Key=='aws:ec2spot:fleet-request-id'].Value" \
        --output text 2>/dev/null || true)

    if [ -z "$FLEET_ID" ] || [ "$FLEET_ID" = "None" ]; then
        echo "No fleet tag found on this instance — skipping fleet cancellation"
        return
    fi

    echo "Cancelling Spot Fleet ${FLEET_ID} (TerminateInstances=true)"
    aws ec2 cancel-spot-fleet-requests \
        --region "$SELF_REGION" \
        --spot-fleet-request-ids "$FLEET_ID" \
        --terminate-instances || true
}

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
# 6. Write docker-compose file and env file, then launch via docker compose
# ---------------------------------------------------------------------------
cat > /home/ec2-user/docker-compose.aws.yml << 'COMPOSE_EOF'
__DOCKER_COMPOSE_CONTENT__
COMPOSE_EOF

cat > /home/ec2-user/.env << EOF
LOCAL_OUTPUTS=${LOCAL_OUTPUTS}
LOCAL_HF_CACHE=${LOCAL_HF_CACHE}
BUCKET=${BUCKET}
S3_PREFIX=${S3_PREFIX}
RUN_NAME=${RUN_NAME}
DOCKER_IMAGE=${DOCKER_IMAGE}
CONFIG_NAME=${CONFIG_NAME}
SEED=${SEED}
LOADING_PATH_ARG=${LOADING_PATH_ARG}
AWS_REGION=${AWS_REGION}
INSTANCE_ID=${INSTANCE_ID}
EOF

echo "MLflow UI will be available at http://<instance-ip>:5000 once training starts"

docker compose -f /home/ec2-user/docker-compose.aws.yml \
    --env-file /home/ec2-user/.env \
    up --exit-code-from magellan && EXIT_CODE=0 || EXIT_CODE=$?

# ---------------------------------------------------------------------------
# 7. Final sync on completion
# ---------------------------------------------------------------------------
kill "$POLL_PID" 2>/dev/null || true
[ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null || true

echo "Training finished (exit code ${EXIT_CODE}) — final sync to S3"
sync_to_s3
cancel_fleet
echo "Done."
