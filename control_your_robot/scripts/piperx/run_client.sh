#!/usr/bin/env bash

set -euo pipefail

# ==================== Edit these settings ====================
REPO_ROOT="/home/standard/workspace/pistar/openpi"
SERVER_HOST="127.0.0.1"
SERVER_PORT=8000
ARM_CAN="can_left_slave"
TASK_PROMPT="put the black plug into the two-hole socket"
CONTROL_FREQ=30
CHUNK_SIZE=10
MAX_STEPS=600
# Positive integer: run a fixed number of episodes. -1: run forever.
NUM_EPISODES=1
# Ordinary pi0.5: leave empty. Future PiStar inference: positive.
ADV_IND=""
# =============================================================

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
CLIENT_SCRIPT="${REPO_ROOT}/control_your_robot/example/deploy/piper_single_on_PI0_websocket.py"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CLIENT_SCRIPT" ]]; then
    echo "ERROR: client script not found: $CLIENT_SCRIPT" >&2
    exit 1
fi

source "$ENV_SCRIPT"
cd "${REPO_ROOT}/control_your_robot"

args=(
    --server-host "$SERVER_HOST"
    --server-port "$SERVER_PORT"
    --arm-can "$ARM_CAN"
    --task-name "$TASK_PROMPT"
    --instruction "$TASK_PROMPT"
    --control-freq "$CONTROL_FREQ"
    --chunk-size "$CHUNK_SIZE"
    --max-step "$MAX_STEPS"
    --num-episode "$NUM_EPISODES"
)

if [[ -n "$ADV_IND" ]]; then
    args+=(--adv-ind "$ADV_IND")
fi

exec "$PYTHON_BIN" "$CLIENT_SCRIPT" "${args[@]}"
