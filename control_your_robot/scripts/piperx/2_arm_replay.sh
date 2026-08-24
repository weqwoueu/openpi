#!/usr/bin/env bash

set -euo pipefail

# Edit these paths manually.
# REPO_ROOT must contain my_env.sh and control_your_robot/.
DATASET_DIR="/home/standard/下载/piperx_black_plug_demo_v3"

EPISODE_INDEX=0
SLAVE_CAN="can_left_slave"
FPS=10
RESET_JOINT_POSITION=(0 1.0 -1.0 1.0 0 0)
GRIPPER_EFFORT=1000

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
CONTROL_ROOT="${REPO_ROOT}/control_your_robot"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
REPLAY_SCRIPT="${CONTROL_ROOT}/scripts/piperx/replay_lerobot_episode.py"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "ERROR: dataset not found: $DATASET_DIR" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

source "$ENV_SCRIPT"
cd "$CONTROL_ROOT"
exec "$PYTHON_BIN" "$REPLAY_SCRIPT" \
    --dataset-dir "$DATASET_DIR" \
    --episode-index "$EPISODE_INDEX" \
    --can-name "$SLAVE_CAN" \
    --fps "$FPS" \
    --gripper-effort "$GRIPPER_EFFORT" \
    --reset-joint-position "${RESET_JOINT_POSITION[@]}"
