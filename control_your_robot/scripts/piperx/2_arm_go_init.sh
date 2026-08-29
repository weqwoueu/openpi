#!/usr/bin/env bash

set -euo pipefail

# Edit this path manually. It must contain my_env.sh and control_your_robot/.
REPO_ROOT="/home/standard/workspace/pistar/openpi"

SPEED=15
GRIPPER=0
MOVE_WAIT_SECONDS=5

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
PIPERX_SCRIPTS="${REPO_ROOT}/control_your_robot/scripts/piperx"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi

source "$ENV_SCRIPT"

echo "[1/2] Move follower to the task initial pose"
bash "${PIPERX_SCRIPTS}/one_arm_go_init.sh" can_left_slave "$SPEED" "$GRIPPER"
sleep "$MOVE_WAIT_SECONDS"

echo "[2/2] Move master to the task initial pose"
bash "${PIPERX_SCRIPTS}/one_arm_go_init.sh" can_left_mas "$SPEED" "$GRIPPER"
