#!/usr/bin/env bash

set -euo pipefail

# Edit this path manually. It must contain my_env.sh and control_your_robot/.
REPO_ROOT="/home/standard/workspace/pistar/openpi"

SPEED=10
GRIPPER=0
MOVE_WAIT_SECONDS=6

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
PIPERX_SCRIPTS="${REPO_ROOT}/control_your_robot/scripts/piperx"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi

source "$ENV_SCRIPT"

echo "[1/3] Move follower to the task initial pose"
bash "${PIPERX_SCRIPTS}/one_arm_go_init.sh" can_left_slave "$SPEED" "$GRIPPER"
sleep "$MOVE_WAIT_SECONDS"

echo "[2/3] Move master to zero"
bash "${PIPERX_SCRIPTS}/one_arm_go_zero.sh" can_left_mas "$SPEED" "$GRIPPER"
sleep "$MOVE_WAIT_SECONDS"

echo "[3/3] Move master to the task initial pose"
bash "${PIPERX_SCRIPTS}/one_arm_go_init.sh" can_left_mas "$SPEED" "$GRIPPER"
