#!/usr/bin/env bash

set -euo pipefail

# Edit this path manually. It must contain my_env.sh and control_your_robot/.
REPO_ROOT="/home/standard/workspace/pistar/openpi"

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
CONTROL_ROOT="${REPO_ROOT}/control_your_robot"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

source "$ENV_SCRIPT"
cd "$CONTROL_ROOT"
exec "$PYTHON_BIN" example/collect/collect_lerobot_master_slave_teleop.py
