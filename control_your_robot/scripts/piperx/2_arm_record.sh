#!/usr/bin/env bash

set -euo pipefail

# Edit the dataset destination here.
REPO_ID="piperx/piperx_black_plug_demo"
OUTPUT_DIR="/home/standard/agilex/lerobot"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${CONTROL_ROOT}/.." && pwd)"
ENV_SCRIPT="${PROJECT_ROOT}/my_env.sh"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi

source "$ENV_SCRIPT"
export PIPERX_REPO_ID="$REPO_ID"
export PIPERX_OUTPUT_DIR="$OUTPUT_DIR"
cd "$CONTROL_ROOT"
exec "$PYTHON_BIN" example/collect/collect_lerobot_master_slave_teleop.py
