#!/usr/bin/env bash

set -euo pipefail

# ==================== Edit these settings ====================
REPO_ROOT="/home/standard/workspace/pistar/openpi"
CHECKPOINT_DIR="/home/standard/workspace/liuzijian/checkpoints/openpi/pi05_piperx_plug_sft/piperx_plug_sft/29999"
TRAIN_CONFIG="pi05_piperx_plug_recap1"
OPENPI_DATA_HOME_DIR="/home/standard/workspace/liuzijian/checkpoints/openpi"
CUDA_VISIBLE_DEVICES_VALUE="0"
XLA_PYTHON_CLIENT_PREALLOCATE_VALUE="false"
HOST="0.0.0.0"
PORT=8000
DEFAULT_PROMPT="put the black plug into the two-hole socket"
# PiStar positive-condition guidance scale.
ADV_GUIDANCE_BETA="2.0"
# =============================================================

SERVER_SCRIPT="${REPO_ROOT}/control_your_robot/scripts/serve_piper_single_pi05star_websocket.py"

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: repository not found: $REPO_ROOT" >&2
    exit 1
fi
if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "ERROR: server script not found: $SERVER_SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

export OPENPI_DATA_HOME="$OPENPI_DATA_HOME_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE_VALUE"
cd "$REPO_ROOT"

args=(
    --checkpoint-dir "$CHECKPOINT_DIR"
    --train-config "$TRAIN_CONFIG"
    --host "$HOST"
    --port "$PORT"
)

if [[ -n "$DEFAULT_PROMPT" ]]; then
    args+=(--default-prompt "$DEFAULT_PROMPT")
fi
if [[ -n "$ADV_GUIDANCE_BETA" ]]; then
    args+=(--adv-guidance-beta "$ADV_GUIDANCE_BETA")
fi

exec uv run --no-sync python "$SERVER_SCRIPT" "${args[@]}"
