#!/usr/bin/env bash

set -euo pipefail

# ==================== Edit these settings ====================
REPO_ROOT="/home/standard/workspace/pistar/openpi"

# Raw rollout/DAgger dataset. Keep it separate from the expert dataset.
REPO_ID="piperx/piperx_plug_dagger_demo"
OUTPUT_DIR="/home/standard/agilex/lerobot"
TASK_PROMPT="put the black plug into the two-hole socket"

SERVER_HOST="127.0.0.1"
SERVER_PORT=8000
MASTER_CAN="can_left_mas"
FOLLOWER_CAN="can_left_slave"

SAMPLE_FPS=30
TELEOP_FPS=60
CHUNK_SIZE=50
# Request the next chunk with this many current actions left. 0 disables overlap.
PREFETCH_THRESHOLD=0
# Positive integer: save this many episodes. -1: keep collecting.
NUM_EPISODES=-1
# Positive integer: automatically stop for labeling at this frame. -1: unlimited RAM growth.
MAX_STEPS=-1

# Episode mix targets. -1 keeps counting the category without setting a target.
TARGET_AUTONOMOUS_SUCCESS=40
TARGET_AUTONOMOUS_FAILURE=40
TARGET_INTERVENTION_SUCCESS=50
TARGET_INTERVENTION_FAILURE=-1

RESET_SETTLE_SECONDS=2.0
ALIGNMENT_TIMEOUT=2.0
FEEDBACK_FALLBACK=false

# Ordinary pi0.5 SFT: leave empty. PiStar inference: for example positive.
# This affects inference only. Saved raw data always uses adv_ind=none.
ADV_IND=""

# Expert takeover filtering, matching the accepted PiperX recorder.
EMA_ENABLED=true
EMA_ALPHA=0.80
SLEW_ENABLED=true
MAX_JOINT_STEP=0.040
MAX_GRIPPER_STEP=0.35714285714285715

JOINT_SIGN=(1 1 1 1 1 1)
JOINT_OFFSET=(0 0 0 0 0 0)
JOINT_SCALE=1.0
GRIPPER_SCALE=1.0
GRIPPER_OFFSET=0.0
# =============================================================

ENV_SCRIPT="${REPO_ROOT}/my_env.sh"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
COLLECT_SCRIPT="${REPO_ROOT}/control_your_robot/example/collect/collect_lerobot_dagger_websocket.py"

if [[ ! -f "$ENV_SCRIPT" ]]; then
    echo "ERROR: my_env.sh not found: $ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$COLLECT_SCRIPT" ]]; then
    echo "ERROR: DAgger collector not found: $COLLECT_SCRIPT" >&2
    exit 1
fi

source "$ENV_SCRIPT"
cd "${REPO_ROOT}/control_your_robot"

args=(
    --server-host "$SERVER_HOST"
    --server-port "$SERVER_PORT"
    --repo-id "$REPO_ID"
    --output-dir "$OUTPUT_DIR"
    --task-name "$TASK_PROMPT"
    --master-can "$MASTER_CAN"
    --follower-can "$FOLLOWER_CAN"
    --sample-fps "$SAMPLE_FPS"
    --teleop-fps "$TELEOP_FPS"
    --chunk-size "$CHUNK_SIZE"
    --prefetch-threshold "$PREFETCH_THRESHOLD"
    --num-episode "$NUM_EPISODES"
    --max-step "$MAX_STEPS"
    --target-autonomous-success "$TARGET_AUTONOMOUS_SUCCESS"
    --target-autonomous-failure "$TARGET_AUTONOMOUS_FAILURE"
    --target-intervention-success "$TARGET_INTERVENTION_SUCCESS"
    --target-intervention-failure "$TARGET_INTERVENTION_FAILURE"
    --reset-settle-seconds "$RESET_SETTLE_SECONDS"
    --alignment-timeout "$ALIGNMENT_TIMEOUT"
    --feedback-fallback "$FEEDBACK_FALLBACK"
    --ema-enabled "$EMA_ENABLED"
    --ema-alpha "$EMA_ALPHA"
    --slew-enabled "$SLEW_ENABLED"
    --max-joint-step "$MAX_JOINT_STEP"
    --max-gripper-step "$MAX_GRIPPER_STEP"
    --joint-sign "${JOINT_SIGN[@]}"
    --joint-offset "${JOINT_OFFSET[@]}"
    --joint-scale "$JOINT_SCALE"
    --gripper-scale "$GRIPPER_SCALE"
    --gripper-offset "$GRIPPER_OFFSET"
)

if [[ -n "$ADV_IND" ]]; then
    args+=(--adv-ind "$ADV_IND")
fi

exec "$PYTHON_BIN" "$COLLECT_SCRIPT" "${args[@]}"
