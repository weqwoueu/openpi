#!/usr/bin/env bash

set -euo pipefail

# ==================== Edit these settings ====================
REPO_ROOT="/home/standard/workspace/pistar/openpi"

# Raw rollout/DAgger dataset. Keep it separate from the expert dataset.
REPO_ID="piperx/piperx_plug_dagger_demo" #这里改数据集名称
OUTPUT_DIR="/home/standard/agilex/lerobot"
TASK_PROMPT="put the black plug into the two-hole socket"

SERVER_HOST="127.0.0.1"
SERVER_PORT=8000
MASTER_CAN="can_left_mas"
FOLLOWER_CAN="can_left_slave"

SAMPLE_FPS=30
TELEOP_FPS=60
CHUNK_SIZE=50
ASYNC_PREFETCH_ENABLED=false
# When enabled, request the next chunk with this many current actions left.
PREFETCH_THRESHOLD=25
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
# Before 0xFA, ramp the master from init to the follower feedback pose.
TAKEOVER_ALIGN_SPEED=30

# Robocoin master-input readiness and feedback/ctrl handoff.
MASTER_ALL_ZERO_ENABLED=true
MASTER_ALL_ZERO_RAD_THRESH=0.001
MASTER_ALL_ZERO_CHECK_JOINTS=6
MASTER_STABLE_TIMEOUT=2.0
MASTER_STABLE_POLL=0.05
MASTER_STABLE_WARMUP=0.6
MASTER_STABLE_READS=3
MASTER_STABLE_MAX_JOINT_DELTA=0.06
# PiStar gripper is normalized to 0..1; this equals robocoin's 0.015 m.
MASTER_STABLE_MAX_GRIPPER_DELTA=0.21428571428571427

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
    --async-prefetch-enabled "$ASYNC_PREFETCH_ENABLED"
    --prefetch-threshold "$PREFETCH_THRESHOLD"
    --num-episode "$NUM_EPISODES"
    --max-step "$MAX_STEPS"
    --target-autonomous-success "$TARGET_AUTONOMOUS_SUCCESS"
    --target-autonomous-failure "$TARGET_AUTONOMOUS_FAILURE"
    --target-intervention-success "$TARGET_INTERVENTION_SUCCESS"
    --target-intervention-failure "$TARGET_INTERVENTION_FAILURE"
    --reset-settle-seconds "$RESET_SETTLE_SECONDS"
    --takeover-align-speed "$TAKEOVER_ALIGN_SPEED"
    --master-all-zero-enabled "$MASTER_ALL_ZERO_ENABLED"
    --master-all-zero-rad-thresh "$MASTER_ALL_ZERO_RAD_THRESH"
    --master-all-zero-check-joints "$MASTER_ALL_ZERO_CHECK_JOINTS"
    --master-stable-timeout "$MASTER_STABLE_TIMEOUT"
    --master-stable-poll "$MASTER_STABLE_POLL"
    --master-stable-warmup "$MASTER_STABLE_WARMUP"
    --master-stable-reads "$MASTER_STABLE_READS"
    --master-stable-max-joint-delta "$MASTER_STABLE_MAX_JOINT_DELTA"
    --master-stable-max-gripper-delta "$MASTER_STABLE_MAX_GRIPPER_DELTA"
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
