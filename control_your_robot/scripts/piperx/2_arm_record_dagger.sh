#!/usr/bin/env bash

set -euo pipefail

# ==================== 在这里修改采集配置 ====================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# DAgger 原始数据集，请勿与专家示教数据集混在一起。
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
# 开启异步预取后，当前 chunk 剩余这么多步时请求下一个 chunk。
PREFETCH_THRESHOLD=25
# 正整数：采集指定条数后退出；-1：持续采集。
NUM_EPISODES=-1
# 正整数：达到指定帧数后自动进入标注；-1：不限制（内存会持续增长）。
MAX_STEPS=-1

# 各类数据的目标条数；-1 表示只统计，不设置目标。
TARGET_AUTONOMOUS_SUCCESS=40
TARGET_AUTONOMOUS_FAILURE=40
TARGET_INTERVENTION_SUCCESS=50
TARGET_INTERVENTION_FAILURE=-1

RESET_SETTLE_SECONDS=2.0
# 切换到 0xFA 控制模式前，将主臂从初始位缓慢对齐到从臂反馈位姿。
TAKEOVER_ALIGN_SPEED=30

# Robocoin 主臂输入就绪检测，以及反馈/控制模式切换配置。
MASTER_ALL_ZERO_ENABLED=true
MASTER_ALL_ZERO_RAD_THRESH=0.001
MASTER_ALL_ZERO_CHECK_JOINTS=6
MASTER_STABLE_TIMEOUT=2.0
MASTER_STABLE_POLL=0.05
MASTER_STABLE_WARMUP=0.6
MASTER_STABLE_READS=3
MASTER_STABLE_MAX_JOINT_DELTA=0.06
# PiStar 夹爪范围归一化为 0..1；此值对应 robocoin 的 0.015 m。
MASTER_STABLE_MAX_GRIPPER_DELTA=0.21428571428571427

# 普通 pi0.5 SFT 留空；PiStar 推理可填写 positive 等条件。
# 仅影响推理，保存的原始数据始终使用 adv_ind=none。
ADV_IND=""

# 人工接管动作滤波，与已验证的 PiperX 采集器保持一致。
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
    echo "错误：未找到环境配置脚本：$ENV_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误：未找到可执行的 Python 环境：$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$COLLECT_SCRIPT" ]]; then
    echo "错误：未找到 DAgger 采集程序：$COLLECT_SCRIPT" >&2
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

printf '%s\n' \
    "" \
    "================ DAgger 数据采集按键 ================" \
    "Enter：开始运行 / 结束当前数据 / 确认保存（根据当前阶段生效）" \
    "空格键：策略运行时切换为人工接管，每条数据只能接管一次" \
    "右方向键：标记成功" \
    "左方向键：标记失败" \
    "Ctrl+C：安全结束采集" \
    "======================================================"

exec "$PYTHON_BIN" "$COLLECT_SCRIPT" "${args[@]}"
