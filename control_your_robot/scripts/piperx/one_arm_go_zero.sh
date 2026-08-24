#!/usr/bin/env bash

# Single-arm zero-position utility for the PiperX plug experiment.
# 功能：将指定 CAN 口的一条 Piper 的所有关节移动到零位置（0度）
# 额外处理：每条臂下发关节指令前，先检查是否为 CAN 指令控制模式（从臂控制模式）；
#          若不是则先切换到从臂模式，并立即下发一条 MotionCtrl_2 以稳定使能/控制状态。
#
# 使用方法：./one_arm_go_zero.sh [can_name] [speed] [gripper]
#   can_name: CAN interface, default can_left_slave
#   speed:    运动速度百分比（0-100），默认为10
#   gripper:  夹爪行程（单位：m），默认为0（闭合）

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
DEFAULT_CAN_NAME="can_left_slave"
DEFAULT_SPEED=10
DEFAULT_GRIPPER=0.0

CAN_NAME="$DEFAULT_CAN_NAME"
SPEED="$DEFAULT_SPEED"
GRIPPER="$DEFAULT_GRIPPER"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTRL_SCRIPT="${SCRIPT_DIR}/ctrl_joint.py"

show_help() {
    local exit_code="${1:-0}"
    echo "PiperX 单机械臂归零脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [can_name] [speed] [gripper]"
    echo "  $0 [speed] [gripper]      # 兼容旧用法，默认 CAN 为 ${DEFAULT_CAN_NAME}"
    echo ""
    echo "参数说明:"
    echo "  can_name: CAN 口名称，默认为 ${DEFAULT_CAN_NAME}"
    echo "            常见值：can_left、can_right、can_left_slave、can_left_mas、can_right_slave、can_right_mas"
    echo "  speed:    运动速度百分比（0-100），默认为 ${DEFAULT_SPEED}"
    echo "  gripper:  夹爪行程（单位：m），默认为 ${DEFAULT_GRIPPER}（闭合）"
    echo ""
    echo "示例:"
    echo "  $0                         # 默认控制 ${DEFAULT_CAN_NAME}"
    echo "  $0 can_left                 # 控制 can_left，默认速度与夹爪闭合"
    echo "  $0 can_right_slave 20       # 控制 can_right_slave，速度20%"
    echo "  $0 can_left_mas 20 0.05     # 控制 can_left_mas，速度20%，夹爪打开到50mm"
    echo "  $0 20 0.05                  # 旧用法：控制 ${DEFAULT_CAN_NAME}，速度20%，夹爪打开到50mm"
    echo ""
    exit "$exit_code"
}

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    show_help 0
fi

is_speed_arg() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

label_for_can() {
    local can_name="$1"
    case "$can_name" in
        can_left) echo "左臂" ;;
        can_right) echo "右臂" ;;
        can_left_slave) echo "左从臂" ;;
        can_right_slave) echo "右从臂" ;;
        can_left_mas|can_left_master) echo "左主臂" ;;
        can_right_mas|can_right_master) echo "右主臂" ;;
        *) echo "$can_name" ;;
    esac
}

parse_args() {
    if [ "$#" -eq 0 ]; then
        return
    fi

    if is_speed_arg "$1"; then
        SPEED="$1"
        GRIPPER="${2:-$DEFAULT_GRIPPER}"
        if [ "$#" -gt 2 ]; then
            echo -e "${RED}错误: 参数过多${NC}"
            show_help 1
        fi
    else
        CAN_NAME="$1"
        SPEED="${2:-$DEFAULT_SPEED}"
        GRIPPER="${3:-$DEFAULT_GRIPPER}"
        if [ "$#" -gt 3 ]; then
            echo -e "${RED}错误: 参数过多${NC}"
            show_help 1
        fi
    fi
}

parse_args "$@"

if [ -z "$CAN_NAME" ]; then
    echo -e "${RED}错误: CAN 口名称不能为空${NC}"
    exit 1
fi

if ! [[ "$SPEED" =~ ^[0-9]+$ ]] || [ "$SPEED" -lt 0 ] || [ "$SPEED" -gt 100 ]; then
    echo -e "${RED}错误: 速度参数必须是0-100之间的整数${NC}"
    exit 1
fi

if ! [[ "$GRIPPER" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo -e "${RED}错误: 夹爪参数必须是数字${NC}"
    exit 1
fi
if awk "BEGIN {exit !($GRIPPER < 0)}"; then
    echo -e "${RED}错误: 夹爪参数必须是非负数${NC}"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi

if [ ! -f "$CTRL_SCRIPT" ]; then
    echo -e "${RED}错误: 未找到控制脚本 $CTRL_SCRIPT${NC}"
    exit 1
fi

# 切换到从臂模式（0xFC）并在切换后立即下发 MotionCtrl_2
ensure_can_slave_mode() {
    local can_name="$1"
    local label="$2"
    python3 - "$can_name" "$label" <<'PY'
import sys
import time

from piper_sdk import C_PiperInterface_V2

can_name = sys.argv[1]
label = sys.argv[2]

print(f"[mode-check] {label} ({can_name}) 开始检查控制模式...")
piper = C_PiperInterface_V2(can_name=can_name)
piper.ConnectPort()
time.sleep(0.1)

need_switch = True
try:
    # 部分 SDK 版本提供 ctrl_mode，但不一定稳定；仅作参考。
    status = piper.GetArmStatus()
    arm_status = getattr(status, "arm_status", None)
    ctrl_mode = getattr(arm_status, "ctrl_mode", None)
    if ctrl_mode == 0x01:
        print(f"[mode-check] {label} 当前显示为 CAN 指令控制模式(0x01)。")
    else:
        print(f"[mode-check] {label} 当前模式={ctrl_mode}，将切换到从臂模式。")
except Exception as e:
    # 与 kai0 中的防御策略一致：取不到可靠状态时，仍执行切换，保证后续控制稳定。
    print(f"[mode-check] {label} 读取模式失败({e})，将执行切换。")

# 与 kai0 的保护逻辑一致：
# 即便看起来已在 CAN 模式，也执行一次从臂切换，避免状态读数不可靠导致误判。
need_switch = True
if need_switch:
    # 0xFC: 从臂跟随；参数与项目内 switch_arm_mode.sh 一致。
    piper.MasterSlaveConfig(
        linkage_config=0xFC,
        feedback_offset=0x00,
        ctrl_offset=0x00,
        linkage_offset=0x00,
    )
    time.sleep(0.5)

    # 关键处理：切换后立刻下发控制命令，恢复控制状态稳定性。
    try:
        piper.MotionCtrl_2(0x01, 0x01, 10, 0x00)
    except TypeError:
        piper.MotionCtrl_2(0x01, 0x01, 10)
    time.sleep(0.1)
    print(f"[mode-check] {label} 已执行从臂切换并下发恢复命令。")
PY
}

echo "=========================================="
echo -e "${BLUE}PiperX 单机械臂归零程序${NC}"
echo "=========================================="
echo "目标 CAN: ${CAN_NAME} ($(label_for_can "$CAN_NAME"))"
echo "速度: ${SPEED}%"
if awk "BEGIN {exit !($GRIPPER > 0)}"; then
    echo "夹爪: 打开到 ${GRIPPER}m"
else
    echo "夹爪: 闭合"
fi
echo ""

# ctrl_joint.py 参数格式: [can_name] [j1] [j2] [j3] [j4] [j5] [j6] [gripper] [speed]
JOINT_ARGS="0 0 0 0 0 0"

run_one_arm() {
    local can_name="$1"
    local label="$2"
    local idx="$3"
    local total="$4"

    echo -e "${YELLOW}[${idx}/${total}] 正在控制${label}归零...${NC}"

    ensure_can_slave_mode "$can_name" "$label"

    if python3 "$CTRL_SCRIPT" "$can_name" $JOINT_ARGS "$GRIPPER" "$SPEED"; then
        echo -e "${GREEN}✓ ${label}归零指令已发送${NC}"
    else
        echo -e "${RED}✗ ${label}归零失败${NC}"
        exit 1
    fi
    echo ""
    sleep 0.5
}

run_one_arm "$CAN_NAME" "$(label_for_can "$CAN_NAME")" 1 1

echo "=========================================="
echo -e "${GREEN}✓ 目标机械臂归零指令已发送完成${NC}"
echo "=========================================="
echo ""
echo "提示: 机械臂正在移动到零位置，请等待运动完成。"
