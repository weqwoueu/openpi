#!/usr/bin/env bash

# Activate only the master and insertion-follower CAN adapters used by this task.
# Other attached CAN adapters are reported and left untouched.

set -euo pipefail

BITRATE=1000000
MASTER_BUS_INFO="1-4.1.1"
SLAVE_BUS_INFO="1-4.1.3"
EXPECTED_BUS_INFOS=("$MASTER_BUS_INFO" "$SLAVE_BUS_INFO")

declare -A TARGET_BY_BUS=(
    ["$MASTER_BUS_INFO"]="can_left_mas"
    ["$SLAVE_BUS_INFO"]="can_left_slave"
)

CHECK_ONLY=false

usage() {
    echo "Usage: $0 [--check]"
    echo
    echo "  no option  Configure the two fixed adapters at ${BITRATE} bit/s."
    echo "  --check    Read-only validation; do not load modules or change links."
}

case "${1:-}" in
    "") ;;
    --check) CHECK_ONLY=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac
if (($# > 1)); then
    echo "ERROR: too many arguments" >&2
    usage >&2
    exit 2
fi

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

bus_info_for() {
    ethtool -i "$1" 2>/dev/null | awk -F ': *' '$1 == "bus-info" {print $2; exit}' || true
}

bitrate_for() {
    ip -details link show dev "$1" 2>/dev/null \
        | awk '/bitrate [0-9]+/ {for (i = 1; i <= NF; i++) if ($i == "bitrate") {print $(i + 1); exit}}' \
        || true
}

require_command ip
require_command ethtool
if ! $CHECK_ONLY; then
    require_command sudo
    sudo modprobe gs_usb
fi

mapfile -t interfaces < <(ip -br link show type can | awk '{print $1}')
if ((${#interfaces[@]} == 0)); then
    echo "ERROR: no CAN interfaces found." >&2
    exit 1
fi

declare -A INTERFACE_BY_BUS=()
declare -A BUS_BY_INTERFACE=()

echo "Detected CAN interfaces:"
for interface in "${interfaces[@]}"; do
    bus_info="$(bus_info_for "$interface")"
    if [[ -z "$bus_info" ]]; then
        echo "ERROR: cannot read bus-info for $interface" >&2
        exit 1
    fi
    if [[ -n "${INTERFACE_BY_BUS[$bus_info]:-}" ]]; then
        echo "ERROR: duplicate bus-info $bus_info for $interface and ${INTERFACE_BY_BUS[$bus_info]}" >&2
        exit 1
    fi
    INTERFACE_BY_BUS["$bus_info"]="$interface"
    BUS_BY_INTERFACE["$interface"]="$bus_info"
    printf '  %-18s %s\n' "$interface" "$bus_info"
done

# Complete all identity and collision checks before changing either target.
for bus_info in "${EXPECTED_BUS_INFOS[@]}"; do
    target="${TARGET_BY_BUS[$bus_info]}"
    source_interface="${INTERFACE_BY_BUS[$bus_info]:-}"
    if [[ -z "$source_interface" ]]; then
        echo "ERROR: expected adapter at $bus_info for $target was not found." >&2
        exit 1
    fi

    if ip link show dev "$target" >/dev/null 2>&1; then
        target_bus_info="$(bus_info_for "$target")"
        if [[ "$target_bus_info" != "$bus_info" ]]; then
            echo "ERROR: $target is occupied by bus-info $target_bus_info; expected $bus_info." >&2
            exit 1
        fi
    fi
done

for interface in "${interfaces[@]}"; do
    bus_info="${BUS_BY_INTERFACE[$interface]}"
    if [[ -z "${TARGET_BY_BUS[$bus_info]:-}" ]]; then
        echo "Leaving unrelated CAN interface untouched: $interface ($bus_info)"
    fi
done

if ! $CHECK_ONLY; then
    for bus_info in "${EXPECTED_BUS_INFOS[@]}"; do
        interface="${INTERFACE_BY_BUS[$bus_info]}"
        target="${TARGET_BY_BUS[$bus_info]}"

        echo "Configuring $bus_info: $interface -> $target @ $BITRATE"
        sudo ip link set dev "$interface" down
        if [[ "$interface" != "$target" ]]; then
            sudo ip link set dev "$interface" name "$target"
        fi
        sudo ip link set dev "$target" type can bitrate "$BITRATE"
        sudo ip link set dev "$target" up
    done
fi

failures=0
for bus_info in "${EXPECTED_BUS_INFOS[@]}"; do
    target="${TARGET_BY_BUS[$bus_info]}"
    if ! ip link show dev "$target" >/dev/null 2>&1; then
        echo "FAIL: expected interface name $target does not exist" >&2
        failures=$((failures + 1))
        continue
    fi
    actual_bus_info="$(bus_info_for "$target")"
    actual_bitrate="$(bitrate_for "$target")"
    state="$(ip -br link show dev "$target" | awk '{print $2}')"

    if [[ "$actual_bus_info" != "$bus_info" || "$actual_bitrate" != "$BITRATE" || "$state" != "UP" ]]; then
        echo "FAIL: $target bus=$actual_bus_info state=$state bitrate=${actual_bitrate:-unset}" >&2
        failures=$((failures + 1))
    else
        echo "PASS: $bus_info -> $target state=UP bitrate=$BITRATE"
    fi
done

if ((failures > 0)); then
    exit 1
fi

echo "CAN_INTERFACE_CHECK=PASS"
echo "Physical role check is still required: init can_left_slave first and verify only the insertion arm moves."
