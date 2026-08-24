#!/usr/bin/env bash

# Read-only CAN inventory. This script never changes interface state or names.

set -euo pipefail

MASTER_BUS_INFO="1-4.1.1:1.0"
SLAVE_BUS_INFO="1-4.1.3:1.0"

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

role_for() {
    case "$1" in
        "$MASTER_BUS_INFO") printf '%s' "master (can_left_mas)" ;;
        "$SLAVE_BUS_INFO") printf '%s' "insertion follower (can_left_slave)" ;;
        *) printf '%s' "not managed by this experiment" ;;
    esac
}

require_command ip
require_command ethtool

mapfile -t interfaces < <(ip -br link show type can | awk '{print $1}')
if ((${#interfaces[@]} == 0)); then
    echo "ERROR: no CAN interfaces found." >&2
    exit 1
fi

printf '%-18s %-20s %-10s %-10s %s\n' "INTERFACE" "BUS-INFO" "STATE" "BITRATE" "EXPECTED ROLE"
for interface in "${interfaces[@]}"; do
    bus_info="$(bus_info_for "$interface")"
    state="$(ip -br link show dev "$interface" | awk '{print $2}')"
    bitrate="$(bitrate_for "$interface")"
    printf '%-18s %-20s %-10s %-10s %s\n' \
        "$interface" "${bus_info:-unknown}" "${state:-unknown}" "${bitrate:-unset}" \
        "$(role_for "$bus_info")"
done

echo
echo "Expected fixed mapping:"
echo "  $MASTER_BUS_INFO -> can_left_mas"
echo "  $SLAVE_BUS_INFO -> can_left_slave"
