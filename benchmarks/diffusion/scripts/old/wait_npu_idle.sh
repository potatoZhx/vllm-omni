#!/usr/bin/env bash
set -euo pipefail

# Wait until NPU memory usage is low enough, then report and exit.
#
# Defaults:
#   - mode=any: exits when any selected device is below threshold
#   - threshold-mb=2048
#   - interval-sec=10
#   - consecutive=3: condition must hold for N consecutive checks
#   - timeout-sec=0: no timeout
#   - devices=all: monitor all detected NPUs

MODE="any"
THRESHOLD_MB=2048
INTERVAL_SEC=10
CONSECUTIVE=3
TIMEOUT_SEC=0
DEVICES="all"
VERBOSE=1

usage() {
  cat <<'EOF'
Usage:
  bash wait_npu_idle.sh [options]

Options:
  --mode any|all           Trigger when any or all selected devices are idle. Default: any
  --threshold-mb N         Idle threshold for used memory (MB). Default: 2048
  --interval-sec N         Poll interval seconds. Default: 10
  --consecutive N          Required consecutive idle checks. Default: 3
  --timeout-sec N          Stop waiting after N seconds; 0 means no timeout. Default: 0
  --devices LIST           Device indices to monitor, e.g. 0,1,3 or all. Default: all
  --quiet                  Reduce periodic log output
  -h, --help               Show this help

Exit codes:
  0  Idle condition met
  2  Timed out
  3  Invalid arguments
  4  npu-smi not found
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 3
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --threshold-mb)
      [[ $# -ge 2 ]] || die "--threshold-mb requires a value"
      THRESHOLD_MB="$2"
      shift 2
      ;;
    --interval-sec)
      [[ $# -ge 2 ]] || die "--interval-sec requires a value"
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --consecutive)
      [[ $# -ge 2 ]] || die "--consecutive requires a value"
      CONSECUTIVE="$2"
      shift 2
      ;;
    --timeout-sec)
      [[ $# -ge 2 ]] || die "--timeout-sec requires a value"
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --devices)
      [[ $# -ge 2 ]] || die "--devices requires a value"
      DEVICES="$2"
      shift 2
      ;;
    --quiet)
      VERBOSE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$MODE" == "any" || "$MODE" == "all" ]] || die "--mode must be any or all"
[[ "$THRESHOLD_MB" =~ ^[0-9]+$ ]] || die "--threshold-mb must be a non-negative integer"
[[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || die "--interval-sec must be a non-negative integer"
[[ "$CONSECUTIVE" =~ ^[1-9][0-9]*$ ]] || die "--consecutive must be a positive integer"
[[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || die "--timeout-sec must be a non-negative integer"

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[ERROR] npu-smi not found in PATH" >&2
  exit 4
fi

parse_used_mems() {
  python3 - <<'PY'
import re
import sys

text = sys.stdin.read()
used = []

# Heuristic parse: capture pairs like "1234 / 65536" and treat large totals as memory.
for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", text):
    u = int(a)
    t = int(b)
    if 1024 <= t <= 200000 and 0 <= u <= t:
      used.append(u)

# De-duplicate near-duplicate streams while keeping order.
out = []
for x in used:
    if not out or out[-1] != x:
        out.append(x)

print(" ".join(str(x) for x in out))
PY
}

select_indices() {
  local count="$1"
  if [[ "$DEVICES" == "all" ]]; then
    seq 0 $((count - 1))
    return
  fi

  local token
  local old_ifs="$IFS"
  IFS=','
  for token in $DEVICES; do
    [[ "$token" =~ ^[0-9]+$ ]] || die "invalid device index: $token"
    if (( token < 0 || token >= count )); then
      die "device index out of range: $token (detected $count devices)"
    fi
    echo "$token"
  done
  IFS="$old_ifs"
}

start_ts=$(date +%s)
pass_count=0

if (( VERBOSE == 1 )); then
  echo "[INFO] Start waiting for NPU idle"
  echo "[INFO] mode=${MODE}, threshold_mb=${THRESHOLD_MB}, interval_sec=${INTERVAL_SEC}, consecutive=${CONSECUTIVE}, timeout_sec=${TIMEOUT_SEC}, devices=${DEVICES}"
fi

while true; do
  now_ts=$(date +%s)
  elapsed=$((now_ts - start_ts))

  if (( TIMEOUT_SEC > 0 && elapsed >= TIMEOUT_SEC )); then
    echo "[TIMEOUT] waited ${elapsed}s, idle condition not met" >&2
    exit 2
  fi

  raw="$(npu-smi info 2>&1 || true)"
  used_line="$(printf '%s' "$raw" | parse_used_mems)"

  if [[ -z "$used_line" ]]; then
    pass_count=0
    echo "[WARN] unable to parse memory usage from npu-smi output; retrying..."
    sleep "$INTERVAL_SEC"
    continue
  fi

  read -r -a used_array <<< "$used_line"
  total_devices=${#used_array[@]}

  if (( total_devices == 0 )); then
    pass_count=0
    echo "[WARN] zero devices parsed; retrying..."
    sleep "$INTERVAL_SEC"
    continue
  fi

  mapfile -t selected_idx < <(select_indices "$total_devices")

  idle_devices=()
  busy_devices=()
  selected_report=()

  for idx in "${selected_idx[@]}"; do
    used_mb=${used_array[$idx]}
    selected_report+=("${idx}:${used_mb}MB")
    if (( used_mb <= THRESHOLD_MB )); then
      idle_devices+=("$idx")
    else
      busy_devices+=("$idx")
    fi
  done

  condition_met=0
  if [[ "$MODE" == "any" ]]; then
    if (( ${#idle_devices[@]} > 0 )); then
      condition_met=1
    fi
  else
    if (( ${#busy_devices[@]} == 0 )); then
      condition_met=1
    fi
  fi

  if (( VERBOSE == 1 )); then
    echo "[CHECK] elapsed=${elapsed}s selected=(${selected_report[*]}) idle=(${idle_devices[*]:-none}) busy=(${busy_devices[*]:-none})"
  fi

  if (( condition_met == 1 )); then
    pass_count=$((pass_count + 1))
    if (( pass_count >= CONSECUTIVE )); then
      ts_now="$(date -Iseconds)"
      echo "[IDLE] condition met at ${ts_now}"
      echo "[IDLE] mode=${MODE}, threshold_mb=${THRESHOLD_MB}, selected=(${selected_report[*]})"
      if (( ${#idle_devices[@]} > 0 )); then
        echo "[IDLE] idle_devices=(${idle_devices[*]})"
      fi
      exit 0
    fi
  else
    pass_count=0
  fi

  sleep "$INTERVAL_SEC"
done
