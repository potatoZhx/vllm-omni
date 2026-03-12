#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/global_scheduler.yaml}"
BENCHMARK_SCRIPT="${REPO_ROOT}/scripts/run_global_scheduler_benchmark.sh"
SCHEDULER_MODULE="${SCHEDULER_MODULE:-vllm_omni.global_scheduler.server}"
SCHEDULER_LOG_FILE="${SCHEDULER_LOG_FILE:-${REPO_ROOT}/logs/global_scheduler_server.log}"
SCHEDULER_READY_TIMEOUT_S="${SCHEDULER_READY_TIMEOUT_S:-60}"
SCHEDULER_POLL_INTERVAL_S="${SCHEDULER_POLL_INTERVAL_S:-1}"
SCHEDULER_PID=""
SCHEDULER_URL=""
_CLEANED_UP=0

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is required." >&2
    exit 1
  fi
}

load_scheduler_env() {
  local config_lines
  config_lines="$(
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${CONFIG_FILE}" <<'PY'
import shlex
import sys
from pathlib import Path

from vllm_omni.global_scheduler.config import load_config

config_path = Path(sys.argv[1]).resolve()
config = load_config(config_path)
host = config.server.host
if host in {"0.0.0.0", "::"}:
    host = "127.0.0.1"

values = {
    "SCHEDULER_URL": f"http://{host}:{config.server.port}",
}

for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
  )"

  while IFS= read -r line; do
    if [[ "${line}" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
      eval "${line}"
    fi
  done <<<"${config_lines}"
}

curl_local() {
  curl --noproxy '*' "$@"
}

wait_scheduler_ready() {
  local deadline
  deadline=$((SECONDS + SCHEDULER_READY_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if curl_local -fsS "${SCHEDULER_URL}/health" >/dev/null 2>&1; then
      echo "[ready] scheduler: ${SCHEDULER_URL}"
      return 0
    fi
    sleep "${SCHEDULER_POLL_INTERVAL_S}"
  done
  echo "Scheduler did not become ready within ${SCHEDULER_READY_TIMEOUT_S}s: ${SCHEDULER_URL}" >&2
  return 1
}

cleanup() {
  if [[ "${_CLEANED_UP}" == "1" ]]; then
    return 0
  fi
  _CLEANED_UP=1

  if [[ -n "${SCHEDULER_PID}" ]] && kill -0 "${SCHEDULER_PID}" >/dev/null 2>&1; then
    echo "[cleanup] stopping scheduler pid=${SCHEDULER_PID}"
    kill -TERM "${SCHEDULER_PID}" >/dev/null 2>&1 || true
    wait "${SCHEDULER_PID}" >/dev/null 2>&1 || true
  fi
}

start_scheduler() {
  mkdir -p "$(dirname "${SCHEDULER_LOG_FILE}")"
  echo "[start] scheduler: ${SCHEDULER_URL}"
  echo "[log] ${SCHEDULER_LOG_FILE}"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python3 -m "${SCHEDULER_MODULE}" --config "${CONFIG_FILE}" >"${SCHEDULER_LOG_FILE}" 2>&1 &
  SCHEDULER_PID="$!"
}

main() {
  require_cmd python3
  require_cmd curl

  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file not found: ${CONFIG_FILE}" >&2
    exit 1
  fi
  if [[ ! -x "${BENCHMARK_SCRIPT}" ]]; then
    echo "Benchmark script is not executable: ${BENCHMARK_SCRIPT}" >&2
    exit 1
  fi

  trap cleanup EXIT INT TERM

  load_scheduler_env
  start_scheduler
  wait_scheduler_ready

  echo "[run] benchmark: ${BENCHMARK_SCRIPT}"
  CONFIG_FILE="${CONFIG_FILE}" "${BENCHMARK_SCRIPT}" "$@"
  cleanup
}

main "$@"
