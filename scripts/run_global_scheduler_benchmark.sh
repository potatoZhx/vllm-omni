#!/usr/bin/env bash
set -euo pipefail

# Preconditions:
# 1) Scheduler is already running, e.g.:
#    python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
# 2) Lifecycle config exists in global_scheduler.yaml for worker ids below.

NUM_PROMPTS="${NUM_PROMPTS:-20}"
REQUEST_RATE="${REQUEST_RATE:-0.1}"
REQUEST_RATES="${REQUEST_RATES:-0.2,0.4,0.6,0.8,1.0}"
REQUEST_DURATION_S="${REQUEST_DURATION_S:-600}"
BACKEND="${BACKEND:-vllm-omni}"
BENCH_RUNNING=0
BENCH_PID=""
_CLEANED_UP=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/benchmarks/diffusion/diffusion_benchmark_serving.py"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/global_scheduler.yaml}"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "Benchmark script not found: ${BENCH_SCRIPT}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

load_benchmark_config() {
  local config_lines
  config_lines="$(
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${CONFIG_FILE}" <<'PY'
import shlex
import sys
from pathlib import Path

from vllm_omni.global_scheduler.config import load_config

config_path = Path(sys.argv[1]).resolve()
config = load_config(config_path)
benchmark = config.benchmark
instances_by_id = {instance.id: instance for instance in config.instances}
worker_ids = benchmark.worker_ids or [instance.id for instance in config.instances]
missing_worker_ids = [worker_id for worker_id in worker_ids if worker_id not in instances_by_id]
if missing_worker_ids:
    raise ValueError(f"benchmark.worker_ids contains unknown instances: {', '.join(missing_worker_ids)}")

model = benchmark.model
if model is None:
    launch_models = {
        instance.launch.model
        for worker_id in worker_ids
        for instance in [instances_by_id[worker_id]]
        if instance.launch is not None and instance.launch.model
    }
    if len(launch_models) != 1:
        raise ValueError("benchmark.model is required when selected workers do not share exactly one launch.model")
    model = next(iter(launch_models))

scheduler_host = config.server.host
if scheduler_host in {"0.0.0.0", "::"}:
    scheduler_host = "127.0.0.1"
scheduler_url = f"http://{scheduler_host}:{config.server.port}"

def resolve_config_path(value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)

values = {
    "SCHEDULER_HOST": scheduler_host,
    "SCHEDULER_URL": scheduler_url,
    "WORKER_IDS": " ".join(worker_ids),
    "WORKER_READY_TIMEOUT_S": str(benchmark.worker_ready_timeout_s),
    "MODEL": model,
    "BACKEND": benchmark.backend,
    "TASK": benchmark.task,
    "DATASET": benchmark.dataset,
    "DATASET_PATH": resolve_config_path(benchmark.dataset_path),
    "RANDOM_REQUEST_CONFIG": benchmark.random_request_config or "",
    "MAX_CONCURRENCY": str(benchmark.max_concurrency),
    "WARMUP_REQUESTS": str(benchmark.warmup_requests),
    "WARMUP_NUM_INFERENCE_STEPS": str(benchmark.warmup_num_inference_steps),
    "OUTPUT_FILE": resolve_config_path(benchmark.output_file),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
  )"

  while IFS= read -r line; do
    if [[ ! "${line}" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
      continue
    fi
    eval "${line}"
  done <<<"${config_lines}"
}

load_benchmark_config

# Keep global proxy settings untouched, but force local loopback traffic
# in this script to bypass proxy.
DEFAULT_NO_PROXY="127.0.0.1,localhost"
if [[ "${SCHEDULER_HOST}" == "127.0.0.1" || "${SCHEDULER_HOST}" == "localhost" ]]; then
  DEFAULT_NO_PROXY="${DEFAULT_NO_PROXY},${SCHEDULER_HOST}"
fi
export NO_PROXY="${NO_PROXY:-${DEFAULT_NO_PROXY}}"
export no_proxy="${no_proxy:-${DEFAULT_NO_PROXY}}"

curl_local() {
  curl --noproxy '*' "$@"
}

check_scheduler_ready() {
  local health
  health="$(curl_local -fsS "${SCHEDULER_URL}/health" || true)"
  if [[ -z "${health}" ]]; then
    return 1
  fi
  return 0
}

is_worker_routable() {
  local worker_id="$1"
  local instances_json
  instances_json="$(curl_local -fsS "${SCHEDULER_URL}/instances" || true)"
  if [[ -z "${instances_json}" ]]; then
    echo "0"
    return 0
  fi
  INSTANCES_JSON="${instances_json}" python3 - "${worker_id}" <<'PY'
import json
import os
import sys

worker_id = sys.argv[1]
raw = os.environ.get("INSTANCES_JSON", "")
if not raw:
    print("0")
    raise SystemExit(0)

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    print("0")
    raise SystemExit(0)

for inst in payload.get("instances", []):
    if inst.get("id") == worker_id:
        print("1" if inst.get("routable") else "0")
        break
else:
    print("0")
PY
}

get_worker_endpoint() {
  local worker_id="$1"
  local instances_json
  instances_json="$(curl_local -fsS "${SCHEDULER_URL}/instances" || true)"
  if [[ -z "${instances_json}" ]]; then
    echo ""
    return 0
  fi
  INSTANCES_JSON="${instances_json}" python3 - "${worker_id}" <<'PY'
import json
import os
import sys

worker_id = sys.argv[1]
raw = os.environ.get("INSTANCES_JSON", "")
if not raw:
    print("")
    raise SystemExit(0)

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)

for inst in payload.get("instances", []):
    if inst.get("id") == worker_id:
        print(inst.get("endpoint", "") or "")
        break
else:
    print("")
PY
}

terminate_benchmark() {
  if [[ "${BENCH_RUNNING}" != "1" || -z "${BENCH_PID}" ]]; then
    return 0
  fi
  if kill -0 "${BENCH_PID}" >/dev/null 2>&1; then
    kill -TERM "${BENCH_PID}" >/dev/null 2>&1 || true
    wait "${BENCH_PID}" >/dev/null 2>&1 || true
  fi
  BENCH_RUNNING=0
  BENCH_PID=""
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

parse_request_rates() {
  local rates_raw="${REQUEST_RATES:-${REQUEST_RATE}}"
  local normalized=()
  local token
  rates_raw="${rates_raw//,/ }"
  for token in ${rates_raw}; do
    token="$(trim "${token}")"
    if [[ -z "${token}" ]]; then
      continue
    fi
    normalized+=("${token}")
  done

  if [[ "${#normalized[@]}" -eq 0 ]]; then
    echo "No request rates configured. Set REQUEST_RATE or REQUEST_RATES." >&2
    return 1
  fi

  REQUEST_RATE_LIST=("${normalized[@]}")
}

sanitize_rate_for_filename() {
  local rate="$1"
  rate="${rate//./p}"
  rate="${rate//[^a-zA-Z0-9_-]/_}"
  printf '%s' "${rate}"
}

resolve_num_prompts_for_rate() {
  local rate="$1"
  if [[ -n "${REQUEST_DURATION_S}" ]]; then
    python3 - "${rate}" "${REQUEST_DURATION_S}" <<'PY'
import math
import sys

rate = float(sys.argv[1])
duration = float(sys.argv[2])
if not math.isfinite(rate):
    raise ValueError("REQUEST_DURATION_S requires a finite request rate")
if duration <= 0:
    raise ValueError("REQUEST_DURATION_S must be > 0")
if rate <= 0:
    raise ValueError("request rate must be > 0 when REQUEST_DURATION_S is set")
print(max(1, math.ceil(rate * duration)))
PY
    return 0
  fi

  printf '%s\n' "${NUM_PROMPTS}"
}

resolve_output_file_for_rate() {
  local rate="$1"
  if [[ -z "${OUTPUT_FILE}" ]]; then
    return 0
  fi

  if [[ "${#REQUEST_RATE_LIST[@]}" -le 1 ]]; then
    printf '%s\n' "${OUTPUT_FILE}"
    return 0
  fi

  python3 - "${OUTPUT_FILE}" "$(sanitize_rate_for_filename "${rate}")" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
suffix = sys.argv[2]
if path.suffix:
    print(str(path.with_name(f"{path.stem}_rps_{suffix}{path.suffix}")))
else:
    print(str(path.with_name(f"{path.name}_rps_{suffix}")))
PY
}

cleanup() {
  if [[ "${_CLEANED_UP}" == "1" ]]; then
    return 0
  fi
  _CLEANED_UP=1

  terminate_benchmark
}

on_signal() {
  echo "[signal] interrupted, cleaning up..."
  cleanup
  exit 130
}

on_exit() {
  cleanup
}

is_worker_api_ready() {
  local endpoint="$1"
  local models_json
  models_json="$(curl_local -fsS --max-time 30 "${endpoint%/}/v1/models" || true)"
  if [[ -z "${models_json}" ]]; then
    echo "0"
    return 0
  fi

  MODELS_JSON="${models_json}" python3 - <<'PY'
import json
import os

raw = os.environ.get("MODELS_JSON", "")
if not raw:
    print("0")
    raise SystemExit(0)

try:
    payload = json.loads(raw)
except json.JSONDecodeError:
    print("0")
    raise SystemExit(0)

models = payload.get("data")
if isinstance(models, list) and len(models) > 0:
    print("1")
else:
    print("0")
PY
}

wait_workers_ready() {
  local timeout_s="${1:-600}"
  local log_interval=60  # 每60秒打印一次等待时间

  for wid in ${WORKER_IDS}; do
    local start_ts
    start_ts="$(date +%s)"
    local last_log_ts="${start_ts}"

    while true; do
      local endpoint routable ready
      # 屏蔽所有 curl 错误输出
      routable="$(is_worker_routable "${wid}" 2>/dev/null)"
      endpoint="$(get_worker_endpoint "${wid}" 2>/dev/null)"
      
      if [[ -n "${endpoint}" ]]; then
        ready="$(is_worker_api_ready "${endpoint}" 2>/dev/null)"
        if [[ "${routable}" == "1" && "${ready}" == "1" ]]; then
          echo "[ready] ${wid} routable=true api_ready=true (${endpoint%/}/v1/models)"
          break
        fi
      fi

      # 计算已等待时间
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))

      # 超时判断
      if (( elapsed > timeout_s )); then
        echo "Timeout waiting for worker ready (${wid}) after ${elapsed}s" >&2
        return 1
      fi

      # 每 60s 输出一次等待时间（不刷屏）
      if (( now - last_log_ts >= log_interval )); then
        echo "[waiting] ${wid} has been waiting for ${elapsed}s ..."
        last_log_ts="${now}"
      fi

      sleep 2
    done
  done
}

run_benchmark_for_rate() {
  local rate="$1"
  local run_num_prompts
  local run_output_file

  run_num_prompts="$(resolve_num_prompts_for_rate "${rate}")"
  run_output_file="$(resolve_output_file_for_rate "${rate}")"

  echo "[bench] start benchmark: rate=${rate}, num_prompts=${run_num_prompts}${REQUEST_DURATION_S:+, request_duration_s=${REQUEST_DURATION_S}}"
  local cmd=(
    python3 "${BENCH_SCRIPT}"
    --base-url "${SCHEDULER_URL}"
    --backend "${BACKEND}"
    --model "${MODEL}"
    --task "${TASK}"
    --dataset "${DATASET}"
    --num-prompts "${run_num_prompts}"
    --max-concurrency "${MAX_CONCURRENCY}"
    --request-rate "${rate}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS}"
  )
  if [[ -n "${DATASET_PATH}" ]]; then
    cmd+=(--dataset-path "${DATASET_PATH}")
  fi
  if [[ -n "${RANDOM_REQUEST_CONFIG}" ]]; then
    cmd+=(--random-request-config "${RANDOM_REQUEST_CONFIG}")
  fi
  if [[ -n "${run_output_file}" ]]; then
    cmd+=(--output-file "${run_output_file}")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}" &
  BENCH_PID="$!"
  BENCH_RUNNING=1
  local bench_status=0
  if ! wait "${BENCH_PID}"; then
    bench_status=$?
  fi
  BENCH_RUNNING=0
  BENCH_PID=""
  return "${bench_status}"
}

main() {
  trap on_signal INT TERM
  trap on_exit EXIT

  parse_request_rates

  echo "[check] scheduler health: ${SCHEDULER_URL}"
  if ! check_scheduler_ready; then
    echo "Scheduler is not reachable at ${SCHEDULER_URL}" >&2
    exit 1
  fi

  wait_workers_ready "${WORKER_READY_TIMEOUT_S}"

  local rate
  for rate in "${REQUEST_RATE_LIST[@]}"; do
    run_benchmark_for_rate "${rate}"
  done
}

main "$@"
