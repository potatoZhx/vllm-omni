#!/usr/bin/env bash
set -euo pipefail

# Preconditions:
# 1) Scheduler is already running, e.g.:
#    python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
# 2) Lifecycle config exists in global_scheduler.yaml for worker ids below.

NUM_PROMPTS="${NUM_PROMPTS:-10}"
REQUEST_RATE="${REQUEST_RATE:-0.5}"
RPS_LIST="${RPS_LIST:-}"
BACKEND="${BACKEND:-vllm-omni}"
STARTED_WORKERS=0
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
  local probe_status

  # For non-chat backends, use lightweight health probe to avoid calling
  # unrelated generation APIs during readiness checks.
  if [[ "${BACKEND}" != "vllm-omni" ]]; then
    probe_status="$(
      curl_local -sS --max-time 10 -o /dev/null -w "%{http_code}" \
        "${endpoint%/}/health" || true
    )"
    if [[ "${probe_status}" == "200" ]]; then
      echo "1"
    else
      echo "0"
    fi
    return 0
  fi

  local tmp_body
  local status
  local payload
  tmp_body="$(mktemp)"
  payload="$(cat <<EOF
{"model":"${MODEL}","messages":[{"role":"user","content":"ready-check"}],"extra_body":{"width":64,"height":64,"num_inference_steps":1}}
EOF
)"

  status="$(
    curl_local -sS --max-time 30 -o "${tmp_body}" -w "%{http_code}" \
      -X POST "${endpoint%/}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "${payload}" || true
  )"
  if [[ "${status}" == "200" ]]; then
    rm -f "${tmp_body}"
    echo "1"
    return 0
  fi
  if grep -qi "model.*not found" "${tmp_body}" 2>/dev/null; then
    rm -f "${tmp_body}"
    echo "MODEL_NOT_FOUND"
    return 0
  fi
  rm -f "${tmp_body}"
  echo "0"
}

wait_workers_ready() {
  local timeout_s="${1:-600}"

  for wid in ${WORKER_IDS}; do
    local start_ts
    start_ts="$(date +%s)"
    while true; do
      local endpoint
      local routable
      local ready
      routable="$(is_worker_routable "${wid}")"
      endpoint="$(get_worker_endpoint "${wid}")"
      if [[ -n "${endpoint}" ]]; then
        ready="$(is_worker_api_ready "${endpoint}")"
        if [[ "${routable}" == "1" && "${ready}" == "1" ]]; then
          if [[ "${BACKEND}" == "vllm-omni" ]]; then
            echo "[ready] ${wid} routable=true api_ready=true (${endpoint%/}/v1/chat/completions)"
          else
            echo "[ready] ${wid} routable=true api_ready=true (${endpoint%/}/health, backend=${BACKEND})"
          fi
          break
        fi
        if [[ "${ready}" == "MODEL_NOT_FOUND" ]]; then
          echo "Model not found on ${wid}: MODEL=${MODEL}" >&2
          return 1
        fi
      fi
      if (( $(date +%s) - start_ts > timeout_s )); then
        echo "Timeout waiting for worker ready (routable + api): ${wid}" >&2
        return 1
      fi
      sleep 2
    done
  done
}

normalize_rps_list() {
  local raw="$1"
  echo "$raw" | sed 's/\[//g; s/\]//g; s/,/ /g'
}

main() {
  trap on_signal INT TERM
  trap on_exit EXIT

  echo "[check] scheduler health: ${SCHEDULER_URL}"
  if ! check_scheduler_ready; then
    echo "Scheduler is not reachable at ${SCHEDULER_URL}" >&2
    exit 1
  fi

  if [[ "${BACKEND}" != "vllm-omni" && "${BACKEND}" != "openai" && "${BACKEND}" != "v1/videos" ]]; then
    echo "Invalid BACKEND: ${BACKEND}. Expected one of: vllm-omni, openai, v1/videos" >&2
    exit 1
  fi

  wait_workers_ready "${WORKER_READY_TIMEOUT_S}"

  local effective_rps_list
  if [[ -n "${RPS_LIST}" ]]; then
    effective_rps_list="${RPS_LIST}"
  else
    effective_rps_list="[${REQUEST_RATE}]"
  fi

  local rps_items
  rps_items="$(normalize_rps_list "${effective_rps_list}")"
  if [[ -z "${rps_items}" ]]; then
    echo "RPS list is empty after parsing: ${effective_rps_list}" >&2
    return 1
  fi

  echo "[bench] start benchmark, rps list: ${effective_rps_list}"
  local bench_status=0
  local rps
  for rps in ${rps_items}; do
    local output_file_for_rps="${OUTPUT_FILE}"
    if [[ -n "${OUTPUT_FILE}" && "${effective_rps_list}" != "[${REQUEST_RATE}]" ]]; then
      local base="${OUTPUT_FILE}"
      local ext=""
      if [[ "${OUTPUT_FILE}" == *.* ]]; then
        base="${OUTPUT_FILE%.*}"
        ext=".${OUTPUT_FILE##*.}"
      fi
      local rps_label="${rps//./_}"
      output_file_for_rps="${base}_rps_${rps_label}${ext}"
    fi

    echo "[bench] running rps=${rps}, num_prompts=${NUM_PROMPTS}"
    cmd=(
      python3 "${BENCH_SCRIPT}"
      --base-url "${SCHEDULER_URL}"
      --model "${MODEL}"
      --backend "${BACKEND}"
      --task "${TASK}"
      --dataset "${DATASET}"
      --num-prompts "${NUM_PROMPTS}"
      --max-concurrency "${MAX_CONCURRENCY}"
      --request-rate "${rps}"
      --warmup-requests "${WARMUP_REQUESTS}"
      --warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS}"
    )
    if [[ -n "${DATASET_PATH}" ]]; then
      cmd+=(--dataset-path "${DATASET_PATH}")
    fi
    if [[ -n "${output_file_for_rps}" ]]; then
      cmd+=(--output-file "${output_file_for_rps}")
    fi

    echo "Running: ${cmd[*]}"
    "${cmd[@]}" &
    BENCH_PID="$!"
    BENCH_RUNNING=1
    if ! wait "${BENCH_PID}"; then
      bench_status=$?
      BENCH_RUNNING=0
      BENCH_PID=""
      return "${bench_status}"
    fi
    BENCH_RUNNING=0
    BENCH_PID=""
  done

  return "${bench_status}"
}

main "$@"
