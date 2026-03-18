#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/diffusion"
BENCH_SCRIPT="${BENCH_DIR}/diffusion_benchmark_serving.py"

MODEL="${MODEL:-Qwen/Qwen-Image}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
NUM_GPUS="${NUM_GPUS:-1}"
POLICY="${POLICY:-sjf}"
ENABLE_STEP_CHUNK="${ENABLE_STEP_CHUNK:-0}"
ENABLE_CHUNK_PREEMPTION="${ENABLE_CHUNK_PREEMPTION:-0}"
CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS:-4}"
SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/benchmarks/dataset/sd3_trace_redistributed.txt}"
STREAM_SERVER_LOGS="${STREAM_SERVER_LOGS:-1}"
INJECT_SCHEDULER_SLO="${INJECT_SCHEDULER_SLO:-0}"
SLO_SCALE="${SLO_SCALE:-3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/trace_scheduler_bench_${RUN_TAG}}"

SERVER_LOG="${OUT_DIR}/server_${POLICY}.log"
BENCH_LOG="${OUT_DIR}/benchmark_${POLICY}.log"
RESULT_JSON="${OUT_DIR}/result_${POLICY}.json"

mkdir -p "${OUT_DIR}"

find_port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' || true
  fi
}

wait_for_port_state() {
  local port="$1"
  local should_exist="$2"
  local timeout_sec="${3:-30}"
  local start_ts now_ts has_pid

  start_ts="$(date +%s)"
  while true; do
    has_pid=0
    if [[ -n "$(find_port_pids "${port}")" ]]; then
      has_pid=1
    fi

    if [[ "${should_exist}" == "1" && "${has_pid}" == "1" ]]; then
      return 0
    fi
    if [[ "${should_exist}" == "0" && "${has_pid}" == "0" ]]; then
      return 0
    fi

    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_sec )); then
      return 1
    fi
    sleep 1
  done
}

kill_server_processes() {
  local pid

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill -TERM "${pid}" 2>/dev/null || true
  done < <(find_port_pids "${PORT}")

  sleep 2

  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    kill -KILL "${pid}" 2>/dev/null || true
  done < <(find_port_pids "${PORT}")

  wait_for_port_state "${PORT}" 0 30 || true
}

cleanup() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "${TAIL_PID}" 2>/dev/null || true
    wait "${TAIL_PID}" 2>/dev/null || true
  fi
  kill_server_processes
}

trap cleanup EXIT

echo "Starting server"
echo "  model: ${MODEL}"
echo "  policy: ${POLICY}"
echo "  step_chunk: ${ENABLE_STEP_CHUNK}"
echo "  chunk_preemption: ${ENABLE_CHUNK_PREEMPTION}"
echo "  small_request_latency_threshold_ms: ${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-<disabled>}"
echo "  port: ${PORT}"
echo "  dataset: ${DATASET_PATH}"
echo "  out_dir: ${OUT_DIR}"

cd "${ROOT_DIR}"

if wait_for_port_state "${PORT}" 1 1; then
  echo "Port ${PORT} is already in use; cleaning up stale service before restart"
  kill_server_processes
fi

setsid nohup vllm serve \
  "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --instance-scheduler-policy "${POLICY}" \
  $([[ "${ENABLE_STEP_CHUNK}" == "1" ]] && printf '%s ' --diffusion-enable-step-chunk) \
  $([[ "${ENABLE_CHUNK_PREEMPTION}" == "1" ]] && printf '%s ' --diffusion-enable-chunk-preemption) \
  $([[ -n "${SMALL_REQUEST_LATENCY_THRESHOLD_MS}" ]] && printf '%s %s ' --diffusion-small-request-latency-threshold-ms "${SMALL_REQUEST_LATENCY_THRESHOLD_MS}") \
  --diffusion-chunk-budget-steps "${CHUNK_BUDGET_STEPS}" \
  >"${SERVER_LOG}" 2>&1 </dev/null &
SERVER_PID=$!

READY_URL="http://${HOST}:${PORT}/health"
READY_TIMEOUT_SEC=600
READY_INTERVAL_SEC=2
READY_START_TS="$(date +%s)"

echo "Waiting for server readiness: ${READY_URL}"
while true; do
  HTTP_CODE="$(curl -sS -o /dev/null -w '%{http_code}' "${READY_URL}" || true)"
  if [[ "${HTTP_CODE}" == "200" ]]; then
    break
  fi
  NOW_TS="$(date +%s)"
  if (( NOW_TS - READY_START_TS >= READY_TIMEOUT_SEC )); then
    echo "Server did not become ready within ${READY_TIMEOUT_SEC}s"
    echo "Last ready probe http_code: ${HTTP_CODE}"
    echo "Last server log lines:"
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
  fi
  echo "  waiting... http_code=${HTTP_CODE}"
  sleep "${READY_INTERVAL_SEC}"
done

echo "Server ready, running benchmark"
if [[ "${STREAM_SERVER_LOGS}" == "1" ]]; then
  tail -n 0 -F "${SERVER_LOG}" 2>/dev/null | sed 's/^/[server] /' &
  TAIL_PID=$!
fi

cd "${BENCH_DIR}"
echo "Benchmark command:"
echo "  python -u diffusion_benchmark_serving.py --backend openai --base-url http://${HOST}:${PORT} --model ${MODEL} --dataset trace --task t2i --dataset-path ${DATASET_PATH} --num-prompts ${NUM_PROMPTS} --warmup-requests ${WARMUP_REQUESTS} --request-rate ${REQUEST_RATE} --max-concurrency ${MAX_CONCURRENCY} --output-file ${RESULT_JSON}"
if [[ "${INJECT_SCHEDULER_SLO}" == "1" ]]; then
  echo "  scheduler SLO injection enabled: --inject-scheduler-slo --slo --slo-scale ${SLO_SCALE}"
fi
python -u diffusion_benchmark_serving.py \
  --backend openai \
  --base-url "http://${HOST}:${PORT}" \
  --model "${MODEL}" \
  --dataset trace \
  --task t2i \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts "${NUM_PROMPTS}" \
  --warmup-requests "${WARMUP_REQUESTS}" \
  --request-rate "${REQUEST_RATE}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --output-file "${RESULT_JSON}" \
  $([[ "${INJECT_SCHEDULER_SLO}" == "1" ]] && printf '%s ' --slo --slo-scale "${SLO_SCALE}" --inject-scheduler-slo) \
  2>&1 | tee "${BENCH_LOG}"

echo
echo "Benchmark finished"
echo "  result_json: ${RESULT_JSON}"
echo "  server_log: ${SERVER_LOG}"
echo
echo "Key scheduler log lines:"
grep -E "QUEUE_REORDER|QUEUE_DEQUEUE|QUEUE_ENQUEUE|REQUEST_DONE|REQUEST_FAIL" "${SERVER_LOG}" | tail -n 80 || true
grep -E "REQUEST_CHUNK_DONE" "${SERVER_LOG}" | tail -n 40 || true

if [[ "${POLICY}" == "slo_first" ]]; then
  echo
  echo "SLO-first evidence:"
  if grep -E "QUEUE_REORDER .*attain_before=.*attain_after=.*self_hit=.*damage_count=" "${SERVER_LOG}" >/dev/null; then
    grep -E "QUEUE_REORDER .*attain_before=.*attain_after=.*self_hit=.*damage_count=" "${SERVER_LOG}" | tail -n 20 || true
  else
    echo "  No slo_first reorder evidence found in server log."
    if [[ "${INJECT_SCHEDULER_SLO}" != "1" ]]; then
      echo "  Hint: set INJECT_SCHEDULER_SLO=1 to inject per-request slo_ms derived from --slo-scale."
    else
      echo "  Hint: the run may have lacked waiting-queue contention, so no reorder was triggered."
    fi
  fi
fi
