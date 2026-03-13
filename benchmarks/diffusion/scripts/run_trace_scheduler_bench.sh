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
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/benchmarks/dataset/sd3_trace_redistributed.txt}"
LOG_LEVEL="${LOG_LEVEL:-info}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/trace_scheduler_bench_${RUN_TAG}}"

SERVER_LOG="${OUT_DIR}/server_${POLICY}.log"
RESULT_JSON="${OUT_DIR}/result_${POLICY}.json"

mkdir -p "${OUT_DIR}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -- "-${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
}

trap cleanup EXIT

echo "Starting server"
echo "  model: ${MODEL}"
echo "  policy: ${POLICY}"
echo "  port: ${PORT}"
echo "  dataset: ${DATASET_PATH}"
echo "  out_dir: ${OUT_DIR}"

cd "${ROOT_DIR}"

setsid vllm serve \
  --model "${MODEL}" \
  --omni \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --instance-scheduler-policy "${POLICY}" \
  --log-level "${LOG_LEVEL}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

READY_URL="http://${HOST}:${PORT}/v1/models"
READY_TIMEOUT_SEC=600
READY_INTERVAL_SEC=2
READY_START_TS="$(date +%s)"

echo "Waiting for server readiness: ${READY_URL}"
while true; do
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    break
  fi
  NOW_TS="$(date +%s)"
  if (( NOW_TS - READY_START_TS >= READY_TIMEOUT_SEC )); then
    echo "Server did not become ready within ${READY_TIMEOUT_SEC}s"
    echo "Last server log lines:"
    tail -n 100 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${READY_INTERVAL_SEC}"
done

echo "Server ready, running benchmark"

cd "${BENCH_DIR}"
python diffusion_benchmark_serving.py \
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
  --output-file "${RESULT_JSON}"

echo
echo "Benchmark finished"
echo "  result_json: ${RESULT_JSON}"
echo "  server_log: ${SERVER_LOG}"
echo
echo "Key scheduler log lines:"
grep -E "QUEUE_REORDER|QUEUE_DEQUEUE|QUEUE_ENQUEUE|REQUEST_DONE|REQUEST_FAIL" "${SERVER_LOG}" | tail -n 80 || true
