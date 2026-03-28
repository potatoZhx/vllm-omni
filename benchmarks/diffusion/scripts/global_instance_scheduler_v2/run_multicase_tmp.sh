#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"

echo "[1/2] running command 1..."
WORKER_IDS=worker0 \
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
BENCHMARK_MODE=fixed_num_prompts \
FIXED_NUM_PROMPTS=100 \
REQUEST_RATES=0.125 \
CHUNK_BUDGET_STEPS=5 \
SMALL_REQUEST_LATENCY_THRESHOLD_MS=6000 \
CASE_MATRIX=$'type_fifo_defer_budget|round_robin|type_fifo_defer_budget|1|1|5' \
"${SCRIPT_DIR}/run_suite.sh"

echo "[2/2] running command 2..."
WORKER_IDS=worker0 \
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
BENCHMARK_MODE=fixed_num_prompts \
FIXED_NUM_PROMPTS=100 \
REQUEST_RATES=0.125 \
CHUNK_BUDGET_STEPS=5 \
SMALL_REQUEST_LATENCY_THRESHOLD_MS=6000 \
CASE_MATRIX=$'sjf_aging_guarded_tail|round_robin|sjf_aging_guarded_tail|1|1|5' \
"${SCRIPT_DIR}/run_suite.sh"

echo "all done"
