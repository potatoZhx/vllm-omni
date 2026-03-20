#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CASE_SCRIPT="${CASE_SCRIPT:-${REPO_ROOT}/benchmarks/diffusion/scripts/run_global_instance_scheduler_case.sh}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/global_scheduler.yaml}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_global_scheduler_benchmark_one_shell.sh}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-global_instance_scheduler_rps_${RUN_TAG}}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/benchmarks/diffusion/results/${SUITE_NAME}}"

REQUEST_RATES="${REQUEST_RATES:-0.2,0.4,0.6,0.8,1.0}"
REQUEST_DURATION_S="${REQUEST_DURATION_S:-600}"
IMAGE_CHUNK_BUDGET_STEPS="${IMAGE_CHUNK_BUDGET_STEPS:-}"
VIDEO_CHUNK_BUDGET_STEPS="${VIDEO_CHUNK_BUDGET_STEPS:-}"
SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-}"

# Format per line:
# case_name|global_policy|instance_policy|enable_step_chunk|enable_chunk_preemption|chunk_budget_steps
CASE_MATRIX="${CASE_MATRIX:-qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4}"

BENCHMARK_MODEL="${BENCHMARK_MODEL:-}"
BENCHMARK_BACKEND="${BENCHMARK_BACKEND:-}"
BENCHMARK_TASK="${BENCHMARK_TASK:-}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-}"
BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH:-}"
BENCHMARK_RANDOM_REQUEST_CONFIG="${BENCHMARK_RANDOM_REQUEST_CONFIG:-}"
BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY:-}"
BENCHMARK_WARMUP_REQUESTS="${BENCHMARK_WARMUP_REQUESTS:-}"
BENCHMARK_WARMUP_NUM_INFERENCE_STEPS="${BENCHMARK_WARMUP_NUM_INFERENCE_STEPS:-}"
WORKER_IDS="${WORKER_IDS:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is required." >&2
    exit 1
  fi
}

write_summary() {
  python3 - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1]).resolve()
rows = []
for metrics_file in sorted(out_root.glob("*/metrics*.json")):
    case_dir = metrics_file.parent
    case_name = case_dir.name
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    rows.append(
        {
            "case": case_name,
            "metrics_file": str(metrics_file),
            "request_rate": metrics.get("request_rate"),
            "completed": metrics.get("completed"),
            "throughput_qps": metrics.get("throughput_qps"),
            "latency_p50": metrics.get("latency_p50"),
            "latency_p95": metrics.get("latency_p95"),
            "latency_p99": metrics.get("latency_p99"),
            "backend": metrics.get("backend"),
            "model": metrics.get("model"),
        }
    )

summary_json = out_root / "summary.json"
summary_csv = out_root / "summary.csv"
summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case",
            "request_rate",
            "completed",
            "throughput_qps",
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "backend",
            "model",
            "metrics_file",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(summary_json)
print(summary_csv)
PY
}

main() {
  require_cmd python3
  require_cmd bash

  if [[ ! -f "${CASE_SCRIPT}" ]]; then
    echo "Case script not found: ${CASE_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Base config not found: ${BASE_CONFIG}" >&2
    exit 1
  fi

  mkdir -p "${OUT_ROOT}"

  echo "[suite] ${SUITE_NAME}"
  echo "[base_config] ${BASE_CONFIG}"
  echo "[rates] ${REQUEST_RATES}"
  echo "[out_root] ${OUT_ROOT}"

  while IFS= read -r case_row; do
    [[ -z "${case_row// }" ]] && continue
    IFS='|' read -r case_name global_policy instance_policy enable_step_chunk enable_chunk_preemption chunk_budget_steps <<<"${case_row}"

    if [[ -z "${case_name}" || -z "${global_policy}" || -z "${instance_policy}" ]]; then
      echo "Invalid CASE_MATRIX row: ${case_row}" >&2
      exit 1
    fi

    case_out_dir="${OUT_ROOT}/${case_name}"
    mkdir -p "${case_out_dir}"

    echo
    echo "=== Running ${case_name} ==="
    echo "global=${global_policy} instance=${instance_policy} step_chunk=${enable_step_chunk} preemption=${enable_chunk_preemption} chunk_budget=${chunk_budget_steps}"

    GLOBAL_POLICY="${global_policy}" \
    INSTANCE_POLICY="${instance_policy}" \
    ENABLE_STEP_CHUNK="${enable_step_chunk:-1}" \
    ENABLE_CHUNK_PREEMPTION="${enable_chunk_preemption:-1}" \
    CHUNK_BUDGET_STEPS="${chunk_budget_steps:-4}" \
    CASE_NAME="${case_name}" \
    BASE_CONFIG="${BASE_CONFIG}" \
    RUNNER_SCRIPT="${RUNNER_SCRIPT}" \
    OUT_DIR="${case_out_dir}" \
    REQUEST_RATES="${REQUEST_RATES}" \
    REQUEST_DURATION_S="${REQUEST_DURATION_S}" \
    IMAGE_CHUNK_BUDGET_STEPS="${IMAGE_CHUNK_BUDGET_STEPS}" \
    VIDEO_CHUNK_BUDGET_STEPS="${VIDEO_CHUNK_BUDGET_STEPS}" \
    SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS}" \
    WORKER_IDS="${WORKER_IDS}" \
    BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
    BENCHMARK_BACKEND="${BENCHMARK_BACKEND}" \
    BENCHMARK_TASK="${BENCHMARK_TASK}" \
    BENCHMARK_DATASET="${BENCHMARK_DATASET}" \
    BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH}" \
    BENCHMARK_RANDOM_REQUEST_CONFIG="${BENCHMARK_RANDOM_REQUEST_CONFIG}" \
    BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY}" \
    BENCHMARK_WARMUP_REQUESTS="${BENCHMARK_WARMUP_REQUESTS}" \
    BENCHMARK_WARMUP_NUM_INFERENCE_STEPS="${BENCHMARK_WARMUP_NUM_INFERENCE_STEPS}" \
    bash "${CASE_SCRIPT}"
  done < <(printf '%s\n' "${CASE_MATRIX}")

  echo
  echo "=== Writing summary ==="
  write_summary >/tmp/global_instance_scheduler_summary_paths.$$ 
  cat /tmp/global_instance_scheduler_summary_paths.$$ | sed '1s/^/[summary_json] /;2s/^/[summary_csv] /'
  rm -f /tmp/global_instance_scheduler_summary_paths.$$
}

main "$@"
