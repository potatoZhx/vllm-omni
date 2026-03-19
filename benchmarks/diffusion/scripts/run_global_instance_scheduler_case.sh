#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/global_scheduler.yaml}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_global_scheduler_benchmark_one_shell.sh}"

GLOBAL_POLICY="${GLOBAL_POLICY:-min_queue_length}"
INSTANCE_POLICY="${INSTANCE_POLICY:-sjf}"
ENABLE_STEP_CHUNK="${ENABLE_STEP_CHUNK:-1}"
ENABLE_CHUNK_PREEMPTION="${ENABLE_CHUNK_PREEMPTION:-1}"
CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS:-4}"

REQUEST_RATES="${REQUEST_RATES:-0.2,0.4,0.6,0.8,1.0}"
REQUEST_DURATION_S="${REQUEST_DURATION_S:-600}"

CASE_NAME="${CASE_NAME:-global_${GLOBAL_POLICY}__instance_${INSTANCE_POLICY}__chunk_${ENABLE_STEP_CHUNK}__preempt_${ENABLE_CHUNK_PREEMPTION}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/benchmarks/diffusion/results/${CASE_NAME}_${RUN_TAG}}"
GENERATED_CONFIG="${OUT_DIR}/global_scheduler.generated.yaml"
BENCH_OUTPUT_FILE="${BENCH_OUTPUT_FILE:-${OUT_DIR}/metrics.json}"
SCHEDULER_LOG_FILE="${SCHEDULER_LOG_FILE:-${OUT_DIR}/global_scheduler_server.log}"

# Optional config overrides written into the generated YAML.
WORKER_IDS="${WORKER_IDS:-}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-}"
BENCHMARK_BACKEND="${BENCHMARK_BACKEND:-}"
BENCHMARK_TASK="${BENCHMARK_TASK:-}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-}"
BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH:-}"
BENCHMARK_RANDOM_REQUEST_CONFIG="${BENCHMARK_RANDOM_REQUEST_CONFIG:-}"
BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY:-}"
BENCHMARK_WARMUP_REQUESTS="${BENCHMARK_WARMUP_REQUESTS:-}"
BENCHMARK_WARMUP_NUM_INFERENCE_STEPS="${BENCHMARK_WARMUP_NUM_INFERENCE_STEPS:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is required." >&2
    exit 1
  fi
}

generate_config() {
  mkdir -p "${OUT_DIR}"
  GLOBAL_POLICY="${GLOBAL_POLICY}" \
  INSTANCE_POLICY="${INSTANCE_POLICY}" \
  ENABLE_STEP_CHUNK="${ENABLE_STEP_CHUNK}" \
  ENABLE_CHUNK_PREEMPTION="${ENABLE_CHUNK_PREEMPTION}" \
  CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS}" \
  WORKER_IDS="${WORKER_IDS}" \
  BENCH_OUTPUT_FILE="${BENCH_OUTPUT_FILE}" \
  BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
  BENCHMARK_BACKEND="${BENCHMARK_BACKEND}" \
  BENCHMARK_TASK="${BENCHMARK_TASK}" \
  BENCHMARK_DATASET="${BENCHMARK_DATASET}" \
  BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH}" \
  BENCHMARK_RANDOM_REQUEST_CONFIG="${BENCHMARK_RANDOM_REQUEST_CONFIG}" \
  BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY}" \
  BENCHMARK_WARMUP_REQUESTS="${BENCHMARK_WARMUP_REQUESTS}" \
  BENCHMARK_WARMUP_NUM_INFERENCE_STEPS="${BENCHMARK_WARMUP_NUM_INFERENCE_STEPS}" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${BASE_CONFIG}" "${GENERATED_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

import yaml


CONFIG_IN = Path(sys.argv[1]).resolve()
CONFIG_OUT = Path(sys.argv[2]).resolve()

GLOBAL_POLICY = os.environ["GLOBAL_POLICY"]
INSTANCE_POLICY = os.environ["INSTANCE_POLICY"]
ENABLE_STEP_CHUNK = os.environ["ENABLE_STEP_CHUNK"] == "1"
ENABLE_CHUNK_PREEMPTION = os.environ["ENABLE_CHUNK_PREEMPTION"] == "1"
CHUNK_BUDGET_STEPS = os.environ["CHUNK_BUDGET_STEPS"]

WORKER_IDS_RAW = os.environ.get("WORKER_IDS", "").replace(",", " ")
WORKER_IDS = [item.strip() for item in WORKER_IDS_RAW.split() if item.strip()]

BENCH_OUTPUT_FILE = os.environ["BENCH_OUTPUT_FILE"]

OPTIONAL_BENCH_OVERRIDES = {
    "model": os.environ.get("BENCHMARK_MODEL", "").strip(),
    "backend": os.environ.get("BENCHMARK_BACKEND", "").strip(),
    "task": os.environ.get("BENCHMARK_TASK", "").strip(),
    "dataset": os.environ.get("BENCHMARK_DATASET", "").strip(),
    "dataset_path": os.environ.get("BENCHMARK_DATASET_PATH", "").strip(),
    "random_request_config": os.environ.get("BENCHMARK_RANDOM_REQUEST_CONFIG", "").strip(),
    "max_concurrency": os.environ.get("BENCHMARK_MAX_CONCURRENCY", "").strip(),
    "warmup_requests": os.environ.get("BENCHMARK_WARMUP_REQUESTS", "").strip(),
    "warmup_num_inference_steps": os.environ.get("BENCHMARK_WARMUP_NUM_INFERENCE_STEPS", "").strip(),
}

payload = yaml.safe_load(CONFIG_IN.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise ValueError(f"Config root must be a mapping: {CONFIG_IN}")

payload.setdefault("policy", {}).setdefault("baseline", {})["algorithm"] = GLOBAL_POLICY

benchmark = payload.setdefault("benchmark", {})
benchmark["output_file"] = str(Path(BENCH_OUTPUT_FILE).resolve())
if WORKER_IDS:
    benchmark["worker_ids"] = WORKER_IDS
for key, value in OPTIONAL_BENCH_OVERRIDES.items():
    if not value:
        continue
    if key in {"max_concurrency", "warmup_requests", "warmup_num_inference_steps"}:
        benchmark[key] = int(value)
    else:
        benchmark[key] = value

target_worker_ids = set(benchmark.get("worker_ids") or [])
instances = payload.get("instances")
if not isinstance(instances, list) or not instances:
    raise ValueError("Config must contain non-empty instances list")

def strip_flag(args: list[str], flag: str) -> list[str]:
    filtered: list[str] = []
    idx = 0
    while idx < len(args):
        item = str(args[idx])
        if item == flag:
            idx += 2
            continue
        if item.startswith(flag + "="):
            idx += 1
            continue
        filtered.append(item)
        idx += 1
    return filtered

managed_flags = [
    "--instance-scheduler-policy",
    "--diffusion-enable-step-chunk",
    "--diffusion-enable-chunk-preemption",
    "--diffusion-chunk-budget-steps",
    "--diffusion-image-chunk-budget-steps",
    "--diffusion-video-chunk-budget-steps",
    "--diffusion-small-request-latency-threshold-ms",
]

matched_instances = []
for instance in instances:
    if not isinstance(instance, dict):
        continue
    instance_id = instance.get("id")
    if target_worker_ids and instance_id not in target_worker_ids:
        continue
    launch = instance.get("launch")
    if not isinstance(launch, dict):
        continue
    args = [str(item) for item in launch.get("args", [])]
    for flag in managed_flags:
        args = strip_flag(args, flag)
    args.extend(["--instance-scheduler-policy", INSTANCE_POLICY])
    if ENABLE_STEP_CHUNK:
        args.append("--diffusion-enable-step-chunk")
    if ENABLE_CHUNK_PREEMPTION:
        args.append("--diffusion-enable-chunk-preemption")
    args.extend(["--diffusion-chunk-budget-steps", str(int(CHUNK_BUDGET_STEPS))])
    launch["args"] = args
    matched_instances.append(instance_id)

if target_worker_ids and target_worker_ids - set(matched_instances):
    missing = ", ".join(sorted(target_worker_ids - set(matched_instances)))
    raise ValueError(f"Requested worker ids missing launch config or instance entry: {missing}")

CONFIG_OUT.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(CONFIG_OUT)
PY
}

main() {
  require_cmd python3
  require_cmd bash

  if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Base config not found: ${BASE_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -e "${RUNNER_SCRIPT}" ]]; then
    echo "Runner script not found: ${RUNNER_SCRIPT}" >&2
    exit 1
  fi

  generate_config

  echo "[case] ${CASE_NAME}"
  echo "[config] base=${BASE_CONFIG}"
  echo "[config] generated=${GENERATED_CONFIG}"
  echo "[policy] global=${GLOBAL_POLICY} instance=${INSTANCE_POLICY} step_chunk=${ENABLE_STEP_CHUNK} preemption=${ENABLE_CHUNK_PREEMPTION} chunk_budget=${CHUNK_BUDGET_STEPS}"
  echo "[rates] ${REQUEST_RATES}"
  echo "[out_dir] ${OUT_DIR}"

  CONFIG_FILE="${GENERATED_CONFIG}" \
  REQUEST_RATES="${REQUEST_RATES}" \
  REQUEST_DURATION_S="${REQUEST_DURATION_S}" \
  SCHEDULER_LOG_FILE="${SCHEDULER_LOG_FILE}" \
  "${RUNNER_SCRIPT}"

  echo "[done] case=${CASE_NAME}"
  echo "[artifacts] ${OUT_DIR}"
}

main "$@"
