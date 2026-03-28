#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/global_scheduler.yaml}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${REPO_ROOT}/scripts/run_global_scheduler_benchmark_one_shell.sh}"

GLOBAL_POLICY="${GLOBAL_POLICY:-}"
INSTANCE_POLICY="${INSTANCE_POLICY:-}"
ENABLE_STEP_CHUNK="${ENABLE_STEP_CHUNK:-}"
ENABLE_CHUNK_PREEMPTION="${ENABLE_CHUNK_PREEMPTION:-}"
CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS:-}"
IMAGE_CHUNK_BUDGET_STEPS="${IMAGE_CHUNK_BUDGET_STEPS:-}"
VIDEO_CHUNK_BUDGET_STEPS="${VIDEO_CHUNK_BUDGET_STEPS:-}"
SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-}"

REQUEST_RATES="${REQUEST_RATES:-0.2,0.4,0.6,0.8,1.0}"
REQUEST_DURATION_S="${REQUEST_DURATION_S:-600}"
BENCHMARK_MODE="${BENCHMARK_MODE:-fixed_duration}"
NUM_PROMPTS_DURATION_SECONDS="${NUM_PROMPTS_DURATION_SECONDS:-${REQUEST_DURATION_S}}"
FIXED_NUM_PROMPTS="${FIXED_NUM_PROMPTS:-20}"

CASE_NAME="${CASE_NAME:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-}"
GENERATED_CONFIG=""
BENCH_OUTPUT_FILE="${BENCH_OUTPUT_FILE:-}"
SCHEDULER_LOG_FILE="${SCHEDULER_LOG_FILE:-}"

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

resolve_case_name() {
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "$1" <<'PY2'
import sys
from pathlib import Path

from vllm_omni.global_scheduler.config import load_config


def get_flag(args: list[str], flag: str) -> str | None:
    idx = 0
    while idx < len(args):
        item = str(args[idx])
        if item == flag:
            if idx + 1 < len(args):
                return str(args[idx + 1])
            return ""
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
        idx += 1
    return None


def has_flag(args: list[str], flag: str) -> bool:
    return any(str(item) == flag for item in args)


config = load_config(Path(sys.argv[1]).resolve())
selected_ids = set(config.benchmark.worker_ids or [instance.id for instance in config.instances])
launch_args = [
    list(instance.launch.args)
    for instance in config.instances
    if instance.id in selected_ids and instance.launch is not None
]

instance_policy = "mixed"
step_chunk = "mixed"
chunk_preemption = "mixed"
if launch_args:
    instance_policies = {get_flag(args, "--instance-scheduler-policy") or "unset" for args in launch_args}
    step_chunks = {"1" if has_flag(args, "--diffusion-enable-step-chunk") else "0" for args in launch_args}
    chunk_preemptions = {
        "1" if has_flag(args, "--diffusion-enable-chunk-preemption") else "0"
        for args in launch_args
    }
    if len(instance_policies) == 1:
        instance_policy = next(iter(instance_policies))
    if len(step_chunks) == 1:
        step_chunk = next(iter(step_chunks))
    if len(chunk_preemptions) == 1:
        chunk_preemption = next(iter(chunk_preemptions))

print(
    f"global_{config.policy.baseline.algorithm}"
    f"__instance_{instance_policy}"
    f"__chunk_{step_chunk}"
    f"__preempt_{chunk_preemption}"
)
PY2
}

generate_config() {
  mkdir -p "${OUT_DIR}"
  GLOBAL_POLICY="${GLOBAL_POLICY}"   INSTANCE_POLICY="${INSTANCE_POLICY}"   ENABLE_STEP_CHUNK="${ENABLE_STEP_CHUNK}"   ENABLE_CHUNK_PREEMPTION="${ENABLE_CHUNK_PREEMPTION}"   CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS}"   IMAGE_CHUNK_BUDGET_STEPS="${IMAGE_CHUNK_BUDGET_STEPS}"   VIDEO_CHUNK_BUDGET_STEPS="${VIDEO_CHUNK_BUDGET_STEPS}"   SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS}"   WORKER_IDS="${WORKER_IDS}"   BENCH_OUTPUT_FILE="${BENCH_OUTPUT_FILE}"   BENCHMARK_MODEL="${BENCHMARK_MODEL}"   BENCHMARK_BACKEND="${BENCHMARK_BACKEND}"   BENCHMARK_TASK="${BENCHMARK_TASK}"   BENCHMARK_DATASET="${BENCHMARK_DATASET}"   BENCHMARK_DATASET_PATH="${BENCHMARK_DATASET_PATH}"   BENCHMARK_RANDOM_REQUEST_CONFIG="${BENCHMARK_RANDOM_REQUEST_CONFIG}"   BENCHMARK_MAX_CONCURRENCY="${BENCHMARK_MAX_CONCURRENCY}"   BENCHMARK_WARMUP_REQUESTS="${BENCHMARK_WARMUP_REQUESTS}"   BENCHMARK_WARMUP_NUM_INFERENCE_STEPS="${BENCHMARK_WARMUP_NUM_INFERENCE_STEPS}"   PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${BASE_CONFIG}" "${GENERATED_CONFIG}" <<'PY2'
import os
import sys
from pathlib import Path

import yaml


CONFIG_IN = Path(sys.argv[1]).resolve()
CONFIG_OUT = Path(sys.argv[2]).resolve()

GLOBAL_POLICY = os.environ.get("GLOBAL_POLICY", "").strip()
INSTANCE_POLICY = os.environ.get("INSTANCE_POLICY", "").strip()
ENABLE_STEP_CHUNK_RAW = os.environ.get("ENABLE_STEP_CHUNK", "").strip()
ENABLE_CHUNK_PREEMPTION_RAW = os.environ.get("ENABLE_CHUNK_PREEMPTION", "").strip()
CHUNK_BUDGET_STEPS = os.environ.get("CHUNK_BUDGET_STEPS", "").strip()
IMAGE_CHUNK_BUDGET_STEPS = os.environ.get("IMAGE_CHUNK_BUDGET_STEPS", "").strip()
VIDEO_CHUNK_BUDGET_STEPS = os.environ.get("VIDEO_CHUNK_BUDGET_STEPS", "").strip()
SMALL_REQUEST_LATENCY_THRESHOLD_MS = os.environ.get("SMALL_REQUEST_LATENCY_THRESHOLD_MS", "").strip()

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

policy_baseline = payload.setdefault("policy", {}).setdefault("baseline", {})
if GLOBAL_POLICY:
    policy_baseline["algorithm"] = GLOBAL_POLICY

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

    if INSTANCE_POLICY:
        args = strip_flag(args, "--instance-scheduler-policy")
        args.extend(["--instance-scheduler-policy", INSTANCE_POLICY])

    if ENABLE_STEP_CHUNK_RAW:
        args = strip_flag(args, "--diffusion-enable-step-chunk")
        if ENABLE_STEP_CHUNK_RAW == "1":
            args.append("--diffusion-enable-step-chunk")

    if ENABLE_CHUNK_PREEMPTION_RAW:
        args = strip_flag(args, "--diffusion-enable-chunk-preemption")
        if ENABLE_CHUNK_PREEMPTION_RAW == "1":
            args.append("--diffusion-enable-chunk-preemption")

    if CHUNK_BUDGET_STEPS:
        args = strip_flag(args, "--diffusion-chunk-budget-steps")
        args.extend(["--diffusion-chunk-budget-steps", str(int(CHUNK_BUDGET_STEPS))])

    if IMAGE_CHUNK_BUDGET_STEPS:
        args = strip_flag(args, "--diffusion-image-chunk-budget-steps")
        args.extend(["--diffusion-image-chunk-budget-steps", str(int(IMAGE_CHUNK_BUDGET_STEPS))])

    if VIDEO_CHUNK_BUDGET_STEPS:
        args = strip_flag(args, "--diffusion-video-chunk-budget-steps")
        args.extend(["--diffusion-video-chunk-budget-steps", str(int(VIDEO_CHUNK_BUDGET_STEPS))])

    if SMALL_REQUEST_LATENCY_THRESHOLD_MS:
        args = strip_flag(args, "--diffusion-small-request-latency-threshold-ms")
        args.extend(
            [
                "--diffusion-small-request-latency-threshold-ms",
                str(float(SMALL_REQUEST_LATENCY_THRESHOLD_MS)),
            ]
        )

    launch["args"] = args
    matched_instances.append(instance_id)

if target_worker_ids and target_worker_ids - set(matched_instances):
    missing = ", ".join(sorted(target_worker_ids - set(matched_instances)))
    raise ValueError(f"Requested worker ids missing launch config or instance entry: {missing}")

CONFIG_OUT.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print(CONFIG_OUT)
PY2
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

  local out_dir_was_provided=0
  local bench_output_was_provided=0
  local scheduler_log_was_provided=0

  if [[ -n "${OUT_DIR}" ]]; then
    out_dir_was_provided=1
  else
    OUT_DIR="${REPO_ROOT}/benchmarks/diffusion/results/.tmp_${RUN_TAG}_$$"
  fi

  if [[ -n "${BENCH_OUTPUT_FILE}" ]]; then
    bench_output_was_provided=1
  else
    BENCH_OUTPUT_FILE="${OUT_DIR}/metrics.json"
  fi

  if [[ -n "${SCHEDULER_LOG_FILE}" ]]; then
    scheduler_log_was_provided=1
  else
    SCHEDULER_LOG_FILE="${OUT_DIR}/global_scheduler_server.log"
  fi

  GENERATED_CONFIG="${OUT_DIR}/global_scheduler.generated.yaml"

  generate_config

  if [[ -z "${CASE_NAME}" ]]; then
    CASE_NAME="$(resolve_case_name "${GENERATED_CONFIG}" | tail -n 1)"
  fi

  if [[ "${out_dir_was_provided}" == "0" ]]; then
    local final_out_dir
    final_out_dir="${REPO_ROOT}/benchmarks/diffusion/results/${CASE_NAME}_${RUN_TAG}"
    mv "${OUT_DIR}" "${final_out_dir}"
    OUT_DIR="${final_out_dir}"
    GENERATED_CONFIG="${OUT_DIR}/global_scheduler.generated.yaml"
    if [[ "${bench_output_was_provided}" == "0" ]]; then
      BENCH_OUTPUT_FILE="${OUT_DIR}/metrics.json"
    fi
    if [[ "${scheduler_log_was_provided}" == "0" ]]; then
      SCHEDULER_LOG_FILE="${OUT_DIR}/global_scheduler_server.log"
    fi
  fi

  echo "[case] ${CASE_NAME}"
  echo "[config] base=${BASE_CONFIG}"
  echo "[config] generated=${GENERATED_CONFIG}"
  echo "[policy] global=${GLOBAL_POLICY:-<inherit>} instance=${INSTANCE_POLICY:-<inherit>} step_chunk=${ENABLE_STEP_CHUNK:-<inherit>} preemption=${ENABLE_CHUNK_PREEMPTION:-<inherit>} chunk_budget=${CHUNK_BUDGET_STEPS:-<inherit>} image_chunk_budget=${IMAGE_CHUNK_BUDGET_STEPS:-<inherit>} video_chunk_budget=${VIDEO_CHUNK_BUDGET_STEPS:-<inherit>} small_latency_threshold_ms=${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-<inherit>}"
  echo "[benchmark_mode] mode=${BENCHMARK_MODE} duration_s=${NUM_PROMPTS_DURATION_SECONDS:-<unset>} fixed_num_prompts=${FIXED_NUM_PROMPTS}"
  echo "[rates] ${REQUEST_RATES}"
  echo "[out_dir] ${OUT_DIR}"

  CONFIG_FILE="${GENERATED_CONFIG}"   REQUEST_RATES="${REQUEST_RATES}"   REQUEST_DURATION_S="${REQUEST_DURATION_S}"   BENCHMARK_MODE="${BENCHMARK_MODE}"   NUM_PROMPTS_DURATION_SECONDS="${NUM_PROMPTS_DURATION_SECONDS}"   FIXED_NUM_PROMPTS="${FIXED_NUM_PROMPTS}"   SCHEDULER_LOG_FILE="${SCHEDULER_LOG_FILE}"   "${RUNNER_SCRIPT}"

  echo "[done] case=${CASE_NAME}"
  echo "[artifacts] ${OUT_DIR}"
}

main "$@"
