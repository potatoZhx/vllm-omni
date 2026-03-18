#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/benchmarks/diffusion/scripts"
RUN_ONE_SCRIPT="${SCRIPT_DIR}/run_trace_scheduler_bench.sh"

MODEL="${MODEL:-Qwen/Qwen-Image}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8091}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/benchmarks/dataset/sd3_trace_redistributed.txt}"
SLO_SCALE="${SLO_SCALE:-3}"
CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS:-4}"
SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS:-}"
INSTANCE_SCHEDULER_AGING_FACTOR="${INSTANCE_SCHEDULER_AGING_FACTOR:-1.0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/scheduler_preemption_compare_${RUN_TAG}}"

mkdir -p "${OUT_DIR}"

declare -a CASES=(
  "fcfs_baseline fcfs 0 0 0"
  "sjf_baseline sjf 0 0 1"
  "sjf_preempt sjf 1 1 1"
  "slo_first_baseline slo_first 0 0 1"
  "slo_first_preempt slo_first 1 1 1"
  "slack_age_baseline slack_age 0 0 1"
  "slack_age_preempt slack_age 1 1 1"
  "slack_cost_age_baseline slack_cost_age 0 0 1"
  "slack_cost_age_preempt slack_cost_age 1 1 1"
)

run_case() {
  local case_name="$1"
  local policy="$2"
  local enable_step_chunk="$3"
  local enable_chunk_preemption="$4"
  local inject_slo="$5"
  local port="$6"
  local case_out_dir="${OUT_DIR}/${case_name}"

  mkdir -p "${case_out_dir}"

  echo
  echo "=== Running ${case_name} ==="
  echo "policy=${policy} step_chunk=${enable_step_chunk} preemption=${enable_chunk_preemption} inject_slo=${inject_slo}"

  env \
    MODEL="${MODEL}" \
    HOST="${HOST}" \
    PORT="${port}" \
    NUM_GPUS="${NUM_GPUS}" \
    POLICY="${policy}" \
    ENABLE_STEP_CHUNK="${enable_step_chunk}" \
    ENABLE_CHUNK_PREEMPTION="${enable_chunk_preemption}" \
    CHUNK_BUDGET_STEPS="${CHUNK_BUDGET_STEPS}" \
    SMALL_REQUEST_LATENCY_THRESHOLD_MS="${SMALL_REQUEST_LATENCY_THRESHOLD_MS}" \
    INSTANCE_SCHEDULER_AGING_FACTOR="${INSTANCE_SCHEDULER_AGING_FACTOR}" \
    NUM_PROMPTS="${NUM_PROMPTS}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY}" \
    REQUEST_RATE="${REQUEST_RATE}" \
    WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
    DATASET_PATH="${DATASET_PATH}" \
    INJECT_SCHEDULER_SLO="${inject_slo}" \
    SLO_SCALE="${SLO_SCALE}" \
    STREAM_SERVER_LOGS="${STREAM_SERVER_LOGS:-1}" \
    OUT_DIR="${case_out_dir}" \
    "${RUN_ONE_SCRIPT}"
}

python_summary() {
  python - "$OUT_DIR" <<'PY'
import json
import pathlib
import re
import string
import sys

out_dir = pathlib.Path(sys.argv[1])
cases = [
    ("fcfs_baseline", "fcfs", "fcfs"),
    ("sjf_baseline", "sjf", "sjf"),
    ("sjf_preempt", "sjf+preempt", "sjf"),
    ("slo_first_baseline", "slo_first", "slo_first"),
    ("slo_first_preempt", "slo_first+preempt", "slo_first"),
    ("slack_age_baseline", "slack_age", "slack_age"),
    ("slack_age_preempt", "slack_age+preempt", "slack_age"),
    ("slack_cost_age_baseline", "slack_cost_age", "slack_cost_age"),
    ("slack_cost_age_preempt", "slack_cost_age+preempt", "slack_cost_age"),
]

def parse_sequence(log_path: pathlib.Path):
    pattern = re.compile(r"QUEUE_DEQUEUE request_id=([^ ]+)")
    ids = []
    if not log_path.exists():
        return ids
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            ids.append(m.group(1))
    return ids

def map_sequence(ids):
    alphabet = list(string.ascii_uppercase)
    mapping = {}
    mapped = []
    next_idx = 0
    for req_id in ids:
        if req_id not in mapping:
            mapping[req_id] = alphabet[next_idx] if next_idx < len(alphabet) else f"R{next_idx+1}"
            next_idx += 1
        mapped.append(mapping[req_id])
    return mapping, mapped

def find_preemption_witness(mapped):
    last_seen = {}
    for idx, label in enumerate(mapped):
        if label in last_seen and idx - last_seen[label] > 1:
            start = last_seen[label]
            return "->".join(mapped[start : idx + 1])
        last_seen[label] = idx
    return None

summary = []
for case_name, display_name, policy_name in cases:
    result_path = out_dir / case_name / f"result_{policy_name}.json"
    server_log = out_dir / case_name / f"server_{policy_name}.log"

    metrics = json.loads(result_path.read_text()) if result_path.exists() else {}
    ids = parse_sequence(server_log)
    _mapping, mapped = map_sequence(ids)
    witness = find_preemption_witness(mapped)
    summary.append(
        {
            "case": display_name,
            "latency_p95_s": metrics.get("latency_p95"),
            "throughput_qps": metrics.get("throughput_qps"),
            "preemption_detected": witness is not None,
            "execution_sequence_sample": "->".join(mapped[:12]),
            "preemption_witness": witness,
            "result_json": str(result_path),
            "server_log": str(server_log),
        }
    )

summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print()
print("=== Scheduler Preemption Comparison ===")
print(f"{'case':<20} {'p95(s)':<12} {'qps':<12} {'preempted':<12} sequence")
for row in summary:
    p95 = f"{row['latency_p95_s']:.4f}" if isinstance(row["latency_p95_s"], (int, float)) else "n/a"
    qps = f"{row['throughput_qps']:.2f}" if isinstance(row["throughput_qps"], (int, float)) else "n/a"
    sequence = row["preemption_witness"] or row["execution_sequence_sample"] or "n/a"
    print(f"{row['case']:<20} {p95:<12} {qps:<12} {str(row['preemption_detected']):<12} {sequence}")
print()
print(f"summary_json: {summary_path}")
PY
}

port="${BASE_PORT}"
for case_row in "${CASES[@]}"; do
  # shellcheck disable=SC2086
  run_case ${case_row} "${port}"
  port=$((port + 1))
done

python_summary
