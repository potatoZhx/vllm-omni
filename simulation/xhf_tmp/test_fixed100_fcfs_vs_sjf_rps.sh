#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_CONFIG="${SCRIPT_DIR}/config/single_instance_datasetX_trace.yaml"
OUT_ROOT="${SCRIPT_DIR}/out_fixed100_fcfs_vs_sjf_rps"

REQUEST_RPS_LIST=(0.2)
INSTANCE_POLICIES=(sjf)

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "[error] BASE_CONFIG not found: ${BASE_CONFIG}" >&2
  exit 1
fi

run_one() {
  local instance_policy="$1"
  local rps="$2"

  local out_dir="${OUT_ROOT}/${instance_policy}/rps_${rps}"
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  echo
  echo "=== Run: instance_policy=${instance_policy} rps=${rps} ==="
  BASE_CONFIG="${BASE_CONFIG}" \
    INSTANCE_POLICY="${instance_policy}" \
    BENCHMARK_MODE="fixed_num_prompts" \
    FIXED_NUM_PROMPTS="100" \
    REQUEST_RATES="${rps}" \
    OUT_DIR="${out_dir}" \
    benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh

  local worker_log="${out_dir}/instance_logs/worker0.log"
  local metrics_json="${out_dir}/metrics.json"

  if [ ! -f "${worker_log}" ]; then
    echo "[warn] worker log not found: ${worker_log}" >&2
    return 0
  fi
  if [ ! -f "${metrics_json}" ]; then
    echo "[warn] metrics json not found: ${metrics_json}" >&2
    return 0
  fi

  # warmup_requests 来自 YAML（warmup_keep）：8
  local expected_measure=100
  local expected_total=$((expected_measure + 8))

  # 注意：completion_ts 字段只会在 REQUEST_COMPLETED 事件里出现（stage1_scheduler 内部日志）
  local completed_with_ts=0
  if command -v rg >/dev/null 2>&1; then
    completed_with_ts="$(rg -c "REQUEST_COMPLETED .*completion_ts=[0-9]" "${worker_log}" || true)"
  else
    # 兼容环境里没装 ripgrep 的情况
    completed_with_ts="$(grep -E -c "REQUEST_COMPLETED .*completion_ts=[0-9]" "${worker_log}" || true)"
  fi

  local completed_requests
  completed_requests="$(
    METRICS_JSON="${metrics_json}" python3 - <<'PY'
import json
import os
from pathlib import Path
metrics_path = Path(os.environ["METRICS_JSON"])
data = json.loads(metrics_path.read_text(encoding="utf-8"))
print(int(data.get("completed_requests", 0)))
PY
  )"

  echo "[check] metrics.completed_requests=${completed_requests} (expected_measure=${expected_measure})"
  echo "[check] log REQUEST_COMPLETED completion_ts lines=${completed_with_ts} (expected_total=${expected_total})"

  if [ "${completed_requests}" -ne "${expected_measure}" ]; then
    echo "[warn] metrics completed_requests != 100, see ${metrics_json}" >&2
  fi

  if [ "${completed_with_ts}" -lt "${expected_measure}" ]; then
    echo "[warn] completion_ts printed lines < 100 (expected at least measurement requests)" >&2
    tail -n 30 "${worker_log}" || true
  fi
}

mkdir -p "${OUT_ROOT}"

for instance_policy in "${INSTANCE_POLICIES[@]}"; do
  for rps in "${REQUEST_RPS_LIST[@]}"; do
    run_one "${instance_policy}" "${rps}"
  done
done

echo
echo "[done] outputs in: ${OUT_ROOT}"

