#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/switch_bench_cases.yaml"
CONFIG_PATH="${DEFAULT_CONFIG}"
MODEL_OVERRIDE=""

print_usage() {
  echo "Usage: $0 [config_path] [model_path]"
  echo "   or: $0 --config <config_path> --model <model_path>"
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      if [[ "${CONFIG_PATH}" == "${DEFAULT_CONFIG}" ]]; then
        CONFIG_PATH="$1"
      elif [[ -z "${MODEL_OVERRIDE}" ]]; then
        MODEL_OVERRIDE="$1"
      else
        echo "Unknown argument: $1" >&2
        print_usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python is required to parse yaml config." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for readiness checks." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH." >&2
  exit 1
fi

if ! python -c "import yaml" >/dev/null 2>&1; then
  echo "PyYAML is required. Install with: pip install pyyaml" >&2
  exit 1
fi

readarray -t GLOBAL_VARS < <(
  python - "${CONFIG_PATH}" "${MODEL_OVERRIDE}" <<'PY'
import json
import shlex
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
g = cfg.get("global", {})
model_override = sys.argv[2] if len(sys.argv) > 2 else ""
model = model_override if model_override else str(g.get("model", "Qwen/Qwen-Image"))
print(f"MODEL={shlex.quote(model)}")
print(f"HOST={shlex.quote(str(g.get('host', '127.0.0.1')))}")
print(f"PORT={int(g.get('port', 8091))}")
print(f"READY_PATH={shlex.quote(str(g.get('ready_path', '/v1/models')))}")
print(f"READY_TIMEOUT_SEC={int(g.get('ready_timeout_sec', 600))}")
print(f"POLL_INTERVAL_SEC={int(g.get('poll_interval_sec', 2))}")
print(f"MEASURED_RUNS={int(g.get('measured_runs', 20))}")
print(f"STOP_TIMEOUT_SEC={int(g.get('stop_timeout_sec', 60))}")
print(f"MAX_GPU_PER_CASE={int(g.get('max_gpu_per_case', 4))}")
print(f"LOG_ROOT={shlex.quote(str(g.get('log_root', './image_switch_results')))}")
print("EXTRA_CLI_ARGS_JSON=" + shlex.quote(json.dumps(g.get("extra_cli_args", []))))
PY
)

for kv in "${GLOBAL_VARS[@]}"; do
  eval "${kv}"
done

if [[ -n "${MODEL_OVERRIDE}" ]]; then
  echo "Using model override: ${MODEL}"
fi

RUN_ROOT="${SCRIPT_DIR}/${LOG_ROOT}"
mkdir -p "${RUN_ROOT}"

SUMMARY_CSV="${RUN_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "case_id,case_name,gpu,sp,cfg,tp,vae_use_slicing,vae_use_tiling,first_startup_s,stop_mean_s,stop_std_s,startup_mean_s,startup_std_s,switch_mean_s,switch_std_s,status,note" > "${SUMMARY_CSV}"
fi

cleanup_server() {
  local pid="$1"
  local timeout_sec="$2"
  local start_ns
  start_ns="$(date +%s%N)"

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -TERM "${pid}" >/dev/null 2>&1 || true
    local elapsed=0
    while kill -0 "${pid}" >/dev/null 2>&1; do
      sleep 1
      elapsed=$((elapsed + 1))
      if (( elapsed >= timeout_sec )); then
        kill -KILL "${pid}" >/dev/null 2>&1 || true
        break
      fi
    done
  fi

  local end_ns
  end_ns="$(date +%s%N)"
  python - <<PY
start_ns=${start_ns}
end_ns=${end_ns}
print((end_ns - start_ns) / 1e9)
PY
}

wait_ready() {
  local url="$1"
  local timeout_sec="$2"
  local interval_sec="$3"
  local start_ns
  start_ns="$(date +%s%N)"
  local waited=0

  while true; do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' "${url}" || true)"
    if [[ "${code}" == "200" ]]; then
      local end_ns
      end_ns="$(date +%s%N)"
      python - <<PY
start_ns=${start_ns}
end_ns=${end_ns}
print((end_ns - start_ns) / 1e9)
PY
      return 0
    fi

    echo "  waiting ready: ${url} (http=${code:-NA}, waited=${waited}s/${timeout_sec}s)"

    sleep "${interval_sec}"
    waited=$((waited + interval_sec))
    if (( waited >= timeout_sec )); then
      return 1
    fi
  done
}

build_cmd_line() {
  local model="$1"
  local port="$2"
  local host="$3"
  local log_file="$4"
  local sp="$5"
  local cfg="$6"
  local tp="$7"
  local vae_use_slicing="$8"
  local vae_use_tiling="$9"
  local extra_cli_json="${10}"

  local cmd=(vllm serve --model "${model}" --omni --port "${port}" --host "${host}" --log-file "${log_file}" --ulysses-degree "${sp}" --cfg-parallel-size "${cfg}" --tensor-parallel-size "${tp}")

  if [[ "${vae_use_slicing}" == "true" ]]; then
    cmd+=(--vae-use-slicing)
  fi
  if [[ "${vae_use_tiling}" == "true" ]]; then
    cmd+=(--vae-use-tiling)
  fi

  mapfile -t extra_args < <(
    python - "${extra_cli_json}" <<'PY'
import json
import shlex
import sys
for item in json.loads(sys.argv[1]):
    print(shlex.quote(str(item)))
PY
  )

  if (( ${#extra_args[@]} > 0 )); then
    cmd+=("${extra_args[@]}")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

start_server() {
  local cmd_line="$1"
  local run_log="$2"
  local clear_log="$3"

  if [[ "${clear_log}" == "true" ]]; then
    : > "${run_log}"
  fi

  # Avoid process substitution here: this function is called via command
  # substitution and tee/process-substitution can block PID capture.
  eval "${cmd_line}" >> "${run_log}" 2>&1 &
  echo "$!"
}

calc_stats() {
  local csv_path="$1"
  python - "${csv_path}" <<'PY'
import csv
import math
import statistics
import sys

stop_vals = []
startup_vals = []
switch_vals = []
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stop_vals.append(float(row["stop_s"]))
        startup_vals.append(float(row["startup_s"]))
        switch_vals.append(float(row["switch_s"]))

def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)

s_mu, s_std = mean_std(stop_vals)
st_mu, st_std = mean_std(startup_vals)
sw_mu, sw_std = mean_std(switch_vals)
print(f"{s_mu},{s_std},{st_mu},{st_std},{sw_mu},{sw_std}")
PY
}

readarray -t CASE_ROWS < <(
  python - "${CONFIG_PATH}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
for case in cfg.get("cases", []):
    print("\t".join([
        str(case.get("id", "")),
        str(case.get("name", "")),
        str(bool(case.get("enabled", True))).lower(),
        str(int(case.get("sp", 1))),
        str(int(case.get("cfg", 1))),
        str(int(case.get("tp", 1))),
        str(bool(case.get("vae_use_slicing", False))).lower(),
        str(bool(case.get("vae_use_tiling", False))).lower(),
    ]))
PY
)

for row in "${CASE_ROWS[@]}"; do
  IFS=$'\t' read -r case_id case_name enabled sp cfg tp vae_use_slicing vae_use_tiling <<<"${row}"

  if [[ "${enabled}" != "true" ]]; then
    continue
  fi

  gpu_count=$((sp * cfg * tp))
  if (( gpu_count > MAX_GPU_PER_CASE )); then
    echo "[SKIP] case ${case_id} (${case_name}) requires ${gpu_count} GPUs (> ${MAX_GPU_PER_CASE})"
    echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},,,,,,,,SKIP,requires>${MAX_GPU_PER_CASE}gpus" >> "${SUMMARY_CSV}"
    continue
  fi

  case_dir="${RUN_ROOT}/case_${case_id}_${case_name}"
  mkdir -p "${case_dir}"
  run_csv="${case_dir}/runs.csv"
  run_log="${case_dir}/service.log"

  echo "run_idx,stop_s,startup_s,switch_s" > "${run_csv}"

  cmd_line="$(build_cmd_line "${MODEL}" "${PORT}" "${HOST}" "${run_log}" "${sp}" "${cfg}" "${tp}" "${vae_use_slicing}" "${vae_use_tiling}" "${EXTRA_CLI_ARGS_JSON}")"
  ready_url="http://${HOST}:${PORT}${READY_PATH}"

  echo "[CASE ${case_id}] ${case_name}"
  echo "  command: ${cmd_line}"
  echo "  ready check: ${ready_url}"
  echo "  service log: ${run_log}"

  server_pid="$(start_server "${cmd_line}" "${run_log}" "true")"

  if ! first_startup_s="$(wait_ready "${ready_url}" "${READY_TIMEOUT_SEC}" "${POLL_INTERVAL_SEC}")"; then
    echo "[FAIL] case ${case_id} first startup did not become ready in ${READY_TIMEOUT_SEC}s"
    cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}" >/dev/null || true
    echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},,,,,,,,FAIL,first-startup-timeout" >> "${SUMMARY_CSV}"
    continue
  fi

  for ((run_idx = 1; run_idx <= MEASURED_RUNS; run_idx++)); do
    stop_s="$(cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}")"

    server_pid="$(start_server "${cmd_line}" "${run_log}" "false")"

    if ! startup_s="$(wait_ready "${ready_url}" "${READY_TIMEOUT_SEC}" "${POLL_INTERVAL_SEC}")"; then
      echo "[FAIL] case ${case_id} run ${run_idx}: startup timeout"
      cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}" >/dev/null || true
      echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},,,,,,,,FAIL,run-${run_idx}-startup-timeout" >> "${SUMMARY_CSV}"
      break
    fi

    switch_s="$(python - <<PY
stop_s=${stop_s}
startup_s=${startup_s}
print(stop_s + startup_s)
PY
)"

    echo "${run_idx},${stop_s},${startup_s},${switch_s}" >> "${run_csv}"
    echo "  run=${run_idx}/${MEASURED_RUNS} stop=${stop_s}s startup=${startup_s}s switch=${switch_s}s"

    if (( run_idx == MEASURED_RUNS )); then
      stats="$(calc_stats "${run_csv}")"
      IFS=',' read -r stop_mu stop_std startup_mu startup_std switch_mu switch_std <<<"${stats}"
      echo "${case_id},${case_name},${gpu_count},${sp},${cfg},${tp},${vae_use_slicing},${vae_use_tiling},${first_startup_s},${stop_mu},${stop_std},${startup_mu},${startup_std},${switch_mu},${switch_std},PASS," >> "${SUMMARY_CSV}"
      echo "[PASS] case ${case_id} first_startup=${first_startup_s}s switch_mean=${switch_mu}s"
      cleanup_server "${server_pid}" "${STOP_TIMEOUT_SEC}" >/dev/null || true
    fi
  done

done

echo "Done. summary: ${SUMMARY_CSV}"
