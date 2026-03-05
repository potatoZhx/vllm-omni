#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_CONFIG_YAML="${SCRIPT_DIR}/diffusion_rps_eval_config.yaml"

CONFIG_YAML="${DEFAULT_CONFIG_YAML}"
if [[ $# -ge 1 ]]; then
  case "$1" in
    --config-yaml)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config-yaml" >&2
        exit 2
      fi
      CONFIG_YAML="$2"
      shift 2
      ;;
    *)
      CONFIG_YAML="$1"
      shift
      ;;
  esac
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config YAML not found: ${CONFIG_YAML}" >&2
  exit 2
fi

log() {
  echo "[$(date +"%F %T")] $*"
}

write_benchmark_config() {
  local yaml_path="$1"
  eval "$(python3 - "${yaml_path}" <<'PY'
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required to read YAML config. Install with: pip install pyyaml"
    ) from exc


def q(value: str) -> str:
    return shlex.quote(value)


yaml_path = Path(sys.argv[1])
cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

runtime = cfg.get("runtime", {})
run_root = Path(str(runtime.get("run_root", "./benchmark_outputs/global_scheduler")))
run_name_prefix = str(runtime.get("run_name_prefix", "run_"))
timestamp_format = str(runtime.get("timestamp_format", "%Y%m%d-%H%M%S"))
log_dir_name = str(runtime.get("log_dir_name", "logs"))
config_dir_name = str(runtime.get("config_dir_name", "configs"))

run_ts = datetime.now().strftime(timestamp_format)
run_dir = run_root / f"{run_name_prefix}{run_ts}"
log_dir = run_dir / log_dir_name
cfg_dir = run_dir / config_dir_name
log_dir.mkdir(parents=True, exist_ok=True)
cfg_dir.mkdir(parents=True, exist_ok=True)

deployment = cfg.get("deployment", {})
benchmark = cfg.get("benchmark", {})
if not isinstance(deployment, dict) or not isinstance(benchmark, dict):
    raise SystemExit("YAML must contain mapping objects for 'deployment' and 'benchmark'")

benchmark["output_dir"] = str(run_dir)
payload = {"deployment": deployment, "benchmark": benchmark}

bench_cfg_path = cfg_dir / "benchmark_config.json"
bench_cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

hf_cfg = cfg.get("hf", {})
if not isinstance(hf_cfg, dict):
    raise SystemExit("YAML field 'hf' must be a mapping")
hf_endpoint = str(hf_cfg.get("endpoint", "https://hf-mirror.com"))
hf_transfer_mode = str(hf_cfg.get("enable_hf_transfer", "auto")).strip().lower()

benchmark_name = str(benchmark.get("benchmark_name", "unnamed_benchmark"))
model = str(benchmark.get("model", ""))
run_mode = str(benchmark.get("run_mode", "fixed_requests"))
duration_per_rps = str(benchmark.get("total_duration_s", ""))

for key, value in {
    "RUN_DIR": str(run_dir),
    "LOG_DIR": str(log_dir),
    "CFG_DIR": str(cfg_dir),
    "BENCH_CFG": str(bench_cfg_path),
    "HF_ENDPOINT_CFG": hf_endpoint,
    "HF_TRANSFER_MODE_CFG": hf_transfer_mode,
    "BENCHMARK_NAME_CFG": benchmark_name,
    "MODEL_CFG": model,
    "RUN_MODE_CFG": run_mode,
    "DURATION_PER_RPS_CFG": duration_per_rps,
}.items():
    print(f"{key}={q(value)}")
PY
)"
}

main() {
  write_benchmark_config "${CONFIG_YAML}"

  # Hugging Face mirror/source settings. YAML values are defaults and can still
  # be overridden by pre-exported environment variables.
  export HF_ENDPOINT="${HF_ENDPOINT:-${HF_ENDPOINT_CFG}}"

  if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER+x}" ]]; then
    case "${HF_TRANSFER_MODE_CFG}" in
      auto)
        if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
        then
          export HF_HUB_ENABLE_HF_TRANSFER="1"
        else
          export HF_HUB_ENABLE_HF_TRANSFER="0"
        fi
        ;;
      1|true|yes|on)
        export HF_HUB_ENABLE_HF_TRANSFER="1"
        ;;
      0|false|no|off)
        export HF_HUB_ENABLE_HF_TRANSFER="0"
        ;;
      *)
        echo "Invalid hf.enable_hf_transfer in YAML: ${HF_TRANSFER_MODE_CFG}" >&2
        exit 2
        ;;
    esac
  fi

  log "Run dir: ${RUN_DIR}"
  log "Config YAML: ${CONFIG_YAML}"
  log "HF_ENDPOINT: ${HF_ENDPOINT}"
  log "HF_HUB_ENABLE_HF_TRANSFER: ${HF_HUB_ENABLE_HF_TRANSFER}"
  log "Model: ${MODEL_CFG}"
  log "Benchmark: ${BENCHMARK_NAME_CFG}"
  log "Run mode: ${RUN_MODE_CFG}"
  if [[ -n "${DURATION_PER_RPS_CFG}" ]]; then
    log "Duration per RPS: ${DURATION_PER_RPS_CFG}s"
  fi

  log "Launching diffusion_rps_eval.py"
  cd "${REPO_ROOT}"
  python3 tests/global_scheduler/scripts/diffusion_rps_eval.py \
    --config "${BENCH_CFG}" \
    --log-dir "${LOG_DIR}"
}

main "$@"
