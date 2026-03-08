#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/resolution_steps_bench_cases.yaml"
PY_RUNNER="${SCRIPT_DIR}/run_resolution_steps_bench.py"
CONFIG_PATH="${DEFAULT_CONFIG}"
MODEL_OVERRIDE=""

print_usage() {
  echo "Usage: $0 [config_path] [model_path]"
  echo "   or: $0 --config <config_path> --model <model_path>"
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      if (( $# < 2 )); then
        echo "--config requires a value" >&2
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model)
      if (( $# < 2 )); then
        echo "--model requires a value" >&2
        exit 1
      fi
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

if [[ ! -f "${PY_RUNNER}" ]]; then
  echo "Python runner not found: ${PY_RUNNER}" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH." >&2
  exit 1
fi

if ! python3 -c "import yaml" >/dev/null 2>&1; then
  echo "PyYAML is required. Install with: pip install pyyaml" >&2
  exit 1
fi

CMD=(python3 "${PY_RUNNER}" --config "${CONFIG_PATH}")
if [[ -n "${MODEL_OVERRIDE}" ]]; then
  CMD+=(--model "${MODEL_OVERRIDE}")
fi

exec "${CMD[@]}"
