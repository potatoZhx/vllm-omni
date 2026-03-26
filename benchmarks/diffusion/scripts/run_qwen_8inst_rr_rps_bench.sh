#!/bin/bash
# Multi-instance benchmark for Qwen-Image:
# - Start 8 single-GPU vLLM-Omni instances.
# - Start one local round-robin proxy port.
# - Run diffusion benchmark by RPS list.
# - Support both fixed_duration and fixed_num_prompts modes.

set -euo pipefail
export PYTHONUNBUFFERED=1

# ========================= Config (edit as needed) =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RESULTS_ROOT_DIR="${REPO_ROOT}/benchmarks/diffusion/results"

MODEL="${MODEL:-/data2/group_谈海生/mumura/models/Qwen-Image}"

# Instance and proxy ports.
NUM_INSTANCES="${NUM_INSTANCES:-8}"
BASE_PORT="${BASE_PORT:-8099}"
RR_PORT="${RR_PORT:-8091}"
BASE_URL="http://localhost:${RR_PORT}"

# Device settings.
# DEVICE_TYPE supports: gpu / npu
DEVICE_TYPE="${DEVICE_TYPE:-gpu}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"

# Benchmark settings.
TASK="${TASK:-t2i}"
DATASET="${DATASET:-random}"
DATASET_TYPE="${DATASET_TYPE:-C}"
BACKEND="${BACKEND:-vllm-omni}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-200}"
RPS_LIST="${RPS_LIST:-[0.2]}" # , 0.4, 0.6, 0.8, 1

# BENCHMARK_MODE supports: fixed_duration / fixed_num_prompts
BENCHMARK_MODE="${BENCHMARK_MODE:-fixed_duration}"
NUM_PROMPTS_DURATION_SECONDS="${NUM_PROMPTS_DURATION_SECONDS:-1800}"
FIXED_NUM_PROMPTS="${FIXED_NUM_PROMPTS:-100}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
WARMUP_NUM_INFERENCE_STEPS="${WARMUP_NUM_INFERENCE_STEPS:-1}"
# ===========================================================================

mkdir -p "$RESULTS_ROOT_DIR"
ts=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT_DIR}/qwen_8inst_rr_rps_${ts}"
mkdir -p "$RUN_DIR"

MASTER_LOG="${RUN_DIR}/benchmark_master.log"
PROXY_LOG="${RUN_DIR}/round_robin_proxy.log"
DEVICE_LOG="${RUN_DIR}/${DEVICE_TYPE}.log"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

echo "===== Run dir: ${RUN_DIR} =====" | tee -a "$MASTER_LOG"
echo "===== Model: ${MODEL} =====" | tee -a "$MASTER_LOG"
echo "===== Num instances: ${NUM_INSTANCES} =====" | tee -a "$MASTER_LOG"
echo "===== Instance base port: ${BASE_PORT} =====" | tee -a "$MASTER_LOG"
echo "===== Round-robin port: ${RR_PORT} =====" | tee -a "$MASTER_LOG"
echo "===== RPS list: ${RPS_LIST} =====" | tee -a "$MASTER_LOG"
echo "===== Benchmark mode: ${BENCHMARK_MODE} =====" | tee -a "$MASTER_LOG"
echo "===== Duration(seconds): ${NUM_PROMPTS_DURATION_SECONDS} =====" | tee -a "$MASTER_LOG"
echo "===== Fixed num-prompts: ${FIXED_NUM_PROMPTS} =====" | tee -a "$MASTER_LOG"
echo "===== Warmup requests: ${WARMUP_REQUESTS} =====" | tee -a "$MASTER_LOG"
echo "===== Warmup num_inference_steps: ${WARMUP_NUM_INFERENCE_STEPS} =====" | tee -a "$MASTER_LOG"
echo "===== Dataset: ${DATASET} =====" | tee -a "$MASTER_LOG"
echo "===== DatasetType: ${DATASET_TYPE} =====" | tee -a "$MASTER_LOG"
echo "===== Task: ${TASK} =====" | tee -a "$MASTER_LOG"
echo "===== Device type: ${DEVICE_TYPE} =====" | tee -a "$MASTER_LOG"
echo "===== Backend: ${BACKEND} =====" | tee -a "$MASTER_LOG"
echo "===== Repo root: ${REPO_ROOT} =====" | tee -a "$MASTER_LOG"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH" | tee -a "$MASTER_LOG"
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH" | tee -a "$MASTER_LOG"
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found in PATH" | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$DEVICE_TYPE" != "gpu" ] && [ "$DEVICE_TYPE" != "npu" ]; then
  echo "Unsupported DEVICE_TYPE=${DEVICE_TYPE}, expected gpu or npu" | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$TASK" != "t2i" ]; then
  echo "Unsupported TASK=${TASK}. This script is for Qwen-Image t2i." | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$DATASET" != "random" ]; then
  echo "Unsupported DATASET=${DATASET}. This script uses random dataset." | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$BENCHMARK_MODE" != "fixed_duration" ] && [ "$BENCHMARK_MODE" != "fixed_num_prompts" ]; then
  echo "Unsupported BENCHMARK_MODE=${BENCHMARK_MODE}, expected fixed_duration or fixed_num_prompts" | tee -a "$MASTER_LOG"
  exit 1
fi
if ! [[ "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [ "$NUM_INSTANCES" -lt 1 ]; then
  echo "NUM_INSTANCES must be a positive integer, got: ${NUM_INSTANCES}" | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$BENCHMARK_MODE" = "fixed_num_prompts" ]; then
  if ! [[ "$FIXED_NUM_PROMPTS" =~ ^[0-9]+$ ]] || [ "$FIXED_NUM_PROMPTS" -lt 1 ]; then
    echo "FIXED_NUM_PROMPTS must be a positive integer, got: ${FIXED_NUM_PROMPTS}" | tee -a "$MASTER_LOG"
    exit 1
  fi
fi
if ! [[ "$WARMUP_REQUESTS" =~ ^[0-9]+$ ]]; then
  echo "WARMUP_REQUESTS must be a non-negative integer, got: ${WARMUP_REQUESTS}" | tee -a "$MASTER_LOG"
  exit 1
fi
if ! [[ "$WARMUP_NUM_INFERENCE_STEPS" =~ ^[0-9]+$ ]] || [ "$WARMUP_NUM_INFERENCE_STEPS" -lt 1 ]; then
  echo "WARMUP_NUM_INFERENCE_STEPS must be a positive integer, got: ${WARMUP_NUM_INFERENCE_STEPS}" | tee -a "$MASTER_LOG"
  exit 1
fi

build_random_request_config() {
  case "$1" in
    A|a)
      cat <<'JSON'
[{"width":512,"height":512,"num_inference_steps":20,"weight":1}]
JSON
      ;;
    B|b)
      cat <<'JSON'
[{"width":1536,"height":1536,"num_inference_steps":35,"weight":1}]
JSON
      ;;
    C|c)
      cat <<'JSON'
[
  {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
  {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
  {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
  {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
]
JSON
      ;;
    *)
      echo "Unsupported DATASET_TYPE=$1, expected A/B/C" >&2
      return 1
      ;;
  esac
}

RANDOM_REQUEST_CONFIG=$(build_random_request_config "$DATASET_TYPE")

normalize_rps_list() {
  local raw="$1"
  # Supports both: "[0.1, 0.5, 1]" and "0.1,0.5,1".
  echo "$raw" | sed 's/\[//g; s/\]//g; s/,/ /g'
}

to_num_prompts() {
  local _rps="$1"
  local _duration="$2"
  python3 - <<PY
from decimal import Decimal, ROUND_HALF_UP
rps = Decimal(${_rps@Q})
duration = Decimal(${_duration@Q})
value = (rps * duration).to_integral_value(rounding=ROUND_HALF_UP)
print(max(1, int(value)))
PY
}

RPS_ITEMS=$(normalize_rps_list "$RPS_LIST")
if [ -z "$RPS_ITEMS" ]; then
  echo "RPS_LIST is empty after parsing: ${RPS_LIST}" | tee -a "$MASTER_LOG"
  exit 1
fi

cd "$REPO_ROOT"

SERVER_PIDS=()
DEVICE_MONITOR_PID=""
RR_PROXY_PID=""

cleanup() {
  if [ -n "${DEVICE_MONITOR_PID:-}" ]; then
    kill "$DEVICE_MONITOR_PID" 2>/dev/null || true
  fi
  if [ -n "${RR_PROXY_PID:-}" ]; then
    kill "$RR_PROXY_PID" 2>/dev/null || true
  fi
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

INSTANCE_BASE_URLS=()

echo "===== Starting ${NUM_INSTANCES} single-device vLLM-Omni instances =====" | tee -a "$MASTER_LOG"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((BASE_PORT + i))
  server_log="${RUN_DIR}/server_${i}.log"
  instance_master_port=$((MASTER_PORT_BASE + i))
  INSTANCE_BASE_URLS+=("http://localhost:${port}")

  if [ "$DEVICE_TYPE" = "gpu" ]; then
    (
      export CUDA_VISIBLE_DEVICES="$i"
      export MASTER_ADDR="$MASTER_ADDR"
      export MASTER_PORT="$instance_master_port"
      vllm serve "$MODEL" \
        --omni \
        --port "$port" \
        --ulysses-degree 1 \
        --cfg-parallel-size 1 \
        --num-weight-load-threads 8 \
        --vae-use-slicing \
        --vae-use-tiling \
        >> "$server_log" 2>&1
    ) &
  else
    (
      export ASCEND_RT_VISIBLE_DEVICES="$i"
      export MASTER_ADDR="$MASTER_ADDR"
      export MASTER_PORT="$instance_master_port"
      vllm serve "$MODEL" \
        --omni \
        --port "$port" \
        --ulysses-degree 1 \
        --cfg-parallel-size 1 \
        --num-weight-load-threads 8 \
        --vae-use-slicing \
        --vae-use-tiling \
        >> "$server_log" 2>&1
    ) &
  fi

  pid=$!
  SERVER_PIDS+=("$pid")
  echo "instance=${i}, port=${port}, pid=${pid}, log=${server_log}" | tee -a "$MASTER_LOG"
done

echo "===== Waiting for all instances ready (max 7200s each) =====" | tee -a "$MASTER_LOG"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((BASE_PORT + i))
  url="http://localhost:${port}"
  echo "waiting for instance ${i} at ${url}" | tee -a "$MASTER_LOG"
  for t in $(seq 1 7200); do
    if curl -s -o /dev/null -w "%{http_code}" "${url}/v1/models" 2>/dev/null | grep -q 200; then
      echo "instance ${i} ready after ${t}s" | tee -a "$MASTER_LOG"
      break
    fi
    if [ "$t" -eq 7200 ]; then
      echo "instance ${i} did not become ready in time. check ${RUN_DIR}/server_${i}.log" | tee -a "$MASTER_LOG"
      exit 1
    fi
    if [ $((t % 60)) -eq 0 ] && [ "$t" -gt 0 ]; then
      echo "still waiting for instance ${i} (${t}s)" | tee -a "$MASTER_LOG"
    fi
    sleep 1
  done
done

echo "===== Starting local round-robin proxy on port ${RR_PORT} =====" | tee -a "$MASTER_LOG"
python3 - "$RR_PORT" "${INSTANCE_BASE_URLS[@]}" >> "$PROXY_LOG" 2>&1 <<'PY' &
import itertools
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if len(sys.argv) < 3:
    raise SystemExit("usage: rr_proxy.py <listen_port> <upstream1> <upstream2> ...")

listen_port = int(sys.argv[1])
upstreams = sys.argv[2:]

_rr_cycle = itertools.cycle(upstreams)
_rr_lock = threading.Lock()


def pick_upstream() -> str:
    with _rr_lock:
        return next(_rr_cycle)


class RRProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def do_PUT(self):
        self._forward()

    def do_PATCH(self):
        self._forward()

    def do_DELETE(self):
        self._forward()

    def log_message(self, fmt, *args):
        # Keep default noisy access logs disabled.
        return

    def _forward(self):
        target_base = pick_upstream()
        target_url = f"{target_base}{self.path}"

        body = b""
        content_length = self.headers.get("Content-Length")
        if content_length:
            body = self.rfile.read(int(content_length))

        req = urllib.request.Request(
            target_url,
            data=body if self.command in {"POST", "PUT", "PATCH"} else None,
            method=self.command,
        )

        for k, v in self.headers.items():
            lk = k.lower()
            if lk in {
                "host",
                "content-length",
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "upgrade",
            }:
                continue
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for hk, hv in resp.getheaders():
                    lhk = hk.lower()
                    if lhk in {
                        "transfer-encoding",
                        "connection",
                        "keep-alive",
                        "proxy-authenticate",
                        "proxy-authorization",
                        "te",
                        "trailers",
                        "upgrade",
                        "content-length",
                    }:
                        continue
                    self.send_header(hk, hv)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                if resp_body:
                    self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            for hk, hv in e.headers.items():
                lhk = hk.lower()
                if lhk in {
                    "transfer-encoding",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                    "content-length",
                }:
                    continue
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            if err_body:
                self.wfile.write(err_body)
        except Exception as e:  # noqa: BLE001
            msg = f"round-robin proxy upstream error: {e}\n".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", listen_port), RRProxyHandler)
    print(f"Round-robin proxy started on :{listen_port}")
    print(f"Upstreams: {upstreams}")
    server.serve_forever()


if __name__ == "__main__":
    main()
PY
RR_PROXY_PID=$!

echo "rr proxy pid=${RR_PROXY_PID}, log=${PROXY_LOG}" | tee -a "$MASTER_LOG"

echo "Waiting for round-robin endpoint ready at ${BASE_URL} (max 120s)..." | tee -a "$MASTER_LOG"
for t in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models" 2>/dev/null | grep -q 200; then
    echo "round-robin endpoint ready after ${t}s" | tee -a "$MASTER_LOG"
    break
  fi
  if [ "$t" -eq 120 ]; then
    echo "round-robin endpoint did not become ready in time. check ${PROXY_LOG}" | tee -a "$MASTER_LOG"
    exit 1
  fi
  sleep 1
done

MONITOR_CMD="nvidia-smi"
if [ "$DEVICE_TYPE" = "npu" ]; then
  MONITOR_CMD="npu-smi info"
fi
(
  while true; do
    echo "=== $(date -Iseconds) ==="
    bash -lc "$MONITOR_CMD" || true
    sleep 5
  done
) >> "$DEVICE_LOG" 2>&1 &
DEVICE_MONITOR_PID=$!

printf '%s\n' "timestamp,mode,rps,duration_seconds,num_prompts,metrics_json,log_file" > "$SUMMARY_CSV"

echo "===== Running benchmark by RPS via round-robin port ${RR_PORT} =====" | tee -a "$MASTER_LOG"
for rps in $RPS_ITEMS; do
  if [ "$BENCHMARK_MODE" = "fixed_duration" ]; then
    num_prompts=$(to_num_prompts "$rps" "$NUM_PROMPTS_DURATION_SECONDS")
    run_duration_seconds="$NUM_PROMPTS_DURATION_SECONDS"
  else
    num_prompts="$FIXED_NUM_PROMPTS"
    run_duration_seconds=$(python3 - <<PY
from decimal import Decimal
rps = Decimal(${rps@Q})
num_prompts = Decimal(${num_prompts@Q})
print(f"{(num_prompts / rps):f}")
PY
)
  fi

  rps_label=${rps//./_}
  METRICS_FILE="${RUN_DIR}/metrics_rps_${rps_label}.json"
  RUN_LOG="${RUN_DIR}/bench_rps_${rps_label}.log"

  echo "[mode=${BENCHMARK_MODE}] [RPS=${rps}] num-prompts=${num_prompts}, duration=${run_duration_seconds}s" | tee -a "$MASTER_LOG"

  python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --task "$TASK" \
    --dataset "$DATASET" \
    --num-prompts "$num_prompts" \
    --request-rate "$rps" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --warmup-num-inference-steps "$WARMUP_NUM_INFERENCE_STEPS" \
    --enable-negative-prompt \
    --random-request-config "$RANDOM_REQUEST_CONFIG" \
    --output-file "$METRICS_FILE" \
    2>&1 | tee "$RUN_LOG"

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$BENCHMARK_MODE" "$rps" "$run_duration_seconds" "$num_prompts" "$METRICS_FILE" "$RUN_LOG" \
    >> "$SUMMARY_CSV"
done

echo "===== Done =====" | tee -a "$MASTER_LOG"
echo "summary: ${SUMMARY_CSV}" | tee -a "$MASTER_LOG"
echo "master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "proxy log: ${PROXY_LOG}" | tee -a "$MASTER_LOG"
echo "device log: ${DEVICE_LOG}" | tee -a "$MASTER_LOG"

exit 0
