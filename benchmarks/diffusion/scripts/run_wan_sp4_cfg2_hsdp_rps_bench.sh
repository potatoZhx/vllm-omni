#!/bin/bash
# 单实例实验：按 sp4_cfg2_hsdp 并行配置启动 Wan 服务，待服务就绪后，
# 按给定 RPS 列表依次压测；每个 RPS 的请求总发送时长一致（duration_seconds），
# 并令 num-prompts = rps * duration_seconds。
# 本脚本为普通 bash 脚本，直接运行：bash run_wan_sp4_cfg2_hsdp_rps_bench.sh
# 说明：所有可调参数都集中在“参数配置区”，手动修改后直接运行即可。

set -euo pipefail
export PYTHONUNBUFFERED=1

# ========================= 参数配置区（手动修改） =========================
# 自动定位仓库根目录（通常无需改）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 所有结果输出目录（可改为绝对路径）。
RESULTS_ROOT_DIR="${REPO_ROOT}/benchmarks/diffusion/results"

# 模型与服务参数。
MODEL="${MODEL:-/data2/group_谈海生/mumura/models/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8091}"
BASE_URL="http://localhost:${PORT}"

# GPU/并行参数（sp4_cfg2_hsdp）。
CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29600}"

# Benchmark 参数。
TASK="${TASK:-t2i}"
DATASET="${DATASET:-trace}"
DATASET_PATH="${DATASET_PATH:-/home/mumura/omni/dataset/trace/cogvideox_trace.txt/cogvideox_trace.txt}"
NUM_PROMPTS_DURATION_SECONDS="${NUM_PROMPTS_DURATION_SECONDS:-10}"
RPS_LIST="${RPS_LIST:-[0.1, 1]}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-200}"
# =======================================================================

# 结果目录：所有日志与 metrics 均写到该目录下。
mkdir -p "$RESULTS_ROOT_DIR"
ts=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_ROOT_DIR}/wan_sp4_cfg2_hsdp_rps_${ts}"
mkdir -p "$RUN_DIR"

SERVER_LOG="${RUN_DIR}/server.log"
MASTER_LOG="${RUN_DIR}/benchmark_master.log"
GPU_LOG="${RUN_DIR}/gpu.log"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

echo "===== Run dir: ${RUN_DIR} =====" | tee -a "$MASTER_LOG"
echo "===== Model: ${MODEL} =====" | tee -a "$MASTER_LOG"
echo "===== RPS list: ${RPS_LIST} =====" | tee -a "$MASTER_LOG"
echo "===== Duration(seconds): ${NUM_PROMPTS_DURATION_SECONDS} =====" | tee -a "$MASTER_LOG"
echo "===== Dataset: ${DATASET} =====" | tee -a "$MASTER_LOG"
echo "===== Dataset path: ${DATASET_PATH} =====" | tee -a "$MASTER_LOG"
echo "===== Repo root: ${REPO_ROOT} =====" | tee -a "$MASTER_LOG"

# 前置校验：脚本假设当前环境已配置好 python3 / vllm。
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH" | tee -a "$MASTER_LOG"
  exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH" | tee -a "$MASTER_LOG"
  exit 1
fi
if [ "$DATASET" = "trace" ] && [ -z "$DATASET_PATH" -o ! -f "$DATASET_PATH" ]; then
  echo "trace dataset file not found: ${DATASET_PATH}" | tee -a "$MASTER_LOG"
  echo "Please set DATASET_PATH to a local trace file path." | tee -a "$MASTER_LOG"
  exit 1
fi

cd "$REPO_ROOT"

cleanup() {
  if [ -n "${GPU_PID:-}" ]; then
    kill "$GPU_PID" 2>/dev/null || true
  fi
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "===== Starting single vLLM-Omni server: sp4_cfg2_hsdp =====" | tee -a "$MASTER_LOG"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_LIST"
export MASTER_ADDR
export MASTER_PORT

vllm serve "$MODEL" \
  --omni \
  --port "$PORT" \
  --num-gpus 8 \
  --tensor-parallel-size 1 \
  --usp 4 \
  --ring 1 \
  --cfg-parallel-size 2 \
  --use-hsdp \
  --hsdp-replicate-size 1 \
  --vae-use-slicing \
  --vae-use-tiling \
  >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "server pid=${SERVER_PID}, port=${PORT}" | tee -a "$MASTER_LOG"
echo "server log: ${SERVER_LOG}" | tee -a "$MASTER_LOG"

echo "Waiting for service ready at ${BASE_URL} (max 7200s)..." | tee -a "$MASTER_LOG"
for t in $(seq 1 7200); do
  if curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models" 2>/dev/null | grep -q 200; then
    echo "Service ready after ${t}s." | tee -a "$MASTER_LOG"
    break
  fi
  if [ "$t" -eq 7200 ]; then
    echo "Service did not become ready in time. Check ${SERVER_LOG}" | tee -a "$MASTER_LOG"
    exit 1
  fi
  if [ $((t % 60)) -eq 0 ] && [ "$t" -gt 0 ]; then
    echo "  still waiting (${t}s)..." | tee -a "$MASTER_LOG"
  fi
  sleep 1
done

# 后台采样 GPU 使用，便于确认服务与压测阶段 GPU 负载。
( while true; do echo "=== $(date -Iseconds) ==="; nvidia-smi; sleep 5; done ) >> "$GPU_LOG" 2>&1 &
GPU_PID=$!

printf '%s\n' "timestamp,rps,duration_seconds,num_prompts,metrics_json,log_file" > "$SUMMARY_CSV"

to_num_prompts() {
  local _rps="$1"
  local _duration="$2"
  # 按要求计算：num-prompts = rps * duration_seconds，四舍五入到整数。
  python3 - <<PY
from decimal import Decimal, ROUND_HALF_UP
rps = Decimal(${_rps@Q})
duration = Decimal(${_duration@Q})
value = (rps * duration).to_integral_value(rounding=ROUND_HALF_UP)
print(max(1, int(value)))
PY
}

normalize_rps_list() {
  local raw="$1"
  # 支持两种写法："[0.1, 0.5, 1]" 或 "0.1,0.5,1"。
  echo "$raw" | sed 's/\[//g; s/\]//g; s/,/ /g'
}

RPS_ITEMS=$(normalize_rps_list "$RPS_LIST")
if [ -z "$RPS_ITEMS" ]; then
  echo "RPS_LIST is empty after parsing: ${RPS_LIST}" | tee -a "$MASTER_LOG"
  exit 1
fi

echo "===== Running benchmark by RPS =====" | tee -a "$MASTER_LOG"
for rps in $RPS_ITEMS; do
  num_prompts=$(to_num_prompts "$rps" "$NUM_PROMPTS_DURATION_SECONDS")
  rps_label=${rps//./_}
  METRICS_FILE="${RUN_DIR}/metrics_rps_${rps_label}.json"
  RUN_LOG="${RUN_DIR}/bench_rps_${rps_label}.log"

  echo "[RPS=${rps}] num-prompts=${num_prompts}, duration=${NUM_PROMPTS_DURATION_SECONDS}s" | tee -a "$MASTER_LOG"

  python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --task "$TASK" \
    --dataset "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --num-prompts "$num_prompts" \
    --request-rate "$rps" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --output-file "$METRICS_FILE" \
    2>&1 | tee "$RUN_LOG"

  printf '%s,%s,%s,%s,%s,%s\n' \
    "$(date -Iseconds)" "$rps" "$NUM_PROMPTS_DURATION_SECONDS" "$num_prompts" "$METRICS_FILE" "$RUN_LOG" \
    >> "$SUMMARY_CSV"
done

echo "===== Done =====" | tee -a "$MASTER_LOG"
echo "summary: ${SUMMARY_CSV}" | tee -a "$MASTER_LOG"
echo "master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
echo "gpu log: ${GPU_LOG}" | tee -a "$MASTER_LOG"

exit 0
