#!/bin/bash
# 8 实例实验：8 张卡各起一个 vLLM-Omni 服务，benchmark 用 trace 数据集、round-robin 发请求，并收集日志与 metrics。
# 作业管理参考：https://saids.hpc.gleamoe.com/guideline/slurm/
# 分区与 QOS 以集群为准，可用 sinfo -o "%P %G" 与 sacctmgr show ass user=$USER format=user,part,qos 查看。
#SBATCH -J diffusion_8inst
#SBATCH -o diffusion_8inst.%j.out
#SBATCH -e diffusion_8inst.%j.err
#SBATCH -p A100
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:8
#SBATCH -t 4:00:00

set -e
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -z "$HF_HOME" ]; then
  _G=$(groups 2>/dev/null | awk '{print $1}')
  _U=$(whoami)
  if [ -n "$_G" ] && [ -d "/data2/$_G/$_U" ] && [ -w "/data2/$_G/$_U" ]; then
    export HF_HOME="/data2/$_G/$_U/xhf/hf_cache"
  else
    export HF_HOME="$HOME/xhf/hf_cache"
  fi
fi

module purge
module load Anaconda3/2025.06
module load cuda/12.9.1

CONDA_ENV=vllm_omni
REPO_DIR="${REPO_DIR:-$HOME/xhf/vllm-omni}"
NUM_INSTANCES=8
BASE_PORT=8099

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
  echo "===== Creating conda env: $CONDA_ENV ====="
  conda create -n $CONDA_ENV python=3.12 -y
  conda activate $CONDA_ENV
  echo "===== Installing vllm and vllm-omni ====="
  cd "$REPO_DIR"
  pip install vllm==0.14.0
  pip install -e .
else
  conda activate $CONDA_ENV
  cd "$REPO_DIR"
  pip install -q vllm==0.14.0
  pip install -q -e .
fi

MODEL="${MODEL:-Qwen/Qwen-Image}"
LOG_DIR="${REPO_DIR}/xhf/logs"
mkdir -p "$LOG_DIR"
ts=$(date +%Y%m%d_%H%M%S)
METRICS_FILE="${LOG_DIR}/8inst_trace_${ts}_metrics.json"
LOG_FILE="${LOG_DIR}/8inst_trace_${ts}.log"

echo "===== Starting ${NUM_INSTANCES} vLLM-Omni servers (1 per GPU) ====="
PIDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  export CUDA_VISIBLE_DEVICES=$i
  # 每个实例用不同的 MASTER_PORT，避免同节点多进程 TCPStore 端口冲突 (EADDRINUSE)
  export MASTER_PORT=$((29600 + i))
  export MASTER_ADDR=127.0.0.1
  port=$((BASE_PORT + i))
  vllm serve "$MODEL" --omni --port "$port" >> "${LOG_DIR}/server_${i}.log" 2>&1 &
  PIDS+=( $! )
  echo "  instance $i: port $port pid $!"
done

echo "Waiting for all ${NUM_INSTANCES} servers to be ready (max 7200s each)..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  port=$((BASE_PORT + i))
  url="http://localhost:${port}"
  echo "  Waiting for instance $i (port $port)..."
  for t in $(seq 1 7200); do
    if curl -s -o /dev/null -w "%{http_code}" "${url}/v1/models" 2>/dev/null | grep -q 200; then
      echo "  instance $i (port $port) ready after ${t}s."
      break
    fi
    if [ $t -eq 7200 ]; then
      echo "Instance $i (port $port) did not become ready in time. Check ${LOG_DIR}/server_${i}.log for errors."
      for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done
      exit 1
    fi
    if [ $((t % 60)) -eq 0 ] && [ $t -gt 0 ]; then
      echo "    still waiting for instance $i (${t}s)..."
    fi
    sleep 1
  done
done

BASE_URLS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  BASE_URLS+=( "http://localhost:$((BASE_PORT + i))" )
done

# 8 卡 × 25 = 200 个请求。可通过 NUM_PROMPTS 覆盖。
NUM_PROMPTS="${NUM_PROMPTS:-200}"
GPU_LOG="${LOG_DIR}/gpu_$(date +%Y%m%d_%H%M%S).log"
# 后台记录 GPU 使用（每 5 秒一帧），便于事后确认 8 张卡是否都在干活；想看实时可另开终端 srun --jobid=<JOBID> --pty nvtop
( while true; do echo "=== $(date -Iseconds) ==="; nvidia-smi; sleep 5; done ) >> "$GPU_LOG" 2>&1 &
GPU_PID=$!
trap "kill $GPU_PID 2>/dev/null || true" EXIT

echo "===== Running diffusion benchmark (trace, t2i, round-robin over ${NUM_INSTANCES} instances, num_prompts=${NUM_PROMPTS}) ====="
echo "===== GPU 采样: $GPU_LOG （实时看 8 卡可另开终端: srun --jobid=\$SLURM_JOB_ID --pty nvtop） ====="
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-urls "${BASE_URLS[@]}" \
  --model "$MODEL" \
  --task t2i \
  --dataset trace \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "${MAX_CONCURRENCY:-200}" \
  --output-file "$METRICS_FILE" \
  2>&1 | tee "$LOG_FILE"

echo "===== Done. Metrics: $METRICS_FILE  Log: $LOG_FILE ====="
for p in "${PIDS[@]}"; do
  kill $p 2>/dev/null || true
done
exit 0
