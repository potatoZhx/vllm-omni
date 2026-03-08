#!/bin/bash
# 并行组合测试：仅测《并行测试表》中 GPU=SP×CFG×TP 的 16 种组合，每配置 5 次样本（1 次首次 + 5 次停→起）。
# 行为与 run_switch_all.sh 一致（不发起推理、仅轮询 /v1/models、同目录 setup_env.sh）。
# 错误累计：每配置内仅统计「未就绪」失败次数（首次或某次样本 wait_ready 超时）；达 10 次则放弃本配置、测下一配置。
# 用法：bash run_switch_parallel.sh 或 sbatch run_switch_parallel.sh
# 可选环境变量：START_CONFIG=从第几个配置开始（计数从 1）；测试中途报错时可设此参数续跑，例如 START_CONFIG=8 bash run_switch_parallel.sh
#SBATCH -J switch_parallel
#SBATCH -o %j_switch_parallel.out
#SBATCH -e %j_switch_parallel.err
#SBATCH -p A100
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH -t 72:00:00

set -e
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -z "$HF_HOME" ]; then
  _G=$(groups 2>/dev/null | awk '{print $1}')
  _U=$(whoami)
  if [ -n "$_G" ] && [ -d "/data2/$_G/$_U" ] && [ -w "/data2/$_G/$_U" ]; then
    export HF_HOME="/data2/$_G/$_U/xhf/hf_cache"
  else
    export HF_HOME="${HF_HOME:-$HOME/xhf/hf_cache}"
  fi
fi

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
  WORK_DIR="${SLURM_SUBMIT_DIR}"
else
  WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
LOG_DIR="${WORK_DIR}/qwen_parallel_log"
mkdir -p "$LOG_DIR"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8099}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
NUM_CONFIGS=16
# 本配置内累计该次数量的「失败事件」则放弃本配置，测下一配置
CONFIG_ERROR_LIMIT="${CONFIG_ERROR_LIMIT:-10}"
# 本配置 server 日志累计 ERROR 关键字超过该值则放弃整个配置，测下一配置
LOG_ERROR_LIMIT="${LOG_ERROR_LIMIT:-20}"
# 从第几个配置开始测（计数从 1）；默认 1；测试中途报错时可设此参数续跑
START_CONFIG="${START_CONFIG:-1}"

CONDA_ENV="${CONDA_ENV:-vllm_omni}"
JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/run_parallel_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/switch_parallel_${JOBID}_${ts}.csv"
STATS_LOG="${LOG_DIR}/switch_parallel_${JOBID}_${ts}_stats.log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "===== switch_time 并行组合测试：仅 GPU=SP×CFG×TP 共 ${NUM_CONFIGS} 种配置，每配置 1 次首次 + ${NUM_SAMPLES} 次停→起 ====="
echo "REPO_DIR=$REPO_DIR  NUM_SAMPLES=$NUM_SAMPLES  START_CONFIG=$START_CONFIG  单配置错误上限=$CONFIG_ERROR_LIMIT  日志累计 ERROR 上限=$LOG_ERROR_LIMIT（本配置超限则放弃整个配置）"

module purge 2>/dev/null || true
module load Anaconda3/2025.06 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true
if [ "${SKIP_SETUP_ENV:-0}" != "1" ]; then
  bash "$WORK_DIR/setup_env.sh"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
cd "$REPO_DIR"
python3 -c "import vllm_omni" || { echo "ERROR: vllm_omni not importable."; exit 1; }

# 《并行测试表》16 种组合：仅 (GPU, SP, CFG, TP)，无 cache/cpu_off/fp8
get_config_params() {
  local id="$1"
  local gpu sp cfg tp
  case "$id" in
    1)  gpu=1; sp=1; cfg=1; tp=1 ;;
    2)  gpu=2; sp=1; cfg=1; tp=2 ;;
    3)  gpu=4; sp=1; cfg=1; tp=4 ;;
    4)  gpu=8; sp=1; cfg=1; tp=8 ;;
    5)  gpu=2; sp=1; cfg=2; tp=1 ;;
    6)  gpu=4; sp=1; cfg=2; tp=2 ;;
    7)  gpu=8; sp=1; cfg=2; tp=4 ;;
    8)  gpu=2; sp=2; cfg=1; tp=1 ;;
    9)  gpu=4; sp=2; cfg=1; tp=2 ;;
    10) gpu=8; sp=2; cfg=1; tp=4 ;;
    11) gpu=4; sp=2; cfg=2; tp=1 ;;
    12) gpu=8; sp=2; cfg=2; tp=2 ;;
    13) gpu=4; sp=4; cfg=1; tp=1 ;;
    14) gpu=8; sp=4; cfg=1; tp=2 ;;
    15) gpu=8; sp=4; cfg=2; tp=1 ;;
    16) gpu=8; sp=8; cfg=1; tp=1 ;;
    *)  echo "ERROR: invalid config id $id"; return 1 ;;
  esac
  local devs="0"
  [ "$gpu" -ge 2 ] && devs="0,1"
  [ "$gpu" -ge 4 ] && devs="0,1,2,3"
  [ "$gpu" -ge 8 ] && devs="0,1,2,3,4,5,6,7"
  local extra=""
  [ "$sp" -gt 1 ] && extra="$extra --ulysses-degree $sp"
  [ "$cfg" -gt 1 ] && extra="$extra --cfg-parallel-size $cfg"
  [ "$tp" -gt 1 ] && extra="$extra --tensor-parallel-size $tp"
  echo "export CUDA_VISIBLE_DEVICES=\"$devs\"; CONFIG_EXTRA='$extra'"
}

start_config_id() {
  local id="$1"
  local port="$2"
  local log="$3"
  eval "$(get_config_params "$id")"
  setsid vllm serve "$MODEL" --omni --port "$port" $CONFIG_EXTRA >> "$log" 2>&1 &
  local leader_pid=$!
  local pgid
  pgid=$(ps -o pgid= -p "$leader_pid" 2>/dev/null | tr -d ' ')
  if [ -z "$pgid" ]; then
    echo "$leader_pid"
  else
    echo "$pgid"
  fi
}

wait_ready() {
  local url="$1"
  local max_wait="${2:-7200}"
  local t=0
  while [ $t -lt "$max_wait" ]; do
    if curl -s -o /dev/null -w "%{http_code}" "${url}/v1/models" 2>/dev/null | grep -q 200; then
      echo "$t"
      return 0
    fi
    sleep 1
    t=$((t + 1))
    if [ $((t % 60)) -eq 0 ] && [ $t -gt 0 ]; then echo "    wait_ready ${t}s..." >&2; fi
  done
  echo "-1"
  return 1
}

stop_server_and_measure() {
  local pgid="$1"
  [ -z "$pgid" ] && return
  local T0 T1
  T0=$(date +%s.%N)
  kill -TERM -"$pgid" 2>/dev/null || true
  local i=0
  while [ $i -lt 60 ]; do
    # awk: 有进程 c>0 → exit 1；无进程 c==0 → exit 0。应在无进程时 break
    if ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
      break
    fi
    sleep 0.5
    i=$((i + 1))
  done
  # 仍有进程时才发 SIGKILL：awk 有进程→exit 1，! awk 为真
  if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
    kill -KILL -"$pgid" 2>/dev/null || true
    local j=0
    while [ $j -lt 25 ]; do
      # 无进程时 break
      if ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
        break
      fi
      sleep 0.2
      j=$((j + 1))
    done
  fi
  T1=$(date +%s.%N)
  python3 -c "print(round($T1 - $T0, 2))"
}

cleanup_gpu_residuals() {
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F',' '{print $1}' | tr -d ' ')
  [ -z "$pids" ] && return 0
  local my_pids=""
  for p in $pids; do
    if ps -o user= -p "$p" 2>/dev/null | grep -q "^$(whoami)$"; then
      my_pids="$my_pids $p"
    fi
  done
  [ -z "$my_pids" ] && return 0
  kill -TERM $my_pids 2>/dev/null || true
  sleep 1
  kill -KILL $my_pids 2>/dev/null || true
}

force_kill_port() {
  local port="$1"
  fuser -k "${port}/tcp" 2>/dev/null || true
  lsof -t -i ":${port}" 2>/dev/null | while read -r p; do kill -9 "$p" 2>/dev/null; done
  sleep 2
}

wait_port_released() {
  local port="$1"
  local max_wait=120
  local i=0
  while [ $i -lt "$max_wait" ]; do
    if ! ss -ltnp 2>/dev/null | grep -q ":${port} "; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
    [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ] && echo "    wait_port_released ${i}s..." >&2
  done
  echo "  [最后尝试] 按端口强制清理后退出" >&2
  force_kill_port "$port"
  cleanup_gpu_residuals
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内未释放，退出" >&2
  exit 1
}

# 统计 server 日志中 ERROR 行数（只输出一个整数，避免多行/空导致 $(( )) 报错）
# 统计 server 日志中 ERROR 行数（排除 429/Too Many Requests，避免 HF 限流误触发放弃配置）
count_log_errors() {
  [ -z "$1" ] && echo "0" && return
  [ ! -f "$1" ] && echo "0" && return
  local c
  c=$(grep "ERROR" "$1" 2>/dev/null | grep -v -E "429|Too Many Requests" | wc -l) || c=0
  c=$(echo "$c" | tr -cd '0-9')
  echo "${c:-0}"
}

log_error_full() {
  local msg="$1"
  local log_file="$2"
  echo ""
  echo "========== ERROR =========="
  echo "  $msg"
  echo "  Time: $(date -Iseconds 2>/dev/null || date)"
  if [ -n "$log_file" ] && [ -f "$log_file" ]; then
    echo "  --- 相关 server 日志末尾 (last 300 lines): $log_file ---"
    tail -n 300 "$log_file" 2>/dev/null || true
    echo "  --- 以上为 $log_file 末尾 ---"
  fi
  echo "========== 继续后续测试 =========="
  echo ""
}

echo "config_id,run,first_startup_s,stop_s,startup_s,switch_s,ready_poll_s" >> "$CSV"

for config_id in $(seq "$START_CONFIG" "$NUM_CONFIGS"); do
  config_errors=0
  log_err_prev=0
  echo "========== 配置 $config_id / $NUM_CONFIGS（并行表，本配置错误达 $CONFIG_ERROR_LIMIT 次则放弃）=========="
  SERVER_LOG="${LOG_DIR}/server_parallel_c${config_id}_${JOBID}.log"

  # ---------- 首次启动 ----------
  log_err_before=$(count_log_errors "$SERVER_LOG")
  log_err_before=${log_err_before:-0}
  pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
  echo "  [首次启动] config $config_id started pgid=$pgid"
  T_start=$(date +%s.%N)
  wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
  log_err_now=$(count_log_errors "$SERVER_LOG")
  log_err_now=${log_err_now:-0}
  log_err_delta=$((log_err_now - log_err_before))
  [ "$log_err_now" -gt "$log_err_prev" ] 2>/dev/null && config_errors=$((config_errors + 1))
  log_err_prev=$log_err_now
  if [ "$wait_poll" = "-1" ]; then
    config_errors=$((config_errors + 1))
    kill -KILL -"$pgid" 2>/dev/null || true
    force_kill_port "$PORT"
    cleanup_gpu_residuals
    log_error_full "config $config_id 首次启动 failed to become ready" "$SERVER_LOG"
    echo "$config_id,0,ERROR,,,,$wait_poll" >> "$CSV"
    echo "  [错误累计] config $config_id 当前 $config_errors 次"
    if [ "$config_errors" -ge "$CONFIG_ERROR_LIMIT" ]; then
      echo "  [放弃] 配置 $config_id 累计错误已达 $CONFIG_ERROR_LIMIT，跳过本配置"
      sleep 2
      continue
    fi
    pgid=""
    sleep 2
    continue
  fi
  # 本配置日志累计 ERROR 超过上限则放弃整个配置
  if [ "${log_err_now:-0}" -gt "${LOG_ERROR_LIMIT:-20}" ] 2>/dev/null; then
    log_error_full "config $config_id 首次启动后 本配置日志累计 ERROR($log_err_now) 超过 $LOG_ERROR_LIMIT，放弃整个配置" "$SERVER_LOG"
    echo "$config_id,0,ERROR,,,,$wait_poll" >> "$CSV"
    if [ -n "$pgid" ]; then
      kill -KILL -"$pgid" 2>/dev/null || true
      force_kill_port "$PORT"
      cleanup_gpu_residuals
      wait_port_released "$PORT" 2>/dev/null || true
    fi
    pgid=""
    sleep 2
    continue
  fi
  first_startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
  echo "  [首次启动] ready in ${first_startup_s}s  ready_poll=${wait_poll}s"
  echo "$config_id,0,$first_startup_s,,,,$wait_poll" >> "$CSV"

  # ---------- NUM_SAMPLES 次停→起 ----------
  for run in $(seq 1 "$NUM_SAMPLES"); do
    if [ "$config_errors" -ge "$CONFIG_ERROR_LIMIT" ]; then
      echo "  [放弃] 配置 $config_id 累计错误已达 $CONFIG_ERROR_LIMIT，不再采样"
      break
    fi
    echo "  --- 样本 $run / $NUM_SAMPLES ---"
    if [ -n "$pgid" ]; then
      stop_s=$(stop_server_and_measure "$pgid")
      cleanup_gpu_residuals
      force_kill_port "$PORT"
      wait_port_released "$PORT"
      sleep 2
    else
      stop_s=""
    fi
    log_err_before=$(count_log_errors "$SERVER_LOG")
    log_err_before=${log_err_before:-0}
    T_start=$(date +%s.%N)
    pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
    wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
    log_err_now=$(count_log_errors "$SERVER_LOG")
    log_err_now=${log_err_now:-0}
    log_err_delta=$((log_err_now - log_err_before))
    [ "$log_err_now" -gt "$log_err_prev" ] 2>/dev/null && config_errors=$((config_errors + 1))
    log_err_prev=$log_err_now
    if [ "$wait_poll" = "-1" ]; then
      config_errors=$((config_errors + 1))
      kill -KILL -"$pgid" 2>/dev/null || true
      force_kill_port "$PORT"
      cleanup_gpu_residuals
      log_error_full "config $config_id run $run failed to become ready" "$SERVER_LOG"
      echo "$config_id,$run,,ERROR,ERROR,ERROR,ERROR" >> "$CSV"
      echo "  [错误累计] config $config_id 当前 $config_errors 次"
      pgid=""
      sleep 2
      continue
    fi
    # 本配置日志累计 ERROR 超过上限则放弃整个配置
    if [ "${log_err_now:-0}" -gt "${LOG_ERROR_LIMIT:-20}" ] 2>/dev/null; then
      log_error_full "config $config_id run $run 本配置日志累计 ERROR($log_err_now) 超过 $LOG_ERROR_LIMIT，放弃整个配置" "$SERVER_LOG"
      echo "$config_id,$run,,ERROR,ERROR,ERROR,ERROR" >> "$CSV"
      break
    fi
    startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
    if [ -n "$stop_s" ]; then
      switch_s=$(python3 -c "print(round($stop_s + $startup_s, 2))")
    else
      switch_s="$startup_s"
    fi
    echo "    stop ${stop_s}s  startup ${startup_s}s  ready_poll=${wait_poll}s  switch ${switch_s}s"
    echo "$config_id,$run,,${stop_s:-},$startup_s,$switch_s,$wait_poll" >> "$CSV"
    if [ "$config_errors" -ge "$CONFIG_ERROR_LIMIT" ]; then
      echo "  [放弃] 配置 $config_id 累计错误已达 $CONFIG_ERROR_LIMIT，停止本配置采样"
      break
    fi
    sleep 2
  done

  if [ -n "$pgid" ]; then
    stop_server_and_measure "$pgid" >/dev/null
    cleanup_gpu_residuals
    force_kill_port "$PORT"
    wait_port_released "$PORT"
  fi
  sleep 2
done

echo "===== 统计（按 config_id 聚合，run 1..${NUM_SAMPLES} 样本）=====" | tee "$STATS_LOG"
python3 - "$CSV" "$NUM_SAMPLES" << 'PYSTATS' | tee -a "$STATS_LOG"
import csv
import sys
from pathlib import Path
from collections import defaultdict

csv_path = Path(sys.argv[1])
num_samples = int(sys.argv[2])
if not csv_path.exists():
    sys.exit(0)
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
first_startups = {r["config_id"]: r.get("first_startup_s") for r in rows if r.get("run") == "0"}

def _valid(r, key):
    v = (r.get(key) or "").strip()
    return v and v != "ERROR"
sample_rows = [r for r in rows if r.get("run") != "0" and _valid(r, "stop_s") and _valid(r, "startup_s") and _valid(r, "switch_s")]
by_config = defaultdict(list)
for r in sample_rows:
    by_config[r["config_id"]].append(r)

print("config_id  first_startup_s  Stop_mu  Stop_sigma  Startup_mu  Startup_sigma  Switch_mu  Switch_sigma")
for config_id in sorted(by_config.keys(), key=int):
    sub = by_config[config_id]
    first_s = first_startups.get(config_id, "")
    stop = [float(r["stop_s"]) for r in sub]
    startup = [float(r["startup_s"]) for r in sub]
    switch = [float(r["switch_s"]) for r in sub]
    n = len(sub)
    if n == 0:
        print(f"{config_id}  {first_s}  —  —  —  —  —  —")
        continue
    mu_s = sum(stop) / n
    mu_u = sum(startup) / n
    mu_w = sum(switch) / n
    sig_s = (sum((x - mu_s) ** 2 for x in stop) / n) ** 0.5
    sig_u = (sum((x - mu_u) ** 2 for x in startup) / n) ** 0.5
    sig_w = (sum((x - mu_w) ** 2 for x in switch) / n) ** 0.5
    print(f"{config_id}  {first_s}  {mu_s:.2f}  {sig_s:.2f}  {mu_u:.2f}  {sig_u:.2f}  {mu_w:.2f}  {sig_w:.2f}")
PYSTATS

echo "===== Done. CSV: $CSV  Stats: $STATS_LOG  Run log: $RUN_LOG ====="
