#!/bin/bash
# 特性切换耗时测试：按测试表「分开测」— 配置 1 测完再测配置 2，依次到 26。每种配置独立：首次启动（单独记录）+ 10 次「停→起」样本（共 11 次/配置）。
# 不向实例发推理请求，仅轮询 /v1/models 判定就绪。依赖同目录 setup_env.sh。
# 用法：bash run_switch_all.sh 或 sbatch run_switch_all.sh
# 可选环境变量：START_CONFIG=从第几个配置开始（计数从 1）；测试中途报错时可设此参数续跑，例如 START_CONFIG=15 bash run_switch_all.sh
# 作业管理参考：https://saids.hpc.gleamoe.com/
# A100 资源：按你集群实际调整。若提交报 Memory/node configuration not available，请 sinfo -p A100 查看后改下面两行。
#SBATCH -J switch_all
#SBATCH -o %j_switch_all.out
#SBATCH -e %j_switch_all.err
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

# sbatch 时优先用提交目录，避免在 /var/spool/slurmd/… 下建 logs 导致 Permission denied
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
  WORK_DIR="${SLURM_SUBMIT_DIR}"
else
  WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8099}"
# 每个配置：1 次首次启动（单独记录）+ NUM_SAMPLES 次「停→起」用于 Stop/Startup/Switch 的 mean/std
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_CONFIGS=26
# 从第几个配置开始测（计数从 1）；默认 1；测试中途报错时可设此参数续跑
START_CONFIG="${START_CONFIG:-1}"

CONDA_ENV="${CONDA_ENV:-vllm_omni}"
JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/run_all_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/switch_all_${JOBID}_${ts}.csv"
STATS_LOG="${LOG_DIR}/switch_all_${JOBID}_${ts}_stats.log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "===== switch_time test: 所有配置 1..26 分开测（策略1测完测策略2，依次类推）====="
echo "REPO_DIR=$REPO_DIR  NUM_SAMPLES=$NUM_SAMPLES  START_CONFIG=$START_CONFIG（每配置 1 次首次 + ${NUM_SAMPLES} 次停→起）"

module purge 2>/dev/null || true
module load Anaconda3/2025.06 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true
if [ "${SKIP_SETUP_ENV:-0}" != "1" ]; then
  bash "$WORK_DIR/setup_env.sh"
fi
# setup_env 在子 shell 中执行，当前 shell 需自行激活 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
cd "$REPO_DIR"
python3 -c "import vllm_omni" || { echo "ERROR: vllm_omni not importable."; exit 1; }

# 根据《Qwen-Image-特性切换测试说明表》表 5.2：编号 → GPU,SP,CFG,TP,cache,cpu_off,fp8
# 输出可 eval 的字符串，设置 CUDA_VISIBLE_DEVICES 与 CONFIG_EXTRA（供 start_config_id 使用）
get_config_params() {
  local id="$1"
  local gpu sp cfg tp cache cpu_off fp8
  case "$id" in
    1)  gpu=1; sp=1; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    2)  gpu=1; sp=1; cfg=1; tp=1; cache=tea_cache; cpu_off=0; fp8=0 ;;
    3)  gpu=1; sp=1; cfg=1; tp=1; cache=cache_dit; cpu_off=0; fp8=0 ;;
    4)  gpu=2; sp=1; cfg=2; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    5)  gpu=2; sp=1; cfg=1; tp=2; cache=none;   cpu_off=0; fp8=0 ;;
    6)  gpu=4; sp=1; cfg=1; tp=4; cache=none;   cpu_off=0; fp8=0 ;;
    7)  gpu=8; sp=1; cfg=1; tp=8; cache=none;   cpu_off=0; fp8=0 ;;
    8)  gpu=1; sp=1; cfg=1; tp=1; cache=none;   cpu_off=1; fp8=0 ;;
    9)  gpu=1; sp=1; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=1 ;;
    10) gpu=2; sp=2; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    11) gpu=2; sp=2; cfg=1; tp=1; cache=tea_cache; cpu_off=0; fp8=0 ;;
    12) gpu=2; sp=2; cfg=1; tp=1; cache=cache_dit; cpu_off=0; fp8=0 ;;
    13) gpu=4; sp=2; cfg=2; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    14) gpu=4; sp=2; cfg=1; tp=2; cache=none;   cpu_off=0; fp8=0 ;;
    15) gpu=8; sp=2; cfg=1; tp=4; cache=none;   cpu_off=0; fp8=0 ;;
    16) gpu=2; sp=2; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=1 ;;
    17) gpu=4; sp=4; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    18) gpu=4; sp=4; cfg=1; tp=1; cache=tea_cache; cpu_off=0; fp8=0 ;;
    19) gpu=4; sp=4; cfg=1; tp=1; cache=cache_dit; cpu_off=0; fp8=0 ;;
    20) gpu=8; sp=4; cfg=2; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    21) gpu=8; sp=4; cfg=1; tp=2; cache=none;   cpu_off=0; fp8=0 ;;
    22) gpu=4; sp=4; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=1 ;;
    23) gpu=8; sp=8; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=0 ;;
    24) gpu=8; sp=8; cfg=1; tp=1; cache=tea_cache; cpu_off=0; fp8=0 ;;
    25) gpu=8; sp=8; cfg=1; tp=1; cache=cache_dit; cpu_off=0; fp8=0 ;;
    26) gpu=8; sp=8; cfg=1; tp=1; cache=none;   cpu_off=0; fp8=1 ;;
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
  [ "$cache" = "tea_cache" ] && extra="$extra --cache-backend tea_cache"
  [ "$cache" = "cache_dit" ] && extra="$extra --cache-backend cache_dit"
  [ "$cpu_off" = "1" ] && extra="$extra --enable-layerwise-offload"
  # FP8 JSON 用单独变量传递，避免 CONFIG_EXTRA 里引号被 eval/展开 吃掉（否则会变成 invalid loads value: "'{method:fp8}'"）
  [ "$fp8" = "1" ] && extra="$extra --quantization-config "'$FP8_JSON'
  echo "export CUDA_VISIBLE_DEVICES=\"$devs\"; export FP8_JSON='{\"method\":\"fp8\"}'; CONFIG_EXTRA='$extra'"
}

# 启动指定编号的配置，返回 PGID（进程组 ID）供 stop 按组杀；不用 eval 包整条命令，减少 $! 不可靠
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

# 停服：按 PGID 杀整组，等进程组清空（不等 wait $pid），超时则 SIGKILL；避免“父死子活”残留
stop_server_and_measure() {
  local pgid="$1"
  [ -z "$pgid" ] && return
  local T0 T1
  T0=$(date +%s.%N)
  kill -TERM -"$pgid" 2>/dev/null || true
  local i=0
  while [ $i -lt 60 ]; do
    if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
      break
    fi
    sleep 0.5
    i=$((i + 1))
  done
  if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
    kill -KILL -"$pgid" 2>/dev/null || true
    local j=0
    while [ $j -lt 25 ]; do
      if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
        break
      fi
      sleep 0.2
      j=$((j + 1))
    done
  fi
  T1=$(date +%s.%N)
  python3 -c "print(round($T1 - $T0, 2))"
}

# 清理本用户在 GPU 上的 python/vllm 残留（multiprocessing.spawn 孤儿 PPID=1 等），不依赖 lsof/fuser
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

# 按端口强制杀掉仍占用该端口的进程（兜底）
force_kill_port() {
  local port="$1"
  fuser -k "${port}/tcp" 2>/dev/null || true
  lsof -t -i ":${port}" 2>/dev/null | while read -r p; do kill -9 "$p" 2>/dev/null; done
  sleep 2
}

# 停服后等待端口释放：先用 ss 判断无 LISTEN，再（可选）curl 不 200；最多 120s
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
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内仍返回 200 或 ss 仍见 LISTEN，未释放，退出" >&2
  exit 1
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
  echo "========== 配置 $config_id / $NUM_CONFIGS（分开测，本配置测完再测下一配置）=========="
  SERVER_LOG="${LOG_DIR}/server_c${config_id}_${JOBID}.log"

  # ---------- 首次启动：单独记录，不参与 mean/std ----------
  pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
  echo "  [首次启动] config $config_id started pgid=$pgid"
  T_start=$(date +%s.%N)
  wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
  if [ "$wait_poll" = "-1" ]; then
    kill -KILL -"$pgid" 2>/dev/null || true
    force_kill_port "$PORT"
    cleanup_gpu_residuals
    log_error_full "config $config_id 首次启动 failed to become ready" "$SERVER_LOG"
    echo "$config_id,0,ERROR,,,,$wait_poll" >> "$CSV"
    sleep 2
    continue
  fi
  first_startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
  echo "  [首次启动] ready in ${first_startup_s}s  ready_poll=${wait_poll}s"
  echo "$config_id,0,$first_startup_s,,,,$wait_poll" >> "$CSV"

  # ---------- NUM_SAMPLES 次「停→起」样本（用于 Stop/Startup/Switch 的 mean/std）----------
  for run in $(seq 1 "$NUM_SAMPLES"); do
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
    T_start=$(date +%s.%N)
    pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
    wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
    if [ "$wait_poll" = "-1" ]; then
      kill -KILL -"$pgid" 2>/dev/null || true
      force_kill_port "$PORT"
      cleanup_gpu_residuals
      log_error_full "config $config_id run $run failed to become ready" "$SERVER_LOG"
      echo "$config_id,$run,,ERROR,ERROR,ERROR,ERROR" >> "$CSV"
      pgid=""
      sleep 2
      continue
    fi
    startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
    if [ -n "$stop_s" ]; then
      switch_s=$(python3 -c "print(round($stop_s + $startup_s, 2))")
    else
      switch_s="$startup_s"
    fi
    echo "    stop ${stop_s}s  startup ${startup_s}s  ready_poll=${wait_poll}s  switch ${switch_s}s"
    echo "$config_id,$run,,${stop_s:-},$startup_s,$switch_s,$wait_poll" >> "$CSV"
    sleep 2
  done

  # 本配置测完，停掉进程再测下一配置
  if [ -n "$pgid" ]; then
    stop_server_and_measure "$pgid" >/dev/null
    cleanup_gpu_residuals
    force_kill_port "$PORT"
    wait_port_released "$PORT"
  fi
  sleep 2
done

echo "===== 统计（按 config_id 聚合，run 1..${NUM_SAMPLES} 样本；首次 Startup 见 run=0 行）=====" | tee "$STATS_LOG"
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
# 首次启动行 run=0：只打印 首次Startup
first_startups = {r["config_id"]: r.get("first_startup_s") for r in rows if r.get("run") == "0"}
# 样本行 run>=1：用于 mean/std（跳过 ERROR、空或 None，避免 float 报错）
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
