#!/bin/bash
# 并行组合测试：仅测《并行测试表》中 GPU=SP×CFG×TP 的 16 种组合，每配置 5 次样本（1 次首次 + 5 次停→起）。
# 不发起推理、仅轮询 /v1/models。
# 用法（推荐 salloc 或直接 bash）：bash run_switch_parallel_qwen.sh
# 配置来自 config/run_switch_parallel_qwen.yaml（含待测 configs 表），环境变量可覆盖。

set -e
set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
SCRIPT_DIR="$SCRIPT_DIR"

# 加载配置（先公共环境，再本脚本 yaml）
if [ -f "$SCRIPT_DIR/load_config.sh" ]; then
  source "$SCRIPT_DIR/load_config.sh"
  load_config "config/common_env.yaml" "config/run_switch_parallel_qwen.yaml"
fi
apply_env
[ -f "$SCRIPT_DIR/common.sh" ] && source "$SCRIPT_DIR/common.sh"

REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
LOG_DIR="${WORK_DIR}/${LOG_SUBDIR:-qwen_parallel_log}"
mkdir -p "$LOG_DIR"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8099}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
CONFIG_IDS="${CONFIG_IDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
START_CONFIG="${START_CONFIG:-1}"

JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/run_parallel_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/switch_parallel_${JOBID}_${ts}.csv"
STATS_LOG="${LOG_DIR}/switch_parallel_${JOBID}_${ts}_stats.log"

exec >> "$RUN_LOG" 2>&1

num_configs=$(echo $CONFIG_IDS | wc -w)
echo "===== switch_time 并行组合测试：共 ${num_configs} 种配置（见 yaml config_ids），每配置 1 次首次 + ${NUM_SAMPLES} 次停→起 ====="
echo "REPO_DIR=$REPO_DIR  NUM_SAMPLES=$NUM_SAMPLES  START_CONFIG=$START_CONFIG"

cd "$REPO_DIR"
SKIP_BENCH=0
python3 -c "import vllm_omni" || { echo "ERROR: vllm_omni not importable，跳过全部测试。" >&2; SKIP_BENCH=1; }

if [ "$SKIP_BENCH" != "1" ]; then
echo "config_id,run,first_startup_s,stop_s,startup_s,switch_s,ready_poll_s" >> "$CSV"

for config_id in $CONFIG_IDS; do
  [ -z "$config_id" ] && continue
  [ "$config_id" -lt "$START_CONFIG" ] 2>/dev/null && continue
  echo "========== 配置 $config_id（见 yaml configs）=========="
  SERVER_LOG="${LOG_DIR}/server_parallel_c${config_id}_${JOBID}.log"

  # ---------- 首次启动 ----------
  pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG") || {
    echo "  [跳过] config $config_id 无效（不在 yaml configs 表），进入下一配置"
    continue
  }
  echo "  [首次启动] config $config_id started pgid=$pgid"
  T_start=$(date +%s.%N)
  wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
  if [ "$wait_poll" = "-1" ]; then
    kill -KILL -"$pgid" 2>/dev/null || true
    force_kill_port "$PORT" || true
    cleanup_gpu_residuals || true
    log_error_full "config $config_id 首次启动 failed to become ready" "$SERVER_LOG"
    echo "$config_id,0,ERROR,,,,$wait_poll" >> "$CSV"
    pgid=""
    sleep 2
    continue
  fi
  first_startup_s=$(python3 -c "print(round($(date +%s.%N) - $T_start, 2))")
  echo "  [首次启动] ready in ${first_startup_s}s  ready_poll=${wait_poll}s"
  echo "$config_id,0,$first_startup_s,,,,$wait_poll" >> "$CSV"

  # ---------- NUM_SAMPLES 次停→起 ----------
  for run in $(seq 1 "$NUM_SAMPLES"); do
    echo "  --- 样本 $run / $NUM_SAMPLES ---"
    if [ -n "$pgid" ]; then
      stop_s=$(stop_server_and_measure "$pgid")
      force_kill_port "$PORT" || true
      cleanup_gpu_residuals || true
      wait_port_released "$PORT" || true
      sleep 2
    else
      stop_s=""
    fi
    T_start=$(date +%s.%N)
    pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
    wait_poll=$(wait_ready "http://127.0.0.1:${PORT}" 7200) || wait_poll="-1"
    if [ "$wait_poll" = "-1" ]; then
      kill -KILL -"$pgid" 2>/dev/null || true
      force_kill_port "$PORT" || true
      cleanup_gpu_residuals || true
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

  if [ -n "$pgid" ]; then
    stop_server_and_measure "$pgid" >/dev/null
    force_kill_port "$PORT" || true
    cleanup_gpu_residuals || true
    wait_port_released "$PORT" || true
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
else
  echo "===== Done（未运行测试：vllm_omni 不可用）。Run log: $RUN_LOG ====="
fi
