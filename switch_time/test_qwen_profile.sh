#!/bin/bash
# Qwen-Image 文生图性能画像：按 profile.md 要求，对每个数据项向每个配置项发起 5 次请求，记录每次完成时间。
# 不修改 vllm-omni-2 源码：仅通过 vllm serve CLI 与 /v1/images/generations HTTP 接口完成。
# 用法：在 switch_time 目录下，先激活 conda 环境后执行：bash test_qwen_profile.sh 或 sbatch test_qwen_profile.sh
# 可选环境变量：START_DATA_ITEM=1（从第几个数据项开始）, START_CONFIG=1（从第几个配置开始）, SKIP_SETUP_ENV=1 跳过 setup_env
#SBATCH -J qwen_profile
#SBATCH -o %j_qwen_profile.out
#SBATCH -e %j_qwen_profile.err
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
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

MODEL="${MODEL:-Qwen/Qwen-Image}"
PORT="${PORT:-8099}"
BASE_URL="http://127.0.0.1:${PORT}"
REQUESTS_PER_CONFIG=5

# profile.md：正负 prompt 固定
PROMPT='A realistic photo of a close-up of a violin on velvet fabric, warm studio lighting, fine wood grain'
NEGATIVE_PROMPT='low quality, blurry'

# profile.md：9 个配置项（编号 1,2,3,5,6,8,9,11,13）
CONFIG_IDS="1 2 3 5 6 8 9 11 13"
# 仅 CFG>1 的配置才在请求中带 negative_prompt（即 5, 6, 11）
CONFIGS_WITH_CFG="5 6 11"

# 数据项：5 种分辨率 × 4 档 steps = 20 种，编码在脚本中
# 格式：每行 "size steps" 如 "128x128 1"
DATA_ITEMS="128x128 1
128x128 5
128x128 10
128x128 50
256x256 1
256x256 5
256x256 10
256x256 50
512x512 1
512x512 5
512x512 10
512x512 50
1024x1024 1
1024x1024 5
1024x1024 10
1024x1024 50
1536x1536 1
1536x1536 5
1536x1536 10
1536x1536 50"

CONDA_ENV="${CONDA_ENV:-vllm_omni}"
JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/qwen_profile_run_${JOBID}_${ts}.log"
RESULT_LOG="${LOG_DIR}/qwen_profile_result_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/qwen_profile_${JOBID}_${ts}.csv"

START_DATA_ITEM="${START_DATA_ITEM:-1}"
START_CONFIG="${START_CONFIG:-1}"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "===== Qwen-Image 文生图性能画像（profile.md）====="
echo "数据项：20 种（5 分辨率 × 4 steps），配置项：9 个，每 (数据项, 配置) 发 ${REQUESTS_PER_CONFIG} 次请求"
echo "REPO_DIR=$REPO_DIR  PORT=$PORT  结果追加到: $RESULT_LOG"

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

# 仅变更待测配置项（SP/CFG/TP），其余特性一律使用默认（如 cache、cpu-offload、FP8 等均不显式传入）
# 与 run_switch_parallel.sh 一致的 9 个配置（profile 表：1,2,3,5,6,8,9,11,13）
get_config_params() {
  local id="$1"
  local gpu sp cfg tp
  case "$id" in
    1)  gpu=1; sp=1; cfg=1; tp=1 ;;
    2)  gpu=2; sp=1; cfg=1; tp=2 ;;
    3)  gpu=4; sp=1; cfg=1; tp=4 ;;
    5)  gpu=2; sp=1; cfg=2; tp=1 ;;
    6)  gpu=4; sp=1; cfg=2; tp=2 ;;
    8)  gpu=2; sp=2; cfg=1; tp=1 ;;
    9)  gpu=4; sp=2; cfg=1; tp=2 ;;
    11) gpu=4; sp=2; cfg=2; tp=1 ;;
    13) gpu=4; sp=4; cfg=1; tp=1 ;;
    *)  echo "ERROR: invalid config id $id (only 1,2,3,5,6,8,9,11,13)"; return 1 ;;
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

# 判断配置是否使用 CFG（需要传 negative_prompt）
config_has_cfg() {
  case "$1" in 5|6|11) return 0 ;; *) return 1 ;; esac
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
  if [ -z "$pgid" ]; then echo "$leader_pid"; else echo "$pgid"; fi
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
    if ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then break; fi
    sleep 0.5
    i=$((i + 1))
  done
  if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
    kill -KILL -"$pgid" 2>/dev/null || true
    local j=0
    while [ $j -lt 25 ]; do
      if ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then break; fi
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
    if ps -o user= -p "$p" 2>/dev/null | grep -q "^$(whoami)$"; then my_pids="$my_pids $p"; fi
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
    if ! ss -ltnp 2>/dev/null | grep -q ":${port} "; then return 0; fi
    sleep 1
    i=$((i + 1))
    [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ] && echo "    wait_port_released ${i}s..." >&2
  done
  force_kill_port "$port"
  cleanup_gpu_residuals
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内未释放" >&2
  exit 1
}

# 发单次文生图请求，输出耗时（秒，保留 4 位小数以准确统计亚秒级请求），失败输出 -1
# 用法：send_one_request "256x256" 50 1 表示 size=256x256 steps=50 且带 negative_prompt（第3参数为 1）
send_one_request() {
  local size="$1"
  local steps="$2"
  local use_neg="${3:-0}"
  local json
  if [ "$use_neg" = "1" ]; then
    json=$(printf '{"prompt":"%s","negative_prompt":"%s","size":"%s","num_inference_steps":%s,"n":1}' "$PROMPT" "$NEGATIVE_PROMPT" "$size" "$steps")
  else
    json=$(printf '{"prompt":"%s","size":"%s","num_inference_steps":%s,"n":1}' "$PROMPT" "$size" "$steps")
  fi
  local t0 t1
  t0=$(date +%s.%N)
  curl -s -o /tmp/qwen_profile_resp_$$.json -X POST "${BASE_URL}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "$json" >/dev/null 2>&1
  t1=$(date +%s.%N)
  if python3 -c "
import json, sys
try:
    with open('/tmp/qwen_profile_resp_$$.json') as f:
        d = json.load(f)
    if d.get('data') and len(d['data']) > 0 and d['data'][0].get('b64_json'):
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    python3 -c "print(round($t1 - $t0, 4))"
  else
    echo "-1"
  fi
  rm -f /tmp/qwen_profile_resp_$$.json
}

# CSV 表头
echo "data_item_id,size,steps,config_id,run,request_time_s" >> "$CSV"

data_item_id=0
while read -r size steps; do
  [ -z "$size" ] && continue
  data_item_id=$((data_item_id + 1))
  if [ "$data_item_id" -lt "$START_DATA_ITEM" ]; then continue; fi

  echo "========== 数据项 $data_item_id / 20  size=$size steps=$steps =========="

  for config_id in $CONFIG_IDS; do
    if [ "$config_id" -lt "$START_CONFIG" ] 2>/dev/null; then continue; fi

    SERVER_LOG="${LOG_DIR}/qwen_profile_server_d${data_item_id}_c${config_id}_${JOBID}_${ts}.log"
    pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
    echo "  [配置 $config_id] 启动 pgid=$pgid"
    wait_poll=$(wait_ready "$BASE_URL" 7200) || wait_poll="-1"
    if [ "$wait_poll" = "-1" ]; then
      echo "  [配置 $config_id] 未就绪，跳过"
      kill -KILL -"$pgid" 2>/dev/null || true
      force_kill_port "$PORT"
      cleanup_gpu_residuals
      wait_port_released "$PORT" 2>/dev/null || true
      continue
    fi
    echo "  [配置 $config_id] 就绪 ready_poll=${wait_poll}s，发 ${REQUESTS_PER_CONFIG} 次请求"

    use_neg=0
    config_has_cfg "$config_id" && use_neg=1
    run=1
    while [ "$run" -le "$REQUESTS_PER_CONFIG" ]; do
      tt=$(send_one_request "$size" "$steps" "$use_neg")
      echo "    run $run / $REQUESTS_PER_CONFIG  request_time_s=$tt"
      echo "$data_item_id,$size,$steps,$config_id,$run,$tt" >> "$CSV"
      run=$((run + 1))
    done

    stop_server_and_measure "$pgid" >/dev/null
    cleanup_gpu_residuals
    force_kill_port "$PORT"
    wait_port_released "$PORT"
    sleep 2
  done

  # 本数据项所有配置测完，整理并追加到结果日志（profile.md：每当一个数据项测试完成所有的配置项后）
  # 均值只统计 run2~5，首请求(run1) 单独列出（避免首请求 warmup 拉高均值）
  {
    echo "--- 数据项 $data_item_id 完成 $(date -Iseconds 2>/dev/null || date) ---"
    echo "  size=$size steps=$steps"
    echo "  各配置：首请求(run1) 单独列出，均值仅 run2~5(秒)："
    awk -F',' -v did="$data_item_id" 'NR>1 && $1==did && $6!="" && $6!="-1" {
      if ($5==1) { run1[$4]=$6 }
      else { sum[$4]+=$6; n[$4]++ }
    }
    END {
      cids="1 2 3 5 6 8 9 11 13"
      ncfg=split(cids, arr)
      for (i=1; i<=ncfg; i++) {
        c=arr[i]+0
        if (!(c in run1) && !(c in n)) next
        r1 = (c in run1) ? sprintf("%.4fs", run1[c]) : "—"
        mu  = (c in n && n[c]>0) ? sprintf("%.4fs", sum[c]/n[c]) : "—"
        printf "    config_%s: 首请求=%s  均值(run2~5)=%s\n", c, r1, mu
      }
    }' "$CSV" 2>/dev/null || true
    echo "  本数据项 均值(仅run2~5)(秒): $(awk -F',' -v did="$data_item_id" 'NR>1 && $1==did && $5>1 && $6!="" && $6!="-1" {s+=$6; c++} END {printf "%.4f", (c>0 ? s/c : 0)}' "$CSV" 2>/dev/null)"
    echo "  明细见 CSV: $CSV (data_item_id=$data_item_id)"
    echo ""
  } >> "$RESULT_LOG"
  echo "  已追加到 $RESULT_LOG"
done <<< "$DATA_ITEMS"

# 全量跑完后追加：请求完成平均时间统计表（均值仅 run2~5，首请求 run1 单独表）
if [ -f "$CSV" ]; then
  STATS_LOG="${LOG_DIR}/qwen_profile_${JOBID}_${ts}_stats.log"
  _stats_content() {
    echo "===== 请求完成平均时间统计（均值仅 run2~5，首请求 run1 单独列出；精度 4 位小数）====="
    echo "--- 均值(仅 run2~5)(秒) ---"
    echo "data_item_id  size         steps  config_1  config_2  config_3  config_5  config_6  config_8  config_9  config_11 config_13   平均(秒)"
    awk -F',' '
      NR>1 && $6!="" && $6!="-1" && $5>1 {
        did=$1; size=$2; steps=$3; cid=$4; t=$6
        sum[did]+=t; n[did]++
        cfg_sum[did,cid]+=t; cfg_n[did,cid]++
        if (!(did in size_done)) { sz[did]=size; st[did]=steps; size_done[did]=1 }
      }
      END {
        cids="1 2 3 5 6 8 9 11 13"
        ncfg=split(cids, arr)
        for (did=1; did<=20; did++) {
          if (!(did in n)) next
          printf "%s  %-12s  %-5s  ", did, sz[did], st[did]
          for (i=1; i<=ncfg; i++) {
            cid=arr[i]+0
            k=did SUBSEP cid
            if (cfg_n[k]>0) printf "%9.4f  ", cfg_sum[k]/cfg_n[k]
            else printf "      —   "
          }
          printf "  %9.4f\n", n[did]>0 ? sum[did]/n[did] : 0
        }
      }
    ' "$CSV" 2>/dev/null
    echo ""
    echo "--- 首请求 run1(秒) 单独 ---"
    echo "data_item_id  size         steps  config_1  config_2  config_3  config_5  config_6  config_8  config_9  config_11 config_13"
    awk -F',' '
      NR>1 && $5==1 && $6!="" && $6!="-1" {
        did=$1; size=$2; steps=$3; cid=$4; t=$6
        run1[did,cid]=t
        if (!(did in size_done)) { sz[did]=size; st[did]=steps; size_done[did]=1 }
      }
      END {
        cids="1 2 3 5 6 8 9 11 13"
        ncfg=split(cids, arr)
        for (did=1; did<=20; did++) {
          if (!(did in size_done)) next
          printf "%s  %-12s  %-5s  ", did, sz[did], st[did]
          for (i=1; i<=ncfg; i++) {
            cid=arr[i]+0
            k=did SUBSEP cid
            if (run1[k]!="") printf "%9.4f  ", run1[k]+0
            else printf "      —   "
          }
          printf "\n"
        }
      }
    ' "$CSV" 2>/dev/null
    echo ""
    echo "说明：均值仅统计 run2~5（排除首请求 warmup）；首请求(run1) 单独成表便于查看。"
  }
  _stats_content >> "$RESULT_LOG"
  _stats_content | tee -a "$STATS_LOG"
fi

echo "===== 完成。结果日志(追加): $RESULT_LOG  明细 CSV: $CSV  运行日志: $RUN_LOG  统计: ${STATS_LOG:-无} ====="
