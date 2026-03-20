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
#SBATCH --export=ALL
#SBATCH -t 72:00:00

# -e: 命令非零退出则立即退出; -u: 使用未定义变量报错; -o pipefail: 管道中任一段失败则整体失败
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -z "${HF_HOME:-}" ]; then
  _G=$(groups 2>/dev/null | awk '{print $1}')
  _U=$(whoami)
  if [ -n "$_G" ] && [ -d "/data2/$_G/$_U" ] && [ -w "/data2/$_G/$_U" ]; then
    export HF_HOME="/data2/$_G/$_U/xhf/hf_cache"
  else
    export HF_HOME="${HF_HOME:-$HOME/xhf/hf_cache}"
  fi
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  WORK_DIR="${SLURM_SUBMIT_DIR}"
else
  WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
SCRIPT_DIR="$WORK_DIR"
REPO_DIR="${REPO_DIR:-$(cd "$WORK_DIR/.." && pwd)}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

# 可选：从 YAML 读取参数（支持 default/random 两种 data_items 模式）
# - CONFIG_FILE: 指定脚本配置 yaml（相对 switch_time 目录），默认 config/test_qwen_profile.yaml
# - DATA_ITEMS_MODE: 覆盖 yaml 里的 data_items_mode（random/default）
CONFIG_FILE="${CONFIG_FILE:-config/test_qwen_profile.yaml}"
DATA_ITEMS_MODE_OVERRIDE="${DATA_ITEMS_MODE:-}"
DATASET_MODE_OVERRIDE="${DATASET_MODE:-}" # 兼容旧字段
if [ -f "$SCRIPT_DIR/load_config.sh" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/load_config.sh"
  # 先加载公共环境，再加载脚本 yaml（两者 key 会合并，脚本 yaml 覆盖公共环境）
  load_config "config/common_env.yaml" "$CONFIG_FILE" || true
  apply_env || true
fi

# 模型：优先使用环境变量 MODEL；未设置时，如果本地路径存在则用本地模型，否则回退到 HuggingFace 名称
if [ -z "${MODEL:-}" ] && [ -d "/data2/group_谈海生/xhf/xhf/Qwen-Image" ]; then
  MODEL="/data2/group_谈海生/xhf/xhf/Qwen-Image"
else
  MODEL="${MODEL:-Qwen/Qwen-Image}"
fi
PORT="${PORT:-8101}"
BASE_URL="http://127.0.0.1:${PORT}"
REQUESTS_PER_CONFIG="${REQUESTS_PER_CONFIG:-5}"

# profile.md：正负 prompt 固定
PROMPT="${PROMPT:-A realistic photo of a close-up of a violin on velvet fabric, warm studio lighting, fine wood grain}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-low quality, blurry}"
# CFG scale（仅 CFG=2 时生效，确保 true CFG 真的启用）
CFG_SCALE="${CFG_SCALE:-4.0}"

# profile.md：并行测试表_qwen.md 中的 16 个配置（编号 1..16）
CONFIG_IDS="${CONFIG_IDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}"
# 仅 CFG>1 的配置才在请求中带 negative_prompt（即 CFG=2 的配置：5, 6, 7, 11, 12, 15）
CONFIGS_WITH_CFG="${CONFIGS_WITH_CFG:-5 6 7 11 12 15}"

# data_items_mode：random/default。优先使用环境变量覆盖，否则使用 yaml 的 data_items_mode；
# 兼容旧字段 DATASET_MODE（random/default）
if [ -n "${DATA_ITEMS_MODE_OVERRIDE:-}" ]; then
  DATA_ITEMS_MODE="$DATA_ITEMS_MODE_OVERRIDE"
fi
if [ -z "${DATA_ITEMS_MODE:-}" ] && [ -n "${DATASET_MODE_OVERRIDE:-}" ]; then
  DATA_ITEMS_MODE="$DATASET_MODE_OVERRIDE"
fi
DATA_ITEMS_MODE="${DATA_ITEMS_MODE:-${DATASET_MODE:-random}}"

# 数据项：
# - 新配置：从 YAML 的 RANDOM_DATA_ITEMS / DEFAULT_DATA_ITEMS 选择
# - 兼容旧配置：若 YAML 仍导出 DATA_ITEMS，则直接用
# - 兜底：脚本内置 default/random 列表
if [ -z "${DATA_ITEMS:-}" ]; then
  if [ "$DATA_ITEMS_MODE" = "default" ] && [ -n "${DEFAULT_DATA_ITEMS:-}" ]; then
    DATA_ITEMS="$DEFAULT_DATA_ITEMS"
  elif [ "$DATA_ITEMS_MODE" = "random" ] && [ -n "${RANDOM_DATA_ITEMS:-}" ]; then
    DATA_ITEMS="$RANDOM_DATA_ITEMS"
  else
    if [ "$DATA_ITEMS_MODE" = "random" ]; then
      DATA_ITEMS="512x512 20
768x768 20
1024x1024 25
1536x1536 35"
    else
      # default：5 分辨率 × 5 steps(1,5,10,30,50) = 25 种
      DATA_ITEMS="128x128 1
128x128 5
128x128 10
128x128 30
128x128 50
256x256 1
256x256 5
256x256 10
256x256 30
256x256 50
512x512 1
512x512 5
512x512 10
512x512 30
512x512 50
1024x1024 1
1024x1024 5
1024x1024 10
1024x1024 30
1024x1024 50
1536x1536 1
1536x1536 5
1536x1536 10
1536x1536 30
1536x1536 50"
    fi
  fi
fi

# 数据项总数：与 DATA_ITEMS 行数一致
NUM_DATA_ITEMS="$(echo "$DATA_ITEMS" | awk 'NF>=2{c++} END{print c+0}')"

CONDA_ENV="${CONDA_ENV:-vllm_omni}"
JOBID="${SLURM_JOB_ID:-local}"
ts=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${LOG_DIR}/qwen_profile_run_${JOBID}_${ts}.log"
RESULT_LOG="${LOG_DIR}/qwen_profile_result_${JOBID}_${ts}.log"
CSV="${LOG_DIR}/qwen_profile_${JOBID}_${ts}.csv"
JSON_OUT="${LOG_DIR}/qwen_profile_${JOBID}_${ts}.json"
SIM_JSON_PATH="${REPO_DIR}/simulation/tmp/tmp.json"
# 默认不写 simulation/tmp/tmp.json，避免覆盖；需要时设 WRITE_SIM_JSON=1
WRITE_SIM_JSON="${WRITE_SIM_JSON:-0}"

START_DATA_ITEM="${START_DATA_ITEM:-${START_DATA_ITEM:-1}}"
START_CONFIG="${START_CONFIG:-${START_CONFIG:-1}}"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "===== Qwen-Image 文生图性能画像（profile.md）====="
num_configs=$(echo $CONFIG_IDS | wc -w)
echo "data_items_mode=${DATA_ITEMS_MODE}  数据项：${NUM_DATA_ITEMS} 种，配置项：${num_configs} 个，每 (数据项, 配置) 发 ${REQUESTS_PER_CONFIG} 次请求"
echo "REPO_DIR=$REPO_DIR  PORT=$PORT"
echo "CSV=$CSV  JSON_OUT=$JSON_OUT  RESULT_LOG(追加)=$RESULT_LOG"

# 直接使用已有的 conda 环境（vllm_omni），不再依赖 module 命令
# 若存在 setup_env.sh 且未设置 SKIP_SETUP_ENV=1，则执行；不存在则跳过
if [ "${SKIP_SETUP_ENV:-0}" != "1" ] && [ -f "$WORK_DIR/setup_env.sh" ]; then
  bash "$WORK_DIR/setup_env.sh"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
cd "$REPO_DIR"
SKIP_BENCH=0
python3 -c "import vllm_omni" || { echo "ERROR: vllm_omni not importable，跳过全部测试，仅正常结束脚本."; SKIP_BENCH=1; }

# 仅变更待测配置项（SP/CFG/TP），其余特性一律使用默认（如 cache、cpu-offload、FP8 等均不显式传入）
# 与《并行测试表_qwen.md》/run_switch_parallel_qwen.sh 一致的 16 个配置（1..16）
get_config_params() {
  local id="$1"
  local gpu sp cfg tp
  case "$id" in
    1)  gpu=1; sp=1; cfg=1; tp=1 ;;  # 1=1×1×1
    2)  gpu=2; sp=1; cfg=1; tp=2 ;;  # 2=1×1×2
    3)  gpu=4; sp=1; cfg=1; tp=4 ;;  # 4=1×1×4
    4)  gpu=8; sp=1; cfg=1; tp=8 ;;  # 8=1×1×8
    5)  gpu=2; sp=1; cfg=2; tp=1 ;;  # 2=1×2×1
    6)  gpu=4; sp=1; cfg=2; tp=2 ;;  # 4=1×2×2
    7)  gpu=8; sp=1; cfg=2; tp=4 ;;  # 8=1×2×4
    8)  gpu=2; sp=2; cfg=1; tp=1 ;;  # 2=2×1×1
    9)  gpu=4; sp=2; cfg=1; tp=2 ;;  # 4=2×1×2
    10) gpu=8; sp=2; cfg=1; tp=4 ;;  # 8=2×1×4
    11) gpu=4; sp=2; cfg=2; tp=1 ;;  # 4=2×2×1
    12) gpu=8; sp=2; cfg=2; tp=2 ;;  # 8=2×2×2
    13) gpu=4; sp=4; cfg=1; tp=1 ;;  # 4=4×1×1
    14) gpu=8; sp=4; cfg=1; tp=2 ;;  # 8=4×1×2
    15) gpu=8; sp=4; cfg=2; tp=1 ;;  # 8=4×2×1
    16) gpu=8; sp=8; cfg=1; tp=1 ;;  # 8=8×1×1
    *)  echo "ERROR: invalid config id $id (only 1..16, see 并行测试表_qwen.md)"; return 1 ;;
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

# 判断配置是否使用 CFG（需要传 negative_prompt），与 CONFIGS_WITH_CFG 一致
config_has_cfg() {
  local id="$1"
  echo " $CONFIGS_WITH_CFG " | grep -q " ${id} "
}

start_config_id() {
  local id="$1"
  local port="$2"
  local log="$3"
  eval "$(get_config_params "$id")"
  setsid vllm serve "$MODEL" --omni --port "$port" --vae-use-slicing --vae-use-tiling $CONFIG_EXTRA >> "$log" 2>&1 &
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
  lsof -t -i ":${port}" 2>/dev/null | while read -r p; do kill -9 "$p" 2>/dev/null || true; done
  sleep 2
}

wait_port_released() {
  local port="$1"
  local max_wait=800
  local i=0
  while [ $i -lt "$max_wait" ]; do
    if ! ss -ltnp 2>/dev/null | grep -q ":${port} "; then return 0; fi
    sleep 1
    i=$((i + 1))
    [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ] && echo "    wait_port_released ${i}s..." >&2
  done
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内未释放，强制清理后不退出，继续下一配置。" >&2
  cleanup_gpu_residuals
  force_kill_port "$port"
  cleanup_gpu_residuals
  return 1
}

# 发单次文生图请求，输出耗时（秒，保留 4 位小数以准确统计亚秒级请求），失败输出 -1
# 用法：send_one_request "256x256" 50 1 表示 size=256x256 steps=50 且带 negative_prompt（第3参数为 1）
send_one_request() {
  local size="$1"
  local steps="$2"
  local use_neg="${3:-0}"
  local json
  if [ "$use_neg" = "1" ]; then
    json=$(printf '{"prompt":"%s","negative_prompt":"%s","true_cfg_scale":%s,"size":"%s","num_inference_steps":%s,"n":1}' "$PROMPT" "$NEGATIVE_PROMPT" "$CFG_SCALE" "$size" "$steps")
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

# steps 归一化（用于覆盖 random/trace 中的 steps 区间）
normalize_steps() {
  local s="$1"
  if [ -z "$s" ]; then
    echo ""
    return 0
  fi
  if [ "$s" -ge 4 ] 2>/dev/null && [ "$s" -le 6 ] 2>/dev/null; then
    echo "5"
    return 0
  fi
  echo "$s"
}

# 从当前 CSV 生成 profiles JSON（run2~5 均值→latency_ms）；可多次调用以支持增量进度
write_profiles_json() {
  [ -f "$CSV" ] || return 0
  python3 - "$CSV" "$JSON_OUT" "$SIM_JSON_PATH" "$WRITE_SIM_JSON" << 'PY'
import csv
import json
import os
import sys
from collections import defaultdict

csv_path = sys.argv[1]
json_out = sys.argv[2]
sim_path = sys.argv[3]
write_sim = sys.argv[4] if len(sys.argv) > 4 else "1"

def parse_size(s: str):
    w, h = s.lower().split("x", 1)
    return int(w), int(h)

cfg_map = {
    1:  (1, 1, 1),
    2:  (1, 1, 2),
    3:  (1, 1, 4),
    4:  (1, 1, 8),
    5:  (1, 2, 1),
    6:  (1, 2, 2),
    7:  (1, 2, 4),
    8:  (2, 1, 1),
    9:  (2, 1, 2),
    10: (2, 1, 4),
    11: (2, 2, 1),
    12: (2, 2, 2),
    13: (4, 1, 1),
    14: (4, 1, 2),
    15: (4, 2, 1),
    16: (8, 1, 1),
}

def instance_type_from_id(config_id: int) -> str:
    sp, cfg, tp = cfg_map.get(config_id, (None, None, None))
    if sp is None:
        return str(config_id)
    return f"sp{sp}_cfg{cfg}_tp{tp}"

times = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            run = int(row["run"])
            if run <= 1:
                continue
            t = float(row["request_time_s"])
            if t < 0:
                continue
            key = (row["size"], int(row["steps"]), int(row["config_id"]))
            times[key].append(t)
        except Exception:
            continue

profiles = []
for (size, steps, config_id), arr in sorted(times.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    if not arr:
        continue
    mean_s = sum(arr) / len(arr)
    latency_ms = int(round(mean_s * 1000.0))
    w, h = parse_size(size)
    profiles.append(
        {
            "instance_type": instance_type_from_id(config_id),
            "task_type": "image",
            "width": w,
            "height": h,
            "num_frames": 1,
            "steps": int(steps),
            "latency_ms": latency_ms,
        }
    )

out_obj = {"profiles": profiles}
os.makedirs(os.path.dirname(json_out), exist_ok=True)
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(out_obj, f, ensure_ascii=False, indent=2)

if str(write_sim) == "1":
    os.makedirs(os.path.dirname(sim_path), exist_ok=True)
    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
PY
}

if [ "$SKIP_BENCH" != "1" ]; then
# CSV 表头
echo "data_item_id,size,steps,config_id,run,request_time_s" >> "$CSV"

# 顺序：先按配置，再按数据项。每个配置只启动一次服务，测完该配置下全部数据项后再停服务，减少启停次数。
for config_id in $CONFIG_IDS; do
  if [ "$config_id" -lt "$START_CONFIG" ] 2>/dev/null; then continue; fi

  SERVER_LOG="${LOG_DIR}/qwen_profile_server_c${config_id}_${JOBID}_${ts}.log"
  pgid=$(start_config_id "$config_id" "$PORT" "$SERVER_LOG")
  echo "========== 配置 $config_id：启动 pgid=$pgid，将测全部 ${NUM_DATA_ITEMS} 个数据项 =========="
  wait_poll=$(wait_ready "$BASE_URL" 7200) || wait_poll="-1"
  if [ "$wait_poll" = "-1" ]; then
    echo "  [配置 $config_id] 未就绪，跳过本配置"
    kill -KILL -"$pgid" 2>/dev/null || true
    force_kill_port "$PORT"
    cleanup_gpu_residuals
    wait_port_released "$PORT" 2>/dev/null || true
    continue
  fi
  echo "  [配置 $config_id] 就绪 ready_poll=${wait_poll}s，开始遍历数据项并发请求"

  use_neg=0
  config_has_cfg "$config_id" && use_neg=1

  data_item_id=0
  while read -r size steps; do
    [ -z "$size" ] && continue
    steps="$(normalize_steps "$steps")"
    data_item_id=$((data_item_id + 1))
    if [ "$data_item_id" -lt "$START_DATA_ITEM" ]; then continue; fi

    echo "  --- 数据项 $data_item_id / ${NUM_DATA_ITEMS}  size=$size steps=$steps ---"
    run=1
    while [ "$run" -le "$REQUESTS_PER_CONFIG" ]; do
      tt=$(send_one_request "$size" "$steps" "$use_neg")
      echo "    run $run / $REQUESTS_PER_CONFIG  request_time_s=$tt"
      echo "$data_item_id,$size,$steps,$config_id,$run,$tt" >> "$CSV"
      run=$((run + 1))
    done
  done <<< "$DATA_ITEMS"

  stop_server_and_measure "$pgid" >/dev/null
  echo "  [配置 $config_id] 已按 pgid 停服，接下来清理端口并杀死本用户在 GPU 上的全部进程..."
  force_kill_port "$PORT" || true
  cleanup_gpu_residuals || true
  sleep 2
  cleanup_gpu_residuals || true
  wait_port_released "$PORT" || true
  sleep 2
  write_profiles_json
  echo "  [配置 $config_id] 已写入/更新 JSON: $JSON_OUT"
done

# 全配置跑完后，按数据项整理并追加到结果日志（与原先「每数据项各配置」汇总格式一致）
data_item_id=0
while read -r size steps; do
  [ -z "$size" ] && continue
  steps="$(normalize_steps "$steps")"
  data_item_id=$((data_item_id + 1))
  if [ "$data_item_id" -lt "$START_DATA_ITEM" ]; then continue; fi

  {
    echo "--- 数据项 $data_item_id 完成 $(date -Iseconds 2>/dev/null || date) ---"
    echo "  size=$size steps=$steps"
    echo "  各配置：首请求(run1) 单独列出，均值仅 run2~5(秒)："
    awk -F',' -v did="$data_item_id" -v cids="$CONFIG_IDS" 'NR>1 && $1==did && $6!="" && $6!="-1" {
      if ($5==1) { run1[$4]=$6 }
      else { sum[$4]+=$6; n[$4]++ }
    }
    END {
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
done <<< "$DATA_ITEMS"
echo "  已按数据项汇总并追加到 $RESULT_LOG"

# 全量跑完后追加：请求完成平均时间统计表（均值仅 run2~5，首请求 run1 单独表）
if [ -f "$CSV" ]; then
  STATS_LOG="${LOG_DIR}/qwen_profile_${JOBID}_${ts}_stats.log"
  _stats_content() {
    echo "===== 请求完成平均时间统计（均值仅 run2~5，首请求 run1 单独列出；精度 4 位小数）====="
    echo "--- 均值(仅 run2~5)(秒) ---"
    _h="data_item_id  size         steps"
    for c in $CONFIG_IDS; do _h="$_h  config_$c"; done
    echo "$_h   平均(秒)"
    awk -F',' -v nitems="$NUM_DATA_ITEMS" -v cids="$CONFIG_IDS" '
      NR>1 && $6!="" && $6!="-1" && $5>1 {
        did=$1; size=$2; steps=$3; cid=$4; t=$6
        sum[did]+=t; n[did]++
        cfg_sum[did,cid]+=t; cfg_n[did,cid]++
        if (!(did in size_done)) { sz[did]=size; st[did]=steps; size_done[did]=1 }
      }
      END {
        ncfg=split(cids, arr)
        for (did=1; did<=nitems; did++) {
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
    _h="data_item_id  size         steps"
    for c in $CONFIG_IDS; do _h="$_h  config_$c"; done
    echo "$_h"
    awk -F',' -v nitems="$NUM_DATA_ITEMS" -v cids="$CONFIG_IDS" '
      NR>1 && $5==1 && $6!="" && $6!="-1" {
        did=$1; size=$2; steps=$3; cid=$4; t=$6
        run1[did,cid]=t
        if (!(did in size_done)) { sz[did]=size; st[did]=steps; size_done[did]=1 }
      }
      END {
        ncfg=split(cids, arr)
        for (did=1; did<=nitems; did++) {
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

  write_profiles_json
  echo "===== 完成。结果日志(追加): $RESULT_LOG  明细 CSV: $CSV  JSON: $JSON_OUT  运行日志: $RUN_LOG  统计: ${STATS_LOG:-无} ====="
else
  echo "===== 完成（未运行测试：vllm_omni 不可用）。结果日志: $RESULT_LOG  运行日志: $RUN_LOG ====="
fi
