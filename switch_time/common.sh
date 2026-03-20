# switch_time 公共函数库：服务就绪检测、停服、端口清理、参数生成与启动
# 用法：在脚本中 source "$SCRIPT_DIR/common.sh"（需已设置 SCRIPT_DIR）
# 依赖：CONFIGS_JSON（由 load_config 从 yaml configs 表加载）、MODEL（由各脚本或 yaml 设置）

# 轮询 /v1/models 直到 200 或超时，输出等待秒数，失败输出 -1
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

# 按进程组停服并输出停服耗时（秒）
stop_server_and_measure() {
  local pgid="$1"
  [ -z "$pgid" ] && return
  local T0 T1
  T0=$(date +%s.%N)
  kill -TERM -"$pgid" 2>/dev/null || true
  local i=0
  while [ $i -lt 60 ]; do
    if ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
      break
    fi
    sleep 0.5
    i=$((i + 1))
  done
  if ! ps -eo pgid=,pid= 2>/dev/null | awk -v p="$pgid" '$1+0==p+0 {c++} END {exit (c>0)?1:0}'; then
    kill -KILL -"$pgid" 2>/dev/null || true
    local j=0
    while [ $j -lt 25 ]; do
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

# 清理本用户在 GPU 上的残留进程
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

# 按端口强制杀进程
force_kill_port() {
  local port="$1"
  fuser -k "${port}/tcp" 2>/dev/null || true
  lsof -t -i ":${port}" 2>/dev/null | while read -r p; do kill -9 "$p" 2>/dev/null || true; done
  sleep 2
}

# 等待端口释放（ss 无 LISTEN），超时则 force_kill_port 并 return 1（不 exit，由调用方 || true 决定）
wait_port_released() {
  local port="$1"
  local max_wait=800
  local i=0
  while [ $i -lt "$max_wait" ]; do
    if ! ss -ltnp 2>/dev/null | grep -q ":${port} "; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
    [ $((i % 30)) -eq 0 ] && [ $i -gt 0 ] && echo "    wait_port_released ${i}s..." >&2
  done
  echo "ERROR: 端口 ${port} 在 ${max_wait}s 内未释放，强制清理后 return 1，不退出脚本。" >&2
  cleanup_gpu_residuals
  force_kill_port "$port"
  cleanup_gpu_residuals
  return 1
}

# 打印错误摘要与相关日志末尾
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

# 从 CONFIGS_JSON 按 id 生成 CUDA_VISIBLE_DEVICES 与 CONFIG_EXTRA（兼容 qwen 型 sp/cfg/tp 与 wan 型 ulysses/ring/hsdp）
get_config_params() {
  local id="$1"
  python3 - "$id" << 'PY'
import json, os, sys
raw = os.environ.get("CONFIGS_JSON", "[]")
configs = json.loads(raw) if raw else []
cid = int(sys.argv[1])
c = next((x for x in configs if x.get("id") == cid), None)
if not c:
    print("echo ERROR invalid config id %s（见 yaml configs 表）>&2" % cid, file=sys.stderr)
    sys.exit(1)
gpu = int(c.get("gpu", 1))
devs = "0"
if gpu >= 2: devs = "0,1"
if gpu >= 4: devs = "0,1,2,3"
if gpu >= 8: devs = "0,1,2,3,4,5,6,7"
extra = []
# Qwen 用 sp，Wan 用 ulysses
if "ulysses" in c and int(c.get("ulysses", 1)) > 1:
    extra.append("--ulysses-degree %s" % c["ulysses"])
elif "sp" in c and int(c.get("sp", 1)) > 1:
    extra.append("--ulysses-degree %s" % c["sp"])
if "ring" in c and int(c.get("ring", 1)) > 1:
    extra.append("--ring %s" % c["ring"])
if int(c.get("cfg", 1)) > 1:
    extra.append("--cfg-parallel-size %s" % c["cfg"])
if int(c.get("tp", 1)) > 1:
    extra.append("--tensor-parallel-size %s" % c["tp"])
if int(c.get("use_hsdp", 0)):
    extra.append("--use-hsdp --hsdp-shard-size %s --hsdp-replicate-size %s" % (c.get("hsdp_shard", 1), c.get("hsdp_rep", 1)))
config_extra = " ".join(extra)
print('export CUDA_VISIBLE_DEVICES="%s"; CONFIG_EXTRA="%s"' % (devs, config_extra))
PY
}

# 启动指定 config id 的 vllm 服务，输出 pgid（或 leader pid）；失败 return 1
start_config_id() {
  local id="$1"
  local port="$2"
  local log="$3"
  eval "$(get_config_params "$id")" || return 1
  setsid vllm serve "$MODEL" --omni --port "$port" --vae-use-slicing --vae-use-tiling $CONFIG_EXTRA >> "$log" 2>&1 &
  local leader_pid=$!
  local pgid
  pgid=$(ps -o pgid= -p "$leader_pid" 2>/dev/null | tr -d ' ')
  if [ -z "$pgid" ]; then
    echo "$leader_pid"
  else
    echo "$pgid"
  fi
}
