# 从 YAML 加载配置并 export 为环境变量（供 switch_time 下各脚本 source）
# 用法: source .../load_config.sh && load_config "config/common_env.yaml" "config/run_switch_parallel_qwen.yaml"（先公共环境，再脚本 yaml）
# 要求: python3 且能 import yaml（pip install pyyaml）

load_config() {
  local env_file="$1"
  local script_file="$2"
  local script_dir="${SCRIPT_DIR:-.}"
  local f

  for f in "$env_file" "$script_file"; do
    [ -z "$f" ] && continue
    [ -f "$script_dir/$f" ] || continue
    eval "$(python3 - "$script_dir/$f" << 'PY'
import yaml
import sys
import os

path = sys.argv[1]
if not os.path.isfile(path):
    sys.exit(0)
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
if not data:
    sys.exit(0)

def esc(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "\\\\").replace("'", "'\"'\"'")
    return s

for k, v in data.items():
    key = k.upper().replace("-", "_")
    if v is None:
        print("export %s=\"\"" % key)
        continue
    if isinstance(v, bool):
        print("export %s=\"%s\"" % (key, "1" if v else "0"))
        continue
    if isinstance(v, list):
        if v and isinstance(v[0], dict):
            import json
            j = json.dumps(v, ensure_ascii=False)
            j_esc = j.replace("\\", "\\\\").replace("'", "'\"'\"'")
            print("export CONFIGS_JSON='%s'" % j_esc)
            continue
        val = "\n".join(str(x) for x in v)
        val_esc = esc(val)
        print("export %s='%s'" % (key, val_esc))
        continue
    val_esc = esc(v)
    print("export %s=\"%s\"" % (key, val_esc))
PY
)"
  done
}

# 应用公共环境（HF_*、CONDA_ENV、REPO_DIR）；假定环境已配置好，不再执行 setup_script / module
apply_env() {
  export PYTHONUNBUFFERED=1
  [ -n "${HF_ENDPOINT:-}" ] && export HF_ENDPOINT
  if [ -z "${HF_HOME:-}" ]; then
    _G=$(groups 2>/dev/null | awk '{print $1}')
    _U=$(whoami)
    if [ -n "$_G" ] && [ -d "/data2/$_G/$_U" ] && [ -w "/data2/$_G/$_U" ]; then
      export HF_HOME="/data2/$_G/$_U/xhf/hf_cache"
    else
      export HF_HOME="${HOME:-/tmp}/xhf/hf_cache"
    fi
  fi
  if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
  fi
}
