## CSV → profiles.json 提取规则说明（prompt 工程）

**目标 JSON 结构**

- 输出 JSON 与 simulation/profile/newest_profile_A100.json 格式对齐，顶层包含 `profiles` 列表：
  - 每个 profile 元素必含：`instance_type`, `task_type`, `width`, `height`, `num_frames`, `steps`, `latency_s`
  - `latency_s` 为浮点数（秒），非 latency_ms

**CSV → 字段映射规则**

- `parallel_name` → `instance_type`
- `model` → `task_type`
  - 对 `model` 字段做不区分大小写的字符串搜索：
    - 如果包含子串 `"wan"`，则 `task_type = "video"`
    - 否则 `task_type = "image"`
- `height` → `height`
- `width` → `width`
- `num_frames` → `num_frames`
- `num_inference_steps` → `steps`
- `latency_seconds` 或 `latency_ms` → `latency_s`（秒，若有 latency_ms 则除以 1000）
- 除以上字段外，CSV 中的其它列全部忽略，不出现在 JSON 中。

**类型与缺失值约定**

- `width`, `height`, `num_frames`, `steps` 在 JSON 中以整数保存；`latency_s` 以浮点数保存。
  - 如果 CSV 中是浮点数，先转为浮点再取 `int`。
  - **若字段缺失或为空，直接报错退出**（fail-fast），不做默认填充。

**脚本和配置位置**

- 转换脚本：`simulation/tmp/csv_to_profiles.py`
  - 从配置文件中读取输入 CSV 路径与输出 JSON 路径。
- 配置文件（示例）：`simulation/tmp/csv_to_profiles.yaml`
  - 字段：
    - `input_csv`: `../../tmp/summary_runs.csv` （相对 `simulation/tmp`，指向根目录下的 `tmp/summary_runs.csv`）
    - `output_json`: `./instance_runtime_profiles.from_csv.json` （输出到 `simulation/tmp` 下）

**调用方式示例**

- 在仓库根目录下执行（需安装 `PyYAML`）：

```bash
python simulation/tmp/csv_to_profiles.py --config simulation/tmp/csv_to_profiles.yaml
```

