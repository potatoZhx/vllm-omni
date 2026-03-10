# 画图模块说明（plot_results.py）

根据 diffusion 服务端压测产出的 `*.json` 画图：横轴为 RPS，纵轴为指定指标。支持单图或多图（按类型拆成 latency / throughput / slo）。



## 输入格式：（假设叫runs.json,可指定）

`runs.json` 是一个 **JSON 数组**，每个元素为一次压测的结果对象。至少需包含：

- **`rps`**（必填）：该次运行的目标 RPS，用于横轴。缺失或无效的记录会被跳过。
- 纵轴指标：如 `latency_mean`、`latency_p95`、`latency_p99`、`throughput_qps`、`slo_attainment_rate` 等（支持 snake_case 或 camelCase，如 `LatencyP95`）。

### 最小示例

```json
[
  { "rps": 0.1, "latency_mean": 6.32, "latency_p95": 6.32, "throughput_qps": 0.16, "slo_attainment_rate": 1.0 },
  { "rps": 0.5, "latency_mean": 30.3, "latency_p95": 35.2, "throughput_qps": 0.07, "slo_attainment_rate": 0.8 },
  { "rps": 1.0, "latency_mean": 65.1, "latency_p95": 72.0, "throughput_qps": 0.05, "slo_attainment_rate": 0.6 }
]
```

### 聚合规则

当 **同一 `rps` 出现多条记录**（例如重复实验）时，脚本会按 `rps` 分组，对每个指标取 **算术平均值** 后再绘图。因此图上的每个横轴点对应的是「该 RPS 下各次运行该指标的平均值」。

---

## 参数规则（必读）

- **`--output`** 与 **`--output-dir` / `--output-prefix` / `--format`** **不能同时使用**。若同时传入会报错退出，避免“悄悄忽略”造成误解。
- **单图**：未指定 `--output` 时弹窗显示、不保存；指定 `--output` 时保存到该路径。
- **多图**（`--split-by-type`）：始终保存；若指定 `--output` 则以其路径的“主干名”为前缀展开多张图，否则使用 `--output-dir` + `--output-prefix` + `--format`。

---

## 输出命名

| 模式 | 规则 | 示例 |
|------|------|------|
| 单图 | `<output-dir>/<output-prefix>.<format>` 或 `--output` 的完整路径 | `figs/benchmark.png` |
| 多图 | `<output-dir>/<output-prefix>_latency.<format>`、`_throughput.<format>`、`_slo.<format>`；若用 `--output` 则用其 stem 作前缀 | `figs/benchmark_latency.png`、`figs/benchmark_throughput.png`、`figs/benchmark_slo.png` |

多图固定为三张：**latency**、**throughput**、**slo**；SLO 图的 y 轴固定为 [0, 1.1]（避免纵坐标为 1.0 的点贴边）。

---

## 命令行用法

### 必选

- **`--input` / `-i`**：输入的 `runs.json` 路径。

### 输出（互斥）

- **`--output` / `-o`**：完整输出文件路径。单图直接使用；多图时用其 stem 展开（如 `figs/qwen.png` → `figs/qwen_latency.png` 等）。
- **`--output-dir`**：输出目录，默认 `.`。**不能与 `--output` 同时使用**。
- **`--output-prefix`**：输出文件名前缀，默认 `benchmark`。**不能与 `--output` 同时使用**。
- **`--format`**：图片格式（png / pdf / svg 等），默认 `png`。**不能与 `--output` 同时使用**。
- **`--dpi`**：保存图片 DPI，默认 150。

### 绘图

- **`--metrics` / `-m`**：纵轴指标，可多个。默认单图为 `LatencyP95`；多图时未指定则用各组默认（latency: Mean/P95/P99，throughput: QPS，slo: slo_attainment_rate）。示例：`-m LatencyP95 LatencyP99` 或 `-m "LatencyP95,ThroughputQps"`。
- **`--split-by-type`**：按类型拆成多图（latency / throughput / slo）。
- **`--title`**：图标题；多图时自动追加 `- Latency`、`- Throughput`、`- SLO`。
- **`--list-metrics`**：列出当前 `runs.json` 中所有记录出现的字段并集后退出，便于选择 `--metrics`。

---

## 示例

```bash
# 单图，弹窗显示（不保存）
python3 plot_results.py -i runs.json -m LatencyP99

# 单图，保存到指定文件
python3 plot_results.py -i runs.json -o figs/benchmark.png

# 多图，用完整路径作前缀（生成 figs/benchmark_latency.png 等）
python3 plot_results.py -i runs.json -o figs/benchmark.png --split-by-type

# 多图，用目录 + 前缀（生成 figs/qwen_image_latency.png 等）
python3 plot_results.py -i runs.json --output-dir figs --output-prefix qwen_image --split-by-type

# 指定 DPI 与格式
python3 plot_results.py -i runs.json --output-dir figs --output-prefix report --format pdf --dpi 200 --split-by-type

# 查看可用指标
python3 plot_results.py -i runs.json --list-metrics
```

---

## Python API

```python
from plot_results import plot_results, list_metrics

# 单图保存
path = plot_results("runs.json", output="figs/benchmark.png")

# 多图保存（返回路径列表）
paths = plot_results("runs.json", output_dir="figs", output_prefix="qwen", split_by_type=True)

# 仅弹窗显示
plot_results("runs.json", metrics=["LatencyP95", "LatencyP99"])

# 列出指标
keys = list_metrics("runs.json")
```

**返回值**：单图保存返回 `str`，多图保存返回 `list[str]`，仅显示不保存返回 `None`。

**参数**：`input_path`, `output=None`, `output_dir="."`, `output_prefix="benchmark"`, `fmt="png"`, `dpi=150`, `metrics=None`, `title="Diffusion serving benchmark"`, `split_by_type=False`。`output` 与 `output_dir`/`output_prefix`/`fmt` 语义互斥，调用时只应指定其一。
