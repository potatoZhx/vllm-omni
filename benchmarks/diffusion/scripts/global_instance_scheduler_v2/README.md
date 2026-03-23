# Global + Instance Scheduler V2

这一套脚本是对旧版 `benchmarks/diffusion/scripts/run_global_instance_scheduler_*.sh` 的重写版本，目标是：

- 保持功能等价
- 把 bash 压到最薄
- 把配置生成、scheduler 启停、benchmark 调度集中到一个 Python 入口
- 避免 bash heredoc Python 和过长调用链

## 文件

- `run_case.sh`
  - 单 case 入口
- `run_suite.sh`
  - 多 case 批量入口
- `orchestrate.py`
  - 真正的实现入口
  - 子命令：`case` / `suite`

## 调用链

旧版：

- `run_global_instance_scheduler_rps_bench.sh`
- `run_global_instance_scheduler_case.sh`
- `run_global_scheduler_benchmark_one_shell.sh`
- `run_global_scheduler_benchmark.sh`
- `diffusion_benchmark_serving.py`

新版：

- `run_suite.sh` 或 `run_case.sh`
- `orchestrate.py`
- `diffusion_benchmark_serving.py`

`orchestrate.py` 直接负责：

- 生成 `global_scheduler.generated.yaml`
- 启动 scheduler
- 等待 scheduler / worker ready
- 逐 RPS 调 benchmark
- 汇总 `summary.json` / `summary.csv`

## 功能范围

当前支持：

- 单 case 运行
- 多 case 批量运行
- 全局调度策略覆盖
- 实例内调度策略覆盖
- step chunk / chunk preemption / chunk budget 覆盖
- benchmark 模型、数据集、随机请求配置、warmup 请求配置覆盖
- YAML 优先，显式环境变量覆盖
- 自动生成结果目录和 summary 汇总

## Benchmark 模式

当前支持两种 benchmark 模式。

### 1. 固定总发送时间

设置：

- `BENCHMARK_MODE=fixed_duration`
- `NUM_PROMPTS_DURATION_SECONDS=<秒数>`

语义：

- 每个 RPS 档的总发送时长固定
- 每个 RPS 档的请求数按 `ceil(rps * duration_seconds)` 自动计算

示例：

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml BENCHMARK_MODE=fixed_duration NUM_PROMPTS_DURATION_SECONDS=600 REQUEST_RATES=0.2,0.4,0.6 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

### 2. 固定总请求数

设置：

- `BENCHMARK_MODE=fixed_num_prompts`
- `FIXED_NUM_PROMPTS=<请求数>`

语义：

- 每个 RPS 档发送相同数量的请求
- 总持续时间随 `request_rate` 自动变化

示例：

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml BENCHMARK_MODE=fixed_num_prompts FIXED_NUM_PROMPTS=100 REQUEST_RATES=0.2,0.4,0.6 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

## 常用环境变量

保留了旧脚本常用环境变量，主要包括：

### 基础运行

- `BASE_CONFIG`
- `CASE_NAME`
- `RUN_TAG`
- `OUT_DIR`
- `REQUEST_RATES`
- `REQUEST_DURATION_S`
- `BENCHMARK_MODE`
- `NUM_PROMPTS_DURATION_SECONDS`
- `FIXED_NUM_PROMPTS`

### 调度策略

- `GLOBAL_POLICY`
- `INSTANCE_POLICY`
- `ENABLE_STEP_CHUNK`
- `ENABLE_CHUNK_PREEMPTION`
- `CHUNK_BUDGET_STEPS`
- `IMAGE_CHUNK_BUDGET_STEPS`
- `VIDEO_CHUNK_BUDGET_STEPS`
- `SMALL_REQUEST_LATENCY_THRESHOLD_MS`

### Benchmark / 数据集

- `WORKER_IDS`
- `BENCHMARK_MODEL`
- `BENCHMARK_BACKEND`
- `BENCHMARK_TASK`
- `BENCHMARK_DATASET`
- `BENCHMARK_DATASET_PATH`
- `BENCHMARK_RANDOM_REQUEST_CONFIG`
- `BENCHMARK_WARMUP_REQUEST_CONFIG`
- `BENCHMARK_MAX_CONCURRENCY`
- `BENCHMARK_WARMUP_REQUESTS`
- `BENCHMARK_WARMUP_NUM_INFERENCE_STEPS`

### 批量 case

- `CASE_MATRIX`
- `SUITE_NAME`
- `OUT_ROOT`

## 默认规则

- YAML 为准
- 只有显式设置对应环境变量时，才覆盖 YAML
- 默认结果目录名从最终生成配置反推，不再硬编码策略名

## CASE_MATRIX 格式

`suite` 模式下，每一行格式为：

```text
case_name|global_policy|instance_policy|enable_step_chunk|enable_chunk_preemption|chunk_budget_steps
```

示例：

```text
qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4
qwen_rr_sjf_preempt|round_robin|sjf|1|1|4
```

## 示例

### 单 case

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml REQUEST_RATES=0.2,0.4 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

### 单 case，显式覆盖策略

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml GLOBAL_POLICY=round_robin INSTANCE_POLICY=sjf ENABLE_STEP_CHUNK=1 ENABLE_CHUNK_PREEMPTION=1 CHUNK_BUDGET_STEPS=4 REQUEST_RATES=0.2,0.4 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

### 单 case，显式指定 warmup request config

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml \
REQUEST_RATES=0.2,0.4 \
BENCHMARK_WARMUP_REQUESTS=4 \
BENCHMARK_WARMUP_REQUEST_CONFIG='[{"width":512,"height":512,"num_inference_steps":20,"weight":0.15},{"width":768,"height":768,"num_inference_steps":20,"weight":0.25},{"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},{"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}]' \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

语义：

- `BENCHMARK_WARMUP_REQUEST_CONFIG` 会直接透传到 `diffusion_benchmark_serving.py --warmup-request-config`。
- 如果 base YAML 里的 `benchmark.warmup_request_config` 已经配置了，环境变量会覆盖它。

### 批量 case

```bash
CASE_MATRIX=$'qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4
qwen_rr_sjf_preempt|round_robin|sjf|1|1|4' BASE_CONFIG=./global_scheduler.qwen.yaml REQUEST_RATES=0.2,0.4 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
```
