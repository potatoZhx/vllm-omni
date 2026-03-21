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

## 环境变量兼容

保留了旧脚本常用环境变量，主要包括：

- `BASE_CONFIG`
- `CASE_NAME`
- `RUN_TAG`
- `OUT_DIR`
- `REQUEST_RATES`
- `REQUEST_DURATION_S`
- `BENCHMARK_MODE`
- `NUM_PROMPTS_DURATION_SECONDS`
- `FIXED_NUM_PROMPTS`
- `GLOBAL_POLICY`
- `INSTANCE_POLICY`
- `ENABLE_STEP_CHUNK`
- `ENABLE_CHUNK_PREEMPTION`
- `CHUNK_BUDGET_STEPS`
- `IMAGE_CHUNK_BUDGET_STEPS`
- `VIDEO_CHUNK_BUDGET_STEPS`
- `SMALL_REQUEST_LATENCY_THRESHOLD_MS`
- `WORKER_IDS`
- `BENCHMARK_MODEL`
- `BENCHMARK_BACKEND`
- `BENCHMARK_TASK`
- `BENCHMARK_DATASET`
- `BENCHMARK_DATASET_PATH`
- `BENCHMARK_RANDOM_REQUEST_CONFIG`
- `BENCHMARK_MAX_CONCURRENCY`
- `BENCHMARK_WARMUP_REQUESTS`
- `BENCHMARK_WARMUP_NUM_INFERENCE_STEPS`
- `CASE_MATRIX`
- `SUITE_NAME`
- `OUT_ROOT`

## 默认规则

- YAML 为准
- 只有显式设置对应环境变量时，才覆盖 YAML
- 默认结果目录名从最终生成配置反推，不再硬编码策略名

## 示例

单 case：

```bash
BASE_CONFIG=./global_scheduler.qwen.yaml REQUEST_RATES=0.2,0.4 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

批量 case：

```bash
CASE_MATRIX=$'qwen_minqlen_sjf_preempt|min_queue_length|sjf|1|1|4
qwen_rr_sjf_preempt|round_robin|sjf|1|1|4' BASE_CONFIG=./global_scheduler.qwen.yaml REQUEST_RATES=0.2,0.4 benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
```
