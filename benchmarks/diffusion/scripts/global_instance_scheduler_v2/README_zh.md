# Global Scheduler Benchmark Orchestrator 中文说明

该目录包含 `v18-base` 上 `global scheduler` 迁移后的精简 benchmark /
orchestration 入口。

这个 orchestrator 的职责很明确：启动 global scheduler，等待 worker 变成
可路由且 API 就绪，然后把
`benchmarks/diffusion/diffusion_benchmark_serving.py`
指向 scheduler URL 运行。

## 当前范围

当前支持：

- 全局路由策略：
  - `min_queue_length`
  - `round_robin`
  - `short_queue_runtime`
- worker 启动参数归一化：
  - 同时支持 `request_scheduler` 和 `step_level_request_scheduler`
  - 只有在选中 `step_level_request_scheduler` 时才注入 `--diffusion-enable-step-chunk`
- 通过 `benchmark.warmup_request_config` 构造 benchmark warmup
- 支持单 case 和 suite 两种运行方式

当前明确不做：

- scheduler 侧等待 / admission blocking
- `diffusion_engine_max_concurrency`
- orchestration 对 `chunk_preemption` / `chunk_budget` 的依赖
- 实例内复杂策略，例如 `sjf`、`p95-first`、`guarded`、`fusion`

关键运行语义：

- 这轮迁移里的 global scheduler 是 `pure routing`
- 它不是 scheduler 侧等待队列
- 请求会被立即路由到某个 worker
- worker 内部是否等待、如何执行，仍由 worker 自己决定

## 目录文件

- `run_case.sh`：执行单个 case
- `run_suite.sh`：执行一组 case 并汇总结果
- `orchestrate.py`：主实现
- `single_instance.qwen.yaml`：单 worker 的 Qwen 图像模板
- `single_instance.wan2_2.yaml`：单 worker 的 Wan2.2 视频模板
- `README.md`：英文版说明

## 执行流程

每个 case 的执行过程如下：

1. 读取一份基础 YAML 配置
2. 套用环境变量覆盖
3. 重写 worker launch args，确保选中的 worker diffusion scheduler backend 被一致应用
4. 在输出目录里写出一份 generated config
5. 启动 `python -m vllm_omni.global_scheduler.server --config <generated-config>`
6. 等待 `/health`、`/instances` 以及 worker `/v1/models` 就绪
7. 调用 `diffusion_benchmark_serving.py` 对 scheduler URL 发压
8. 停掉 scheduler，并保留所有产物

`run_suite.sh` 会对 `CASE_MATRIX` 中的每一行重复上述流程，最后额外生成：

- `summary.json`
- `summary.csv`

## 快速开始

Qwen 单 case：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
GLOBAL_POLICY=min_queue_length \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

使用同一份模板切到原始 `v18-base` worker 路径 `request_scheduler`：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
DIFFUSION_SCHEDULER_BACKEND=request_scheduler \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Wan2.2 单 case：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml \
GLOBAL_POLICY=short_queue_runtime \
REQUEST_RATES=0.05,0.1 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Suite 示例：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
REQUEST_RATES=0.2,0.4 \
CASE_MATRIX=$'mql|min_queue_length\nrr|round_robin\nsqr|short_queue_runtime' \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
```

## Benchmark 模式

通过 `BENCHMARK_MODE` 支持两种模式：

- `fixed_duration`
- `fixed_num_prompts`

`fixed_duration` 的实际语义：

- 使用 `REQUEST_RATES` 和 `NUM_PROMPTS_DURATION_SECONDS`
- 计算 `num_prompts = ceil(request_rate * duration_seconds)`
- 它不是 benchmark 内部“到点硬停止”的 wall-clock 截止
- 更准确地说，这是“根据目标时长推导请求总数”

`fixed_num_prompts` 的实际语义：

- 使用 `REQUEST_RATES` 和 `FIXED_NUM_PROMPTS`
- 每个 request rate 都发送相同数量的请求

示例：

```bash
BENCHMARK_MODE=fixed_duration \
NUM_PROMPTS_DURATION_SECONDS=600 \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

```bash
BENCHMARK_MODE=fixed_num_prompts \
FIXED_NUM_PROMPTS=20 \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

## 基础 YAML 结构

基础 YAML 主要由 5 部分组成：

- `server`：scheduler 的 host / port / timeout
- `scheduler`：tie-breaker 和 EWMA 参数
- `policy`：默认全局策略
- `benchmark`：benchmark 运行参数
- `instances`：worker 定义，以及 launch / stop 命令

`benchmark` 中常用字段：

- `worker_ids`：本次要纳入的 worker
- `worker_ready_timeout_s`：等待 worker 变成可路由且 API-ready 的超时时间
- `model`：benchmark 侧使用的模型名
- `backend`：传给 `diffusion_benchmark_serving.py`
- `task`：例如 `t2i`、`t2v`
- `dataset`：`random`、`trace` 或 `vbench`
- `dataset_path`：可选的数据集路径
- `random_request_config`：`dataset=random` 时的请求混合分布
- `warmup_requests`：warmup 请求数量
- `warmup_num_inference_steps`：warmup 的兜底 steps
- `warmup_request_config`：可选 warmup profile 列表
- `max_concurrency`：客户端侧 in-flight 请求上限
- metrics 输出路径由 orchestrator 接管，不建议在基础 YAML 中维护

`instances[*]` 中常用字段：

- `id`：逻辑 worker id
- `endpoint`：worker base URL
- `instance_type`：`short_queue_runtime` 估算时会使用
- `backends`：scheduler 可路由到该实例的 backend 列表
- `launch`：如何启动 worker
- `stop`：如何停止 worker

worker diffusion scheduler backend 的选择规则：

- 如果设置了 `DIFFUSION_SCHEDULER_BACKEND`，它会覆盖基础 YAML 中的 backend 配置
- 如果没设置，就沿用 `launch.args` 里原本的 backend
- 如果两边都没有，就回退到 `request_scheduler`
- 如果选中 `step_level_request_scheduler`，orchestrator 会强制注入 step chunk
- 如果选中 `request_scheduler`，会自动剥离 step-level 专用参数

## Warmup 语义

`benchmark.warmup_request_config` 会透传为：

- `diffusion_benchmark_serving.py --warmup-request-config`

它的行为是：

- warmup 请求仍然基于主 benchmark 选择的数据集构造
- 然后再用 profile 覆盖 `width`、`height`、`num_frames`、`fps`、`num_inference_steps` 等字段
- 当你希望 warmup 覆盖一组固定服务形态时，这是推荐做法

当前限制：

- 这里没有独立的 `warmup_dataset`
- 也就是说，warmup 不能在这个 orchestrator 里使用一份与主 benchmark 不同的数据集

## 环境变量

case 级变量：

- `BASE_CONFIG`：基础 YAML 路径，默认是 `single_instance.qwen.yaml`
- `GLOBAL_POLICY`：覆盖全局路由策略
- `DIFFUSION_SCHEDULER_BACKEND`：可选的 worker backend 覆盖。支持 `request_scheduler`、`step_level_request_scheduler`
- `ENABLE_STEP_CHUNK`：可选的 step-chunk 布尔覆盖，只对 `step_level_request_scheduler` 有意义
- `REQUEST_RATES`：逗号或空格分隔的 request rate，例如 `0.2,0.4,0.6`
- `BENCHMARK_MODE`：`fixed_duration` 或 `fixed_num_prompts`
- `NUM_PROMPTS_DURATION_SECONDS`：`fixed_duration` 使用
- `FIXED_NUM_PROMPTS`：`fixed_num_prompts` 使用
- `CASE_NAME`：可选的 case 名覆盖
- `RUN_TAG`：可选的运行标签覆盖
- `OUT_DIR`：单 case 显式输出目录
- `BENCH_OUTPUT_FILE`：显式 metrics JSON 路径
- `SCHEDULER_LOG_FILE`：显式 scheduler 日志路径
- `WORKER_IDS`：可选，仅运行部分 worker

benchmark 覆盖变量：

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

suite 专用变量：

- `SUITE_NAME`：suite 输出目录名
- `OUT_ROOT`：显式 suite 输出根目录
- `CASE_MATRIX`：每行一个 case，格式是 `case_name|global_policy`

示例：

```bash
CASE_MATRIX=$'mql|min_queue_length\nrr|round_robin\nsqr|short_queue_runtime'
```

## 输出目录结构

执行 `run_case.sh` 时，默认会在下面生成一个 case 目录：

- `benchmarks/diffusion/results/<case_name>_<run_tag>/`

常见产物包括：

- `global_scheduler.generated.yaml`：套用环境变量后的最终配置
- `global_scheduler_server.log`：scheduler 的 stdout / stderr
- `instance_logs/`：如果 scheduler 把 worker 日志写到这里，就会出现在该目录下
- `metrics.json` 或 `metrics_rps_<rate>.json`：benchmark 输出的 metrics

补充说明：

- 基础 YAML 不需要再维护 `benchmark.output_file`
- 如果没有显式传 `BENCH_OUTPUT_FILE`，orchestrator 会自动把 metrics 写到 case 输出目录里

执行 `run_suite.sh` 时，suite 根目录还会额外包含：

- `summary.json`
- `summary.csv`

当前 `summary.csv` 的列包括：

- `case`
- `request_rate`
- `completed`
- `throughput_qps`
- `latency_p50`
- `latency_p95`
- `latency_p99`
- `backend`
- `model`
- `metrics_file`

## 模板说明

`single_instance.qwen.yaml`：

- 图像生成模板
- 默认 backend 是 `vllm-omni`
- 默认 task 是 `t2i`
- 内置了 4 类 shape 的 warmup profile
- 当前模板中的 worker launch args 默认是 `step_level_request_scheduler`

`single_instance.wan2_2.yaml`：

- 视频生成模板
- 默认 backend 是 `v1/videos`
- 默认 task 是 `t2v`
- 内置了 3 类视频请求的 warmup mix
- 给了一份 4 卡 `usp/cfg/hsdp` 的 worker 启动样例
- 当前模板中的 worker launch args 默认是 `step_level_request_scheduler`

这些模板只是起点，不是冻结不变的生产配置。
你仍然需要根据本地环境修改端口、模型路径、GPU 可见性和 worker 启动参数。

## 已知限制

- README 示例默认你已经具备 `python3`、`vllm` 和所需 Python 依赖
- `fixed_duration` 是“按目标时长推导请求数”，不是 benchmark 内部的硬截止时长
- 这个 orchestrator 目前只面向这轮迁移保留的 3 个全局策略
- scheduler 侧等待在这轮迁移里是刻意不支持的
- `request_scheduler` 与 `ENABLE_STEP_CHUNK=1` 不能同时使用
