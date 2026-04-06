# Global Scheduler Benchmark Orchestrator 中文说明

该目录保存了 `v18-base` 上 `global scheduler` 迁移后的精简 benchmark / orchestration 入口。

当前范围：

- 只支持覆盖全局路由策略
- worker 启动参数只强制保留：
  - `--diffusion-scheduler-backend step_level_request_scheduler`
  - `--diffusion-enable-step-chunk`
- 支持通过 `benchmark.warmup_request_config` 构造 warmup 请求
- orchestration 不再依赖：
  - `diffusion_engine_max_concurrency`
  - `chunk_preemption`
  - `chunk_budget`

当前支持的全局策略：

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

当前提供的单实例模板：

- `single_instance.qwen.yaml`
- `single_instance.wan2_2.yaml`

Warmup 构造方式：

- `benchmark.warmup_request_config` 会透传为 `diffusion_benchmark_serving.py --warmup-request-config`
- warmup 请求仍然基于主 benchmark 选定的数据集构造，再由 `warmup_request_config` 中的 profile 覆盖尺寸、steps、frames 等字段
- 当你希望 warmup 覆盖一组固定的服务形态，而不是只复用前几个数据集请求时，优先使用这种方式

Qwen 单 case 示例：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
GLOBAL_POLICY=min_queue_length \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Wan2.2 单 case 示例：

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml \
GLOBAL_POLICY=short_queue_runtime \
REQUEST_RATES=0.05,0.1 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```
