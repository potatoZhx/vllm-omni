# Global Scheduler Benchmark Orchestrator

This directory contains the slimmed-down benchmark/orchestration entrypoints
for the `v18-base` global scheduler migration.

Current scope:

- global policy override only
- worker launch args only force:
  - `--diffusion-scheduler-backend step_level_request_scheduler`
  - `--diffusion-enable-step-chunk`
- benchmark warmup can be constructed via `benchmark.warmup_request_config`
- no orchestration dependency on:
  - `diffusion_engine_max_concurrency`
  - `chunk_preemption`
  - `chunk_budget`

Supported global policies:

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

Available single-instance templates:

- `single_instance.qwen.yaml`
- `single_instance.wan2_2.yaml`

Warmup construction:

- `benchmark.warmup_request_config` maps to `diffusion_benchmark_serving.py --warmup-request-config`
- warmup requests are still built from the selected benchmark dataset and then overridden by the listed warmup profiles
- use this when you want warmup to cover a fixed serving mix instead of only reusing the first few dataset requests

Single-case example:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
GLOBAL_POLICY=min_queue_length \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Wan2.2 example:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml \
GLOBAL_POLICY=short_queue_runtime \
REQUEST_RATES=0.05,0.1 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```
