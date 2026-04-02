# Global Scheduler Benchmark Orchestrator

This directory contains the slimmed-down benchmark/orchestration entrypoints
for the `v18-base` global scheduler migration.

Current scope:

- global policy override only
- worker launch args only force:
  - `--diffusion-scheduler-backend step_level_request_scheduler`
  - `--diffusion-enable-step-chunk`
- no orchestration dependency on:
  - `diffusion_engine_max_concurrency`
  - `chunk_preemption`
  - `chunk_budget`

Supported global policies:

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

Single-case example:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
GLOBAL_POLICY=min_queue_length \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```
