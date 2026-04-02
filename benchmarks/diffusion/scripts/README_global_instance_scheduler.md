# Global Scheduler Experiment Notes

This repository now carries a reduced `global scheduler` benchmark surface for
the `v16-base -> v18-base` migration.

What remains in scope:

- global routing policies:
  - `min_queue_length`
  - `round_robin`
  - `short_queue_runtime`
- worker metadata pass-through:
  - `slo_ms`
  - `estimated_cost_s`
- worker launch requirements:
  - `--diffusion-scheduler-backend step_level_request_scheduler`
  - `--diffusion-enable-step-chunk`

What is intentionally out of scope in this migration:

- global scheduler side waiting / admission blocking
- `diffusion_engine_max_concurrency`
- orchestration dependence on `chunk_preemption` or `chunk_budget`
- instance-level complex policies such as `sjf` / `p95-first` / `fusion`
