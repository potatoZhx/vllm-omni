# vLLM-Omni Global Scheduler

This package provides a minimal global request router for multi-instance
diffusion serving on `v18-base`.

Supported global policies in this migration:

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

Current behavior is intentionally narrow:

- the global scheduler does immediate routing only
- it does not maintain a scheduler-side waiting queue
- worker-side step-level execution stays controlled by the worker launch args
- `short_queue_runtime` uses request `estimated_cost_s` when provided, otherwise
  it falls back to EWMA / optional runtime profiles
