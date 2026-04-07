# Global Scheduler Benchmark Orchestrator

This directory contains the reduced benchmark / orchestration entrypoints used
to exercise the `v18-base` global scheduler migration.

The orchestrator is intentionally narrow: it starts the global scheduler,
waits until workers become routable and API-ready, then runs
`benchmarks/diffusion/diffusion_benchmark_serving.py` against the scheduler URL.

## Scope

What this directory supports:

- global routing policies:
  - `min_queue_length`
  - `round_robin`
  - `short_queue_runtime`
- worker launch normalization:
  - supports both `request_scheduler` and `step_level_request_scheduler`
  - injects `--diffusion-enable-step-chunk` only when `step_level_request_scheduler` is selected
- benchmark-side warmup profile construction via `benchmark.warmup_request_config`
- single-case runs and multi-case suites
- worker-side migrated instance policies can remain in `launch.args`, including:
  - `fcfs`
  - `sjf`
  - `sjf_aging`
  - `sjf_aging_guarded`
  - `sjf_aging_guarded_tail`
  - `p95-first`

What is intentionally out of scope:

- scheduler-side waiting / admission blocking
- `diffusion_engine_max_concurrency`
- orchestration dependency on `chunk_preemption` or `chunk_budget`
- orchestration-side synthesis of policy-specific tuning knobs for guarded /
  tail / p95 variants

Important runtime semantics:

- The global scheduler in this migration is `pure routing`, not a scheduler-side queue.
- Requests are routed immediately to a worker.
- Worker-local waiting and execution are still decided by the worker itself.

## Files

- `run_case.sh`: run one case
- `run_suite.sh`: run multiple cases and aggregate summaries
- `orchestrate.py`: main implementation
- `single_instance.qwen.yaml`: single-worker Qwen image template
- `single_instance.wan2_2.yaml`: single-worker Wan2.2 video template
- `README_zh.md`: Chinese version of this document

## How It Works

For each case, the orchestrator does the following:

1. Read a base YAML config.
2. Apply environment-variable overrides.
3. Rewrite worker launch args so the selected worker diffusion scheduler backend is applied consistently.
4. Write a generated config file into the output directory.
5. Start `python -m vllm_omni.global_scheduler.server --config <generated-config>`.
6. Wait for `/health`, `/instances`, and worker `/v1/models` readiness.
7. Launch `diffusion_benchmark_serving.py` against the scheduler URL.
8. Stop the scheduler and leave all artifacts on disk.

For suites, `run_suite.sh` repeats the case flow for each row in `CASE_MATRIX`,
then writes an aggregated `summary.json` and `summary.csv`.

## Quick Start

Single case with the Qwen template:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
GLOBAL_POLICY=min_queue_length \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Run the original `v18-base` worker path (`request_scheduler`) with the same template:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
DIFFUSION_SCHEDULER_BACKEND=request_scheduler \
REQUEST_RATES=0.2,0.4 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Single case with the Wan2.2 template:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.wan2_2.yaml \
GLOBAL_POLICY=short_queue_runtime \
REQUEST_RATES=0.05,0.1 \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_case.sh
```

Suite example:

```bash
BASE_CONFIG=./benchmarks/diffusion/scripts/global_instance_scheduler_v2/single_instance.qwen.yaml \
REQUEST_RATES=0.2,0.4 \
CASE_MATRIX=$'mql|min_queue_length\nrr|round_robin\nsqr|short_queue_runtime' \
benchmarks/diffusion/scripts/global_instance_scheduler_v2/run_suite.sh
```

## Benchmark Modes

The orchestrator supports two modes via `BENCHMARK_MODE`:

- `fixed_duration`
- `fixed_num_prompts`

`fixed_duration` behavior:

- Uses `REQUEST_RATES` plus `NUM_PROMPTS_DURATION_SECONDS`.
- Derives `num_prompts = ceil(request_rate * duration_seconds)`.
- This is not a hard wall-clock stop inside the benchmark script.
- In practice it means "derive a request count from a target send duration".

`fixed_num_prompts` behavior:

- Uses `REQUEST_RATES` plus `FIXED_NUM_PROMPTS`.
- Each rate runs with the same number of requests.

Examples:

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

## Base YAML Structure

The base YAML contains four main sections:

- `server`: scheduler host / port / timeouts
- `scheduler`: tie-breaker and EWMA settings
- `policy`: default global routing policy
- `benchmark`: benchmark runtime settings
- `instances`: worker definitions and launch / stop commands

Important `benchmark` fields:

- `worker_ids`: workers to include in this run
- `worker_ready_timeout_s`: timeout for routable + API-ready workers
- `model`: benchmark-side model name
- `backend`: forwarded to `diffusion_benchmark_serving.py`
- `task`: `t2i`, `t2v`, etc.
- `dataset`: `random`, `trace`, or `vbench`
- `dataset_path`: optional dataset path
- `random_request_config`: synthetic request mix for `dataset=random`
- `warmup_requests`: number of warmup requests
- `warmup_num_inference_steps`: fallback warmup steps
- `warmup_request_config`: optional warmup profile list
- `max_concurrency`: client-side in-flight request cap
- metrics output path is orchestrator-managed, not expected to be edited in the base YAML

Important `instances[*]` fields:

- `id`: logical worker id
- `endpoint`: worker base URL
- `instance_type`: used by `short_queue_runtime` estimation
- `backends`: routable backends advertised to the scheduler
- `launch`: how the orchestrator starts the worker
- `stop`: how the orchestrator stops the worker

Worker diffusion scheduler backend selection:

- if `DIFFUSION_SCHEDULER_BACKEND` is set, it overrides the backend encoded in the base YAML
- otherwise the orchestrator follows the backend already present in `launch.args`
- if neither is present, it falls back to `request_scheduler`
- when `step_level_request_scheduler` is selected, step chunk is always injected
- when `request_scheduler` is selected, step-level-only flags are stripped
- when the final backend is `request_scheduler`, `--instance-scheduler-policy` is also omitted from worker args
- if `INSTANCE_POLICY` is set, it overrides the worker
  `--instance-scheduler-policy` from the base YAML or launch args
- otherwise the existing worker instance policy is preserved from the base YAML
  or launch args

## Warmup Behavior

`benchmark.warmup_request_config` is mapped to:

- `diffusion_benchmark_serving.py --warmup-request-config`

Semantics:

- warmup requests are still built from the selected benchmark dataset
- then the listed warmup profiles override shape-related fields such as
  `width`, `height`, `num_frames`, `fps`, and `num_inference_steps`
- this is the recommended way to make warmup cover a fixed serving mix

Current limitation:

- there is no separate `warmup_dataset`
- warmup cannot use a different dataset from the main benchmark in this orchestrator

## Environment Variables

Case-level variables:

- `BASE_CONFIG`: base YAML file. Default is `single_instance.qwen.yaml`
- `GLOBAL_POLICY`: override global routing policy
- `INSTANCE_POLICY`: optional worker instance policy override, for example `fcfs`, `sjf`, or `sjf_aging`
- `DIFFUSION_SCHEDULER_BACKEND`: optional worker backend override. Supported values: `request_scheduler`, `step_level_request_scheduler`
- `ENABLE_STEP_CHUNK`: optional explicit override for the step-chunk launch flag. Only meaningful with `step_level_request_scheduler`
- `REQUEST_RATES`: comma- or space-separated rates, for example `0.2,0.4,0.6`
- `BENCHMARK_MODE`: `fixed_duration` or `fixed_num_prompts`
- `NUM_PROMPTS_DURATION_SECONDS`: used by `fixed_duration`
- `FIXED_NUM_PROMPTS`: used by `fixed_num_prompts`
- `CASE_NAME`: optional case name override
- `RUN_TAG`: optional run tag override
- `OUT_DIR`: explicit output directory for one case
- `BENCH_OUTPUT_FILE`: explicit metrics JSON path
- `SCHEDULER_LOG_FILE`: explicit scheduler log path
- `WORKER_IDS`: optional subset of workers to run

Benchmark override variables:

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

Suite-only variables:

- `SUITE_NAME`: output directory name for suite mode
- `OUT_ROOT`: explicit suite output directory
- `CASE_MATRIX`: one row per case. Supported formats:
  - `case_name|global_policy`
  - `case_name|global_policy|instance_policy`
  - `case_name|global_policy|instance_policy|scheduler_backend_flag`
  - longer legacy rows are accepted, but only the first 4 columns are used

The 4th column `scheduler_backend_flag` supports:

- `0`: use `request_scheduler`
- `1`: use `step_level_request_scheduler`
- literal backend names are also accepted

Additional semantics:

- when the 4th column is `0`, the orchestrator runs the case with `request_scheduler`
- in that mode, `--diffusion-enable-step-chunk` is not applied
- in that mode, the `instance_policy` column and `INSTANCE_POLICY` env var are ignored

Example:

```bash
CASE_MATRIX=$'mql|min_queue_length\nrr|round_robin\nsqr|short_queue_runtime'
```

```bash
CASE_MATRIX=$'fcfs|round_robin|fcfs\nsjf_aging|round_robin|sjf_aging'
```

```bash
CASE_MATRIX=$'sjf_req|round_robin|sjf|0\nsjf_aging_step|round_robin|sjf_aging|1'
```

```bash
CASE_MATRIX=$'sjf|round_robin|sjf|0|0|5\nsjf_aging|round_robin|sjf_aging|1|1|5'
```

## Output Layout

For `run_case.sh`, the orchestrator creates a case directory under:

- `benchmarks/diffusion/results/<case_name>_<run_tag>/`

Common artifacts:

- `global_scheduler.generated.yaml`: generated config after env overrides
- `global_scheduler_server.log`: scheduler stdout / stderr
- `instance_logs/`: worker logs if the scheduler writes them there
- `metrics.json` or `metrics_rps_<rate>.json`: benchmark metrics

Note:

- the base YAML does not need a user-maintained `benchmark.output_file`
- if `BENCH_OUTPUT_FILE` is not provided, the orchestrator writes metrics into the case output directory automatically

For `run_suite.sh`, the suite root additionally contains:

- `summary.json`
- `summary.csv`

`summary.csv` currently includes:

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

## Template Notes

`single_instance.qwen.yaml`:

- image generation template
- default backend is `vllm-omni`
- default task is `t2i`
- includes a four-shape warmup profile
- base worker launch args currently default to `step_level_request_scheduler`

`single_instance.wan2_2.yaml`:

- video generation template
- default backend is `v1/videos`
- default task is `t2v`
- includes a three-profile warmup mix
- includes a 4-GPU worker launch example with `usp/cfg/hsdp`
- base worker launch args currently default to `step_level_request_scheduler`

These templates are starting points, not frozen production configs.
You are expected to edit ports, model paths, GPU visibility, and worker launch args
for your environment.

## Known Limitations

- The README examples assume `python3`, `vllm`, and required Python deps are already available.
- `fixed_duration` is request-count derivation, not a hard benchmark-side wall-clock cutoff.
- The orchestrator only documents and validates the three migrated global policies.
- Scheduler-side waiting is intentionally absent in this migration.
- `ENABLE_STEP_CHUNK=1` is invalid with `request_scheduler`.
