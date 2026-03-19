# Global Scheduler Serving Guide

This directory contains the vLLM-Omni global scheduler proxy.
It exposes OpenAI-compatible entrypoints and routes requests to
multiple upstream vLLM instances.

Main module:

- `vllm_omni/global_scheduler/server.py`

## 1. Quick Start

### 1.1 Create scheduler config

Create `global_scheduler.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 1800
  instance_health_check_interval_s: 5.0
  instance_health_check_timeout_s: 1.0

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: short_queue_runtime  # fcfs | min_queue_length | round_robin | short_queue_runtime | estimated_completion_time
    runtime_profile_path: ./runtime_profile.json

benchmark:
  worker_ids: [worker-0, worker-1]
  worker_ready_timeout_s: 600
  model: Qwen/Qwen-Image
  backend: vllm-omni
  task: t2i
  dataset: trace
  max_concurrency: 20
  auto_stop: true

instances:
  - id: worker-0
    endpoint: http://127.0.0.1:9001
    instance_type: qwen-image-tp2
    numa_node: 0
    backends: [vllm-omni, openai]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "0,1"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port {endpoint_port}"]
  - id: worker-1
    endpoint: http://127.0.0.1:9002
    instance_type: wan-video-tp2
    numa_node: 1
    backends: [v1/videos]
    launch:
      executable: vllm
      model: Wan/Wan2.2
      args: ["--omni", "--max-concurrency", "2"]
      env:
        CUDA_VISIBLE_DEVICES: "2,3"
```

Notes:

- `policy.baseline.runtime_profile_path`
  - profiling JSON used by `short_queue_runtime` and `estimated_completion_time`
- `instances[].instance_type`
  - instance-type label used to match profile records
- `instances[].numa_node`
  - if `numactl` exists on the host, scheduler adds NUMA binding automatically on start
- `launch.args`
  - provide only extra args; scheduler adds `vllm serve <model> --port <endpoint_port>` itself
- `stop.args`
  - placeholders are supported: `{instance_id}`, `{endpoint}`, `{endpoint_host}`, `{endpoint_port}`
- `benchmark`
  - colocated benchmark config consumed by `scripts/run_global_scheduler_benchmark*.sh`

### 1.2 Start global scheduler

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
```

The scheduler listens on `http://<host>:<port>` from config (default `8089`).

Important current behavior:

- This command starts the scheduler service itself.
- If an instance has `launch` config, server startup automatically issues one `start`.
- After auto-start, an instance usually reaches `process_state=running` before it becomes `healthy=true`.
- If an instance has no `launch` config, scheduler will not start it for you.
- On server shutdown, instances with `stop` config are best-effort stopped.

### 1.3 Trigger probe or lifecycle operations manually

To refresh routability immediately:

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

If an instance was not auto-started, or you want to restart it manually:

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-1/restart
```

### 1.4 Check readiness

```bash
curl -sS http://127.0.0.1:8089/health
curl -sS http://127.0.0.1:8089/instances
```

Ensure at least one instance has:

- `enabled=true`
- `healthy=true`
- `draining=false`
- `process_state=running`
- `routable=true`

### 1.5 Smoke test with one request

```bash
curl -sS http://127.0.0.1:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image",
    "messages": [{"role": "user", "content": "a cute orange cat"}],
    "extra_body": {
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 20
    }
  }'
```

## 2. Runtime APIs

### 2.1 Request entrypoints

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `POST /v1/videos`

Backend routing:

- `/v1/chat/completions` routes to backend `vllm-omni`
- `/v1/images/generations` routes to backend `openai`
- `/v1/videos` routes to backend `v1/videos`
- `instances[].backends` restricts which backends an instance can receive
- if `instances[].backends` is omitted or empty, that instance is treated as compatible with all supported backends

Scheduler extracts these request fields when available for routing:

- `width`
- `height`
- `num_frames`
- `num_inference_steps`

Extraction sources:

- `extra_body` in chat/images JSON
- top-level chat/images JSON fields
- OpenAI image `size`
- multipart form fields for `/v1/videos`

Response headers include:

- `X-Routed-Instance`: selected instance id
- `X-Route-Reason`: routing reason string
- `X-Route-Score`: routing score as stringified float

### 2.2 Health and instance status

- `GET /health`
  - returns `status`, `instance_count`, `version`
  - `checks` currently includes:
    - `config_loaded`
    - `has_instances`
- `GET /instances`
  - returns lifecycle and runtime snapshot for every instance

Each `/instances` item currently includes:

- `id`
- `endpoint`
- `backends`
- `enabled`
- `healthy`
- `draining`
- `process_state`
- `last_operation`
- `last_operation_ts_s`
- `last_operation_error`
- `last_check_ts_s`
- `last_error`
- `log_path`
- `routable`
- `queue_len`
- `inflight`
- `ewma_service_time_s`

Where:

- `routable = enabled && healthy && !draining && process_state == "running"`
- `log_path` defaults to `./logs/global_scheduler/<instance_id>.log`
  - override via `GLOBAL_SCHEDULER_LOG_DIR`

### 2.3 Lifecycle APIs (current implementation)

- `POST /instances/{id}/disable`
- `POST /instances/{id}/enable`
- `POST /instances/{id}/stop`
- `POST /instances/{id}/start`
- `POST /instances/{id}/restart`
- `POST /instances/reload`
- `POST /instances/probe`

Examples:

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/disable
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/enable
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/stop
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/restart
curl -sS -X POST http://127.0.0.1:8089/instances/reload
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

Notes:

- `disable/enable`
  - only changes routing availability inside scheduler; it does not directly manage the process
- `start/restart`
  - require `instances[].launch`
- `stop/restart`
  - require `instances[].stop`
- `start`
  - is idempotent for an instance that is already `running` and whose last operation was also `start`
- `reload`
  - requires server startup via `--config`, then reloads YAML, rebuilds policy, and syncs instance inventory

## 3. Routing Policies

Configure via YAML:

- `policy.baseline.algorithm=fcfs`
- `policy.baseline.algorithm=min_queue_length`
- `policy.baseline.algorithm=round_robin`
- `policy.baseline.algorithm=short_queue_runtime`
- `policy.baseline.algorithm=estimated_completion_time`

Related knobs:

- `scheduler.tie_breaker`
  - `random` or `lexical`
- `scheduler.ewma_alpha`
  - EWMA smoothing factor for per-instance service time `(0, 1]`
- `policy.baseline.runtime_profile_path`
  - runtime profile JSON path
- `instances[].instance_type`
  - instance label used to select profile records

Policy behavior:

- `fcfs`
  - picks the first available instance; ties are broken by tie-breaker
- `min_queue_length`
  - picks the instance with smallest `queue_len`
- `round_robin`
  - rotates across available instances
- `short_queue_runtime`
  - picks the instance with smallest estimated outstanding queue runtime
  - sums profiled/EWMA waiting-request runtime and adds `inflight * ewma_service_time_s`
- `estimated_completion_time`
  - picks the instance with smallest estimated completion time for the current request
  - current approximation is `queue_len * current_request_runtime + current_request_runtime`

Additional notes:

- `short_queue_runtime` and `estimated_completion_time` fall back to EWMA if profile data is missing
- runtime profile JSON must contain a `profiles` array and use `latency_ms`
- `--max-concurrency` is used by scheduler to infer per-instance routing capacity, but is stripped before spawning the real `vllm serve` child process

## 4. Error Semantics

Request proxy paths and most lifecycle operations return a normalized body:

```json
{
  "error": {
    "code": "GS_...",
    "message": "...",
    "request_id": "..."
  }
}
```

Common error codes:

- `GS_NO_ROUTABLE_INSTANCE` (503)
- `GS_UPSTREAM_TIMEOUT` (502)
- `GS_UPSTREAM_NETWORK_ERROR` (502)
- `GS_UPSTREAM_HTTP_ERROR` (upstream status code)
- `GS_LIFECYCLE_CONFLICT` (409)
- `GS_LIFECYCLE_UNSUPPORTED` (400)
- `GS_LIFECYCLE_EXEC_ERROR` (502)
- `GS_UNKNOWN_INSTANCE` (404)

Additional note:

- some management-path failures from `reload`, `enable`, and `disable` can still return the default FastAPI error shape instead of `GS_*`

## 5. Benchmark Through Scheduler

Point `--base-url` at scheduler to benchmark the full routed path.

Example:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8089 \
  --model Qwen/Qwen-Image \
  --task t2i \
  --dataset vbench \
  --num-prompts 20 \
  --max-concurrency 4
```

Full path:

- benchmark client -> global scheduler -> selected upstream instance

Helper scripts in this repo:

- `scripts/run_global_scheduler_benchmark.sh`
- `scripts/run_global_scheduler_benchmark_one_shell.sh`
- `scripts/run_global_scheduler_benchmark_one_shell_cleanup.sh`

These scripts read the colocated `benchmark` section from the same YAML file.

## 6. Troubleshooting

### 6.1 `GS_NO_ROUTABLE_INSTANCE`

Check:

- `GET /instances` shows at least one instance with
  - `enabled=true`
  - `healthy=true`
  - `draining=false`
  - `process_state=running`
  - `routable=true`
- endpoint in config is reachable (`http://host:port`, no path)
- if service just started and workers were auto-started, run `POST /instances/probe` once

### 6.2 Frequent `GS_UPSTREAM_TIMEOUT`

Check:

- `server.request_timeout_s` is large enough
- upstream is overloaded (`inflight` near instance concurrency limit)
- health probe timeout is not too aggressive
- `short_queue_runtime` / `estimated_completion_time` is not missing runtime profile data for your workload

### 6.3 Config validation failed at startup

Common causes:

- duplicate `instances[].id`
- invalid `policy.baseline.algorithm`
- empty `policy.baseline.runtime_profile_path`
- invalid endpoint format (must be `http://host:port` and must not include path)
- invalid backend in `instances[].backends`
- empty `instances[].instance_type`
- `instances[].numa_node < 0`
- invalid structured `launch` / `stop` config

### 6.4 Lifecycle call succeeded but instance is still not routable

Check:

- after `start`, an instance can be `process_state=running` while HTTP is still not ready
- inspect `last_error`
  - common values include `awaiting_http_ready_after_start`
  - or `awaiting_probe_after_start`
- inspect the file at `log_path` to confirm the upstream process actually booted
