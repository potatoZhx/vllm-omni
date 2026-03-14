# Global Scheduler Serving Guide

This folder contains the global scheduler proxy for vLLM-Omni.
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
    algorithm: fcfs  # fcfs | min_queue_length | round_robin | short_queue_runtime | estimated_completion_time

instances:
  - id: worker-0
    endpoint: http://127.0.0.1:9001
    backends: [vllm-omni, openai]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "0,1"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port 9001"]
  - id: worker-1
    endpoint: http://127.0.0.1:9002
    backends: [v1/videos]
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args: ["--omni", "--max-concurrency", "2", "--ulysses-degree", "2", "--cfg-parallel-size", "2", "--hsdp"]
      env:
        CUDA_VISIBLE_DEVICES: "2,3"
    stop:
      executable: pkill
      args: ["-f", "vllm serve Qwen/Qwen-Image --port 9002"]
```

### 1.2 Start global scheduler

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
```

The scheduler listens at `http://<host>:<port>` from config (default `8089`).

Important behavior:

- This command starts the scheduler service itself.
- It does not automatically start all upstream workers on boot.
- Use lifecycle APIs (`/instances/{id}/start|stop|restart`) to control workers through scheduler.

### 1.3 Start workers via scheduler lifecycle APIs

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/start
curl -sS -X POST http://127.0.0.1:8089/instances/worker-1/start
```

### 1.4 Check readiness

```bash
curl -sS http://127.0.0.1:8089/instances
```

Ensure at least one instance has:

- `enabled=true`
- `healthy=true`
- `draining=false`
- `process_state=running`

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

Backend routing note:

- `/v1/chat/completions` routes to backend `vllm-omni`
- `/v1/images/generations` routes to backend `openai`
- `/v1/videos` routes to backend `v1/videos`
- `instances[].backends` controls which backends an instance can receive.
- If `instances[].backends` is omitted or empty, that instance is compatible with all backends.

Response headers include:

- `X-Routed-Instance`: selected instance id
- `X-Route-Reason`: routing reason string
- `X-Route-Score`: routing score (stringified float)

### 2.2 Health and instance status

- `GET /health`: config and service health summary
- `GET /instances`: instance lifecycle and runtime stats snapshot

Example:

```bash
curl -sS http://127.0.0.1:8089/health
curl -sS http://127.0.0.1:8089/instances
```

### 2.3 Lifecycle operation APIs (current implementation)

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

- `start/restart` require `instances[].launch` config.
- `stop/restart` require `instances[].stop` config.
- `reload` requires server started with `--config` and reload-capable loader (default path in module entrypoint supports this).

## 3. Routing Policies

Set in YAML:

- `policy.baseline.algorithm=fcfs`
- `policy.baseline.algorithm=min_queue_length`
- `policy.baseline.algorithm=round_robin`
- `policy.baseline.algorithm=short_queue_runtime`
- `policy.baseline.algorithm=estimated_completion_time`

Related knobs:

- `scheduler.tie_breaker`: `random` or `lexical`
- `scheduler.ewma_alpha`: EWMA smoothing factor `(0, 1]`
- per-instance routing concurrency is inferred from `instances[].launch.args`
  - recommended flag: `--max-concurrency`

## 4. Error Semantics

Global scheduler returns normalized error body:

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

## 5. Benchmark Through Scheduler

You can benchmark through scheduler by pointing `--base-url` to scheduler address.

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

This exercises the full path:

- benchmark client -> global scheduler -> selected upstream instance

## 6. Troubleshooting

### 6.1 `GS_NO_ROUTABLE_INSTANCE`

Check:

- `GET /instances` shows at least one instance with
  - `enabled=true`
  - `healthy=true`
  - `draining=false`
  - `process_state=running`
- Upstream endpoint in config is reachable (`http://host:port`, no path)

### 6.2 Frequent `GS_UPSTREAM_TIMEOUT`

Check:

- `server.request_timeout_s` is large enough for model/task
- Upstream service is overloaded (`inflight` close to `max_concurrency`)
- Health probe timeout is not too aggressive for your environment

### 6.3 Config validation failed at startup

Common causes:

- duplicate `instances[].id`
- invalid `policy.baseline.algorithm`
- invalid endpoint format (must be `http://host:port`)
- malformed structured `launch/stop` config

## 7. Current Limitations

- No dynamic SP scheduling.
- No `/metrics` endpoint yet in current implementation.
- Local process control depends on command templates; production environments should prefer orchestrator-native adapters.
