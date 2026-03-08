# Global Scheduler Serving Guide

This folder contains the global scheduler proxy for vLLM-Omni.
It exposes a single OpenAI-compatible entrypoint and routes requests to
multiple upstream vLLM instances.

Main module:

- `vllm_omni/global_scheduler/server.py`

## 1. Quick Start

### 1.1 Start upstream vLLM instances

Example (two instances on different ports):

```bash
vllm serve Qwen/Qwen-Image --omni --port 9001
vllm serve Qwen/Qwen-Image --omni --port 9002
```

### 1.2 Create scheduler config

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
    algorithm: fcfs  # fcfs | short_queue_runtime | estimated_completion_time

instances:
  - id: worker-0
    endpoint: http://127.0.0.1:9001
    sp_size: 1
    max_concurrency: 2
  - id: worker-1
    endpoint: http://127.0.0.1:9002
    sp_size: 1
    max_concurrency: 2
```

### 1.3 Start global scheduler

```bash
python3 -m vllm_omni.global_scheduler.server --config ./global_scheduler.yaml
```

The scheduler listens at `http://<host>:<port>` from config (default `8089`).

### 1.4 Smoke test with one request

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

### 2.1 Request entrypoint

- `POST /v1/chat/completions`

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
- `POST /instances/reload`
- `POST /instances/probe`

Examples:

```bash
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/disable
curl -sS -X POST http://127.0.0.1:8089/instances/worker-0/enable
curl -sS -X POST http://127.0.0.1:8089/instances/reload
curl -sS -X POST http://127.0.0.1:8089/instances/probe
```

Notes:

- `stop/start/restart` APIs are design-stage items and are not exposed yet.
- `reload` requires server started with `--config` and reload-capable loader (default path in module entrypoint supports this).

## 3. Routing Policies

Set in YAML:

- `policy.baseline.algorithm=fcfs`
- `policy.baseline.algorithm=short_queue_runtime`
- `policy.baseline.algorithm=estimated_completion_time`

Related knobs:

- `scheduler.tie_breaker`: `random` or `lexical`
- `scheduler.ewma_alpha`: EWMA smoothing factor `(0, 1]`

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
- `sp_size != 1` in current stage

## 7. Current Limitations

- No dynamic SP scheduling.
- No `/metrics` endpoint yet in current implementation.
- No process-level `stop/start/restart` API yet.
