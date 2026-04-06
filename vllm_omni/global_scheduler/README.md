# vLLM-Omni Global Scheduler

This package provides the `v18-base` global scheduler used for multi-instance
diffusion serving.

In this migration, the scheduler is intentionally minimal:

- it is a global request router, not a scheduler-side waiting queue
- it routes immediately to a routable worker
- it tracks per-instance runtime bookkeeping for scoring
- worker-local execution and step-level scheduling stay inside each worker

The implementation is designed to be operationally usable while keeping the
scope narrow enough to land on top of the current `v18-base` diffusion stack.

## Scope of This Migration

Supported global policies:

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

Supported upstream backend families:

- `vllm-omni`
- `openai`
- `v1/videos`

Intentionally out of scope:

- scheduler-side waiting / admission blocking
- global capacity gating
- `fcfs`
- `estimated_completion_time`
- instance-local advanced policies such as `sjf`, `p95-first`, `guarded`, `fusion`
- worker-side fields such as `slo_target_ms` and `deadline_ts`

## Runtime Model

The scheduler keeps two kinds of state:

1. Static instance configuration
2. Runtime bookkeeping for already-routed requests

Important consequence:

- there is no scheduler-side pending queue in this implementation
- `queue_len` is currently always `0`
- the meaningful live signals are `inflight`, `outstanding_runtime_s`, and `ewma_service_time_s`

That means:

- `min_queue_length` effectively routes by current `inflight` count
- `round_robin` ignores load and rotates over routable instances
- `short_queue_runtime` routes by accumulated reserved runtime when request cost is available

## High-Level Request Flow

For each incoming request:

1. The scheduler extracts a `RequestMeta` from the request payload.
2. It filters instances by backend compatibility and lifecycle routability.
3. It runs the configured global policy on the current runtime snapshot.
4. It reserves runtime bookkeeping for the chosen instance.
5. It forwards the original HTTP request to the chosen worker.
6. When the upstream response finishes or errors, it releases the reservation and updates EWMA.

The scheduler does not hold requests until some global capacity becomes free.

## Public HTTP Endpoints

Health and inventory:

- `GET /health`
- `GET /instances`
- `POST /instances/reload`
- `POST /instances/probe`

Lifecycle operations:

- `POST /instances/{id}/enable`
- `POST /instances/{id}/disable`
- `POST /instances/{id}/start`
- `POST /instances/{id}/stop`
- `POST /instances/{id}/restart`

Request proxy endpoints:

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `POST /v1/videos`

Routing response headers:

- `X-Routed-Instance`
- `X-Route-Reason`
- `X-Route-Score`

These headers are attached when a route decision has been made, including the
common proxied success and upstream-error paths.

## Backend Mapping

The scheduler maps ingress routes to backend names as follows:

- `/v1/chat/completions` -> `vllm-omni`
- `/v1/images/generations` -> `openai`
- `/v1/videos` -> `v1/videos`

Each instance may declare a `backends` allowlist. If `backends` is empty, the
instance is considered compatible with all supported backends.

## Request Metadata Used for Routing

The scheduler currently extracts the following request-visible fields:

- `width`
- `height`
- `num_frames`
- `num_inference_steps`
- `estimated_cost_s`
- `slo_ms`

How extraction works:

- for `/v1/chat/completions` and `/v1/images/generations`, metadata can come
  from top-level fields or `extra_body`
- for image-style OpenAI requests, `size="WxH"` is also parsed
- for `/v1/videos`, metadata is extracted from multipart form fields

Current routing usage:

- `estimated_cost_s` is the main useful scheduler-visible cost signal
- `slo_ms` is parsed and preserved in `RequestMeta.extra`, but is not used by
  the current routing policies

## Routing Policies

### `min_queue_length`

Selects the instance with minimum:

- `inflight + queue_len`

In the current migration, `queue_len` stays `0`, so this behaves like
"pick the instance with the fewest active routed requests".

### `round_robin`

Routes across routable instances in round-robin order.

Notes:

- it does not inspect request shape
- it does not inspect runtime cost
- the returned score is the selected instance's current `inflight`

### `short_queue_runtime`

Selects the instance with minimum `outstanding_runtime_s`.

Important detail:

- if the incoming request does not carry `estimated_cost_s`, the policy falls
  back to `min_queue_length`

So, for good results with this policy, benchmark clients or upstream callers
should provide `estimated_cost_s`.

## Runtime Estimation

Reserved runtime for each routed request is computed by `RuntimeEstimator`.
The estimator uses this order:

1. request `estimated_cost_s`
2. runtime profile exact match
3. runtime profile interpolation across neighboring `steps`
4. per-instance EWMA fallback

This estimate is used to maintain `outstanding_runtime_s` in the runtime store.

Important nuance:

- runtime bookkeeping can still use runtime profile / EWMA fallback
- but `short_queue_runtime` route selection itself currently falls back to
  `min_queue_length` when the incoming request does not provide `estimated_cost_s`

## Runtime Profile Format

If `policy.baseline.runtime_profile_path` is configured, the scheduler loads a
JSON file with a top-level `profiles` array.

Supported fields per entry:

- `instance_type`
- `width`
- `height`
- `num_frames`
- `steps`
- `latency_ms`

Example:

```json
{
  "profiles": [
    {
      "instance_type": "wan-video-tp2",
      "width": 1280,
      "height": 720,
      "num_frames": 16,
      "steps": 50,
      "latency_ms": 8210
    }
  ]
}
```

The loaded table is keyed by:

- `(instance_type, width, height, num_frames, steps)`

and stored internally in seconds.

## Lifecycle and Health Model

Each instance has lifecycle state with:

- `enabled`
- `healthy`
- `draining`
- `process_state`

An instance is routable only when:

- `enabled == true`
- `healthy == true`
- `draining == false`
- `process_state == "running"`

### Health Probing

The scheduler runs a background probe loop.

After `start` or `restart`:

- the scheduler probes `GET /v1/models` until the worker exposes at least one model

During steady state:

- the scheduler probes `GET /health`

Failures are accumulated, and the instance becomes unhealthy after
`server.instance_health_check_failures_before_unhealthy` consecutive failures.

### Disable vs Stop

`disable`:

- prevents new routing
- marks the instance as draining
- does not stop the process

`stop`:

- executes the configured stop command
- marks the instance unavailable

`enable`:

- re-enables routing
- marks the instance healthy immediately

`start` / `restart`:

- run the configured launch command
- re-enable the instance
- mark it unhealthy until readiness probe succeeds

### Config Reload

`POST /instances/reload` reloads the same YAML path used to start the server.

Reload behavior:

- new instances are added
- changed instance configs are refreshed
- removed instances are deleted immediately if they have no inflight requests
- removed instances with inflight requests become draining and are removed after drain completes

## Process Control

Lifecycle operations are backed by `LocalProcessController`.

Start command shape:

- `<launch.executable> serve <launch.model> --port <endpoint_port> ...launch.args`

Stop command:

- `stop.executable` plus `stop.args`

Supported placeholder expansion in `stop.args`:

- `{instance_id}`
- `{endpoint}`
- `{endpoint_host}`
- `{endpoint_port}`

Instance logs:

- by default, managed worker logs are written under `./logs/global_scheduler`
- you can override this with `GLOBAL_SCHEDULER_LOG_DIR`

NUMA behavior:

- if `numactl` is available and the instance has `numa_node` configured, the
  launch command is prefixed with `numactl --cpunodebind=... --membind=...`

## Config Structure

The root config sections are:

- `server`
- `scheduler`
- `policy`
- `benchmark`
- `instances`

Runtime server actually consumes:

- `server`
- `scheduler`
- `policy`
- `instances`

The `benchmark` section is colocated for orchestration convenience and is mainly
used by the benchmark scripts under `benchmarks/diffusion/scripts/`.

### `server`

Main fields:

- `host`
- `port`
- `request_timeout_s`
- `instance_health_check_interval_s`
- `instance_health_check_timeout_s`
- `instance_health_check_failures_before_unhealthy`

### `scheduler`

Main fields:

- `tie_breaker`: `random` or `lexical`
- `ewma_alpha`

### `policy.baseline`

Main fields:

- `algorithm`
- `runtime_profile_path`

Supported algorithms:

- `min_queue_length`
- `round_robin`
- `short_queue_runtime`

### `instances[]`

Main fields:

- `id`
- `endpoint`
- `instance_type`
- `numa_node`
- `backends`
- `launch`
- `stop`

`endpoint` must be:

- `http://host:port`

No path is allowed in the endpoint.

## Example Config

```yaml
server:
  host: 0.0.0.0
  port: 8089
  request_timeout_s: 600
  instance_health_check_interval_s: 5.0
  instance_health_check_timeout_s: 1.0

scheduler:
  tie_breaker: random
  ewma_alpha: 0.2

policy:
  baseline:
    algorithm: short_queue_runtime
    runtime_profile_path: ./runtime_profile.json

instances:
  - id: worker0
    endpoint: http://127.0.0.1:8001
    instance_type: qwen-image
    backends:
      - vllm-omni
      - openai
    launch:
      executable: vllm
      model: Qwen/Qwen-Image
      args:
        - --omni
        - --diffusion-scheduler-backend
        - step_level_request_scheduler
        - --diffusion-enable-step-chunk
    stop:
      executable: pkill
      args:
        - -f
        - vllm serve Qwen/Qwen-Image --port {endpoint_port}
```

## Running the Server

Start with:

```bash
python -m vllm_omni.global_scheduler.server --config /path/to/config.yaml
```

Basic smoke checks:

```bash
curl http://127.0.0.1:8089/health
curl http://127.0.0.1:8089/instances
```

## Operational Notes

- Auto-start happens at scheduler startup for instances that provide `launch`.
- Managed instances are stopped on scheduler shutdown if they provide `stop`.
- Lifecycle operations are serialized per instance.
- Reload is rejected while another reload or lifecycle operation is active.

## Known Limitations

- No scheduler-side waiting queue exists in this migration.
- No global capacity enforcement exists in this migration.
- `queue_len` is not used as a real waiting metric.
- `short_queue_runtime` needs request `estimated_cost_s` for best routing behavior.
- `/v1/videos` metadata extraction is currently based on multipart form fields.
- The colocated `benchmark` section is validated in config, but not used by the
  runtime routing path itself.
