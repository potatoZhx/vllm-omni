# Global Scheduler Ops Runbook

## Scope

This runbook covers the vLLM-Omni global scheduler operational loop:

- request routing observability (`ROUTE_*` logs)
- lifecycle auditability (`LIFECYCLE_AUDIT` logs)
- health and metrics endpoints
- safe rollback actions

## Startup

1. Prepare scheduler config (`docs/serving/examples/global_scheduler.yaml`).
2. Start service:

```bash
python -m vllm_omni.global_scheduler.server --config /path/to/global_scheduler.yaml
```

3. Check health:

```bash
curl -s http://127.0.0.1:8089/health | jq
```

Expected result:

- `status=ok`
- `checks.config_loaded=true`
- `checks.has_instances=true`

## Metrics Checks

Query runtime metrics:

```bash
curl -s http://127.0.0.1:8089/metrics | jq
```

Key fields:

- `global.request_total`, `global.request_success`, `global.request_failure`
- per-instance `queue_len`, `inflight`, `ewma_service_time_s`
- per-instance lifecycle flags: `enabled`, `healthy`, `draining`

## Request Routing Logs

Every request should emit a complete route chain:

- `ROUTE_BEGIN`: `request_id`, `policy`, `candidate_count`
- `ROUTE_DECISION`: `request_id`, `instance_id`, `reason`, `score`
- `ROUTE_DONE` or `ROUTE_FAIL`: `request_id`, `status`, `latency_s`, optional `error_code`

A single request is considered traceable only when these logs can be correlated by the same `request_id`.

## Lifecycle Audit Logs

Lifecycle APIs (`enable`, `disable`, `reload`, `probe`) emit `LIFECYCLE_AUDIT` logs with mandatory fields:

- `op`
- `instance_id` (`*` for batch operations)
- `operator` (from request header `x-operator`, default `unknown`)
- `result` (`ok` or `failed`)
- `error_code` (nullable)

## Lifecycle Conflict Semantics

`POST /instances/reload` is single-flight:

- concurrent reload returns `409`
- payload error code is `GS_LIFECYCLE_CONFLICT`

## Failure Triage

1. `GS_NO_ROUTABLE_INSTANCE`:
   - check `/instances` and `/metrics` lifecycle flags
   - verify upstream health via `POST /instances/probe`
2. `GS_UPSTREAM_TIMEOUT` / `GS_UPSTREAM_NETWORK_ERROR`:
   - check network reachability and `server.request_timeout_s`
3. `GS_UPSTREAM_HTTP_ERROR`:
   - inspect upstream app logs for non-2xx response cause

## Rollback

1. Switch algorithm to `fcfs` and reload:

```yaml
policy:
  baseline:
    algorithm: fcfs
```

2. If scheduler risk remains, bypass scheduler entry and route traffic directly to one upstream instance.
3. Keep scheduler process running for troubleshooting until incident closure.
