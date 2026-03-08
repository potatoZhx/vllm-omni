from __future__ import annotations

import argparse
import asyncio
import contextlib
from contextlib import asynccontextmanager
import json
import socket
import time
import uuid
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import BaseModel

from vllm_omni.version import __version__

from .config import GlobalSchedulerConfig, load_config
from .lifecycle import InstanceLifecycleManager
from .router import build_policy
from .state import RuntimeStateStore
from .types import InstanceSpec, RequestMeta


class UpstreamHTTPError(Exception):
    """Represents non-2xx HTTP response returned by upstream instance."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body


def _build_error_payload(code: str, message: str, request_id: str) -> dict[str, Any]:
    """Build normalized error body following GS_* contract.

    Args:
        code: Stable global scheduler error code.
        message: Human-readable error description.
        request_id: Current request identifier.

    Returns:
        Error payload object under `error` field.
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


def _extract_request_meta(payload: dict[str, Any], request_id: str) -> RequestMeta:
    """Extract scheduler metadata from OpenAI-compatible request payload.

    Args:
        payload: JSON payload from incoming request body.
        request_id: Stable request identifier used for tracing.

    Returns:
        Parsed request metadata used by routing policies.
    """
    extra_body = payload.get("extra_body") if isinstance(payload.get("extra_body"), dict) else {}

    return RequestMeta(
        request_id=request_id,
        width=extra_body.get("width"),
        height=extra_body.get("height"),
        num_frames=extra_body.get("num_frames"),
        num_inference_steps=extra_body.get("num_inference_steps"),
    )


def _filter_forward_headers(headers: Any) -> dict[str, str]:
    """Filter inbound headers before forwarding request upstream.

    Args:
        headers: Incoming request headers mapping.

    Returns:
        Sanitized headers suitable for upstream POST request.
    """
    skipped = {"host", "content-length", "connection"}
    return {key: value for key, value in headers.items() if key.lower() not in skipped}


def _proxy_chat_completion(
    endpoint: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: int,
) -> bytes:
    """Forward chat completion request to one upstream endpoint.

    Args:
        endpoint: Base upstream endpoint.
        body: Serialized request body.
        headers: Forwarded HTTP headers.
        timeout_s: Upstream request timeout in seconds.

    Returns:
        Raw response body from upstream on success.

    Raises:
        TimeoutError: Upstream timeout.
        OSError: Network-level transport failure.
        UpstreamHTTPError: Upstream returned non-2xx response.
    """
    url = f"{endpoint}/v1/chat/completions"
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as upstream_response:  # noqa: S310
            return upstream_response.read()
    except urllib_error.HTTPError as exc:
        raise UpstreamHTTPError(status_code=exc.code, body=exc.read()) from exc
    except urllib_error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("upstream request timed out") from exc
        raise OSError(f"upstream network error: {reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise TimeoutError("upstream request timed out") from exc


def _build_health_payload(config: Any) -> tuple[int, dict[str, Any]]:
    """Build health response payload from current app config state.

    Args:
        config: Current scheduler config object, or None when unavailable.

    Returns:
        Tuple of HTTP status code and JSON response payload.
    """
    checks = {
        "config_loaded": isinstance(config, GlobalSchedulerConfig),
        "has_instances": False,
    }

    instance_count = 0
    if isinstance(config, GlobalSchedulerConfig):
        instance_count = len(config.instances)
        checks["has_instances"] = instance_count > 0

    healthy = all(checks.values())
    payload: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "instance_count": instance_count,
        "checks": checks,
        "version": __version__,
    }
    return (200 if healthy else 503), payload


def _to_instance_specs(config: GlobalSchedulerConfig) -> list[InstanceSpec]:
    """Convert validated config model to instance specs used at runtime.

    Args:
        config: Global scheduler config.

    Returns:
        Instance specification list for runtime state components.
    """
    return [
        InstanceSpec(
            id=instance.id,
            endpoint=instance.endpoint,
            sp_size=instance.sp_size,
            max_concurrency=instance.max_concurrency,
        )
        for instance in config.instances
    ]


class ReloadResponse(BaseModel):
    status: str
    instance_count: int


def create_app(config: GlobalSchedulerConfig, config_loader: Any = None) -> FastAPI:
    """Create FastAPI app with scheduler lifecycle endpoints.

    Args:
        config: Initial validated scheduler config.
        config_loader: Optional callable used by reload endpoint.

    Returns:
        Configured FastAPI application instance.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _run() -> None:
            while True:
                current_config = getattr(app.state, "global_scheduler_config", config)
                timeout_s = current_config.server.instance_health_check_timeout_s
                interval_s = current_config.server.instance_health_check_interval_s

                await asyncio.to_thread(
                    app.state.instance_lifecycle_manager.probe_all,
                    timeout_s,
                )
                app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
                await asyncio.sleep(interval_s)

        app.state.health_probe_task = asyncio.create_task(_run())
        try:
            yield
        finally:
            task = getattr(app.state, "health_probe_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="vLLM-Omni Global Scheduler", version=__version__, lifespan=lifespan)
    app.state.global_scheduler_config = config
    instance_specs = _to_instance_specs(config)
    app.state.runtime_state_store = RuntimeStateStore(
        instances=instance_specs,
        ewma_alpha=config.scheduler.ewma_alpha,
    )
    app.state.instance_lifecycle_manager = InstanceLifecycleManager(instance_specs)
    app.state.policy = build_policy(config)
    app.state.config_loader = config_loader
    app.state.health_probe_task = None
    instance_specs = _to_instance_specs(config)
    app.state.runtime_state_store = RuntimeStateStore(
        instances=instance_specs,
        ewma_alpha=config.scheduler.ewma_alpha,
    )
    app.state.instance_lifecycle_manager = InstanceLifecycleManager(instance_specs)
    app.state.policy = build_policy(config)
    app.state.config_loader = config_loader
    app.state.health_probe_task = None

    @app.get("/health")
    async def health() -> JSONResponse:
        status_code, payload = _build_health_payload(getattr(app.state, "global_scheduler_config", None))
        return JSONResponse(
            status_code=status_code,
            content=payload,
        )

    @app.get("/instances")
    async def instances() -> JSONResponse:
        runtime_snapshot = app.state.runtime_state_store.snapshot()
        lifecycle_snapshot = app.state.instance_lifecycle_manager.snapshot()
        payload = []
        for instance_id, lifecycle in lifecycle_snapshot.items():
            stats = runtime_snapshot.get(instance_id)
            payload.append(
                {
                    "id": instance_id,
                    "endpoint": lifecycle.instance.endpoint,
                    "enabled": lifecycle.enabled,
                    "healthy": lifecycle.healthy,
                    "draining": lifecycle.draining,
                    "routable": lifecycle.enabled and lifecycle.healthy and not lifecycle.draining,
                    "queue_len": stats.queue_len if stats else 0,
                    "inflight": stats.inflight if stats else 0,
                    "ewma_service_time_s": stats.ewma_service_time_s if stats else None,
                    "last_check_ts_s": lifecycle.last_check_ts_s,
                    "last_error": lifecycle.last_error,
                }
            )
        return JSONResponse(status_code=200, content={"instances": payload})

    @app.post("/instances/{instance_id}/disable")
    async def disable_instance(instance_id: str) -> JSONResponse:
        try:
            status = app.state.instance_lifecycle_manager.set_enabled(instance_id, enabled=False)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            status_code=200,
            content={"id": status.instance.id, "enabled": status.enabled, "draining": status.draining},
        )

    @app.post("/instances/{instance_id}/enable")
    async def enable_instance(instance_id: str) -> JSONResponse:
        try:
            status = app.state.instance_lifecycle_manager.set_enabled(instance_id, enabled=True)
            app.state.instance_lifecycle_manager.mark_health(instance_id, healthy=True, error=None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            status_code=200,
            content={"id": status.instance.id, "enabled": status.enabled, "draining": status.draining},
        )

    @app.post("/instances/reload", response_model=ReloadResponse)
    async def reload_instances() -> ReloadResponse:
        loader = getattr(app.state, "config_loader", None)
        if loader is None:
            raise HTTPException(status_code=501, detail="config reload is not enabled")

        new_config = loader()
        app.state.global_scheduler_config = new_config
        new_instance_specs = _to_instance_specs(new_config)

        app.state.runtime_state_store.sync_instances(new_instance_specs)
        app.state.instance_lifecycle_manager.sync_instances(
            new_instance_specs,
            runtime_snapshot=app.state.runtime_state_store.snapshot(),
        )
        app.state.policy = build_policy(new_config)
        app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
        return ReloadResponse(status="ok", instance_count=len(new_instance_specs))

    @app.post("/instances/probe")
    async def probe_instances() -> JSONResponse:
        current_config = getattr(app.state, "global_scheduler_config", config)
        await asyncio.to_thread(
            app.state.instance_lifecycle_manager.probe_all,
            current_config.server.instance_health_check_timeout_s,
        )
        app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        payload = await request.json()
        request_id = request.headers.get("x-request-id") or payload.get("request_id") or str(uuid.uuid4())
        request_meta = _extract_request_meta(payload, request_id=request_id)
        runtime_snapshot = app.state.runtime_state_store.snapshot()
        candidates = app.state.instance_lifecycle_manager.get_routable_instances()
        if not candidates:
            return JSONResponse(
                status_code=503,
                content=_build_error_payload(
                    code="GS_NO_ROUTABLE_INSTANCE",
                    message="No routable instance is available",
                    request_id=request_id,
                ),
            )

        try:
            decision = app.state.policy.select_instance(
                request=request_meta,
                instances=candidates,
                runtime_stats=runtime_snapshot,
            )
        except (ValueError, KeyError) as exc:
            return JSONResponse(
                status_code=503,
                content=_build_error_payload(
                    code="GS_NO_ROUTABLE_INSTANCE",
                    message=str(exc),
                    request_id=request_id,
                ),
            )

        app.state.runtime_state_store.on_request_start(decision.instance_id)
        started_at = time.monotonic()
        current_config = getattr(app.state, "global_scheduler_config", config)
        response: Response
        try:
            response_body = await asyncio.to_thread(
                _proxy_chat_completion,
                decision.endpoint,
                json.dumps(payload).encode("utf-8"),
                _filter_forward_headers(request.headers),
                current_config.server.request_timeout_s,
            )
            response = Response(status_code=200, content=response_body, media_type="application/json")
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=True,
            )
        except UpstreamHTTPError as exc:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=False,
            )
            response = JSONResponse(
                status_code=exc.status_code,
                content=_build_error_payload(
                    code="GS_UPSTREAM_HTTP_ERROR",
                    message=f"Upstream returned HTTP {exc.status_code}",
                    request_id=request_id,
                ),
            )
        except TimeoutError:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=False,
            )
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_TIMEOUT",
                    message="Upstream request timed out",
                    request_id=request_id,
                ),
            )
        except OSError as exc:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=False,
            )
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_NETWORK_ERROR",
                    message=str(exc),
                    request_id=request_id,
                ),
            )

        response.headers["X-Routed-Instance"] = decision.instance_id
        response.headers["X-Route-Reason"] = decision.reason
        response.headers["X-Route-Score"] = str(decision.score)
        return response

    return app


def run_server(config_path: str) -> None:
    """Run scheduler API server from YAML config path.

    Args:
        config_path: Path to scheduler YAML config.
    """
    config = load_config(config_path)
    app = create_app(config, config_loader=lambda: load_config(config_path))
    app = create_app(config, config_loader=lambda: load_config(config_path))
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for scheduler standalone server entrypoint.

    Returns:
        Argument parser with scheduler server options.
    """
    parser = argparse.ArgumentParser(description="Run vLLM-Omni global scheduler server")
    parser.add_argument("--config", required=True, help="Path to global scheduler YAML config")
    return parser


def main() -> None:
    """CLI entrypoint for standalone global scheduler server."""
    args = build_arg_parser().parse_args()
    run_server(args.config)


if __name__ == "__main__":
    main()
