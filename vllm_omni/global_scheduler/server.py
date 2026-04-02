from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default as email_default_policy
from threading import RLock
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vllm_omni.version import __version__

from .config import GlobalSchedulerConfig, load_config
from .lifecycle import InstanceLifecycleManager
from .process_controller import (
    LifecycleExecutionError,
    LifecycleUnsupportedError,
    LocalProcessController,
    ProcessController,
    get_instance_log_path,
)
from .router import build_policy, build_runtime_estimator
from .state import RuntimeStateStore
from .types import InstanceSpec, RequestMeta, RouteDecision, SUPPORTED_BACKENDS

logger = logging.getLogger("vllm_omni.global_scheduler.server")
_NO_PROXY_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))


class UpstreamHTTPError(Exception):
    """Represents a non-2xx response returned by one upstream worker."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body


class UpstreamResult(BaseModel):
    """HTTP response returned by the non-stream proxy path."""

    status_code: int
    body: bytes
    headers: dict[str, str]


class ReloadResponse(BaseModel):
    status: str
    instance_count: int


class LifecycleOpResponse(BaseModel):
    id: str
    operation: str
    status: str
    process_state: str
    message: str
    log_path: str | None = None


def _build_error_payload(code: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


def _select_candidates_for_backend(all_candidates: list[InstanceSpec], backend: str) -> list[InstanceSpec]:
    if backend not in SUPPORTED_BACKENDS:
        return []
    return [instance for instance in all_candidates if not instance.backends or backend in instance.backends]


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_openai_size(size: Any) -> tuple[int | None, int | None]:
    if not isinstance(size, str):
        return None, None
    parts = size.lower().split("x", 1)
    if len(parts) != 2:
        return None, None
    return _coerce_int(parts[0]), _coerce_int(parts[1])


def _extract_request_meta_from_payload(payload: dict[str, Any], request_id: str) -> RequestMeta:
    extra_body = payload.get("extra_body") if isinstance(payload.get("extra_body"), dict) else {}

    width = _coerce_int(extra_body.get("width"))
    height = _coerce_int(extra_body.get("height"))
    num_frames = _coerce_int(extra_body.get("num_frames"))
    num_inference_steps = _coerce_int(extra_body.get("num_inference_steps"))
    estimated_cost_s = _coerce_float(extra_body.get("estimated_cost_s"))
    slo_ms = _coerce_float(extra_body.get("slo_ms"))

    if width is None:
        width = _coerce_int(payload.get("width"))
    if height is None:
        height = _coerce_int(payload.get("height"))
    if num_frames is None:
        num_frames = _coerce_int(payload.get("num_frames"))
    if num_inference_steps is None:
        num_inference_steps = _coerce_int(payload.get("num_inference_steps"))
    if estimated_cost_s is None:
        estimated_cost_s = _coerce_float(payload.get("estimated_cost_s"))
    if slo_ms is None:
        slo_ms = _coerce_float(payload.get("slo_ms"))

    if width is None or height is None:
        parsed_width, parsed_height = _parse_openai_size(payload.get("size"))
        if width is None:
            width = parsed_width
        if height is None:
            height = parsed_height

    extra: dict[str, Any] = {}
    if slo_ms is not None:
        extra["slo_ms"] = slo_ms

    return RequestMeta(
        request_id=request_id,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        estimated_cost_s=estimated_cost_s,
        extra=extra,
    )


def _extract_multipart_form_fields(body: bytes, content_type: str | None) -> dict[str, str]:
    if not body or not content_type or "multipart/form-data" not in content_type.lower():
        return {}

    mime_message = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    message = BytesParser(policy=email_default_policy).parsebytes(mime_message)
    if not message.is_multipart():
        return {}

    fields: dict[str, str] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="content-disposition")
        if not field_name or part.get_filename() is not None:
            continue
        value = part.get_content()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        fields[field_name] = str(value).strip()
    return fields


def _extract_request_meta_from_form_fields(form_fields: dict[str, str], request_id: str) -> RequestMeta:
    extra: dict[str, Any] = {}
    slo_ms = _coerce_float(form_fields.get("slo_ms"))
    if slo_ms is not None:
        extra["slo_ms"] = slo_ms
    return RequestMeta(
        request_id=request_id,
        width=_coerce_int(form_fields.get("width")),
        height=_coerce_int(form_fields.get("height")),
        num_frames=_coerce_int(form_fields.get("num_frames")),
        num_inference_steps=_coerce_int(form_fields.get("num_inference_steps")),
        estimated_cost_s=_coerce_float(form_fields.get("estimated_cost_s")),
        extra=extra,
    )


def _instance_debug_snapshot(app: FastAPI) -> list[dict[str, Any]]:
    lifecycle_snapshot = app.state.instance_lifecycle_manager.snapshot()
    runtime_snapshot = app.state.runtime_state_store.snapshot()
    payload: list[dict[str, Any]] = []
    for instance_id, lifecycle in lifecycle_snapshot.items():
        runtime = runtime_snapshot.get(instance_id)
        payload.append(
            {
                "id": instance_id,
                "enabled": lifecycle.enabled,
                "healthy": lifecycle.healthy,
                "draining": lifecycle.draining,
                "process_state": lifecycle.process_state,
                "queue_len": runtime.queue_len if runtime else 0,
                "inflight": runtime.inflight if runtime else 0,
                "last_error": lifecycle.last_error,
            }
        )
    return payload


def _no_routable_instance_response(app: FastAPI, backend: str, request_id: str) -> JSONResponse:
    logger.warning(
        "route.reject.no_routable request_id=%s backend=%s instances=%s",
        request_id,
        backend,
        _instance_debug_snapshot(app),
    )
    return JSONResponse(
        status_code=503,
        content=_build_error_payload(
            code="GS_NO_ROUTABLE_INSTANCE",
            message=f"No routable instance is available for backend {backend}",
            request_id=request_id,
        ),
    )


async def _wait_and_reserve_route(
    app: FastAPI,
    *,
    backend: str,
    request_meta: RequestMeta,
    request_id: str,
) -> tuple[RouteDecision, int] | JSONResponse:
    candidates = _select_candidates_for_backend(app.state.instance_lifecycle_manager.get_routable_instances(), backend)
    if not candidates:
        return _no_routable_instance_response(app, backend, request_id)

    async with app.state.routing_lock:
        runtime_snapshot = app.state.runtime_state_store.snapshot()
        try:
            decision = app.state.policy.select_instance(
                request=request_meta,
                instances=candidates,
                runtime_stats=runtime_snapshot,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "route.reject.selection_error request_id=%s backend=%s candidates=%s error=%s instances=%s",
                request_id,
                backend,
                [instance.id for instance in candidates],
                str(exc),
                _instance_debug_snapshot(app),
            )
            return JSONResponse(
                status_code=503,
                content=_build_error_payload(
                    code="GS_NO_ROUTABLE_INSTANCE",
                    message=str(exc),
                    request_id=request_id,
                ),
            )

        app.state.runtime_state_store.on_request_start(decision.instance_id, request=request_meta)
        return decision, len(candidates)


def _filter_forward_headers(headers: Any) -> dict[str, str]:
    skipped = {"host", "content-length", "connection"}
    return {key: value for key, value in headers.items() if key.lower() not in skipped}


def _select_upstream_response_headers(headers: Any) -> dict[str, str]:
    blocked = {"content-length", "transfer-encoding", "connection", "keep-alive"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _proxy_request(
    endpoint: str,
    upstream_path: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: int,
) -> UpstreamResult:
    url = f"{endpoint}{upstream_path}"
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with _NO_PROXY_OPENER.open(request, timeout=timeout_s) as upstream_response:  # noqa: S310
            return UpstreamResult(
                status_code=upstream_response.status,
                body=upstream_response.read(),
                headers={key: value for key, value in upstream_response.headers.items()},
            )
    except urllib_error.HTTPError as exc:
        raise UpstreamHTTPError(status_code=exc.code, body=exc.read()) from exc
    except urllib_error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("upstream request timed out") from exc
        raise OSError(f"upstream network error: {reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise TimeoutError("upstream request timed out") from exc


async def _open_streaming_upstream(
    endpoint: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: int,
    upstream_path: str = "/v1/chat/completions",
) -> tuple[int, dict[str, str], str | None, AsyncIterator[bytes]]:
    url = f"{endpoint}{upstream_path}"
    timeout = httpx.Timeout(timeout=timeout_s)
    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    stream_ctx = client.stream("POST", url=url, content=body, headers=headers)
    try:
        upstream_response = await stream_ctx.__aenter__()
    except httpx.TimeoutException as exc:
        await client.aclose()
        raise TimeoutError("upstream request timed out") from exc
    except httpx.HTTPError as exc:
        await client.aclose()
        raise OSError(f"upstream network error: {exc}") from exc

    if upstream_response.status_code >= 400:
        try:
            body_bytes = await upstream_response.aread()
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()
        raise UpstreamHTTPError(status_code=upstream_response.status_code, body=body_bytes)

    status_code = upstream_response.status_code
    response_headers = _select_upstream_response_headers(upstream_response.headers)
    media_type = upstream_response.headers.get("content-type")

    async def _body_iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()

    return status_code, response_headers, media_type, _body_iter()


def _build_health_payload(config: Any) -> tuple[int, dict[str, Any]]:
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
    return [
        InstanceSpec(
            id=instance.id,
            endpoint=instance.endpoint,
            instance_type=instance.instance_type,
            numa_node=instance.numa_node,
            launch_executable=instance.launch.executable if instance.launch is not None else None,
            launch_model=instance.launch.model if instance.launch is not None else None,
            launch_args=list(instance.launch.args) if instance.launch is not None else [],
            launch_env=dict(instance.launch.env) if instance.launch is not None else {},
            stop_executable=instance.stop.executable if instance.stop is not None else None,
            stop_args=list(instance.stop.args) if instance.stop is not None else [],
            backends=list(instance.backends),
        )
        for instance in config.instances
    ]


async def _auto_start_configured_instances(app: FastAPI) -> None:
    manager: InstanceLifecycleManager = app.state.instance_lifecycle_manager
    controller: ProcessController = app.state.process_controller

    for instance_id, status in manager.snapshot().items():
        instance_spec = status.instance
        if instance_spec.launch_executable is None or instance_spec.launch_model is None:
            continue

        manager.set_process_state(instance_id, process_state="starting", operation="start", error=None)
        try:
            await asyncio.to_thread(controller.start, instance_spec)
        except LifecycleUnsupportedError as exc:
            manager.set_process_state(instance_id, process_state="error", operation="start", error=str(exc))
            raise RuntimeError(f"auto-start unsupported for {instance_id}: {exc}") from exc
        except LifecycleExecutionError as exc:
            manager.set_process_state(instance_id, process_state="error", operation="start", error=str(exc))
            raise RuntimeError(f"auto-start failed for {instance_id}: {exc}") from exc
        except Exception as exc:
            manager.set_process_state(instance_id, process_state="error", operation="start", error=str(exc))
            raise RuntimeError(f"auto-start failed for {instance_id}: {exc}") from exc

        manager.set_enabled(instance_id, enabled=True)
        manager.mark_health(instance_id, healthy=False, error="awaiting_http_ready_after_start")
        manager.set_process_state(instance_id, process_state="running", operation="start", error=None)


async def _stop_managed_instances_on_shutdown(app: FastAPI) -> None:
    manager: InstanceLifecycleManager = app.state.instance_lifecycle_manager
    controller: ProcessController = app.state.process_controller

    for instance_id, status in manager.snapshot().items():
        instance = status.instance
        if instance.stop_executable is None or status.process_state == "stopped":
            continue

        manager.set_enabled(instance_id, enabled=False)
        manager.set_process_state(instance_id, process_state="stopping", operation="stop", error=None)
        try:
            await asyncio.to_thread(controller.stop, instance)
        except LifecycleUnsupportedError as exc:
            manager.set_process_state(instance_id, process_state="error", operation="stop", error=str(exc))
            logger.warning("shutdown.stop unsupported instance_id=%s error=%s", instance_id, exc)
            continue
        except LifecycleExecutionError as exc:
            manager.set_process_state(instance_id, process_state="error", operation="stop", error=str(exc))
            logger.warning("shutdown.stop failed instance_id=%s error=%s", instance_id, exc)
            continue
        except Exception:
            logger.exception("shutdown.stop unexpected_error instance_id=%s", instance_id)
            continue

        manager.mark_health(instance_id, healthy=False, error="stopped_on_shutdown")
        manager.set_process_state(instance_id, process_state="stopped", operation="stop", error=None)


def create_app(
    config: GlobalSchedulerConfig,
    config_loader: Any = None,
    process_controller: ProcessController | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def _run() -> None:
            while True:
                current_config = getattr(app.state, "global_scheduler_config", config)
                await asyncio.to_thread(
                    app.state.instance_lifecycle_manager.probe_all,
                    current_config.server.instance_health_check_timeout_s,
                    current_config.server.instance_health_check_failures_before_unhealthy,
                )
                app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
                await asyncio.sleep(current_config.server.instance_health_check_interval_s)

        await _auto_start_configured_instances(app)
        app.state.health_probe_task = asyncio.create_task(_run())
        try:
            yield
        finally:
            task = getattr(app.state, "health_probe_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await _stop_managed_instances_on_shutdown(app)

    app = FastAPI(title="vLLM-Omni Global Scheduler", version=__version__, lifespan=lifespan)
    app.state.global_scheduler_config = config
    instance_specs = _to_instance_specs(config)
    runtime_estimator = build_runtime_estimator(config)
    app.state.runtime_estimator = runtime_estimator
    app.state.runtime_state_store = RuntimeStateStore(
        instances=instance_specs,
        ewma_alpha=config.scheduler.ewma_alpha,
        estimator=runtime_estimator,
    )
    app.state.instance_lifecycle_manager = InstanceLifecycleManager(instance_specs)
    app.state.policy = build_policy(config, estimator=runtime_estimator)
    app.state.config_loader = config_loader
    app.state.health_probe_task = None
    app.state.process_controller = process_controller or LocalProcessController()
    app.state.reload_in_progress = False
    app.state.lifecycle_operation_locks = {}
    app.state.lifecycle_operation_locks_guard = RLock()
    app.state.routing_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> JSONResponse:
        status_code, payload = _build_health_payload(getattr(app.state, "global_scheduler_config", None))
        return JSONResponse(status_code=status_code, content=payload)

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
                    "backends": lifecycle.instance.backends,
                    "enabled": lifecycle.enabled,
                    "healthy": lifecycle.healthy,
                    "draining": lifecycle.draining,
                    "process_state": lifecycle.process_state,
                    "last_operation": lifecycle.last_operation,
                    "last_operation_ts_s": lifecycle.last_operation_ts_s,
                    "last_operation_error": lifecycle.last_operation_error,
                    "log_path": get_instance_log_path(instance_id),
                    "routable": (
                        lifecycle.enabled
                        and lifecycle.healthy
                        and not lifecycle.draining
                        and lifecycle.process_state == "running"
                    ),
                    "queue_len": stats.queue_len if stats else 0,
                    "inflight": stats.inflight if stats else 0,
                    "ewma_service_time_s": stats.ewma_service_time_s if stats else None,
                    "outstanding_runtime_s": stats.outstanding_runtime_s if stats else 0.0,
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
        if app.state.reload_in_progress:
            raise HTTPException(status_code=409, detail="reload is already in progress")
        if any(lock.locked() for lock in app.state.lifecycle_operation_locks.values()):
            raise HTTPException(status_code=409, detail="lifecycle operation in progress")

        loader = getattr(app.state, "config_loader", None)
        if loader is None:
            raise HTTPException(status_code=501, detail="config reload is not enabled")

        app.state.reload_in_progress = True
        try:
            new_config = loader()
            new_instance_specs = _to_instance_specs(new_config)
            runtime_estimator = build_runtime_estimator(new_config)
            app.state.global_scheduler_config = new_config
            app.state.runtime_estimator = runtime_estimator
            app.state.runtime_state_store._estimator = runtime_estimator
            app.state.runtime_state_store.sync_instances(new_instance_specs)
            app.state.instance_lifecycle_manager.sync_instances(
                new_instance_specs,
                runtime_snapshot=app.state.runtime_state_store.snapshot(),
            )
            app.state.policy = build_policy(new_config, estimator=runtime_estimator)
            app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
            return ReloadResponse(status="ok", instance_count=len(new_instance_specs))
        finally:
            app.state.reload_in_progress = False

    @app.post("/instances/probe")
    async def probe_instances() -> JSONResponse:
        current_config = getattr(app.state, "global_scheduler_config", config)
        await asyncio.to_thread(
            app.state.instance_lifecycle_manager.probe_all,
            current_config.server.instance_health_check_timeout_s,
            current_config.server.instance_health_check_failures_before_unhealthy,
        )
        app.state.instance_lifecycle_manager.converge_draining(app.state.runtime_state_store.snapshot())
        return JSONResponse(status_code=200, content={"status": "ok"})

    def _build_lifecycle_error(
        *,
        status_code: int,
        code: str,
        message: str,
        request_id: str,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=_build_error_payload(code=code, message=message, request_id=request_id),
        )

    def _instance_op_lock(instance_id: str) -> asyncio.Lock:
        with app.state.lifecycle_operation_locks_guard:
            lock = app.state.lifecycle_operation_locks.get(instance_id)
            if lock is None:
                lock = asyncio.Lock()
                app.state.lifecycle_operation_locks[instance_id] = lock
            return lock

    async def _run_lifecycle_operation(instance_id: str, operation: str) -> JSONResponse:
        request_id = str(uuid.uuid4())
        if app.state.reload_in_progress:
            return _build_lifecycle_error(
                status_code=409,
                code="GS_LIFECYCLE_CONFLICT",
                message="reload is in progress",
                request_id=request_id,
            )

        lock = _instance_op_lock(instance_id)
        if lock.locked():
            return _build_lifecycle_error(
                status_code=409,
                code="GS_LIFECYCLE_CONFLICT",
                message=f"{operation} conflicts with another lifecycle operation",
                request_id=request_id,
            )

        async with lock:
            snapshot = app.state.instance_lifecycle_manager.snapshot()
            status = snapshot.get(instance_id)
            if status is None:
                return _build_lifecycle_error(
                    status_code=404,
                    code="GS_UNKNOWN_INSTANCE",
                    message=f"Unknown instance id: {instance_id}",
                    request_id=request_id,
                )

            if status.process_state in {"starting", "stopping", "restarting"}:
                return _build_lifecycle_error(
                    status_code=409,
                    code="GS_LIFECYCLE_CONFLICT",
                    message=f"instance is in {status.process_state}",
                    request_id=request_id,
                )

            if operation == "start" and status.process_state == "running" and status.last_operation == "start":
                return JSONResponse(
                    status_code=200,
                    content=LifecycleOpResponse(
                        id=instance_id,
                        operation=operation,
                        status="completed",
                        process_state=status.process_state,
                        message="start skipped: already running",
                        log_path=get_instance_log_path(instance_id),
                    ).model_dump(),
                )

            manager = app.state.instance_lifecycle_manager
            controller: ProcessController = app.state.process_controller
            instance_spec = status.instance

            if operation == "stop":
                manager.set_enabled(instance_id, enabled=False)
                manager.set_process_state(instance_id, process_state="stopping", operation=operation, error=None)
                execute = controller.stop
                final_state = "stopped"
            elif operation == "start":
                manager.set_process_state(instance_id, process_state="starting", operation=operation, error=None)
                execute = controller.start
                final_state = "running"
            elif operation == "restart":
                manager.set_enabled(instance_id, enabled=False)
                manager.set_process_state(instance_id, process_state="restarting", operation=operation, error=None)
                execute = controller.restart
                final_state = "running"
            else:
                return _build_lifecycle_error(
                    status_code=400,
                    code="GS_LIFECYCLE_UNSUPPORTED",
                    message=f"unsupported operation: {operation}",
                    request_id=request_id,
                )

            try:
                await asyncio.to_thread(execute, instance_spec)
            except LifecycleUnsupportedError as exc:
                manager.set_process_state(instance_id, process_state="error", operation=operation, error=str(exc))
                return _build_lifecycle_error(
                    status_code=400,
                    code="GS_LIFECYCLE_UNSUPPORTED",
                    message=str(exc),
                    request_id=request_id,
                )
            except LifecycleExecutionError as exc:
                manager.set_process_state(instance_id, process_state="error", operation=operation, error=str(exc))
                return _build_lifecycle_error(
                    status_code=502,
                    code="GS_LIFECYCLE_EXEC_ERROR",
                    message=str(exc),
                    request_id=request_id,
                )
            except Exception as exc:
                manager.set_process_state(instance_id, process_state="error", operation=operation, error=str(exc))
                return _build_lifecycle_error(
                    status_code=502,
                    code="GS_LIFECYCLE_EXEC_ERROR",
                    message=str(exc),
                    request_id=request_id,
                )

            if operation in {"start", "restart"}:
                manager.set_enabled(instance_id, enabled=True)
                manager.mark_health(instance_id, healthy=False, error=f"awaiting_probe_after_{operation}")
            else:
                manager.mark_health(instance_id, healthy=False, error="stopped_by_operator")

            finished = manager.set_process_state(instance_id, process_state=final_state, operation=operation, error=None)
            return JSONResponse(
                status_code=200,
                content=LifecycleOpResponse(
                    id=instance_id,
                    operation=operation,
                    status="completed",
                    process_state=finished.process_state,
                    message=f"{operation} completed",
                    log_path=get_instance_log_path(instance_id),
                ).model_dump(),
            )

    @app.post("/instances/{instance_id}/stop", response_model=LifecycleOpResponse)
    async def stop_instance(instance_id: str) -> JSONResponse:
        return await _run_lifecycle_operation(instance_id=instance_id, operation="stop")

    @app.post("/instances/{instance_id}/start", response_model=LifecycleOpResponse)
    async def start_instance(instance_id: str) -> JSONResponse:
        return await _run_lifecycle_operation(instance_id=instance_id, operation="start")

    @app.post("/instances/{instance_id}/restart", response_model=LifecycleOpResponse)
    async def restart_instance(instance_id: str) -> JSONResponse:
        return await _run_lifecycle_operation(instance_id=instance_id, operation="restart")

    def _attach_route_headers(response: Response, decision: RouteDecision) -> Response:
        response.headers["X-Routed-Instance"] = decision.instance_id
        response.headers["X-Route-Reason"] = decision.reason
        response.headers["X-Route-Score"] = str(decision.score)
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        payload = await request.json()
        request_id = request.headers.get("x-request-id") or payload.get("request_id") or str(uuid.uuid4())
        request_meta = _extract_request_meta_from_payload(payload, request_id=request_id)
        route_result = await _wait_and_reserve_route(
            app,
            backend="vllm-omni",
            request_meta=request_meta,
            request_id=request_id,
        )
        if isinstance(route_result, JSONResponse):
            return route_result

        decision, _ = route_result
        started_at = time.monotonic()
        current_config = getattr(app.state, "global_scheduler_config", config)
        filtered_headers = _filter_forward_headers(request.headers)
        payload_bytes = json.dumps(payload).encode("utf-8")
        is_stream = isinstance(payload, dict) and bool(payload.get("stream"))

        if is_stream:
            ok = False
            try:
                status_code, upstream_headers, upstream_media_type, body_iter = await _open_streaming_upstream(
                    decision.endpoint,
                    payload_bytes,
                    filtered_headers,
                    current_config.server.request_timeout_s,
                )
                ok = True
            except UpstreamHTTPError as exc:
                app.state.runtime_state_store.on_request_finish(
                    decision.instance_id,
                    latency_s=time.monotonic() - started_at,
                    ok=False,
                    request_id=request_id,
                )
                return _attach_route_headers(
                    JSONResponse(
                        status_code=exc.status_code,
                        content=_build_error_payload(
                            code="GS_UPSTREAM_HTTP_ERROR",
                            message=f"Upstream returned HTTP {exc.status_code}",
                            request_id=request_id,
                        ),
                    ),
                    decision,
                )
            except TimeoutError:
                app.state.runtime_state_store.on_request_finish(
                    decision.instance_id,
                    latency_s=time.monotonic() - started_at,
                    ok=False,
                    request_id=request_id,
                )
                return _attach_route_headers(
                    JSONResponse(
                        status_code=502,
                        content=_build_error_payload(
                            code="GS_UPSTREAM_TIMEOUT",
                            message="Upstream request timed out",
                            request_id=request_id,
                        ),
                    ),
                    decision,
                )
            except OSError as exc:
                app.state.runtime_state_store.on_request_finish(
                    decision.instance_id,
                    latency_s=time.monotonic() - started_at,
                    ok=False,
                    request_id=request_id,
                )
                return _attach_route_headers(
                    JSONResponse(
                        status_code=502,
                        content=_build_error_payload(
                            code="GS_UPSTREAM_NETWORK_ERROR",
                            message=str(exc),
                            request_id=request_id,
                        ),
                    ),
                    decision,
                )

            async def _tracked_stream() -> AsyncIterator[bytes]:
                stream_ok = ok
                try:
                    async for chunk in body_iter:
                        yield chunk
                except Exception:
                    stream_ok = False
                    raise
                finally:
                    app.state.runtime_state_store.on_request_finish(
                        decision.instance_id,
                        latency_s=time.monotonic() - started_at,
                        ok=stream_ok,
                        request_id=request_id,
                    )

            response = StreamingResponse(
                _tracked_stream(),
                status_code=status_code,
                media_type=upstream_media_type or "text/event-stream",
            )
            for key, value in upstream_headers.items():
                response.headers[key] = value
            return _attach_route_headers(response, decision)

        ok = False
        try:
            upstream_result = await asyncio.to_thread(
                _proxy_request,
                decision.endpoint,
                "/v1/chat/completions",
                payload_bytes,
                filtered_headers,
                current_config.server.request_timeout_s,
            )
            if isinstance(upstream_result, bytes):
                response = Response(status_code=200, content=upstream_result, media_type="application/json")
            else:
                response = Response(
                    status_code=upstream_result.status_code,
                    content=upstream_result.body,
                    media_type=upstream_result.headers.get("content-type") or "application/json",
                )
                for key, value in _select_upstream_response_headers(upstream_result.headers).items():
                    response.headers[key] = value
            ok = True
        except UpstreamHTTPError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content=_build_error_payload(
                    code="GS_UPSTREAM_HTTP_ERROR",
                    message=f"Upstream returned HTTP {exc.status_code}",
                    request_id=request_id,
                ),
            )
        except TimeoutError:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_TIMEOUT",
                    message="Upstream request timed out",
                    request_id=request_id,
                ),
            )
        except OSError as exc:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_NETWORK_ERROR",
                    message=str(exc),
                    request_id=request_id,
                ),
            )
        finally:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=ok,
                request_id=request_id,
            )

        return _attach_route_headers(response, decision)

    @app.post("/v1/images/generations")
    async def image_generations(request: Request) -> Response:
        payload = await request.json()
        request_id = request.headers.get("x-request-id") or payload.get("request_id") or str(uuid.uuid4())
        request_meta = _extract_request_meta_from_payload(payload, request_id=request_id)
        route_result = await _wait_and_reserve_route(
            app,
            backend="openai",
            request_meta=request_meta,
            request_id=request_id,
        )
        if isinstance(route_result, JSONResponse):
            return route_result

        decision, _ = route_result
        started_at = time.monotonic()
        current_config = getattr(app.state, "global_scheduler_config", config)
        filtered_headers = _filter_forward_headers(request.headers)
        payload_bytes = json.dumps(payload).encode("utf-8")

        ok = False
        try:
            upstream_result = await asyncio.to_thread(
                _proxy_request,
                decision.endpoint,
                "/v1/images/generations",
                payload_bytes,
                filtered_headers,
                current_config.server.request_timeout_s,
            )
            response = Response(
                status_code=upstream_result.status_code,
                content=upstream_result.body,
                media_type=upstream_result.headers.get("content-type") or "application/json",
            )
            for key, value in _select_upstream_response_headers(upstream_result.headers).items():
                response.headers[key] = value
            ok = True
        except UpstreamHTTPError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content=_build_error_payload(
                    code="GS_UPSTREAM_HTTP_ERROR",
                    message=f"Upstream returned HTTP {exc.status_code}",
                    request_id=request_id,
                ),
            )
        except TimeoutError:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_TIMEOUT",
                    message="Upstream request timed out",
                    request_id=request_id,
                ),
            )
        except OSError as exc:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_NETWORK_ERROR",
                    message=str(exc),
                    request_id=request_id,
                ),
            )
        finally:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=ok,
                request_id=request_id,
            )

        return _attach_route_headers(response, decision)

    @app.post("/v1/videos")
    async def videos(request: Request) -> Response:
        payload_bytes = await request.body()
        form_fields = _extract_multipart_form_fields(payload_bytes, request.headers.get("content-type"))
        request_id = request.headers.get("x-request-id") or form_fields.get("request_id") or str(uuid.uuid4())
        request_meta = _extract_request_meta_from_form_fields(form_fields, request_id=request_id)
        route_result = await _wait_and_reserve_route(
            app,
            backend="v1/videos",
            request_meta=request_meta,
            request_id=request_id,
        )
        if isinstance(route_result, JSONResponse):
            return route_result

        decision, _ = route_result
        started_at = time.monotonic()
        current_config = getattr(app.state, "global_scheduler_config", config)
        filtered_headers = _filter_forward_headers(request.headers)

        ok = False
        try:
            upstream_result = await asyncio.to_thread(
                _proxy_request,
                decision.endpoint,
                "/v1/videos",
                payload_bytes,
                filtered_headers,
                current_config.server.request_timeout_s,
            )
            response = Response(
                status_code=upstream_result.status_code,
                content=upstream_result.body,
                media_type=upstream_result.headers.get("content-type") or "application/json",
            )
            for key, value in _select_upstream_response_headers(upstream_result.headers).items():
                response.headers[key] = value
            ok = True
        except UpstreamHTTPError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content=_build_error_payload(
                    code="GS_UPSTREAM_HTTP_ERROR",
                    message=f"Upstream returned HTTP {exc.status_code}",
                    request_id=request_id,
                ),
            )
        except TimeoutError:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_TIMEOUT",
                    message="Upstream request timed out",
                    request_id=request_id,
                ),
            )
        except OSError as exc:
            response = JSONResponse(
                status_code=502,
                content=_build_error_payload(
                    code="GS_UPSTREAM_NETWORK_ERROR",
                    message=str(exc),
                    request_id=request_id,
                ),
            )
        finally:
            app.state.runtime_state_store.on_request_finish(
                decision.instance_id,
                latency_s=time.monotonic() - started_at,
                ok=ok,
                request_id=request_id,
            )

        return _attach_route_headers(response, decision)

    return app


def run_server(config_path: str) -> None:
    config = load_config(config_path)
    app = create_app(config, config_loader=lambda: load_config(config_path))
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vLLM-Omni global scheduler server")
    parser.add_argument("--config", required=True, help="Path to global scheduler YAML config")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_server(args.config)


if __name__ == "__main__":
    main()
