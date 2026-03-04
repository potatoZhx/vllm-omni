from __future__ import annotations

import argparse
import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm_omni.version import __version__

from .config import GlobalSchedulerConfig, load_config
from .lifecycle import InstanceLifecycleManager
from .state import RuntimeStateStore
from .types import InstanceSpec


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
            sp_size=instance.sp_size,
            max_concurrency=instance.max_concurrency,
        )
        for instance in config.instances
    ]


class ReloadResponse(BaseModel):
    status: str
    instance_count: int


def create_app(config: GlobalSchedulerConfig, config_loader: Any = None) -> FastAPI:
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
