from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from threading import RLock
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from .types import InstanceSpec, RuntimeStats

PROCESS_STATES = {"running", "stopped", "stopping", "starting", "restarting", "error"}
_READY_PATH = "/v1/models"
_RUNTIME_HEALTH_PATH = "/health"
_NO_PROXY_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))


@dataclass(slots=True)
class InstanceLifecycleStatus:
    """Lifecycle and health state tracked for one instance."""

    instance: InstanceSpec
    enabled: bool = True
    healthy: bool = True
    draining: bool = False
    process_state: str = "running"
    last_operation: str = "none"
    last_operation_ts_s: float | None = None
    last_operation_error: str | None = None
    last_check_ts_s: float | None = None
    last_error: str | None = None
    consecutive_probe_failures: int = 0


class InstanceLifecycleManager:
    """Manage per-instance lifecycle, health, and routability state."""

    def __init__(self, instances: list[InstanceSpec]) -> None:
        if not instances:
            raise ValueError("instances must not be empty")

        self._lock = RLock()
        self._instances: dict[str, InstanceLifecycleStatus] = {
            item.id: InstanceLifecycleStatus(instance=item) for item in instances
        }

    def snapshot(self) -> dict[str, InstanceLifecycleStatus]:
        with self._lock:
            return {
                instance_id: self._copy_status(status)
                for instance_id, status in self._instances.items()
            }

    def get_routable_instances(self) -> list[InstanceSpec]:
        with self._lock:
            return [
                status.instance
                for status in self._instances.values()
                if status.enabled and status.healthy and not status.draining and status.process_state == "running"
            ]

    def set_process_state(
        self,
        instance_id: str,
        process_state: str,
        operation: str | None = None,
        error: str | None = None,
    ) -> InstanceLifecycleStatus:
        if process_state not in PROCESS_STATES:
            raise ValueError(f"invalid process_state: {process_state}")
        with self._lock:
            status = self._get_status(instance_id)
            status.process_state = process_state
            if operation is not None:
                status.last_operation = operation
                status.last_operation_ts_s = time.time()
            status.last_operation_error = error
            return self._copy_status(status)

    def set_enabled(self, instance_id: str, enabled: bool) -> InstanceLifecycleStatus:
        with self._lock:
            status = self._get_status(instance_id)
            status.enabled = enabled
            status.draining = not enabled
            return self._copy_status(status)

    def mark_health(self, instance_id: str, healthy: bool, error: str | None = None) -> InstanceLifecycleStatus:
        with self._lock:
            status = self._get_status(instance_id)
            status.healthy = healthy
            status.last_check_ts_s = time.time()
            status.last_error = error
            status.consecutive_probe_failures = 0
            return self._copy_status(status)

    def record_probe_result(
        self,
        instance_id: str,
        *,
        healthy: bool,
        error: str | None = None,
        unhealthy_after_failures: int = 3,
    ) -> InstanceLifecycleStatus:
        if unhealthy_after_failures < 1:
            raise ValueError("unhealthy_after_failures must be >= 1")

        with self._lock:
            status = self._get_status(instance_id)
            status.last_check_ts_s = time.time()
            if healthy:
                status.healthy = True
                status.last_error = None
                status.consecutive_probe_failures = 0
                return self._copy_status(status)

            status.last_error = error
            status.consecutive_probe_failures += 1
            if status.consecutive_probe_failures >= unhealthy_after_failures:
                status.healthy = False
            return self._copy_status(status)

    def probe_all(self, timeout_s: float, unhealthy_after_failures: int = 3) -> None:
        with self._lock:
            statuses = list(self._instances.values())

        for status in statuses:
            if not status.enabled:
                continue
            if _should_use_readiness_probe(status):
                healthy, error = _probe_http_ready(status.instance.endpoint, timeout_s)
            else:
                healthy, error = _probe_http_health(status.instance.endpoint, timeout_s)
            self.record_probe_result(
                status.instance.id,
                healthy=healthy,
                error=error,
                unhealthy_after_failures=unhealthy_after_failures,
            )

    def sync_instances(self, instances: list[InstanceSpec], runtime_snapshot: dict[str, RuntimeStats]) -> None:
        desired = {item.id: item for item in instances}
        with self._lock:
            for instance_id, status in list(self._instances.items()):
                if instance_id in desired:
                    status.instance = desired[instance_id]
                    continue

                current_runtime = runtime_snapshot.get(instance_id)
                has_pending = bool(current_runtime and current_runtime.inflight > 0)
                if has_pending:
                    status.enabled = False
                    status.draining = True
                    status.last_error = "removed_by_reload_draining"
                else:
                    del self._instances[instance_id]

            for instance_id, instance in desired.items():
                if instance_id not in self._instances:
                    self._instances[instance_id] = InstanceLifecycleStatus(instance=instance)

    def converge_draining(self, runtime_snapshot: dict[str, RuntimeStats]) -> None:
        with self._lock:
            for instance_id, status in list(self._instances.items()):
                if not status.draining:
                    continue
                stats = runtime_snapshot.get(instance_id)
                if stats is None or stats.inflight == 0:
                    if status.last_error == "removed_by_reload_draining":
                        del self._instances[instance_id]
                    else:
                        status.draining = False

    def _get_status(self, instance_id: str) -> InstanceLifecycleStatus:
        if instance_id not in self._instances:
            raise KeyError(f"Unknown instance id: {instance_id}")
        return self._instances[instance_id]

    @staticmethod
    def _copy_status(status: InstanceLifecycleStatus) -> InstanceLifecycleStatus:
        return InstanceLifecycleStatus(
            instance=status.instance,
            enabled=status.enabled,
            healthy=status.healthy,
            draining=status.draining,
            process_state=status.process_state,
            last_operation=status.last_operation,
            last_operation_ts_s=status.last_operation_ts_s,
            last_operation_error=status.last_operation_error,
            last_check_ts_s=status.last_check_ts_s,
            last_error=status.last_error,
            consecutive_probe_failures=status.consecutive_probe_failures,
        )


def _should_use_readiness_probe(status: InstanceLifecycleStatus) -> bool:
    if status.last_error == "awaiting_http_ready_after_start":
        return True
    return bool(status.last_error and status.last_error.startswith("awaiting_probe_after_"))


def _probe_http_health(endpoint: str, timeout_s: float) -> tuple[bool, str | None]:
    try:
        parsed = urlparse(endpoint)
        if parsed.hostname is None or parsed.port is None:
            return False, "invalid_endpoint"
        request = urllib_request.Request(url=f"{endpoint.rstrip('/')}{_RUNTIME_HEALTH_PATH}", method="GET")
        with _NO_PROXY_OPENER.open(request, timeout=timeout_s):  # noqa: S310
            return True, None
    except urllib_error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except urllib_error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return False, "runtime_probe_timeout"
        return False, str(reason)
    except OSError as exc:
        return False, str(exc)


def _probe_http_ready(endpoint: str, timeout_s: float) -> tuple[bool, str | None]:
    try:
        parsed = urlparse(endpoint)
        if parsed.hostname is None or parsed.port is None:
            return False, "invalid_endpoint"
        request = urllib_request.Request(url=f"{endpoint.rstrip('/')}{_READY_PATH}", method="GET")
        with _NO_PROXY_OPENER.open(request, timeout=timeout_s) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data")
        if isinstance(models, list) and len(models) > 0:
            return True, None
        return False, "ready_probe_empty_models"
    except urllib_error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except urllib_error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return False, "ready_probe_timeout"
        return False, str(reason)
    except json.JSONDecodeError:
        return False, "ready_probe_invalid_json"
    except OSError as exc:
        return False, str(exc)
