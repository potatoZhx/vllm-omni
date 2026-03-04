from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from threading import RLock
from urllib.parse import urlparse

from .types import InstanceSpec, RuntimeStats


@dataclass(slots=True)
class InstanceLifecycleStatus:
    instance: InstanceSpec
    enabled: bool = True
    healthy: bool = True
    draining: bool = False
    last_check_ts_s: float | None = None
    last_error: str | None = None


class InstanceLifecycleManager:
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
                instance_id: InstanceLifecycleStatus(
                    instance=status.instance,
                    enabled=status.enabled,
                    healthy=status.healthy,
                    draining=status.draining,
                    last_check_ts_s=status.last_check_ts_s,
                    last_error=status.last_error,
                )
                for instance_id, status in self._instances.items()
            }

    def get_routable_instances(self) -> list[InstanceSpec]:
        with self._lock:
            return [
                status.instance
                for status in self._instances.values()
                if status.enabled and status.healthy and not status.draining
            ]

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
            return self._copy_status(status)

    def probe_all(self, timeout_s: float) -> None:
        with self._lock:
            statuses = list(self._instances.values())

        for status in statuses:
            if not status.enabled:
                continue
            healthy, error = _probe_tcp_alive(status.instance.endpoint, timeout_s)
            self.mark_health(status.instance.id, healthy=healthy, error=error)

    def sync_instances(self, instances: list[InstanceSpec], runtime_snapshot: dict[str, RuntimeStats]) -> None:
        desired = {item.id: item for item in instances}
        with self._lock:
            for instance_id, status in list(self._instances.items()):
                if instance_id in desired:
                    incoming = desired[instance_id]
                    status.instance = incoming
                    status.draining = False
                    continue

                current_runtime = runtime_snapshot.get(instance_id)
                has_pending = bool(current_runtime and (current_runtime.queue_len > 0 or current_runtime.inflight > 0))
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
                if stats is None or (stats.queue_len == 0 and stats.inflight == 0):
                    if not status.enabled:
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
            last_check_ts_s=status.last_check_ts_s,
            last_error=status.last_error,
        )


def _probe_tcp_alive(endpoint: str, timeout_s: float) -> tuple[bool, str | None]:
    try:
        parsed = urlparse(endpoint)
        if parsed.hostname is None or parsed.port is None:
            return False, "invalid_endpoint"
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout_s):
            return True, None
    except OSError as exc:
        return False, str(exc)
