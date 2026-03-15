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
_NO_PROXY_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))


@dataclass(slots=True)
class InstanceLifecycleStatus:
    """Lifecycle and health state tracked for a single instance."""

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


class InstanceLifecycleManager:
    """Manage per-instance lifecycle, health, and routability state."""

    def __init__(self, instances: list[InstanceSpec]) -> None:
        """Initialize lifecycle manager with configured instances.

        Args:
            instances: Initial instance specs from scheduler config.
        """
        if not instances:
            raise ValueError("instances must not be empty")

        self._lock = RLock()
        self._instances: dict[str, InstanceLifecycleStatus] = {
            item.id: InstanceLifecycleStatus(instance=item) for item in instances
        }

    def snapshot(self) -> dict[str, InstanceLifecycleStatus]:
        """Return an immutable snapshot of lifecycle state.

        Returns:
            Mapping from instance id to copied lifecycle status.
        """
        with self._lock:
            return {
                instance_id: InstanceLifecycleStatus(
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
                )
                for instance_id, status in self._instances.items()
            }

    def get_routable_instances(self) -> list[InstanceSpec]:
        """List instances currently eligible for routing.

        Returns:
            Instances that are enabled, healthy, and not draining.
        """
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
        """Update process lifecycle state for one instance."""
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
        """Set operator enabled flag for an instance.

        Args:
            instance_id: Target instance id.
            enabled: Whether the instance should accept new traffic.

        Returns:
            A copied lifecycle status after update.
        """
        with self._lock:
            status = self._get_status(instance_id)
            status.enabled = enabled
            status.draining = not enabled
            return self._copy_status(status)

    def mark_health(self, instance_id: str, healthy: bool, error: str | None = None) -> InstanceLifecycleStatus:
        """Update health-check result for an instance.

        Args:
            instance_id: Target instance id.
            healthy: Probing result.
            error: Optional health-check error message.

        Returns:
            A copied lifecycle status after update.
        """
        with self._lock:
            status = self._get_status(instance_id)
            status.healthy = healthy
            status.last_check_ts_s = time.time()
            status.last_error = error
            return self._copy_status(status)

    def probe_all(self, timeout_s: float) -> None:
        """Probe all enabled instances and update their health status.

        Args:
            timeout_s: TCP probing timeout in seconds.
        """
        with self._lock:
            statuses = list(self._instances.values())

        for status in statuses:
            if not status.enabled:
                continue
            healthy, error = _probe_http_ready(status.instance.endpoint, timeout_s)
            self.mark_health(status.instance.id, healthy=healthy, error=error)

    def sync_instances(self, instances: list[InstanceSpec], runtime_snapshot: dict[str, RuntimeStats]) -> None:
        """Synchronize lifecycle entries with latest configured instances.

        Existing desired instances keep operator state (`enabled`/`draining`) so
        reload does not override manual lifecycle operations. Instances removed by
        reload are put into draining when they still have pending runtime work.

        Args:
            instances: New configured instances after reload.
            runtime_snapshot: Runtime counters used to determine pending work.
        """
        desired = {item.id: item for item in instances}
        with self._lock:
            for instance_id, status in list(self._instances.items()):
                if instance_id in desired:
                    incoming = desired[instance_id]
                    status.instance = incoming
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
        """Converge draining instances based on current runtime stats.

        Args:
            runtime_snapshot: Runtime counters for all tracked instances.
        """
        with self._lock:
            for instance_id, status in list(self._instances.items()):
                if not status.draining:
                    continue
                stats = runtime_snapshot.get(instance_id)
                if stats is None or (stats.queue_len == 0 and stats.inflight == 0):
                    if status.last_error == "removed_by_reload_draining":
                        del self._instances[instance_id]
                    else:
                        status.draining = False

    def _get_status(self, instance_id: str) -> InstanceLifecycleStatus:
        """Fetch mutable lifecycle status for one instance.

        Args:
            instance_id: Target instance id.

        Returns:
            Internal mutable lifecycle status object.
        """
        if instance_id not in self._instances:
            raise KeyError(f"Unknown instance id: {instance_id}")
        return self._instances[instance_id]

    @staticmethod
    def _copy_status(status: InstanceLifecycleStatus) -> InstanceLifecycleStatus:
        """Create a detached copy of lifecycle status.

        Args:
            status: Source lifecycle status.

        Returns:
            Copied lifecycle status safe for external consumers.
        """
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
        )


def _probe_http_ready(endpoint: str, timeout_s: float) -> tuple[bool, str | None]:
    """Probe endpoint HTTP readiness for health-checking.

    Args:
        endpoint: Upstream endpoint in `http://host:port` format.
        timeout_s: HTTP readiness timeout in seconds.

    Returns:
        Tuple `(healthy, error_message)` where error is `None` on success.
    """
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
