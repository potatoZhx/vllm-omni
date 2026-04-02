from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .policies.runtime_estimator import RuntimeEstimator
from .types import InstanceSpec, RequestMeta, RuntimeStats


class RuntimeStateStore:
    """Thread-safe runtime bookkeeping for routed requests.

    The global scheduler in this migration is pure routing: it does not hold a
    scheduler-side waiting queue. We only track active routed requests and the
    outstanding runtime estimate used by the routing policies.
    """

    def __init__(
        self,
        instances: list[InstanceSpec],
        ewma_alpha: float = 0.2,
        default_ewma_service_time_s: float = 1.0,
        estimator: RuntimeEstimator | None = None,
    ) -> None:
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if default_ewma_service_time_s <= 0.0:
            raise ValueError("default_ewma_service_time_s must be > 0")
        if not instances:
            raise ValueError("instances must not be empty")

        self._ewma_alpha = ewma_alpha
        self._default_ewma_service_time_s = default_ewma_service_time_s
        self._estimator = estimator or RuntimeEstimator()
        self._lock = RLock()
        self._draining_instance_ids: set[str] = set()
        self._instance_specs: dict[str, InstanceSpec] = {instance.id: instance for instance in instances}
        self._active_request_runtime_s: dict[str, dict[str, float]] = {instance.id: {} for instance in instances}
        self._stats: dict[str, RuntimeStats] = {
            instance.id: RuntimeStats(
                queue_len=0,
                inflight=0,
                ewma_service_time_s=default_ewma_service_time_s,
                waiting_requests=(),
            )
            for instance in instances
        }

    def snapshot(self) -> dict[str, RuntimeStats]:
        with self._lock:
            return {instance_id: replace(stats) for instance_id, stats in self._stats.items()}

    def sync_instances(self, instances: list[InstanceSpec]) -> None:
        with self._lock:
            desired_ids = {instance.id for instance in instances}

            for instance_id in list(self._stats):
                if instance_id in desired_ids:
                    self._draining_instance_ids.discard(instance_id)
                    continue

                stats = self._stats[instance_id]
                if stats.inflight == 0:
                    del self._stats[instance_id]
                    self._active_request_runtime_s.pop(instance_id, None)
                    self._draining_instance_ids.discard(instance_id)
                else:
                    self._draining_instance_ids.add(instance_id)

            for instance in instances:
                self._instance_specs[instance.id] = instance
                if instance.id not in self._stats:
                    self._stats[instance.id] = RuntimeStats(
                        queue_len=0,
                        inflight=0,
                        ewma_service_time_s=self._default_ewma_service_time_s,
                        waiting_requests=(),
                    )
                self._active_request_runtime_s.setdefault(instance.id, {})

            for instance_id in list(self._instance_specs):
                if instance_id not in desired_ids and instance_id not in self._stats:
                    del self._instance_specs[instance_id]

    def on_request_start(self, instance_id: str, request: RequestMeta) -> RuntimeStats:
        with self._lock:
            stats = self._get_stats(instance_id)
            reserved_runtime_s = self._estimate_request_runtime_s(instance_id, request, stats)
            self._active_request_runtime_s.setdefault(instance_id, {})[request.request_id] = reserved_runtime_s
            stats.inflight += 1
            stats.outstanding_runtime_s += reserved_runtime_s
            return replace(stats)

    def on_request_finish(
        self,
        instance_id: str,
        latency_s: float,
        ok: bool,
        request_id: str | None = None,
    ) -> RuntimeStats:
        del ok
        with self._lock:
            stats = self._get_stats(instance_id)
            reserved_runtime_s = self._pop_reserved_runtime_s(instance_id, request_id)
            stats.outstanding_runtime_s = max(0.0, stats.outstanding_runtime_s - reserved_runtime_s)
            if stats.inflight > 0:
                stats.inflight -= 1

            if latency_s >= 0.0:
                stats.ewma_service_time_s = (
                    self._ewma_alpha * latency_s + (1.0 - self._ewma_alpha) * stats.ewma_service_time_s
                )

            if instance_id in self._draining_instance_ids and stats.inflight == 0:
                self._draining_instance_ids.remove(instance_id)
                del self._stats[instance_id]
                self._active_request_runtime_s.pop(instance_id, None)
                self._instance_specs.pop(instance_id, None)
                return RuntimeStats(
                    queue_len=0,
                    inflight=0,
                    ewma_service_time_s=stats.ewma_service_time_s,
                    outstanding_runtime_s=0.0,
                )

            return replace(stats)

    def instance_has_capacity(
        self,
        instance_id: str,
        runtime_snapshot: dict[str, RuntimeStats] | None = None,
    ) -> bool:
        if runtime_snapshot is not None:
            if instance_id not in runtime_snapshot:
                raise KeyError(f"Unknown instance id: {instance_id}")
            return True

        with self._lock:
            self._get_stats(instance_id)
            return True

    def wait_for_available_capacity(self, instance_ids: list[str], timeout_s: float | None = None) -> bool:
        del timeout_s
        with self._lock:
            return any(instance_id in self._stats for instance_id in instance_ids)

    def _get_stats(self, instance_id: str) -> RuntimeStats:
        if instance_id not in self._stats:
            raise KeyError(f"Unknown instance id: {instance_id}")
        return self._stats[instance_id]

    def _estimate_request_runtime_s(self, instance_id: str, request: RequestMeta, stats: RuntimeStats) -> float:
        instance = self._instance_specs.get(instance_id)
        return self._estimator.estimate_runtime_s(
            request=request,
            ewma_fallback_s=stats.ewma_service_time_s,
            instance_type=instance.instance_type if instance is not None else None,
        )

    def _pop_reserved_runtime_s(self, instance_id: str, request_id: str | None) -> float:
        request_runtimes = self._active_request_runtime_s.setdefault(instance_id, {})
        if request_id is not None:
            return request_runtimes.pop(request_id, 0.0)
        if not request_runtimes:
            return 0.0
        first_request_id = next(iter(request_runtimes))
        return request_runtimes.pop(first_request_id, 0.0)
