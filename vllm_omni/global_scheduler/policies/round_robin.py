from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class RoundRobinPolicy(PolicyBase):
    """Route requests in round-robin order."""

    def __init__(self, tie_breaker: str = "random") -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._cursor = 0

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        del request
        if not instances:
            raise ValueError("No instances configured")

        total = len(instances)
        selected = instances[self._cursor % total]
        self._cursor = (self._cursor + 1) % total
        selected_stats = runtime_stats[selected.id]
        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason="algorithm=round_robin",
            score=float(selected_stats.inflight),
        )
