from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class FirstComeFirstServedPolicy(PolicyBase):
    """FCFS baseline policy with load-aware fallback tie handling."""

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        """Choose first available instance, fallback to lowest inflight.

        Args:
            request: Request metadata (unused by FCFS).
            instances: Candidate instances in input order.
            runtime_stats: Runtime snapshot for candidate instances.

        Returns:
            Route decision with selected instance and score.
        """
        del request
        if not instances:
            raise ValueError("No instances configured")

        available = [item for item in instances if self._is_available(item, runtime_stats)]
        if available:
            selected = available[0]
            selected_stats = runtime_stats[selected.id]
            return RouteDecision(
                instance_id=selected.id,
                endpoint=selected.endpoint,
                reason="algorithm=fcfs,available=true",
                score=float(selected_stats.inflight),
            )

        lowest_inflight = min(runtime_stats[item.id].inflight for item in instances)
        tie_group = [item for item in instances if runtime_stats[item.id].inflight == lowest_inflight]
        selected = self._break_tie(tie_group)
        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason="algorithm=fcfs,available=false",
            score=float(runtime_stats[selected.id].inflight),
        )
