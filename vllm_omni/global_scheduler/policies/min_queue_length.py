from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class MinQueueLengthPolicy(PolicyBase):
    """Route to the instance with the smallest outstanding request count."""

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        """Select the instance with the fewest outstanding requests.

        Outstanding requests are counted as ``RuntimeStats.inflight +
        RuntimeStats.queue_len`` so the policy reflects both running and waiting
        work. The policy prefers available instances first; if all are busy it
        falls back to the full instance set.
        """
        del request
        if not instances:
            raise ValueError("No instances configured")

        candidates = [item for item in instances if self._is_available(item, runtime_stats)]
        if not candidates:
            candidates = list(instances)

        scored = [
            (
                instance,
                float(runtime_stats[instance.id].inflight + runtime_stats[instance.id].queue_len),
            )
            for instance in candidates
        ]
        min_score = min(score for _, score in scored)
        tie_group = [instance for instance, score in scored if score == min_score]
        selected = tie_group[0] if len(tie_group) == 1 else self._break_tie(tie_group)

        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason="algorithm=min_queue_length",
            score=min_score,
        )
