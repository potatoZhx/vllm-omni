from __future__ import annotations

from vllm_omni.global_scheduler.policies.min_queue_length import MinQueueLengthPolicy
from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class ShortQueueRuntimePolicy(PolicyBase):
    """Route to the instance with minimum estimated outstanding runtime."""

    def __init__(self, estimator: RuntimeEstimator, tie_breaker: str = "random") -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._estimator = estimator
        self._min_queue_length_policy = MinQueueLengthPolicy(tie_breaker=tie_breaker)

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        if not instances:
            raise ValueError("No instances configured")

        if request.estimated_cost_s is None:
            fallback_decision = self._min_queue_length_policy.select_instance(
                request=request,
                instances=instances,
                runtime_stats=runtime_stats,
            )
            return RouteDecision(
                instance_id=fallback_decision.instance_id,
                endpoint=fallback_decision.endpoint,
                reason="algorithm=short_queue_runtime,fallback=min_queue_length",
                score=fallback_decision.score,
            )

        scored = [
            (instance, runtime_stats[instance.id].outstanding_runtime_s)
            for instance in instances
        ]
        min_score = min(score for _, score in scored)
        tie_group = [instance for instance, score in scored if score == min_score]
        selected = tie_group[0] if len(tie_group) == 1 else self._break_tie(tie_group)

        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason="algorithm=short_queue_runtime",
            score=min_score,
        )
