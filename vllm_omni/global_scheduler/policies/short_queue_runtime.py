from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class ShortQueueRuntimePolicy(PolicyBase):
    def __init__(
        self,
        estimator: RuntimeEstimator,
        tie_breaker: str = "random",
    ) -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._estimator = estimator

    def _estimate_queue_runtime_s(self, request: RequestMeta, stats: RuntimeStats) -> float:
        est_service_time_s = self._estimator.estimate_runtime_s(
            request=request,
            ewma_fallback_s=stats.ewma_service_time_s,
        )
        return float(stats.queue_len) * est_service_time_s

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        if not instances:
            raise ValueError("No instances configured")

        candidates = [item for item in instances if self._is_available(item, runtime_stats)]
        if not candidates:
            candidates = list(instances)

        scored = [
            (instance, self._estimate_queue_runtime_s(request, runtime_stats[instance.id]))
            for instance in candidates
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
