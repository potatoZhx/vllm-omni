from __future__ import annotations

from vllm_omni.global_scheduler.policies.policy_base import PolicyBase
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class EstimatedCompletionTimePolicy(PolicyBase):
    """Route to instance with minimum estimated completion time (ECT)."""

    def __init__(
        self,
        estimator: RuntimeEstimator,
        tie_breaker: str = "random",
    ) -> None:
        """Initialize ECT policy.

        Args:
            estimator: Runtime estimator with profiling/EWMA fallback support.
            tie_breaker: Strategy for equal-score candidates.
        """
        super().__init__(tie_breaker=tie_breaker)
        self._estimator = estimator

    def _estimate_completion_time_s(
        self,
        request: RequestMeta,
        instance: InstanceSpec,
        stats: RuntimeStats,
    ) -> float:
        service_time_s = self._estimator.estimate_runtime_s(
            request=request,
            ewma_fallback_s=stats.ewma_service_time_s,
            instance_type=instance.instance_type,
        )
        queue_runtime_s = float(stats.queue_len) * service_time_s
        return queue_runtime_s + service_time_s

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        """Select instance with lowest estimated completion time.

        Args:
            request: Parsed request metadata.
            instances: Candidate upstream instances.
            runtime_stats: Runtime snapshot for candidates.

        Returns:
            Route decision with minimum completion-time score.
        """
        if not instances:
            raise ValueError("No instances configured")

        candidates = [item for item in instances if self._is_available(item, runtime_stats)]
        if not candidates:
            candidates = list(instances)

        scored = [
            (instance, self._estimate_completion_time_s(request, instance, runtime_stats[instance.id]))
            for instance in candidates
        ]
        min_score = min(score for _, score in scored)
        tie_group = [instance for instance, score in scored if score == min_score]
        selected = tie_group[0] if len(tie_group) == 1 else self._break_tie(tie_group)

        return RouteDecision(
            instance_id=selected.id,
            endpoint=selected.endpoint,
            reason="algorithm=estimated_completion_time",
            score=min_score,
        )
