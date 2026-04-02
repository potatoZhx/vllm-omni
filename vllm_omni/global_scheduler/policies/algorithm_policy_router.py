from __future__ import annotations

from .min_queue_length import MinQueueLengthPolicy
from .policy_base import PolicyBase
from .round_robin import RoundRobinPolicy
from .runtime_estimator import RuntimeEstimator
from .short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class AlgorithmPolicyRouter(PolicyBase):
    """Policy router delegating baseline algorithms by config value."""

    def __init__(
        self,
        algorithm: str,
        tie_breaker: str = "random",
        estimator: RuntimeEstimator | None = None,
    ) -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._algorithm = algorithm
        estimator = estimator or RuntimeEstimator()
        self._delegate: PolicyBase
        if algorithm == "min_queue_length":
            self._delegate = MinQueueLengthPolicy(tie_breaker=tie_breaker)
        elif algorithm == "round_robin":
            self._delegate = RoundRobinPolicy(tie_breaker=tie_breaker)
        elif algorithm == "short_queue_runtime":
            self._delegate = ShortQueueRuntimePolicy(tie_breaker=tie_breaker, estimator=estimator)
        else:
            raise ValueError(
                "Unsupported baseline algorithm. expected one of: min_queue_length, round_robin, short_queue_runtime"
            )

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        decision = self._delegate.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
        decision.reason = f"router={self._algorithm};{decision.reason}"
        return decision
