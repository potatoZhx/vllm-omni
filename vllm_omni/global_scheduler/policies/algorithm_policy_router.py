from __future__ import annotations

from .estimated_completion_time import EstimatedCompletionTimePolicy
from .first_come_first_served import FirstComeFirstServedPolicy
from .policy_base import PolicyBase
from .runtime_estimator import RuntimeEstimator
from .short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class AlgorithmPolicyRouter(PolicyBase):
    def __init__(self, algorithm: str, tie_breaker: str = "random") -> None:
        super().__init__(tie_breaker=tie_breaker)
        self._algorithm = algorithm
        self._delegate: PolicyBase
        if algorithm == "fcfs":
            self._delegate = FirstComeFirstServedPolicy(tie_breaker=tie_breaker)
        elif algorithm == "short_queue_runtime":
            self._delegate = ShortQueueRuntimePolicy(
                tie_breaker=tie_breaker,
                estimator=RuntimeEstimator(),
            )
        elif algorithm == "estimated_completion_time":
            self._delegate = EstimatedCompletionTimePolicy(
                tie_breaker=tie_breaker,
                estimator=RuntimeEstimator(),
            )
        else:
            raise ValueError(
                "Unsupported baseline algorithm. expected one of: fcfs, short_queue_runtime, estimated_completion_time"
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
