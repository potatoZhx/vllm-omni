from __future__ import annotations

from .estimated_completion_time import EstimatedCompletionTimePolicy
from .first_come_first_served import FirstComeFirstServedPolicy
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
        """Build router and instantiate selected baseline policy.

        Args:
            algorithm: Baseline algorithm name.
            tie_breaker: Strategy for equal-score candidates.
            estimator: Optional runtime estimator shared by profiling-aware policies.
        """
        super().__init__(tie_breaker=tie_breaker)
        self._algorithm = algorithm
        estimator = estimator or RuntimeEstimator()
        self._delegate: PolicyBase
        if algorithm == "fcfs":
            self._delegate = FirstComeFirstServedPolicy(tie_breaker=tie_breaker)
        elif algorithm == "min_queue_length":
            self._delegate = MinQueueLengthPolicy(tie_breaker=tie_breaker)
        elif algorithm == "round_robin":
            self._delegate = RoundRobinPolicy(tie_breaker=tie_breaker)
        elif algorithm == "short_queue_runtime":
            self._delegate = ShortQueueRuntimePolicy(
                tie_breaker=tie_breaker,
                estimator=estimator,
            )
        elif algorithm == "estimated_completion_time":
            self._delegate = EstimatedCompletionTimePolicy(
                tie_breaker=tie_breaker,
                estimator=estimator,
            )
        else:
            raise ValueError(
                "Unsupported baseline algorithm. expected one of: fcfs, min_queue_length, round_robin, short_queue_runtime, estimated_completion_time"
            )

    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        """Delegate request routing to selected baseline policy.

        Args:
            request: Parsed request metadata.
            instances: Candidate upstream instances.
            runtime_stats: Runtime snapshot for candidates.

        Returns:
            Route decision augmented with router algorithm tag.
        """
        decision = self._delegate.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
        decision.reason = f"router={self._algorithm};{decision.reason}"
        return decision
