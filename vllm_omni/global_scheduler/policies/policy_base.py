from __future__ import annotations

import random
from abc import ABC, abstractmethod

from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class PolicyBase(ABC):
    def __init__(self, tie_breaker: str = "random") -> None:
        if tie_breaker not in {"random", "lexical"}:
            raise ValueError("tie_breaker must be one of: random, lexical")
        self._tie_breaker = tie_breaker

    @abstractmethod
    def select_instance(
        self,
        request: RequestMeta,
        instances: list[InstanceSpec],
        runtime_stats: dict[str, RuntimeStats],
    ) -> RouteDecision:
        raise NotImplementedError

    def _is_available(self, instance: InstanceSpec, runtime_stats: dict[str, RuntimeStats]) -> bool:
        stats = runtime_stats[instance.id]
        return stats.inflight < instance.max_concurrency

    def _break_tie(self, instances: list[InstanceSpec]) -> InstanceSpec:
        if self._tie_breaker == "lexical":
            return sorted(instances, key=lambda item: item.id)[0]
        return random.choice(instances)
