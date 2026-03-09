from __future__ import annotations

import random
from abc import ABC, abstractmethod

from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RouteDecision, RuntimeStats


class PolicyBase(ABC):
    """Base interface for global scheduler routing policies."""

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
        """Select one upstream instance for a request.

        Args:
            request: Parsed routing metadata for the incoming request.
            instances: Candidate upstream instances.
            runtime_stats: Snapshot runtime counters keyed by instance id.

        Returns:
            Route decision containing target endpoint and routing score.
        """
        raise NotImplementedError

    def _is_available(self, instance: InstanceSpec, runtime_stats: dict[str, RuntimeStats]) -> bool:
        stats = runtime_stats[instance.id]
        return stats.inflight < self._max_concurrency_from_args(instance.launch_args)

    @staticmethod
    def _max_concurrency_from_args(args: list[str]) -> int:
        keys = {"--max-concurrency"}
        for idx, item in enumerate(args):
            if "=" in item:
                key, value = item.split("=", 1)
                if key in keys:
                    try:
                        parsed = int(value)
                    except ValueError:
                        return 1
                    return max(parsed, 1)
            if item in keys and idx + 1 < len(args):
                try:
                    parsed = int(args[idx + 1])
                except ValueError:
                    return 1
                return max(parsed, 1)
        return 1

    def _break_tie(self, instances: list[InstanceSpec]) -> InstanceSpec:
        if self._tie_breaker == "lexical":
            return sorted(instances, key=lambda item: item.id)[0]
        return random.choice(instances)
