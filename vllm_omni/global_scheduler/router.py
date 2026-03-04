from __future__ import annotations

from vllm_omni.global_scheduler.config import GlobalSchedulerConfig
from vllm_omni.global_scheduler.policies import AlgorithmPolicyRouter


def build_policy(config: GlobalSchedulerConfig):
    if config.scheduler.type == "baseline":
        return AlgorithmPolicyRouter(
            algorithm=config.policy.baseline.algorithm,
            tie_breaker=config.scheduler.tie_breaker,
        )
    if config.scheduler.type == "ondisc":
        return AlgorithmPolicyRouter(
            algorithm="estimated_completion_time",
            tie_breaker=config.scheduler.tie_breaker,
        )
    raise ValueError("Unsupported scheduler.type. expected one of: baseline, ondisc")
