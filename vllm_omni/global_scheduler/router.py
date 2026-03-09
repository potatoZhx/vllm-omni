from __future__ import annotations

from vllm_omni.global_scheduler.config import GlobalSchedulerConfig
from vllm_omni.global_scheduler.policies import AlgorithmPolicyRouter


def build_policy(config: GlobalSchedulerConfig):
    """Build baseline policy router from validated scheduler config.

    Args:
        config: Loaded global scheduler config.

    Returns:
        Algorithm policy router configured by `policy.baseline.algorithm`.
    """
    return AlgorithmPolicyRouter(
        algorithm=config.policy.baseline.algorithm,
        tie_breaker=config.scheduler.tie_breaker,
    )
