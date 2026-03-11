from __future__ import annotations

from vllm_omni.global_scheduler.config import GlobalSchedulerConfig
from vllm_omni.global_scheduler.policies import AlgorithmPolicyRouter
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.runtime_profile import load_runtime_profile


def build_policy(config: GlobalSchedulerConfig):
    """Build baseline policy router from validated scheduler config.

    Args:
        config: Loaded global scheduler config.

    Returns:
        Algorithm policy router configured by `policy.baseline.algorithm`.
    """
    runtime_profile_path = config.policy.baseline.runtime_profile_path
    estimator = RuntimeEstimator(
        profiling_data=load_runtime_profile(runtime_profile_path) if runtime_profile_path is not None else None
    )

    return AlgorithmPolicyRouter(
        algorithm=config.policy.baseline.algorithm,
        tie_breaker=config.scheduler.tie_breaker,
        estimator=estimator,
    )
