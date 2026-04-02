from __future__ import annotations

from vllm_omni.global_scheduler.config import GlobalSchedulerConfig
from vllm_omni.global_scheduler.policies import AlgorithmPolicyRouter
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.runtime_profile import load_runtime_profile


def build_runtime_estimator(config: GlobalSchedulerConfig) -> RuntimeEstimator:
    runtime_profile_path = config.policy.baseline.runtime_profile_path
    return RuntimeEstimator(
        profiling_data=load_runtime_profile(runtime_profile_path) if runtime_profile_path is not None else None
    )


def build_policy(config: GlobalSchedulerConfig, estimator: RuntimeEstimator | None = None) -> AlgorithmPolicyRouter:
    estimator = estimator or build_runtime_estimator(config)
    return AlgorithmPolicyRouter(
        algorithm=config.policy.baseline.algorithm,
        tie_breaker=config.scheduler.tie_breaker,
        estimator=estimator,
    )
