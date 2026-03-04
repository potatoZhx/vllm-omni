import pytest

from vllm_omni.global_scheduler.policies.baseline_estimated_completion_time import (
    BaselineEstimatedCompletionTimePolicy,
)
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_estimated_completion_time_selects_lowest_score():
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.0})
    policy = BaselineEstimatedCompletionTimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(request_id="r1", width=1280, height=720, num_inference_steps=50)
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=2, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)


def test_estimated_completion_time_uses_fallback_when_profiling_missing():
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.0})
    policy = BaselineEstimatedCompletionTimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(request_id="r2", width=640, height=360, num_inference_steps=20)
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=1, inflight=0, ewma_service_time_s=0.5),
        "worker-1": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=2.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-0"
    assert decision.score == pytest.approx(1.0)
