import pytest

from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.policies.short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_short_queue_runtime_prefers_lower_estimated_queue_runtime():
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.0})
    policy = ShortQueueRuntimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(request_id="r1", width=1280, height=720, num_inference_steps=50)
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=3, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)


def test_short_queue_runtime_uses_ewma_fallback_when_profile_missing():
    estimator = RuntimeEstimator(profiling_data={(1280, 720, 50): 2.0})
    policy = ShortQueueRuntimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(request_id="r2", width=640, height=360, num_inference_steps=20)
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=2, inflight=0, ewma_service_time_s=0.5),
        "worker-1": RuntimeStats(queue_len=1, inflight=0, ewma_service_time_s=1.5),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-0"
    assert decision.score == pytest.approx(1.0)
