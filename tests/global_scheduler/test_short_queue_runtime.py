"""Short queue runtime policy behavior tests."""

import pytest

from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.policies.short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_short_queue_runtime_prefers_lower_estimated_outstanding_runtime():
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 2.0})
    policy = ShortQueueRuntimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(
        request_id="incoming",
        width=1280,
        height=720,
        num_frames=16,
        num_inference_steps=50,
        estimated_cost_s=2.0,
    )
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", instance_type="wan-video-tp2"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", instance_type="wan-video-tp2"),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(inflight=3, ewma_service_time_s=1.0, outstanding_runtime_s=5.0),
        "worker-1": RuntimeStats(inflight=2, ewma_service_time_s=1.0, outstanding_runtime_s=3.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(3.0)


def test_short_queue_runtime_falls_back_to_min_queue_length_without_request_estimate():
    estimator = RuntimeEstimator(profiling_data={("wan-video-tp2", 1280, 720, 16, 50): 9.0})
    policy = ShortQueueRuntimePolicy(estimator=estimator, tie_breaker="lexical")
    request = RequestMeta(request_id="incoming", width=1280, height=720, num_frames=16, num_inference_steps=50)
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", instance_type="wan-video-tp2"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", instance_type="wan-video-tp2"),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(inflight=1, ewma_service_time_s=1.0, outstanding_runtime_s=9.0),
        "worker-1": RuntimeStats(inflight=2, ewma_service_time_s=1.0, outstanding_runtime_s=2.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
    assert decision.instance_id == "worker-0"
    assert decision.score == pytest.approx(1.0)
    assert decision.reason == "algorithm=short_queue_runtime,fallback=min_queue_length"
