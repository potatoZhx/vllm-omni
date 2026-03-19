"""Minimum queue length policy behavior tests."""

import pytest

from vllm_omni.global_scheduler.policies.min_queue_length import MinQueueLengthPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_min_queue_length_prefers_smaller_total_outstanding_requests():
    """Policy should prefer the instance with fewer outstanding requests."""
    policy = MinQueueLengthPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r1")
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", launch_args=["--max-concurrency", "1000"]),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", launch_args=["--max-concurrency", "1000"]),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=3, inflight=0, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)


def test_min_queue_length_considers_inflight_when_queue_lengths_are_equal():
    """Policy should compare running requests even when both waiting queues are empty."""
    policy = MinQueueLengthPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r1")
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", launch_args=["--max-concurrency", "1000"]),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", launch_args=["--max-concurrency", "1000"]),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=0, inflight=8, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=2, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)


def test_min_queue_length_falls_back_to_lowest_total_outstanding_when_needed():
    """Policy should still pick the lowest outstanding count when all instances are busy."""
    policy = MinQueueLengthPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r2")
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", launch_args=["--max-concurrency", "1"]),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", launch_args=["--max-concurrency", "1"]),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=2, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)
    assert "algorithm=min_queue_length" in decision.reason
