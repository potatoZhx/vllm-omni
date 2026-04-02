"""Minimum queue length policy behavior tests."""

import pytest

from vllm_omni.global_scheduler.policies.min_queue_length import MinQueueLengthPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_min_queue_length_prefers_smaller_total_outstanding_requests():
    policy = MinQueueLengthPolicy(tie_breaker="lexical")
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002"),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=0, inflight=3, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=2, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(RequestMeta(request_id="r1"), instances, runtime_stats)
    assert decision.instance_id == "worker-1"
    assert decision.score == pytest.approx(2.0)


def test_min_queue_length_breaks_ties_lexically():
    policy = MinQueueLengthPolicy(tie_breaker="lexical")
    instances = [
        InstanceSpec(id="worker-b", endpoint="http://127.0.0.1:9002"),
        InstanceSpec(id="worker-a", endpoint="http://127.0.0.1:9001"),
    ]
    runtime_stats = {
        "worker-a": RuntimeStats(queue_len=0, inflight=1, ewma_service_time_s=1.0),
        "worker-b": RuntimeStats(queue_len=0, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(RequestMeta(request_id="r2"), instances, runtime_stats)
    assert decision.instance_id == "worker-a"
