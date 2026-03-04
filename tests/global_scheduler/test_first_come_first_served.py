import pytest

from vllm_omni.global_scheduler.policies.first_come_first_served import FirstComeFirstServedPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fcfs_selects_first_available_instance():
    policy = FirstComeFirstServedPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r1")
    instances = [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-0"
    assert decision.endpoint == "http://127.0.0.1:9001"
    assert "algorithm=fcfs" in decision.reason


def test_fcfs_falls_back_to_lexical_when_all_busy():
    policy = FirstComeFirstServedPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r2")
    instances = [
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=1),
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=1),
    ]
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=2, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=3, inflight=1, ewma_service_time_s=1.0),
    }

    decision = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert decision.instance_id == "worker-0"
    assert "available=false" in decision.reason
