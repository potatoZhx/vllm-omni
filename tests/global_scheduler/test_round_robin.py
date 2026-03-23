"""Round-robin routing policy tests."""

import pytest

from vllm_omni.global_scheduler.policies.round_robin import RoundRobinPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _instances() -> list[InstanceSpec]:
    return [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", launch_args=["--diffusion-engine-max-concurrency", "1"]),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", launch_args=["--diffusion-engine-max-concurrency", "1"]),
        InstanceSpec(id="worker-2", endpoint="http://127.0.0.1:9003", launch_args=["--diffusion-engine-max-concurrency", "1"]),
    ]


def test_round_robin_rotates_across_available_instances():
    """Round robin should rotate in stable order when all instances are available."""
    policy = RoundRobinPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r1")
    instances = _instances()
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
        "worker-2": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
    }

    decisions = [
        policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
        for _ in range(4)
    ]

    assert [decision.instance_id for decision in decisions] == ["worker-0", "worker-1", "worker-2", "worker-0"]
    assert all("algorithm=round_robin" in decision.reason for decision in decisions)
    assert all("available=true" in decision.reason for decision in decisions)


def test_round_robin_skips_busy_instance_and_keeps_rotation():
    """Round robin should skip busy entries and continue from chosen instance."""
    policy = RoundRobinPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r2")
    instances = _instances()
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
        "worker-2": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0),
    }

    first = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
    second = policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)

    assert first.instance_id == "worker-1"
    assert second.instance_id == "worker-2"
    assert "available=true" in first.reason


def test_round_robin_rotates_when_all_busy():
    """Round robin should still rotate across all instances when none are available."""
    policy = RoundRobinPolicy(tie_breaker="lexical")
    request = RequestMeta(request_id="r3")
    instances = _instances()
    runtime_stats = {
        "worker-0": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
        "worker-2": RuntimeStats(queue_len=1, inflight=1, ewma_service_time_s=1.0),
    }

    decisions = [
        policy.select_instance(request=request, instances=instances, runtime_stats=runtime_stats)
        for _ in range(3)
    ]

    assert [decision.instance_id for decision in decisions] == ["worker-0", "worker-1", "worker-2"]
    assert all("available=false" in decision.reason for decision in decisions)
