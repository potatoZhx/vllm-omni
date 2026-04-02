"""Round-robin routing policy tests."""

import pytest

from vllm_omni.global_scheduler.policies.round_robin import RoundRobinPolicy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _instances() -> list[InstanceSpec]:
    return [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002"),
        InstanceSpec(id="worker-2", endpoint="http://127.0.0.1:9003"),
    ]


def test_round_robin_rotates_in_stable_order():
    policy = RoundRobinPolicy(tie_breaker="lexical")
    runtime_stats = {
        "worker-0": RuntimeStats(inflight=0, ewma_service_time_s=1.0),
        "worker-1": RuntimeStats(inflight=0, ewma_service_time_s=1.0),
        "worker-2": RuntimeStats(inflight=0, ewma_service_time_s=1.0),
    }

    decisions = [
        policy.select_instance(RequestMeta(request_id=f"r{idx}"), _instances(), runtime_stats)
        for idx in range(4)
    ]
    assert [decision.instance_id for decision in decisions] == ["worker-0", "worker-1", "worker-2", "worker-0"]
    assert all(decision.reason == "algorithm=round_robin" for decision in decisions)
