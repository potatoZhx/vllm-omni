"""Router construction and delegation behavior tests."""

import json
import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.policies.min_queue_length import MinQueueLengthPolicy
from vllm_omni.global_scheduler.policies.round_robin import RoundRobinPolicy
from vllm_omni.global_scheduler.policies.short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.router import build_policy
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta, RuntimeStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_router_rejects_unknown_scheduler_type(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              unexpected_key: unsupported_type
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(config_path)


def test_router_builds_min_queue_length_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                algorithm: min_queue_length
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )
    policy = build_policy(load_config(config_path))
    assert isinstance(policy._delegate, MinQueueLengthPolicy)


def test_router_builds_round_robin_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                algorithm: round_robin
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )
    policy = build_policy(load_config(config_path))
    assert isinstance(policy._delegate, RoundRobinPolicy)


def test_router_builds_short_queue_runtime_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                algorithm: short_queue_runtime
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )
    policy = build_policy(load_config(config_path))
    assert isinstance(policy._delegate, ShortQueueRuntimePolicy)


def test_router_loads_runtime_profile_from_sample_format(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "instance_type": "wan-video-tp2",
                        "task_type": "video",
                        "width": 1280,
                        "height": 720,
                        "num_frames": 16,
                        "steps": 50,
                        "latency_ms": 8210,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        textwrap.dedent(
            f"""
            policy:
              baseline:
                algorithm: short_queue_runtime
                runtime_profile_path: {profile_path}
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                instance_type: wan-video-tp2
            """
        ),
        encoding="utf-8",
    )

    policy = build_policy(load_config(config_path))
    assert isinstance(policy._delegate, ShortQueueRuntimePolicy)
    assert policy._delegate._estimator.profiling_data[("wan-video-tp2", 1280, 720, 16, 50)] == pytest.approx(8.21)


def test_router_reason_uses_router_prefix(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                algorithm: min_queue_length
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )

    policy = build_policy(load_config(config_path))
    decision = policy.select_instance(
        request=RequestMeta(request_id="req-1"),
        instances=[InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")],
        runtime_stats={"worker-0": RuntimeStats(queue_len=0, inflight=0, ewma_service_time_s=1.0)},
    )
    assert decision.reason == "router=min_queue_length;algorithm=min_queue_length"
