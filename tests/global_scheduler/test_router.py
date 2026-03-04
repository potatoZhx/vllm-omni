import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.policies.estimated_completion_time import EstimatedCompletionTimePolicy
from vllm_omni.global_scheduler.policies.first_come_first_served import FirstComeFirstServedPolicy
from vllm_omni.global_scheduler.policies.short_queue_runtime import ShortQueueRuntimePolicy
from vllm_omni.global_scheduler.router import build_policy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_router_builds_fcfs_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline
            policy:
              baseline:
                algorithm: fcfs
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    policy = build_policy(config)

    assert isinstance(policy._delegate, FirstComeFirstServedPolicy)


def test_router_rejects_unknown_scheduler_type(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: unsupported_type
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scheduler.type must be one of: baseline, ondisc"):
      load_config(config_path)


def test_router_builds_short_queue_runtime_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline
            policy:
              baseline:
                algorithm: short_queue_runtime
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    policy = build_policy(config)

    assert isinstance(policy._delegate, ShortQueueRuntimePolicy)


def test_router_builds_estimated_completion_time_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline
            policy:
              baseline:
                algorithm: estimated_completion_time
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    policy = build_policy(config)

    assert isinstance(policy._delegate, EstimatedCompletionTimePolicy)
