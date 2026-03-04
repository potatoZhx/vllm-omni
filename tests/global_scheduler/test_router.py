import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.policies.baseline_estimated_completion_time import (
  BaselineEstimatedCompletionTimePolicy,
)
from vllm_omni.global_scheduler.policies.baseline_fcfs import BaselineFCFSPolicy
from vllm_omni.global_scheduler.policies.baseline_short_queue_runtime import BaselineShortQueueRuntimePolicy
from vllm_omni.global_scheduler.router import build_policy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_router_builds_baseline_fcfs_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            policy:
              baseline_sp1:
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

    assert isinstance(policy._delegate, BaselineFCFSPolicy)


def test_router_rejects_unknown_scheduler_type(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: ondisc_sp1
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

    with pytest.raises(ValueError, match="Unsupported scheduler.type"):
        build_policy(config)


def test_router_builds_baseline_short_queue_runtime_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            policy:
              baseline_sp1:
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

    assert isinstance(policy._delegate, BaselineShortQueueRuntimePolicy)


def test_router_builds_baseline_estimated_completion_time_policy(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            policy:
              baseline_sp1:
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

    assert isinstance(policy._delegate, BaselineEstimatedCompletionTimePolicy)
