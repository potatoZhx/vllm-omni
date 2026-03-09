"""Config loading and validation tests for global scheduler."""

import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_config_success(tmp_path):
    """Config with valid server and instances should load successfully."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              host: 0.0.0.0
              port: 8089
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
              - id: worker-1
                endpoint: http://127.0.0.1:9002
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.server.port == 8089
    assert len(config.instances) == 2


def test_load_config_duplicate_instance_id(tmp_path):
    """Duplicate instance ids should be rejected by config validation."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
              - id: worker-0
                endpoint: http://127.0.0.1:9002
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_config(config_path)


def test_load_config_invalid_endpoint(tmp_path):
    """Only http://host:port endpoints should be accepted."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: https://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="http://host:port"):
        load_config(config_path)


def test_load_config_baseline_algorithm_success(tmp_path):
    """Supported baseline algorithm values should pass validation."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
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

    assert config.policy.baseline.algorithm == "fcfs"


def test_load_config_short_queue_runtime_algorithm(tmp_path):
    """short_queue_runtime should be accepted as a baseline algorithm."""
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
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.policy.baseline.algorithm == "short_queue_runtime"


def test_load_config_invalid_baseline_algorithm(tmp_path):
    """Unknown baseline algorithm names should fail validation."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                algorithm: unknown_algo
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy.baseline.algorithm"):
        load_config(config_path)


def test_load_config_legacy_sp1_keys_are_rejected(tmp_path):
    """Legacy baseline_sp1 config keys should be rejected."""
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

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(config_path)


def test_load_config_legacy_mode_key_is_rejected(tmp_path):
    """Legacy mode key should be rejected in baseline policy config."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            policy:
              baseline:
                mode: short_queue_runtime
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(config_path)


def test_load_config_instance_lifecycle_commands_success(tmp_path):
    """Lifecycle command fields should be accepted when non-empty."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
                start_command: "echo start"
                stop_command: "echo stop"
                restart_command: "echo restart"
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.instances[0].start_command == "echo start"
    assert config.instances[0].stop_command == "echo stop"
    assert config.instances[0].restart_command == "echo restart"


def test_load_config_empty_lifecycle_command_rejected(tmp_path):
    """Lifecycle command field should reject blank strings."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 2
                start_command: "   "
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lifecycle commands cannot be empty"):
        load_config(config_path)
