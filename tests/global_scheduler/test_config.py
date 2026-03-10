"""Config loading and validation tests for global scheduler."""

"""Config loading and validation tests for global scheduler."""

import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_config_success(tmp_path):
    """Config with valid server and instances should load successfully."""
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
              - id: worker-1
                endpoint: http://127.0.0.1:9002
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.server.port == 8089
    assert len(config.instances) == 2


def test_load_config_benchmark_section_success(tmp_path):
    """Benchmark section should be accepted and parsed with typed fields."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            benchmark:
              worker_ids: [worker-0, worker-1]
              worker_ready_timeout_s: 900
              model: Qwen/Qwen-Image
              task: t2i
              dataset: trace
              dataset_path: /tmp/prompts.txt
              max_concurrency: 16
              warmup_requests: 2
              warmup_num_inference_steps: 3
              output_file: /tmp/metrics.json
              auto_stop: false
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
              - id: worker-1
                endpoint: http://127.0.0.1:9002
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.benchmark.worker_ids == ["worker-0", "worker-1"]
    assert config.benchmark.worker_ready_timeout_s == 900
    assert config.benchmark.model == "Qwen/Qwen-Image"
    assert config.benchmark.max_concurrency == 16
    assert config.benchmark.auto_stop is False


def test_load_config_duplicate_instance_id(tmp_path):
    """Duplicate instance ids should be rejected by config validation."""
    """Duplicate instance ids should be rejected by config validation."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
              - id: worker-0
                endpoint: http://127.0.0.1:9002
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="globally unique"):
        load_config(config_path)


def test_load_config_invalid_endpoint(tmp_path):
    """Only http://host:port endpoints should be accepted."""
    """Only http://host:port endpoints should be accepted."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: https://127.0.0.1:9001
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
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.policy.baseline.algorithm == "short_queue_runtime"


def test_load_config_round_robin_algorithm(tmp_path):
    """round_robin should be accepted as a baseline algorithm."""
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

    config = load_config(config_path)

    assert config.policy.baseline.algorithm == "round_robin"


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
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(config_path)


def test_load_config_instance_lifecycle_structured_fields_success(tmp_path):
    """Structured launch/stop fields should be accepted."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                launch:
                  model: Qwen/Qwen-Image
                  executable: vllm
                  args: ["--omni", "--ulysses-degree", "2"]
                  env:
                    CUDA_VISIBLE_DEVICES: "0,1"
                stop:
                  executable: pkill
                  args: ["-f", "vllm serve Qwen/Qwen-Image --port 9001"]
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.instances[0].launch is not None
    assert config.instances[0].launch.model == "Qwen/Qwen-Image"
    assert config.instances[0].launch.args == ["--omni", "--ulysses-degree", "2"]
    assert config.instances[0].launch.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert config.instances[0].stop is not None
    assert config.instances[0].stop.args[0] == "-f"


def test_load_config_empty_launch_arg_rejected(tmp_path):
    """Structured launch args should reject blank items."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                launch:
                  model: Qwen/Qwen-Image
                  args: ["--omni", "   "]
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="launch.args cannot include empty items"):
        load_config(config_path)
