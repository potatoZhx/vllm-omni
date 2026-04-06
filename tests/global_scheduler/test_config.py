"""Config loading and validation tests for global scheduler."""

import textwrap

import pytest

from vllm_omni.global_scheduler.config import load_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_load_config_success(tmp_path):
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
    assert config.policy.baseline.algorithm == "min_queue_length"


def test_load_config_benchmark_section_success(tmp_path):
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
              warmup_request_config: '[{"width":512,"height":512,"num_inference_steps":20}]'
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
    assert config.benchmark.model == "Qwen/Qwen-Image"
    assert config.benchmark.warmup_request_config == '[{"width":512,"height":512,"num_inference_steps":20}]'
    assert config.benchmark.max_concurrency == 16
    assert config.benchmark.auto_stop is False


def test_load_config_duplicate_instance_id(tmp_path):
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


@pytest.mark.parametrize("algorithm", ["min_queue_length", "round_robin", "short_queue_runtime"])
def test_load_config_supported_baseline_algorithms(tmp_path, algorithm):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            policy:
              baseline:
                algorithm: {algorithm}
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.policy.baseline.algorithm == algorithm


def test_load_config_runtime_profile_path_success(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    profile_path = tmp_path / "runtime_profile.json"
    profile_path.write_text('{"profiles": []}', encoding="utf-8")
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

    config = load_config(config_path)
    assert config.policy.baseline.runtime_profile_path == str(profile_path)
    assert config.instances[0].instance_type == "wan-video-tp2"


def test_load_config_invalid_baseline_algorithm(tmp_path):
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
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline_sp1
            policy:
              baseline_sp1:
                algorithm: min_queue_length
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
                  args: ["--omni", "--diffusion-scheduler-backend", "step_level_request_scheduler"]
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
    assert config.instances[0].launch.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert config.instances[0].stop is not None


def test_load_config_empty_launch_arg_rejected(tmp_path):
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
