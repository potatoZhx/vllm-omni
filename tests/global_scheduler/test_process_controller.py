"""Process controller tests for lifecycle command execution."""

import pytest

from vllm_omni.global_scheduler.process_controller import (
    LifecycleExecutionError,
    LifecycleUnsupportedError,
    LocalProcessController,
)
from vllm_omni.global_scheduler.types import InstanceSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_process_controller_requires_command():
    """Missing launch config should raise LifecycleUnsupportedError."""
    controller = LocalProcessController()
    instance = InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")

    with pytest.raises(LifecycleUnsupportedError, match="launch config not provided"):
        controller.start(instance)


def test_local_process_controller_exec_error():
    """Failing command should raise LifecycleExecutionError with details."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_executable="sh",
        stop_args=["-c", "echo boom 1>&2; exit 7"],
        launch_executable="sh",
        launch_model="ignored-model",
        launch_args=["-c", "exit 0"],
    )

    with pytest.raises(LifecycleExecutionError, match="stop failed"):
        controller.restart(instance)


def test_local_process_controller_success_command():
    """Successful command should return without raising."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_executable="sh",
        stop_args=["-c", "exit 0"],
    )

    controller.stop(instance)


def test_local_process_controller_expands_stop_placeholders(monkeypatch):
    """stop() should expand endpoint-derived placeholders in stop args."""
    captured: dict[str, object] = {}

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr("vllm_omni.global_scheduler.process_controller.subprocess.run", _fake_run)

    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_executable="pkill",
        stop_args=["-f", "vllm serve --host {endpoint_host} --port {endpoint_port} --tag {instance_id}"],
    )

    controller.stop(instance)

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv == [
        "pkill",
        "-f",
        "vllm serve --host 127.0.0.1 --port 9001 --tag worker-0",
    ]


def test_local_process_controller_start_with_structured_launch():
    """Start should execute structured launch command with endpoint port."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        launch_executable="true",
        launch_model="ignored-model",
        launch_args=["--ulysses-degree", "2"],
    )

    controller.start(instance)


def test_local_process_controller_strips_scheduler_only_max_concurrency(monkeypatch):
    """start() should not forward scheduler-only --max-concurrency to vllm serve."""
    captured: dict[str, object] = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        class _Proc:
            pid = 1

        return _Proc()

    monkeypatch.setattr("vllm_omni.global_scheduler.process_controller.subprocess.Popen", _fake_popen)

    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        launch_executable="vllm",
        launch_model="Qwen/Qwen-Image",
        launch_args=["--omni", "--max-concurrency", "100", "--ulysses-degree", "1"],
    )

    controller.start(instance)
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "--max-concurrency" not in argv
    assert "--ulysses-degree" in argv


def test_local_process_controller_start_writes_to_instance_log(monkeypatch, tmp_path):
    """Background start should route stdout/stderr to per-instance log path."""
    monkeypatch.setenv("GLOBAL_SCHEDULER_LOG_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class _Proc:
            pid = 12345

        return _Proc()

    monkeypatch.setattr("vllm_omni.global_scheduler.process_controller.subprocess.Popen", _fake_popen)

    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        launch_executable="vllm",
        launch_model="Qwen/Qwen-Image",
        launch_args=["--omni"],
    )
    controller.start(instance)

    kwargs = captured["kwargs"]
    assert kwargs["stderr"] is not None
    assert kwargs["stdout"].name.endswith("worker-0.log")
