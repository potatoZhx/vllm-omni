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
    """Missing command should raise LifecycleUnsupportedError."""
    controller = LocalProcessController()
    instance = InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")

    with pytest.raises(LifecycleUnsupportedError, match="not configured"):
        controller.start(instance)


def test_local_process_controller_exec_error():
    """Failing command should raise LifecycleExecutionError with details."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        restart_command='sh -c "echo boom 1>&2; exit 7"',
    )

    with pytest.raises(LifecycleExecutionError, match="restart failed"):
        controller.restart(instance)


def test_local_process_controller_success_command():
    """Successful command should return without raising."""
    controller = LocalProcessController()
    instance = InstanceSpec(
        id="worker-0",
        endpoint="http://127.0.0.1:9001",
        stop_command='sh -c "exit 0"',
    )

    controller.stop(instance)
