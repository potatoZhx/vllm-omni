from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod

from .types import InstanceSpec


class LifecycleUnsupportedError(ValueError):
    """Raised when lifecycle operation is not configured for an instance."""


class LifecycleExecutionError(RuntimeError):
    """Raised when lifecycle operation command execution fails."""


class ProcessController(ABC):
    """Abstract process controller for instance lifecycle operations."""

    @abstractmethod
    def stop(self, instance: InstanceSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def start(self, instance: InstanceSpec) -> None:
        raise NotImplementedError

    @abstractmethod
    def restart(self, instance: InstanceSpec) -> None:
        raise NotImplementedError


class LocalProcessController(ProcessController):
    """Execute lifecycle command templates on local shell."""

    def stop(self, instance: InstanceSpec) -> None:
        self._run(operation="stop", instance=instance, command=instance.stop_command)

    def start(self, instance: InstanceSpec) -> None:
        self._run(operation="start", instance=instance, command=instance.start_command)

    def restart(self, instance: InstanceSpec) -> None:
        self._run(operation="restart", instance=instance, command=instance.restart_command)

    @staticmethod
    def _run(operation: str, instance: InstanceSpec, command: str | None) -> None:
        if command is None:
            raise LifecycleUnsupportedError(f"{operation} command not configured for instance {instance.id}")
        try:
            subprocess.run(  # noqa: S602
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            raise LifecycleExecutionError(f"{operation} failed for {instance.id}: {detail}") from exc
