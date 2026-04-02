from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from .types import InstanceSpec


class LifecycleUnsupportedError(ValueError):
    """Raised when lifecycle operation is not configured for an instance."""


class LifecycleExecutionError(RuntimeError):
    """Raised when lifecycle operation command execution fails."""


def get_instance_log_path(instance_id: str) -> str:
    """Return absolute log path for one managed instance."""

    log_root = Path(os.environ.get("GLOBAL_SCHEDULER_LOG_DIR", "./logs/global_scheduler")).resolve()
    return str(log_root / f"{instance_id}.log")


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
    """Execute structured local lifecycle commands."""

    def stop(self, instance: InstanceSpec) -> None:
        if instance.stop_executable is None:
            raise LifecycleUnsupportedError(f"stop config not provided for instance {instance.id}")
        if not instance.stop_args:
            raise LifecycleUnsupportedError(f"stop.args is empty for instance {instance.id}")
        stop_args = self._expand_instance_placeholders(instance.stop_args, instance)
        self._run(operation="stop", instance=instance, argv=[instance.stop_executable, *stop_args], env=None)

    def start(self, instance: InstanceSpec) -> None:
        argv, env = self._build_start_argv_and_env(instance)
        self._start_background(operation="start", instance=instance, argv=argv, env=env)

    def restart(self, instance: InstanceSpec) -> None:
        self.stop(instance)
        self.start(instance)

    @staticmethod
    def _build_start_argv_and_env(instance: InstanceSpec) -> tuple[list[str], dict[str, str] | None]:
        if instance.launch_executable is None or instance.launch_model is None:
            raise LifecycleUnsupportedError(f"launch config not provided for instance {instance.id}")
        parsed = urlparse(instance.endpoint)
        if parsed.port is None:
            raise LifecycleUnsupportedError(f"endpoint has no port for instance {instance.id}")
        argv = [
            instance.launch_executable,
            "serve",
            instance.launch_model,
            "--port",
            str(parsed.port),
            *list(instance.launch_args),
        ]
        env = os.environ.copy()
        env.update(instance.launch_env)
        if instance.numa_node is not None:
            env.setdefault("VLLM_OMNI_NUMA_NODE", str(instance.numa_node))
        argv = LocalProcessController._maybe_prefix_numactl(argv, env)
        return argv, env

    @staticmethod
    def _maybe_prefix_numactl(argv: list[str], env: dict[str, str]) -> list[str]:
        if shutil.which("numactl") is None:
            return argv

        explicit_node = env.get("VLLM_OMNI_NUMA_NODE", "").strip()
        if explicit_node:
            return ["numactl", f"--cpunodebind={explicit_node}", f"--membind={explicit_node}", *argv]
        return argv

    @staticmethod
    def _expand_instance_placeholders(args: list[str], instance: InstanceSpec) -> list[str]:
        parsed = urlparse(instance.endpoint)
        endpoint_port = parsed.port
        if endpoint_port is None:
            raise LifecycleUnsupportedError(f"endpoint has no port for instance {instance.id}")
        endpoint_host = parsed.hostname
        if endpoint_host is None:
            raise LifecycleUnsupportedError(f"endpoint has no host for instance {instance.id}")

        replacements = {
            "{instance_id}": instance.id,
            "{endpoint}": instance.endpoint,
            "{endpoint_host}": endpoint_host,
            "{endpoint_port}": str(endpoint_port),
        }
        expanded: list[str] = []
        for arg in args:
            for placeholder, value in replacements.items():
                arg = arg.replace(placeholder, value)
            expanded.append(arg)
        return expanded

    @staticmethod
    def _run(operation: str, instance: InstanceSpec, argv: list[str], env: dict[str, str] | None) -> None:
        try:
            subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or f"exit code {exc.returncode}"
            raise LifecycleExecutionError(f"{operation} failed for {instance.id}: {detail}") from exc

    @staticmethod
    def _start_background(operation: str, instance: InstanceSpec, argv: list[str], env: dict[str, str] | None) -> None:
        log_path = Path(get_instance_log_path(instance.id))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("wb") as log_file:
                subprocess.Popen(  # noqa: S603
                    argv,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as exc:
            raise LifecycleExecutionError(f"{operation} failed for {instance.id}: {exc}") from exc
