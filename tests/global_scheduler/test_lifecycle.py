import pytest

from vllm_omni.global_scheduler.lifecycle import (
    InstanceLifecycleManager,
    _probe_http_health,
    _probe_http_ready,
)
from vllm_omni.global_scheduler.state import RuntimeStateStore
from vllm_omni.global_scheduler.types import InstanceSpec, RequestMeta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _instances() -> list[InstanceSpec]:
    """Build a deterministic two-instance fixture for lifecycle tests.

    Returns:
        Static instance list used by lifecycle state assertions.
    """
    """Build a deterministic two-instance fixture for lifecycle tests.

    Returns:
        Static instance list used by lifecycle state assertions.
    """
    return [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002"),
    ]


def _request(request_id: str) -> RequestMeta:
    return RequestMeta(request_id=request_id)


def test_unhealthy_or_disabled_instances_are_excluded_from_routable_set():
    """Routable set should exclude disabled or unhealthy instances."""
    """Routable set should exclude disabled or unhealthy instances."""
    manager = InstanceLifecycleManager(_instances())

    manager.mark_health("worker-1", healthy=False, error="dial failed")
    routable_ids = [item.id for item in manager.get_routable_instances()]
    assert routable_ids == ["worker-0"]

    manager.set_enabled("worker-0", enabled=False)
    routable_ids_after_disable = [item.id for item in manager.get_routable_instances()]
    assert routable_ids_after_disable == []

    manager.set_enabled("worker-0", enabled=True)
    manager.mark_health("worker-0", healthy=True)
    restored_ids = [item.id for item in manager.get_routable_instances()]
    assert restored_ids == ["worker-0"]


def test_reload_keeps_removed_instance_until_inflight_converges():
    """Removed instances should remain draining until runtime work converges."""
    """Removed instances should remain draining until runtime work converges."""
    instances = _instances()
    store = RuntimeStateStore(instances=instances)
    manager = InstanceLifecycleManager(instances)

    store.on_request_start("worker-1", _request("r1"))
    store.sync_instances([InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")])

    manager.sync_instances(
        [InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")],
        runtime_snapshot=store.snapshot(),
    )

    draining_snapshot = manager.snapshot()
    assert draining_snapshot["worker-1"].draining is True
    assert draining_snapshot["worker-1"].enabled is False

    store.on_request_finish("worker-1", latency_s=0.5, ok=False)
    manager.converge_draining(store.snapshot())

    final_manager_snapshot = manager.snapshot()
    assert "worker-1" not in final_manager_snapshot
    final_runtime_snapshot = store.snapshot()
    assert "worker-1" not in final_runtime_snapshot


def test_reload_does_not_clear_operator_draining_state_for_desired_instance():
    """Reload should preserve operator-intended draining for desired instances."""
    instances = _instances()
    manager = InstanceLifecycleManager(instances)

    manager.set_enabled("worker-0", enabled=False)
    before_reload = manager.snapshot()
    assert before_reload["worker-0"].draining is True

    manager.sync_instances(instances, runtime_snapshot={})

    after_reload = manager.snapshot()
    assert after_reload["worker-0"].enabled is False
    assert after_reload["worker-0"].draining is True


def test_user_disabled_instance_is_kept_after_drain_converges():
    """User-disabled instances should remain tracked after drain convergence."""
    instances = _instances()
    store = RuntimeStateStore(instances=instances)
    manager = InstanceLifecycleManager(instances)

    manager.set_enabled("worker-1", enabled=False)
    store.on_request_start("worker-1", _request("r1"))

    manager.converge_draining(store.snapshot())
    mid = manager.snapshot()
    assert mid["worker-1"].draining is True

    store.on_request_finish("worker-1", latency_s=0.3, ok=False)
    manager.converge_draining(store.snapshot())

    final = manager.snapshot()
    assert "worker-1" in final
    assert final["worker-1"].enabled is False
    assert final["worker-1"].draining is False


def test_non_running_process_state_is_excluded_from_routable_set():
    """Instances in non-running process states must not be routed."""
    manager = InstanceLifecycleManager(_instances())

    manager.set_process_state("worker-0", process_state="restarting", operation="restart")
    routable_ids = [item.id for item in manager.get_routable_instances()]
    assert routable_ids == ["worker-1"]

    snapshot = manager.snapshot()
    assert snapshot["worker-0"].process_state == "restarting"
    assert snapshot["worker-0"].last_operation == "restart"
    assert snapshot["worker-0"].last_operation_ts_s is not None


def test_probe_http_ready_reports_success_for_non_empty_models(monkeypatch):
    """HTTP readiness probe should succeed only when models are returned."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data":[{"id":"demo"}]}'

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._NO_PROXY_OPENER.open",
        lambda request, timeout: _Response(),
    )

    healthy, error = _probe_http_ready("http://127.0.0.1:9001", 0.5)

    assert healthy is True
    assert error is None


def test_probe_http_ready_reports_empty_models_as_unhealthy(monkeypatch):
    """HTTP readiness probe should reject empty model lists."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data":[]}'

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._NO_PROXY_OPENER.open",
        lambda request, timeout: _Response(),
    )

    healthy, error = _probe_http_ready("http://127.0.0.1:9001", 0.5)

    assert healthy is False
    assert error == "ready_probe_empty_models"


def test_probe_http_health_reports_success_on_http_200(monkeypatch):
    """Runtime health probe should accept a lightweight HTTP 200 response."""

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    seen = {}

    def _fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._NO_PROXY_OPENER.open",
        _fake_open,
    )

    healthy, error = _probe_http_health("http://127.0.0.1:9001", 0.5)

    assert healthy is True
    assert error is None
    assert seen == {"url": "http://127.0.0.1:9001/health", "timeout": 0.5}


def test_probe_all_uses_runtime_health_for_steady_state_instance(monkeypatch):
    """Steady-state instances should use the lightweight runtime probe."""
    manager = InstanceLifecycleManager(_instances())
    calls: list[tuple[str, str, float]] = []

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_ready",
        lambda endpoint, timeout_s: calls.append(("ready", endpoint, timeout_s)) or (True, None),
    )
    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_health",
        lambda endpoint, timeout_s: calls.append(("health", endpoint, timeout_s)) or (True, None),
    )

    manager.probe_all(0.75)

    assert calls == [
        ("health", "http://127.0.0.1:9001", 0.75),
        ("health", "http://127.0.0.1:9002", 0.75),
    ]


def test_probe_all_keeps_newly_started_instance_on_readiness_probe(monkeypatch):
    """Instances awaiting startup readiness should keep probing `/v1/models`."""
    manager = InstanceLifecycleManager(_instances())
    manager.mark_health("worker-0", healthy=False, error="awaiting_http_ready_after_start")
    manager.mark_health("worker-1", healthy=False, error="awaiting_probe_after_restart")
    calls: list[tuple[str, str, float]] = []

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_ready",
        lambda endpoint, timeout_s: calls.append(("ready", endpoint, timeout_s)) or (True, None),
    )
    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_health",
        lambda endpoint, timeout_s: calls.append(("health", endpoint, timeout_s)) or (True, None),
    )

    manager.probe_all(1.25)

    assert calls == [
        ("ready", "http://127.0.0.1:9001", 1.25),
        ("ready", "http://127.0.0.1:9002", 1.25),
    ]


def test_probe_failures_require_threshold_before_marking_unhealthy(monkeypatch):
    """Transient probe failures should not eject a healthy instance immediately."""
    manager = InstanceLifecycleManager(_instances())

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_health",
        lambda endpoint, timeout_s: (False, "timed out"),
    )

    manager.probe_all(0.5, unhealthy_after_failures=3)
    after_first = manager.snapshot()["worker-0"]
    assert after_first.healthy is True
    assert after_first.last_error == "timed out"

    manager.probe_all(0.5, unhealthy_after_failures=3)
    after_second = manager.snapshot()["worker-0"]
    assert after_second.healthy is True

    manager.probe_all(0.5, unhealthy_after_failures=3)
    after_third = manager.snapshot()["worker-0"]
    assert after_third.healthy is False
    assert after_third.consecutive_probe_failures == 3


def test_successful_probe_resets_failure_streak(monkeypatch):
    """A successful probe should reset the consecutive failure counter."""
    manager = InstanceLifecycleManager([InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001")])
    outcomes = iter([(False, "timed out"), (True, None), (False, "timed out")])

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_health",
        lambda endpoint, timeout_s: next(outcomes),
    )

    manager.probe_all(0.5, unhealthy_after_failures=2)
    after_first = manager.snapshot()["worker-0"]
    assert after_first.healthy is True
    assert after_first.consecutive_probe_failures == 1

    manager.probe_all(0.5, unhealthy_after_failures=2)
    after_success = manager.snapshot()["worker-0"]
    assert after_success.healthy is True
    assert after_success.consecutive_probe_failures == 0

    manager.probe_all(0.5, unhealthy_after_failures=2)
    after_reset_fail = manager.snapshot()["worker-0"]
    assert after_reset_fail.healthy is True
    assert after_reset_fail.consecutive_probe_failures == 1
