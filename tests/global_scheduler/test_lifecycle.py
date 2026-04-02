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
    return [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001"),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002"),
    ]


def _request(request_id: str) -> RequestMeta:
    return RequestMeta(request_id=request_id)


def test_unhealthy_or_disabled_instances_are_excluded_from_routable_set():
    manager = InstanceLifecycleManager(_instances())

    manager.mark_health("worker-1", healthy=False, error="dial failed")
    assert [item.id for item in manager.get_routable_instances()] == ["worker-0"]

    manager.set_enabled("worker-0", enabled=False)
    assert manager.get_routable_instances() == []

    manager.set_enabled("worker-0", enabled=True)
    manager.mark_health("worker-0", healthy=True)
    assert [item.id for item in manager.get_routable_instances()] == ["worker-0"]


def test_reload_keeps_removed_instance_until_inflight_converges():
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

    store.on_request_finish("worker-1", latency_s=0.5, ok=False, request_id="r1")
    manager.converge_draining(store.snapshot())
    assert "worker-1" not in manager.snapshot()
    assert "worker-1" not in store.snapshot()


def test_user_disabled_instance_is_kept_after_drain_converges():
    instances = _instances()
    store = RuntimeStateStore(instances=instances)
    manager = InstanceLifecycleManager(instances)

    manager.set_enabled("worker-1", enabled=False)
    store.on_request_start("worker-1", _request("r1"))

    manager.converge_draining(store.snapshot())
    assert manager.snapshot()["worker-1"].draining is True

    store.on_request_finish("worker-1", latency_s=0.3, ok=False, request_id="r1")
    manager.converge_draining(store.snapshot())

    final = manager.snapshot()
    assert "worker-1" in final
    assert final["worker-1"].enabled is False
    assert final["worker-1"].draining is False


def test_probe_http_ready_reports_success_for_non_empty_models(monkeypatch):
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

    monkeypatch.setattr("vllm_omni.global_scheduler.lifecycle._NO_PROXY_OPENER.open", _fake_open)

    healthy, error = _probe_http_health("http://127.0.0.1:9001", 0.5)
    assert healthy is True
    assert error is None
    assert seen == {"url": "http://127.0.0.1:9001/health", "timeout": 0.5}


def test_probe_failures_require_threshold_before_marking_unhealthy(monkeypatch):
    manager = InstanceLifecycleManager(_instances())
    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_health",
        lambda endpoint, timeout_s: (False, "timed out"),
    )

    manager.probe_all(0.5, unhealthy_after_failures=3)
    assert manager.snapshot()["worker-0"].healthy is True
    manager.probe_all(0.5, unhealthy_after_failures=3)
    assert manager.snapshot()["worker-0"].healthy is True
    manager.probe_all(0.5, unhealthy_after_failures=3)
    assert manager.snapshot()["worker-0"].healthy is False
