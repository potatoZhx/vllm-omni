"""Scheduler server endpoint and lifecycle API tests."""

import textwrap

import pytest
from fastapi.testclient import TestClient

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.server import UpstreamHTTPError, create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_health_endpoint_returns_scheduler_and_ok(tmp_path):
    """Health endpoint should report ok when config is present and valid."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["instance_count"] == 1
    assert payload["checks"]["config_loaded"] is True
    assert payload["checks"]["has_instances"] is True
    assert "version" in payload


def test_health_endpoint_returns_503_when_config_missing(tmp_path):
    """Health endpoint should degrade when app config is missing."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    app.state.global_scheduler_config = None
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["config_loaded"] is False
    assert payload["checks"]["has_instances"] is False


def test_load_config_missing_file_raises_clear_error(tmp_path):
    """Missing config file should produce a clear validation error."""
    missing = tmp_path / "not_found.yaml"

    with pytest.raises(ValueError, match="Config file not found"):
        load_config(missing)


def test_instance_lifecycle_control_endpoints(tmp_path):
    """Enable/disable endpoints should update routable state as expected."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
              instance_health_check_timeout_s: 0.1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    list_before = client.get("/instances")
    assert list_before.status_code == 200
    assert list_before.json()["instances"][0]["routable"] is True

    disable_response = client.post("/instances/worker-0/disable")
    assert disable_response.status_code == 200
    assert disable_response.json()["enabled"] is False

    list_after_disable = client.get("/instances")
    assert list_after_disable.json()["instances"][0]["routable"] is False

    enable_response = client.post("/instances/worker-0/enable")
    assert enable_response.status_code == 200
    assert enable_response.json()["enabled"] is True

    list_after_enable = client.get("/instances")
    assert list_after_enable.json()["instances"][0]["routable"] is True


def test_reload_endpoint_replaces_instance_set(tmp_path):
    """Reload endpoint should reconcile and replace configured instances."""
    initial_path = tmp_path / "scheduler.yaml"
    reloaded_path = tmp_path / "scheduler_reloaded.yaml"

    initial_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    reloaded_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
            instances:
              - id: worker-1
                endpoint: http://127.0.0.1:9002
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(initial_path)
    app = create_app(config, config_loader=lambda: load_config(reloaded_path))
    client = TestClient(app)

    before_reload = client.get("/instances").json()["instances"]
    assert [item["id"] for item in before_reload] == ["worker-0"]

    reload_response = client.post("/instances/reload")
    assert reload_response.status_code == 200
    assert reload_response.json()["status"] == "ok"
    assert reload_response.json()["instance_count"] == 1

    after_reload = client.get("/instances").json()["instances"]
    assert [item["id"] for item in after_reload] == ["worker-1"]


def test_reload_endpoint_returns_501_without_loader(tmp_path):
    """Reload endpoint should return 501 when reload loader is not configured."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    response = client.post("/instances/reload")
    assert response.status_code == 501


def test_probe_endpoint_runs_probe_in_to_thread(tmp_path, monkeypatch):
    """Probe endpoint should delegate probe work to asyncio.to_thread."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
              instance_health_check_timeout_s: 0.1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    app = create_app(config)
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def _fake_to_thread(func, *args, **kwargs):
        """Record to_thread invocation without executing background work."""
        calls.append((func, args, kwargs))
        return None

    monkeypatch.setattr("vllm_omni.global_scheduler.server.asyncio.to_thread", _fake_to_thread)

    client = TestClient(app)
    response = client.post("/instances/probe")

    assert response.status_code == 200
    assert len(calls) == 1


def test_metrics_endpoint_reports_global_and_instance_metrics(tmp_path, monkeypatch):
    """Metrics endpoint should expose request counters and instance runtime stats."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              request_timeout_s: 2
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.server._proxy_chat_completion",
        lambda *_args, **_kwargs: b'{"id":"ok"}',
    )
    client = TestClient(app)

    success = client.post("/v1/chat/completions", json={"model": "demo", "messages": []})
    assert success.status_code == 200
    app.state.instance_lifecycle_manager.set_enabled("worker-0", enabled=False)
    failure = client.post("/v1/chat/completions", json={"model": "demo", "messages": []})
    assert failure.status_code == 503

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["global"]["request_total"] == 2
    assert payload["global"]["request_success"] == 1
    assert payload["global"]["request_failure"] == 1
    assert payload["global"]["uptime_s"] >= 0.0
    assert payload["instances"][0]["id"] == "worker-0"
    assert payload["instances"][0]["queue_len"] == 0
    assert payload["instances"][0]["inflight"] == 0


def test_lifecycle_audit_logs_include_required_fields(tmp_path, monkeypatch):
    """Lifecycle operations should emit audit logs with required fields."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
              instance_health_check_timeout_s: 0.1
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)
    client = TestClient(app)

    calls: list[dict[str, object]] = []

    def _capture_log_event(_level, event, **fields):
        calls.append({"event": event, **fields})

    monkeypatch.setattr("vllm_omni.global_scheduler.server._log_event", _capture_log_event)
    response = client.post("/instances/worker-0/disable", headers={"x-operator": "ops-user"})
    assert response.status_code == 200

    logs = [item for item in calls if item["event"] == "LIFECYCLE_AUDIT"]
    assert logs
    entry = logs[-1]
    assert entry["op"] == "disable"
    assert entry["instance_id"] == "worker-0"
    assert entry["operator"] == "ops-user"
    assert entry["result"] == "ok"
    assert "error_code" in entry


def test_reload_conflict_returns_409_with_gs_lifecycle_conflict(tmp_path):
    """Reload should reject concurrent execution with GS_LIFECYCLE_CONFLICT."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config, config_loader=lambda: config)

    class _LockedReload:
        def locked(self):
            return True

    app.state.reload_lock = _LockedReload()
    client = TestClient(app)
    response = client.post("/instances/reload")

    assert response.status_code == 409
    detail = response.json()["detail"]["error"]
    assert detail["code"] == "GS_LIFECYCLE_CONFLICT"


def test_chat_completions_success_sets_route_headers_and_state(tmp_path, monkeypatch):
    """Proxy success path should add route headers and release counters."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              request_timeout_s: 2
              instance_health_check_interval_s: 100
            policy:
              baseline:
                algorithm: fcfs
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)

    def _fake_proxy(endpoint, body, headers, timeout_s):
        """Return a deterministic JSON response without real upstream call."""
        assert endpoint == "http://127.0.0.1:9001"
        assert timeout_s == 2
        assert b'"model": "demo"' in body
        assert headers["content-type"] == "application/json"
        return b'{"id": "resp-1"}'

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_chat_completion", _fake_proxy)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"content-type": "application/json", "x-request-id": "req-1"},
        json={"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "resp-1"
    assert response.headers["X-Routed-Instance"] == "worker-0"
    assert "router=fcfs" in response.headers["X-Route-Reason"]
    assert float(response.headers["X-Route-Score"]) >= 0.0

    snapshot = app.state.runtime_state_store.snapshot()
    assert snapshot["worker-0"].queue_len == 0
    assert snapshot["worker-0"].inflight == 0


def test_chat_completions_emits_route_structured_logs(tmp_path, monkeypatch):
    """Route logs should form begin -> decision -> done structured events."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              request_timeout_s: 2
              instance_health_check_interval_s: 100
            policy:
              baseline:
                algorithm: fcfs
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)
    monkeypatch.setattr(
        "vllm_omni.global_scheduler.server._proxy_chat_completion",
        lambda *_args, **_kwargs: b'{"id":"resp-2"}',
    )
    client = TestClient(app)
    calls: list[dict[str, object]] = []

    def _capture_log_event(_level, event, **fields):
        calls.append({"event": event, **fields})

    monkeypatch.setattr("vllm_omni.global_scheduler.server._log_event", _capture_log_event)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-log"},
        json={"model": "demo", "messages": []},
    )
    assert response.status_code == 200

    logs = [item for item in calls if item.get("request_id") == "req-log"]
    events = [item["event"] for item in logs]
    assert "ROUTE_BEGIN" in events
    assert "ROUTE_DECISION" in events
    assert "ROUTE_DONE" in events


def test_chat_completions_returns_503_when_no_routable_instance(tmp_path):
    """Proxy should return GS_NO_ROUTABLE_INSTANCE when no instance is routable."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)
    app.state.instance_lifecycle_manager.set_enabled("worker-0", enabled=False)
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json={"model": "demo", "messages": []})

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "GS_NO_ROUTABLE_INSTANCE"
    assert payload["error"]["request_id"]


@pytest.mark.parametrize(
    "raised,status_code,error_code",
    [
        (UpstreamHTTPError(status_code=503, body=b"{}"), 503, "GS_UPSTREAM_HTTP_ERROR"),
        (TimeoutError("timeout"), 502, "GS_UPSTREAM_TIMEOUT"),
        (OSError("network down"), 502, "GS_UPSTREAM_NETWORK_ERROR"),
    ],
)
def test_chat_completions_error_semantics_and_state_cleanup(tmp_path, monkeypatch, raised, status_code, error_code):
    """Proxy should classify upstream errors and always cleanup runtime state."""
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              request_timeout_s: 3
              instance_health_check_interval_s: 100
            instances:
              - id: worker-0
                endpoint: http://127.0.0.1:9001
                sp_size: 1
                max_concurrency: 1
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    app = create_app(config)

    def _fake_proxy(*_args, **_kwargs):
        """Raise synthetic upstream errors for classification tests."""
        raise raised

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_chat_completion", _fake_proxy)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-error"},
        json={"model": "demo", "messages": []},
    )

    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == error_code
    assert payload["error"]["request_id"] == "req-error"
    assert response.headers["X-Routed-Instance"] == "worker-0"

    snapshot = app.state.runtime_state_store.snapshot()
    assert snapshot["worker-0"].queue_len == 0
    assert snapshot["worker-0"].inflight == 0
