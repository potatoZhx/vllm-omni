"""Scheduler server endpoint and lifecycle API tests."""

from collections.abc import AsyncIterator
import textwrap

import pytest
from fastapi.testclient import TestClient

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.process_controller import ProcessController
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


def test_chat_completions_unexpected_exception_still_cleans_state(tmp_path, monkeypatch):
    """Unexpected proxy exceptions should still release runtime counters."""
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
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_chat_completion", _fake_proxy)

    # Keep server exception inside HTTP 500 response to assert post-state.
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-unexpected"},
        json={"model": "demo", "messages": []},
    )

    assert response.status_code == 500
    snapshot = app.state.runtime_state_store.snapshot()
    assert snapshot["worker-0"].queue_len == 0
    assert snapshot["worker-0"].inflight == 0


def test_chat_completions_streaming_passthrough_and_state_cleanup(tmp_path, monkeypatch):
    """Stream path should passthrough upstream metadata and cleanup counters."""
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

    async def _fake_stream() -> AsyncIterator[bytes]:
        yield b"data: part-1\n\n"
        yield b"data: [DONE]\n\n"

    async def _fake_open_streaming_upstream(*_args, **_kwargs):
        return (
            200,
            {"x-upstream-header": "ok", "content-type": "text/event-stream"},
            "text/event-stream",
            _fake_stream(),
        )

    monkeypatch.setattr(
        "vllm_omni.global_scheduler.server._open_streaming_upstream",
        _fake_open_streaming_upstream,
    )

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-stream"},
        json={"model": "demo", "messages": [], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-upstream-header"] == "ok"
    assert response.headers["X-Routed-Instance"] == "worker-0"
    assert "data: part-1" in response.text
    assert "data: [DONE]" in response.text

    snapshot = app.state.runtime_state_store.snapshot()
    assert snapshot["worker-0"].queue_len == 0
    assert snapshot["worker-0"].inflight == 0


class _FakeProcessController(ProcessController):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def stop(self, instance) -> None:
        self.calls.append(("stop", instance.id))

    def start(self, instance) -> None:
        self.calls.append(("start", instance.id))

    def restart(self, instance) -> None:
        self.calls.append(("restart", instance.id))


def test_instance_lifecycle_ops_endpoints_update_process_state(tmp_path):
    """stop/start/restart endpoints should update state and use process controller."""
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
    controller = _FakeProcessController()
    config = load_config(config_path)
    app = create_app(config, process_controller=controller)
    client = TestClient(app)

    stop_resp = client.post("/instances/worker-0/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["process_state"] == "stopped"

    start_resp = client.post("/instances/worker-0/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["process_state"] == "running"

    restart_resp = client.post("/instances/worker-0/restart")
    assert restart_resp.status_code == 200
    assert restart_resp.json()["process_state"] == "running"

    assert controller.calls == [("stop", "worker-0"), ("start", "worker-0"), ("restart", "worker-0")]
    instance_view = client.get("/instances").json()["instances"][0]
    assert instance_view["process_state"] == "running"
    assert instance_view["last_operation"] == "restart"


def test_lifecycle_ops_conflict_with_reload_returns_409(tmp_path):
    """Lifecycle op should reject requests while reload is in progress."""
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
    app = create_app(config, process_controller=_FakeProcessController())
    app.state.reload_in_progress = True
    client = TestClient(app)

    response = client.post("/instances/worker-0/restart")
    assert response.status_code == 409
    payload = response.json()
    assert payload["error"]["code"] == "GS_LIFECYCLE_CONFLICT"


def test_lifecycle_ops_unconfigured_command_returns_400(tmp_path):
    """Missing lifecycle command should return GS_LIFECYCLE_UNSUPPORTED."""
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

    response = client.post("/instances/worker-0/stop")
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "GS_LIFECYCLE_UNSUPPORTED"
