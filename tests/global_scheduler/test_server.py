"""Scheduler server endpoint and lifecycle API tests."""

import textwrap

import pytest
from fastapi.testclient import TestClient

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.process_controller import ProcessController
from vllm_omni.global_scheduler.server import (
    UpstreamHTTPError,
    _extract_request_meta_from_form_fields,
    _extract_request_meta_from_payload,
    create_app,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeProcessController(ProcessController):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def stop(self, instance) -> None:
        self.calls.append(("stop", instance.id))

    def start(self, instance) -> None:
        self.calls.append(("start", instance.id))

    def restart(self, instance) -> None:
        self.calls.append(("restart", instance.id))


def _write_config(path, body: str):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_config(path)


def test_extract_request_meta_from_payload_reads_estimated_cost_s():
    request_meta = _extract_request_meta_from_payload(
        {
            "model": "demo",
            "extra_body": {
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 30,
                "estimated_cost_s": 1.25,
                "slo_ms": 1500,
            },
        },
        request_id="req-est-1",
    )

    assert request_meta.width == 1024
    assert request_meta.height == 1024
    assert request_meta.num_inference_steps == 30
    assert request_meta.estimated_cost_s == pytest.approx(1.25)
    assert request_meta.extra["slo_ms"] == pytest.approx(1500)


def test_extract_request_meta_from_form_fields_reads_estimated_cost_s():
    request_meta = _extract_request_meta_from_form_fields(
        {
            "width": "832",
            "height": "480",
            "num_frames": "33",
            "num_inference_steps": "4",
            "estimated_cost_s": "2.5",
            "slo_ms": "900",
        },
        request_id="req-est-form-1",
    )

    assert request_meta.width == 832
    assert request_meta.height == 480
    assert request_meta.num_frames == 33
    assert request_meta.num_inference_steps == 4
    assert request_meta.estimated_cost_s == pytest.approx(2.5)
    assert request_meta.extra["slo_ms"] == pytest.approx(900)


def test_health_endpoint_returns_ok(tmp_path):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )

    client = TestClient(create_app(config))
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["instance_count"] == 1


def test_instance_lifecycle_control_endpoints(tmp_path):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          instance_health_check_interval_s: 100
          instance_health_check_timeout_s: 0.1
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )

    client = TestClient(create_app(config))
    assert client.get("/instances").json()["instances"][0]["routable"] is True
    assert client.post("/instances/worker-0/disable").json()["enabled"] is False
    assert client.get("/instances").json()["instances"][0]["routable"] is False
    assert client.post("/instances/worker-0/enable").json()["enabled"] is True
    assert client.get("/instances").json()["instances"][0]["routable"] is True


def test_reload_endpoint_replaces_instance_set(tmp_path):
    initial = tmp_path / "scheduler.yaml"
    reloaded = tmp_path / "scheduler_reloaded.yaml"
    config = _write_config(
        initial,
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )
    _write_config(
        reloaded,
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-1
            endpoint: http://127.0.0.1:9002
        """,
    )

    app = create_app(config, config_loader=lambda: load_config(reloaded))
    client = TestClient(app)
    assert [item["id"] for item in client.get("/instances").json()["instances"]] == ["worker-0"]
    response = client.post("/instances/reload")
    assert response.status_code == 200
    assert response.json()["instance_count"] == 1
    assert [item["id"] for item in client.get("/instances").json()["instances"]] == ["worker-1"]


def test_chat_completions_success_sets_route_headers_and_state(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          request_timeout_s: 2
          instance_health_check_interval_s: 100
        policy:
          baseline:
            algorithm: min_queue_length
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )
    app = create_app(config)

    def _fake_proxy(endpoint, upstream_path, body, headers, timeout_s):
        assert endpoint == "http://127.0.0.1:9001"
        assert upstream_path == "/v1/chat/completions"
        assert timeout_s == 2
        assert b'"model": "demo"' in body
        assert headers["content-type"] == "application/json"
        return type(
            "_Resp",
            (),
            {
                "status_code": 200,
                "body": b'{"id": "resp-1"}',
                "headers": {"content-type": "application/json"},
            },
        )()

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_request", _fake_proxy)

    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"content-type": "application/json", "x-request-id": "req-1"},
        json={"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "resp-1"
    assert response.headers["X-Routed-Instance"] == "worker-0"
    assert "router=min_queue_length" in response.headers["X-Route-Reason"]
    snapshot = app.state.runtime_state_store.snapshot()
    assert snapshot["worker-0"].inflight == 0


def test_chat_completions_returns_503_when_no_routable_instance(tmp_path):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )
    app = create_app(config)
    app.state.instance_lifecycle_manager.set_enabled("worker-0", enabled=False)
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json={"model": "demo", "messages": []})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GS_NO_ROUTABLE_INSTANCE"


@pytest.mark.parametrize(
    "raised,status_code,error_code",
    [
        (UpstreamHTTPError(status_code=503, body=b"{}"), 503, "GS_UPSTREAM_HTTP_ERROR"),
        (TimeoutError("timeout"), 502, "GS_UPSTREAM_TIMEOUT"),
        (OSError("network down"), 502, "GS_UPSTREAM_NETWORK_ERROR"),
    ],
)
def test_chat_completions_error_semantics_and_state_cleanup(tmp_path, monkeypatch, raised, status_code, error_code):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          request_timeout_s: 3
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )
    app = create_app(config)

    def _fake_proxy(*_args, **_kwargs):
        raise raised

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_request", _fake_proxy)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-error"},
        json={"model": "demo", "messages": []},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == error_code
    assert app.state.runtime_state_store.snapshot()["worker-0"].inflight == 0


def test_image_generations_success_sets_route_headers_and_state(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          request_timeout_s: 2
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
            backends: [openai]
        """,
    )
    app = create_app(config)

    def _fake_proxy(endpoint, upstream_path, body, headers, timeout_s):
        assert endpoint == "http://127.0.0.1:9001"
        assert upstream_path == "/v1/images/generations"
        assert timeout_s == 2
        assert b'"estimated_cost_s": 1.2' in body
        return type(
            "_Resp",
            (),
            {
                "status_code": 200,
                "body": b'{"created": 1, "data": []}',
                "headers": {"content-type": "application/json"},
            },
        )()

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_request", _fake_proxy)

    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        headers={"content-type": "application/json", "x-request-id": "req-img-1"},
        json={"model": "demo", "prompt": "test", "size": "1024x1024", "estimated_cost_s": 1.2},
    )

    assert response.status_code == 200
    assert response.headers["X-Routed-Instance"] == "worker-0"
    assert app.state.runtime_state_store.snapshot()["worker-0"].inflight == 0


def test_videos_success_sets_route_headers_and_state(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          request_timeout_s: 2
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
            backends: [v1/videos]
        """,
    )
    app = create_app(config)

    def _fake_proxy(endpoint, upstream_path, body, headers, timeout_s):
        assert endpoint == "http://127.0.0.1:9001"
        assert upstream_path == "/v1/videos"
        assert timeout_s == 2
        assert b"estimated_cost_s" in body
        return type(
            "_Resp",
            (),
            {
                "status_code": 200,
                "body": b'{"id": "video-1"}',
                "headers": {"content-type": "application/json"},
            },
        )()

    monkeypatch.setattr("vllm_omni.global_scheduler.server._proxy_request", _fake_proxy)

    client = TestClient(app)
    response = client.post(
        "/v1/videos",
        data={
            "prompt": "city at sunset",
            "width": "832",
            "height": "480",
            "num_frames": "33",
            "num_inference_steps": "4",
            "estimated_cost_s": "2.4",
            "request_id": "req-video-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Routed-Instance"] == "worker-0"
    assert app.state.runtime_state_store.snapshot()["worker-0"].inflight == 0


def test_instance_lifecycle_ops_endpoints_update_process_state(tmp_path):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
        """,
    )
    controller = _FakeProcessController()
    client = TestClient(create_app(config, process_controller=controller))

    assert client.post("/instances/worker-0/stop").json()["process_state"] == "stopped"
    assert client.post("/instances/worker-0/start").json()["process_state"] == "running"
    assert client.post("/instances/worker-0/restart").json()["process_state"] == "running"
    assert controller.calls == [("stop", "worker-0"), ("start", "worker-0"), ("restart", "worker-0")]


def test_server_startup_auto_starts_instances_with_launch_config(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
            launch:
              executable: vllm
              model: Qwen/Qwen-Image
              args:
                - --omni
        """,
    )
    controller = _FakeProcessController()
    app = create_app(config, process_controller=controller)
    monkeypatch.setattr(
        "vllm_omni.global_scheduler.lifecycle._probe_http_ready",
        lambda endpoint, timeout_s: (False, "ready_probe_timeout"),
    )

    with TestClient(app) as client:
        response = client.get("/instances")

    assert response.status_code == 200
    assert controller.calls == [("start", "worker-0")]
    instance_view = response.json()["instances"][0]
    assert instance_view["process_state"] == "running"
    assert instance_view["healthy"] is False
    assert instance_view["routable"] is False


def test_server_shutdown_stops_instances_with_stop_config(tmp_path):
    config = _write_config(
        tmp_path / "scheduler.yaml",
        """
        server:
          instance_health_check_interval_s: 100
        instances:
          - id: worker-0
            endpoint: http://127.0.0.1:9001
            stop:
              executable: pkill
              args:
                - -f
                - worker-0
        """,
    )
    controller = _FakeProcessController()
    app = create_app(config, process_controller=controller)

    with TestClient(app) as client:
        response = client.get("/instances")
        assert response.status_code == 200

    assert controller.calls == [("stop", "worker-0")]
    assert app.state.instance_lifecycle_manager.snapshot()["worker-0"].process_state == "stopped"
