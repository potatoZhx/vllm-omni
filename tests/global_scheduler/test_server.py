import textwrap

import pytest
from fastapi.testclient import TestClient

from vllm_omni.global_scheduler.config import load_config
from vllm_omni.global_scheduler.server import create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_health_endpoint_returns_scheduler_and_ok(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: ondisc
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
    assert payload["scheduler"] == "ondisc"
    assert payload["instance_count"] == 1
    assert payload["checks"]["config_loaded"] is True
    assert payload["checks"]["has_instances"] is True
    assert payload["checks"]["scheduler_type_valid"] is True
    assert "version" in payload


def test_health_endpoint_returns_503_when_config_missing(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            scheduler:
              type: baseline
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
    assert payload["checks"]["scheduler_type_valid"] is False


def test_load_config_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "not_found.yaml"

    with pytest.raises(ValueError, match="Config file not found"):
        load_config(missing)


def test_instance_lifecycle_control_endpoints(tmp_path):
    config_path = tmp_path / "scheduler.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            server:
              instance_health_check_interval_s: 100
              instance_health_check_timeout_s: 0.1
            scheduler:
              type: baseline
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
