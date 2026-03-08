"""Lifecycle manager health, draining, and reload behavior tests."""

import pytest

from vllm_omni.global_scheduler.lifecycle import InstanceLifecycleManager
from vllm_omni.global_scheduler.state import RuntimeStateStore
from vllm_omni.global_scheduler.types import InstanceSpec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _instances() -> list[InstanceSpec]:
    """Build a deterministic two-instance fixture for lifecycle tests.

    Returns:
        Static instance list used by lifecycle state assertions.
    """
    return [
        InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2),
        InstanceSpec(id="worker-1", endpoint="http://127.0.0.1:9002", max_concurrency=2),
    ]


def test_unhealthy_or_disabled_instances_are_excluded_from_routable_set():
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
    instances = _instances()
    store = RuntimeStateStore(instances=instances)
    manager = InstanceLifecycleManager(instances)

    store.on_request_start("worker-1")
    store.sync_instances([InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2)])

    manager.sync_instances(
        [InstanceSpec(id="worker-0", endpoint="http://127.0.0.1:9001", max_concurrency=2)],
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
    store.on_request_start("worker-1")

    manager.converge_draining(store.snapshot())
    mid = manager.snapshot()
    assert mid["worker-1"].draining is True

    store.on_request_finish("worker-1", latency_s=0.3, ok=False)
    manager.converge_draining(store.snapshot())

    final = manager.snapshot()
    assert "worker-1" in final
    assert final["worker-1"].enabled is False
    assert final["worker-1"].draining is False
