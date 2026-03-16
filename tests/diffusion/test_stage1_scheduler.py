# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections import deque
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.stage1_scheduler import Stage1Scheduler

pytestmark = [pytest.mark.diffusion]


def _tagged_output(tag: str) -> DiffusionOutput:
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _mock_request(
    tag: str,
    *,
    num_inference_steps: int = 1,
    resolution: int = 1024,
    num_outputs_per_prompt: int = 1,
    extra_args: dict | None = None,
) -> Mock:
    req = Mock()
    req.request_ids = [tag]
    req.sampling_params = SimpleNamespace(
        num_inference_steps=num_inference_steps,
        resolution=resolution,
        num_outputs_per_prompt=num_outputs_per_prompt,
        num_frames=1,
        width=None,
        height=None,
        extra_args=extra_args or {},
    )
    return req


def _make_stage1_scheduler(
    *,
    policy: str = "fcfs",
    slo_target_ms: float | None = None,
    aging_factor: float = 0.0,
    profile_path: str | None = None,
    profile_name: str | None = None,
):
    sched = Stage1Scheduler()
    sched.num_workers = 1
    sched.od_config = SimpleNamespace(
        num_gpus=1,
        instance_scheduler_policy=policy,
        instance_scheduler_slo_target_ms=slo_target_ms,
        instance_scheduler_slo_floor_ms=0.0,
        instance_scheduler_aging_factor=aging_factor,
        instance_runtime_profile_path=profile_path,
        instance_runtime_profile_name=profile_name,
    )
    sched.initialize(sched.od_config)

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    mock_mq = Mock()
    mock_mq.enqueue = req_q.put

    mock_rmq = Mock()
    mock_rmq.dequeue = lambda timeout=None: res_q.get(timeout=timeout if timeout else 10)

    sched.mq = mock_mq
    sched.result_mq = mock_rmq
    return sched, req_q, res_q


def test_stage1_scheduler_attaches_metrics_on_success():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-success")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1])))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error is None
    assert output.request_id == "req-success"
    assert output.metrics["scheduler_policy"] == "fcfs"
    assert output.metrics["scheduler_latency_ms"] >= 0
    assert output.metrics["queue_wait_ms"] >= 0
    assert output.metrics["scheduler_execute_ms"] >= 0
    assert output.metrics["width"] == 1024
    assert output.metrics["height"] == 1024
    assert output.metrics["total_steps"] == 1
    assert output.metrics["remaining_steps"] == 1
    assert req.first_enqueue_time is not None
    assert req.first_dispatch_time is not None
    assert req.completion_time is not None


def test_stage1_scheduler_normalizes_worker_error_dict():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-fail")

    def _worker():
        req_q.get(timeout=5)
        res_q.put({"status": "error", "error": "boom"})

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.error == "boom"
    assert output.error_code == "WORKER_EXEC_FAILED"
    assert output.request_id == "req-fail"
    assert output.metrics["scheduler_policy"] == "fcfs"


def test_stage1_scheduler_preserves_fcfs_order():
    sched, req_q, res_q = _make_stage1_scheduler()
    seen_request_ids: list[str] = []

    def _worker():
        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            seen_request_ids.append(rpc_request["args"][0].request_ids[0])
            request_id = rpc_request["args"][0].request_ids[0]
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    outputs: dict[str, DiffusionOutput] = {}

    def _run(tag: str):
        outputs[tag] = sched.add_req(_mock_request(tag))

    t1 = threading.Thread(target=_run, args=("req-1",), daemon=True)
    t2 = threading.Thread(target=_run, args=("req-2",), daemon=True)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)
    worker.join(5)

    assert seen_request_ids == ["req-1", "req-2"]
    assert outputs["req-1"].request_id == "req-1"
    assert outputs["req-2"].request_id == "req-2"


def test_stage1_scheduler_keeps_request_running_for_unfinished_output():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-chunk")

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=None, finished=False, metrics={"executed_steps": 2}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.finished is False
    assert req.request_state == "running"
    assert req.last_preempted_time is not None


def test_stage1_scheduler_slo_first_reorders_waiting_queue():
    sched, req_q, res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    active = threading.Thread(
        target=lambda: results.setdefault(
            "active",
            sched.add_req(_mock_request("active", num_inference_steps=1, extra_args={"slo_ms": 5000.0})),
        ),
        daemon=True,
    )
    delayed = threading.Thread(
        target=lambda: results.setdefault(
            "delayed",
            sched.add_req(_mock_request("delayed", num_inference_steps=18, extra_args={"slo_ms": 20000.0})),
        ),
        daemon=True,
    )
    urgent = threading.Thread(
        target=lambda: results.setdefault(
            "urgent",
            sched.add_req(_mock_request("urgent", num_inference_steps=1, extra_args={"slo_ms": 2000.0})),
        ),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    delayed.start()
    urgent.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    delayed.join(5)
    urgent.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "urgent", "delayed"]
    assert results["urgent"].metrics["scheduler_policy"] == "slo_first"
    assert results["urgent"].metrics["self_hit"] == 1
    assert results["urgent"].metrics["queue_reorder_count"] == 1


def test_stage1_scheduler_sjf_uses_remaining_steps():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="sjf")
    long_req = _mock_request("long", num_inference_steps=10)
    short_remaining_req = _mock_request("short-remaining", num_inference_steps=10)
    short_remaining_req.executed_steps = 8

    with sched._queue_cv:
        sched._enqueue_request_locked(long_req)
        sched._enqueue_request_locked(short_remaining_req)

        ordered = [queued.request.request_ids[0] for queued in sched._waiting_queue]

    assert ordered == ["short-remaining", "long"]


def test_stage1_scheduler_deadline_uses_request_arrival_time():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    req = _mock_request("req-deadline", num_inference_steps=10, extra_args={"slo_ms": 2000.0})
    req.arrival_time = 100.0

    with sched._queue_cv:
        queued = sched._enqueue_request_locked(req)

    assert sched._deadline_ts(queued) == pytest.approx(102.0)


def test_stage1_scheduler_sjf_reorders_waiting_queue():
    sched, req_q, res_q = _make_stage1_scheduler(policy="sjf")
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    active = threading.Thread(
        target=lambda: results.setdefault("active", sched.add_req(_mock_request("active", num_inference_steps=1))),
        daemon=True,
    )
    long_waiting = threading.Thread(
        target=lambda: results.setdefault("long", sched.add_req(_mock_request("long", num_inference_steps=20))),
        daemon=True,
    )
    short_waiting = threading.Thread(
        target=lambda: results.setdefault("short", sched.add_req(_mock_request("short", num_inference_steps=1))),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    long_waiting.start()
    short_waiting.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    long_waiting.join(5)
    short_waiting.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "short", "long"]
    assert results["short"].metrics["scheduler_policy"] == "sjf"
    assert results["short"].metrics["queue_reorder_count"] == 1
    assert results["short"].metrics["estimated_cost_s"] < results["long"].metrics["estimated_cost_s"]


def test_stage1_scheduler_sjf_uses_profile_runtime_estimation(tmp_path):
    profile_path = tmp_path / "runtime.json"
    profile_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "instance_type": "profile-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 10,
                        "latency_s": 5.0,
                    },
                    {
                        "instance_type": "profile-a",
                        "task_type": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 50,
                        "latency_s": 0.2,
                    },
                ]
            }
        )
    )
    sched, req_q, res_q = _make_stage1_scheduler(
        policy="sjf",
        profile_path=str(profile_path),
        profile_name="profile-a",
    )
    enqueue_order: list[str] = []
    release_first = threading.Event()
    first_enqueued = threading.Event()
    second_waiting = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        enqueue_order.append(first["args"][0].request_ids[0])
        first_enqueued.set()
        second_waiting.wait(timeout=5)
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        for _ in range(2):
            rpc_request = req_q.get(timeout=5)
            request_id = rpc_request["args"][0].request_ids[0]
            enqueue_order.append(request_id)
            res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id=request_id))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(target=lambda: sched.add_req(_mock_request("active", num_inference_steps=1)), daemon=True)
    short_profiled = threading.Thread(
        target=lambda: sched.add_req(_mock_request("profile-short", num_inference_steps=50)),
        daemon=True,
    )
    long_profiled = threading.Thread(
        target=lambda: sched.add_req(_mock_request("profile-long", num_inference_steps=10)),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    long_profiled.start()
    short_profiled.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    long_profiled.join(5)
    short_profiled.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "profile-short", "profile-long"]


def test_stage1_scheduler_estimate_cost_counts_num_outputs_once_without_profile():
    sched, _, _ = _make_stage1_scheduler(policy="sjf")

    single_output_cost = sched._estimate_cost_seconds(  # noqa: SLF001
        _mock_request("single", num_inference_steps=10, num_outputs_per_prompt=1)
    )
    multi_output_cost = sched._estimate_cost_seconds(  # noqa: SLF001
        _mock_request("multi", num_inference_steps=10, num_outputs_per_prompt=2)
    )

    assert single_output_cost == pytest.approx(10.0)
    assert multi_output_cost == pytest.approx(20.0)
    assert multi_output_cost == pytest.approx(single_output_cost * 2.0)


def test_stage1_scheduler_caches_estimated_cost_for_waiting_plan():
    sched, _, _ = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    original_estimate = sched._estimate_cost_seconds  # noqa: SLF001
    call_count = 0

    def _counting_estimate(request):
        nonlocal call_count
        call_count += 1
        return original_estimate(request)

    sched._estimate_cost_seconds = _counting_estimate  # type: ignore[method-assign]  # noqa: SLF001

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req-1", num_inference_steps=2, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req-2", num_inference_steps=3, extra_args={"slo_ms": 5000.0})
        )
        waiting_requests = list(sched._waiting_queue)  # noqa: SLF001

    sched._build_waiting_plan(waiting_requests, now=time.monotonic())  # noqa: SLF001
    assert call_count == 2

    sched._build_waiting_plan(waiting_requests, now=time.monotonic())  # noqa: SLF001
    sched._build_sjf_queue(waiting_requests, now=time.monotonic())  # noqa: SLF001
    assert call_count == 2


@pytest.mark.parametrize(
    ("method_name", "expected_state"),
    [("finish_request", "finished"), ("fail_request", "failed")],
)
def test_stage1_scheduler_request_terminal_state_updates_hold_queue_lock(method_name: str, expected_state: str):
    sched, _, _ = _make_stage1_scheduler()
    req = _mock_request("req-terminal")
    finished = threading.Event()

    with sched._queue_cv:  # noqa: SLF001
        worker = threading.Thread(target=lambda: (getattr(sched, method_name)(req), finished.set()), daemon=True)
        worker.start()
        time.sleep(0.05)
        assert finished.is_set() is False

    worker.join(5)

    assert finished.is_set() is True
    assert getattr(req, "request_state") == expected_state


def test_stage1_scheduler_reports_waiting_queue_len_and_load():
    sched, req_q, res_q = _make_stage1_scheduler()
    release_first = threading.Event()
    first_enqueued = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        assert first["args"][0].request_ids[0] == "active"
        first_enqueued.set()
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

        second = req_q.get(timeout=5)
        assert second["args"][0].request_ids[0] == "waiting"
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="waiting"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(target=lambda: sched.add_req(_mock_request("active")), daemon=True)
    waiting = threading.Thread(target=lambda: sched.add_req(_mock_request("waiting")), daemon=True)

    active.start()
    first_enqueued.wait(timeout=5)
    waiting.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if sched.estimate_waiting_queue_len() == 1:
            break
        time.sleep(0.01)

    assert sched.estimate_waiting_queue_len() == 1
    assert sched.estimate_scheduler_load() == {
        "waiting_queue_len": 1,
        "active_request_count": 1,
        "paused_context_count": 0,
    }

    release_first.set()
    active.join(5)
    waiting.join(5)
    worker.join(5)


def test_stage1_scheduler_abort_removes_waiting_request():
    sched, req_q, res_q = _make_stage1_scheduler()
    release_first = threading.Event()
    first_enqueued = threading.Event()
    results: dict[str, DiffusionOutput] = {}

    def _worker():
        first = req_q.get(timeout=5)
        assert first["args"][0].request_ids[0] == "active"
        first_enqueued.set()
        release_first.wait(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([0]), request_id="active"))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    active = threading.Thread(
        target=lambda: results.setdefault("active", sched.add_req(_mock_request("active"))),
        daemon=True,
    )
    waiting = threading.Thread(
        target=lambda: results.setdefault("waiting", sched.add_req(_mock_request("waiting"))),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    waiting.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if sched.estimate_waiting_queue_len() == 1:
            break
        time.sleep(0.01)

    assert sched.abort_request("waiting") is True
    release_first.set()

    active.join(5)
    waiting.join(5)
    worker.join(5)

    assert results["waiting"].error_code == "REQUEST_ABORTED"
    assert results["waiting"].error == "Request aborted before dispatch"
    assert sched.estimate_waiting_queue_len() == 0
