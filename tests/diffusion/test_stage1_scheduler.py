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
    assert output.metrics["executed_steps"] == 1
    assert output.metrics["remaining_steps"] == 0
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
    req = _mock_request("req-chunk", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=None, finished=False, metrics={"executed_steps": 2}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.finished is False
    assert req.request_state == "waiting"
    assert req.executed_steps == 2
    assert output.metrics["executed_steps"] == 2
    assert output.metrics["remaining_steps"] == 2
    assert req.last_preempted_time is not None
    with sched._queue_cv:
        assert len(sched._waiting_queue) == 1
        assert sched._waiting_queue[0].request is req


def test_stage1_scheduler_reuses_waiting_entry_for_resumed_request():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-resume", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=None, finished=False, metrics={"executed_steps": 2}))
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, metrics={"executed_steps": 4}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    first_output = sched.add_req(req)
    assert first_output.finished is False

    enqueue_calls: list[str] = []
    original_enqueue = sched._enqueue_request_locked

    def _record_enqueue(request):
        enqueue_calls.append(request.request_ids[0])
        return original_enqueue(request)

    sched._enqueue_request_locked = _record_enqueue
    second_output = sched.add_req(req)
    worker.join(5)

    assert second_output.finished is True
    assert enqueue_calls == []


def test_stage1_scheduler_sjf_prioritizes_requeued_shorter_request_over_new_longer_request():
    sched, req_q, res_q = _make_stage1_scheduler(policy="sjf")
    req1 = _mock_request("req-1", num_inference_steps=41, extra_args={"estimated_cost_s": 2.0})
    req2 = _mock_request("req-2", num_inference_steps=28, extra_args={"estimated_cost_s": 10.0})
    dispatch_order: list[str] = []
    req1_requeued = threading.Event()
    allow_req1_resume = threading.Event()

    def _worker():
        first = req_q.get(timeout=5)
        dispatch_order.append(first["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=None, finished=False, request_id="req-1", metrics={"executed_steps": 30}))

        second = req_q.get(timeout=5)
        dispatch_order.append(second["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-1", metrics={"executed_steps": 41}))

        third = req_q.get(timeout=5)
        dispatch_order.append(third["args"][0].request_ids[0])
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True, request_id="req-2", metrics={"executed_steps": 28}))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    results: dict[str, DiffusionOutput] = {}

    def _run_req1():
        first_output = sched.add_req(req1)
        results["req-1-first"] = first_output
        req1_requeued.set()
        allow_req1_resume.wait(timeout=5)
        results["req-1-second"] = sched.add_req(req1)

    req1_thread = threading.Thread(target=_run_req1, daemon=True)
    req1_thread.start()
    assert req1_requeued.wait(timeout=5)

    req2_thread = threading.Thread(
        target=lambda: results.setdefault("req-2", sched.add_req(req2)),
        daemon=True,
    )
    req2_thread.start()

    time.sleep(0.1)
    allow_req1_resume.set()

    req1_thread.join(5)
    req2_thread.join(5)
    worker.join(5)

    assert dispatch_order == ["req-1", "req-1", "req-2"]
    assert results["req-1-first"].finished is False
    assert results["req-1-second"].finished is True
    assert results["req-2"].finished is True


def test_stage1_scheduler_marks_finished_request_as_fully_executed_without_worker_metric():
    sched, req_q, res_q = _make_stage1_scheduler()
    req = _mock_request("req-finished", num_inference_steps=4)

    def _worker():
        req_q.get(timeout=5)
        res_q.put(DiffusionOutput(output=torch.tensor([1]), finished=True))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    output = sched.add_req(req)
    worker.join(5)

    assert output.finished is True
    assert req.request_state == "finished"
    assert req.executed_steps == 4
    assert output.metrics["executed_steps"] == 4
    assert output.metrics["remaining_steps"] == 0


def test_stage1_scheduler_slo_first_reorders_waiting_queue_by_slack_over_remaining_cost():
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
    tighter = threading.Thread(
        target=lambda: results.setdefault(
            "tighter",
            sched.add_req(
                _mock_request(
                    "tighter",
                    num_inference_steps=2,
                    extra_args={"slo_ms": 3000.0, "estimated_cost_s": 2.0},
                )
            ),
        ),
        daemon=True,
    )
    looser = threading.Thread(
        target=lambda: results.setdefault(
            "looser",
            sched.add_req(
                _mock_request(
                    "looser",
                    num_inference_steps=1,
                    extra_args={"slo_ms": 2500.0, "estimated_cost_s": 1.0},
                )
            ),
        ),
        daemon=True,
    )

    active.start()
    first_enqueued.wait(timeout=5)
    tighter.start()
    looser.start()
    second_waiting.set()
    release_first.set()

    active.join(5)
    tighter.join(5)
    looser.join(5)
    worker.join(5)

    assert enqueue_order == ["active", "looser", "tighter"]
    assert results["looser"].metrics["scheduler_policy"] == "slo_first"
    assert results["looser"].metrics["self_hit"] == 1
    assert results["looser"].metrics["queue_reorder_count"] == 1


def test_stage1_scheduler_slo_first_splits_on_time_and_best_effort_sets():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-4", num_inference_steps=4, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-3", num_inference_steps=3, extra_args={"slo_ms": 5000.0})
        )
        sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-2", num_inference_steps=2, extra_args={"slo_ms": 5000.0})
        )
        last = sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("cost-1", num_inference_steps=1, extra_args={"slo_ms": 5000.0})
        )

        ordered = [queued.request.request_ids[0] for queued in sched._waiting_queue]  # noqa: SLF001

    assert ordered == ["cost-2", "cost-1", "cost-3", "cost-4"]
    assert last.schedule_metrics["on_time_set_size"] == 2
    assert last.schedule_metrics["best_effort_set_size"] == 2
    assert last.schedule_metrics["dispatch_group"] == "on_time"


def test_stage1_scheduler_slo_first_orders_on_time_set_by_slack_over_remaining_cost():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0)
    tighter = _mock_request(
        "tighter",
        num_inference_steps=2,
        extra_args={"slo_ms": 3000.0, "estimated_cost_s": 2.0},
    )
    looser = _mock_request(
        "looser",
        num_inference_steps=1,
        extra_args={"slo_ms": 2500.0, "estimated_cost_s": 1.0},
    )

    with sched._queue_cv:  # noqa: SLF001
        q1 = sched._enqueue_request_locked(tighter)  # noqa: SLF001
        q2 = sched._enqueue_request_locked(looser)  # noqa: SLF001
        plan = sched._build_waiting_plan(list(sched._waiting_queue), now=q1.enqueue_time)  # noqa: SLF001

    assert q1.sequence_id in plan.feasible_ids
    assert q2.sequence_id in plan.feasible_ids
    assert [queued.request.request_ids[0] for queued in plan.on_time_queue] == [
        "tighter",
        "looser",
    ]


def test_stage1_scheduler_slack_age_prefers_older_request_when_slack_is_tied():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slack_age", slo_target_ms=5000.0, aging_factor=1.0)
    older = _mock_request("older", num_inference_steps=2, extra_args={"deadline_ts": 24.0, "estimated_cost_s": 2.0})
    newer = _mock_request("newer", num_inference_steps=1, extra_args={"deadline_ts": 23.0, "estimated_cost_s": 1.0})
    older.arrival_time = 10.0
    newer.arrival_time = 15.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.on_time_queue] == ["older", "newer"]


def test_stage1_scheduler_slack_cost_age_penalizes_large_request_when_slack_and_age_match():
    sched, _req_q, _res_q = _make_stage1_scheduler(policy="slack_cost_age", slo_target_ms=5000.0, aging_factor=0.0)
    large = _mock_request("large", num_inference_steps=4, extra_args={"slo_ms": 8000.0, "estimated_cost_s": 4.0})
    small = _mock_request("small", num_inference_steps=1, extra_args={"slo_ms": 5000.0, "estimated_cost_s": 1.0})
    large.arrival_time = 10.0
    small.arrival_time = 10.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(large)  # noqa: SLF001
        sched._enqueue_request_locked(small)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=10.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.on_time_queue] == ["small", "large"]


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


def test_stage1_scheduler_scales_injected_estimated_cost_by_remaining_steps():
    sched, _, _ = _make_stage1_scheduler(policy="sjf")
    req = _mock_request(
        "chunked",
        num_inference_steps=20,
        extra_args={"estimated_cost_s": 5.0},
    )

    initial_cost = sched._estimate_cost_seconds(req)  # noqa: SLF001
    req.executed_steps = 10
    resumed_cost = sched._estimate_cost_seconds(req)  # noqa: SLF001

    assert initial_cost == pytest.approx(5.0)
    assert resumed_cost == pytest.approx(2.5)


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


def test_stage1_scheduler_best_effort_aging_uses_request_arrival_time():
    sched, _, _ = _make_stage1_scheduler(policy="slo_first", slo_target_ms=5000.0, aging_factor=1.0)
    older = _mock_request("older", num_inference_steps=8, extra_args={"slo_ms": 5000.0})
    newer = _mock_request("newer", num_inference_steps=7, extra_args={"slo_ms": 5000.0})
    older.arrival_time = 10.0
    newer.arrival_time = 18.0

    with sched._queue_cv:  # noqa: SLF001
        sched._enqueue_request_locked(older)  # noqa: SLF001
        sched._enqueue_request_locked(newer)  # noqa: SLF001
        waiting = list(sched._waiting_queue)  # noqa: SLF001

    plan = sched._build_waiting_plan(waiting, now=20.0)  # noqa: SLF001

    assert [queued.request.request_ids[0] for queued in plan.best_effort_queue] == ["older", "newer"]


@pytest.mark.parametrize("policy", ["slack_age", "slack_cost_age"])
def test_stage1_scheduler_deadline_aware_policies_report_policy_name(policy: str):
    sched, _req_q, _res_q = _make_stage1_scheduler(policy=policy, slo_target_ms=5000.0, aging_factor=1.0)

    with sched._queue_cv:  # noqa: SLF001
        queued = sched._enqueue_request_locked(  # noqa: SLF001
            _mock_request("req", num_inference_steps=2, extra_args={"slo_ms": 5000.0, "estimated_cost_s": 2.0})
        )

    assert queued.schedule_metrics["scheduler_policy"] == policy


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
