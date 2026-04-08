# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from time import monotonic
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    DiffusionRequestStatus,
    RequestScheduler,
    Scheduler,
    SchedulerInterface,
    StepLevelRequestScheduler,
)
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionExecutionState,
    DiffusionRequestState,
    NewRequestData,
)
from vllm_omni.diffusion.sched.policy import (
    P95FirstSelectionPolicy,
    SJFAgingGuardedSelectionPolicy,
    SJFAgingGuardedTailSelectionPolicy,
    SJFAgingSelectionPolicy,
    SJFSelectionPolicy,
    build_request_selection_policy,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(
    req_id: str,
    *,
    num_inference_steps: int = 1,
    estimated_cost_s: float | None = None,
    resolution: int = 1024,
    num_outputs_per_prompt: int = 1,
) -> OmniDiffusionRequest:
    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=num_inference_steps,
        resolution=resolution,
        num_outputs_per_prompt=num_outputs_per_prompt,
    )
    if estimated_cost_s is not None:
        sampling_params.extra_args["estimated_cost_s"] = estimated_cost_s
    return OmniDiffusionRequest(
        prompts=[f"prompt_{req_id}"],
        sampling_params=sampling_params,
        request_ids=[req_id],
    )


def _make_request_output(req_id: str, *, error: str | None = None) -> DiffusionOutput:
    del req_id
    return DiffusionOutput(output=None, error=error)


def _new_ids(sched_output) -> list[str]:
    return [req.sched_req_id for req in sched_output.scheduled_new_reqs]


def _cached_ids(sched_output) -> list[str]:
    return list(sched_output.scheduled_cached_reqs.sched_req_ids)


def _set_request_age(scheduler: StepLevelRequestScheduler, sched_req_id: str, age_s: float) -> None:
    scheduler.get_execution_state(sched_req_id).arrival_time = monotonic() - age_s


def _render_log_messages(log_calls) -> list[str]:
    return [call.args[0] % call.args[1:] for call in log_calls.call_args_list]


class _StubScheduler(SchedulerInterface):
    def __init__(self, request: OmniDiffusionRequest, output: DiffusionOutput) -> None:
        self._request = request
        self._output = output
        self.initialized_with = None
        self._sched_req_id = request.request_ids[0]
        self._state = None
        self._scheduled = False
        self._execution_state = None

    def initialize(self, od_config) -> None:
        self.initialized_with = od_config

    def add_request(self, request: OmniDiffusionRequest) -> str:
        assert request is self._request
        self._state = Mock(sched_req_id=self._sched_req_id, req=request)
        self._execution_state = DiffusionExecutionState(sched_req_id=self._sched_req_id)
        return self._sched_req_id

    def schedule(self):
        if self._scheduled or self._state is None:
            return Mock(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                scheduled_req_ids=[],
                is_empty=True,
            )
        self._scheduled = True
        return Mock(
            scheduled_new_reqs=[NewRequestData.from_state(self._state)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            scheduled_req_ids=[self._state.sched_req_id],
            is_empty=False,
        )

    def update_from_output(self, sched_output, output: DiffusionOutput) -> set[str]:
        del sched_output
        assert output is self._output
        return {self._sched_req_id}

    def has_requests(self) -> bool:
        return not self._scheduled

    def get_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def get_execution_state(self, sched_req_id: str):
        del sched_req_id
        return self._execution_state

    def get_sched_req_id(self, request_id: str) -> str | None:
        if request_id in self._request.request_ids:
            return self._sched_req_id
        return None

    def pop_request_state(self, sched_req_id: str):
        del sched_req_id
        return self._state

    def preempt_request(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def mark_abort_pending(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def is_abort_pending(self, sched_req_id: str) -> bool:
        del sched_req_id
        return False

    def finish_requests(self, sched_req_ids, status) -> None:
        del sched_req_ids, status
        return None

    def close(self) -> None:
        return None


class TestRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler: RequestScheduler = RequestScheduler()
        self.scheduler.initialize(Mock())

    def test_single_request_success_lifecycle(self) -> None:
        req_id = self.scheduler.add_request(_make_request("a"))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.WAITING

        sched_output = self.scheduler.schedule()
        assert _new_ids(sched_output) == [req_id]
        assert _cached_ids(sched_output) == []
        assert sched_output.num_running_reqs == 1
        assert sched_output.num_waiting_reqs == 0

        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_error_output_marks_finished_error(self) -> None:
        req_id = self.scheduler.add_request(_make_request("err"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(
            sched_output,
            _make_request_output(req_id, error="worker failed"),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ERROR
        assert state.error == "worker failed"

    def test_empty_output_without_error_marks_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("empty"))

        sched_output = self.scheduler.schedule()
        finished = self.scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_fifo_single_request_scheduling(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id_a]
        assert _cached_ids(first) == []
        assert first.num_running_reqs == 1
        assert first.num_waiting_reqs == 1

        # Request A is still running; scheduling again should not pull B.
        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id_a]
        assert second.num_running_reqs == 1
        assert second.num_waiting_reqs == 1

        self.scheduler.update_from_output(first, _make_request_output(req_id_a))

        third = self.scheduler.schedule()
        assert _new_ids(third) == [req_id_b]
        assert _cached_ids(third) == []
        assert third.num_running_reqs == 1
        assert third.num_waiting_reqs == 0

    def test_abort_request_for_waiting_and_running(self) -> None:
        req_id_a = self.scheduler.add_request(_make_request("a"))
        req_id_b = self.scheduler.add_request(_make_request("b"))

        # Abort waiting request.
        self.scheduler.finish_requests(req_id_b, DiffusionRequestStatus.FINISHED_ABORTED)
        state_b = self.scheduler.get_request_state(req_id_b)
        assert state_b.status == DiffusionRequestStatus.FINISHED_ABORTED

        # A should still run normally.
        output_a = self.scheduler.schedule()
        assert _new_ids(output_a) == [req_id_a]

        # Abort running request.
        self.scheduler.finish_requests(req_id_a, DiffusionRequestStatus.FINISHED_ABORTED)
        state_a = self.scheduler.get_request_state(req_id_a)
        assert state_a.status == DiffusionRequestStatus.FINISHED_ABORTED

        assert self.scheduler.has_requests() is False
        assert self.scheduler.schedule().scheduled_req_ids == []

    def test_has_requests_state_transition(self) -> None:
        assert self.scheduler.has_requests() is False

        req_id = self.scheduler.add_request(_make_request("has"))
        assert self.scheduler.has_requests() is True

        sched_output = self.scheduler.schedule()
        assert self.scheduler.has_requests() is True

        self.scheduler.update_from_output(sched_output, _make_request_output(req_id))
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        assert self.scheduler.has_requests() is False

    def test_request_id_mapping_lifecycle(self) -> None:
        request = OmniDiffusionRequest(
            prompts=["prompt_map_a", "prompt_map_b"],
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_ids=["map-a", "map-b"],
        )

        sched_req_id = self.scheduler.add_request(request)

        assert self.scheduler.get_sched_req_id("map-a") == sched_req_id
        assert self.scheduler.get_sched_req_id("map-b") == sched_req_id

        self.scheduler.pop_request_state(sched_req_id)

        assert self.scheduler.get_sched_req_id("map-a") is None
        assert self.scheduler.get_sched_req_id("map-b") is None


class TestStepLevelRequestScheduler:
    def setup_method(self) -> None:
        self.scheduler = StepLevelRequestScheduler()
        self.scheduler.initialize(Mock(instance_scheduler_policy="fcfs"))

    def _make_scheduler(self, policy: str) -> StepLevelRequestScheduler:
        scheduler = StepLevelRequestScheduler()
        scheduler.initialize(
            Mock(
                instance_scheduler_policy=policy,
                instance_runtime_profile_path=None,
                instance_runtime_profile_name=None,
            )
        )
        return scheduler

    def test_single_request_requeues_as_cached_after_unfinished_step(self) -> None:
        req_id = self.scheduler.add_request(_make_request("step"))

        first = self.scheduler.schedule()
        assert _new_ids(first) == [req_id]
        assert _cached_ids(first) == []

        finished = self.scheduler.update_from_output(
            first,
            RunnerOutput(req_id=req_id, step_index=1, finished=False, result=None),
        )
        assert finished == set()
        state = self.scheduler.get_request_state(req_id)
        exec_state = self.scheduler.get_execution_state(req_id)
        assert state.status == DiffusionRequestStatus.PREEMPTED
        assert exec_state.executed_steps == 1

        second = self.scheduler.schedule()
        assert _new_ids(second) == []
        assert _cached_ids(second) == [req_id]

    def test_fcfs_resumes_original_arrival_before_later_request(self) -> None:
        first_req_id = self.scheduler.add_request(_make_request("first", num_inference_steps=2))
        second_req_id = self.scheduler.add_request(_make_request("second", num_inference_steps=2))

        first = self.scheduler.schedule()
        assert first.scheduled_req_ids == [first_req_id]

        finished = self.scheduler.update_from_output(
            first,
            RunnerOutput(req_id=first_req_id, step_index=1, finished=False, result=None),
        )

        assert finished == set()
        assert list(self.scheduler._waiting) == [second_req_id, first_req_id]  # noqa: SLF001

        second = self.scheduler.schedule()

        assert second.scheduled_req_ids == [first_req_id]
        assert _new_ids(second) == []
        assert _cached_ids(second) == [first_req_id]

    def test_finished_runner_output_marks_request_completed(self) -> None:
        req_id = self.scheduler.add_request(_make_request("done"))
        sched_output = self.scheduler.schedule()

        finished = self.scheduler.update_from_output(
            sched_output,
            RunnerOutput(
                req_id=req_id,
                step_index=2,
                finished=True,
                result=DiffusionOutput(output=None),
            ),
        )

        assert finished == {req_id}
        assert self.scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_logs_per_step_schedule_and_completion_details(self) -> None:
        req_id = self.scheduler.add_request(
            _make_request("step-log", num_inference_steps=2, estimated_cost_s=4.0),
        )

        with (
            patch("vllm_omni.diffusion.sched.step_level_request_scheduler.logger.isEnabledFor", return_value=True),
            patch("vllm_omni.diffusion.sched.step_level_request_scheduler.logger.info") as log_info,
        ):
            first = self.scheduler.schedule()
            self.scheduler.update_from_output(
                first,
                RunnerOutput(req_id=req_id, step_index=1, finished=False, result=None),
            )

            second = self.scheduler.schedule()
            self.scheduler.update_from_output(
                second,
                RunnerOutput(
                    req_id=req_id,
                    step_index=2,
                    finished=True,
                    result=DiffusionOutput(output=None),
                ),
            )

        rendered = _render_log_messages(log_info)
        assert any(
            "[StepSchedule]" in message
            and "policy=fcfs" in message
            and "selected=step-log" in message
            and "kind=new" in message
            and "progress=0/2" in message
            for message in rendered
        )
        assert any(
            "[StepComplete]" in message
            and "req=step-log" in message
            and "progress=0->1/2" in message
            and "status=PREEMPTED" in message
            for message in rendered
        )
        assert any(
            "[StepSchedule]" in message
            and "selected=step-log" in message
            and "kind=resumed" in message
            and "progress=1/2" in message
            for message in rendered
        )
        assert any(
            "[StepComplete]" in message
            and "req=step-log" in message
            and "progress=1->2/2" in message
            and "status=FINISHED_COMPLETED" in message
            for message in rendered
        )

    def test_abort_pending_finishes_on_next_scheduler_update(self) -> None:
        req_id = self.scheduler.add_request(_make_request("abort"))
        sched_output = self.scheduler.schedule()
        assert self.scheduler.mark_abort_pending(req_id) is True

        finished = self.scheduler.update_from_output(
            sched_output,
            RunnerOutput(req_id=req_id, step_index=1, finished=False, result=None),
        )

        assert finished == {req_id}
        state = self.scheduler.get_request_state(req_id)
        assert state.status == DiffusionRequestStatus.FINISHED_ABORTED

    def test_policy_builder_returns_migrated_policy_implementations(self) -> None:
        assert isinstance(build_request_selection_policy("sjf"), SJFSelectionPolicy)
        assert isinstance(build_request_selection_policy("sjf_aging"), SJFAgingSelectionPolicy)
        assert isinstance(build_request_selection_policy("sjf_aging_guarded"), SJFAgingGuardedSelectionPolicy)
        assert isinstance(build_request_selection_policy("sjf_aging_guard"), SJFAgingGuardedSelectionPolicy)
        assert isinstance(build_request_selection_policy("sjf_aging_guarded_tail"), SJFAgingGuardedTailSelectionPolicy)
        assert isinstance(build_request_selection_policy("p95-first"), P95FirstSelectionPolicy)

    def test_sjf_uses_remaining_estimated_runtime(self) -> None:
        scheduler = self._make_scheduler("sjf")
        long_req_id = scheduler.add_request(_make_request("long", num_inference_steps=10, estimated_cost_s=10.0))
        short_req_id = scheduler.add_request(_make_request("short-remaining", num_inference_steps=10, estimated_cost_s=10.0))
        scheduler.get_execution_state(short_req_id).executed_steps = 8

        sched_output = scheduler.schedule()

        assert sched_output.scheduled_req_ids == [short_req_id]
        assert long_req_id in scheduler._waiting  # noqa: SLF001

    def test_sjf_aging_promotes_old_request_over_short_new_request(self) -> None:
        scheduler = self._make_scheduler("sjf_aging")
        old_req_id = scheduler.add_request(_make_request("old", num_inference_steps=10, estimated_cost_s=10.0))
        new_req_id = scheduler.add_request(_make_request("new", num_inference_steps=1, estimated_cost_s=1.0))
        _set_request_age(scheduler, old_req_id, 20.0)
        _set_request_age(scheduler, new_req_id, 0.0)

        sched_output = scheduler.schedule()

        assert sched_output.scheduled_req_ids == [old_req_id]

    def test_sjf_aging_guarded_prioritizes_protected_request(self) -> None:
        scheduler = self._make_scheduler("sjf_aging_guarded")
        old_req_id = scheduler.add_request(_make_request("old-large", num_inference_steps=35, estimated_cost_s=37.0))
        new_req_id = scheduler.add_request(_make_request("new-medium", num_inference_steps=25, estimated_cost_s=12.0))
        _set_request_age(scheduler, old_req_id, 80.0)
        _set_request_age(scheduler, new_req_id, 0.0)

        sched_output = scheduler.schedule()

        assert sched_output.scheduled_req_ids == [old_req_id]

    def test_sjf_aging_guarded_tail_sinks_old_super_heavy_request(self) -> None:
        scheduler = self._make_scheduler("sjf_aging_guarded_tail")
        policy = scheduler._policy
        assert isinstance(policy, SJFAgingGuardedTailSelectionPolicy)
        for index in range(20):
            prime_req_id = f"prime-{index}"
            prime_state = DiffusionRequestState(
                sched_req_id=prime_req_id,
                req=_make_request(prime_req_id, estimated_cost_s=1.0),
            )
            prime_exec_state = DiffusionExecutionState(
                sched_req_id=prime_req_id,
                arrival_time=monotonic(),
                estimated_runtime_s=1.0,
            )
            policy.on_request_arrival(prime_req_id, prime_state, prime_exec_state)

        heavy_req_id = scheduler.add_request(_make_request("heavy", num_inference_steps=35, estimated_cost_s=40.0))
        short_a_req_id = scheduler.add_request(_make_request("short-a", num_inference_steps=10, estimated_cost_s=2.0))
        short_b_req_id = scheduler.add_request(_make_request("short-b", num_inference_steps=10, estimated_cost_s=3.0))
        _set_request_age(scheduler, heavy_req_id, 130.0)
        _set_request_age(scheduler, short_a_req_id, 2.0)
        _set_request_age(scheduler, short_b_req_id, 1.0)

        ordered_waiting = policy.order_waiting(
            list(scheduler._waiting),  # noqa: SLF001
            scheduler._request_states,  # noqa: SLF001
            scheduler._execution_states,  # noqa: SLF001
        )

        assert ordered_waiting == [short_a_req_id, short_b_req_id, heavy_req_id]

    def test_p95_first_orders_by_normalized_tail_pressure(self) -> None:
        scheduler = self._make_scheduler("p95-first")
        policy = scheduler._policy
        assert isinstance(policy, P95FirstSelectionPolicy)
        policy._observed_service_ms_per_work_unit = 1000.0  # noqa: SLF001
        policy._slowdown_history.append(2.0)  # noqa: SLF001

        old_req_id = scheduler.add_request(_make_request("old", num_inference_steps=10, estimated_cost_s=10.0))
        new_req_id = scheduler.add_request(_make_request("new", num_inference_steps=1, estimated_cost_s=1.0))
        _set_request_age(scheduler, old_req_id, 20.0)
        _set_request_age(scheduler, new_req_id, 0.0)

        ordered_waiting = policy.order_waiting(
            list(scheduler._waiting),  # noqa: SLF001
            scheduler._request_states,  # noqa: SLF001
            scheduler._execution_states,  # noqa: SLF001
        )

        assert ordered_waiting == [old_req_id, new_req_id]

    def test_p95_first_updates_service_rate_and_slowdown_from_runtime_hooks(self) -> None:
        policy = P95FirstSelectionPolicy()
        policy.initialize(Mock(instance_runtime_profile_path=None, instance_runtime_profile_name=None))

        request_state = DiffusionRequestState(
            sched_req_id="p95",
            req=_make_request("p95", num_inference_steps=2, estimated_cost_s=2.0),
        )
        execution_state = DiffusionExecutionState(
            sched_req_id="p95",
            arrival_time=0.0,
            estimated_runtime_s=2.0,
            executed_steps=1,
            cumulative_execute_time_s=0.4,
        )

        policy.on_step_complete(
            "p95",
            request_state,
            execution_state,
            RunnerOutput(req_id="p95", step_index=1, finished=False, result=None),
            0,
            0.4,
        )
        policy.on_request_finished(
            "p95",
            request_state,
            execution_state,
            DiffusionRequestStatus.FINISHED_COMPLETED,
            1.0,
        )

        assert policy._observed_service_ms_per_work_unit == pytest.approx(400.0)  # noqa: SLF001
        assert policy._learned_slowdown_p95() == pytest.approx(2.5)  # noqa: SLF001


class TestDiffusionEngine:
    def test_add_req_and_wait_for_response_single_path(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = RequestScheduler()
        engine.scheduler.initialize(Mock())
        engine.executor = Mock()
        engine._rpc_lock = threading.Lock()

        request = _make_request("engine")
        expected = DiffusionOutput(output=None)
        engine.executor.add_req.return_value = expected

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_supports_scheduler_interface_injection(self) -> None:
        request = _make_request("engine_iface")
        expected = DiffusionOutput(output=None)
        scheduler = _StubScheduler(request, expected)

        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.scheduler = scheduler
        engine.executor = Mock()
        engine.executor.add_req = Mock(return_value=expected)
        engine._rpc_lock = threading.Lock()

        output = engine.add_req_and_wait_for_response(request)

        assert output is expected
        engine.executor.add_req.assert_called_once_with(request)

    def test_initializes_injected_scheduler(self) -> None:
        request = _make_request("init")
        scheduler = _StubScheduler(request, DiffusionOutput(output=None))
        od_config = Mock(model_class_name="mock_model")
        fake_executor_cls = Mock(return_value=Mock())

        with (
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func", return_value=None),
            patch("vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class", return_value=fake_executor_cls),
            patch.object(DiffusionEngine, "_dummy_run", return_value=None),
        ):
            DiffusionEngine(od_config, scheduler=scheduler)

        assert scheduler.initialized_with is od_config
        fake_executor_cls.assert_called_once_with(od_config)

    def test_scheduler_alias_keeps_default_request_scheduler(self) -> None:
        scheduler = Scheduler()
        scheduler.initialize(Mock())

        req_id = scheduler.add_request(_make_request("alias"))
        sched_output = scheduler.schedule()
        finished = scheduler.update_from_output(sched_output, _make_request_output(req_id))

        assert req_id in finished
        assert scheduler.get_request_state(req_id).status == DiffusionRequestStatus.FINISHED_COMPLETED

    def test_dummy_run_raises_on_output_error(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        engine.od_config = Mock(model_class_name="mock_model")
        engine.pre_process_func = None
        engine.add_req_and_wait_for_response = Mock(return_value=DiffusionOutput(error="boom"))

        with pytest.raises(RuntimeError, match="Dummy run failed: boom"):
            engine._dummy_run()

    def test_step_level_engine_waits_for_terminal_output(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        scheduler = StepLevelRequestScheduler()
        scheduler.initialize(Mock(instance_scheduler_policy="fcfs"))
        engine.scheduler = scheduler
        engine.executor = Mock()
        engine._engine_lock = threading.Lock()
        engine._rpc_lock = threading.Lock()
        engine._scheduler_cv = threading.Condition(engine._engine_lock)
        engine._request_events = {}
        engine._request_outputs = {}
        engine._fatal_error = None
        engine._closed = False
        expected = DiffusionOutput(output=None)
        request = _make_request("step_engine")

        def _execute_stepwise(sched_output):
            sched_req_id = sched_output.scheduled_req_ids[0]
            exec_state = scheduler.get_execution_state(sched_req_id)
            if exec_state.executed_steps == 0:
                return RunnerOutput(req_id=sched_req_id, step_index=1, finished=False, result=None)
            return RunnerOutput(req_id=sched_req_id, step_index=2, finished=True, result=expected)

        engine.executor.execute_stepwise.side_effect = _execute_stepwise
        engine.executor.shutdown = Mock()

        scheduler_thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
        engine._scheduler_thread = scheduler_thread
        scheduler_thread.start()

        output = engine.add_req_and_wait_for_response(request)
        engine.close()

        assert output is expected
        assert engine.executor.execute_stepwise.call_count == 2

    def test_step_level_engine_close_unblocks_waiter(self) -> None:
        engine = DiffusionEngine.__new__(DiffusionEngine)
        scheduler = StepLevelRequestScheduler()
        scheduler.initialize(Mock(instance_scheduler_policy="fcfs"))
        engine.scheduler = scheduler
        engine.executor = Mock()
        engine._engine_lock = threading.Lock()
        engine._rpc_lock = threading.Lock()
        engine._scheduler_cv = threading.Condition(engine._engine_lock)
        engine._request_events = {}
        engine._request_outputs = {}
        engine._fatal_error = None
        engine._closed = False
        engine.executor.shutdown = Mock()

        def _blocked_execute_stepwise(sched_output):
            threading.Event().wait(0.2)
            return RunnerOutput(
                req_id=sched_output.scheduled_req_ids[0],
                step_index=1,
                finished=False,
                result=None,
            )

        engine.executor.execute_stepwise.side_effect = _blocked_execute_stepwise

        scheduler_thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
        engine._scheduler_thread = scheduler_thread
        scheduler_thread.start()

        results = {}

        def _run_request():
            results["output"] = engine.add_req_and_wait_for_response(_make_request("close_me"))

        request_thread = threading.Thread(target=_run_request, daemon=True)
        request_thread.start()
        threading.Event().wait(0.05)

        engine.close()
        request_thread.join(timeout=5)

        assert results["output"].error == "DiffusionEngine closed."
