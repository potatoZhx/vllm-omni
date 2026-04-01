# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import _BaseScheduler
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    ExecutionOutput,
    NewRequestData,
)
from vllm_omni.diffusion.sched.policy import (
    RequestSelectionPolicy,
    build_request_selection_policy,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)


class StepLevelRequestScheduler(_BaseScheduler):
    """Single-request step-level scheduler for MVP stepwise diffusion."""

    def __init__(self, policy: RequestSelectionPolicy | None = None) -> None:
        super().__init__()
        self._policy = policy

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        if self._policy is None:
            self._policy = build_request_selection_policy(od_config.instance_scheduler_policy)

    def add_request(self, request: OmniDiffusionRequest) -> str:
        if len(request.prompts) != 1:
            raise NotImplementedError(
                "Step-level diffusion scheduling currently supports only single-prompt requests."
            )

        sched_req_id = self._make_sched_req_id(request)
        state = DiffusionRequestState(sched_req_id=sched_req_id, req=request)
        self._request_states[sched_req_id] = state
        self._ensure_execution_state(sched_req_id)
        self._register_request_ids(request.request_ids, sched_req_id)
        self._waiting.append(sched_req_id)
        logger.debug("StepLevelRequestScheduler add_request: %s (waiting=%d)", sched_req_id, len(self._waiting))
        return sched_req_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_req_ids: list[str] = []

        if self._running:
            for sched_req_id in self._running:
                state = self._request_states.get(sched_req_id)
                if state is not None and not state.is_finished():
                    scheduled_cached_req_ids.append(sched_req_id)

        while self._waiting and len(self._running) < self._max_batch_size and not scheduled_cached_req_ids:
            assert self._policy is not None
            ordered_waiting = self._policy.order_waiting(
                list(self._waiting),
                self._request_states,
                self._execution_states,
            )
            if not ordered_waiting:
                break

            sched_req_id = ordered_waiting[0]
            try:
                self._waiting.remove(sched_req_id)
            except ValueError:
                logger.warning("Waiting request %s disappeared before scheduling", sched_req_id)
                continue

            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue

            was_new_request = state.status == DiffusionRequestStatus.WAITING
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(sched_req_id)

            exec_state = self._ensure_execution_state(sched_req_id)
            exec_state.dispatch_epoch += 1
            exec_state.planned_chunk_budget_steps = 1

            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_req_ids.append(sched_req_id)

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(sched_req_ids=scheduled_cached_req_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
        )
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def update_from_output(self, sched_output: DiffusionSchedulerOutput, output: ExecutionOutput) -> set[str]:
        if not isinstance(output, RunnerOutput):
            raise TypeError(f"StepLevelRequestScheduler expects RunnerOutput, got {type(output)!r}")

        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()
        if len(scheduled_req_ids) != 1:
            raise ValueError(
                "Step-level diffusion scheduling currently supports batch_size=1, "
                f"but got {len(scheduled_req_ids)} scheduled requests."
            )

        sched_req_id = scheduled_req_ids[0]
        if output.req_id != sched_req_id:
            raise ValueError(
                f"RunnerOutput req_id mismatch: expected {sched_req_id!r}, got {output.req_id!r}"
            )

        finished_req_ids = {
            running_req_id for running_req_id in scheduled_req_ids if running_req_id in self._finished_req_ids
        }

        state = self._request_states.get(sched_req_id)
        if state is None or state.is_finished():
            return finished_req_ids

        if sched_req_id in self._running:
            self._running.remove(sched_req_id)

        exec_state = self._ensure_execution_state(sched_req_id)
        if output.step_index is not None:
            exec_state.executed_steps = max(exec_state.executed_steps, output.step_index)

        if self.is_abort_pending(sched_req_id):
            finished_req_ids |= self._finish_requests(
                {sched_req_id: DiffusionRequestStatus.FINISHED_ABORTED},
            )
            return finished_req_ids

        if output.finished:
            if output.result is None:
                finished_req_ids |= self._finish_requests(
                    {sched_req_id: DiffusionRequestStatus.FINISHED_ERROR},
                    {sched_req_id: "Step-level diffusion finished without a terminal result."},
                )
            elif output.result.error:
                finished_req_ids |= self._finish_requests(
                    {sched_req_id: DiffusionRequestStatus.FINISHED_ERROR},
                    {sched_req_id: output.result.error},
                )
            else:
                finished_req_ids |= self._finish_requests(
                    {sched_req_id: DiffusionRequestStatus.FINISHED_COMPLETED},
                )
            return finished_req_ids

        state.status = DiffusionRequestStatus.PREEMPTED
        self._waiting.append(sched_req_id)
        return finished_req_ids
