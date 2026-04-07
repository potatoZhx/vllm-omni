# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from time import monotonic

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
    normalize_request_selection_policy_name,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

logger = init_logger(__name__)
_WAITING_PREVIEW_LIMIT = 5


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _safe_optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _request_total_steps(request: OmniDiffusionRequest) -> int:
    return max(_safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)


class StepLevelRequestScheduler(_BaseScheduler):
    """Single-request step-level scheduler for MVP stepwise diffusion."""

    def __init__(self, policy: RequestSelectionPolicy | None = None) -> None:
        super().__init__()
        self._policy = policy
        self._policy_name = type(policy).__name__ if policy is not None else "uninitialized"

    def initialize(self, od_config) -> None:
        super().initialize(od_config)
        if self._policy is None:
            self._policy = build_request_selection_policy(od_config.instance_scheduler_policy)
        self._policy.initialize(od_config)
        configured_policy = getattr(od_config, "instance_scheduler_policy", None)
        if isinstance(configured_policy, str) and configured_policy.strip():
            self._policy_name = normalize_request_selection_policy_name(configured_policy)
        else:
            self._policy_name = type(self._policy).__name__

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
        assert self._policy is not None
        self._policy.on_request_arrival(sched_req_id, state, self._execution_states[sched_req_id])
        self._waiting.append(sched_req_id)
        logger.debug("StepLevelRequestScheduler add_request: %s (waiting=%d)", sched_req_id, len(self._waiting))
        return sched_req_id

    def _estimated_total_runtime_s(self, state: DiffusionRequestState, exec_state) -> float | None:
        if exec_state.estimated_runtime_s is not None:
            return max(float(exec_state.estimated_runtime_s), 1e-9)
        extra_args = getattr(state.req.sampling_params, "extra_args", {}) or {}
        estimated_cost_s = _safe_optional_float(extra_args.get("estimated_cost_s"))
        if estimated_cost_s is None:
            return None
        return max(estimated_cost_s, 1e-9)

    def _remaining_estimated_runtime_s(self, state: DiffusionRequestState, exec_state) -> float | None:
        estimated_total_runtime_s = self._estimated_total_runtime_s(state, exec_state)
        if estimated_total_runtime_s is None:
            return None
        total_steps = _request_total_steps(state.req)
        remaining_steps = max(total_steps - exec_state.executed_steps, 0)
        if remaining_steps <= 0:
            return 0.0
        return max(estimated_total_runtime_s * (float(remaining_steps) / float(total_steps)), 0.0)

    def _format_runtime_s(self, runtime_s: float | None) -> str:
        if runtime_s is None:
            return "?"
        return f"{runtime_s:.3f}s"

    def _format_request_summary(self, sched_req_id: str) -> str:
        state = self._request_states[sched_req_id]
        exec_state = self._execution_states[sched_req_id]
        total_steps = _request_total_steps(state.req)
        remaining_est_s = self._remaining_estimated_runtime_s(state, exec_state)
        return (
            f"{sched_req_id}(step={exec_state.executed_steps}/{total_steps},"
            f"dispatch={exec_state.dispatch_epoch},rem_est={self._format_runtime_s(remaining_est_s)})"
        )

    def _log_schedule_decision(
        self,
        step_id: int,
        ordered_waiting: list[str],
        sched_req_id: str,
        *,
        was_new_request: bool,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        state = self._request_states[sched_req_id]
        exec_state = self._execution_states[sched_req_id]
        queue_head = ", ".join(self._format_request_summary(req_id) for req_id in ordered_waiting[:_WAITING_PREVIEW_LIMIT])
        if len(ordered_waiting) > _WAITING_PREVIEW_LIMIT:
            queue_head = f"{queue_head}, +{len(ordered_waiting) - _WAITING_PREVIEW_LIMIT} more"
        logger.info(
            "[StepSchedule] turn=%d policy=%s selected=%s kind=%s dispatch_epoch=%d progress=%d/%d waiting_before=%d queue_head=[%s]",
            step_id,
            self._policy_name,
            sched_req_id,
            "new" if was_new_request else "resumed",
            exec_state.dispatch_epoch,
            exec_state.executed_steps,
            _request_total_steps(state.req),
            len(ordered_waiting),
            queue_head,
        )

    def _log_step_outcome(
        self,
        step_id: int,
        sched_req_id: str,
        state: DiffusionRequestState,
        exec_state,
        *,
        previous_executed_steps: int,
        output: RunnerOutput,
        step_latency_s: float | None,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        latency_ms = "?"
        if step_latency_s is not None:
            latency_ms = f"{step_latency_s * 1000.0:.2f}"
        logger.info(
            "[StepComplete] turn=%d req=%s progress=%d->%d/%d finished=%s status=%s latency_ms=%s cumulative_execute_ms=%.2f rem_est=%s waiting_after=%d running_after=%d",
            step_id,
            sched_req_id,
            previous_executed_steps,
            exec_state.executed_steps,
            _request_total_steps(state.req),
            output.finished,
            state.status.name,
            latency_ms,
            exec_state.cumulative_execute_time_s * 1000.0,
            self._format_runtime_s(self._remaining_estimated_runtime_s(state, exec_state)),
            len(self._waiting),
            len(self._running),
        )

    def schedule(self) -> DiffusionSchedulerOutput:
        step_id = self._step_id
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
            exec_state.last_dispatch_time = monotonic()
            self._policy.on_request_scheduled(sched_req_id, state, exec_state)
            self._log_schedule_decision(step_id, ordered_waiting, sched_req_id, was_new_request=was_new_request)

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
        previous_executed_steps = exec_state.executed_steps
        if output.step_index is not None:
            exec_state.executed_steps = max(exec_state.executed_steps, output.step_index)
        step_latency_s = None
        if exec_state.last_dispatch_time is not None:
            step_latency_s = max(monotonic() - exec_state.last_dispatch_time, 0.0)
            exec_state.cumulative_execute_time_s += step_latency_s
        exec_state.last_dispatch_time = None
        assert self._policy is not None
        self._policy.on_step_complete(
            sched_req_id,
            state,
            exec_state,
            output,
            previous_executed_steps,
            step_latency_s,
        )

        if self.is_abort_pending(sched_req_id):
            finished_req_ids |= self._finish_requests(
                {sched_req_id: DiffusionRequestStatus.FINISHED_ABORTED},
            )
            self._notify_finished_requests(finished_req_ids)
            self._log_step_outcome(
                sched_output.step_id,
                sched_req_id,
                state,
                exec_state,
                previous_executed_steps=previous_executed_steps,
                output=output,
                step_latency_s=step_latency_s,
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
            self._notify_finished_requests(finished_req_ids)
            self._log_step_outcome(
                sched_output.step_id,
                sched_req_id,
                state,
                exec_state,
                previous_executed_steps=previous_executed_steps,
                output=output,
                step_latency_s=step_latency_s,
            )
            return finished_req_ids

        state.status = DiffusionRequestStatus.PREEMPTED
        self._waiting.append(sched_req_id)
        self._log_step_outcome(
            sched_output.step_id,
            sched_req_id,
            state,
            exec_state,
            previous_executed_steps=previous_executed_steps,
            output=output,
            step_latency_s=step_latency_s,
        )
        return finished_req_ids

    def finish_requests(self, sched_req_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(sched_req_ids, str):
            sched_req_ids = [sched_req_ids]
        finished_req_ids = self._finish_requests({sched_req_id: status for sched_req_id in sched_req_ids})
        self._notify_finished_requests(finished_req_ids)

    def _notify_finished_requests(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids or self._policy is None:
            return
        finished_at = monotonic()
        for sched_req_id in finished_req_ids:
            state = self._request_states.get(sched_req_id)
            exec_state = self._execution_states.get(sched_req_id)
            if state is None or exec_state is None:
                continue
            self._policy.on_request_finished(
                sched_req_id,
                state,
                exec_state,
                state.status,
                finished_at,
            )
