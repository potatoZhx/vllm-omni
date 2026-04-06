# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from math import ceil, sqrt
from time import monotonic
from typing import TYPE_CHECKING

from vllm_omni.diffusion.sched.interface import (
    DiffusionExecutionState,
    DiffusionRequestState,
    DiffusionRequestStatus,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.global_scheduler.policies.runtime_estimator import RuntimeEstimator
from vllm_omni.global_scheduler.runtime_profile import load_runtime_profile
from vllm_omni.global_scheduler.types import RequestMeta

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

_SJF_AGING_DEFAULT_FACTOR = 1.0
_SJF_AGING_COST_REF_S = 12.0
_SJF_AGING_COST_WEIGHT_MAX = 4.0
_SJF_AGING_GUARDED_MIN_WAIT_S = 45.0
_SJF_AGING_GUARDED_MAX_WAIT_S = 120.0
_SJF_AGING_GUARDED_WAIT_COST_RATIO = 2.0
_SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO = 0.05
_SJF_AGING_GUARDED_TAIL_WINDOW_MAXLEN = 128
_SJF_AGING_GUARDED_TAIL_COST_SCALE = 1.5
_SJF_AGING_GUARDED_TAIL_DEFER_WAIT_MULTIPLIER = 1.0
_SJF_AGING_GUARDED_TAIL_DEFER_COST_MULTIPLIER = 1.5
_SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_WAIT_MULTIPLIER = 100.0
_SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_COST_MULTIPLIER = 100.0
_P95_FIRST_HISTORY_MAXLEN = 128
_P95_FIRST_MIN_HISTORY_FOR_QUANTILE = 20
_P95_FIRST_SERVICE_RATE_EMA_ALPHA = 0.1

_POLICY_ALIASES = {
    "sjf_aging_guard": "sjf_aging_guarded",
}
_SUPPORTED_STEP_LEVEL_SELECTION_POLICIES = (
    "fcfs",
    "sjf",
    "sjf_aging",
    "sjf_aging_guarded",
    "sjf_aging_guarded_tail",
    "p95-first",
)


def normalize_request_selection_policy_name(name: str) -> str:
    normalized = name.strip()
    return _POLICY_ALIASES.get(normalized, normalized)


def supported_step_level_selection_policies() -> tuple[str, ...]:
    return _SUPPORTED_STEP_LEVEL_SELECTION_POLICIES


class RequestSelectionPolicy:
    """Order waiting requests for one scheduling turn."""

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        del od_config

    def on_request_arrival(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> None:
        del sched_req_id, request_state, execution_state

    def on_request_scheduled(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> None:
        del sched_req_id, request_state, execution_state

    def on_step_complete(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        output: RunnerOutput,
        previous_executed_steps: int,
        step_latency_s: float | None,
    ) -> None:
        del sched_req_id, request_state, execution_state, output, previous_executed_steps, step_latency_s

    def on_request_finished(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        status: DiffusionRequestStatus,
        finished_at: float,
    ) -> None:
        del sched_req_id, request_state, execution_state, status, finished_at

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        """Return waiting scheduler request ids ordered by selection priority."""
        raise NotImplementedError


class _EstimatedRuntimePolicy(RequestSelectionPolicy):
    def __init__(self) -> None:
        self._runtime_estimator: RuntimeEstimator | None = None
        self._instance_type: str | None = None

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        profile_path = getattr(od_config, "instance_runtime_profile_path", None)
        profiling_data = load_runtime_profile(profile_path) if profile_path else None
        self._runtime_estimator = RuntimeEstimator(profiling_data=profiling_data)
        self._instance_type = getattr(od_config, "instance_runtime_profile_name", None)

    def on_request_arrival(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> None:
        del sched_req_id
        execution_state.estimated_runtime_s = self._estimate_total_runtime_s(request_state.req)

    def _estimate_total_runtime_s(self, request) -> float:
        extra_args = getattr(request.sampling_params, "extra_args", {}) or {}
        estimated_cost_s = _safe_optional_float(extra_args.get("estimated_cost_s"))
        if estimated_cost_s is not None:
            return max(estimated_cost_s, 1e-3)

        heuristic_estimate = self._heuristic_runtime_s(request)
        if self._runtime_estimator is None:
            return heuristic_estimate

        request_meta = RequestMeta(
            request_id=(request.request_ids[0] if request.request_ids else "<missing-request-id>"),
            width=_request_width(request),
            height=_request_height(request),
            num_frames=_request_num_frames(request),
            num_inference_steps=_request_total_steps(request),
        )
        return max(
            self._runtime_estimator.estimate_runtime_s(
                request_meta,
                ewma_fallback_s=heuristic_estimate,
                instance_type=self._instance_type,
            ),
            1e-3,
        )

    def _heuristic_runtime_s(self, request) -> float:
        area_scale = max(float(_request_width(request) * _request_height(request)) / float(1024 * 1024), 0.0)
        heuristic = (
            float(_request_total_steps(request) * _request_num_frames(request) * _request_num_outputs(request))
            * max(area_scale, 0.0625)
        )
        return max(heuristic, 1e-3)

    def _remaining_estimated_runtime_s(
        self,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> float:
        total_steps = _request_total_steps(request_state.req)
        remaining_steps = max(total_steps - execution_state.executed_steps, 1)
        total_estimated_runtime_s = execution_state.estimated_runtime_s
        if total_estimated_runtime_s is None:
            total_estimated_runtime_s = self._estimate_total_runtime_s(request_state.req)
            execution_state.estimated_runtime_s = total_estimated_runtime_s
        return max(total_estimated_runtime_s * (float(remaining_steps) / float(total_steps)), 1e-9)

    def _remaining_work_units(
        self,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> float:
        return _request_work_units(request_state.req, remaining_steps=max(_request_total_steps(request_state.req) - execution_state.executed_steps, 1))


class FCFSSelectionPolicy(RequestSelectionPolicy):
    """Preserve arrival order from the waiting deque."""

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        del request_states, execution_states
        return list(waiting)


class SJFSelectionPolicy(_EstimatedRuntimePolicy):
    """Order by estimated remaining runtime."""

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        order = {req_id: index for index, req_id in enumerate(waiting)}
        return sorted(
            waiting,
            key=lambda req_id: (
                self._remaining_estimated_runtime_s(request_states[req_id], execution_states[req_id]),
                order[req_id],
            ),
        )


class SJFAgingSelectionPolicy(_EstimatedRuntimePolicy):
    """SJF with wait-time aging to reduce starvation."""

    def _aging_cost_weight(self, estimated_cost_s: float) -> float:
        normalized_cost = max(float(estimated_cost_s), 1e-9) / _SJF_AGING_COST_REF_S
        return min(max(sqrt(normalized_cost), 1.0), _SJF_AGING_COST_WEIGHT_MAX)

    def _aged_cost_s(
        self,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        *,
        now: float,
    ) -> tuple[float, float]:
        estimated_cost_s = self._remaining_estimated_runtime_s(request_state, execution_state)
        age_s = _request_age_s(execution_state, now)
        cost_weight = self._aging_cost_weight(estimated_cost_s)
        aged_cost_s = estimated_cost_s / (1.0 + (_SJF_AGING_DEFAULT_FACTOR * cost_weight * age_s))
        return aged_cost_s, estimated_cost_s

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        now = monotonic()
        order = {req_id: index for index, req_id in enumerate(waiting)}
        return sorted(
            waiting,
            key=lambda req_id: (
                *self._aged_cost_s(request_states[req_id], execution_states[req_id], now=now),
                order[req_id],
            ),
        )


class SJFAgingGuardedSelectionPolicy(SJFAgingSelectionPolicy):
    """Aging SJF with a learned protected queue for old requests."""

    def __init__(self) -> None:
        super().__init__()
        self._wait_history_ms: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)

    def _learned_wait_guard_s(self) -> float:
        if not self._wait_history_ms:
            return _SJF_AGING_GUARDED_MIN_WAIT_S

        samples = sorted(self._wait_history_ms)
        if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
            learned_wait_s = float(samples[-1]) / 1000.0
        else:
            index = max(ceil(len(samples) * 0.95) - 1, 0)
            learned_wait_s = float(samples[index]) / 1000.0
        return min(max(learned_wait_s, _SJF_AGING_GUARDED_MIN_WAIT_S), _SJF_AGING_GUARDED_MAX_WAIT_S)

    def _protection_threshold_s(
        self,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> float:
        estimated_cost_s = self._remaining_estimated_runtime_s(request_state, execution_state)
        return max(self._learned_wait_guard_s(), _SJF_AGING_GUARDED_WAIT_COST_RATIO * estimated_cost_s)

    def on_request_finished(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        status: DiffusionRequestStatus,
        finished_at: float,
    ) -> None:
        del sched_req_id, request_state
        if status == DiffusionRequestStatus.FINISHED_ABORTED:
            return
        if execution_state.arrival_time is None or execution_state.cumulative_execute_time_s <= 0.0:
            return

        latency_ms = max((finished_at - execution_state.arrival_time) * 1000.0, 0.0)
        wait_ms = max(latency_ms - (execution_state.cumulative_execute_time_s * 1000.0), 0.0)
        self._wait_history_ms.append(wait_ms)

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        now = monotonic()
        aged_costs: dict[str, tuple[float, float]] = {}
        tail_protected: dict[str, bool] = {}
        for req_id in waiting:
            request_state = request_states[req_id]
            execution_state = execution_states[req_id]
            aged_costs[req_id] = self._aged_cost_s(request_state, execution_state, now=now)
            tail_protected[req_id] = _request_age_s(execution_state, now) >= self._protection_threshold_s(
                request_state,
                execution_state,
            )

        order = {req_id: index for index, req_id in enumerate(waiting)}
        return sorted(
            waiting,
            key=lambda req_id: (
                0 if tail_protected[req_id] else 1,
                execution_states[req_id].arrival_time if tail_protected[req_id] else aged_costs[req_id][0],
                aged_costs[req_id][1],
                order[req_id],
            ),
        )


class SJFAgingGuardedTailSelectionPolicy(SJFAgingGuardedSelectionPolicy):
    """Guarded SJF with a bounded tail-sink budget for super-heavy requests."""

    def __init__(self) -> None:
        super().__init__()
        self._arrived_request_ids: set[str] = set()
        self._deferred_request_ids: set[str] = set()
        self._active_sunk_request_ids: set[str] = set()
        self._arrival_window_ids: deque[str] = deque()
        self._arrival_window_id_set: set[str] = set()
        self._window_deferred_request_ids: set[str] = set()

    def on_request_arrival(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> None:
        super().on_request_arrival(sched_req_id, request_state, execution_state)
        if sched_req_id in self._arrived_request_ids:
            return

        self._arrived_request_ids.add(sched_req_id)
        self._arrival_window_ids.append(sched_req_id)
        self._arrival_window_id_set.add(sched_req_id)
        while len(self._arrival_window_ids) > _SJF_AGING_GUARDED_TAIL_WINDOW_MAXLEN:
            evicted_req_id = self._arrival_window_ids.popleft()
            self._arrival_window_id_set.discard(evicted_req_id)
            self._window_deferred_request_ids.discard(evicted_req_id)

    def on_request_finished(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        status: DiffusionRequestStatus,
        finished_at: float,
    ) -> None:
        super().on_request_finished(sched_req_id, request_state, execution_state, status, finished_at)
        self._active_sunk_request_ids.discard(sched_req_id)

    def _budget_status(self) -> dict[str, int]:
        global_arrived_unique = len(self._arrived_request_ids)
        global_limit = int(global_arrived_unique * _SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO)
        window_arrived_unique = len(self._arrival_window_id_set)
        window_limit = int(window_arrived_unique * _SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO)
        return {
            "global_budget_remaining": max(global_limit - len(self._deferred_request_ids), 0),
            "window_budget_remaining": max(window_limit - len(self._window_deferred_request_ids), 0),
        }

    def _mark_deferred(self, sched_req_id: str) -> None:
        self._deferred_request_ids.add(sched_req_id)
        self._active_sunk_request_ids.add(sched_req_id)
        if sched_req_id in self._arrival_window_id_set:
            self._window_deferred_request_ids.add(sched_req_id)

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        now = monotonic()
        queue_costs = sorted(
            self._remaining_estimated_runtime_s(request_states[req_id], execution_states[req_id]) for req_id in waiting
        )
        if not queue_costs:
            return []

        mid = len(queue_costs) // 2
        queue_median_cost_s = (
            queue_costs[mid]
            if len(queue_costs) % 2 == 1
            else (queue_costs[mid - 1] + queue_costs[mid]) / 2.0
        )
        queue_p75_cost_s = queue_costs[max(int(len(queue_costs) * 0.75) - 1, 0)]
        p95_relief_min_count = max(ceil(len(waiting) * 0.05), 1)

        aged_costs: dict[str, tuple[float, float]] = {}
        protected: dict[str, bool] = {}
        sunk: dict[str, bool] = {}
        hard_escape: dict[str, bool] = {}
        sink_threshold: dict[str, float] = {}

        for req_id in waiting:
            request_state = request_states[req_id]
            execution_state = execution_states[req_id]
            aged_cost_s, estimated_cost_s = self._aged_cost_s(request_state, execution_state, now=now)
            aged_costs[req_id] = (aged_cost_s, estimated_cost_s)
            protection_threshold_s = self._protection_threshold_s(request_state, execution_state)
            age_s = _request_age_s(execution_state, now)
            protected[req_id] = age_s >= protection_threshold_s
            sink_threshold[req_id] = max(
                _SJF_AGING_GUARDED_TAIL_DEFER_WAIT_MULTIPLIER * protection_threshold_s,
                _SJF_AGING_GUARDED_TAIL_DEFER_COST_MULTIPLIER * estimated_cost_s,
            )
            hard_escape_threshold_s = max(
                _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_WAIT_MULTIPLIER * self._learned_wait_guard_s(),
                _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_COST_MULTIPLIER * estimated_cost_s,
            )
            hard_escape[req_id] = age_s >= hard_escape_threshold_s
            if req_id in self._active_sunk_request_ids and hard_escape[req_id]:
                self._active_sunk_request_ids.discard(req_id)
            sunk[req_id] = req_id in self._active_sunk_request_ids

        budget_status = self._budget_status()
        sink_budget_limit = min(
            1 if len(waiting) > 1 else 0,
            budget_status["global_budget_remaining"],
            budget_status["window_budget_remaining"],
        )

        if sink_budget_limit > 0:
            candidates: list[tuple[float, float, float, int, str]] = []
            for req_id in waiting:
                estimated_cost_s = aged_costs[req_id][1]
                age_s = _request_age_s(execution_states[req_id], now)
                large_request_threshold_s = max(
                    _SJF_AGING_GUARDED_TAIL_COST_SCALE * queue_median_cost_s,
                    queue_p75_cost_s,
                )
                if (
                    not protected[req_id]
                    or sunk[req_id]
                    or hard_escape[req_id]
                    or estimated_cost_s < large_request_threshold_s
                    or len(waiting) < 3
                ):
                    continue

                lighter_costs = [
                    aged_costs[other_req_id][1]
                    for other_req_id in waiting
                    if other_req_id != req_id and aged_costs[other_req_id][1] < estimated_cost_s
                ]
                lighter_request_count = len(lighter_costs)
                if lighter_request_count == 0 or lighter_request_count < p95_relief_min_count:
                    continue

                lighter_mean_cost_s = sum(lighter_costs) / float(lighter_request_count)
                defer_relief_score = float(lighter_request_count) * max(estimated_cost_s - lighter_mean_cost_s, 0.0)
                defer_harm_score = estimated_cost_s * max(
                    age_s
                    / max(
                        max(
                            _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_WAIT_MULTIPLIER * self._learned_wait_guard_s(),
                            _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_COST_MULTIPLIER * estimated_cost_s,
                        ),
                        1e-9,
                    ),
                    1.0,
                )
                if age_s >= sink_threshold[req_id] and defer_relief_score > defer_harm_score:
                    candidates.append(
                        (
                            -defer_relief_score,
                            -float(lighter_request_count),
                            -age_s,
                            waiting.index(req_id),
                            req_id,
                        )
                    )

            if candidates:
                _, _, _, _, sunk_req_id = min(candidates)
                self._mark_deferred(sunk_req_id)
                sunk[sunk_req_id] = True

        order = {req_id: index for index, req_id in enumerate(waiting)}
        return sorted(
            waiting,
            key=lambda req_id: (
                3 if sunk[req_id] else 0 if hard_escape[req_id] else 1 if protected[req_id] else 2,
                execution_states[req_id].arrival_time
                if (sunk[req_id] or hard_escape[req_id] or protected[req_id])
                else aged_costs[req_id][0],
                aged_costs[req_id][1],
                order[req_id],
            ),
        )


class P95FirstSelectionPolicy(_EstimatedRuntimePolicy):
    """Normalized tail-pressure ranking learned from observed step runtime."""

    def __init__(self) -> None:
        super().__init__()
        self._slowdown_history: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._observed_service_ms_per_work_unit: float | None = None

    def _learned_slowdown_p95(self) -> float:
        if not self._slowdown_history:
            return 1.0

        samples = sorted(self._slowdown_history)
        if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
            return max(float(samples[-1]), 1.0)
        index = max(ceil(len(samples) * 0.95) - 1, 0)
        return max(float(samples[index]), 1.0)

    def on_step_complete(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        output: RunnerOutput,
        previous_executed_steps: int,
        step_latency_s: float | None,
    ) -> None:
        del sched_req_id, output
        if step_latency_s is None or step_latency_s <= 0.0:
            return

        completed_steps = max(execution_state.executed_steps - previous_executed_steps, 0)
        if completed_steps <= 0:
            return

        work_units = _request_work_units(request_state.req, remaining_steps=completed_steps)
        if work_units <= 0.0:
            return

        observed_ms_per_work_unit = (step_latency_s * 1000.0) / max(work_units, 1e-9)
        if self._observed_service_ms_per_work_unit is None:
            self._observed_service_ms_per_work_unit = observed_ms_per_work_unit
            return

        alpha = _P95_FIRST_SERVICE_RATE_EMA_ALPHA
        self._observed_service_ms_per_work_unit = (
            (1.0 - alpha) * self._observed_service_ms_per_work_unit
            + alpha * observed_ms_per_work_unit
        )

    def on_request_finished(
        self,
        sched_req_id: str,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
        status: DiffusionRequestStatus,
        finished_at: float,
    ) -> None:
        del sched_req_id, request_state
        if status == DiffusionRequestStatus.FINISHED_ABORTED:
            return
        if execution_state.arrival_time is None or execution_state.cumulative_execute_time_s <= 0.0:
            return

        latency_ms = max((finished_at - execution_state.arrival_time) * 1000.0, 0.0)
        slowdown = max(latency_ms / max(execution_state.cumulative_execute_time_s * 1000.0, 1e-9), 1.0)
        self._slowdown_history.append(slowdown)

    def _estimated_service_ms(
        self,
        request_state: DiffusionRequestState,
        execution_state: DiffusionExecutionState,
    ) -> float:
        if self._observed_service_ms_per_work_unit is None:
            return max(self._remaining_estimated_runtime_s(request_state, execution_state) * 1000.0, 1e-9)
        return max(
            self._observed_service_ms_per_work_unit * self._remaining_work_units(request_state, execution_state),
            1e-9,
        )

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        now = monotonic()
        learned_slowdown_p95 = self._learned_slowdown_p95()
        waiting_order = {req_id: index for index, req_id in enumerate(waiting)}
        remaining = list(waiting)
        ordered: list[str] = []
        cursor_ms = 0.0

        while remaining:
            best_req_id: str | None = None
            best_key: tuple[float, float, int] | None = None
            for req_id in remaining:
                request_state = request_states[req_id]
                execution_state = execution_states[req_id]
                estimated_service_ms = self._estimated_service_ms(request_state, execution_state)
                age_s = _request_age_s(execution_state, now)
                predicted_finish_latency_ms = max((age_s * 1000.0) + cursor_ms + estimated_service_ms, 0.0)
                target_latency_ms = max(learned_slowdown_p95 * estimated_service_ms, 1e-9)
                pressure_ratio = predicted_finish_latency_ms / target_latency_ms
                candidate_key = (-pressure_ratio, estimated_service_ms, waiting_order[req_id])
                if best_key is None or candidate_key < best_key:
                    best_req_id = req_id
                    best_key = candidate_key

            assert best_req_id is not None
            ordered.append(best_req_id)
            cursor_ms += self._estimated_service_ms(request_states[best_req_id], execution_states[best_req_id])
            remaining.remove(best_req_id)

        return ordered


def build_request_selection_policy(name: str) -> RequestSelectionPolicy:
    normalized_name = normalize_request_selection_policy_name(name)
    if normalized_name == "fcfs":
        return FCFSSelectionPolicy()
    if normalized_name == "sjf":
        return SJFSelectionPolicy()
    if normalized_name == "sjf_aging":
        return SJFAgingSelectionPolicy()
    if normalized_name == "sjf_aging_guarded":
        return SJFAgingGuardedSelectionPolicy()
    if normalized_name == "sjf_aging_guarded_tail":
        return SJFAgingGuardedTailSelectionPolicy()
    if normalized_name == "p95-first":
        return P95FirstSelectionPolicy()
    raise NotImplementedError(f"Unsupported diffusion step-level selection policy: {name!r}")


def _request_total_steps(request) -> int:
    return max(_safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)


def _request_width(request) -> int:
    sampling_params = request.sampling_params
    resolution = _safe_int(getattr(sampling_params, "resolution", 1024), 1024)
    width = _safe_int(getattr(sampling_params, "width", None), 0)
    return width or resolution


def _request_height(request) -> int:
    sampling_params = request.sampling_params
    resolution = _safe_int(getattr(sampling_params, "resolution", 1024), 1024)
    height = _safe_int(getattr(sampling_params, "height", None), 0)
    return height or resolution


def _request_num_frames(request) -> int:
    return max(_safe_int(getattr(request.sampling_params, "num_frames", 1), 1), 1)


def _request_num_outputs(request) -> int:
    return max(_safe_int(getattr(request.sampling_params, "num_outputs_per_prompt", 1), 1), 1)


def _request_work_units(request, *, remaining_steps: int) -> float:
    if remaining_steps <= 0:
        return 0.0
    area_scale = max(float(_request_width(request) * _request_height(request)) / float(1024 * 1024), 0.0)
    return max(
        float(remaining_steps * _request_num_frames(request) * _request_num_outputs(request)) * max(area_scale, 0.0625),
        1e-9,
    )


def _request_age_s(execution_state: DiffusionExecutionState, now: float) -> float:
    if execution_state.arrival_time is None:
        return 0.0
    return max(now - execution_state.arrival_time, 0.0)


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
