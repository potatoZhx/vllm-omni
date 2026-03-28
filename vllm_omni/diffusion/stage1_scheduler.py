# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from math import ceil, inf, sqrt
from typing import Any

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_profile import RuntimeProfileEstimator
from vllm_omni.diffusion.scheduler import Scheduler

logger = init_logger(__name__)

_DEADLINE_AWARE_POLICIES = {"slo_first", "slack_age", "slack_cost_age"}
_P95_FIRST_POLICY = "p95-first"
_P95_FIRST_DEADLINE_POLICY = "p95-first-deadline"
_P95_BUCKET_SJF_NORMALIZED_POLICY = "p95-bucket-sjf-normalized"
_P95_BUCKET_SJF_POLICY = "p95-bucket-sjf"
_SJF_AGING_POLICY = "sjf_aging"
_SJF_AGING_GUARDED_POLICY = "sjf_aging_guarded"
_SJF_AGING_GUARDED_TAIL_POLICY = "sjf_aging_guarded_tail"
_BYPASS_GUARD_SJF_POLICY = "bypass_guard_sjf"
_SIZE_BUCKET_SJF_AGING_POLICY = "size_bucket_sjf_aging"
_TYPE_FIFO_DEFER_BUDGET_POLICY = "type_fifo_defer_budget"
_SLACK_HYBRID_POLICY = "slack_hybrid"
_P95_FUSION_POLICY = "p95-fusion"
_P95_FIRST_HISTORY_MAXLEN = 128
_P95_FIRST_MIN_HISTORY_FOR_QUANTILE = 20
_P95_FIRST_SERVICE_RATE_EMA_ALPHA = 0.1
_SLACK_COST_ALPHA = 0.25
_SJF_AGING_DEFAULT_FACTOR = 1.0
_SJF_AGING_COST_REF_S = 12.0
_SJF_AGING_COST_WEIGHT_MAX = 4.0
_SJF_AGING_GUARDED_MIN_WAIT_S = 45.0
_SJF_AGING_GUARDED_MAX_WAIT_S = 120.0
_SJF_AGING_GUARDED_WAIT_COST_RATIO = 2.0
_SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO = 0.05
_SJF_AGING_GUARDED_TAIL_BOOTSTRAP_MIN_UNIQUE = 2
_SJF_AGING_GUARDED_TAIL_WINDOW_MAXLEN = _P95_FIRST_HISTORY_MAXLEN
_SJF_AGING_GUARDED_TAIL_COST_SCALE = 1.5
_SJF_AGING_GUARDED_TAIL_DEFER_WAIT_MULTIPLIER = 1.0
_SJF_AGING_GUARDED_TAIL_DEFER_COST_MULTIPLIER = 1.5
_SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_WAIT_MULTIPLIER = 2.0
_SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_COST_MULTIPLIER = 4.0
_BYPASS_GUARD_MIN_WAIT_S = 45.0
_BYPASS_GUARD_MAX_WAIT_S = 120.0
_BYPASS_GUARD_WAIT_COST_RATIO = 2.0
_TYPE_FIFO_DEFER_DEFAULT_RATIO = 0.05
_TYPE_FIFO_DEFER_MIN_BUDGET = 0
_TYPE_FIFO_DEFER_ADAPTIVE_MIN_QUEUE_DEPTH = 8
_TYPE_FIFO_DEFER_WINDOW_MAXLEN = _P95_FIRST_HISTORY_MAXLEN
_TYPE_FIFO_DEFER_BASE_MIN_WAIT_S = 15.0
_TYPE_FIFO_DEFER_DYNAMIC_FLOOR_COST_RATIO = 0.5
_TYPE_FIFO_DEFER_WAIT_COST_RATIO = 2.0
_TYPE_FIFO_DEFER_OVERSTARVED_WAIT_P95_MULTIPLIER = 2.0
_TYPE_FIFO_DEFER_OVERSTARVED_COST_MULTIPLIER = 3.0
_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS = (512, 768, 1024)
_SIZE_BUCKET_PROMOTION_WINDOW_S = 10.0
_SLACK_HYBRID_DEFAULT_PANIC_THRESHOLD = 1.0
_P95_FUSION_MAX_LANE_RATIO = 0.35
_NORMALIZED_P95_POLICIES = {
    _P95_FIRST_POLICY,
    _P95_FIRST_DEADLINE_POLICY,
    _P95_BUCKET_SJF_NORMALIZED_POLICY,
    _P95_FUSION_POLICY,
}
_LEARNED_P95_DEADLINE_POLICIES = _DEADLINE_AWARE_POLICIES | {_P95_FIRST_POLICY, _SLACK_HYBRID_POLICY}


@dataclass
class _QueuedRequest:
    request: OmniDiffusionRequest
    enqueue_time: float
    sequence_id: int
    schedule_metrics: dict[str, Any] = field(default_factory=dict)
    estimated_cost_s: float | None = None


@dataclass
class _WaitingPlan:
    ordered_queue: list[_QueuedRequest]
    on_time_queue: list[_QueuedRequest]
    best_effort_queue: list[_QueuedRequest]
    feasible_ids: set[int]
    completion_ts: dict[int, float]
    regret_drop_count: int
    dynamic_p95_ms: float | None = None
    learned_p95_ms: float | None = None
    backlog_adjusted_p95_ms: float | None = None
    uses_learned_deadline: bool = False


class Stage1Scheduler(Scheduler):
    """Stage-1 FCFS scheduler with queue observability and normalized failures."""

    def initialize(self, od_config: OmniDiffusionConfig):
        super().initialize(od_config)
        self._queue_cv = threading.Condition()
        self._waiting_queue: deque[_QueuedRequest] = deque()
        self._active_request: _QueuedRequest | None = None
        self._active_started_at: float | None = None
        self._enqueue_seq = 0
        self._aborted_request_ids: set[str] = set()
        self._runtime_estimator = RuntimeProfileEstimator.from_path(
            getattr(od_config, "instance_runtime_profile_path", None),
            instance_type=getattr(od_config, "instance_runtime_profile_name", None),
        )
        self._p95_first_latency_history_ms: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._p95_first_slowdown_history: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._sjf_aging_guarded_wait_history_ms: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._sjf_aging_guarded_tail_arrived_request_ids: set[str] = set()
        self._sjf_aging_guarded_tail_deferred_request_ids: set[str] = set()
        self._sjf_aging_guarded_tail_active_sunk_request_ids: set[str] = set()
        self._sjf_aging_guarded_tail_arrival_window_ids: deque[str] = deque()
        self._sjf_aging_guarded_tail_arrival_window_id_set: set[str] = set()
        self._sjf_aging_guarded_tail_window_deferred_request_ids: set[str] = set()
        self._bypass_guard_wait_history_ms: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._type_fifo_defer_wait_history_ms: deque[float] = deque(maxlen=_P95_FIRST_HISTORY_MAXLEN)
        self._type_fifo_defer_arrived_request_ids: set[str] = set()
        self._type_fifo_defer_deferred_request_ids: set[str] = set()
        self._type_fifo_defer_arrival_window_ids: deque[str] = deque()
        self._type_fifo_defer_arrival_window_id_set: set[str] = set()
        self._type_fifo_defer_window_deferred_request_ids: set[str] = set()
        self._p95_first_observed_service_ms_per_work_unit: float | None = None
        self._p95_first_cold_start_max_ms = 0.0
        self._p95_fusion_nonheavy_streak = 0
        self._p95_fusion_arrival_count = 0
        self._p95_fusion_borrowed_cap = 0

    @staticmethod
    def _request_label(request: OmniDiffusionRequest) -> str:
        request_ids = getattr(request, "request_ids", None) or []
        if request_ids:
            return ",".join(request_ids)
        return "<missing-request-id>"

    def _build_generate_rpc_request(self, request: OmniDiffusionRequest) -> dict:
        return {
            "type": "rpc",
            "method": "generate",
            "args": (request,),
            "kwargs": {},
            "output_rank": 0,
            "exec_all_ranks": True,
        }

    @staticmethod
    def _set_request_state(request: OmniDiffusionRequest, state: str) -> None:
        setattr(request, "request_state", state)

    @staticmethod
    def _request_ids(request: OmniDiffusionRequest) -> list[str]:
        return list(getattr(request, "request_ids", None) or [])

    @classmethod
    def _request_summary(cls, request: OmniDiffusionRequest) -> dict[str, int]:
        sampling_params = getattr(request, "sampling_params", None)
        resolution = cls._safe_int(getattr(sampling_params, "resolution", 1024), 1024)
        width = cls._safe_int(getattr(sampling_params, "width", None), 0) or resolution
        height = cls._safe_int(getattr(sampling_params, "height", None), 0) or resolution
        total_steps = max(cls._safe_int(getattr(sampling_params, "num_inference_steps", 1), 1), 1)
        executed_steps = max(cls._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        return {
            "width": width,
            "height": height,
            "resolution": resolution,
            "num_frames": max(cls._safe_int(getattr(sampling_params, "num_frames", 1), 1), 1),
            "total_steps": total_steps,
            "executed_steps": executed_steps,
            "remaining_steps": max(total_steps - executed_steps, 0),
        }

    @staticmethod
    def _safe_optional_float(value: Any) -> float | None:
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

    @classmethod
    def _request_time_summary(cls, request: OmniDiffusionRequest) -> dict[str, float | None]:
        return {
            "arrival_ts": cls._safe_optional_float(getattr(request, "arrival_time", None)),
            "first_enqueue_ts": cls._safe_optional_float(getattr(request, "first_enqueue_time", None)),
            "first_dispatch_ts": cls._safe_optional_float(getattr(request, "first_dispatch_time", None)),
            "last_dispatch_ts": cls._safe_optional_float(getattr(request, "last_dispatch_time", None)),
            "last_preempted_ts": cls._safe_optional_float(getattr(request, "last_preempted_time", None)),
            "completion_ts": cls._safe_optional_float(getattr(request, "completion_time", None)),
            "failure_ts": cls._safe_optional_float(getattr(request, "failure_time", None)),
            "aborted_ts": cls._safe_optional_float(getattr(request, "aborted_time", None)),
        }

    @classmethod
    def _request_elapsed_ms(cls, request: OmniDiffusionRequest, now: float | None = None) -> float | None:
        base_ts = cls._safe_optional_float(getattr(request, "arrival_time", None))
        if base_ts is None:
            base_ts = cls._safe_optional_float(getattr(request, "first_enqueue_time", None))
        if base_ts is None:
            return None
        current_ts = time.monotonic() if now is None else float(now)
        return max((current_ts - base_ts) * 1000.0, 0.0)

    @classmethod
    def _request_log_payload(cls, request: OmniDiffusionRequest) -> dict[str, Any]:
        payload = dict(cls._request_summary(request))
        payload.update(cls._request_time_summary(request))
        payload["dispatch_epoch"] = cls._safe_int(getattr(request, "dispatch_epoch", 0), 0)
        payload["chunk_budget_steps"] = cls._safe_optional_int(getattr(request, "max_steps_this_turn", None))
        return payload

    @classmethod
    def _log_request_event(cls, event: str, request: OmniDiffusionRequest, **extra_fields: Any) -> None:
        payload = cls._request_log_payload(request)
        payload.update(extra_fields)
        logger.info(
            "%s request_id=%s width=%s height=%s total_steps=%s executed_steps=%s remaining_steps=%s dispatch_epoch=%s chunk_budget_steps=%s arrival_ts=%s first_enqueue_ts=%s first_dispatch_ts=%s last_dispatch_ts=%s last_preempted_ts=%s completion_ts=%s failure_ts=%s aborted_ts=%s queue_len=%s queue_wait_ms=%s latency_ms=%s policy=%s",
            event,
            cls._request_label(request),
            payload.get("width"),
            payload.get("height"),
            payload.get("total_steps"),
            payload.get("executed_steps"),
            payload.get("remaining_steps"),
            payload.get("dispatch_epoch"),
            payload.get("chunk_budget_steps"),
            payload.get("arrival_ts"),
            payload.get("first_enqueue_ts"),
            payload.get("first_dispatch_ts"),
            payload.get("last_dispatch_ts"),
            payload.get("last_preempted_ts"),
            payload.get("completion_ts"),
            payload.get("failure_ts"),
            payload.get("aborted_ts"),
            payload.get("queue_len"),
            payload.get("queue_wait_ms"),
            payload.get("latency_ms"),
            payload.get("scheduler_policy"),
        )

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
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

    @classmethod
    def _safe_optional_int(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float, bool, str)):
            return cls._safe_int(value)
        return None

    def _is_request_aborted(self, request: OmniDiffusionRequest) -> bool:
        request_ids = self._request_ids(request)
        return any(request_id in self._aborted_request_ids for request_id in request_ids)

    def _find_waiting_request_locked(self, request: OmniDiffusionRequest) -> _QueuedRequest | None:
        request_ids = set(self._request_ids(request))
        for queued_request in self._waiting_queue:
            if queued_request.request is request:
                return queued_request
            queued_request_ids = set(self._request_ids(queued_request.request))
            if request_ids and queued_request_ids and request_ids & queued_request_ids:
                return queued_request
        return None

    def _policy_name(self) -> str:
        return getattr(self.od_config, "instance_scheduler_policy", "fcfs")

    def _estimate_cost_seconds(self, request: OmniDiffusionRequest) -> float:
        sampling_params = request.sampling_params
        extra_args = getattr(sampling_params, "extra_args", {}) or {}
        total_steps = max(self._safe_int(getattr(sampling_params, "num_inference_steps", 1), 1), 1)
        executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        num_steps = max(total_steps - executed_steps, 1)
        if extra_args.get("estimated_cost_s") is not None:
            total_cost_s = max(float(extra_args["estimated_cost_s"]), 0.0)
            return max(total_cost_s * (float(num_steps) / float(total_steps)), 0.0)

        num_outputs = max(self._safe_int(getattr(sampling_params, "num_outputs_per_prompt", 1), 1), 1)
        num_frames = max(self._safe_int(getattr(sampling_params, "num_frames", 1), 1), 1)
        width = getattr(sampling_params, "width", None)
        height = getattr(sampling_params, "height", None)
        resolution = getattr(sampling_params, "resolution", None) or 1024
        if width and height:
            area_scale = max((float(width) * float(height)) / float(1024 * 1024), 0.0)
        else:
            area_scale = max((float(resolution) * float(resolution)) / float(1024 * 1024), 0.0)
        heuristic_estimate = max(float(num_steps * num_frames) * max(area_scale, 0.0625), 0.001)

        request_width = int(width or resolution)
        request_height = int(height or resolution)
        task_type = "video" if num_frames > 1 else "image"
        profiled_estimate = self._runtime_estimator.estimate_runtime_s(
            task_type=task_type,
            width=request_width,
            height=request_height,
            num_frames=num_frames,
            steps=num_steps,
            fallback_s=heuristic_estimate,
        )
        return max(profiled_estimate * float(num_outputs), 0.001)

    def _queued_cost_seconds(self, queued_request: _QueuedRequest) -> float:
        if queued_request.estimated_cost_s is None:
            queued_request.estimated_cost_s = self._estimate_cost_seconds(queued_request.request)
        return queued_request.estimated_cost_s

    def _active_total_remaining_cost_seconds(self, now: float) -> float:
        if self._active_request is None or self._active_started_at is None:
            return 0.0
        elapsed = max(now - self._active_started_at, 0.0)
        return max(self._queued_cost_seconds(self._active_request) - elapsed, 0.0)

    def _active_chunk_remaining_cost_seconds(self, now: float) -> float:
        if self._active_request is None or self._active_started_at is None:
            return 0.0

        request = self._active_request.request
        total_remaining_cost_s = self._queued_cost_seconds(self._active_request)
        elapsed = max(now - self._active_started_at, 0.0)
        total_steps = max(self._safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)
        executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        remaining_steps_at_dispatch = max(total_steps - executed_steps, 1)
        chunk_steps_this_turn = self._safe_optional_int(getattr(request, "max_steps_this_turn", None))
        if chunk_steps_this_turn is None or chunk_steps_this_turn <= 0:
            chunk_steps_this_turn = remaining_steps_at_dispatch
        else:
            chunk_steps_this_turn = min(max(chunk_steps_this_turn, 1), remaining_steps_at_dispatch)

        estimated_chunk_cost_s = total_remaining_cost_s * (
            float(chunk_steps_this_turn) / float(remaining_steps_at_dispatch)
        )
        return max(estimated_chunk_cost_s - elapsed, 0.0)

    def _request_work_units(
        self,
        request: OmniDiffusionRequest,
        *,
        step_count: int | None = None,
        total: bool = False,
    ) -> float:
        sampling_params = request.sampling_params
        total_steps = max(self._safe_int(getattr(sampling_params, "num_inference_steps", 1), 1), 1)
        if step_count is None:
            if total:
                step_count = total_steps
            else:
                executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
                step_count = max(total_steps - executed_steps, 0)
        else:
            step_count = max(self._safe_int(step_count, 0), 0)

        if step_count <= 0:
            return 0.0

        request_summary = self._request_summary(request)
        area_scale = max(float(request_summary["width"] * request_summary["height"]) / float(1024 * 1024), 0.0)
        num_frames = max(self._safe_int(getattr(sampling_params, "num_frames", 1), 1), 1)
        num_outputs = max(self._safe_int(getattr(sampling_params, "num_outputs_per_prompt", 1), 1), 1)
        return max(float(step_count * num_frames * num_outputs) * max(area_scale, 0.0625), 1e-9)

    def _chunk_work_units_this_turn(self, queued_request: _QueuedRequest) -> float:
        request = queued_request.request
        total_steps = max(self._safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)
        executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        remaining_steps_at_dispatch = max(total_steps - executed_steps, 1)
        chunk_steps_this_turn = self._safe_optional_int(getattr(request, "max_steps_this_turn", None))
        if chunk_steps_this_turn is None or chunk_steps_this_turn <= 0:
            chunk_steps_this_turn = remaining_steps_at_dispatch
        else:
            chunk_steps_this_turn = min(max(chunk_steps_this_turn, 1), remaining_steps_at_dispatch)
        return self._request_work_units(request, step_count=chunk_steps_this_turn)

    def _completed_chunk_work_units(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
        previous_executed_steps: int,
    ) -> float:
        if getattr(output, "error", None):
            return 0.0

        total_steps = max(self._safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)
        metrics = dict(getattr(output, "metrics", {}) or {})
        if "executed_steps" in metrics:
            executed_after = self._safe_int(metrics["executed_steps"], previous_executed_steps)
        elif getattr(output, "finished", True):
            executed_after = total_steps
        else:
            executed_after = previous_executed_steps

        executed_after = min(max(executed_after, 0), total_steps)
        delta_steps = max(executed_after - previous_executed_steps, 0)
        return self._request_work_units(request, step_count=delta_steps)

    def _record_p95_first_cost_observation(self, queued_request: _QueuedRequest) -> None:
        observed_ms = max(self._queued_cost_seconds(queued_request) * 1000.0, 0.0)
        self._p95_first_cold_start_max_ms = max(self._p95_first_cold_start_max_ms, observed_ms)

    def _record_p95_first_execute_sample(self, execute_latency_ms: float | None, work_units: float) -> None:
        if execute_latency_ms is None or work_units <= 0.0:
            return

        observed_service_ms_per_work_unit = max(float(execute_latency_ms), 0.0) / max(work_units, 1e-9)
        if observed_service_ms_per_work_unit <= 0.0:
            return

        if self._p95_first_observed_service_ms_per_work_unit is None:
            self._p95_first_observed_service_ms_per_work_unit = observed_service_ms_per_work_unit
            return

        alpha = _P95_FIRST_SERVICE_RATE_EMA_ALPHA
        self._p95_first_observed_service_ms_per_work_unit = (
            (1.0 - alpha) * self._p95_first_observed_service_ms_per_work_unit
            + alpha * observed_service_ms_per_work_unit
        )

    def _record_p95_first_latency_ms(
        self,
        latency_ms: float | None,
        *,
        request: OmniDiffusionRequest | None = None,
    ) -> None:
        if latency_ms is None:
            return
        value = max(float(latency_ms), 0.0)
        self._p95_first_latency_history_ms.append(value)
        self._p95_first_cold_start_max_ms = max(self._p95_first_cold_start_max_ms, value)

        if request is None or self._policy_name() not in _NORMALIZED_P95_POLICIES:
            return

        total_execute_ms = self._safe_optional_float(getattr(request, "_p95_first_cumulative_execute_ms", None))
        if total_execute_ms is None or total_execute_ms <= 0.0:
            return
        slowdown = max(value / max(total_execute_ms, 1e-9), 1.0)
        self._p95_first_slowdown_history.append(slowdown)

    def _learned_p95_ms(self) -> float:
        if self._p95_first_latency_history_ms:
            samples = sorted(self._p95_first_latency_history_ms)
            if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
                return float(samples[-1])
            index = max(ceil(len(samples) * 0.95) - 1, 0)
            return float(samples[index])

        base_ms = self._safe_optional_float(getattr(self.od_config, "instance_scheduler_p95_first_base_ms", None))
        min_ms = float(getattr(self.od_config, "instance_scheduler_p95_first_min_ms", 0.0) or 0.0)
        candidates = [min_ms, self._p95_first_cold_start_max_ms]
        if base_ms is not None:
            candidates.append(base_ms)
        return max(candidates)

    def _compute_local_backlog_seconds(self, waiting_requests: list[_QueuedRequest], now: float) -> float:
        waiting_cost_s = sum(self._queued_cost_seconds(queued_request) for queued_request in waiting_requests)
        return self._active_total_remaining_cost_seconds(now) + waiting_cost_s

    def _compute_dynamic_p95_ms(self, waiting_requests: list[_QueuedRequest], now: float) -> tuple[float, float, float, float]:
        min_ms = float(getattr(self.od_config, "instance_scheduler_p95_first_min_ms", 0.0) or 0.0)
        max_ms = self._safe_optional_float(getattr(self.od_config, "instance_scheduler_p95_first_max_ms", None))
        backlog_alpha = float(getattr(self.od_config, "instance_scheduler_p95_first_backlog_alpha", 1.0) or 0.0)
        base_ms = self._safe_optional_float(getattr(self.od_config, "instance_scheduler_p95_first_base_ms", None))
        if base_ms is None:
            base_ms = min_ms

        backlog_s = self._compute_local_backlog_seconds(waiting_requests, now)
        learned_p95_ms = max(self._learned_p95_ms(), min_ms)
        backlog_adjusted_p95_ms = max(base_ms, min_ms) + (backlog_alpha * backlog_s * 1000.0)
        dynamic_p95_ms = max(learned_p95_ms, backlog_adjusted_p95_ms)
        if max_ms is not None:
            dynamic_p95_ms = min(dynamic_p95_ms, max_ms)
        dynamic_p95_ms = max(dynamic_p95_ms, min_ms)
        return dynamic_p95_ms, backlog_s, learned_p95_ms, backlog_adjusted_p95_ms

    def _learned_p95_first_slowdown(self) -> float:
        if self._p95_first_slowdown_history:
            samples = sorted(self._p95_first_slowdown_history)
            if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
                return max(float(samples[-1]), 1.0)
            index = max(ceil(len(samples) * 0.95) - 1, 0)
            return max(float(samples[index]), 1.0)
        return 1.0

    def _p95_first_estimated_service_ms(self, queued_request: _QueuedRequest) -> float:
        if self._p95_first_observed_service_ms_per_work_unit is None:
            return max(self._queued_cost_seconds(queued_request) * 1000.0, 1e-9)
        work_units = self._request_work_units(queued_request.request)
        return max(self._p95_first_observed_service_ms_per_work_unit * max(work_units, 1e-9), 1e-9)

    def _p95_first_active_chunk_blocking_ms(self, now: float) -> float:
        if self._active_request is None or self._active_started_at is None:
            return 0.0

        if self._p95_first_observed_service_ms_per_work_unit is None:
            return max(self._active_chunk_remaining_cost_seconds(now) * 1000.0, 0.0)

        elapsed_ms = max((now - self._active_started_at) * 1000.0, 0.0)
        predicted_chunk_ms = self._p95_first_observed_service_ms_per_work_unit * max(
            self._chunk_work_units_this_turn(self._active_request),
            1e-9,
        )
        return max(predicted_chunk_ms - elapsed_ms, 0.0)

    def _p95_first_candidate_metrics(
        self,
        queued_request: _QueuedRequest,
        *,
        now: float,
        cursor_ms: float,
        learned_slowdown_p95: float,
        instance_backlog_total_s: float,
        active_chunk_blocking_ms: float,
        queue_rank: int,
    ) -> dict[str, Any]:
        remaining_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
        work_units = max(self._request_work_units(queued_request.request), 1e-9)
        estimated_service_ms = self._p95_first_estimated_service_ms(queued_request)
        age_s = self._request_age_seconds(queued_request, now)
        predicted_finish_latency_ms = max((age_s * 1000.0) + cursor_ms + estimated_service_ms, 0.0)
        target_latency_ms = max(learned_slowdown_p95 * estimated_service_ms, 1e-9)
        pressure_ratio = predicted_finish_latency_ms / target_latency_ms
        risk_ms = predicted_finish_latency_ms - target_latency_ms
        base_score = -pressure_ratio
        size_penalty = float(getattr(self.od_config, "instance_scheduler_p95_first_size_bias", 0.0) or 0.0) * (
            estimated_service_ms / 1000.0
        )
        aging_bonus = float(getattr(self.od_config, "instance_scheduler_p95_first_age_bias", 0.0) or 0.0) * age_s
        starvation_threshold_s = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_first_starvation_threshold_s", None)
        )
        starvation_boost = 0.0
        if starvation_threshold_s is not None and age_s >= starvation_threshold_s:
            starvation_boost = float(getattr(self.od_config, "instance_scheduler_p95_first_starvation_boost", 0.0) or 0.0)
        final_priority_score = base_score + size_penalty - aging_bonus - starvation_boost
        return {
            "scheduler_policy": _P95_FIRST_POLICY,
            "queue_reorder_count": 1,
            "learned_slowdown_p95": learned_slowdown_p95,
            "observed_service_rate_ms_per_work_unit": self._p95_first_observed_service_ms_per_work_unit,
            "service_rate_source": (
                "observed_runtime"
                if self._p95_first_observed_service_ms_per_work_unit is not None
                else "estimated_cost_fallback"
            ),
            "backlog_s_at_schedule": instance_backlog_total_s,
            "instance_backlog_total_s": instance_backlog_total_s,
            "active_chunk_blocking_ms": active_chunk_blocking_ms,
            "active_chunk_blocking_s": active_chunk_blocking_ms / 1000.0,
            "estimated_cost_s": remaining_cost_s,
            "work_units": work_units,
            "estimated_service_ms": estimated_service_ms,
            "age_s": age_s,
            "predicted_finish_latency_ms": predicted_finish_latency_ms,
            "target_latency_ms": target_latency_ms,
            "pressure_ratio": pressure_ratio,
            "risk_ms": risk_ms,
            "base_score": base_score,
            "size_penalty": size_penalty,
            "aging_bonus": aging_bonus,
            "starvation_boost": starvation_boost,
            "final_priority_score": final_priority_score,
            "queue_rank": queue_rank,
        }

    def _build_p95_first_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        learned_slowdown_p95 = self._learned_p95_first_slowdown()
        instance_backlog_total_s = self._compute_local_backlog_seconds(waiting_requests, now)
        active_chunk_blocking_ms = self._p95_first_active_chunk_blocking_ms(now)
        cursor_ms = active_chunk_blocking_ms
        remaining = list(waiting_requests)
        ordered_queue: list[_QueuedRequest] = []
        metrics_by_sequence: dict[int, dict[str, Any]] = {}

        while remaining:
            best_request: _QueuedRequest | None = None
            best_metrics: dict[str, Any] | None = None
            best_key: tuple[float, float, float, float, int] | None = None
            queue_rank = len(ordered_queue) + 1
            for queued_request in remaining:
                candidate_metrics = self._p95_first_candidate_metrics(
                    queued_request,
                    now=now,
                    cursor_ms=cursor_ms,
                    learned_slowdown_p95=learned_slowdown_p95,
                    instance_backlog_total_s=instance_backlog_total_s,
                    active_chunk_blocking_ms=active_chunk_blocking_ms,
                    queue_rank=queue_rank,
                )
                candidate_key = (
                    float(candidate_metrics["final_priority_score"]),
                    -float(candidate_metrics["pressure_ratio"]),
                    float(candidate_metrics["estimated_service_ms"]),
                    queued_request.enqueue_time,
                    queued_request.sequence_id,
                )
                if best_key is None or candidate_key < best_key:
                    best_request = queued_request
                    best_metrics = candidate_metrics
                    best_key = candidate_key

            assert best_request is not None
            assert best_metrics is not None
            ordered_queue.append(best_request)
            metrics_by_sequence[best_request.sequence_id] = best_metrics
            cursor_ms += float(best_metrics["estimated_service_ms"])
            remaining.remove(best_request)

        return ordered_queue, metrics_by_sequence

    def _p95_first_deadline_candidate_metrics(
        self,
        queued_request: _QueuedRequest,
        *,
        now: float,
        availability_ts: float,
        learned_slowdown_p95: float,
        instance_backlog_total_s: float,
        active_chunk_blocking_ms: float,
        queue_rank: int,
    ) -> dict[str, Any]:
        remaining_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
        work_units = max(self._request_work_units(queued_request.request), 1e-9)
        estimated_service_ms = self._p95_first_estimated_service_ms(queued_request)
        age_s = self._request_age_seconds(queued_request, now)
        target_latency_ms = max(learned_slowdown_p95 * estimated_service_ms, 1e-9)
        synthetic_deadline_ts = self._request_base_arrival_time(queued_request) + (target_latency_ms / 1000.0)
        slack_s = synthetic_deadline_ts - availability_ts - (estimated_service_ms / 1000.0)
        urgency_ms = (synthetic_deadline_ts - availability_ts) * 1000.0
        predicted_finish_latency_ms = max(((availability_ts - self._request_base_arrival_time(queued_request)) * 1000.0) + estimated_service_ms, 0.0)
        pressure_ratio = predicted_finish_latency_ms / target_latency_ms
        return {
            "scheduler_policy": _P95_FIRST_DEADLINE_POLICY,
            "queue_reorder_count": 1,
            "learned_slowdown_p95": learned_slowdown_p95,
            "observed_service_rate_ms_per_work_unit": self._p95_first_observed_service_ms_per_work_unit,
            "service_rate_source": (
                "observed_runtime"
                if self._p95_first_observed_service_ms_per_work_unit is not None
                else "estimated_cost_fallback"
            ),
            "backlog_s_at_schedule": instance_backlog_total_s,
            "instance_backlog_total_s": instance_backlog_total_s,
            "active_chunk_blocking_ms": active_chunk_blocking_ms,
            "active_chunk_blocking_s": active_chunk_blocking_ms / 1000.0,
            "availability_ts": availability_ts,
            "estimated_cost_s": remaining_cost_s,
            "work_units": work_units,
            "estimated_service_ms": estimated_service_ms,
            "age_s": age_s,
            "target_latency_ms": target_latency_ms,
            "synthetic_deadline_ts": synthetic_deadline_ts,
            "urgency_ms": urgency_ms,
            "slack_s": slack_s,
            "predicted_finish_latency_ms": predicted_finish_latency_ms,
            "pressure_ratio": pressure_ratio,
            "queue_rank": queue_rank,
        }

    def _build_p95_first_deadline_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        learned_slowdown_p95 = self._learned_p95_first_slowdown()
        availability_ts = self._availability_ts(now)
        instance_backlog_total_s = self._compute_local_backlog_seconds(waiting_requests, now)
        active_chunk_blocking_ms = self._p95_first_active_chunk_blocking_ms(now)
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        for queued_request in waiting_requests:
            metrics_by_sequence[queued_request.sequence_id] = self._p95_first_deadline_candidate_metrics(
                queued_request,
                now=now,
                availability_ts=availability_ts,
                learned_slowdown_p95=learned_slowdown_p95,
                instance_backlog_total_s=instance_backlog_total_s,
                active_chunk_blocking_ms=active_chunk_blocking_ms,
                queue_rank=0,
            )

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                float(metrics_by_sequence[queued.sequence_id]["slack_s"]),
                float(metrics_by_sequence[queued.sequence_id]["synthetic_deadline_ts"]),
                float(metrics_by_sequence[queued.sequence_id]["estimated_service_ms"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _p95_bucket_target_ms(self, queued_request: _QueuedRequest) -> tuple[float, float]:
        estimated_cost_ms = max(self._queued_cost_seconds(queued_request) * 1000.0, 0.0)
        history_p95_ms = max(
            self._learned_p95_ms(),
            float(getattr(self.od_config, "instance_scheduler_p95_first_min_ms", 0.0) or 0.0),
        )
        return max(history_p95_ms, estimated_cost_ms), history_p95_ms

    def _build_p95_bucket_sjf_normalized_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        bucket_count = max(self._safe_int(getattr(self.od_config, "instance_scheduler_p95_bucket_count", 4), 4), 1)
        min_window_ms = float(getattr(self.od_config, "instance_scheduler_p95_bucket_min_window_ms", 200.0) or 200.0)
        starvation_threshold_s = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_bucket_starvation_threshold_s", None)
        )
        promote_levels = max(
            self._safe_int(getattr(self.od_config, "instance_scheduler_p95_bucket_starvation_promote_levels", 1), 1),
            0,
        )
        availability_ts = self._availability_ts(now)
        learned_slowdown_p95 = self._learned_p95_first_slowdown()
        instance_backlog_total_s = self._compute_local_backlog_seconds(waiting_requests, now)
        active_chunk_blocking_ms = self._p95_first_active_chunk_blocking_ms(now)
        service_rate_source = (
            "observed_runtime"
            if self._p95_first_observed_service_ms_per_work_unit is not None
            else "estimated_cost_fallback"
        )
        precomputed: dict[int, dict[str, float]] = {}
        max_target_latency_ms = min_window_ms
        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            work_units = max(self._request_work_units(queued_request.request), 1e-9)
            estimated_service_ms = self._p95_first_estimated_service_ms(queued_request)
            target_latency_ms = max(learned_slowdown_p95 * estimated_service_ms, 1e-9)
            synthetic_deadline_ts = self._request_base_arrival_time(queued_request) + (target_latency_ms / 1000.0)
            urgency_ms = (synthetic_deadline_ts - availability_ts) * 1000.0
            age_s = self._request_age_seconds(queued_request, now)
            precomputed[queued_request.sequence_id] = {
                "estimated_cost_s": estimated_cost_s,
                "work_units": work_units,
                "estimated_service_ms": estimated_service_ms,
                "target_latency_ms": target_latency_ms,
                "synthetic_deadline_ts": synthetic_deadline_ts,
                "urgency_ms": urgency_ms,
                "age_s": age_s,
            }
            max_target_latency_ms = max(max_target_latency_ms, target_latency_ms)

        bucket_width_ms = max(max_target_latency_ms / float(bucket_count), 1e-9)
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        buckets: dict[int, list[_QueuedRequest]] = {bucket_id: [] for bucket_id in range(bucket_count)}
        for queued_request in waiting_requests:
            values = precomputed[queued_request.sequence_id]
            urgency_ms = float(values["urgency_ms"])
            if urgency_ms <= 0.0:
                raw_bucket_id = 0
            else:
                raw_bucket_id = min(int(urgency_ms / bucket_width_ms), bucket_count - 1)
            starvation_promoted = 0
            effective_bucket_id = raw_bucket_id
            age_s = float(values["age_s"])
            if starvation_threshold_s is not None and age_s >= starvation_threshold_s and promote_levels > 0:
                effective_bucket_id = max(raw_bucket_id - promote_levels, 0)
                starvation_promoted = 1
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _P95_BUCKET_SJF_NORMALIZED_POLICY,
                "queue_reorder_count": 1,
                "learned_slowdown_p95": learned_slowdown_p95,
                "observed_service_rate_ms_per_work_unit": self._p95_first_observed_service_ms_per_work_unit,
                "service_rate_source": service_rate_source,
                "backlog_s_at_schedule": instance_backlog_total_s,
                "instance_backlog_total_s": instance_backlog_total_s,
                "active_chunk_blocking_ms": active_chunk_blocking_ms,
                "active_chunk_blocking_s": active_chunk_blocking_ms / 1000.0,
                "estimated_cost_s": float(values["estimated_cost_s"]),
                "work_units": float(values["work_units"]),
                "estimated_service_ms": float(values["estimated_service_ms"]),
                "target_latency_ms": float(values["target_latency_ms"]),
                "synthetic_deadline_ts": float(values["synthetic_deadline_ts"]),
                "urgency_ms": urgency_ms,
                "raw_bucket_id": raw_bucket_id,
                "effective_bucket_id": effective_bucket_id,
                "bucket_width_ms": bucket_width_ms,
                "age_s": age_s,
                "starvation_promoted": starvation_promoted,
            }
            buckets[effective_bucket_id].append(queued_request)

        ordered_queue: list[_QueuedRequest] = []
        for bucket_id in range(bucket_count):
            bucket = sorted(
                buckets[bucket_id],
                key=lambda queued: (
                    float(metrics_by_sequence[queued.sequence_id]["estimated_service_ms"]),
                    queued.enqueue_time,
                    queued.sequence_id,
                ),
            )
            ordered_queue.extend(bucket)

        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _build_p95_bucket_sjf_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        bucket_count = max(self._safe_int(getattr(self.od_config, "instance_scheduler_p95_bucket_count", 4), 4), 1)
        min_window_ms = float(getattr(self.od_config, "instance_scheduler_p95_bucket_min_window_ms", 200.0) or 200.0)
        starvation_threshold_s = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_bucket_starvation_threshold_s", None)
        )
        promote_levels = max(
            self._safe_int(getattr(self.od_config, "instance_scheduler_p95_bucket_starvation_promote_levels", 1), 1),
            0,
        )
        availability_ts = self._availability_ts(now)
        history_p95_ms = max(
            self._learned_p95_ms(),
            float(getattr(self.od_config, "instance_scheduler_p95_first_min_ms", 0.0) or 0.0),
        )
        max_estimated_cost_ms = max(self._queued_cost_seconds(queued) * 1000.0 for queued in waiting_requests)
        anchor_window_ms = max(history_p95_ms, max_estimated_cost_ms, min_window_ms)
        bucket_width_ms = max(anchor_window_ms / float(bucket_count), 1e-9)
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        buckets: dict[int, list[_QueuedRequest]] = {bucket_id: [] for bucket_id in range(bucket_count)}

        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            estimated_cost_ms = estimated_cost_s * 1000.0
            target_p95_ms = max(history_p95_ms, estimated_cost_ms)
            age_s = self._request_age_seconds(queued_request, now)
            base_arrival_time = getattr(queued_request.request, "arrival_time", queued_request.enqueue_time)
            if not isinstance(base_arrival_time, (int, float)):
                base_arrival_time = queued_request.enqueue_time
            deadline_ts = float(base_arrival_time) + (target_p95_ms / 1000.0)
            urgency_ms = (deadline_ts - availability_ts) * 1000.0
            if urgency_ms <= 0:
                raw_bucket_id = 0
            else:
                raw_bucket_id = min(int(urgency_ms / bucket_width_ms), bucket_count - 1)
            starvation_promoted = 0
            effective_bucket_id = raw_bucket_id
            if starvation_threshold_s is not None and age_s >= starvation_threshold_s and promote_levels > 0:
                effective_bucket_id = max(raw_bucket_id - promote_levels, 0)
                starvation_promoted = 1
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _P95_BUCKET_SJF_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "history_p95_ms": history_p95_ms,
                "target_p95_ms": target_p95_ms,
                "deadline_ts": deadline_ts,
                "urgency_ms": urgency_ms,
                "raw_bucket_id": raw_bucket_id,
                "effective_bucket_id": effective_bucket_id,
                "bucket_width_ms": bucket_width_ms,
                "age_s": age_s,
                "starvation_promoted": starvation_promoted,
            }
            buckets[effective_bucket_id].append(queued_request)

        ordered_queue: list[_QueuedRequest] = []
        for bucket_id in range(bucket_count):
            bucket = sorted(
                buckets[bucket_id],
                key=lambda queued: (
                    self._queued_cost_seconds(queued),
                    queued.enqueue_time,
                    queued.sequence_id,
                ),
            )
            ordered_queue.extend(bucket)

        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _request_base_arrival_time(self, queued_request: _QueuedRequest) -> float:
        request = queued_request.request
        base_arrival_time = getattr(request, "arrival_time", None)
        if not isinstance(base_arrival_time, (int, float)):
            base_arrival_time = getattr(request, "first_enqueue_time", None)
        if not isinstance(base_arrival_time, (int, float)):
            base_arrival_time = queued_request.enqueue_time
        return float(base_arrival_time)

    def _explicit_deadline_ts(self, queued_request: _QueuedRequest) -> float | None:
        extra_args = getattr(queued_request.request.sampling_params, "extra_args", {}) or {}
        if extra_args.get("deadline_ts") is not None:
            return float(extra_args["deadline_ts"])

        slo_target_ms = extra_args.get("slo_target_ms")
        if slo_target_ms is None:
            slo_target_ms = extra_args.get("slo_ms")
        if slo_target_ms is None:
            slo_target_ms = getattr(self.od_config, "instance_scheduler_slo_target_ms", None)
        if slo_target_ms is None:
            return None

        floor_ms = float(getattr(self.od_config, "instance_scheduler_slo_floor_ms", 0.0) or 0.0)
        effective_target_ms = max(float(slo_target_ms), floor_ms)
        return self._request_base_arrival_time(queued_request) + (effective_target_ms / 1000.0)

    def _uses_learned_deadline(self, queued_request: _QueuedRequest) -> bool:
        return self._explicit_deadline_ts(queued_request) is None

    def _deadline_ts(self, queued_request: _QueuedRequest, dynamic_p95_ms: float | None = None) -> float:
        explicit_deadline_ts = self._explicit_deadline_ts(queued_request)
        if explicit_deadline_ts is not None:
            return explicit_deadline_ts
        if dynamic_p95_ms is None:
            return inf
        return self._request_base_arrival_time(queued_request) + (float(dynamic_p95_ms) / 1000.0)

    def _learned_deadline_context(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
        *,
        include_active: bool = False,
    ) -> tuple[float | None, float | None, float | None, bool]:
        needs_learned_deadline = any(self._uses_learned_deadline(queued_request) for queued_request in waiting_requests)
        if include_active and self._active_request is not None:
            needs_learned_deadline = needs_learned_deadline or self._uses_learned_deadline(self._active_request)
        if not needs_learned_deadline:
            return None, None, None, False

        dynamic_p95_ms, _backlog_s, learned_p95_ms, backlog_adjusted_p95_ms = self._compute_dynamic_p95_ms(
            waiting_requests, now
        )
        return dynamic_p95_ms, learned_p95_ms, backlog_adjusted_p95_ms, True

    def _request_age_seconds(self, queued_request: _QueuedRequest, now: float) -> float:
        return max(now - self._request_base_arrival_time(queued_request), 0.0)

    def _best_effort_score(self, queued_request: _QueuedRequest, now: float) -> float:
        aging_factor = float(getattr(self.od_config, "instance_scheduler_aging_factor", 0.0) or 0.0)
        age_s = self._request_age_seconds(queued_request, now)
        return self._queued_cost_seconds(queued_request) / (1.0 + aging_factor * age_s)

    def _on_time_score(
        self,
        queued_request: _QueuedRequest,
        now: float,
        dynamic_p95_ms: float | None = None,
    ) -> tuple[float, float, float, float, int]:
        remaining_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
        slack_s = self._deadline_ts(queued_request, dynamic_p95_ms) - now - remaining_cost_s
        age_s = self._request_age_seconds(queued_request, now)
        aging_factor = float(getattr(self.od_config, "instance_scheduler_aging_factor", 0.0) or 0.0)
        policy = self._policy_name()
        if policy == "slack_age":
            return (
                slack_s - (aging_factor * age_s),
                slack_s,
                remaining_cost_s,
                queued_request.enqueue_time,
                queued_request.sequence_id,
            )
        if policy == "slack_cost_age":
            return (
                slack_s + (_SLACK_COST_ALPHA * remaining_cost_s) - (aging_factor * age_s),
                slack_s,
                remaining_cost_s,
                queued_request.enqueue_time,
                queued_request.sequence_id,
            )
        return (
            slack_s / remaining_cost_s,
            slack_s,
            remaining_cost_s,
            queued_request.enqueue_time,
            queued_request.sequence_id,
        )

    def _availability_ts(self, now: float) -> float:
        if self._active_request is None or self._active_started_at is None:
            return now
        elapsed = max(now - self._active_started_at, 0.0)
        remaining = max(self._queued_cost_seconds(self._active_request) - elapsed, 0.0)
        return now + remaining

    def _build_waiting_plan(self, waiting_requests: list[_QueuedRequest], now: float) -> _WaitingPlan:
        if not waiting_requests:
            return _WaitingPlan(
                ordered_queue=[],
                on_time_queue=[],
                best_effort_queue=[],
                feasible_ids=set(),
                completion_ts={},
                regret_drop_count=0,
            )

        availability_ts = self._availability_ts(now)
        policy = self._policy_name()
        dynamic_p95_ms, learned_p95_ms, backlog_adjusted_p95_ms, uses_learned_deadline = self._learned_deadline_context(
            waiting_requests,
            now,
        )
        if policy in {"slack_age", "slack_cost_age"}:
            ordered_queue = sorted(
                waiting_requests,
                key=lambda queued: self._on_time_score(queued, now, dynamic_p95_ms),
            )
            completion_ts: dict[int, float] = {}
            cursor = availability_ts
            for queued in ordered_queue:
                cursor += self._queued_cost_seconds(queued)
                completion_ts[queued.sequence_id] = cursor
            return _WaitingPlan(
                ordered_queue=ordered_queue,
                on_time_queue=list(ordered_queue),
                best_effort_queue=[],
                feasible_ids={queued.sequence_id for queued in ordered_queue},
                completion_ts=completion_ts,
                regret_drop_count=0,
                dynamic_p95_ms=dynamic_p95_ms,
                learned_p95_ms=learned_p95_ms,
                backlog_adjusted_p95_ms=backlog_adjusted_p95_ms,
                uses_learned_deadline=uses_learned_deadline,
            )

        deadline_sorted = sorted(
            waiting_requests,
            key=lambda queued: (self._deadline_ts(queued, dynamic_p95_ms), queued.enqueue_time, queued.sequence_id),
        )

        prefix: list[_QueuedRequest] = []
        work = 0.0
        regret_drop_count = 0
        for queued in deadline_sorted:
            prefix.append(queued)
            work += self._queued_cost_seconds(queued)
            if availability_ts + work > self._deadline_ts(queued, dynamic_p95_ms):
                longest = max(
                    prefix,
                    key=lambda candidate: (self._queued_cost_seconds(candidate), candidate.sequence_id),
                )
                prefix.remove(longest)
                work -= self._queued_cost_seconds(longest)
                regret_drop_count += 1

        feasible_ids = {queued.sequence_id for queued in prefix}
        on_time_queue = sorted(
            prefix,
            key=lambda queued: self._on_time_score(queued, now, dynamic_p95_ms),
        )
        best_effort_queue = [queued for queued in waiting_requests if queued.sequence_id not in feasible_ids]
        best_effort_queue = sorted(
            best_effort_queue,
            key=lambda queued: (self._best_effort_score(queued, now), queued.enqueue_time, queued.sequence_id),
        )
        ordered_queue = on_time_queue + best_effort_queue

        completion_ts: dict[int, float] = {}
        cursor = availability_ts
        for queued in ordered_queue:
            cursor += self._queued_cost_seconds(queued)
            completion_ts[queued.sequence_id] = cursor

        return _WaitingPlan(
            ordered_queue=ordered_queue,
            on_time_queue=on_time_queue,
            best_effort_queue=best_effort_queue,
            feasible_ids=feasible_ids,
            completion_ts=completion_ts,
            regret_drop_count=regret_drop_count,
            dynamic_p95_ms=dynamic_p95_ms,
            learned_p95_ms=learned_p95_ms,
            backlog_adjusted_p95_ms=backlog_adjusted_p95_ms,
            uses_learned_deadline=uses_learned_deadline,
        )

    def _build_sjf_queue(self, waiting_requests: list[_QueuedRequest], now: float) -> list[_QueuedRequest]:
        del now
        return sorted(
            waiting_requests,
            key=lambda queued: (
                self._queued_cost_seconds(queued),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )

    def _effective_sjf_aging_factor(self) -> float:
        aging_factor = float(getattr(self.od_config, "instance_scheduler_aging_factor", 0.0) or 0.0)
        if aging_factor > 0.0:
            return aging_factor
        return _SJF_AGING_DEFAULT_FACTOR

    def _sjf_aging_cost_weight(self, estimated_cost_s: float) -> float:
        normalized_cost = max(float(estimated_cost_s), 1e-9) / _SJF_AGING_COST_REF_S
        return min(max(sqrt(normalized_cost), 1.0), _SJF_AGING_COST_WEIGHT_MAX)

    def _record_sjf_aging_guarded_wait_ms(
        self,
        latency_ms: float | None,
        *,
        request: OmniDiffusionRequest | None = None,
    ) -> None:
        if latency_ms is None or request is None:
            return
        total_execute_ms = self._safe_optional_float(getattr(request, "_sjf_aging_guarded_cumulative_execute_ms", None))
        if total_execute_ms is None or total_execute_ms <= 0.0:
            return
        wait_ms = max(float(latency_ms) - total_execute_ms, 0.0)
        self._sjf_aging_guarded_wait_history_ms.append(wait_ms)

    def _learned_sjf_aging_guarded_wait_s(self) -> float:
        if self._sjf_aging_guarded_wait_history_ms:
            samples = sorted(self._sjf_aging_guarded_wait_history_ms)
            if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
                learned_wait_s = float(samples[-1]) / 1000.0
            else:
                index = max(ceil(len(samples) * 0.95) - 1, 0)
                learned_wait_s = float(samples[index]) / 1000.0
            return min(max(learned_wait_s, _SJF_AGING_GUARDED_MIN_WAIT_S), _SJF_AGING_GUARDED_MAX_WAIT_S)
        return _SJF_AGING_GUARDED_MIN_WAIT_S

    def _sjf_aging_guard_threshold_s(self, estimated_cost_s: float) -> float:
        estimated_cost_s = max(float(estimated_cost_s), 1e-9)
        return max(self._learned_sjf_aging_guarded_wait_s(), _SJF_AGING_GUARDED_WAIT_COST_RATIO * estimated_cost_s)

    def _sjf_aging_guarded_tail_request_key(self, request: OmniDiffusionRequest) -> str:
        return self._request_label(request)

    def _track_sjf_aging_guarded_tail_arrival(self, request: OmniDiffusionRequest) -> None:
        request_key = self._sjf_aging_guarded_tail_request_key(request)
        if request_key in self._sjf_aging_guarded_tail_arrived_request_ids:
            return
        self._sjf_aging_guarded_tail_arrived_request_ids.add(request_key)
        self._sjf_aging_guarded_tail_arrival_window_ids.append(request_key)
        self._sjf_aging_guarded_tail_arrival_window_id_set.add(request_key)
        while len(self._sjf_aging_guarded_tail_arrival_window_ids) > _SJF_AGING_GUARDED_TAIL_WINDOW_MAXLEN:
            evicted_request_key = self._sjf_aging_guarded_tail_arrival_window_ids.popleft()
            self._sjf_aging_guarded_tail_arrival_window_id_set.discard(evicted_request_key)
            self._sjf_aging_guarded_tail_window_deferred_request_ids.discard(evicted_request_key)

    def _sjf_aging_guarded_tail_budget_status(self) -> dict[str, int]:
        global_arrived_unique = len(self._sjf_aging_guarded_tail_arrived_request_ids)
        global_limit = int(global_arrived_unique * _SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO)
        if global_arrived_unique >= _SJF_AGING_GUARDED_TAIL_BOOTSTRAP_MIN_UNIQUE:
            global_limit = max(global_limit, 1)
        global_used = len(self._sjf_aging_guarded_tail_deferred_request_ids)
        window_arrived_unique = len(self._sjf_aging_guarded_tail_arrival_window_id_set)
        window_limit = int(
            window_arrived_unique * _SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO
        )
        if window_arrived_unique >= _SJF_AGING_GUARDED_TAIL_BOOTSTRAP_MIN_UNIQUE:
            window_limit = max(window_limit, 1)
        window_used = len(self._sjf_aging_guarded_tail_window_deferred_request_ids)
        return {
            "global_arrived_unique": global_arrived_unique,
            "global_budget_limit": global_limit,
            "global_budget_used": global_used,
            "global_budget_remaining": max(global_limit - global_used, 0),
            "window_arrived_unique": window_arrived_unique,
            "window_budget_limit": window_limit,
            "window_budget_used": window_used,
            "window_budget_remaining": max(window_limit - window_used, 0),
        }

    def _mark_sjf_aging_guarded_tail_deferred(self, request: OmniDiffusionRequest) -> None:
        request_key = self._sjf_aging_guarded_tail_request_key(request)
        self._sjf_aging_guarded_tail_deferred_request_ids.add(request_key)
        self._sjf_aging_guarded_tail_active_sunk_request_ids.add(request_key)
        if request_key in self._sjf_aging_guarded_tail_arrival_window_id_set:
            self._sjf_aging_guarded_tail_window_deferred_request_ids.add(request_key)

    def _clear_sjf_aging_guarded_tail_sunk_state(self, request: OmniDiffusionRequest) -> None:
        request_key = self._sjf_aging_guarded_tail_request_key(request)
        self._sjf_aging_guarded_tail_active_sunk_request_ids.discard(request_key)
        setattr(request, "tail_sunk", False)

    def _record_bypass_guard_wait_ms(
        self,
        latency_ms: float | None,
        *,
        request: OmniDiffusionRequest | None = None,
    ) -> None:
        if latency_ms is None or request is None:
            return
        total_execute_ms = self._safe_optional_float(getattr(request, "_bypass_guard_cumulative_execute_ms", None))
        if total_execute_ms is None or total_execute_ms <= 0.0:
            return
        wait_ms = max(float(latency_ms) - total_execute_ms, 0.0)
        self._bypass_guard_wait_history_ms.append(wait_ms)

    def _learned_bypass_guard_wait_s(self) -> float:
        if self._bypass_guard_wait_history_ms:
            samples = sorted(self._bypass_guard_wait_history_ms)
            if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
                learned_wait_s = float(samples[-1]) / 1000.0
            else:
                index = max(ceil(len(samples) * 0.95) - 1, 0)
                learned_wait_s = float(samples[index]) / 1000.0
            return min(max(learned_wait_s, _BYPASS_GUARD_MIN_WAIT_S), _BYPASS_GUARD_MAX_WAIT_S)
        return _BYPASS_GUARD_MIN_WAIT_S

    def _bypass_guard_threshold_s(self, estimated_cost_s: float) -> float:
        estimated_cost_s = max(float(estimated_cost_s), 1e-9)
        return max(self._learned_bypass_guard_wait_s(), _BYPASS_GUARD_WAIT_COST_RATIO * estimated_cost_s)

    def _record_type_fifo_defer_wait_ms(
        self,
        latency_ms: float | None,
        *,
        request: OmniDiffusionRequest | None = None,
    ) -> None:
        if latency_ms is None or request is None:
            return
        total_execute_ms = self._safe_optional_float(getattr(request, "_type_fifo_defer_cumulative_execute_ms", None))
        if total_execute_ms is None or total_execute_ms <= 0.0:
            return
        wait_ms = max(float(latency_ms) - total_execute_ms, 0.0)
        self._type_fifo_defer_wait_history_ms.append(wait_ms)

    def _learned_type_fifo_defer_wait_s(self) -> float:
        if self._type_fifo_defer_wait_history_ms:
            samples = sorted(self._type_fifo_defer_wait_history_ms)
            if len(samples) < _P95_FIRST_MIN_HISTORY_FOR_QUANTILE:
                return float(samples[-1]) / 1000.0
            index = max(ceil(len(samples) * 0.95) - 1, 0)
            return float(samples[index]) / 1000.0
        return 0.0

    def _type_fifo_defer_wait_floor_s(
        self,
        waiting_requests: list[_QueuedRequest] | None = None,
    ) -> float:
        if waiting_requests:
            queue_costs = sorted(max(self._queued_cost_seconds(queued_request), 1e-9) for queued_request in waiting_requests)
            mid = len(queue_costs) // 2
            if len(queue_costs) % 2 == 1:
                median_cost_s = queue_costs[mid]
            else:
                median_cost_s = (queue_costs[mid - 1] + queue_costs[mid]) / 2.0
        else:
            median_cost_s = 0.0
        return max(_TYPE_FIFO_DEFER_BASE_MIN_WAIT_S, _TYPE_FIFO_DEFER_DYNAMIC_FLOOR_COST_RATIO * median_cost_s)

    def _type_fifo_defer_wait_p95_s(
        self,
        waiting_requests: list[_QueuedRequest] | None = None,
    ) -> float:
        return max(self._learned_type_fifo_defer_wait_s(), self._type_fifo_defer_wait_floor_s(waiting_requests))

    def _fixed_size_bucket_id(self, queued_request: _QueuedRequest) -> int:
        request_summary = self._request_summary(queued_request.request)
        max_dim = max(request_summary["width"], request_summary["height"])
        for bucket_id, threshold in enumerate(_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS):
            if max_dim <= threshold:
                return bucket_id
        return len(_FIXED_SIZE_BUCKET_MAX_DIM_THRESHOLDS)

    def _type_fifo_defer_budget_ratio(self) -> float:
        ratio = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_type_fifo_defer_budget_ratio", None)
        )
        if ratio is None:
            return _TYPE_FIFO_DEFER_DEFAULT_RATIO
        return min(max(float(ratio), 0.0), 1.0)

    def _type_fifo_defer_threshold_s(self, estimated_cost_s: float, *, wait_p95_s: float | None = None) -> float:
        estimated_cost_s = max(float(estimated_cost_s), 1e-9)
        effective_wait_p95_s = self._type_fifo_defer_wait_p95_s() if wait_p95_s is None else max(float(wait_p95_s), 0.0)
        return max(effective_wait_p95_s, _TYPE_FIFO_DEFER_WAIT_COST_RATIO * estimated_cost_s)

    def _type_fifo_defer_overstarved_threshold_s(self, estimated_cost_s: float, *, wait_p95_s: float | None = None) -> float:
        estimated_cost_s = max(float(estimated_cost_s), 1e-9)
        effective_wait_p95_s = self._type_fifo_defer_wait_p95_s() if wait_p95_s is None else max(float(wait_p95_s), 0.0)
        return max(
            _TYPE_FIFO_DEFER_OVERSTARVED_WAIT_P95_MULTIPLIER * effective_wait_p95_s,
            _TYPE_FIFO_DEFER_OVERSTARVED_COST_MULTIPLIER * estimated_cost_s,
        )

    def _type_fifo_defer_request_key(self, request: OmniDiffusionRequest) -> str:
        return self._request_label(request)

    def _track_type_fifo_defer_arrival(self, request: OmniDiffusionRequest) -> None:
        request_key = self._type_fifo_defer_request_key(request)
        if request_key in self._type_fifo_defer_arrived_request_ids:
            return
        self._type_fifo_defer_arrived_request_ids.add(request_key)
        self._type_fifo_defer_arrival_window_ids.append(request_key)
        self._type_fifo_defer_arrival_window_id_set.add(request_key)
        while len(self._type_fifo_defer_arrival_window_ids) > _TYPE_FIFO_DEFER_WINDOW_MAXLEN:
            evicted_request_key = self._type_fifo_defer_arrival_window_ids.popleft()
            self._type_fifo_defer_arrival_window_id_set.discard(evicted_request_key)
            self._type_fifo_defer_window_deferred_request_ids.discard(evicted_request_key)

    def _type_fifo_defer_budget_status(self, target_ratio: float) -> dict[str, int]:
        global_limit = int(len(self._type_fifo_defer_arrived_request_ids) * target_ratio)
        global_used = len(self._type_fifo_defer_deferred_request_ids)
        window_limit = int(len(self._type_fifo_defer_arrival_window_id_set) * target_ratio)
        window_used = len(self._type_fifo_defer_window_deferred_request_ids)
        return {
            "global_arrived_unique": len(self._type_fifo_defer_arrived_request_ids),
            "global_budget_limit": global_limit,
            "global_budget_used": global_used,
            "global_budget_remaining": max(global_limit - global_used, 0),
            "window_arrived_unique": len(self._type_fifo_defer_arrival_window_id_set),
            "window_budget_limit": window_limit,
            "window_budget_used": window_used,
            "window_budget_remaining": max(window_limit - window_used, 0),
        }

    def _mark_type_fifo_deferred(self, request: OmniDiffusionRequest) -> None:
        request_key = self._type_fifo_defer_request_key(request)
        self._type_fifo_defer_deferred_request_ids.add(request_key)
        if request_key in self._type_fifo_defer_arrival_window_id_set:
            self._type_fifo_defer_window_deferred_request_ids.add(request_key)

    def _type_fifo_defer_distribution_stats(
        self,
        grouped_requests: dict[tuple[int, int, int, int, int], list[_QueuedRequest]],
        type_costs: dict[tuple[int, int, int, int, int], float],
    ) -> dict[str, float]:
        total_requests = sum(len(grouped) for grouped in grouped_requests.values())
        if total_requests <= 0 or not type_costs:
            return {
                "queue_depth": 0.0,
                "type_count": 0.0,
                "dominant_type_share": 0.0,
                "heavy_type_share": 0.0,
                "lighter_request_share": 0.0,
                "weighted_mean_type_cost_s": 0.0,
            }

        heaviest_type_cost_s = max(type_costs.values())
        dominant_type_size = max(len(grouped) for grouped in grouped_requests.values())
        heavy_request_count = sum(
            len(grouped_requests[type_key])
            for type_key, type_cost_s in type_costs.items()
            if type_cost_s == heaviest_type_cost_s
        )
        weighted_mean_type_cost_s = (
            sum(type_costs[type_key] * len(grouped) for type_key, grouped in grouped_requests.items())
            / float(total_requests)
        )
        heavy_type_share = float(heavy_request_count) / float(total_requests)
        lighter_request_share = max(1.0 - heavy_type_share, 0.0)
        return {
            "queue_depth": float(total_requests),
            "type_count": float(len(grouped_requests)),
            "dominant_type_share": float(dominant_type_size) / float(total_requests),
            "heavy_type_share": heavy_type_share,
            "lighter_request_share": lighter_request_share,
            "weighted_mean_type_cost_s": float(weighted_mean_type_cost_s),
        }

    def _adaptive_type_fifo_defer_budget_limit(
        self,
        waiting_queue_len: int,
        *,
        target_ratio: float,
        distribution_stats: dict[str, float],
    ) -> tuple[int, float]:
        if waiting_queue_len <= 1 or target_ratio <= 0.0:
            return 0, 0.0

        dominant_type_share = float(distribution_stats.get("dominant_type_share", 0.0) or 0.0)
        heavy_type_share = float(distribution_stats.get("heavy_type_share", 0.0) or 0.0)
        lighter_request_share = float(distribution_stats.get("lighter_request_share", 0.0) or 0.0)
        type_count = int(distribution_stats.get("type_count", 0.0) or 0.0)
        adaptive_ratio = min(
            1.0,
            target_ratio
            * (
                1.0
                + (dominant_type_share * lighter_request_share)
                + max(heavy_type_share - target_ratio, 0.0)
            ),
        )
        defer_budget_limit = min(int(waiting_queue_len * adaptive_ratio), waiting_queue_len - 1)
        if (
            defer_budget_limit <= 0
            and type_count >= 2
            and waiting_queue_len >= _TYPE_FIFO_DEFER_ADAPTIVE_MIN_QUEUE_DEPTH
            and lighter_request_share > 0.0
            and heavy_type_share >= max(target_ratio, 1.0 / float(waiting_queue_len))
        ):
            defer_budget_limit = 1

        defer_budget_limit = max(defer_budget_limit, _TYPE_FIFO_DEFER_MIN_BUDGET)
        return defer_budget_limit, adaptive_ratio

    @classmethod
    def _request_type_key(cls, queued_request: _QueuedRequest) -> tuple[int, int, int, int, int]:
        request_summary = cls._request_summary(queued_request.request)
        sampling_params = getattr(queued_request.request, "sampling_params", None)
        num_outputs = max(cls._safe_int(getattr(sampling_params, "num_outputs_per_prompt", 1), 1), 1)
        return (
            request_summary["width"],
            request_summary["height"],
            request_summary["num_frames"],
            request_summary["total_steps"],
            num_outputs,
        )

    @staticmethod
    def _format_type_key(type_key: tuple[int, int, int, int, int]) -> str:
        width, height, num_frames, total_steps, num_outputs = type_key
        return f"{width}x{height}:{num_frames}f:{total_steps}s:{num_outputs}o"

    def _build_type_fifo_defer_budget_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        defer_budget_ratio = self._type_fifo_defer_budget_ratio()
        wait_p95_s = self._type_fifo_defer_wait_p95_s(waiting_requests)
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        grouped_requests: dict[tuple[int, int, int, int, int], list[_QueuedRequest]] = {}
        type_costs: dict[tuple[int, int, int, int, int], float] = {}

        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            cost_weight = self._sjf_aging_cost_weight(estimated_cost_s)
            aged_cost_s = estimated_cost_s / (1.0 + (aging_factor * cost_weight * age_s))
            type_key = self._request_type_key(queued_request)
            grouped_requests.setdefault(type_key, []).append(queued_request)
            type_costs[type_key] = max(type_costs.get(type_key, 0.0), estimated_cost_s)
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _TYPE_FIFO_DEFER_BUDGET_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aging_cost_weight": cost_weight,
                "aged_cost_s": aged_cost_s,
                "type_key": self._format_type_key(type_key),
                "type_cost_s": 0.0,
                "same_type_rank": 0,
                "same_type_size": 0,
                "defer_budget_ratio": defer_budget_ratio,
                "defer_budget_limit": 0,
                "defer_threshold_s": self._type_fifo_defer_threshold_s(estimated_cost_s, wait_p95_s=wait_p95_s),
                "over_starved_threshold_s": self._type_fifo_defer_overstarved_threshold_s(
                    estimated_cost_s,
                    wait_p95_s=wait_p95_s,
                ),
                "deferred": 0,
                "defer_candidate": 0,
                "dispatch_group": "normal",
            }

        for type_key, grouped in grouped_requests.items():
            grouped.sort(
                key=lambda queued: (
                    float(getattr(queued.request, "arrival_time", queued.enqueue_time)),
                    queued.enqueue_time,
                    queued.sequence_id,
                )
            )
            type_cost_s = type_costs[type_key]
            for same_type_rank, queued_request in enumerate(grouped, start=1):
                request_metrics = metrics_by_sequence[queued_request.sequence_id]
                request_metrics["type_cost_s"] = type_cost_s
                request_metrics["same_type_rank"] = same_type_rank
                request_metrics["same_type_size"] = len(grouped)

        distribution_stats = self._type_fifo_defer_distribution_stats(grouped_requests, type_costs)
        queue_defer_budget_limit, adaptive_budget_ratio = self._adaptive_type_fifo_defer_budget_limit(
            len(waiting_requests),
            target_ratio=defer_budget_ratio,
            distribution_stats=distribution_stats,
        )
        budget_status = self._type_fifo_defer_budget_status(defer_budget_ratio)
        defer_budget_limit = min(
            queue_defer_budget_limit,
            budget_status["global_budget_remaining"],
            budget_status["window_budget_remaining"],
        )
        for request_metrics in metrics_by_sequence.values():
            request_metrics["defer_budget_limit"] = defer_budget_limit
            request_metrics["queue_defer_budget_limit"] = queue_defer_budget_limit
            request_metrics["adaptive_defer_budget_ratio"] = adaptive_budget_ratio
            request_metrics["dominant_type_share"] = distribution_stats["dominant_type_share"]
            request_metrics["heavy_type_share"] = distribution_stats["heavy_type_share"]
            request_metrics["lighter_request_share"] = distribution_stats["lighter_request_share"]
            request_metrics["weighted_mean_type_cost_s"] = distribution_stats["weighted_mean_type_cost_s"]
            request_metrics["global_arrived_unique"] = budget_status["global_arrived_unique"]
            request_metrics["global_budget_limit"] = budget_status["global_budget_limit"]
            request_metrics["global_budget_used"] = budget_status["global_budget_used"]
            request_metrics["global_budget_remaining"] = budget_status["global_budget_remaining"]
            request_metrics["window_arrived_unique"] = budget_status["window_arrived_unique"]
            request_metrics["window_budget_limit"] = budget_status["window_budget_limit"]
            request_metrics["window_budget_used"] = budget_status["window_budget_used"]
            request_metrics["window_budget_remaining"] = budget_status["window_budget_remaining"]
            request_metrics["wait_p95_s"] = wait_p95_s
            request_metrics["defer_relief_score"] = 0.0
            request_metrics["defer_harm_score"] = 0.0
            request_metrics["over_starved"] = 0

        deferred_requests: list[_QueuedRequest] = []
        grouped_remaining = {type_key: list(grouped) for type_key, grouped in grouped_requests.items()}

        while len(deferred_requests) < defer_budget_limit:
            remaining_type_costs = {
                type_key: type_costs[type_key] for type_key, grouped in grouped_remaining.items() if grouped
            }
            if len(remaining_type_costs) <= 1:
                break
            max_type_cost_s = max(remaining_type_costs.values())
            lighter_type_exists = any(type_cost_s < max_type_cost_s for type_cost_s in remaining_type_costs.values())
            if not lighter_type_exists:
                break
            remaining_request_count = sum(len(grouped) for grouped in grouped_remaining.values())
            lighter_request_count = sum(
                len(grouped)
                for type_key, grouped in grouped_remaining.items()
                if grouped and remaining_type_costs[type_key] < max_type_cost_s
            )
            lighter_type_count = sum(
                1 for type_key, grouped in grouped_remaining.items() if grouped and remaining_type_costs[type_key] < max_type_cost_s
            )
            lighter_mean_cost_s = (
                sum(
                    remaining_type_costs[type_key] * len(grouped)
                    for type_key, grouped in grouped_remaining.items()
                    if grouped and remaining_type_costs[type_key] < max_type_cost_s
                )
                / float(lighter_request_count)
                if lighter_request_count > 0
                else max_type_cost_s
            )
            lighter_request_share = (
                float(lighter_request_count) / float(remaining_request_count)
                if remaining_request_count > 0
                else 0.0
            )

            candidate_type_keys = [
                type_key for type_key, type_cost_s in remaining_type_costs.items() if type_cost_s == max_type_cost_s
            ]
            eligible_candidates: list[tuple[float, float, int, tuple[int, int, int, int, int], _QueuedRequest]] = []
            for type_key in candidate_type_keys:
                head_request = grouped_remaining[type_key][0]
                request_metrics = metrics_by_sequence[head_request.sequence_id]
                request_metrics["defer_candidate"] = 1
                defer_threshold_s = float(request_metrics["defer_threshold_s"] or 0.0)
                estimated_cost_s = max(float(request_metrics["estimated_cost_s"] or 0.0), 1e-9)
                age_s = float(request_metrics["age_s"] or 0.0)
                over_starved_threshold_s = float(request_metrics["over_starved_threshold_s"] or 0.0)
                defer_relief_score = float(lighter_request_count) * max(max_type_cost_s - lighter_mean_cost_s, 0.0)
                defer_harm_score = estimated_cost_s * max(age_s / max(defer_threshold_s, 1e-9), 1.0)
                over_starved = int(age_s >= over_starved_threshold_s)
                request_metrics["lighter_request_count"] = lighter_request_count
                request_metrics["lighter_type_count"] = lighter_type_count
                request_metrics["lighter_request_share"] = lighter_request_share
                request_metrics["lighter_mean_cost_s"] = lighter_mean_cost_s
                request_metrics["defer_relief_score"] = defer_relief_score
                request_metrics["defer_harm_score"] = defer_harm_score
                request_metrics["over_starved"] = over_starved
                if (
                    age_s >= defer_threshold_s
                    and not over_starved
                    and defer_relief_score > defer_harm_score
                ):
                    eligible_candidates.append(
                        (
                            -defer_relief_score,
                            -age_s,
                            head_request.enqueue_time,
                            head_request.sequence_id,
                            type_key,
                            head_request,
                        )
                    )
            if not eligible_candidates:
                break

            _neg_relief_score, _neg_age_s, _enqueue_time, _sequence_id, deferred_type_key, deferred_request = min(
                eligible_candidates
            )
            grouped_remaining[deferred_type_key].pop(0)
            deferred_requests.append(deferred_request)
            deferred_metrics = metrics_by_sequence[deferred_request.sequence_id]
            deferred_metrics["deferred"] = 1
            deferred_metrics["dispatch_group"] = "deferred"
            self._mark_type_fifo_deferred(deferred_request.request)

        ordered_queue: list[_QueuedRequest] = []
        while True:
            head_candidates: list[tuple[float, float, float, int, tuple[int, int, int, int, int], _QueuedRequest]] = []
            for type_key, grouped in grouped_remaining.items():
                if not grouped:
                    continue
                head_request = grouped[0]
                request_metrics = metrics_by_sequence[head_request.sequence_id]
                head_candidates.append(
                    (
                        float(request_metrics["aged_cost_s"]),
                        float(request_metrics["estimated_cost_s"]),
                        float(getattr(head_request.request, "arrival_time", head_request.enqueue_time)),
                        head_request.sequence_id,
                        type_key,
                        head_request,
                    )
                )
            if not head_candidates:
                break

            _aged_cost_s, _estimated_cost_s, _arrival_time, _sequence_id, selected_type_key, selected_request = min(
                head_candidates
            )
            grouped_remaining[selected_type_key].pop(0)
            ordered_queue.append(selected_request)

        deferred_requests.sort(
            key=lambda queued: (
                float(getattr(queued.request, "arrival_time", queued.enqueue_time)),
                queued.enqueue_time,
                queued.sequence_id,
            )
        )
        ordered_queue.extend(deferred_requests)

        final_budget_status = self._type_fifo_defer_budget_status(defer_budget_ratio)

        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank
            metrics_by_sequence[queued_request.sequence_id]["deferred_queue_size"] = len(deferred_requests)
            metrics_by_sequence[queued_request.sequence_id]["deferred_budget_used"] = len(deferred_requests)
            metrics_by_sequence[queued_request.sequence_id]["global_budget_used"] = final_budget_status["global_budget_used"]
            metrics_by_sequence[queued_request.sequence_id]["global_budget_remaining"] = final_budget_status["global_budget_remaining"]
            metrics_by_sequence[queued_request.sequence_id]["window_budget_used"] = final_budget_status["window_budget_used"]
            metrics_by_sequence[queued_request.sequence_id]["window_budget_remaining"] = final_budget_status["window_budget_remaining"]

        return ordered_queue, metrics_by_sequence

    def _build_size_bucket_sjf_aging_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            aged_cost_s = estimated_cost_s / (1.0 + aging_factor * age_s)
            request_summary = self._request_summary(queued_request.request)
            max_dim = max(request_summary["width"], request_summary["height"])
            raw_size_bucket_id = self._fixed_size_bucket_id(queued_request)
            bucket_promotion_levels = int((aging_factor * age_s) / _SIZE_BUCKET_PROMOTION_WINDOW_S)
            effective_size_bucket_id = max(raw_size_bucket_id - bucket_promotion_levels, 0)
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _SIZE_BUCKET_SJF_AGING_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aged_cost_s": aged_cost_s,
                "max_dim": max_dim,
                "raw_size_bucket_id": raw_size_bucket_id,
                "effective_size_bucket_id": effective_size_bucket_id,
                "bucket_promotion_levels": bucket_promotion_levels,
            }

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                int(metrics_by_sequence[queued.sequence_id]["effective_size_bucket_id"]),
                float(metrics_by_sequence[queued.sequence_id]["aged_cost_s"]),
                float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _slack_hybrid_panic_threshold(self) -> float:
        threshold = self._safe_optional_float(getattr(self.od_config, "instance_scheduler_slack_panic_threshold", None))
        if threshold is None:
            return _SLACK_HYBRID_DEFAULT_PANIC_THRESHOLD
        return max(float(threshold), 0.0)

    def _slack_hybrid_swap_overhead_seconds(self) -> float:
        swap_overhead_ms = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_slack_swap_overhead_ms", None)
        )
        if swap_overhead_ms is None:
            return 0.0
        return max(float(swap_overhead_ms), 0.0) / 1000.0

    def _slack_hybrid_ratio(
        self,
        *,
        deadline_ts: float,
        now: float,
        remaining_cost_s: float,
        swap_overhead_s: float,
    ) -> float:
        if deadline_ts == inf:
            return inf
        return (deadline_ts - now - swap_overhead_s) / max(remaining_cost_s, 1e-9)

    def _active_slack_hybrid_ratio(
        self,
        now: float,
        swap_overhead_s: float,
        dynamic_p95_ms: float | None = None,
    ) -> float:
        if self._active_request is None:
            return inf
        remaining_cost_s = max(self._active_total_remaining_cost_seconds(now), 1e-9)
        deadline_ts = self._deadline_ts(self._active_request, dynamic_p95_ms)
        return self._slack_hybrid_ratio(
            deadline_ts=deadline_ts,
            now=now,
            remaining_cost_s=remaining_cost_s,
            swap_overhead_s=swap_overhead_s,
        )

    def _build_slack_hybrid_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = float(getattr(self.od_config, "instance_scheduler_aging_factor", 0.0) or 0.0)
        panic_threshold = self._slack_hybrid_panic_threshold()
        swap_overhead_s = self._slack_hybrid_swap_overhead_seconds()
        dynamic_p95_ms, learned_p95_ms, backlog_adjusted_p95_ms, uses_learned_deadline = self._learned_deadline_context(
            waiting_requests,
            now,
            include_active=True,
        )
        active_slack_ratio = self._active_slack_hybrid_ratio(now, swap_overhead_s, dynamic_p95_ms)
        panic_mode = active_slack_ratio < panic_threshold
        metrics_by_sequence: dict[int, dict[str, Any]] = {}

        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            deadline_ts = self._deadline_ts(queued_request, dynamic_p95_ms)
            slack_ratio = self._slack_hybrid_ratio(
                deadline_ts=deadline_ts,
                now=now,
                remaining_cost_s=estimated_cost_s,
                swap_overhead_s=swap_overhead_s,
            )
            throughput_priority = estimated_cost_s - (aging_factor * age_s)
            is_urgent = int(slack_ratio < panic_threshold)
            panic_mode = panic_mode or bool(is_urgent)
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _SLACK_HYBRID_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "deadline_ts": deadline_ts,
                "slack_ratio": slack_ratio,
                "throughput_priority": throughput_priority,
                "panic_threshold": panic_threshold,
                "swap_overhead_ms": swap_overhead_s * 1000.0,
                "active_slack_ratio": active_slack_ratio,
                "is_urgent": is_urgent,
                "dispatch_group": "single_queue",
                "learned_deadline_fallback": int(uses_learned_deadline and self._uses_learned_deadline(queued_request)),
                "dynamic_p95_ms": dynamic_p95_ms,
                "learned_p95_ms": learned_p95_ms,
                "backlog_adjusted_p95_ms": backlog_adjusted_p95_ms,
            }

        if panic_mode:
            ordered_queue = sorted(
                waiting_requests,
                key=lambda queued: (
                    0 if metrics_by_sequence[queued.sequence_id]["is_urgent"] else 1,
                    float(metrics_by_sequence[queued.sequence_id]["deadline_ts"]),
                    float(metrics_by_sequence[queued.sequence_id]["slack_ratio"]),
                    float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                    queued.enqueue_time,
                    queued.sequence_id,
                ),
            )
            mode = "panic_edf"
        else:
            ordered_queue = sorted(
                waiting_requests,
                key=lambda queued: (
                    float(metrics_by_sequence[queued.sequence_id]["throughput_priority"]),
                    float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                    queued.enqueue_time,
                    queued.sequence_id,
                ),
            )
            mode = "throughput_srpt"

        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["hybrid_mode"] = mode
            metrics_by_sequence[queued_request.sequence_id]["priority_score"] = (
                float(metrics_by_sequence[queued_request.sequence_id]["deadline_ts"])
                if panic_mode
                else float(metrics_by_sequence[queued_request.sequence_id]["throughput_priority"])
            )
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _build_sjf_aging_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            cost_weight = self._sjf_aging_cost_weight(estimated_cost_s)
            aged_cost_s = estimated_cost_s / (1.0 + (aging_factor * cost_weight * age_s))
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _SJF_AGING_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aging_cost_weight": cost_weight,
                "aged_cost_s": aged_cost_s,
            }

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                float(metrics_by_sequence[queued.sequence_id]["aged_cost_s"]),
                float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _build_sjf_aging_guarded_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        learned_wait_guard_s = self._learned_sjf_aging_guarded_wait_s()
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            cost_weight = self._sjf_aging_cost_weight(estimated_cost_s)
            aged_cost_s = estimated_cost_s / (1.0 + (aging_factor * cost_weight * age_s))
            protection_threshold_s = max(learned_wait_guard_s, _SJF_AGING_GUARDED_WAIT_COST_RATIO * estimated_cost_s)
            tail_protected = age_s >= protection_threshold_s
            setattr(queued_request.request, "tail_protected", tail_protected)
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _SJF_AGING_GUARDED_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aging_cost_weight": cost_weight,
                "aged_cost_s": aged_cost_s,
                "tail_protected": int(tail_protected),
                "protection_threshold_s": protection_threshold_s,
                "wait_ratio": age_s / estimated_cost_s,
                "learned_wait_guard_s": learned_wait_guard_s,
                "dispatch_group": "protected" if tail_protected else "normal",
            }

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                0 if metrics_by_sequence[queued.sequence_id]["tail_protected"] else 1,
                float(getattr(queued.request, "arrival_time", queued.enqueue_time))
                if metrics_by_sequence[queued.sequence_id]["tail_protected"]
                else float(metrics_by_sequence[queued.sequence_id]["aged_cost_s"]),
                queued.enqueue_time
                if metrics_by_sequence[queued.sequence_id]["tail_protected"]
                else float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _build_sjf_aging_guarded_tail_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        learned_wait_guard_s = self._learned_sjf_aging_guarded_wait_s()
        queue_costs = sorted(max(self._queued_cost_seconds(queued_request), 1e-9) for queued_request in waiting_requests)
        mid = len(queue_costs) // 2
        if len(queue_costs) % 2 == 1:
            queue_median_cost_s = queue_costs[mid]
        else:
            queue_median_cost_s = (queue_costs[mid - 1] + queue_costs[mid]) / 2.0
        queue_p75_cost_s = queue_costs[max(int(len(queue_costs) * 0.75) - 1, 0)]
        queue_p90_cost_s = queue_costs[max(ceil(len(queue_costs) * 0.9) - 1, 0)]
        metrics_by_sequence: dict[int, dict[str, Any]] = {}

        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            cost_weight = self._sjf_aging_cost_weight(estimated_cost_s)
            aged_cost_s = estimated_cost_s / (1.0 + (aging_factor * cost_weight * age_s))
            protection_threshold_s = max(learned_wait_guard_s, _SJF_AGING_GUARDED_WAIT_COST_RATIO * estimated_cost_s)
            tail_protected = age_s >= protection_threshold_s
            large_request_threshold_s = max(
                _SJF_AGING_GUARDED_TAIL_COST_SCALE * queue_median_cost_s,
                queue_p75_cost_s,
            )
            sink_threshold_s = max(
                _SJF_AGING_GUARDED_TAIL_DEFER_WAIT_MULTIPLIER * protection_threshold_s,
                _SJF_AGING_GUARDED_TAIL_DEFER_COST_MULTIPLIER * estimated_cost_s,
            )
            hard_escape_threshold_s = max(
                _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_WAIT_MULTIPLIER * learned_wait_guard_s,
                _SJF_AGING_GUARDED_TAIL_HARD_ESCAPE_COST_MULTIPLIER * estimated_cost_s,
            )
            request_key = self._sjf_aging_guarded_tail_request_key(queued_request.request)
            tail_sunk = request_key in self._sjf_aging_guarded_tail_active_sunk_request_ids
            hard_escape = False
            setattr(queued_request.request, "tail_protected", tail_protected)
            setattr(queued_request.request, "tail_sunk", tail_sunk)
            setattr(queued_request.request, "tail_hard_escape", hard_escape)
            dispatch_group = "normal"
            if tail_sunk:
                dispatch_group = "sunk_tail"
            elif tail_protected:
                dispatch_group = "protected_soft"
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _SJF_AGING_GUARDED_TAIL_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aging_cost_weight": cost_weight,
                "aged_cost_s": aged_cost_s,
                "tail_protected": int(tail_protected),
                "tail_sunk": int(tail_sunk),
                "super_heavy": int(estimated_cost_s >= large_request_threshold_s),
                "protection_threshold_s": protection_threshold_s,
                "sink_threshold_s": sink_threshold_s,
                "hard_escape_threshold_s": hard_escape_threshold_s,
                "hard_escape": int(hard_escape),
                "wait_ratio": age_s / estimated_cost_s,
                "learned_wait_guard_s": learned_wait_guard_s,
                "queue_median_cost_s": queue_median_cost_s,
                "queue_p75_cost_s": queue_p75_cost_s,
                "queue_p90_cost_s": queue_p90_cost_s,
                "defer_relief_score": 0.0,
                "defer_harm_score": 0.0,
                "lighter_request_count": 0,
                "lighter_mean_cost_s": estimated_cost_s,
                "dispatch_group": dispatch_group,
            }

        budget_status = self._sjf_aging_guarded_tail_budget_status()
        sink_budget_limit = 1 if len(waiting_requests) > 1 else 0
        sink_budget_limit = min(
            sink_budget_limit,
            budget_status["global_budget_remaining"],
            budget_status["window_budget_remaining"],
        )
        for request_metrics in metrics_by_sequence.values():
            request_metrics["tail_defer_budget_ratio"] = _SJF_AGING_GUARDED_TAIL_DEFER_BUDGET_RATIO
            request_metrics["tail_defer_budget_limit"] = sink_budget_limit
            request_metrics["global_arrived_unique"] = budget_status["global_arrived_unique"]
            request_metrics["global_budget_limit"] = budget_status["global_budget_limit"]
            request_metrics["global_budget_used"] = budget_status["global_budget_used"]
            request_metrics["global_budget_remaining"] = budget_status["global_budget_remaining"]
            request_metrics["window_arrived_unique"] = budget_status["window_arrived_unique"]
            request_metrics["window_budget_limit"] = budget_status["window_budget_limit"]
            request_metrics["window_budget_used"] = budget_status["window_budget_used"]
            request_metrics["window_budget_remaining"] = budget_status["window_budget_remaining"]

        eligible_candidates: list[tuple[float, float, float, int, _QueuedRequest]] = []
        if sink_budget_limit > 0:
            for queued_request in waiting_requests:
                request_metrics = metrics_by_sequence[queued_request.sequence_id]
                if (
                    not request_metrics["tail_protected"]
                    or not request_metrics["super_heavy"]
                    or request_metrics["tail_sunk"]
                ):
                    continue
                estimated_cost_s = float(request_metrics["estimated_cost_s"])
                age_s = float(request_metrics["age_s"])
                sink_threshold_s = float(request_metrics["sink_threshold_s"])
                hard_escape_threshold_s = float(request_metrics["hard_escape_threshold_s"])
                lighter_costs = [
                    float(metrics["estimated_cost_s"])
                    for sequence_id, metrics in metrics_by_sequence.items()
                    if sequence_id != queued_request.sequence_id and float(metrics["estimated_cost_s"]) < estimated_cost_s
                ]
                lighter_request_count = len(lighter_costs)
                if lighter_request_count == 0 or len(waiting_requests) < 3:
                    continue
                lighter_mean_cost_s = sum(lighter_costs) / float(lighter_request_count)
                defer_relief_score = float(lighter_request_count) * max(estimated_cost_s - lighter_mean_cost_s, 0.0)
                defer_harm_score = estimated_cost_s * max(age_s / max(hard_escape_threshold_s, 1e-9), 1.0)
                request_metrics["lighter_request_count"] = lighter_request_count
                request_metrics["lighter_mean_cost_s"] = lighter_mean_cost_s
                request_metrics["defer_relief_score"] = defer_relief_score
                request_metrics["defer_harm_score"] = defer_harm_score
                request_metrics["hard_escape"] = 0
                if age_s >= sink_threshold_s:
                    eligible_candidates.append(
                        (
                            -defer_relief_score,
                            -float(lighter_request_count),
                            -age_s,
                            queued_request.enqueue_time,
                            queued_request.sequence_id,
                            queued_request,
                        )
                    )

        sunk_request_ids: set[int] = set()
        for _ in range(sink_budget_limit):
            if not eligible_candidates:
                break
            _neg_relief_score, _neg_lighter_count, _neg_age_s, _enqueue_time, _sequence_id, sunk_request = min(eligible_candidates)
            if sunk_request.sequence_id in sunk_request_ids:
                continue
            sunk_request_ids.add(sunk_request.sequence_id)
            setattr(sunk_request.request, "tail_sunk", True)
            sunk_metrics = metrics_by_sequence[sunk_request.sequence_id]
            sunk_metrics["tail_sunk"] = 1
            sunk_metrics["dispatch_group"] = "sunk_tail"
            self._mark_sjf_aging_guarded_tail_deferred(sunk_request.request)

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                2
                if metrics_by_sequence[queued.sequence_id]["tail_sunk"]
                else 1
                if metrics_by_sequence[queued.sequence_id]["tail_protected"]
                else 0,
                float(getattr(queued.request, "arrival_time", queued.enqueue_time))
                if metrics_by_sequence[queued.sequence_id]["tail_protected"]
                or metrics_by_sequence[queued.sequence_id]["tail_sunk"]
                else float(metrics_by_sequence[queued.sequence_id]["aged_cost_s"]),
                queued.enqueue_time
                if metrics_by_sequence[queued.sequence_id]["tail_protected"]
                or metrics_by_sequence[queued.sequence_id]["tail_sunk"]
                else float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _build_bypass_guard_sjf_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        aging_factor = self._effective_sjf_aging_factor()
        learned_wait_guard_s = self._learned_bypass_guard_wait_s()
        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        for queued_request in waiting_requests:
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            age_s = self._request_age_seconds(queued_request, now)
            cost_weight = self._sjf_aging_cost_weight(estimated_cost_s)
            aged_cost_s = estimated_cost_s / (1.0 + (aging_factor * cost_weight * age_s))
            guard_threshold_s = max(learned_wait_guard_s, _BYPASS_GUARD_WAIT_COST_RATIO * estimated_cost_s)
            previous_can_bypass = self._safe_optional_int(getattr(queued_request.request, "can_bypass", None))
            if previous_can_bypass is None:
                previous_can_bypass = 1
            can_bypass = 0 if (previous_can_bypass == 0 or age_s >= guard_threshold_s) else 1
            setattr(queued_request.request, "can_bypass", can_bypass)
            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _BYPASS_GUARD_SJF_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "age_s": age_s,
                "aging_factor": aging_factor,
                "aging_cost_weight": cost_weight,
                "aged_cost_s": aged_cost_s,
                "can_bypass": can_bypass,
                "guard_threshold_s": guard_threshold_s,
                "wait_ratio": age_s / estimated_cost_s,
                "learned_wait_guard_s": learned_wait_guard_s,
                "dispatch_group": "locked" if can_bypass == 0 else "normal",
            }

        ordered_queue = sorted(
            waiting_requests,
            key=lambda queued: (
                int(metrics_by_sequence[queued.sequence_id]["can_bypass"]),
                float(getattr(queued.request, "arrival_time", queued.enqueue_time))
                if metrics_by_sequence[queued.sequence_id]["can_bypass"] == 0
                else float(metrics_by_sequence[queued.sequence_id]["aged_cost_s"]),
                queued.enqueue_time
                if metrics_by_sequence[queued.sequence_id]["can_bypass"] == 0
                else float(metrics_by_sequence[queued.sequence_id]["estimated_cost_s"]),
                queued.sequence_id,
            ),
        )
        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics_by_sequence[queued_request.sequence_id]["queue_rank"] = queue_rank

        return ordered_queue, metrics_by_sequence

    def _p95_fusion_tail_budget_ratio(self) -> float:
        value = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_fusion_tail_budget_ratio", None)
        )
        if value is None:
            return 0.10
        return min(max(value, 1e-9), 1.0)

    def _p95_fusion_heavy_threshold_s(self) -> float:
        value = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_fusion_heavy_threshold_s", None)
        )
        if value is None:
            return 20.0
        return max(value, 1e-9)

    def _p95_fusion_urgent_slack_ratio(self) -> float:
        value = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_fusion_urgent_slack_ratio", None)
        )
        if value is None:
            return 1.0
        return max(value, 0.0)

    def _p95_fusion_promote_wait_s(self) -> float:
        value = self._safe_optional_float(
            getattr(self.od_config, "instance_scheduler_p95_fusion_promote_wait_s", None)
        )
        if value is None:
            return 60.0
        return max(value, 1e-9)

    def _p95_fusion_nonheavy_streak_limit(self) -> int:
        return max(
            self._safe_int(
                getattr(self.od_config, "instance_scheduler_p95_fusion_nonheavy_streak_limit", 4),
                4,
            ),
            1,
        )

    def _p95_fusion_growth_every(self) -> int:
        return max(
            self._safe_int(getattr(self.od_config, "instance_scheduler_p95_fusion_growth_every", 20), 20),
            1,
        )

    def _p95_fusion_borrowed_cap_max(self) -> int:
        return max(
            self._safe_int(
                getattr(self.od_config, "instance_scheduler_p95_fusion_borrowed_cap_max", 4),
                4,
            ),
            0,
        )

    def _p95_fusion_min_chunk_steps(self) -> int:
        return max(
            self._safe_int(
                getattr(self.od_config, "instance_scheduler_p95_fusion_min_chunk_steps", 1),
                1,
            ),
            1,
        )

    def _p95_fusion_max_chunk_steps(self) -> int:
        return max(
            self._safe_int(
                getattr(self.od_config, "instance_scheduler_p95_fusion_max_chunk_steps", 8),
                8,
            ),
            self._p95_fusion_min_chunk_steps(),
        )

    @staticmethod
    def _reset_scheduler_chunk_annotations(request: OmniDiffusionRequest) -> None:
        setattr(request, "scheduler_force_run_to_completion", False)
        setattr(request, "scheduler_chunk_budget_steps", None)

    def _build_p95_fusion_queue(
        self,
        waiting_requests: list[_QueuedRequest],
        now: float,
    ) -> tuple[list[_QueuedRequest], dict[int, dict[str, Any]]]:
        if not waiting_requests:
            return [], {}

        dynamic_p95_ms, backlog_s, learned_p95_ms, backlog_adjusted_p95_ms = self._compute_dynamic_p95_ms(
            waiting_requests,
            now,
        )
        learned_slowdown_p95 = self._learned_p95_first_slowdown()
        active_chunk_blocking_ms = self._p95_first_active_chunk_blocking_ms(now)
        heavy_threshold_s = self._p95_fusion_heavy_threshold_s()
        urgent_slack_ratio = self._p95_fusion_urgent_slack_ratio()
        promote_wait_s = self._p95_fusion_promote_wait_s()
        min_chunk_steps = self._p95_fusion_min_chunk_steps()
        max_chunk_steps = self._p95_fusion_max_chunk_steps()

        metrics_by_sequence: dict[int, dict[str, Any]] = {}
        tail_candidates: list[_QueuedRequest] = []

        for queued_request in waiting_requests:
            request = queued_request.request
            self._reset_scheduler_chunk_annotations(request)
            estimated_cost_s = max(self._queued_cost_seconds(queued_request), 1e-9)
            estimated_service_ms = max(self._p95_first_estimated_service_ms(queued_request), 1e-9)
            estimated_service_s = estimated_service_ms / 1000.0
            age_s = self._request_age_seconds(queued_request, now)
            target_latency_ms = max(dynamic_p95_ms, learned_slowdown_p95 * estimated_service_ms, 1e-9)
            predicted_finish_latency_ms = max((age_s * 1000.0) + active_chunk_blocking_ms + estimated_service_ms, 0.0)
            pressure_ratio = predicted_finish_latency_ms / max(target_latency_ms, 1e-9)
            slack_s = (target_latency_ms / 1000.0) - age_s - estimated_service_s
            slack_ratio = slack_s / max(estimated_service_s, 1e-9)
            is_heavy = estimated_service_s >= heavy_threshold_s
            is_urgent = slack_s <= 0.0 or slack_ratio <= urgent_slack_ratio or (is_heavy and age_s >= promote_wait_s)
            tail_candidate = is_heavy

            total_steps = max(self._safe_int(getattr(request.sampling_params, "num_inference_steps", 1), 1), 1)
            executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
            remaining_steps = max(total_steps - executed_steps, 0)
            chunk_budget_override_steps: int | None = None
            force_run_to_completion = False
            if is_heavy and remaining_steps > 0:
                overdue_s = max(-slack_s, 0.0)
                if is_urgent and overdue_s >= estimated_service_s:
                    force_run_to_completion = True
                else:
                    chunk_budget_override_steps = min(max_chunk_steps, remaining_steps)
                    chunk_budget_override_steps = max(min(chunk_budget_override_steps, remaining_steps), min_chunk_steps)

            metrics_by_sequence[queued_request.sequence_id] = {
                "scheduler_policy": _P95_FUSION_POLICY,
                "queue_reorder_count": 1,
                "estimated_cost_s": estimated_cost_s,
                "estimated_service_ms": estimated_service_ms,
                "estimated_service_s": estimated_service_s,
                "dynamic_p95_ms": dynamic_p95_ms,
                "learned_p95_ms": learned_p95_ms,
                "backlog_adjusted_p95_ms": backlog_adjusted_p95_ms,
                "learned_slowdown_p95": learned_slowdown_p95,
                "backlog_s_at_schedule": backlog_s,
                "instance_backlog_total_s": backlog_s,
                "active_chunk_blocking_ms": active_chunk_blocking_ms,
                "active_chunk_blocking_s": active_chunk_blocking_ms / 1000.0,
                "target_latency_ms": target_latency_ms,
                "predicted_finish_latency_ms": predicted_finish_latency_ms,
                "pressure_ratio": pressure_ratio,
                "slack_s": slack_s,
                "slack_ratio": slack_ratio,
                "age_s": age_s,
                "is_heavy": int(is_heavy),
                "is_urgent": int(is_urgent),
                "tail_candidate": int(tail_candidate),
                "tail_lane_selected": 0,
                "tail_lane_rank": 0,
                "scheduler_force_run_to_completion": int(force_run_to_completion),
                "scheduler_chunk_budget_steps": chunk_budget_override_steps,
                "remaining_steps": remaining_steps,
                "dispatch_group": "rest",
                "nonheavy_streak_seed": self._p95_fusion_nonheavy_streak,
            }
            if tail_candidate:
                tail_candidates.append(queued_request)

        ratio_cap = ceil(len(waiting_requests) * self._p95_fusion_tail_budget_ratio())
        borrowed_cap = min(self._p95_fusion_borrowed_cap, self._p95_fusion_borrowed_cap_max())
        lane_cap = min(max(ratio_cap, borrowed_cap), ceil(len(waiting_requests) * _P95_FUSION_MAX_LANE_RATIO))
        tail_lane_sorted = sorted(
            tail_candidates,
            key=lambda queued: (
                0 if metrics_by_sequence[queued.sequence_id]["is_urgent"] else 1,
                float(metrics_by_sequence[queued.sequence_id]["slack_s"]),
                -float(metrics_by_sequence[queued.sequence_id]["age_s"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        if tail_lane_sorted:
            lane_cap = min(max(lane_cap, 1), len(tail_lane_sorted))
        tail_lane = tail_lane_sorted[:lane_cap]
        for queued_request in waiting_requests:
            request = queued_request.request
            metrics = metrics_by_sequence[queued_request.sequence_id]
            if bool(metrics["scheduler_force_run_to_completion"]):
                setattr(request, "scheduler_force_run_to_completion", True)
                setattr(request, "scheduler_chunk_budget_steps", None)
            elif metrics["scheduler_chunk_budget_steps"] is not None:
                setattr(request, "scheduler_chunk_budget_steps", int(metrics["scheduler_chunk_budget_steps"]))

        for tail_lane_rank, queued_request in enumerate(tail_lane, start=1):
            metrics = metrics_by_sequence[queued_request.sequence_id]
            metrics["tail_lane_selected"] = 1
            metrics["tail_lane_rank"] = tail_lane_rank
            metrics["dispatch_group"] = "tail_lane"

        urgent_all = sorted(
            [
                queued
                for queued in waiting_requests
                if metrics_by_sequence[queued.sequence_id]["is_urgent"]
            ],
            key=lambda queued: (
                float(metrics_by_sequence[queued.sequence_id]["slack_s"]),
                float(metrics_by_sequence[queued.sequence_id]["estimated_service_ms"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        short_normal = sorted(
            [
                queued
                for queued in waiting_requests
                if not metrics_by_sequence[queued.sequence_id]["is_urgent"]
                and not metrics_by_sequence[queued.sequence_id]["is_heavy"]
            ],
            key=lambda queued: (
                float(metrics_by_sequence[queued.sequence_id]["estimated_service_ms"]),
                -float(metrics_by_sequence[queued.sequence_id]["pressure_ratio"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )
        nonurgent_tail_lane = [
            queued
            for queued in tail_lane
            if not metrics_by_sequence[queued.sequence_id]["is_urgent"]
        ]
        rest = sorted(
            [
                queued
                for queued in waiting_requests
                if queued.sequence_id not in {req.sequence_id for req in urgent_all}
                and queued.sequence_id not in {req.sequence_id for req in short_normal}
                and queued.sequence_id not in {req.sequence_id for req in nonurgent_tail_lane}
            ],
            key=lambda queued: (
                -float(metrics_by_sequence[queued.sequence_id]["pressure_ratio"]),
                float(metrics_by_sequence[queued.sequence_id]["estimated_service_ms"]),
                queued.enqueue_time,
                queued.sequence_id,
            ),
        )

        ordered_queue: list[_QueuedRequest] = []
        local_nonheavy_streak = self._p95_fusion_nonheavy_streak
        nonheavy_streak_limit = self._p95_fusion_nonheavy_streak_limit()
        pending_urgent = deque(urgent_all)
        pending_short = deque(short_normal)
        pending_tail = deque(nonurgent_tail_lane)
        pending_rest = deque(rest)

        while pending_urgent or pending_short or pending_tail or pending_rest:
            next_request: _QueuedRequest | None = None
            if pending_urgent:
                next_request = pending_urgent.popleft()
            elif local_nonheavy_streak >= nonheavy_streak_limit and pending_tail:
                next_request = pending_tail.popleft()
            elif pending_short:
                next_request = pending_short.popleft()
            elif pending_tail:
                next_request = pending_tail.popleft()
            elif pending_rest:
                next_request = pending_rest.popleft()

            assert next_request is not None
            ordered_queue.append(next_request)
            if metrics_by_sequence[next_request.sequence_id]["is_heavy"]:
                local_nonheavy_streak = 0
            else:
                local_nonheavy_streak += 1

        for queue_rank, queued_request in enumerate(ordered_queue, start=1):
            metrics = metrics_by_sequence[queued_request.sequence_id]
            metrics["queue_rank"] = queue_rank
            metrics["tail_budget_ratio"] = self._p95_fusion_tail_budget_ratio()
            metrics["ratio_cap"] = ratio_cap
            metrics["borrowed_cap"] = borrowed_cap
            metrics["lane_cap"] = lane_cap
            metrics["nonheavy_streak_limit"] = nonheavy_streak_limit
            if metrics["is_urgent"]:
                metrics["dispatch_group"] = "urgent"
            elif metrics["tail_lane_selected"]:
                metrics["dispatch_group"] = "tail_lane"
            elif not metrics["is_heavy"]:
                metrics["dispatch_group"] = "short_normal"
            else:
                metrics["dispatch_group"] = "rest"

        return ordered_queue, metrics_by_sequence

    def _maybe_reorder_waiting_queue(self, new_request: _QueuedRequest, now: float) -> None:
        policy = self._policy_name()
        if policy == "fcfs":
            return

        if policy == _SJF_AGING_POLICY:
            ordered_queue, metrics_by_sequence = self._build_sjf_aging_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s estimated_cost_s=%.4f aged_cost_s=%.4f age_s=%.4f queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == _SJF_AGING_GUARDED_POLICY:
            ordered_queue, metrics_by_sequence = self._build_sjf_aging_guarded_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s estimated_cost_s=%.4f aged_cost_s=%.4f age_s=%.4f tail_protected=%s protection_threshold_s=%.4f queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("tail_protected"),
                float(request_metrics.get("protection_threshold_s", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == _SJF_AGING_GUARDED_TAIL_POLICY:
            ordered_queue, metrics_by_sequence = self._build_sjf_aging_guarded_tail_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s estimated_cost_s=%.4f aged_cost_s=%.4f age_s=%.4f tail_protected=%s tail_sunk=%s hard_escape=%s protection_threshold_s=%.4f sink_threshold_s=%.4f queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("tail_protected"),
                request_metrics.get("tail_sunk"),
                request_metrics.get("hard_escape"),
                float(request_metrics.get("protection_threshold_s", 0.0) or 0.0),
                float(request_metrics.get("sink_threshold_s", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == _BYPASS_GUARD_SJF_POLICY:
            ordered_queue, metrics_by_sequence = self._build_bypass_guard_sjf_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s estimated_cost_s=%.4f aged_cost_s=%.4f age_s=%.4f can_bypass=%s guard_threshold_s=%.4f queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("can_bypass"),
                float(request_metrics.get("guard_threshold_s", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == _TYPE_FIFO_DEFER_BUDGET_POLICY:
            ordered_queue, metrics_by_sequence = self._build_type_fifo_defer_budget_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s type_key=%s aged_cost_s=%.4f age_s=%.4f deferred=%s over_starved=%s defer_threshold_s=%.4f over_starved_threshold_s=%.4f defer_budget_limit=%s global_budget_remaining=%s window_budget_remaining=%s queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                request_metrics.get("type_key"),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("deferred"),
                request_metrics.get("over_starved"),
                float(request_metrics.get("defer_threshold_s", 0.0) or 0.0),
                float(request_metrics.get("over_starved_threshold_s", 0.0) or 0.0),
                request_metrics.get("defer_budget_limit"),
                request_metrics.get("global_budget_remaining"),
                request_metrics.get("window_budget_remaining"),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == _SIZE_BUCKET_SJF_AGING_POLICY:
            ordered_queue, metrics_by_sequence = self._build_size_bucket_sjf_aging_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s raw_size_bucket=%s effective_size_bucket=%s aged_cost_s=%.4f age_s=%.4f queue_rank=%s",
                self._request_label(new_request.request),
                policy,
                request_metrics.get("raw_size_bucket_id"),
                request_metrics.get("effective_size_bucket_id"),
                float(request_metrics.get("aged_cost_s", 0.0) or 0.0),
                float(request_metrics.get("age_s", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
            )
            return

        if policy == "sjf":
            self._waiting_queue = deque(self._build_sjf_queue(list(self._waiting_queue), now))
            new_request.schedule_metrics.update(
                {
                    "scheduler_policy": "sjf",
                    "queue_reorder_count": 1,
                    "estimated_cost_s": self._queued_cost_seconds(new_request),
                }
            )
            request_summary = self._request_summary(new_request.request)
            logger.info(
                "QUEUE_REORDER request_id=%s policy=sjf estimated_cost_s=%.4f width=%d height=%d total_steps=%d remaining_steps=%d",
                self._request_label(new_request.request),
                new_request.schedule_metrics["estimated_cost_s"],
                request_summary["width"],
                request_summary["height"],
                request_summary["total_steps"],
                request_summary["remaining_steps"],
            )
            return

        if policy == _P95_FIRST_POLICY:
            ordered_queue, metrics_by_sequence = self._build_p95_first_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s learned_slowdown_p95=%.4f pressure_ratio=%.4f final_priority_score=%.4f queue_rank=%s estimated_service_ms=%.2f risk_ms=%.2f",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("learned_slowdown_p95", 0.0) or 0.0),
                float(request_metrics.get("pressure_ratio", 0.0) or 0.0),
                float(request_metrics.get("final_priority_score", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
                float(request_metrics.get("estimated_service_ms", 0.0) or 0.0),
                float(request_metrics.get("risk_ms", 0.0) or 0.0),
            )
            return

        if policy == _P95_FIRST_DEADLINE_POLICY:
            ordered_queue, metrics_by_sequence = self._build_p95_first_deadline_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s synthetic_deadline_ts=%.4f slack_s=%.4f pressure_ratio=%.4f queue_rank=%s estimated_service_ms=%.2f",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("synthetic_deadline_ts", 0.0) or 0.0),
                float(request_metrics.get("slack_s", 0.0) or 0.0),
                float(request_metrics.get("pressure_ratio", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
                float(request_metrics.get("estimated_service_ms", 0.0) or 0.0),
            )
            return

        if policy == _P95_BUCKET_SJF_NORMALIZED_POLICY:
            ordered_queue, metrics_by_sequence = self._build_p95_bucket_sjf_normalized_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s learned_slowdown_p95=%.4f target_latency_ms=%.2f urgency_ms=%.2f bucket=%s queue_rank=%s estimated_service_ms=%.2f",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("learned_slowdown_p95", 0.0) or 0.0),
                float(request_metrics.get("target_latency_ms", 0.0) or 0.0),
                float(request_metrics.get("urgency_ms", 0.0) or 0.0),
                request_metrics.get("effective_bucket_id"),
                request_metrics.get("queue_rank"),
                float(request_metrics.get("estimated_service_ms", 0.0) or 0.0),
            )
            return

        if policy == _P95_BUCKET_SJF_POLICY:
            ordered_queue, metrics_by_sequence = self._build_p95_bucket_sjf_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s history_p95_ms=%.2f target_p95_ms=%.2f urgency_ms=%.2f bucket=%s queue_rank=%s estimated_cost_s=%.4f",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("history_p95_ms", 0.0) or 0.0),
                float(request_metrics.get("target_p95_ms", 0.0) or 0.0),
                float(request_metrics.get("urgency_ms", 0.0) or 0.0),
                request_metrics.get("effective_bucket_id"),
                request_metrics.get("queue_rank"),
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
            )
            return

        if policy == _SLACK_HYBRID_POLICY:
            ordered_queue, metrics_by_sequence = self._build_slack_hybrid_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s mode=%s slack_ratio=%.4f priority_score=%.4f queue_rank=%s estimated_cost_s=%.4f urgent=%s",
                self._request_label(new_request.request),
                policy,
                request_metrics.get("hybrid_mode"),
                float(request_metrics.get("slack_ratio", 0.0) or 0.0),
                float(request_metrics.get("priority_score", 0.0) or 0.0),
                request_metrics.get("queue_rank"),
                float(request_metrics.get("estimated_cost_s", 0.0) or 0.0),
                request_metrics.get("is_urgent"),
            )
            return

        if policy == _P95_FUSION_POLICY:
            ordered_queue, metrics_by_sequence = self._build_p95_fusion_queue(list(self._waiting_queue), now)
            self._waiting_queue = deque(ordered_queue)
            for queued_request in self._waiting_queue:
                metrics = metrics_by_sequence.get(queued_request.sequence_id)
                if metrics is not None:
                    queued_request.schedule_metrics.update(metrics)
            request_metrics = metrics_by_sequence.get(new_request.sequence_id, {})
            logger.info(
                "QUEUE_REORDER request_id=%s policy=%s estimated_service_ms=%.2f pressure_ratio=%.4f slack_s=%.4f heavy=%s urgent=%s tail_lane=%s queue_rank=%s chunk_budget_steps=%s force_run=%s",
                self._request_label(new_request.request),
                policy,
                float(request_metrics.get("estimated_service_ms", 0.0) or 0.0),
                float(request_metrics.get("pressure_ratio", 0.0) or 0.0),
                float(request_metrics.get("slack_s", 0.0) or 0.0),
                request_metrics.get("is_heavy"),
                request_metrics.get("is_urgent"),
                request_metrics.get("tail_lane_selected"),
                request_metrics.get("queue_rank"),
                request_metrics.get("scheduler_chunk_budget_steps"),
                request_metrics.get("scheduler_force_run_to_completion"),
            )
            return

        if policy not in _DEADLINE_AWARE_POLICIES:
            return

        waiting_before = list(self._waiting_queue)[:-1]
        before_plan = self._build_waiting_plan(waiting_before, now)
        after_plan = self._build_waiting_plan(list(self._waiting_queue), now)
        self._waiting_queue = deque(after_plan.ordered_queue)

        attain_before = len(before_plan.feasible_ids)
        attain_after = len(after_plan.feasible_ids)
        damage_count = len(before_plan.feasible_ids - after_plan.feasible_ids)
        self_hit = 1 if new_request.sequence_id in after_plan.feasible_ids else 0
        deadline_ts = self._deadline_ts(new_request, after_plan.dynamic_p95_ms)
        completion_ts = after_plan.completion_ts.get(new_request.sequence_id)
        slack_ms = None
        if completion_ts is not None and deadline_ts != inf:
            slack_ms = (deadline_ts - completion_ts) * 1000.0

        is_single_queue_slack = policy in {"slack_age", "slack_cost_age"}
        new_request.schedule_metrics.update(
            {
                "scheduler_policy": policy,
                "attain_before": attain_before,
                "attain_after": attain_after,
                "self_hit": self_hit,
                "damage_count": damage_count,
                "on_time_set_size": len(after_plan.ordered_queue) if is_single_queue_slack else len(after_plan.on_time_queue),
                "best_effort_set_size": 0 if is_single_queue_slack else len(after_plan.best_effort_queue),
                "tail_set_size": 0 if is_single_queue_slack else len(after_plan.best_effort_queue),
                "regret_drop_count": after_plan.regret_drop_count,
                "queue_reorder_count": 1,
                "deadline_slack_ms": slack_ms,
                "dispatch_group": "single_queue"
                if is_single_queue_slack
                else ("on_time" if new_request.sequence_id in after_plan.feasible_ids else "best_effort"),
                "estimated_cost_s": self._queued_cost_seconds(new_request),
                "learned_deadline_fallback": int(after_plan.uses_learned_deadline and self._uses_learned_deadline(new_request)),
                "dynamic_p95_ms": after_plan.dynamic_p95_ms,
                "learned_p95_ms": after_plan.learned_p95_ms,
                "backlog_adjusted_p95_ms": after_plan.backlog_adjusted_p95_ms,
            }
        )
        request_summary = self._request_summary(new_request.request)
        logger.info(
            "QUEUE_REORDER request_id=%s attain_before=%d attain_after=%d self_hit=%d damage_count=%d width=%d height=%d total_steps=%d remaining_steps=%d",
            self._request_label(new_request.request),
            attain_before,
            attain_after,
            self_hit,
            damage_count,
            request_summary["width"],
            request_summary["height"],
            request_summary["total_steps"],
            request_summary["remaining_steps"],
        )

    def _annotate_output(
        self,
        output: DiffusionOutput,
        queued_request: _QueuedRequest,
        request: OmniDiffusionRequest,
        queue_wait_ms: float,
        execute_latency_ms: float,
    ) -> DiffusionOutput:
        request_label = self._request_label(request)
        now = time.monotonic()
        chunk_latency_ms = queue_wait_ms + execute_latency_ms
        end_to_end_latency_ms = self._request_elapsed_ms(request, now)
        metrics = dict(getattr(output, "metrics", {}) or {})
        queued_metrics = dict(queued_request.schedule_metrics)
        metrics.update(
            {
                "scheduler_policy": self._policy_name(),
                "queue_wait_ms": queue_wait_ms,
                "scheduler_execute_ms": execute_latency_ms,
                "scheduler_chunk_latency_ms": chunk_latency_ms,
                "scheduler_latency_ms": end_to_end_latency_ms if end_to_end_latency_ms is not None else chunk_latency_ms,
                "queue_len": len(self._waiting_queue),
                "dispatch_epoch": self._safe_int(getattr(request, "dispatch_epoch", 0), 0),
                "executed_steps": self._safe_int(getattr(request, "executed_steps", 0), 0),
                "remaining_steps": max(
                    self._safe_int(getattr(request.sampling_params, "num_inference_steps", 0), 0)
                    - self._safe_int(getattr(request, "executed_steps", 0), 0),
                    0,
                ),
                "chunk_budget_steps": self._safe_optional_int(getattr(request, "max_steps_this_turn", None)),
            }
        )
        metrics.update(self._request_summary(request))
        metrics.update(self._request_time_summary(request))
        metrics.update(queued_metrics)
        output.metrics = metrics
        output.request_id = output.request_id or request_label
        return output

    def _refresh_output_metrics(
        self,
        output: DiffusionOutput,
        request: OmniDiffusionRequest,
        *,
        queue_len: int | None = None,
    ) -> None:
        metrics = dict(getattr(output, "metrics", {}) or {})
        metrics.update(self._request_summary(request))
        metrics.update(self._request_time_summary(request))
        metrics["dispatch_epoch"] = self._safe_int(getattr(request, "dispatch_epoch", 0), 0)
        metrics["executed_steps"] = self._safe_int(getattr(request, "executed_steps", 0), 0)
        metrics["remaining_steps"] = max(
            self._safe_int(getattr(request.sampling_params, "num_inference_steps", 0), 0)
            - self._safe_int(getattr(request, "executed_steps", 0), 0),
            0,
        )
        metrics["chunk_budget_steps"] = self._safe_optional_int(getattr(request, "max_steps_this_turn", None))
        elapsed_ms = self._request_elapsed_ms(request)
        if elapsed_ms is not None:
            metrics["scheduler_latency_ms"] = elapsed_ms
        if queue_len is not None:
            metrics["queue_len"] = queue_len
        output.metrics = metrics

    def _sync_request_progress_from_output(self, request: OmniDiffusionRequest, output: DiffusionOutput) -> None:
        metrics = dict(getattr(output, "metrics", {}) or {})
        total_steps = max(self._safe_int(getattr(request.sampling_params, "num_inference_steps", 0), 0), 0)
        current_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        current_dispatch_epoch = self._safe_int(getattr(request, "dispatch_epoch", 0), 0)

        if "executed_steps" in metrics:
            executed_steps = self._safe_int(metrics["executed_steps"], current_steps)
        elif getattr(output, "finished", True) and not output.error:
            executed_steps = total_steps
        else:
            executed_steps = current_steps

        if total_steps > 0:
            executed_steps = min(max(executed_steps, 0), total_steps)
        else:
            executed_steps = max(executed_steps, 0)

        request.executed_steps = executed_steps
        request.dispatch_epoch = self._safe_int(metrics.get("dispatch_epoch"), current_dispatch_epoch)
        metrics["executed_steps"] = executed_steps
        metrics["remaining_steps"] = max(total_steps - executed_steps, 0)
        metrics["dispatch_epoch"] = request.dispatch_epoch
        output.metrics = metrics

    def _normalize_error_output(self, request: OmniDiffusionRequest, error: str, error_code: str) -> DiffusionOutput:
        request_label = self._request_label(request)
        return DiffusionOutput(
            error=error,
            error_code=error_code,
            request_id=request_label,
            metrics={"scheduler_policy": self._policy_name(), "queue_len": len(self._waiting_queue)},
        )

    def _queue_request_locked(self, request: OmniDiffusionRequest, *, is_new_arrival: bool) -> _QueuedRequest:
        enqueue_time = time.monotonic()
        if is_new_arrival:
            arrival_time = getattr(request, "arrival_time", None)
            if not isinstance(arrival_time, (int, float)):
                arrival_time = enqueue_time
            setattr(request, "arrival_time", float(arrival_time))
        elif getattr(request, "arrival_time", None) is None:
            setattr(request, "arrival_time", enqueue_time)
        if getattr(request, "first_enqueue_time", None) is None:
            setattr(request, "first_enqueue_time", enqueue_time)
        self._reset_scheduler_chunk_annotations(request)
        self._set_request_state(request, "waiting")
        for request_id in self._request_ids(request):
            self._aborted_request_ids.discard(request_id)
        self._enqueue_seq += 1
        queued_request = _QueuedRequest(
            request=request,
            enqueue_time=enqueue_time,
            sequence_id=self._enqueue_seq,
            schedule_metrics={"scheduler_policy": self._policy_name()},
        )
        self._waiting_queue.append(queued_request)
        if is_new_arrival and self._policy_name() == _P95_FUSION_POLICY:
            self._p95_fusion_arrival_count += 1
            if self._p95_fusion_arrival_count % self._p95_fusion_growth_every() == 0:
                self._p95_fusion_borrowed_cap = min(
                    self._p95_fusion_borrowed_cap + 1,
                    self._p95_fusion_borrowed_cap_max(),
                )
        if is_new_arrival and self._policy_name() == _SJF_AGING_GUARDED_TAIL_POLICY:
            self._track_sjf_aging_guarded_tail_arrival(request)
        if is_new_arrival and self._policy_name() == _TYPE_FIFO_DEFER_BUDGET_POLICY:
            self._track_type_fifo_defer_arrival(request)
        if self._policy_name() in {_P95_FIRST_POLICY, _P95_FIRST_DEADLINE_POLICY, _P95_BUCKET_SJF_NORMALIZED_POLICY, _P95_BUCKET_SJF_POLICY, "slo_first", "slack_age", "slack_cost_age", _SLACK_HYBRID_POLICY, _P95_FUSION_POLICY}:
            self._record_p95_first_cost_observation(queued_request)
        self._maybe_reorder_waiting_queue(queued_request, enqueue_time)
        self._log_request_event(
            "QUEUE_ENQUEUE",
            request,
            queue_len=len(self._waiting_queue),
            scheduler_policy=self._policy_name(),
        )
        if is_new_arrival:
            self._log_request_event(
                "REQUEST_ARRIVED",
                request,
                queue_len=len(self._waiting_queue),
                scheduler_policy=self._policy_name(),
            )
        self._queue_cv.notify_all()
        return queued_request

    def _enqueue_request_locked(self, request: OmniDiffusionRequest) -> _QueuedRequest:
        return self._queue_request_locked(request, is_new_arrival=True)

    def _requeue_request_locked(self, request: OmniDiffusionRequest) -> _QueuedRequest:
        return self._queue_request_locked(request, is_new_arrival=False)

    def pop_next_request(self) -> OmniDiffusionRequest | None:
        with self._queue_cv:
            if self._active_request is not None or not self._waiting_queue:
                return None
            queued_request = self._waiting_queue.popleft()
            self._active_request = queued_request
            self._active_started_at = time.monotonic()
            self._set_request_state(queued_request.request, "running")
            dispatch_time = self._active_started_at
            queue_wait_ms = max((dispatch_time - queued_request.enqueue_time) * 1000.0, 0.0)
            if getattr(queued_request.request, "first_dispatch_time", None) is None:
                setattr(queued_request.request, "first_dispatch_time", dispatch_time)
            setattr(queued_request.request, "last_dispatch_time", dispatch_time)
            self._log_request_event(
                "QUEUE_DEQUEUE",
                queued_request.request,
                queue_len=len(self._waiting_queue),
                queue_wait_ms=queue_wait_ms,
                scheduler_policy=self._policy_name(),
            )
            self._log_request_event(
                "REQUEST_RESUMED" if getattr(queued_request.request, "last_preempted_time", None) is not None else "REQUEST_STARTED",
                queued_request.request,
                queue_len=len(self._waiting_queue),
                queue_wait_ms=queue_wait_ms,
                scheduler_policy=self._policy_name(),
            )
            return queued_request.request

    def estimate_waiting_queue_len(self) -> int:
        with self._queue_cv:
            return len(self._waiting_queue)

    def estimate_scheduler_load(self) -> dict[str, int]:
        with self._queue_cv:
            return {
                "waiting_queue_len": len(self._waiting_queue),
                "active_request_count": int(self._active_request is not None),
                "paused_context_count": 0,
            }

    def finish_request(self, request: OmniDiffusionRequest) -> None:
        with self._queue_cv:
            for request_id in self._request_ids(request):
                self._aborted_request_ids.discard(request_id)
            if self._policy_name() == _SJF_AGING_GUARDED_TAIL_POLICY:
                self._clear_sjf_aging_guarded_tail_sunk_state(request)
            self._set_request_state(request, "finished")

    def mark_request_unfinished(self, request: OmniDiffusionRequest) -> None:
        with self._queue_cv:
            self._set_request_state(request, "waiting")

    def fail_request(self, request: OmniDiffusionRequest) -> None:
        with self._queue_cv:
            for request_id in self._request_ids(request):
                self._aborted_request_ids.discard(request_id)
            if self._policy_name() == _SJF_AGING_GUARDED_TAIL_POLICY:
                self._clear_sjf_aging_guarded_tail_sunk_state(request)
            self._set_request_state(request, "failed")

    def abort_request(self, request_id: str) -> bool:
        with self._queue_cv:
            for queued_request in list(self._waiting_queue):
                if request_id in self._request_ids(queued_request.request):
                    self._waiting_queue.remove(queued_request)
                    self._aborted_request_ids.add(request_id)
                    if self._policy_name() == _SJF_AGING_GUARDED_TAIL_POLICY:
                        self._clear_sjf_aging_guarded_tail_sunk_state(queued_request.request)
                    self._set_request_state(queued_request.request, "aborted")
                    setattr(queued_request.request, "aborted_time", time.monotonic())
                    self._log_request_event(
                        "REQUEST_ABORTED",
                        queued_request.request,
                        queue_len=len(self._waiting_queue),
                        scheduler_policy=self._policy_name(),
                    )
                    self._queue_cv.notify_all()
                    return True

            if self._active_request is not None and request_id in self._request_ids(self._active_request.request):
                self._aborted_request_ids.add(request_id)
                if self._policy_name() == _SJF_AGING_GUARDED_TAIL_POLICY:
                    self._clear_sjf_aging_guarded_tail_sunk_state(self._active_request.request)
                self._set_request_state(self._active_request.request, "aborted")
                setattr(self._active_request.request, "aborted_time", time.monotonic())
                self._log_request_event(
                    "REQUEST_ABORTED",
                    self._active_request.request,
                    queue_len=len(self._waiting_queue),
                    scheduler_policy=self._policy_name(),
                )
                return True

            return False

    def add_req(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        request_label = self._request_label(request)

        with self._queue_cv:
            queued_request = self._find_waiting_request_locked(request)
            if queued_request is None:
                queued_request = self._enqueue_request_locked(request)
            while True:
                if self._is_request_aborted(request):
                    return self._normalize_error_output(
                        request=request,
                        error="Request aborted before dispatch",
                        error_code="REQUEST_ABORTED",
                    )
                if self._active_request is None and self._waiting_queue and self._waiting_queue[0] is queued_request:
                    self._waiting_queue.popleft()
                    self._active_request = queued_request
                    self._active_started_at = time.monotonic()
                    self._set_request_state(request, "running")
                    if getattr(request, "first_dispatch_time", None) is None:
                        setattr(request, "first_dispatch_time", self._active_started_at)
                    setattr(request, "last_dispatch_time", self._active_started_at)
                    queue_wait_ms = (time.monotonic() - queued_request.enqueue_time) * 1000
                    self._log_request_event(
                        "QUEUE_DEQUEUE",
                        request,
                        queue_len=len(self._waiting_queue),
                        queue_wait_ms=queue_wait_ms,
                        scheduler_policy=self._policy_name(),
                    )
                    self._log_request_event(
                        "REQUEST_RESUMED" if getattr(request, "last_preempted_time", None) is not None else "REQUEST_STARTED",
                        request,
                        queue_len=len(self._waiting_queue),
                        queue_wait_ms=queue_wait_ms,
                        scheduler_policy=self._policy_name(),
                    )
                    break
                self._queue_cv.wait()

        previous_executed_steps = self._safe_int(getattr(request, "executed_steps", 0), 0)
        execute_start = time.monotonic()
        try:
            with self._lock:
                self.mq.enqueue(self._build_generate_rpc_request(request))
                if self.result_mq is None:
                    output = self._normalize_error_output(
                        request=request,
                        error="Result queue not initialized",
                        error_code="RESULT_QUEUE_NOT_INITIALIZED",
                    )
                else:
                    raw_output = self.result_mq.dequeue()
                    if isinstance(raw_output, dict) and raw_output.get("status") == "error":
                        output = self._normalize_error_output(
                            request=request,
                            error=raw_output.get("error", "worker error"),
                            error_code="WORKER_EXEC_FAILED",
                        )
                    else:
                        output = raw_output
        except zmq.error.Again as exc:
            self.fail_request(request)
            setattr(request, "failure_time", time.monotonic())
            self._log_request_event(
                "REQUEST_FAILED",
                request,
                queue_len=len(self._waiting_queue),
                queue_wait_ms=queue_wait_ms,
                scheduler_policy=self._policy_name(),
            )
            logger.error("REQUEST_FAIL request_id=%s error_code=SCHEDULER_TIMEOUT", request_label)
            raise TimeoutError("Scheduler did not respond in time.") from exc

        execute_latency_ms = (time.monotonic() - execute_start) * 1000
        chunk_work_units = self._completed_chunk_work_units(request, output, previous_executed_steps)
        self._sync_request_progress_from_output(request, output)
        output = self._annotate_output(output, queued_request, request, queue_wait_ms, execute_latency_ms)
        if self._policy_name() == _P95_FUSION_POLICY and not output.error:
            if int(output.metrics.get("is_heavy", 0) or 0):
                self._p95_fusion_nonheavy_streak = 0
            else:
                self._p95_fusion_nonheavy_streak += 1
        if self._policy_name() in _NORMALIZED_P95_POLICIES and not output.error:
            previous_total_execute_ms = self._safe_optional_float(getattr(request, "_p95_first_cumulative_execute_ms", None))
            total_execute_ms = (previous_total_execute_ms or 0.0) + execute_latency_ms
            setattr(request, "_p95_first_cumulative_execute_ms", total_execute_ms)
            self._record_p95_first_execute_sample(execute_latency_ms, chunk_work_units)
        if self._policy_name() in {_SJF_AGING_GUARDED_POLICY, _SJF_AGING_GUARDED_TAIL_POLICY} and not output.error:
            previous_total_execute_ms = self._safe_optional_float(getattr(request, "_sjf_aging_guarded_cumulative_execute_ms", None))
            total_execute_ms = (previous_total_execute_ms or 0.0) + execute_latency_ms
            setattr(request, "_sjf_aging_guarded_cumulative_execute_ms", total_execute_ms)
        if self._policy_name() == _BYPASS_GUARD_SJF_POLICY and not output.error:
            previous_total_execute_ms = self._safe_optional_float(getattr(request, "_bypass_guard_cumulative_execute_ms", None))
            total_execute_ms = (previous_total_execute_ms or 0.0) + execute_latency_ms
            setattr(request, "_bypass_guard_cumulative_execute_ms", total_execute_ms)
        if self._policy_name() == _TYPE_FIFO_DEFER_BUDGET_POLICY and not output.error:
            previous_total_execute_ms = self._safe_optional_float(getattr(request, "_type_fifo_defer_cumulative_execute_ms", None))
            total_execute_ms = (previous_total_execute_ms or 0.0) + execute_latency_ms
            setattr(request, "_type_fifo_defer_cumulative_execute_ms", total_execute_ms)

        if output.error:
            self.fail_request(request)
            setattr(request, "failure_time", time.monotonic())
            if output.error_code is None:
                output.error_code = "REQUEST_EXEC_FAILED"
            self._refresh_output_metrics(output, request, queue_len=len(self._waiting_queue))
            self._log_request_event(
                "REQUEST_FAILED",
                request,
                queue_len=output.metrics.get("queue_len", -1),
                queue_wait_ms=output.metrics.get("queue_wait_ms"),
                latency_ms=output.metrics.get("scheduler_latency_ms", -1.0),
                scheduler_policy=output.metrics.get("scheduler_policy"),
            )
            logger.error(
                "REQUEST_FAIL request_id=%s queue_len=%d latency_ms=%.2f error_code=%s error=%s width=%s height=%s total_steps=%s executed_steps=%s remaining_steps=%s",
                output.request_id,
                output.metrics.get("queue_len", -1),
                output.metrics.get("scheduler_latency_ms", -1.0),
                output.error_code,
                output.error,
                output.metrics.get("width"),
                output.metrics.get("height"),
                output.metrics.get("total_steps"),
                output.metrics.get("executed_steps"),
                output.metrics.get("remaining_steps"),
            )
        elif not getattr(output, "finished", True):
            with self._queue_cv:
                setattr(request, "last_preempted_time", time.monotonic())
                self._requeue_request_locked(request)
                queue_len = len(self._waiting_queue)
            self._refresh_output_metrics(output, request, queue_len=queue_len)
            self._log_request_event(
                "REQUEST_PREEMPTED",
                request,
                queue_len=output.metrics.get("queue_len", -1),
                queue_wait_ms=output.metrics.get("queue_wait_ms"),
                latency_ms=output.metrics.get("scheduler_latency_ms", -1.0),
                scheduler_policy=output.metrics.get("scheduler_policy"),
            )
            logger.info(
                "REQUEST_CHUNK_DONE request_id=%s queue_len=%d latency_ms=%.2f width=%s height=%s total_steps=%s executed_steps=%s remaining_steps=%s chunk_budget_steps=%s dispatch_epoch=%s",
                output.request_id,
                output.metrics.get("queue_len", -1),
                output.metrics.get("scheduler_latency_ms", -1.0),
                output.metrics.get("width"),
                output.metrics.get("height"),
                output.metrics.get("total_steps"),
                output.metrics.get("executed_steps"),
                output.metrics.get("remaining_steps"),
                output.metrics.get("chunk_budget_steps"),
                output.metrics.get("dispatch_epoch"),
            )
        else:
            self.finish_request(request)
            setattr(request, "completion_time", time.monotonic())
            self._refresh_output_metrics(output, request, queue_len=len(self._waiting_queue))
            if self._policy_name() in {_P95_FIRST_POLICY, _P95_FIRST_DEADLINE_POLICY, _P95_BUCKET_SJF_NORMALIZED_POLICY, _P95_BUCKET_SJF_POLICY, "slo_first", "slack_age", "slack_cost_age", _SLACK_HYBRID_POLICY, _P95_FUSION_POLICY}:
                self._record_p95_first_latency_ms(
                    self._safe_optional_float(output.metrics.get("scheduler_latency_ms")),
                    request=request,
                )
            if self._policy_name() in {_SJF_AGING_GUARDED_POLICY, _SJF_AGING_GUARDED_TAIL_POLICY}:
                self._record_sjf_aging_guarded_wait_ms(
                    self._safe_optional_float(output.metrics.get("scheduler_latency_ms")),
                    request=request,
                )
            if self._policy_name() == _BYPASS_GUARD_SJF_POLICY:
                self._record_bypass_guard_wait_ms(
                    self._safe_optional_float(output.metrics.get("scheduler_latency_ms")),
                    request=request,
                )
            if self._policy_name() == _TYPE_FIFO_DEFER_BUDGET_POLICY:
                self._record_type_fifo_defer_wait_ms(
                    self._safe_optional_float(output.metrics.get("scheduler_latency_ms")),
                    request=request,
                )
            self._log_request_event(
                "REQUEST_COMPLETED",
                request,
                queue_len=output.metrics.get("queue_len", -1),
                queue_wait_ms=output.metrics.get("queue_wait_ms"),
                latency_ms=output.metrics.get("scheduler_latency_ms", -1.0),
                scheduler_policy=output.metrics.get("scheduler_policy"),
            )
            logger.info(
                "REQUEST_DONE request_id=%s queue_len=%d latency_ms=%.2f width=%s height=%s total_steps=%s executed_steps=%s remaining_steps=%s dispatch_epoch=%s",
                output.request_id,
                output.metrics.get("queue_len", -1),
                output.metrics.get("scheduler_latency_ms", -1.0),
                output.metrics.get("width"),
                output.metrics.get("height"),
                output.metrics.get("total_steps"),
                output.metrics.get("executed_steps"),
                output.metrics.get("remaining_steps"),
                output.metrics.get("dispatch_epoch"),
            )

        with self._queue_cv:
            if self._policy_name() == _P95_FUSION_POLICY and not self._waiting_queue:
                self._p95_fusion_nonheavy_streak = 0
            self._active_request = None
            self._active_started_at = None
            self._queue_cv.notify_all()

        return output
