# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from math import inf
from typing import Any

import zmq
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_profile import RuntimeProfileEstimator
from vllm_omni.diffusion.scheduler import Scheduler

logger = init_logger(__name__)


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
    feasible_ids: set[int]
    completion_ts: dict[int, float]
    regret_drop_count: int


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
            "%s request_id=%s width=%s height=%s total_steps=%s executed_steps=%s remaining_steps=%s dispatch_epoch=%s chunk_budget_steps=%s arrival_ts=%s first_enqueue_ts=%s first_dispatch_ts=%s last_dispatch_ts=%s last_preempted_ts=%s completion_ts=%s failure_ts=%s aborted_ts=%s queue_len=%s latency_ms=%s policy=%s",
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

    def _policy_name(self) -> str:
        return getattr(self.od_config, "instance_scheduler_policy", "fcfs")

    def _estimate_cost_seconds(self, request: OmniDiffusionRequest) -> float:
        sampling_params = request.sampling_params
        extra_args = getattr(sampling_params, "extra_args", {}) or {}
        if extra_args.get("estimated_cost_s") is not None:
            return max(float(extra_args["estimated_cost_s"]), 0.0)

        total_steps = max(self._safe_int(getattr(sampling_params, "num_inference_steps", 1), 1), 1)
        executed_steps = max(self._safe_int(getattr(request, "executed_steps", 0), 0), 0)
        num_steps = max(total_steps - executed_steps, 1)
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

    def _deadline_ts(self, queued_request: _QueuedRequest) -> float:
        extra_args = getattr(queued_request.request.sampling_params, "extra_args", {}) or {}
        if extra_args.get("deadline_ts") is not None:
            return float(extra_args["deadline_ts"])

        slo_target_ms = extra_args.get("slo_target_ms")
        if slo_target_ms is None:
            slo_target_ms = extra_args.get("slo_ms")
        if slo_target_ms is None:
            slo_target_ms = getattr(self.od_config, "instance_scheduler_slo_target_ms", None)
        if slo_target_ms is None:
            return inf

        floor_ms = float(getattr(self.od_config, "instance_scheduler_slo_floor_ms", 0.0) or 0.0)
        effective_target_ms = max(float(slo_target_ms), floor_ms)
        base_arrival_time = getattr(queued_request.request, "arrival_time", queued_request.enqueue_time)
        if not isinstance(base_arrival_time, (int, float)):
            base_arrival_time = queued_request.enqueue_time
        base_arrival_time = float(base_arrival_time)
        return base_arrival_time + (effective_target_ms / 1000.0)

    def _tail_priority(self, queued_request: _QueuedRequest, now: float) -> float:
        aging_factor = float(getattr(self.od_config, "instance_scheduler_aging_factor", 0.0) or 0.0)
        wait_s = max(now - queued_request.enqueue_time, 0.0)
        return self._queued_cost_seconds(queued_request) / (1.0 + aging_factor * wait_s)

    def _availability_ts(self, now: float) -> float:
        if self._active_request is None or self._active_started_at is None:
            return now
        elapsed = max(now - self._active_started_at, 0.0)
        remaining = max(self._queued_cost_seconds(self._active_request) - elapsed, 0.0)
        return now + remaining

    def _build_waiting_plan(self, waiting_requests: list[_QueuedRequest], now: float) -> _WaitingPlan:
        if not waiting_requests:
            return _WaitingPlan(ordered_queue=[], feasible_ids=set(), completion_ts={}, regret_drop_count=0)

        availability_ts = self._availability_ts(now)
        deadline_sorted = sorted(
            waiting_requests,
            key=lambda queued: (self._deadline_ts(queued), queued.enqueue_time, queued.sequence_id),
        )

        prefix: list[_QueuedRequest] = []
        work = 0.0
        regret_drop_count = 0
        for queued in deadline_sorted:
            prefix.append(queued)
            work += self._queued_cost_seconds(queued)
            if availability_ts + work > self._deadline_ts(queued):
                longest = max(
                    prefix,
                    key=lambda candidate: (self._queued_cost_seconds(candidate), candidate.sequence_id),
                )
                prefix.remove(longest)
                work -= self._queued_cost_seconds(longest)
                regret_drop_count += 1

        feasible_ids = {queued.sequence_id for queued in prefix}
        prefix_ordered = sorted(
            prefix,
            key=lambda queued: (self._deadline_ts(queued), queued.enqueue_time, queued.sequence_id),
        )
        tail = [queued for queued in waiting_requests if queued.sequence_id not in feasible_ids]
        tail_ordered = sorted(
            tail,
            key=lambda queued: (self._tail_priority(queued, now), queued.enqueue_time, queued.sequence_id),
        )
        ordered_queue = prefix_ordered + tail_ordered

        completion_ts: dict[int, float] = {}
        cursor = availability_ts
        for queued in ordered_queue:
            cursor += self._queued_cost_seconds(queued)
            completion_ts[queued.sequence_id] = cursor

        return _WaitingPlan(
            ordered_queue=ordered_queue,
            feasible_ids=feasible_ids,
            completion_ts=completion_ts,
            regret_drop_count=regret_drop_count,
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

    def _maybe_reorder_waiting_queue(self, new_request: _QueuedRequest, now: float) -> None:
        policy = self._policy_name()
        if policy == "fcfs":
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

        waiting_before = list(self._waiting_queue)[:-1]
        before_plan = self._build_waiting_plan(waiting_before, now)
        after_plan = self._build_waiting_plan(list(self._waiting_queue), now)
        self._waiting_queue = deque(after_plan.ordered_queue)

        attain_before = len(before_plan.feasible_ids)
        attain_after = len(after_plan.feasible_ids)
        damage_count = len(before_plan.feasible_ids - after_plan.feasible_ids)
        self_hit = 1 if new_request.sequence_id in after_plan.feasible_ids else 0
        deadline_ts = self._deadline_ts(new_request)
        completion_ts = after_plan.completion_ts.get(new_request.sequence_id)
        slack_ms = None
        if completion_ts is not None and deadline_ts != inf:
            slack_ms = (deadline_ts - completion_ts) * 1000.0

        new_request.schedule_metrics.update(
            {
                "scheduler_policy": "slo_first",
                "attain_before": attain_before,
                "attain_after": attain_after,
                "self_hit": self_hit,
                "damage_count": damage_count,
                "tail_set_size": len(after_plan.ordered_queue) - len(after_plan.feasible_ids),
                "regret_drop_count": after_plan.regret_drop_count,
                "queue_reorder_count": 1,
                "deadline_slack_ms": slack_ms,
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
        metrics = dict(getattr(output, "metrics", {}) or {})
        queued_metrics = dict(queued_request.schedule_metrics)
        metrics.update(
            {
                "scheduler_policy": self._policy_name(),
                "queue_wait_ms": queue_wait_ms,
                "scheduler_execute_ms": execute_latency_ms,
                "scheduler_latency_ms": queue_wait_ms + execute_latency_ms,
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

    def _normalize_error_output(self, request: OmniDiffusionRequest, error: str, error_code: str) -> DiffusionOutput:
        request_label = self._request_label(request)
        return DiffusionOutput(
            error=error,
            error_code=error_code,
            request_id=request_label,
            metrics={"scheduler_policy": self._policy_name(), "queue_len": len(self._waiting_queue)},
        )

    def _enqueue_request_locked(self, request: OmniDiffusionRequest) -> _QueuedRequest:
        enqueue_time = time.monotonic()
        setattr(request, "arrival_time", getattr(request, "arrival_time", enqueue_time) or enqueue_time)
        if getattr(request, "first_enqueue_time", None) is None:
            setattr(request, "first_enqueue_time", enqueue_time)
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
        self._maybe_reorder_waiting_queue(queued_request, enqueue_time)
        self._log_request_event(
            "QUEUE_ENQUEUE",
            request,
            queue_len=len(self._waiting_queue),
            scheduler_policy=self._policy_name(),
        )
        if enqueue_time == getattr(request, "first_enqueue_time", None):
            self._log_request_event(
                "REQUEST_ARRIVED",
                request,
                queue_len=len(self._waiting_queue),
                scheduler_policy=self._policy_name(),
            )
        self._queue_cv.notify_all()
        return queued_request

    def pop_next_request(self) -> OmniDiffusionRequest | None:
        with self._queue_cv:
            if self._active_request is not None or not self._waiting_queue:
                return None
            queued_request = self._waiting_queue.popleft()
            self._active_request = queued_request
            self._active_started_at = time.monotonic()
            self._set_request_state(queued_request.request, "running")
            dispatch_time = self._active_started_at
            if getattr(queued_request.request, "first_dispatch_time", None) is None:
                setattr(queued_request.request, "first_dispatch_time", dispatch_time)
            setattr(queued_request.request, "last_dispatch_time", dispatch_time)
            self._log_request_event(
                "QUEUE_DEQUEUE",
                queued_request.request,
                queue_len=len(self._waiting_queue),
                scheduler_policy=self._policy_name(),
            )
            self._log_request_event(
                "REQUEST_RESUMED" if getattr(queued_request.request, "last_preempted_time", None) is not None else "REQUEST_STARTED",
                queued_request.request,
                queue_len=len(self._waiting_queue),
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
            self._set_request_state(request, "finished")

    def mark_request_unfinished(self, request: OmniDiffusionRequest) -> None:
        with self._queue_cv:
            self._set_request_state(request, "running")

    def fail_request(self, request: OmniDiffusionRequest) -> None:
        with self._queue_cv:
            for request_id in self._request_ids(request):
                self._aborted_request_ids.discard(request_id)
            self._set_request_state(request, "failed")

    def abort_request(self, request_id: str) -> bool:
        with self._queue_cv:
            for queued_request in list(self._waiting_queue):
                if request_id in self._request_ids(queued_request.request):
                    self._waiting_queue.remove(queued_request)
                    self._aborted_request_ids.add(request_id)
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
                        scheduler_policy=self._policy_name(),
                    )
                    self._log_request_event(
                        "REQUEST_RESUMED" if getattr(request, "last_preempted_time", None) is not None else "REQUEST_STARTED",
                        request,
                        queue_len=len(self._waiting_queue),
                        scheduler_policy=self._policy_name(),
                    )
                    break
                self._queue_cv.wait()

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
                scheduler_policy=self._policy_name(),
            )
            logger.error("REQUEST_FAIL request_id=%s error_code=SCHEDULER_TIMEOUT", request_label)
            raise TimeoutError("Scheduler did not respond in time.") from exc
        finally:
            with self._queue_cv:
                self._active_request = None
                self._active_started_at = None
                self._queue_cv.notify_all()

        execute_latency_ms = (time.monotonic() - execute_start) * 1000
        output = self._annotate_output(output, queued_request, request, queue_wait_ms, execute_latency_ms)

        if output.error:
            self.fail_request(request)
            setattr(request, "failure_time", time.monotonic())
            if output.error_code is None:
                output.error_code = "REQUEST_EXEC_FAILED"
            self._log_request_event(
                "REQUEST_FAILED",
                request,
                queue_len=output.metrics.get("queue_len", -1),
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
            return output

        if not getattr(output, "finished", True):
            self.mark_request_unfinished(request)
            setattr(request, "last_preempted_time", time.monotonic())
            self._log_request_event(
                "REQUEST_PREEMPTED",
                request,
                queue_len=output.metrics.get("queue_len", -1),
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
            return output

        self.finish_request(request)
        setattr(request, "completion_time", time.monotonic())
        self._log_request_event(
            "REQUEST_COMPLETED",
            request,
            queue_len=output.metrics.get("queue_len", -1),
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
        return output
