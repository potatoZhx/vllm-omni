# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import (
    DiffusionRequestStatus,
    RequestScheduler,
    SchedulerInterface,
    StepLevelRequestScheduler,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


def supports_audio_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_input", False))


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.scheduler = self._make_scheduler(scheduler)
        self.scheduler.initialize(od_config)
        self._engine_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._scheduler_cv = threading.Condition(self._engine_lock)
        self._request_events: dict[str, threading.Event] = {}
        self._request_outputs: dict[str, DiffusionOutput] = {}
        self._fatal_error: str | None = None
        self._closed = False
        self._scheduler_thread: threading.Thread | None = None

        if self._uses_step_level_scheduler():
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="DiffusionStepScheduler",
                daemon=True,
            )
            self._scheduler_thread.start()

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def _make_scheduler(self, scheduler: SchedulerInterface | None) -> SchedulerInterface:
        if scheduler is not None:
            return scheduler
        if self.od_config.uses_step_level_scheduler:
            return StepLevelRequestScheduler()
        return RequestScheduler()

    def _get_engine_lock(self) -> threading.Lock:
        return getattr(self, "_engine_lock", getattr(self, "_rpc_lock"))

    def _get_rpc_lock(self) -> threading.Lock:
        return getattr(self, "_rpc_lock", self._get_engine_lock())

    def _uses_step_level_scheduler(self) -> bool:
        return isinstance(self.scheduler, StepLevelRequestScheduler)

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        diffusion_engine_start_time = time.perf_counter()

        # Apply pre-processing if available
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        exec_start_time = time.perf_counter()
        output = self.add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time

        if output.error:
            raise Exception(f"{output.error}")
        logger.info("Generation completed successfully.")

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics={},
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        # When CPU offload is enabled, move output to CPU before
        # post-processing to avoid device OOM — model weights may still
        # reside on the device and leave no headroom for intermediates.
        output_data = output.output
        if (
            self.od_config.enable_cpu_offload
            and isinstance(output_data, torch.Tensor)
            and output_data.device.type != "cpu"
        ):
            output_data = output_data.cpu()

        postprocess_start_time = time.perf_counter()
        outputs = self.post_process_func(output_data) if self.post_process_func is not None else output_data
        audio_payload = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            outputs = outputs.get("video", outputs)
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
        logger.info(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        # Convert to OmniRequestOutput format
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": (time.perf_counter() - diffusion_engine_start_time) * 1000,
            "diffusion_engine_total_time_ms": exec_total_time * 1000,
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }
        if self.pre_process_func is not None:
            metrics["preprocessing_time_ms"] = preprocess_time * 1000

        # Handle single request or multiple requests
        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if supports_audio_output(self.od_config.model_class_name):
                request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": request_audio_payload},
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
            else:
                mm_output = {}
                if audio_payload is not None:
                    mm_output["audio"] = audio_payload
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        custom_output=output.custom_output or {},
                        multimodal_output=mm_output,
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
        else:
            # Multiple requests: return list of OmniRequestOutput
            # Split images based on num_outputs_per_prompt for each request
            results = []
            output_idx = 0

            for i, prompt in enumerate(request.prompts):
                request_id = request.request_ids[i] if i < len(request.request_ids) else ""

                # Get images for this request
                num_outputs = request.sampling_params.num_outputs_per_prompt
                start_idx = output_idx
                end_idx = start_idx + num_outputs
                request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
                output_idx = end_idx

                if supports_audio_output(self.od_config.model_class_name):
                    request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            multimodal_output={"audio": request_audio_payload},
                            final_output_type="audio",
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )
                else:
                    mm_output = {}
                    if audio_payload is not None:
                        sliced_audio = audio_payload
                        if isinstance(audio_payload, (list, tuple)):
                            sliced_audio = audio_payload[start_idx:end_idx]
                            if len(sliced_audio) == 1:
                                sliced_audio = sliced_audio[0]
                        elif hasattr(audio_payload, "shape") and getattr(audio_payload, "shape", None) is not None:
                            if len(audio_payload.shape) > 0 and audio_payload.shape[0] >= end_idx:
                                sliced_audio = audio_payload[start_idx:end_idx]
                                if num_outputs == 1:
                                    sliced_audio = sliced_audio[0]
                        mm_output["audio"] = sliced_audio
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            custom_output=output.custom_output or {},
                            multimodal_output=mm_output,
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )

            return results

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if self._uses_step_level_scheduler():
            return self._add_step_level_request_and_wait_for_response(request)

        engine_lock = self._get_engine_lock()
        with engine_lock:
            target_sched_req_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                # NOTE: add_req_and_wait_for_response() is synchronous, and
                # the scheduler currently enforces _max_batch_size = 1 (see
                # vllm_omni/diffusion/sched/base_scheduler.py), so we directly
                # take the single scheduled request here.
                sched_req_id = sched_output.scheduled_req_ids[0]
                req = sched_output.scheduled_new_reqs[0].req
                try:
                    output = self.executor.add_req(req)
                except Exception as exc:
                    logger.error(
                        "Execution failed for diffusion request %s",
                        sched_req_id,
                        exc_info=True,
                    )
                    output = DiffusionOutput(error=str(exc))

                finished_req_ids = self.scheduler.update_from_output(sched_output, output)
                if target_sched_req_id in finished_req_ids:
                    self.scheduler.pop_request_state(target_sched_req_id)
                    return output

    def _add_step_level_request_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise NotImplementedError(
                "Step-level diffusion scheduling currently supports only single-prompt requests."
            )

        engine_lock = self._get_engine_lock()
        with engine_lock:
            if self._closed:
                raise RuntimeError("DiffusionEngine is closed.")
            if self._fatal_error is not None:
                return DiffusionOutput(error=self._fatal_error)

            target_sched_req_id = self.scheduler.add_request(request)
            event = threading.Event()
            self._request_events[target_sched_req_id] = event
            self._scheduler_cv.notify_all()

        event.wait()

        with engine_lock:
            output = self._request_outputs.pop(target_sched_req_id, None)
            self._request_events.pop(target_sched_req_id, None)
            self.scheduler.pop_request_state(target_sched_req_id)
            if output is None:
                if self._fatal_error is not None:
                    output = DiffusionOutput(error=self._fatal_error)
                else:
                    output = DiffusionOutput(
                        error=f"Step-level diffusion request {target_sched_req_id} completed without an output."
                    )
            return output

    def _scheduler_loop(self) -> None:
        try:
            while True:
                with self._get_engine_lock():
                    while not self._closed and not self.scheduler.has_requests():
                        self._scheduler_cv.wait()

                    if self._closed:
                        return

                    sched_output = self.scheduler.schedule()
                    if sched_output.is_empty:
                        continue

                with self._get_rpc_lock():
                    runner_output = self.executor.execute_stepwise(sched_output)

                with self._get_engine_lock():
                    if self._closed:
                        continue
                    finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
                    self._publish_finished_requests_locked(
                        finished_req_ids,
                        runner_output=runner_output,
                    )
        except Exception as exc:
            logger.error("Step-level diffusion scheduler thread failed", exc_info=True)
            with self._get_engine_lock():
                self._fatal_error = str(exc)
                self._complete_pending_requests_locked(str(exc))
                self._scheduler_cv.notify_all()

    def _publish_finished_requests_locked(
        self,
        finished_req_ids: set[str],
        runner_output: object | None = None,
    ) -> None:
        for sched_req_id in finished_req_ids:
            state = self.scheduler.get_request_state(sched_req_id)
            if state is None:
                continue

            output = self._make_terminal_output(
                sched_req_id,
                state,
                runner_output=runner_output,
            )
            self._request_outputs[sched_req_id] = output
            event = self._request_events.get(sched_req_id)
            if event is not None:
                event.set()

    def _make_terminal_output(
        self,
        sched_req_id: str,
        state,
        runner_output: object | None = None,
    ) -> DiffusionOutput:
        if (
            runner_output is not None
            and getattr(runner_output, "req_id", None) == sched_req_id
            and getattr(runner_output, "finished", False)
        ):
            result = getattr(runner_output, "result", None)
            if isinstance(result, DiffusionOutput):
                return result

        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            return DiffusionOutput(error=f"Diffusion request {sched_req_id} aborted.")
        if state.status == DiffusionRequestStatus.FINISHED_ERROR:
            return DiffusionOutput(error=state.error or f"Diffusion request {sched_req_id} failed.")
        return DiffusionOutput(error=f"Diffusion request {sched_req_id} completed without a terminal output.")

    def _complete_pending_requests_locked(self, error_message: str) -> None:
        pending_req_ids = list(self._request_events)
        for sched_req_id in pending_req_ids:
            state = self.scheduler.get_request_state(sched_req_id)
            if state is not None and not state.is_finished():
                self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ERROR)
            self._request_outputs.setdefault(sched_req_id, DiffusionOutput(error=error_message))
            event = self._request_events.get(sched_req_id)
            if event is not None:
                event.set()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop torch profiling on all diffusion workers.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename (vLLM compat).

        Note:
            Matches vLLM's worker.profile() signature for consistency.
            Traces are saved automatically via on_trace_ready callback.
        """
        if is_start:
            if profile_prefix is None:
                profile_prefix = f"diffusion_{int(time.time())}"
            logger.info(f"Starting diffusion profiling with prefix: {profile_prefix}")
        else:
            logger.info("Stopping diffusion profiling...")

        try:
            self.collective_rpc(method="profile", args=(is_start, profile_prefix))
        except Exception as e:
            action = "start" if is_start else "stop"
            logger.error(f"Failed to {action} profiling on workers", exc_info=True)
            if is_start:
                raise RuntimeError(f"Could not {action} profiler: {e}") from e

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 1024
        width = 1024
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None

        if supports_audio_input(self.od_config.model_class_name):
            audio_sr = 16000
            audio_duration_sec = 4
            audio_array = np.random.randn(audio_sr * audio_duration_sec).astype(np.float32)
            dummy_audio = audio_array[audio_sr * 1 : audio_sr * 3]
        else:
            dummy_audio = None

        prompt: OmniTextPrompt = {
            "prompt": "dummy run",
            "multi_modal_data": {"image": dummy_image, "audio": dummy_audio},
        }
        req = OmniDiffusionRequest(
            prompts=[prompt],
            request_ids=["dummy_req_id"],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                # Disable CFG for warmup to avoid triggering CFG parallel
                # validation when cfg_parallel_size > 1.
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"

        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            rpc_lock = self._get_rpc_lock()
            if deadline is None:
                rpc_lock.acquire()
                acquired = True
            else:
                lock_timeout = max(0, deadline - time.monotonic())
                acquired = rpc_lock.acquire(timeout=lock_timeout)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for rpc lock.")

            rpc_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            if deadline is not None and rpc_timeout <= 0:
                raise TimeoutError(f"RPC call to {method} timed out.")

            return self.executor.collective_rpc(
                method=method,
                timeout=rpc_timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        finally:
            if acquired:
                rpc_lock.release()

    def close(self) -> None:
        scheduler_thread = None
        if hasattr(self, "scheduler"):
            with self._get_engine_lock():
                if getattr(self, "_closed", False):
                    scheduler_thread = getattr(self, "_scheduler_thread", None)
                else:
                    self._closed = True
                    if hasattr(self, "_scheduler_cv"):
                        self._complete_pending_requests_locked("DiffusionEngine closed.")
                        self._scheduler_cv.notify_all()
                    scheduler_thread = getattr(self, "_scheduler_thread", None)

        if (
            scheduler_thread is not None
            and scheduler_thread.is_alive()
            and scheduler_thread is not threading.current_thread()
        ):
            scheduler_thread.join(timeout=30)

        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        with self._get_engine_lock():
            for public_req_id in request_ids:
                sched_req_id = self.scheduler.get_sched_req_id(public_req_id)
                if sched_req_id is None:
                    continue

                state = self.scheduler.get_request_state(sched_req_id)
                if state is None or state.is_finished():
                    continue

                if state.status == DiffusionRequestStatus.RUNNING:
                    self.scheduler.mark_abort_pending(sched_req_id)
                    continue

                self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
                self._request_outputs[sched_req_id] = DiffusionOutput(
                    error=f"Diffusion request {public_req_id} aborted."
                )
                event = self._request_events.get(sched_req_id)
                if event is not None:
                    event.set()

            if hasattr(self, "_scheduler_cv"):
                self._scheduler_cv.notify_all()
