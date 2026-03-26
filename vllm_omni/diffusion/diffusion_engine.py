# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections.abc import Iterable
from typing import Any

import PIL.Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.runtime_profile import RuntimeProfileEstimator
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


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

    def __init__(self, od_config: OmniDiffusionConfig):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        self.runtime_estimator = RuntimeProfileEstimator.from_path(
            getattr(od_config, "instance_runtime_profile_path", None),
            instance_type=getattr(od_config, "instance_runtime_profile_name", None),
        )

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        # Apply pre-processing if available
        scheduler_metrics: dict[str, Any] = {}
        if self.pre_process_func is not None:
            preprocess_start_time = time.time()
            request = self.pre_process_func(request)
            preprocess_time = time.time() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        aggregated_chunk_metrics: dict[str, Any] = {}
        output = None
        while True:
            if getattr(self.od_config, "diffusion_enable_step_chunk", False):
                request.max_steps_this_turn = self._plan_chunk_budget(request)
                total_steps = max(int(getattr(request.sampling_params, "num_inference_steps", 1) or 1), 1)
                width = int(getattr(request.sampling_params, "width", None) or getattr(request.sampling_params, "resolution", 1024) or 1024)
                height = int(
                    getattr(request.sampling_params, "height", None) or getattr(request.sampling_params, "resolution", 1024) or 1024
                )
                estimated_remaining_latency_ms = self._estimate_remaining_runtime_s(request) * 1000.0
                logger.info(
                    "STEP_CHUNK_PLAN request_id=%s width=%d height=%d total_steps=%d executed_steps=%d remaining_steps=%d chunk_budget_steps=%d estimated_remaining_latency_ms=%.2f preemption_enabled=%s",
                    ",".join(getattr(request, "request_ids", []) or []) or "<missing-request-id>",
                    width,
                    height,
                    total_steps,
                    int(getattr(request, "executed_steps", 0) or 0),
                    max(total_steps - int(getattr(request, "executed_steps", 0) or 0), 0),
                    int(request.max_steps_this_turn or 0),
                    estimated_remaining_latency_ms,
                    bool(getattr(self.od_config, "diffusion_enable_chunk_preemption", False)),
                )

            output = self.add_req_and_wait_for_response(request)
            scheduler_metrics = dict(getattr(output, "metrics", {}) or {})
            self._accumulate_chunk_metrics(aggregated_chunk_metrics, scheduler_metrics)

            if output.error:
                error_code = getattr(output, "error_code", None) or "REQUEST_EXEC_FAILED"
                request_label = getattr(output, "request_id", None) or ",".join(getattr(request, "request_ids", []) or [])
                raise RuntimeError(f"[{error_code}] request_id={request_label} {output.error}")

            if getattr(output, "finished", True):
                output.metrics = {**aggregated_chunk_metrics, **scheduler_metrics}
                logger.info("Generation completed successfully.")
                break

            if not getattr(self.od_config, "diffusion_enable_step_chunk", False):
                logger.info(
                    "Generation unfinished for request_id=%s executed_steps=%s",
                    getattr(output, "request_id", None) or ",".join(getattr(request, "request_ids", []) or []),
                    scheduler_metrics.get("executed_steps"),
                )
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                        images=[],
                        prompt=prompt,
                        metrics={**aggregated_chunk_metrics, **scheduler_metrics, "unfinished": True},
                        latents=None,
                    )
                    for i, prompt in enumerate(request.prompts)
                ]

        assert output is not None
        scheduler_metrics = dict(getattr(output, "metrics", {}) or {})
        request.max_steps_this_turn = None

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics=scheduler_metrics,
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        postprocess_start_time = time.time()
        outputs = self.post_process_func(output.output) if self.post_process_func is not None else output.output
        postprocess_time = time.time() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        # Convert to OmniRequestOutput format
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }
        metrics.update(scheduler_metrics)
        if self.pre_process_func is not None:
            metrics["preprocessing_time_ms"] = preprocess_time * 1000

        # Handle single request or multiple requests
        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if supports_audio_output(self.od_config.model_class_name):
                audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        multimodal_output={"audio": audio_payload},
                        final_output_type="audio",
                    ),
                ]
            else:
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
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
                request_outputs = outputs[output_idx : output_idx + num_outputs] if output_idx < len(outputs) else []
                output_idx += num_outputs

                if supports_audio_output(self.od_config.model_class_name):
                    audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            multimodal_output={"audio": audio_payload},
                            final_output_type="audio",
                        )
                    )
                else:
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                        )
                    )

            return results

    def _plan_chunk_budget(self, request: OmniDiffusionRequest) -> int:
        total_steps = max(int(getattr(request.sampling_params, "num_inference_steps", 1) or 1), 1)
        executed_steps = max(int(getattr(request, "executed_steps", 0) or 0), 0)
        remaining_steps = max(total_steps - executed_steps, 0)
        if remaining_steps == 0:
            return 1

        if self._is_small_request_by_runtime(request):
            return remaining_steps

        if not getattr(self.od_config, "diffusion_enable_chunk_preemption", False):
            return remaining_steps

        return min(remaining_steps, self._chunk_budget_steps_for_request(request))

    def _is_small_request_by_runtime(self, request: OmniDiffusionRequest) -> bool:
        threshold_ms = getattr(self.od_config, "diffusion_small_request_latency_threshold_ms", None)
        if threshold_ms is None:
            return False
        estimated_runtime_s = self._estimate_remaining_runtime_s(request)
        return (estimated_runtime_s * 1000.0) <= float(threshold_ms)

    def _estimate_remaining_runtime_s(self, request: OmniDiffusionRequest) -> float:
        sampling_params = request.sampling_params
        width = int(getattr(sampling_params, "width", None) or getattr(sampling_params, "resolution", 1024) or 1024)
        height = int(getattr(sampling_params, "height", None) or getattr(sampling_params, "resolution", 1024) or 1024)
        total_steps = max(int(getattr(sampling_params, "num_inference_steps", 1) or 1), 1)
        executed_steps = max(int(getattr(request, "executed_steps", 0) or 0), 0)
        remaining_steps = max(total_steps - executed_steps, 1)
        num_frames = max(int(getattr(sampling_params, "num_frames", 1) or 1), 1)
        task_type = "video" if num_frames > 1 else "image"
        fallback_s = max(float(remaining_steps * num_frames), 0.001)
        runtime_estimator = getattr(self, "runtime_estimator", None) or RuntimeProfileEstimator()
        return runtime_estimator.estimate_runtime_s(
            task_type=task_type,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=remaining_steps,
            fallback_s=fallback_s,
        )

    def _chunk_budget_steps_for_request(self, request: OmniDiffusionRequest) -> int:
        sampling_params = request.sampling_params
        num_frames = max(int(getattr(sampling_params, "num_frames", 1) or 1), 1)
        if num_frames > 1:
            budget_steps = getattr(self.od_config, "diffusion_video_chunk_budget_steps", None)
        else:
            budget_steps = getattr(self.od_config, "diffusion_image_chunk_budget_steps", None)

        if budget_steps is None:
            budget_steps = getattr(self.od_config, "diffusion_chunk_budget_steps", 4)
        return max(int(budget_steps or 4), 1)

    @staticmethod
    def _accumulate_chunk_metrics(aggregate: dict[str, Any], chunk_metrics: dict[str, Any]) -> None:
        aggregate["chunk_count"] = int(aggregate.get("chunk_count", 0)) + 1
        for key in ("queue_wait_ms", "scheduler_execute_ms", "scheduler_latency_ms"):
            aggregate[key] = float(aggregate.get(key, 0.0)) + float(chunk_metrics.get(key, 0.0) or 0.0)
        for key in (
            "scheduler_policy",
            "dispatch_epoch",
            "executed_steps",
            "remaining_steps",
            "chunk_budget_steps",
            "queue_len",
        ):
            if key in chunk_metrics:
                aggregate[key] = chunk_metrics[key]

    @staticmethod
    def make_engine(config: OmniDiffusionConfig) -> "DiffusionEngine":
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest):
        return self.executor.add_req(request)

    def estimate_waiting_queue_len(self) -> int:
        scheduler = getattr(self.executor, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "estimate_waiting_queue_len"):
            return 0
        return scheduler.estimate_waiting_queue_len()

    def estimate_scheduler_load(self) -> dict[str, int]:
        scheduler = getattr(self.executor, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "estimate_scheduler_load"):
            return {
                "waiting_queue_len": 0,
                "active_request_count": 0,
                "paused_context_count": 0,
            }
        return scheduler.estimate_scheduler_load()

    def start_profile(self, trace_filename: str | None = None) -> None:
        """
        Start torch profiling on all diffusion workers.

        Creates a directory (if needed) and sets up a base filename template
        for per-rank profiler traces (typically saved as <template>_rank<N>.json).

        Args:
            trace_filename: Optional base filename (without extension or rank suffix).
                            If None, generates one using current timestamp.
        """
        if trace_filename is None:
            trace_filename = f"stage_0_diffusion_{int(time.time())}_rank"

        trace_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR", "./profiles")

        # Expand ~ and ~user, then make absolute (robust against cwd changes)
        trace_dir = os.path.expanduser(trace_dir)
        trace_dir = os.path.abspath(trace_dir)

        try:
            os.makedirs(trace_dir, exist_ok=True)
        except OSError as exc:
            logger.error(f"Failed to create profiler directory {trace_dir}: {exc}")
            raise

        # Build final template path (without rank or extension — torch.profiler appends those)
        full_template = os.path.join(trace_dir, trace_filename)

        expected_pattern = f"{full_template}*.json"
        logger.info(f"Starting diffusion profiling → {expected_pattern}")

        # Also log the absolute directory once (useful in multi-node or containers)
        logger.debug(f"Profiler output directory: {trace_dir}")

        # Propagate to all workers
        try:
            self.collective_rpc(method="start_profile", args=(full_template,))
        except Exception as e:
            logger.error("Failed to start profiling on workers", exc_info=True)
            raise RuntimeError(f"Could not start profiler: {e}") from e

    def stop_profile(self) -> dict:
        """
        Stop profiling on all workers and collect the final trace/table paths.

        The worker (torch_profiler.py) now handles trace export, compression to .gz,
        and deletion of the original .json file. This method only collects and
        reports the paths returned by the workers.

        Returns:
            dict with keys:
            - "traces": list of final trace file paths (usually .json.gz)
            - "tables": list of table strings (one per rank)
        """
        logger.info("Stopping diffusion profiling and collecting results...")

        try:
            # Give worker enough time — export + compression + table can be slow
            results = self.collective_rpc(method="stop_profile", timeout=600)
        except Exception:
            logger.error("Failed to stop profiling on workers", exc_info=True)
            return {"traces": [], "tables": []}

        output_files = {"traces": [], "tables": []}
        successful_traces = 0

        if not results:
            logger.warning("No profiling results returned from any rank")
            return output_files

        for rank, res in enumerate(results):
            if not isinstance(res, dict):
                logger.warning(f"Rank {rank}: invalid result format (got {type(res)})")
                continue

            # 1. Trace file — should be .json.gz if compression succeeded
            trace_path = res.get("trace")
            if trace_path:
                # We trust the worker — it created/compressed the file
                logger.info(f"[Rank {rank}] Final trace: {trace_path}")
                output_files["traces"].append(trace_path)
                successful_traces += 1

                # Optional: warn if path looks suspicious (e.g. still .json)
                if not trace_path.endswith((".json.gz", ".json")):
                    logger.warning(f"Rank {rank}: unusual trace path extension: {trace_path}")

            # 2. Table file — plain text
            table = res.get("table")
            if table:
                output_files["tables"].append(table)

        # Final summary logging
        num_ranks = len(results)
        if successful_traces > 0:
            final_paths_str = ", ".join(output_files["traces"][:3])
            if len(output_files["traces"]) > 3:
                final_paths_str += f" ... (+{len(output_files['traces']) - 3} more)"

            logger.info(
                f"Profiling stopped. Collected {successful_traces} trace file(s) "
                f"from {num_ranks} rank(s). "
                f"Final trace paths: {final_paths_str}"
            )
        elif output_files["traces"]:
            logger.info(
                f"Profiling stopped but no traces were successfully collected. "
                f"Reported paths: {', '.join(output_files['traces'][:3])}"
                f"{' ...' if len(output_files['traces']) > 3 else ''}"
            )
        else:
            logger.info("Profiling stopped — no trace files were collected from any rank.")

        if output_files["tables"]:
            logger.debug(f"Collected {len(output_files['tables'])} profiling table(s)")

        return output_files

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
        prompt: OmniTextPrompt = {"prompt": "dummy run", "multi_modal_data": {"image": dummy_image}}
        req = OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        self.add_req_and_wait_for_response(request)

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
        return self.executor.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )

    def close(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        scheduler = getattr(self.executor, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "abort_request"):
            logger.warning("DiffusionEngine abort is not available on current executor")
            return

        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in request_ids:
            aborted = scheduler.abort_request(req_id)
            if not aborted:
                logger.info("REQUEST_ABORT request_id=%s status=not_found", req_id)
