#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline profiling worker for one parallel config.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-types-config", required=True)
    parser.add_argument("--output-json", required=True)

    parser.add_argument("--parallel-name", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--ulysses-degree", type=int, required=True)
    parser.add_argument("--ring-degree", type=int, required=True)
    parser.add_argument("--cfg-parallel-size", type=int, required=True)
    parser.add_argument("--vae-patch-parallel-size", type=int, required=True)
    parser.add_argument("--use-hsdp", action="store_true")
    parser.add_argument("--hsdp-shard-size", type=int, default=-1)
    parser.add_argument("--hsdp-replicate-size", type=int, default=1)

    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--prompt", default="A cat walking on a beach at sunset.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.875)
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument(
        "--num-weight-load-threads",
        type=int,
        default=8,
        help="Number of threads used for model weight loading.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Per-request timeout (includes warmup requests). 0 means disabled. "
            "If a request exceeds this timeout, the worker process group is killed."
        ),
    )
    parser.add_argument(
        "--warmup-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Warmup-specific timeout. 0 means disabled. "
            "If >0, it overrides request-timeout for warmup requests."
        ),
    )
    parser.add_argument(
        "--timeout-grace-seconds",
        type=int,
        default=15,
        help=(
            "Grace period after SIGTERM before SIGKILL when timeout happens."
        ),
    )
    parser.add_argument("--request-fail-fast", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output-json by skipping completed OK entries.",
    )
    return parser.parse_args()


def load_request_types(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("request_profiles"), list):
        reqs = []
        for profile in payload["request_profiles"]:
            if not isinstance(profile, dict):
                continue
            fps = profile.get("fps", None)
            fps_suffix = f"_fps{int(fps)}" if fps is not None else ""
            reqs.append(
                {
                    "height": int(profile["height"]),
                    "width": int(profile["width"]),
                    "num_frames": int(profile["num_frames"]),
                    "num_inference_steps": int(profile["num_inference_steps"]),
                    "fps": int(fps) if fps is not None else None,
                    "request_type_id": (
                        f"h{int(profile['height'])}_w{int(profile['width'])}_"
                        f"f{int(profile['num_frames'])}_s{int(profile['num_inference_steps'])}{fps_suffix}"
                    ),
                }
            )
        return reqs

    reqs = []
    for (height, width), num_frames, num_steps in itertools.product(
        payload["height_width"],
        payload["num_frames"],
        payload["num_inference_steps"],
    ):
        reqs.append(
            {
                "height": int(height),
                "width": int(width),
                "num_frames": int(num_frames),
                "num_inference_steps": int(num_steps),
                "fps": None,
                "request_type_id": f"h{int(height)}_w{int(width)}_f{int(num_frames)}_s{int(num_steps)}",
            }
        )
    return reqs


def run_one(
    omni: Omni,
    req: dict,
    seed: int,
    guidance_scale: float,
    prompt: str,
    negative_prompt: str,
) -> float:
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    start = time.perf_counter()
    _ = omni.generate(
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=req["height"],
            width=req["width"],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=req["num_inference_steps"],
            num_frames=req["num_frames"],
            fps=req.get("fps"),
        ),
    )
    end = time.perf_counter()
    return end - start


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(row["request_type_id"], []).append(float(row["latency_seconds"]))

    out = []
    for request_type_id, values in sorted(grouped.items()):
        arr = np.array(values, dtype=np.float64)
        out.append(
            {
                "request_type_id": request_type_id,
                "count": int(arr.size),
                "latency_seconds_mean": float(arr.mean()),
                "latency_seconds_median": float(np.median(arr)),
                "latency_seconds_p50": float(np.percentile(arr, 50)),
                "latency_seconds_p90": float(np.percentile(arr, 90)),
                "latency_seconds_p99": float(np.percentile(arr, 99)),
                "latency_ms_mean": float(arr.mean() * 1000.0),
            }
        )
    return out


def _row_key(row: dict) -> tuple[str, int]:
    return (str(row.get("request_type_id", "")), int(row.get("repeat_id", 0)))


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results", [])
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
        return []
    except Exception:
        return []


def _serialize_rows(rows_by_key: dict[tuple[str, int], dict], order_map: dict[str, int]) -> list[dict]:
    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (order_map.get(str(r.get("request_type_id", "")), 10**9), int(r.get("repeat_id", 0))))
    return rows


def _save_payload(
    out_path: Path,
    args: argparse.Namespace,
    rows_by_key: dict[tuple[str, int], dict],
    order_map: dict[str, int],
    planned_total: int,
    warmup_done: bool,
    warmup_stats: dict,
    model_load_stats: dict,
) -> None:
    rows = _serialize_rows(rows_by_key, order_map)
    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    failed_count = sum(1 for r in rows if r.get("status") == "failed")
    timeout_count = sum(1 for r in rows if r.get("status") == "timeout")
    payload = {
        "model": args.model,
        "parallel_name": args.parallel_name,
        "num_gpus": args.num_gpus,
        "repeats": args.repeats,
        "warmup_iters": args.warmup_iters,
        "planned_total_runs": planned_total,
        "completed_entries": len(rows),
        "completed_ok": ok_count,
        "completed_failed": failed_count,
        "completed_timeout": timeout_count,
        "warmup_done": warmup_done,
        "warmup_stats": warmup_stats,
        "model_load_stats": model_load_stats,
        "results": rows,
        "summary_by_request_type": summarize(rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def main() -> None:
    args = parse_args()
    def _pids_in_group(pgid: int) -> list[int]:
        pids: list[int] = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                if os.getpgid(pid) == pgid:
                    pids.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
        return pids

    def _terminate_group_with_grace() -> None:
        pgid = os.getpgrp()
        self_pid = os.getpid()
        grace = max(0, int(args.timeout_grace_seconds))

        members = [pid for pid in _pids_in_group(pgid) if pid != self_pid]
        for pid in members:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            alive = []
            for pid in members:
                try:
                    os.kill(pid, 0)
                    alive.append(pid)
                except ProcessLookupError:
                    continue
            if not alive:
                break
            time.sleep(0.5)

        remaining = []
        for pid in members:
            try:
                os.kill(pid, 0)
                remaining.append(pid)
            except ProcessLookupError:
                continue

        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


    request_types = load_request_types(Path(args.request_types_config))
    if not request_types:
        raise ValueError("No request types found.")

    out_path = Path(args.output_json)
    order_map = {req["request_type_id"]: idx for idx, req in enumerate(request_types)}
    planned_total = len(request_types) * args.repeats

    rows_by_key: dict[tuple[str, int], dict] = {}
    warmup_done = False
    warmup_stats: dict[str, object] = {
        "request_type_id": "",
        "planned_iters": int(args.warmup_iters),
        "completed_iters": 0,
        "latency_seconds_total": 0.0,
        "latency_seconds_mean": 0.0,
        "latency_seconds": [],
    }
    model_load_stats: dict[str, object] = {
        "status": "not_started",
        "latency_seconds": 0.0,
        "latency_ms": 0.0,
        "error": "",
    }
    if args.resume:
        existing_rows = _load_existing_rows(out_path)
        for row in existing_rows:
            try:
                rows_by_key[_row_key(row)] = row
            except Exception:
                continue
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                warmup_done = bool(payload.get("warmup_done", False))
                loaded_warmup_stats = payload.get("warmup_stats", None)
                if isinstance(loaded_warmup_stats, dict):
                    warmup_stats = {
                        "request_type_id": str(loaded_warmup_stats.get("request_type_id", "")),
                        "planned_iters": int(loaded_warmup_stats.get("planned_iters", args.warmup_iters)),
                        "completed_iters": int(loaded_warmup_stats.get("completed_iters", 0)),
                        "latency_seconds_total": float(loaded_warmup_stats.get("latency_seconds_total", 0.0)),
                        "latency_seconds_mean": float(loaded_warmup_stats.get("latency_seconds_mean", 0.0)),
                        "latency_seconds": [
                            float(x)
                            for x in loaded_warmup_stats.get("latency_seconds", [])
                        ],
                    }
                loaded_model_stats = payload.get("model_load_stats", None)
                if isinstance(loaded_model_stats, dict):
                    model_load_stats = {
                        "status": str(loaded_model_stats.get("status", "not_started")),
                        "latency_seconds": float(loaded_model_stats.get("latency_seconds", 0.0)),
                        "latency_ms": float(loaded_model_stats.get("latency_ms", 0.0)),
                        "error": str(loaded_model_stats.get("error", "")),
                    }
            except Exception:
                warmup_done = False
        print(f"[Worker] Resume enabled. Existing entries: {len(rows_by_key)}")

    completed_ok_keys = {
        key for key, row in rows_by_key.items() if row.get("status") == "ok"
    }

    request_watch_state: dict[str, object] = {
        "active": False,
        "label": "",
        "start": 0.0,
        "meta": {},
    }
    request_watch_lock = threading.Lock()
    save_lock = threading.Lock()

    def mark_request_start(label: str, meta: dict[str, object] | None = None) -> None:
        with request_watch_lock:
            request_watch_state["active"] = True
            request_watch_state["label"] = label
            request_watch_state["start"] = time.monotonic()
            request_watch_state["meta"] = dict(meta or {})

    def mark_request_end() -> None:
        with request_watch_lock:
            request_watch_state["active"] = False
            request_watch_state["label"] = ""
            request_watch_state["start"] = 0.0
            request_watch_state["meta"] = {}

    def _watchdog_loop() -> None:
        if args.request_timeout_seconds <= 0 and args.warmup_timeout_seconds <= 0:
            return
        while True:
            time.sleep(2)
            with request_watch_lock:
                active = bool(request_watch_state["active"])
                label = str(request_watch_state["label"])
                start = float(request_watch_state["start"])
                meta = dict(request_watch_state.get("meta", {}))
            if not active:
                continue
            timeout_s = args.request_timeout_seconds
            if label.startswith("warmup_") and args.warmup_timeout_seconds > 0:
                timeout_s = args.warmup_timeout_seconds
            if timeout_s <= 0:
                continue
            elapsed = time.monotonic() - start
            if elapsed > timeout_s:
                try:
                    if bool(meta.get("is_warmup", False)):
                        warmup_stats["request_type_id"] = str(meta.get("request_type_id", ""))
                        warmup_stats["planned_iters"] = int(args.warmup_iters)
                        warmup_stats["status"] = "timeout"
                        warmup_stats["timeout_seconds"] = float(timeout_s)
                        warmup_stats["timeout_elapsed_seconds"] = float(elapsed)
                    else:
                        request_type_id = str(meta.get("request_type_id", ""))
                        repeat_id = int(meta.get("repeat_id", 0))
                        key = (request_type_id, repeat_id)
                        timeout_row = {
                            "parallel_name": args.parallel_name,
                            "num_gpus": args.num_gpus,
                            "tensor_parallel_size": args.tensor_parallel_size,
                            "ulysses_degree": args.ulysses_degree,
                            "ring_degree": args.ring_degree,
                            "cfg_parallel_size": args.cfg_parallel_size,
                            "vae_patch_parallel_size": args.vae_patch_parallel_size,
                            "request_type_id": request_type_id,
                            "height": int(meta.get("height", 0)),
                            "width": int(meta.get("width", 0)),
                            "num_frames": int(meta.get("num_frames", 0)),
                            "num_inference_steps": int(meta.get("num_inference_steps", 0)),
                            "fps": int(meta["fps"]) if meta.get("fps") is not None else "",
                            "repeat_id": repeat_id,
                            "seed": int(meta.get("seed", 0)),
                            "latency_seconds": float(elapsed),
                            "latency_ms": float(elapsed * 1000.0),
                            "status": "timeout",
                            "error": (
                                f"request timeout after {timeout_s}s "
                                f"(elapsed={elapsed:.4f}s)"
                            ),
                        }
                        rows_by_key[key] = timeout_row

                    with save_lock:
                        _save_payload(
                            out_path,
                            args,
                            rows_by_key,
                            order_map,
                            planned_total,
                            warmup_done,
                            warmup_stats,
                            model_load_stats,
                        )
                except Exception as persist_exc:
                    print(f"[Worker][Timeout] Failed to persist timeout result: {persist_exc}", flush=True)

                print(
                    f"[Worker][Timeout] Request '{label}' exceeded {timeout_s}s "
                    f"(elapsed={elapsed:.1f}s). Sending SIGTERM then SIGKILL (grace={args.timeout_grace_seconds}s).",
                    flush=True,
                )
                try:
                    _terminate_group_with_grace()
                except Exception:
                    pass
                os._exit(124)

    if args.request_timeout_seconds > 0 or args.warmup_timeout_seconds > 0:
        watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        watchdog_thread.start()

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    print("[Worker] Loading Omni model...")
    model_load_start = time.perf_counter()
    try:
        omni = Omni(
            model=args.model,
            parallel_config=parallel_config,
            vae_use_slicing=args.vae_use_slicing,
            vae_use_tiling=args.vae_use_tiling,
            boundary_ratio=args.boundary_ratio,
            flow_shift=args.flow_shift,
            enable_cpu_offload=args.enable_cpu_offload,
            enable_layerwise_offload=args.enable_layerwise_offload,
            enforce_eager=args.enforce_eager,
            num_weight_load_threads=args.num_weight_load_threads,
        )
        model_load_elapsed_s = time.perf_counter() - model_load_start
        model_load_stats = {
            "status": "ok",
            "latency_seconds": float(model_load_elapsed_s),
            "latency_ms": float(model_load_elapsed_s * 1000.0),
            "error": "",
        }
        print(f"[Worker] Model load done: {model_load_elapsed_s:.4f}s")
    except Exception as exc:
        model_load_elapsed_s = time.perf_counter() - model_load_start
        model_load_stats = {
            "status": "failed",
            "latency_seconds": float(model_load_elapsed_s),
            "latency_ms": float(model_load_elapsed_s * 1000.0),
            "error": str(exc),
        }
        _save_payload(
            out_path,
            args,
            rows_by_key,
            order_map,
            planned_total,
            warmup_done,
            warmup_stats,
            model_load_stats,
        )
        raise

    if not warmup_done:
        warmup_req = request_types[0]
        print(f"[Worker] Warmup start: iters={args.warmup_iters}, request={warmup_req['request_type_id']}")
        warmup_latencies: list[float] = []
        for warmup_idx in range(args.warmup_iters):
            warmup_label = f"warmup_{warmup_req['request_type_id']}_iter{warmup_idx + 1}"
            mark_request_start(
                warmup_label,
                {
                    "is_warmup": True,
                    "request_type_id": warmup_req["request_type_id"],
                },
            )
            try:
                warmup_latency_s = run_one(
                    omni=omni,
                    req=warmup_req,
                    seed=args.seed + warmup_idx,
                    guidance_scale=args.guidance_scale,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                )
                warmup_latencies.append(float(warmup_latency_s))
            finally:
                mark_request_end()

        warmup_total = float(sum(warmup_latencies))
        warmup_count = len(warmup_latencies)
        warmup_stats = {
            "request_type_id": str(warmup_req["request_type_id"]),
            "planned_iters": int(args.warmup_iters),
            "completed_iters": int(warmup_count),
            "latency_seconds_total": warmup_total,
            "latency_seconds_mean": (warmup_total / warmup_count) if warmup_count > 0 else 0.0,
            "latency_seconds": warmup_latencies,
        }
        warmup_done = True
        _save_payload(
            out_path,
            args,
            rows_by_key,
            order_map,
            planned_total,
            warmup_done,
            warmup_stats,
            model_load_stats,
        )
        print("[Worker] Warmup done")
    else:
        print("[Worker] Warmup skipped due to resume state.")

    total = planned_total
    done_ok = len(completed_ok_keys)
    print(f"[Worker] Completed OK before run: {done_ok}/{total}")

    for req_idx, req in enumerate(request_types):
        for repeat_id in range(1, args.repeats + 1):
            key = (req["request_type_id"], repeat_id)
            if key in completed_ok_keys:
                continue

            global_ord = req_idx * args.repeats + repeat_id
            seed = args.seed + global_ord

            try:
                req_label = f"{req['request_type_id']}_r{repeat_id}"
                mark_request_start(
                    req_label,
                    {
                        "is_warmup": False,
                        "request_type_id": req["request_type_id"],
                        "repeat_id": repeat_id,
                        "seed": seed,
                        "height": req["height"],
                        "width": req["width"],
                        "num_frames": req["num_frames"],
                        "num_inference_steps": req["num_inference_steps"],
                        "fps": req.get("fps"),
                    },
                )
                try:
                    latency_s = run_one(
                        omni=omni,
                        req=req,
                        seed=seed,
                        guidance_scale=args.guidance_scale,
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                    )
                finally:
                    mark_request_end()
                row = {
                    "parallel_name": args.parallel_name,
                    "num_gpus": args.num_gpus,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "ulysses_degree": args.ulysses_degree,
                    "ring_degree": args.ring_degree,
                    "cfg_parallel_size": args.cfg_parallel_size,
                    "vae_patch_parallel_size": args.vae_patch_parallel_size,
                    "request_type_id": req["request_type_id"],
                    "height": req["height"],
                    "width": req["width"],
                    "num_frames": req["num_frames"],
                    "num_inference_steps": req["num_inference_steps"],
                    "fps": req.get("fps", ""),
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "latency_seconds": latency_s,
                    "latency_ms": latency_s * 1000.0,
                    "status": "ok",
                    "error": "",
                }
                rows_by_key[key] = row
                completed_ok_keys.add(key)
                print(f"[Worker] [{len(completed_ok_keys)}/{total}] {req['request_type_id']} r{repeat_id} latency={latency_s:.4f}s")
            except Exception as exc:
                row = {
                    "parallel_name": args.parallel_name,
                    "num_gpus": args.num_gpus,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "ulysses_degree": args.ulysses_degree,
                    "ring_degree": args.ring_degree,
                    "cfg_parallel_size": args.cfg_parallel_size,
                    "vae_patch_parallel_size": args.vae_patch_parallel_size,
                    "request_type_id": req["request_type_id"],
                    "height": req["height"],
                    "width": req["width"],
                    "num_frames": req["num_frames"],
                    "num_inference_steps": req["num_inference_steps"],
                    "fps": req.get("fps", ""),
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "latency_seconds": "",
                    "latency_ms": "",
                    "status": "failed",
                    "error": str(exc),
                }
                rows_by_key[key] = row
                print(f"[Worker][Error] {req['request_type_id']} r{repeat_id}: {exc}")
                with save_lock:
                    _save_payload(
                        out_path,
                        args,
                        rows_by_key,
                        order_map,
                        planned_total,
                        warmup_done,
                        warmup_stats,
                        model_load_stats,
                    )
                if args.request_fail_fast:
                    raise
                continue

            with save_lock:
                _save_payload(
                    out_path,
                    args,
                    rows_by_key,
                    order_map,
                    planned_total,
                    warmup_done,
                    warmup_stats,
                    model_load_stats,
                )

    print(f"[Worker] Saved: {out_path}")


if __name__ == "__main__":
    main()
