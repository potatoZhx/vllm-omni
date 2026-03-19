#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# python profile/videoGen/run_video_parallel_bench.py --request-types-config /home/mumura/omni/vllm-omni/profile/videoGen/test_req.json   --card-counts 1,2,4,8 --gpu-device-ids 0,1,2,3,4,5,6,7 --warmup-iters 1 --repeats 1 --request-timeout-seconds 200 --warmup-timeout-seconds 300 --timeout-grace-seconds 20 
# python profile/videoGen/run_video_parallel_bench_random.py  --card-counts 4,8 --gpu-device-ids 0,1,2,3,4,5,6,7 --warmup-iters 1 --repeats 2 --request-timeout-seconds 450 --warmup-timeout-seconds 600 --timeout-grace-seconds 20 


DEFAULT_MODEL = "/data2/group_谈海生/mumura/models/Wan2.2-T2V-A14B-Diffusers"


@dataclass
class RequestType:
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    fps: int | None = None

    @property
    def request_type_id(self) -> str:
        fps_suffix = f"_fps{self.fps}" if self.fps is not None else ""
        return f"h{self.height}_w{self.width}_f{self.num_frames}_s{self.num_inference_steps}{fps_suffix}"


@dataclass
class ParallelConfig:
    num_gpus: int
    name: str
    vae_patch_parallel_size: int
    tensor_parallel_size: int
    ulysses_degree: int
    ring_degree: int
    cfg_parallel_size: int
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_degree * self.ring_degree

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.sp_size * self.cfg_parallel_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline video generation latency profiling across 1/2/4/8 GPU "
            "parallel configurations and request types from random dataset profiles."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--request-types-config",
        default="profile/videoGen/random_dataset_c_request_profiles.json",
    )
    parser.add_argument(
        "--parallel-matrix-config",
        default="profile/videoGen/parallel_matrix.json",
    )
    parser.add_argument(
        "--worker-script",
        default="profile/videoGen/offline_profile_worker.py",
    )
    parser.add_argument(
        "--output-root",
        default="profile/videoGen/results",
    )
    parser.add_argument(
        "--gpu-device-ids",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated physical GPU IDs. First N ids will be used for N-gpu runs.",
    )
    parser.add_argument(
        "--card-counts",
        default="1,2,4,8",
        help="Comma-separated gpu counts to run, e.g. 2,4,8",
    )
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
            "Per-request timeout enforced inside worker. 0 means disabled. "
            "If one request hangs, worker kills itself quickly."
        ),
    )
    parser.add_argument(
        "--warmup-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Warmup-specific timeout in worker. 0 means disabled. "
            "If >0, it overrides request-timeout for warmup requests."
        ),
    )
    parser.add_argument(
        "--timeout-grace-seconds",
        type=int,
        default=15,
        help=(
            "Grace period after SIGTERM before SIGKILL when worker request timeout happens."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume worker from existing worker_results.json for each config.",
    )
    parser.add_argument(
        "--request-fail-fast",
        action="store_true",
        help="Fail worker immediately when a single request fails.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=0,
        help=(
            "Timeout for each parallel-config worker process. 0 means no timeout. "
            "When timeout is reached, worker process group will be force-killed."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
    )
    return parser.parse_args()


def load_request_types(path: Path) -> list[RequestType]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_types: list[RequestType] = []

    # RandomDataset-style request profiles: keep exact tuples without Cartesian product.
    if isinstance(payload, dict) and isinstance(payload.get("request_profiles"), list):
        for profile in payload["request_profiles"]:
            if not isinstance(profile, dict):
                continue
            request_types.append(
                RequestType(
                    height=int(profile["height"]),
                    width=int(profile["width"]),
                    num_frames=int(profile["num_frames"]),
                    num_inference_steps=int(profile["num_inference_steps"]),
                    fps=int(profile["fps"]) if profile.get("fps") is not None else None,
                )
            )
        if not request_types:
            raise ValueError(f"No valid request_profiles found in {path}")
        return request_types

    # Legacy request types format: expand Cartesian product.
    height_width = payload["height_width"]
    num_frames = payload["num_frames"]
    num_steps = payload["num_inference_steps"]
    for height, width in height_width:
        for frames in num_frames:
            for steps in num_steps:
                request_types.append(
                    RequestType(
                        height=int(height),
                        width=int(width),
                        num_frames=int(frames),
                        num_inference_steps=int(steps),
                    )
                )
    return request_types


def load_parallel_configs(path: Path, selected_cards: set[int]) -> list[ParallelConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs: list[ParallelConfig] = []
    for key, cfg_list in payload.items():
        num_gpus = int(key)
        if num_gpus not in selected_cards:
            continue
        for cfg in cfg_list:
            if not cfg.get("enabled", True):
                continue
            item = ParallelConfig(
                num_gpus=num_gpus,
                name=str(cfg["name"]),
                vae_patch_parallel_size=int(cfg["vae_patch_parallel_size"]),
                tensor_parallel_size=int(cfg["tensor_parallel_size"]),
                ulysses_degree=int(cfg["ulysses_degree"]),
                ring_degree=int(cfg["ring_degree"]),
                cfg_parallel_size=int(cfg["cfg_parallel_size"]),
                use_hsdp=bool(cfg.get("use_hsdp", False)),
                hsdp_shard_size=int(cfg.get("hsdp_shard_size", -1)),
                hsdp_replicate_size=int(cfg.get("hsdp_replicate_size", 1)),
            )

            core_world_size = item.world_size
            expected_world_size = core_world_size

            if not item.use_hsdp and core_world_size != num_gpus:
                raise ValueError(
                    f"Invalid config {item.name}: TP*USP*CFG={core_world_size}, "
                    f"expected {num_gpus}."
                )

            if item.use_hsdp:
                if item.tensor_parallel_size != 1:
                    raise ValueError(
                        f"Invalid config {item.name}: HSDP requires tensor_parallel_size=1, "
                        f"got {item.tensor_parallel_size}."
                    )
                if item.hsdp_replicate_size <= 0:
                    raise ValueError(
                        f"Invalid config {item.name}: hsdp_replicate_size must be > 0, "
                        f"got {item.hsdp_replicate_size}."
                    )

                shard_size = item.hsdp_shard_size
                if shard_size == -1:
                    if core_world_size == 1:
                        raise ValueError(
                            f"Invalid config {item.name}: cannot auto-calculate hsdp_shard_size "
                            "when TP/SP/CFG are all 1; please set hsdp_shard_size explicitly."
                        )
                    if core_world_size % item.hsdp_replicate_size != 0:
                        raise ValueError(
                            f"Invalid config {item.name}: hsdp_replicate_size={item.hsdp_replicate_size} "
                            f"must divide TP*USP*CFG world_size={core_world_size}."
                        )
                    hsdp_world_size = core_world_size
                else:
                    if shard_size <= 0:
                        raise ValueError(
                            f"Invalid config {item.name}: hsdp_shard_size must be > 0 or -1, got {shard_size}."
                        )
                    hsdp_world_size = shard_size * item.hsdp_replicate_size

                if core_world_size == 1:
                    expected_world_size = hsdp_world_size
                else:
                    if hsdp_world_size != core_world_size:
                        raise ValueError(
                            f"Invalid config {item.name}: HSDP dimensions "
                            f"({item.hsdp_replicate_size}x{shard_size}={hsdp_world_size}) "
                            f"must equal TP*USP*CFG world_size ({core_world_size})."
                        )
                    expected_world_size = core_world_size

            if expected_world_size != num_gpus:
                raise ValueError(
                    f"Invalid config {item.name}: world_size={expected_world_size}, expected {num_gpus}."
                )
            if item.vae_patch_parallel_size not in (1, num_gpus):
                raise ValueError(
                    f"Invalid config {item.name}: vae_patch_parallel_size={item.vae_patch_parallel_size}, "
                    f"expected 1 or {num_gpus}."
                )
            configs.append(item)
    configs.sort(key=lambda x: (x.num_gpus, x.name))
    return configs

def build_worker_cmd(
    args: argparse.Namespace,
    cfg: ParallelConfig,
    request_types_path: Path,
    output_json: Path,
) -> list[str]:
    cmd = [
        args.python_executable,
        args.worker_script,
        "--model",
        args.model,
        "--request-types-config",
        str(request_types_path),
        "--output-json",
        str(output_json),
        "--parallel-name",
        cfg.name,
        "--num-gpus",
        str(cfg.num_gpus),
        "--repeats",
        str(args.repeats),
        "--warmup-iters",
        str(args.warmup_iters),
        "--seed",
        str(args.seed),
        "--guidance-scale",
        str(args.guidance_scale),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--flow-shift",
        str(args.flow_shift),
        "--boundary-ratio",
        str(args.boundary_ratio),
        "--ulysses-degree",
        str(cfg.ulysses_degree),
        "--ring-degree",
        str(cfg.ring_degree),
        "--cfg-parallel-size",
        str(cfg.cfg_parallel_size),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--vae-patch-parallel-size",
        str(cfg.vae_patch_parallel_size),
    ]
    cmd.append("--vae-use-slicing")
    cmd.append("--vae-use-tiling")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.enable_cpu_offload:
        cmd.append("--enable-cpu-offload")
    if args.enable_layerwise_offload:
        cmd.append("--enable-layerwise-offload")
    if args.num_weight_load_threads > 0:
        cmd.extend(["--num-weight-load-threads", str(args.num_weight_load_threads)])
    if args.request_timeout_seconds > 0:
        cmd.extend(["--request-timeout-seconds", str(args.request_timeout_seconds)])
    if args.warmup_timeout_seconds > 0:
        cmd.extend(["--warmup-timeout-seconds", str(args.warmup_timeout_seconds)])
    if args.timeout_grace_seconds > 0:
        cmd.extend(["--timeout-grace-seconds", str(args.timeout_grace_seconds)])
    if cfg.use_hsdp:
        cmd.append("--use-hsdp")
        if cfg.hsdp_shard_size > 0:
            cmd.extend(["--hsdp-shard-size", str(cfg.hsdp_shard_size)])
        cmd.extend(["--hsdp-replicate-size", str(cfg.hsdp_replicate_size)])
    if not args.resume:
        cmd.append("--no-resume")
    if args.request_fail_fast:
        cmd.append("--request-fail-fast")
    return cmd


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = [
        "timestamp",
        "model",
        "num_gpus",
        "gpu_ids",
        "parallel_name",
        "vae_patch_parallel_size",
        "tensor_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "cfg_parallel_size",
        "request_type_id",
        "height",
        "width",
        "num_frames",
        "num_inference_steps",
        "fps",
        "repeat_id",
        "seed",
        "latency_seconds",
        "latency_ms",
        "run_dir",
        "metrics_json",
        "stdout_log",
        "model_load_seconds",
        "is_timeout",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def main() -> None:
    args = parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    request_types_path = (workspace_root / args.request_types_config).resolve()
    matrix_path = (workspace_root / args.parallel_matrix_config).resolve()
    worker_script = (workspace_root / args.worker_script).resolve()

    request_types = load_request_types(request_types_path)
    card_counts = {int(x.strip()) for x in args.card_counts.split(",") if x.strip()}
    parallel_configs = load_parallel_configs(matrix_path, card_counts)

    gpu_ids = [x.strip() for x in args.gpu_device_ids.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("gpu-device-ids cannot be empty")

    total_runs = len(request_types) * len(parallel_configs) * args.repeats
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (workspace_root / args.output_root / run_tag).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plan_file = output_root / "plan.json"
    plan_payload = {
        "model": args.model,
        "request_types_count": len(request_types),
        "parallel_config_count": len(parallel_configs),
        "repeats": args.repeats,
        "warmup_iters": args.warmup_iters,
        "request_timeout_seconds": args.request_timeout_seconds,
        "warmup_timeout_seconds": args.warmup_timeout_seconds,
        "timeout_grace_seconds": args.timeout_grace_seconds,
        "resume": args.resume,
        "request_fail_fast": args.request_fail_fast,
        "num_weight_load_threads": args.num_weight_load_threads,
        "worker_timeout_seconds": args.worker_timeout_seconds,
        "total_runs": total_runs,
        "request_types_config": str(request_types_path),
        "parallel_matrix_config": str(matrix_path),
        "worker_script": str(worker_script),
    }
    plan_file.write_text(json.dumps(plan_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Plan] request types: {len(request_types)}")
    print(f"[Plan] parallel configs: {len(parallel_configs)}")
    print(f"[Plan] repeats: {args.repeats}")
    print(f"[Plan] total runs: {total_runs}")
    print(f"[Plan] output: {output_root}")

    rows: list[dict[str, Any]] = []
    for cfg in parallel_configs:
        if cfg.num_gpus > len(gpu_ids):
            raise ValueError(
                f"Need {cfg.num_gpus} GPUs for config {cfg.name}, "
                f"but only {len(gpu_ids)} ids provided"
            )

        selected_gpu_ids = gpu_ids[: cfg.num_gpus]
        cuda_visible_devices = ",".join(selected_gpu_ids)
        cfg_dir = output_root / f"g{cfg.num_gpus}_{cfg.name}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = cfg_dir / "worker.log"
        metrics_json = cfg_dir / "worker_results.json"

        print(
            f"[Config] g{cfg.num_gpus}_{cfg.name} "
            f"(CUDA_VISIBLE_DEVICES={cuda_visible_devices})"
        )

        worker_cmd = build_worker_cmd(
            args=args,
            cfg=cfg,
            request_types_path=request_types_path,
            output_json=metrics_json,
        )

        if args.dry_run:
            print(
                f"[DryRun] Would load model once, warmup {args.warmup_iters}, "
                f"then run {len(request_types)} request types x {args.repeats} repeats."
            )
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        try:
            with stdout_log.open("w", encoding="utf-8") as log_f:
                worker_proc = subprocess.Popen(
                    worker_cmd,
                    cwd=str(workspace_root),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid,
                )

                timeout_s = args.worker_timeout_seconds if args.worker_timeout_seconds > 0 else None
                try:
                    worker_return_code = worker_proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired as exc:
                    terminate_process_group(worker_proc)
                    raise TimeoutError(
                        f"worker timeout after {args.worker_timeout_seconds}s for config {cfg.name}"
                    ) from exc

            class _Result:
                def __init__(self, returncode: int):
                    self.returncode = returncode

            result = _Result(worker_return_code)

            payload = None
            if metrics_json.exists():
                payload = json.loads(metrics_json.read_text(encoding="utf-8"))

            if payload is not None:
                model_load_seconds = ""
                model_load_stats = payload.get("model_load_stats", {})
                if isinstance(model_load_stats, dict):
                    try:
                        model_load_seconds = float(model_load_stats.get("latency_seconds", ""))
                    except Exception:
                        model_load_seconds = ""
                for item in payload.get("results", []):
                    rows.append(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "model": args.model,
                            "num_gpus": cfg.num_gpus,
                            "gpu_ids": cuda_visible_devices,
                            "parallel_name": cfg.name,
                            "vae_patch_parallel_size": cfg.vae_patch_parallel_size,
                            "tensor_parallel_size": cfg.tensor_parallel_size,
                            "ulysses_degree": cfg.ulysses_degree,
                            "ring_degree": cfg.ring_degree,
                            "cfg_parallel_size": cfg.cfg_parallel_size,
                            "request_type_id": item.get("request_type_id", ""),
                            "height": item.get("height", ""),
                            "width": item.get("width", ""),
                            "num_frames": item.get("num_frames", ""),
                            "num_inference_steps": item.get("num_inference_steps", ""),
                            "fps": item.get("fps", ""),
                            "repeat_id": item.get("repeat_id", ""),
                            "seed": item.get("seed", ""),
                            "latency_seconds": item.get("latency_seconds", ""),
                            "latency_ms": item.get("latency_ms", ""),
                            "run_dir": str(cfg_dir),
                            "metrics_json": str(metrics_json),
                            "stdout_log": str(stdout_log),
                            "model_load_seconds": model_load_seconds,
                            "is_timeout": str(item.get("status", "ok")) == "timeout",
                            "status": item.get("status", "ok"),
                            "error": item.get("error", ""),
                        }
                    )
                save_csv(rows, output_root / "summary_runs.csv")

            if result.returncode != 0:
                raise RuntimeError(f"worker failed with exit code {result.returncode}")

            if payload is None:
                raise FileNotFoundError(f"Worker output not found: {metrics_json}")

        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": args.model,
                    "num_gpus": cfg.num_gpus,
                    "gpu_ids": cuda_visible_devices,
                    "parallel_name": cfg.name,
                    "vae_patch_parallel_size": cfg.vae_patch_parallel_size,
                    "tensor_parallel_size": cfg.tensor_parallel_size,
                    "ulysses_degree": cfg.ulysses_degree,
                    "ring_degree": cfg.ring_degree,
                    "cfg_parallel_size": cfg.cfg_parallel_size,
                    "request_type_id": "",
                    "height": "",
                    "width": "",
                    "num_frames": "",
                    "num_inference_steps": "",
                    "fps": "",
                    "repeat_id": "",
                    "seed": "",
                    "latency_seconds": "",
                    "latency_ms": "",
                    "run_dir": str(cfg_dir),
                    "metrics_json": str(metrics_json),
                    "stdout_log": str(stdout_log),
                    "model_load_seconds": "",
                    "is_timeout": False,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            save_csv(rows, output_root / "summary_runs.csv")

    summary_csv = output_root / "summary_runs.csv"
    save_csv(rows, summary_csv)
    print(f"[Done] Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
