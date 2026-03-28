#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
BENCH_SCRIPT = ROOT_DIR / "benchmarks/diffusion/diffusion_benchmark_serving.py"

IMAGE_ENDPOINT = "/v1/images/generations"
VIDEO_ENDPOINT = "/v1/videos"
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4"}
REQUEST_ID_PATTERN = re.compile(r"request_id=([^ ]+)")


@dataclass
class ServerHandle:
    process: subprocess.Popen[Any]
    log_path: Path
    port: int


@dataclass
class ModalityResult:
    modality: str
    compare_passed: bool
    preemption_passed: bool
    saved_count: int
    completed_requests: int
    failed_requests: int
    preempted_completed_ids: list[str]
    compare_detail: dict[str, Any]
    result_dir: Path


class CheckError(RuntimeError):
    pass


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_b64(payload: str) -> bytes:
    return base64.b64decode(payload)


def _fingerprint_image_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise CheckError(f"Image response does not contain non-empty data list: {response_json}")

    fingerprints: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str):
            raise CheckError(f"Image response item #{idx} missing b64_json: {item}")
        image_bytes = _decode_b64(item["b64_json"])
        with Image.open(BytesIO(image_bytes)) as img:
            normalized = img.convert("RGB")
            fingerprints.append(
                {
                    "index": idx,
                    "raw_sha256": _sha256_hex(image_bytes),
                    "semantic_sha256": _sha256_hex(normalized.tobytes()),
                    "size": list(normalized.size),
                }
            )
    return fingerprints


def _fingerprint_video_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise CheckError(f"Video response does not contain non-empty data list: {response_json}")

    fingerprints: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("b64_json"), str):
            raise CheckError(f"Video response item #{idx} missing b64_json: {item}")
        video_bytes = _decode_b64(item["b64_json"])
        fingerprints.append(
            {
                "index": idx,
                "raw_sha256": _sha256_hex(video_bytes),
                "byte_length": len(video_bytes),
            }
        )
    return fingerprints


def _compare_fingerprints(left: list[dict[str, Any]], right: list[dict[str, Any]], semantic_key: str) -> bool:
    if len(left) != len(right):
        return False
    return all(a.get(semantic_key) == b.get(semantic_key) for a, b in zip(left, right, strict=True))


def _ensure_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise CheckError(f"Port {port} is already in use.")


def _wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    health_url = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise CheckError(f"Server did not become ready within {timeout_s}s: {health_url}")


def _start_server(
    *,
    model: str,
    host: str,
    port: int,
    num_gpus: int,
    policy: str,
    enable_step_chunk: bool,
    enable_chunk_preemption: bool,
    chunk_budget_steps: int,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
    ready_timeout_s: int,
) -> ServerHandle:
    _ensure_port_free(port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")

    cmd = [
        "vllm",
        "serve",
        model,
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--num-gpus",
        str(num_gpus),
        "--instance-scheduler-policy",
        policy,
        "--diffusion-chunk-budget-steps",
        str(chunk_budget_steps),
    ]
    if enable_step_chunk:
        cmd.append("--diffusion-enable-step-chunk")
    if enable_chunk_preemption:
        cmd.append("--diffusion-enable-chunk-preemption")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env,
    )
    try:
        _wait_for_server(f"http://{host}:{port}", ready_timeout_s)
    except Exception:
        _stop_server(ServerHandle(process=process, log_path=log_path, port=port))
        raise
    finally:
        log_fp.close()

    return ServerHandle(process=process, log_path=log_path, port=port)


def _stop_server(handle: ServerHandle | None) -> None:
    if handle is None:
        return
    process = handle.process
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def _send_image_request(base_url: str, model: str, seed: int, timeout_s: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": "A bright toy robot waving in a studio, sharp details",
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json",
        "seed": seed,
        "num_inference_steps": 20,
    }
    response = requests.post(base_url.rstrip("/") + IMAGE_ENDPOINT, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _send_video_request(base_url: str, model: str, seed: int, timeout_s: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": "A small red drone flying above a calm lake at sunset",
        "n": "1",
        "size": "854x480",
        "response_format": "b64_json",
        "seed": str(seed),
        "num_inference_steps": "4",
        "num_frames": "33",
        "fps": "16",
    }
    response = requests.post(base_url.rstrip("/") + VIDEO_ENDPOINT, data=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _build_image_trace() -> str:
    return "\n".join(
        [
            "Request(request_id='img-long', timestamp=0.0, width=1536, height=1536, num_frames=1, prompt='A cinematic portrait of a futuristic city square at sunrise', num_inference_steps=35)",
            "Request(request_id='img-short-1', timestamp=0.1, width=512, height=512, num_frames=1, prompt='A minimalist icon of a blue bird', num_inference_steps=18)",
            "Request(request_id='img-short-2', timestamp=0.2, width=512, height=512, num_frames=1, prompt='A minimalist icon of a yellow fish', num_inference_steps=18)",
            "Request(request_id='img-short-3', timestamp=0.3, width=512, height=512, num_frames=1, prompt='A minimalist icon of a green leaf', num_inference_steps=18)",
        ]
    ) + "\n"


def _build_video_trace() -> str:
    return "\n".join(
        [
            "Request(request_id='vid-long', timestamp=0.0, width=854, height=480, num_frames=81, fps=16, prompt='A spaceship gliding through clouds with dramatic lighting', num_inference_steps=6)",
            "Request(request_id='vid-short-1', timestamp=0.1, width=854, height=480, num_frames=17, fps=16, prompt='A paper boat floating on still water', num_inference_steps=3)",
            "Request(request_id='vid-short-2', timestamp=0.2, width=854, height=480, num_frames=17, fps=16, prompt='A red lantern swaying gently in the wind', num_inference_steps=3)",
            "Request(request_id='vid-short-3', timestamp=0.3, width=854, height=480, num_frames=17, fps=16, prompt='A small fox looking around in a snowy field', num_inference_steps=3)",
        ]
    ) + "\n"


def _run_benchmark(
    *,
    modality: str,
    model: str,
    host: str,
    port: int,
    trace_path: Path,
    result_json: Path,
    bench_log: Path,
    output_dir: Path,
    request_rate: float,
) -> dict[str, Any]:
    bench_log.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if modality == "image":
        cmd = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--backend",
            "openai",
            "--base-url",
            f"http://{host}:{port}",
            "--model",
            model,
            "--dataset",
            "trace",
            "--dataset-path",
            str(trace_path),
            "--task",
            "t2i",
            "--num-prompts",
            "4",
            "--warmup-requests",
            "0",
            "--request-rate",
            str(request_rate),
            "--max-concurrency",
            "4",
            "--save-output-dir",
            str(output_dir),
            "--output-file",
            str(result_json),
            "--disable-tqdm",
        ]
    else:
        cmd = [
            sys.executable,
            str(BENCH_SCRIPT),
            "--backend",
            "v1/videos",
            "--base-url",
            f"http://{host}:{port}",
            "--model",
            model,
            "--dataset",
            "trace",
            "--dataset-path",
            str(trace_path),
            "--task",
            "t2v",
            "--num-prompts",
            "4",
            "--warmup-requests",
            "0",
            "--request-rate",
            str(request_rate),
            "--max-concurrency",
            "4",
            "--save-output-dir",
            str(output_dir),
            "--output-file",
            str(result_json),
            "--disable-tqdm",
        ]

    with bench_log.open("w", encoding="utf-8") as fp:
        subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=fp, stderr=subprocess.STDOUT, check=True)

    return json.loads(result_json.read_text(encoding="utf-8"))


def _analyze_preemption(log_path: Path) -> dict[str, Any]:
    preempted: set[str] = set()
    resumed: set[str] = set()
    completed: set[str] = set()
    failed_lines: list[str] = []

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = REQUEST_ID_PATTERN.search(line)
        request_id = match.group(1) if match else None
        if "REQUEST_PREEMPTED" in line and request_id:
            preempted.add(request_id)
        if "REQUEST_RESUMED" in line and request_id:
            resumed.add(request_id)
        if "REQUEST_COMPLETED" in line and request_id:
            completed.add(request_id)
        if "REQUEST_FAIL" in line or "REQUEST_FAILED" in line:
            failed_lines.append(line)

    preempted_completed_ids = sorted(preempted & resumed & completed)
    return {
        "preempted_ids": sorted(preempted),
        "resumed_ids": sorted(resumed),
        "completed_ids": sorted(completed),
        "preempted_completed_ids": preempted_completed_ids,
        "failed_lines": failed_lines,
    }


def _count_saved_outputs(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES)


def _compare_image_outputs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_fp = _fingerprint_image_response(baseline)
    candidate_fp = _fingerprint_image_response(candidate)
    passed = _compare_fingerprints(baseline_fp, candidate_fp, "semantic_sha256")
    return {
        "passed": passed,
        "baseline": baseline_fp,
        "candidate": candidate_fp,
    }


def _compare_video_outputs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_fp = _fingerprint_video_response(baseline)
    candidate_fp = _fingerprint_video_response(candidate)
    passed = _compare_fingerprints(baseline_fp, candidate_fp, "raw_sha256")
    return {
        "passed": passed,
        "baseline": baseline_fp,
        "candidate": candidate_fp,
    }


def _run_modality(
    *,
    modality: str,
    model: str,
    num_gpus: int,
    base_port: int,
    host: str,
    chunk_budget_steps: int,
    request_timeout_s: int,
    ready_timeout_s: int,
    request_rate: float,
    out_dir: Path,
) -> ModalityResult:
    modality_dir = out_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    baseline_port = base_port
    candidate_port = base_port + 1
    baseline_log = modality_dir / "baseline_server.log"
    candidate_log = modality_dir / "candidate_server.log"
    baseline_response_path = modality_dir / "baseline_response.json"
    candidate_response_path = modality_dir / "candidate_response.json"
    trace_path = modality_dir / f"{modality}_preemption_trace.txt"
    result_json = modality_dir / "benchmark_result.json"
    bench_log = modality_dir / "benchmark.log"
    saved_output_dir = modality_dir / "saved_outputs"

    baseline: ServerHandle | None = None
    candidate: ServerHandle | None = None
    try:
        baseline = _start_server(
            model=model,
            host=host,
            port=baseline_port,
            num_gpus=num_gpus,
            policy="fcfs",
            enable_step_chunk=False,
            enable_chunk_preemption=False,
            chunk_budget_steps=chunk_budget_steps,
            log_path=baseline_log,
            ready_timeout_s=ready_timeout_s,
        )
        if modality == "image":
            baseline_response = _send_image_request(f"http://{host}:{baseline_port}", model, 20250323, request_timeout_s)
        else:
            baseline_response = _send_video_request(f"http://{host}:{baseline_port}", model, 20250323, request_timeout_s)
        baseline_response_path.write_text(json.dumps(baseline_response, indent=2), encoding="utf-8")
    finally:
        _stop_server(baseline)

    try:
        candidate = _start_server(
            model=model,
            host=host,
            port=candidate_port,
            num_gpus=num_gpus,
            policy="sjf",
            enable_step_chunk=True,
            enable_chunk_preemption=True,
            chunk_budget_steps=chunk_budget_steps,
            log_path=candidate_log,
            ready_timeout_s=ready_timeout_s,
        )
        if modality == "image":
            candidate_response = _send_image_request(f"http://{host}:{candidate_port}", model, 20250323, request_timeout_s)
            trace_path.write_text(_build_image_trace(), encoding="utf-8")
            compare_detail = _compare_image_outputs(baseline_response, candidate_response)
        else:
            candidate_response = _send_video_request(f"http://{host}:{candidate_port}", model, 20250323, request_timeout_s)
            trace_path.write_text(_build_video_trace(), encoding="utf-8")
            compare_detail = _compare_video_outputs(baseline_response, candidate_response)
        candidate_response_path.write_text(json.dumps(candidate_response, indent=2), encoding="utf-8")

        metrics = _run_benchmark(
            modality=modality,
            model=model,
            host=host,
            port=candidate_port,
            trace_path=trace_path,
            result_json=result_json,
            bench_log=bench_log,
            output_dir=saved_output_dir,
            request_rate=request_rate,
        )
    finally:
        _stop_server(candidate)

    preemption_detail = _analyze_preemption(candidate_log)
    saved_count = _count_saved_outputs(saved_output_dir)
    completed_requests = int(metrics.get("completed_requests", 0))
    failed_requests = int(metrics.get("failed_requests", 0))
    preemption_passed = (
        bool(preemption_detail["preempted_completed_ids"])
        and not preemption_detail["failed_lines"]
        and failed_requests == 0
        and saved_count >= completed_requests
        and completed_requests > 0
    )

    summary = {
        "modality": modality,
        "model": model,
        "compare": compare_detail,
        "preemption": preemption_detail,
        "benchmark_metrics": metrics,
        "saved_count": saved_count,
    }
    (modality_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ModalityResult(
        modality=modality,
        compare_passed=bool(compare_detail["passed"]),
        preemption_passed=preemption_passed,
        saved_count=saved_count,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        preempted_completed_ids=preemption_detail["preempted_completed_ids"],
        compare_detail=compare_detail,
        result_dir=modality_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick correctness test for SJF preemption on diffusion image/video generation.")
    parser.add_argument("--run", choices=["image", "video", "both"], default="both")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=18091)
    parser.add_argument("--image-model", default=os.environ.get("IMAGE_MODEL", "Qwen/Qwen-Image"))
    parser.add_argument("--video-model", default=os.environ.get("VIDEO_MODEL", ""))
    parser.add_argument("--image-num-gpus", type=int, default=int(os.environ.get("IMAGE_NUM_GPUS", "1")))
    parser.add_argument("--video-num-gpus", type=int, default=int(os.environ.get("VIDEO_NUM_GPUS", "4")))
    parser.add_argument("--chunk-budget-steps", type=int, default=4)
    parser.add_argument("--request-timeout-s", type=int, default=900)
    parser.add_argument("--server-ready-timeout-s", type=int, default=900)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument(
        "--out-dir",
        default=str(Path.cwd() / "tmp" / f"sjf_preemption_correctness_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities: list[tuple[str, str, int, int]] = []
    if args.run in {"image", "both"}:
        modalities.append(("image", args.image_model, args.image_num_gpus, args.base_port))
    if args.run in {"video", "both"}:
        if not args.video_model:
            raise CheckError("--video-model is required when --run is video or both.")
        modalities.append(("video", args.video_model, args.video_num_gpus, args.base_port + 10))

    results: list[ModalityResult] = []
    for modality, model, num_gpus, base_port in modalities:
        print(f"\n=== Running {modality} correctness check ===")
        print(f"model={model}")
        print(f"num_gpus={num_gpus}")
        print(f"ports={base_port},{base_port + 1}")
        result = _run_modality(
            modality=modality,
            model=model,
            num_gpus=num_gpus,
            base_port=base_port,
            host=args.host,
            chunk_budget_steps=args.chunk_budget_steps,
            request_timeout_s=args.request_timeout_s,
            ready_timeout_s=args.server_ready_timeout_s,
            request_rate=args.request_rate,
            out_dir=out_dir,
        )
        results.append(result)
        overall = result.compare_passed and result.preemption_passed
        print(f"compare_passed={result.compare_passed}")
        print(f"preemption_passed={result.preemption_passed}")
        print(f"saved_count={result.saved_count} completed={result.completed_requests} failed={result.failed_requests}")
        print(f"preempted_completed_ids={result.preempted_completed_ids}")
        print(f"result_dir={result.result_dir}")
        print(f"status={'PASS' if overall else 'FAIL'}")

    summary = {
        "results": [
            {
                "modality": r.modality,
                "compare_passed": r.compare_passed,
                "preemption_passed": r.preemption_passed,
                "saved_count": r.saved_count,
                "completed_requests": r.completed_requests,
                "failed_requests": r.failed_requests,
                "preempted_completed_ids": r.preempted_completed_ids,
                "result_dir": str(r.result_dir),
            }
            for r in results
        ]
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary_json={summary_path}")

    return 0 if all(r.compare_passed and r.preemption_passed for r in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
