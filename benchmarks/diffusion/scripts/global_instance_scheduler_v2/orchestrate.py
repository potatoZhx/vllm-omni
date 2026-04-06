#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "diffusion" / "results"
DEFAULT_SCHEDULER_MODULE = "vllm_omni.global_scheduler.server"
DEFAULT_BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "diffusion" / "diffusion_benchmark_serving.py"
NO_PROXY_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))
SUPPORTED_GLOBAL_POLICIES = {"min_queue_length", "round_robin", "short_queue_runtime"}


def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_request_rates(raw: str) -> list[str]:
    tokens = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    if not tokens:
        raise ValueError("No request rates configured. Set REQUEST_RATE or REQUEST_RATES.")
    return tokens


def parse_case_matrix(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 2:
            raise ValueError(f"Invalid CASE_MATRIX row: {line}")
        case_name, global_policy = parts
        rows.append({"case_name": case_name, "global_policy": global_policy})
    if not rows:
        raise ValueError("CASE_MATRIX is empty.")
    return rows


def split_worker_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def strip_flag(args: list[str], flag: str) -> list[str]:
    filtered: list[str] = []
    idx = 0
    while idx < len(args):
        item = str(args[idx])
        if item == flag:
            idx += 2
            continue
        if item.startswith(flag + "="):
            idx += 1
            continue
        filtered.append(item)
        idx += 1
    return filtered


def strip_boolean_flag(args: list[str], flag: str) -> list[str]:
    return [str(item) for item in args if str(item) != flag]


def resolve_case_name(config_path: Path) -> str:
    payload = read_yaml(config_path)
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    baseline = policy.get("baseline") if isinstance(policy.get("baseline"), dict) else {}
    global_policy = baseline.get("algorithm", "min_queue_length")
    return f"global_{global_policy}"


def generate_config(base_config: Path, generated_config: Path, options: dict[str, str], benchmark_output_file: Path) -> None:
    payload = read_yaml(base_config)
    benchmark = payload.setdefault("benchmark", {})
    policy_baseline = payload.setdefault("policy", {}).setdefault("baseline", {})
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("Config must contain non-empty instances list.")

    global_policy = options.get("GLOBAL_POLICY", "").strip()
    if global_policy:
        if global_policy not in SUPPORTED_GLOBAL_POLICIES:
            raise ValueError(f"Unsupported GLOBAL_POLICY={global_policy}")
        policy_baseline["algorithm"] = global_policy

    benchmark["output_file"] = str(benchmark_output_file.resolve())
    worker_ids = split_worker_ids(options.get("WORKER_IDS", ""))
    if worker_ids:
        benchmark["worker_ids"] = worker_ids

    for env_name, config_key in [
        ("BENCHMARK_MODEL", "model"),
        ("BENCHMARK_BACKEND", "backend"),
        ("BENCHMARK_TASK", "task"),
        ("BENCHMARK_DATASET", "dataset"),
        ("BENCHMARK_DATASET_PATH", "dataset_path"),
        ("BENCHMARK_RANDOM_REQUEST_CONFIG", "random_request_config"),
        ("BENCHMARK_WARMUP_REQUEST_CONFIG", "warmup_request_config"),
    ]:
        value = options.get(env_name, "").strip()
        if value:
            benchmark[config_key] = value

    for env_name, config_key in [
        ("BENCHMARK_MAX_CONCURRENCY", "max_concurrency"),
        ("BENCHMARK_WARMUP_REQUESTS", "warmup_requests"),
        ("BENCHMARK_WARMUP_NUM_INFERENCE_STEPS", "warmup_num_inference_steps"),
    ]:
        value = options.get(env_name, "").strip()
        if value:
            benchmark[config_key] = int(value)

    enable_step_chunk = options.get("ENABLE_STEP_CHUNK", "1").strip()
    target_worker_ids = set(benchmark.get("worker_ids") or [])
    matched_instances: list[str] = []
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        instance_id = str(instance.get("id"))
        if target_worker_ids and instance_id not in target_worker_ids:
            continue

        launch = instance.get("launch")
        if not isinstance(launch, dict):
            continue
        args = [str(item) for item in launch.get("args", [])]
        args = strip_flag(args, "--diffusion-scheduler-backend")
        args = strip_boolean_flag(args, "--diffusion-enable-step-chunk")
        args = strip_boolean_flag(args, "--diffusion-enable-chunk-preemption")
        args = strip_flag(args, "--diffusion-chunk-budget-steps")
        args = strip_flag(args, "--diffusion-engine-max-concurrency")
        args.extend(["--diffusion-scheduler-backend", "step_level_request_scheduler"])
        if enable_step_chunk != "0":
            args.append("--diffusion-enable-step-chunk")
        launch["args"] = args
        matched_instances.append(instance_id)

    if target_worker_ids and (target_worker_ids - set(matched_instances)):
        missing = ", ".join(sorted(target_worker_ids - set(matched_instances)))
        raise ValueError(f"Requested worker ids missing launch config or instance entry: {missing}")

    write_yaml(generated_config, payload)


def normalize_shell_value(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def resolve_benchmark_runtime(config_path: Path) -> dict[str, Any]:
    payload = read_yaml(config_path)
    server = payload.get("server") if isinstance(payload.get("server"), dict) else {}
    benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), dict) else {}
    instances = payload.get("instances") if isinstance(payload.get("instances"), list) else []
    instances_by_id = {
        str(instance.get("id")): instance
        for instance in instances
        if isinstance(instance, dict) and instance.get("id")
    }

    worker_ids = benchmark.get("worker_ids") or list(instances_by_id.keys())
    missing = [worker_id for worker_id in worker_ids if worker_id not in instances_by_id]
    if missing:
        raise ValueError(f"benchmark.worker_ids contains unknown instances: {', '.join(missing)}")

    model = benchmark.get("model")
    if not model:
        launch_models = {
            instance["launch"]["model"]
            for worker_id in worker_ids
            for instance in [instances_by_id[worker_id]]
            if isinstance(instance.get("launch"), dict) and instance["launch"].get("model")
        }
        if len(launch_models) != 1:
            raise ValueError("benchmark.model is required when selected workers do not share one launch.model")
        model = next(iter(launch_models))

    host = str(server.get("host", "0.0.0.0"))
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(server.get("port", 8089))

    def resolve_config_path(value: str | None) -> str:
        if not value:
            return ""
        path = Path(str(value))
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        return str(path)

    return {
        "scheduler_url": f"http://{host}:{port}",
        "worker_ids": [str(worker_id) for worker_id in worker_ids],
        "worker_ready_timeout_s": int(benchmark.get("worker_ready_timeout_s", 600)),
        "model": str(model),
        "backend": str(benchmark.get("backend", "vllm-omni")),
        "task": str(benchmark.get("task", "t2i")),
        "dataset": str(benchmark.get("dataset", "trace")),
        "dataset_path": resolve_config_path(benchmark.get("dataset_path")),
        "random_request_config": normalize_shell_value(benchmark.get("random_request_config")),
        "warmup_request_config": normalize_shell_value(benchmark.get("warmup_request_config")),
        "max_concurrency": int(benchmark.get("max_concurrency", 20)),
        "warmup_requests": int(benchmark.get("warmup_requests", 0)),
        "warmup_num_inference_steps": int(benchmark.get("warmup_num_inference_steps", 1)),
        "output_file": resolve_config_path(benchmark.get("output_file")),
        "save_output_dir": "",
    }


def scheduler_health(scheduler_url: str) -> bool:
    try:
        with NO_PROXY_OPENER.open(f"{scheduler_url}/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def fetch_json(url: str, timeout: float = 30.0) -> dict[str, Any] | None:
    try:
        with NO_PROXY_OPENER.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def is_worker_routable(scheduler_url: str, worker_id: str) -> bool:
    payload = fetch_json(f"{scheduler_url}/instances")
    if not payload:
        return False
    for item in payload.get("instances", []):
        if item.get("id") == worker_id:
            return bool(item.get("routable"))
    return False


def get_worker_endpoint(scheduler_url: str, worker_id: str) -> str:
    payload = fetch_json(f"{scheduler_url}/instances")
    if not payload:
        return ""
    for item in payload.get("instances", []):
        if item.get("id") == worker_id:
            return str(item.get("endpoint") or "")
    return ""


def is_worker_api_ready(endpoint: str) -> bool:
    payload = fetch_json(f"{endpoint.rstrip('/')}/v1/models")
    models = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(models, list) and len(models) > 0


def wait_scheduler_ready(scheduler_url: str, timeout_s: int = 60, poll_interval_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if scheduler_health(scheduler_url):
            print(f"[ready] scheduler: {scheduler_url}")
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Scheduler did not become ready within {timeout_s}s: {scheduler_url}")


def wait_workers_ready(scheduler_url: str, worker_ids: list[str], timeout_s: int) -> None:
    for worker_id in worker_ids:
        start_ts = time.time()
        last_log_ts = start_ts
        while True:
            endpoint = get_worker_endpoint(scheduler_url, worker_id)
            routable = is_worker_routable(scheduler_url, worker_id)
            ready = bool(endpoint) and is_worker_api_ready(endpoint)
            if routable and ready:
                print(f"[ready] {worker_id} routable=true api_ready=true ({endpoint.rstrip('/')}/v1/models)")
                break

            now = time.time()
            elapsed = int(now - start_ts)
            if elapsed > timeout_s:
                raise TimeoutError(f"Timeout waiting for worker ready ({worker_id}) after {elapsed}s")
            if now - last_log_ts >= 60:
                print(f"[waiting] {worker_id} has been waiting for {elapsed}s ...")
                last_log_ts = now
            time.sleep(2)


def sanitize_rate_for_filename(rate: str) -> str:
    value = rate.replace(".", "p")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def resolve_num_prompts(mode: str, rate: str, duration_s: str, fixed_num_prompts: str) -> int:
    if mode == "fixed_duration":
        rate_value = float(rate)
        duration_value = float(duration_s)
        if rate_value <= 0 or duration_value <= 0:
            raise ValueError("fixed_duration mode requires positive request rate and duration.")
        return max(1, math.ceil(rate_value * duration_value))
    return int(fixed_num_prompts)


def resolve_run_duration(mode: str, rate: str, duration_s: str, fixed_num_prompts: str) -> str:
    if mode == "fixed_duration":
        return duration_s
    rate_value = float(rate)
    if rate_value <= 0:
        raise ValueError("fixed_num_prompts mode requires positive request rate.")
    return f"{int(fixed_num_prompts) / rate_value:.6f}"


def resolve_output_file(base_output_file: str, rate: str, request_rates: list[str]) -> str:
    if not base_output_file:
        return ""
    if len(request_rates) <= 1:
        return base_output_file
    path = Path(base_output_file)
    suffix = sanitize_rate_for_filename(rate)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_rps_{suffix}{path.suffix}"))
    return str(path.with_name(f"{path.name}_rps_{suffix}"))


def build_benchmark_command(
    runtime: dict[str, Any],
    rate: str,
    request_rates: list[str],
    mode: str,
    duration_s: str,
    fixed_num_prompts: str,
) -> list[str]:
    num_prompts = resolve_num_prompts(mode, rate, duration_s, fixed_num_prompts)
    run_duration = resolve_run_duration(mode, rate, duration_s, fixed_num_prompts)
    output_file = resolve_output_file(runtime["output_file"], rate, request_rates)

    print(
        f"[bench] start benchmark: mode={mode}, rate={rate}, "
        f"num_prompts={num_prompts}, request_duration_s={run_duration}"
    )

    cmd = [
        sys.executable,
        str(DEFAULT_BENCHMARK_SCRIPT),
        "--base-url",
        runtime["scheduler_url"],
        "--backend",
        runtime["backend"],
        "--model",
        runtime["model"],
        "--task",
        runtime["task"],
        "--dataset",
        runtime["dataset"],
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(runtime["max_concurrency"]),
        "--request-rate",
        rate,
        "--warmup-requests",
        str(runtime["warmup_requests"]),
        "--warmup-num-inference-steps",
        str(runtime["warmup_num_inference_steps"]),
    ]
    if runtime["dataset_path"]:
        cmd.extend(["--dataset-path", runtime["dataset_path"]])
    if runtime["random_request_config"]:
        cmd.extend(["--random-request-config", runtime["random_request_config"]])
    if runtime["warmup_request_config"]:
        cmd.extend(["--warmup-request-config", runtime["warmup_request_config"]])
    if output_file:
        cmd.extend(["--output-file", output_file])
    return cmd


def start_scheduler(config_file: Path, log_file: Path, instance_log_dir: Path | None = None) -> subprocess.Popen[str]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if instance_log_dir is not None and not env.get("GLOBAL_SCHEDULER_LOG_DIR"):
        env["GLOBAL_SCHEDULER_LOG_DIR"] = str(instance_log_dir.resolve())
    log_handle = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            DEFAULT_SCHEDULER_MODULE,
            "--config",
            str(config_file),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._log_handle = log_handle  # type: ignore[attr-defined]
    return process


def stop_scheduler(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    log_handle = getattr(process, "_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def write_summary(out_root: Path) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    for metrics_file in sorted(out_root.glob("*/metrics*.json")):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append(
            {
                "case": metrics_file.parent.name,
                "metrics_file": str(metrics_file),
                "request_rate": metrics.get("request_rate"),
                "completed": metrics.get("completed"),
                "throughput_qps": metrics.get("throughput_qps"),
                "latency_p50": metrics.get("latency_p50"),
                "latency_p95": metrics.get("latency_p95"),
                "latency_p99": metrics.get("latency_p99"),
                "backend": metrics.get("backend"),
                "model": metrics.get("model"),
            }
        )

    summary_json = out_root / "summary.json"
    summary_csv = out_root / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "request_rate",
                "completed",
                "throughput_qps",
                "latency_p50",
                "latency_p95",
                "latency_p99",
                "backend",
                "model",
                "metrics_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary_json, summary_csv


def collect_case_options() -> dict[str, str]:
    return {
        "BASE_CONFIG": env_str(
            "BASE_CONFIG",
            str(REPO_ROOT / "benchmarks" / "diffusion" / "scripts" / "global_instance_scheduler_v2" / "single_instance.qwen.yaml"),
        ),
        "GLOBAL_POLICY": env_str("GLOBAL_POLICY"),
        "ENABLE_STEP_CHUNK": env_str("ENABLE_STEP_CHUNK", "1"),
        "REQUEST_RATES": env_str("REQUEST_RATES", env_str("REQUEST_RATE", "0.2,0.4,0.6")),
        "REQUEST_DURATION_S": env_str("REQUEST_DURATION_S", "600"),
        "BENCHMARK_MODE": env_str("BENCHMARK_MODE", "fixed_duration"),
        "NUM_PROMPTS_DURATION_SECONDS": env_str("NUM_PROMPTS_DURATION_SECONDS", env_str("REQUEST_DURATION_S", "600")),
        "FIXED_NUM_PROMPTS": env_str("FIXED_NUM_PROMPTS", "20"),
        "CASE_NAME": env_str("CASE_NAME"),
        "RUN_TAG": env_str("RUN_TAG", time.strftime("%Y%m%d_%H%M%S")),
        "OUT_DIR": env_str("OUT_DIR"),
        "BENCH_OUTPUT_FILE": env_str("BENCH_OUTPUT_FILE"),
        "SCHEDULER_LOG_FILE": env_str("SCHEDULER_LOG_FILE"),
        "WORKER_IDS": env_str("WORKER_IDS"),
        "BENCHMARK_MODEL": env_str("BENCHMARK_MODEL"),
        "BENCHMARK_BACKEND": env_str("BENCHMARK_BACKEND"),
        "BENCHMARK_TASK": env_str("BENCHMARK_TASK"),
        "BENCHMARK_DATASET": env_str("BENCHMARK_DATASET"),
        "BENCHMARK_DATASET_PATH": env_str("BENCHMARK_DATASET_PATH"),
        "BENCHMARK_RANDOM_REQUEST_CONFIG": env_str("BENCHMARK_RANDOM_REQUEST_CONFIG"),
        "BENCHMARK_MAX_CONCURRENCY": env_str("BENCHMARK_MAX_CONCURRENCY"),
        "BENCHMARK_WARMUP_REQUESTS": env_str("BENCHMARK_WARMUP_REQUESTS"),
        "BENCHMARK_WARMUP_NUM_INFERENCE_STEPS": env_str("BENCHMARK_WARMUP_NUM_INFERENCE_STEPS"),
    }


def collect_suite_options() -> dict[str, str]:
    options = collect_case_options()
    options.update(
        {
            "SUITE_NAME": env_str("SUITE_NAME", f"global_instance_scheduler_rps_{options['RUN_TAG']}"),
            "OUT_ROOT": env_str("OUT_ROOT"),
            "CASE_MATRIX": env_str(
                "CASE_MATRIX",
                "min_queue_length|min_queue_length\nround_robin|round_robin\nshort_queue_runtime|short_queue_runtime",
            ),
        }
    )
    return options


def run_case(options: dict[str, str]) -> Path:
    base_config = Path(options["BASE_CONFIG"]).resolve()
    require_file(base_config, "Base config")
    require_file(DEFAULT_BENCHMARK_SCRIPT, "Benchmark script")

    benchmark_mode = options["BENCHMARK_MODE"]
    if benchmark_mode not in {"fixed_duration", "fixed_num_prompts"}:
        raise ValueError(
            f"Unsupported BENCHMARK_MODE={benchmark_mode}, expected fixed_duration or fixed_num_prompts"
        )

    request_rates = parse_request_rates(options["REQUEST_RATES"])
    run_tag = options["RUN_TAG"]
    explicit_out_dir = options["OUT_DIR"].strip()
    case_name = options["CASE_NAME"].strip()

    if explicit_out_dir:
        out_dir = Path(explicit_out_dir).resolve()
        out_dir_was_provided = True
    else:
        out_dir = (DEFAULT_RESULTS_ROOT / f".tmp_{run_tag}_{os.getpid()}").resolve()
        out_dir_was_provided = False

    out_dir.mkdir(parents=True, exist_ok=True)
    bench_output_file = Path(options["BENCH_OUTPUT_FILE"]).resolve() if options["BENCH_OUTPUT_FILE"].strip() else out_dir / "metrics.json"
    scheduler_log_file = Path(options["SCHEDULER_LOG_FILE"]).resolve() if options["SCHEDULER_LOG_FILE"].strip() else out_dir / "global_scheduler_server.log"
    generated_config = out_dir / "global_scheduler.generated.yaml"
    generate_config(base_config, generated_config, options, bench_output_file)

    if not case_name:
        case_name = resolve_case_name(generated_config)

    if not out_dir_was_provided:
        final_out_dir = (DEFAULT_RESULTS_ROOT / f"{case_name}_{run_tag}").resolve()
        if final_out_dir.exists():
            shutil.rmtree(final_out_dir)
        shutil.move(str(out_dir), str(final_out_dir))
        out_dir = final_out_dir
        generated_config = out_dir / "global_scheduler.generated.yaml"
        if not options["BENCH_OUTPUT_FILE"].strip():
            bench_output_file = out_dir / "metrics.json"
        if not options["SCHEDULER_LOG_FILE"].strip():
            scheduler_log_file = out_dir / "global_scheduler_server.log"

    runtime = resolve_benchmark_runtime(generated_config)
    if not options["BENCH_OUTPUT_FILE"].strip():
        runtime["output_file"] = str(bench_output_file.resolve())

    scheduler_process: subprocess.Popen[str] | None = None
    try:
        scheduler_process = start_scheduler(generated_config, scheduler_log_file, out_dir / "instance_logs")
        wait_scheduler_ready(runtime["scheduler_url"], timeout_s=60, poll_interval_s=1.0)
        wait_workers_ready(runtime["scheduler_url"], runtime["worker_ids"], runtime["worker_ready_timeout_s"])
        for rate in request_rates:
            cmd = build_benchmark_command(
                runtime=runtime,
                rate=rate,
                request_rates=request_rates,
                mode=benchmark_mode,
                duration_s=options["NUM_PROMPTS_DURATION_SECONDS"],
                fixed_num_prompts=options["FIXED_NUM_PROMPTS"],
            )
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    finally:
        stop_scheduler(scheduler_process)

    print(f"[done] case={case_name}")
    print(f"[artifacts] {out_dir}")
    return out_dir


def run_suite(options: dict[str, str]) -> Path:
    out_root = Path(options["OUT_ROOT"]).resolve() if options["OUT_ROOT"].strip() else DEFAULT_RESULTS_ROOT / options["SUITE_NAME"]
    out_root.mkdir(parents=True, exist_ok=True)

    for case in parse_case_matrix(options["CASE_MATRIX"]):
        case_options = dict(options)
        case_options.update(
            {
                "CASE_NAME": case["case_name"],
                "GLOBAL_POLICY": case["global_policy"],
                "OUT_DIR": str((out_root / case["case_name"]).resolve()),
                "BENCH_OUTPUT_FILE": "",
                "SCHEDULER_LOG_FILE": "",
            }
        )
        run_case(case_options)

    summary_json, summary_csv = write_summary(out_root)
    print(f"[summary_json] {summary_json}")
    print(f"[summary_csv] {summary_csv}")
    return out_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Global scheduler benchmark orchestrator")
    parser.add_argument("command", choices=["case", "suite"])
    args = parser.parse_args()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: sys.exit(130))

    if args.command == "case":
        run_case(collect_case_options())
    else:
        run_suite(collect_suite_options())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
