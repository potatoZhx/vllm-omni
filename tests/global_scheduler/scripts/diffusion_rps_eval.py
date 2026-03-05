#!/usr/bin/env python3
"""Run RPS sweep benchmark and plot SLO attainment + P95 latency curves.

This script is intentionally config-driven so the companion bash script can
write deployment and benchmark parameters in one place and invoke this script.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import shutil
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"slo_rps_benchmark_{ts}.log"

    logger = logging.getLogger("slo_rps_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)

    logger.info("Log file: %s", log_path)
    return logger


def _load_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc


def _parse_rps_values(raw: Any) -> list[float]:
    if isinstance(raw, list):
        values = [float(x) for x in raw]
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        values = [float(p) for p in parts]
    else:
        raise ValueError("benchmark.rps_values must be a list or comma-separated string")

    if not values:
        raise ValueError("benchmark.rps_values cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError("benchmark.rps_values must be > 0")
    return values


def _to_cli_args(extra_args: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in extra_args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
            continue
        args.extend([flag, str(value)])
    return args


def _extract_p95_latency_ms(result: dict[str, Any], metric: str) -> float:
    # vllm bench serve style keys (already in ms)
    candidates = [
        f"p95_{metric}_ms",
        "p95_e2el_ms",
        "p95_ttft_ms",
        "p95_itl_ms",
        "p95_tpot_ms",
    ]
    for key in candidates:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    for key, value in result.items():
        if not isinstance(value, (int, float)):
            continue
        key_l = key.lower()
        if "p95" in key_l and "_ms" in key_l:
            return float(value)

    # diffusion benchmark style keys are in seconds.
    sec_candidates = ["latency_p95"]
    for key in sec_candidates:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value) * 1000.0

    raise ValueError("Cannot find P95 latency field in benchmark result JSON")


@dataclass
class SweepRow:
    rps: float
    requested_num_prompts: int
    target_duration_s: float | None
    completed: int
    failed: int
    request_throughput: float
    request_goodput: float
    slo_attainment_rate: float
    p95_latency_ms: float
    raw_result_file: str


class ServiceOrchestrator:
    """Manage lifecycle of local benchmark services (instances + scheduler)."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger, run_dir: Path) -> None:
        self.logger = logger
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.configs_dir = run_dir / "configs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        deployment = config.get("deployment", {})

        self.num_instances = int(deployment.get("num_instances", 1))
        self.num_gpus = int(deployment.get("num_gpus", self.num_instances))
        self.model = str(deployment.get("serving_model") or config.get("benchmark", {}).get("model", ""))
        self.instance_host = str(deployment.get("instance_host", "127.0.0.1"))
        self.instance_port_base = int(deployment.get("instance_port_base", 19091))
        self.instance_gpu_map = deployment.get("instance_gpu_map", [0])
        self.instance_extra_args = deployment.get("instance_extra_args", [])
        self.stage_config_path = str(deployment.get("stage_config_path", "")).strip()
        self.instance_max_concurrency = int(deployment.get("instance_max_concurrency", 1))
        self.instance_health_timeout_s = int(deployment.get("instance_health_timeout_s", 1200))
        self.auto_kill_port_owner = int(deployment.get("auto_kill_port_owner", 1))

        self.scheduler_host = str(deployment.get("scheduler_host", "127.0.0.1"))
        self.scheduler_port = int(deployment.get("scheduler_port", 18089))
        self.scheduler_algorithm = str(deployment.get("scheduler_algorithm", "estimated_completion_time"))
        self.scheduler_tie_breaker = str(deployment.get("scheduler_tie_breaker", "random"))
        self.scheduler_ewma_alpha = float(deployment.get("scheduler_ewma_alpha", 0.2))

        self._procs: list[tuple[str, subprocess.Popen[str], Path]] = []

    @staticmethod
    def _tail_lines(path: Path, n: int = 40) -> str:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])

    @staticmethod
    def _http_status(url: str, timeout_s: float = 1.0) -> int | None:
        req = urllib_request.Request(url=url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
                return int(getattr(resp, "status", 200))
        except urllib_error.HTTPError as exc:
            return int(exc.code)
        except Exception:
            return None

    def _is_port_listening(self, port: int) -> bool:
        if shutil_which("ss"):
            cp = subprocess.run(["ss", "-ltn", f"( sport = :{port} )"], capture_output=True, text=True, check=False)
            lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
            return len(lines) > 1
        if shutil_which("lsof"):
            cp = subprocess.run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, check=False)
            return bool(cp.stdout.strip())
        return False

    def _get_listen_pids(self, port: int) -> list[int]:
        pids: set[int] = set()
        if shutil_which("lsof"):
            cp = subprocess.run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True, check=False)
            for token in cp.stdout.split():
                if token.isdigit():
                    pids.add(int(token))
        if not pids and shutil_which("fuser"):
            cp = subprocess.run(["fuser", "-n", "tcp", str(port)], capture_output=True, text=True, check=False)
            for token in cp.stdout.split():
                if token.isdigit():
                    pids.add(int(token))
        return sorted(pids)

    @staticmethod
    def _read_pid_cmdline(pid: int) -> str:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return ""
        if not raw:
            return ""
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _pid_matches_managed_service(pid_cmdline: str, label: str) -> bool:
        cmd = pid_cmdline.lower()
        if not cmd:
            return False

        if label.startswith("instance_"):
            # vLLM can be launched either via `vllm serve` CLI or Python module entrypoints.
            return "vllm" in cmd and (" serve" in cmd or "api_server" in cmd)

        if label == "scheduler":
            return "vllm_omni.global_scheduler.server" in cmd

        return False

    def _ensure_port_available(self, port: int, label: str) -> None:
        if not self._is_port_listening(port):
            return
        pids = self._get_listen_pids(port)
        if not pids:
            raise RuntimeError(
                f"Port {port} for {label} is LISTENing but owner PID is not visible; free it manually or change port"
            )
        self.logger.warning("Port %s for %s is occupied by PIDs: %s", port, label, pids)
        if self.auto_kill_port_owner != 1:
            raise RuntimeError(f"Port {port} occupied and auto_kill_port_owner=0")

        safe_pids: list[int] = []
        unsafe_entries: list[str] = []
        for pid in pids:
            cmdline = self._read_pid_cmdline(pid)
            if self._pid_matches_managed_service(cmdline, label):
                safe_pids.append(pid)
            else:
                unsafe_entries.append(f"{pid}:{cmdline or '<unknown>'}")

        if unsafe_entries:
            joined = "; ".join(unsafe_entries)
            raise RuntimeError(
                f"Refusing to kill non-managed process(es) on port {port} for {label}: {joined}"
            )

        if not safe_pids:
            raise RuntimeError(
                f"Port {port} occupied for {label} but no verifiable managed PID found; aborting cleanup"
            )

        for pid in safe_pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)

        deadline = time.time() + 30
        while time.time() < deadline:
            if not self._is_port_listening(port):
                return
            time.sleep(1)

        raise RuntimeError(f"Port {port} still occupied after cleanup attempt")

    def _wait_health(self, name: str, url: str, timeout_s: int, proc: subprocess.Popen[str], log_path: Path) -> None:
        start = time.time()
        while time.time() - start < timeout_s:
            code = self._http_status(url)
            if code == 200:
                return
            if proc.poll() is not None:
                tail = self._tail_lines(log_path, n=80)
                raise RuntimeError(f"{name} exited before health ready. Last logs:\n{tail}")
            time.sleep(2)

        tail = self._tail_lines(log_path, n=80)
        raise RuntimeError(f"Timeout waiting {name} health at {url}. Last logs:\n{tail}")

    def _start_instance(self, idx: int, gpu_id: int, port: int) -> None:
        self._ensure_port_available(port, f"instance_{idx}")
        log_path = self.logs_dir / f"instance_{idx}.log"

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.instance_host,
            "--port",
            str(port),
        ]
        if self.stage_config_path:
            cmd.extend(["--stage-configs-path", self.stage_config_path])
        if isinstance(self.instance_extra_args, list):
            cmd.extend([str(x) for x in self.instance_extra_args])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_f = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, text=True, env=env)
        self._procs.append((f"instance_{idx}", proc, log_path))
        self.logger.info("Started instance_%s pid=%s on gpu=%s port=%s", idx, proc.pid, gpu_id, port)

        health_url = f"http://{self.instance_host}:{port}/health"
        self._wait_health(f"instance_{idx}", health_url, self.instance_health_timeout_s, proc, log_path)

    def _build_scheduler_yaml(self) -> Path:
        path = self.configs_dir / "global_scheduler.yaml"
        lines = [
            "server:",
            f"  host: {self.scheduler_host}",
            f"  port: {self.scheduler_port}",
            "  request_timeout_s: 1800",
            "scheduler:",
            f"  tie_breaker: {self.scheduler_tie_breaker}",
            f"  ewma_alpha: {self.scheduler_ewma_alpha}",
            "policy:",
            "  baseline:",
            f"    algorithm: {self.scheduler_algorithm}",
            "instances:",
        ]
        for i in range(self.num_instances):
            port = self.instance_port_base + i
            lines.extend(
                [
                    f"  - id: worker-{i}",
                    f"    endpoint: http://{self.instance_host}:{port}",
                    "    sp_size: 1",
                    f"    max_concurrency: {self.instance_max_concurrency}",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _start_scheduler(self) -> str:
        self._ensure_port_available(self.scheduler_port, "scheduler")
        yaml_path = self._build_scheduler_yaml()
        log_path = self.logs_dir / "scheduler.log"
        cmd = [sys.executable, "-m", "vllm_omni.global_scheduler.server", "--config", str(yaml_path)]
        log_f = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, text=True)
        self._procs.append(("scheduler", proc, log_path))
        self.logger.info("Started scheduler pid=%s on %s:%s", proc.pid, self.scheduler_host, self.scheduler_port)

        health_url = f"http://{self.scheduler_host}:{self.scheduler_port}/health"
        self._wait_health("scheduler", health_url, 180, proc, log_path)
        return f"http://{self.scheduler_host}:{self.scheduler_port}"

    def start(self) -> str:
        if self.num_instances <= 0:
            raise ValueError("deployment.num_instances must be > 0")
        if self.num_gpus < self.num_instances:
            raise ValueError("deployment.num_gpus must be >= deployment.num_instances")
        if not isinstance(self.instance_gpu_map, list) or len(self.instance_gpu_map) < self.num_instances:
            raise ValueError("deployment.instance_gpu_map must include one GPU id per instance")

        for i in range(self.num_instances):
            gpu = int(self.instance_gpu_map[i])
            port = self.instance_port_base + i
            self._start_instance(i, gpu, port)

        if self.num_instances == 1:
            return f"http://{self.instance_host}:{self.instance_port_base}"
        return self._start_scheduler()

    def stop(self) -> None:
        for name, proc, _ in reversed(self._procs):
            if proc.poll() is not None:
                continue
            self.logger.info("Stopping %s pid=%s", name, proc.pid)
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=15)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
        self._procs.clear()


def shutil_which(cmd: str) -> str | None:
    return shutil.which(cmd)


class BenchmarkRunner:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        benchmark_cfg = config.get("benchmark", {})
        deployment_cfg = config.get("deployment", {})

        self.host: str | None = None
        self.port: int | None = None
        target_base_url = benchmark_cfg.get("target_base_url")
        if target_base_url:
            self.set_target_base_url(str(target_base_url))
        self.model = str(benchmark_cfg.get("model", ""))
        if not self.model:
            raise ValueError("benchmark.model is required")
        self.benchmark_name = str(benchmark_cfg.get("benchmark_name", "unnamed_benchmark"))
        self.benchmark_tool = str(benchmark_cfg.get("benchmark_tool", "diffusion_serving")).strip().lower()

        self.rps_values = _parse_rps_values(benchmark_cfg.get("rps_values", []))
        self.run_mode = str(benchmark_cfg.get("run_mode", "fixed_requests")).strip().lower()
        self.total_duration_s: float | None = None
        self.fixed_num_prompts: int | None = None

        if self.run_mode == "duration":
            self.total_duration_s = float(benchmark_cfg.get("total_duration_s", 0.0))
            if self.total_duration_s <= 0:
                raise ValueError("benchmark.total_duration_s must be > 0 when run_mode=duration")
        else:
            total_requests_raw = benchmark_cfg.get("total_requests", benchmark_cfg.get("num_prompts", 100))
            self.fixed_num_prompts = int(total_requests_raw)
            if self.fixed_num_prompts <= 0:
                raise ValueError("benchmark.total_requests (or benchmark.num_prompts) must be > 0")

        self.endpoint = str(benchmark_cfg.get("endpoint", "/v1/chat/completions"))
        self.backend = str(benchmark_cfg.get("backend", "vllm-omni"))
        self.task = str(benchmark_cfg.get("task", "t2i"))
        self.dataset = str(benchmark_cfg.get("dataset", "trace"))
        self.dataset_path = str(benchmark_cfg.get("dataset_path", "")).strip()
        self.percentile_metrics = str(benchmark_cfg.get("percentile_metrics", "e2el"))
        self.metric_percentiles = str(benchmark_cfg.get("metric_percentiles", "95"))
        self.latency_metric = str(benchmark_cfg.get("latency_metric", "e2el"))
        self.max_retries = int(benchmark_cfg.get("max_retries", 1))
        self.extra_args = benchmark_cfg.get("extra_bench_args", {})
        if not isinstance(self.extra_args, dict):
            raise ValueError("benchmark.extra_bench_args must be a JSON object")

        self.slo_metric = str(benchmark_cfg.get("slo_metric", "e2el"))
        self.slo_base_latency_ms = float(benchmark_cfg.get("slo_base_latency_ms", 0.0))
        self.slo_multiplier = float(benchmark_cfg.get("slo_multiplier", 1.0))
        if self.slo_base_latency_ms <= 0:
            raise ValueError("benchmark.slo_base_latency_ms must be > 0")
        if self.slo_multiplier <= 0:
            raise ValueError("benchmark.slo_multiplier must be > 0")
        self.slo_threshold_ms = self.slo_base_latency_ms * self.slo_multiplier

        configured_output_dir = str(benchmark_cfg.get("output_dir", "")).strip()
        if configured_output_dir:
            self.output_dir = Path(configured_output_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = Path("./benchmark_outputs") / f"run_{ts}"
        self.raw_result_dir = self.output_dir / "raw_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_result_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler_algorithm = str(deployment_cfg.get("scheduler_algorithm", "disabled(single-instance)"))
        if int(deployment_cfg.get("num_instances", 1)) <= 1:
            self.scheduler_algorithm = "disabled(single-instance)"

        self.estimated_total_requests_all_rps = sum(self._num_prompts_for_rps(rps) for rps in self.rps_values)

    def set_target_base_url(self, target_base_url: str) -> None:
        parsed = urlparse(target_base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("benchmark.target_base_url must start with http:// or https://")
        if not parsed.hostname or parsed.port is None:
            raise ValueError("benchmark.target_base_url must include host and port")
        self.host = parsed.hostname
        self.port = parsed.port

    def _num_prompts_for_rps(self, rps: float) -> int:
        if self.run_mode == "duration":
            assert self.total_duration_s is not None
            return max(1, int(math.ceil(rps * self.total_duration_s)))

        assert self.fixed_num_prompts is not None
        return self.fixed_num_prompts

    def _build_cmd(self, rps: float, num_prompts: int, result_filename: str) -> list[str]:
        if self.host is None or self.port is None:
            raise ValueError("Benchmark target URL is not set. Provide benchmark.target_base_url or enable Python-managed services.")

        result_file_path = self.raw_result_dir / result_filename

        if self.benchmark_tool == "diffusion_serving":
            cmd = [
                "python3",
                "benchmarks/diffusion/diffusion_benchmark_serving.py",
                "--base-url",
                f"http://{self.host}:{self.port}",
                "--model",
                self.model,
                "--backend",
                self.backend,
                "--dataset",
                self.dataset,
                "--task",
                self.task,
                "--num-prompts",
                str(num_prompts),
                "--request-rate",
                str(rps),
                "--output-file",
                str(result_file_path),
            ]
            if self.dataset_path:
                cmd.extend(["--dataset-path", self.dataset_path])
            cmd.extend(_to_cli_args(self.extra_args))
            return cmd

        # Backward-compatible fallback for legacy vllm bench serve mode.
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--omni",
            "--backend",
            self.backend,
            "--endpoint",
            self.endpoint,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model",
            self.model,
            "--request-rate",
            str(rps),
            "--num-prompts",
            str(num_prompts),
            "--percentile-metrics",
            self.percentile_metrics,
            "--metric-percentiles",
            self.metric_percentiles,
            "--goodput",
            f"{self.slo_metric}:{self.slo_threshold_ms}",
            "--save-result",
            "--result-dir",
            str(self.raw_result_dir),
            "--result-filename",
            result_filename,
        ]
        cmd.extend(_to_cli_args(self.extra_args))
        return cmd

    def _run_one(self, rps: float) -> SweepRow:
        requested_num_prompts = self._num_prompts_for_rps(rps)
        result_filename = f"result_rps_{str(rps).replace('.', '_')}.json"
        cmd = self._build_cmd(rps, requested_num_prompts, result_filename)
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        self.logger.info("Running benchmark for RPS=%s", rps)
        if self.run_mode == "duration":
            self.logger.info(
                "Duration mode: duration=%.3fs, requested_num_prompts=%d",
                self.total_duration_s,
                requested_num_prompts,
            )
        self.logger.info("Command: %s", cmd_str)

        for attempt in range(1, self.max_retries + 2):
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError as exc:
                self.logger.warning("Attempt %d failed for RPS=%s (exit=%s)", attempt, rps, exc.returncode)
                if attempt >= self.max_retries + 1:
                    raise RuntimeError(f"Benchmark failed after retries for RPS={rps}") from exc

        result_path = self.raw_result_dir / result_filename
        if not result_path.exists():
            raise FileNotFoundError(f"Expected benchmark result file not found: {result_path}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        completed = int(result.get("completed", result.get("completed_requests", 0)))
        failed = int(result.get("failed", result.get("failed_requests", 0)))
        request_throughput = float(result.get("request_throughput", result.get("throughput_qps", 0.0)))

        request_goodput = float(result.get("request_goodput", 0.0))
        if request_goodput <= 0.0 and self.benchmark_tool == "diffusion_serving":
            slo_rate_raw = result.get("slo_attainment_rate")
            if isinstance(slo_rate_raw, (int, float)) and request_throughput > 0:
                request_goodput = float(request_throughput) * float(slo_rate_raw)

        p95_latency_ms = _extract_p95_latency_ms(result, metric=self.latency_metric)

        if request_throughput <= 0:
            slo_rate = 0.0
        else:
            slo_rate = request_goodput / request_throughput
            if not math.isfinite(slo_rate):
                slo_rate = 0.0
            slo_rate = max(0.0, min(1.0, slo_rate))

        self.logger.info(
            "RPS=%s done: throughput=%.4f, goodput=%.4f, slo_rate=%.4f, p95_ms=%.2f",
            rps,
            request_throughput,
            request_goodput,
            slo_rate,
            p95_latency_ms,
        )

        return SweepRow(
            rps=rps,
            requested_num_prompts=requested_num_prompts,
            target_duration_s=self.total_duration_s if self.run_mode == "duration" else None,
            completed=completed,
            failed=failed,
            request_throughput=request_throughput,
            request_goodput=request_goodput,
            slo_attainment_rate=slo_rate,
            p95_latency_ms=p95_latency_ms,
            raw_result_file=str(result_path),
        )

    def _write_summary(self, rows: list[SweepRow]) -> tuple[Path, Path]:
        csv_path = self.output_dir / "summary.csv"
        json_path = self.output_dir / "summary.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rps",
                    "requested_num_prompts",
                    "target_duration_s",
                    "completed",
                    "failed",
                    "request_throughput",
                    "request_goodput",
                    "slo_attainment_rate",
                    "p95_latency_ms",
                    "raw_result_file",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)

        payload = {
            "generated_at": datetime.now().isoformat(),
            "benchmark_name": self.benchmark_name,
            "benchmark_tool": self.benchmark_tool,
            "model": self.model,
            "dataset": self.dataset,
            "task": self.task,
            "scheduler_algorithm": self.scheduler_algorithm,
            "run_mode": self.run_mode,
            "total_duration_s": self.total_duration_s,
            "total_requests_per_rps": self.fixed_num_prompts,
            "estimated_total_requests_all_rps": self.estimated_total_requests_all_rps,
            "slo_metric": self.slo_metric,
            "slo_base_latency_ms": self.slo_base_latency_ms,
            "slo_multiplier": self.slo_multiplier,
            "slo_threshold_ms": self.slo_threshold_ms,
            "latency_metric": self.latency_metric,
            "rows": [row.__dict__ for row in rows],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return csv_path, json_path

    def _plot(self, rows: list[SweepRow]) -> dict[str, Path]:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for plotting. Install it via `pip install matplotlib`.") from exc

        rows = sorted(rows, key=lambda x: x.rps)
        x = [r.rps for r in rows]
        slo_y = [r.slo_attainment_rate for r in rows]
        p95_y = [r.p95_latency_ms for r in rows]

        # 1) Combined view (dual axis)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        line1 = ax1.plot(x, slo_y, marker="o", linewidth=2.0, color="#1f77b4", label="SLO attainment rate")
        ax1.set_xlabel("RPS (req/s)")
        ax1.set_ylabel("SLO Attainment Rate", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        line2 = ax2.plot(x, p95_y, marker="s", linewidth=2.0, color="#d62728", label="P95 Latency (ms)")
        ax2.set_ylabel("P95 Latency (ms)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        title = (
            f"{self.benchmark_name} | Model={self.model.split('/')[-1]} | "
            f"RPS Sweep | Algo={self.scheduler_algorithm} | "
            f"SLO {self.slo_metric}<={self.slo_threshold_ms:.2f}ms"
        )
        ax1.set_title(title)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()
        combined_plot_path = self.output_dir / "slo_attainment_vs_p95_latency.png"
        fig.savefig(combined_plot_path, dpi=180)
        plt.close(fig)

        # 2) SLO-only curve
        fig_slo, ax_slo = plt.subplots(figsize=(10, 6))
        ax_slo.plot(x, slo_y, marker="o", linewidth=2.2, color="#1f77b4", label="SLO attainment rate")
        ax_slo.set_xlabel("RPS (req/s)")
        ax_slo.set_ylabel("SLO Attainment Rate")
        ax_slo.set_ylim(0.0, 1.05)
        ax_slo.grid(True, alpha=0.3)
        ax_slo.set_title(
            f"{self.benchmark_name} | Model={self.model.split('/')[-1]} | "
            f"SLO Attainment vs RPS | SLO {self.slo_metric}<={self.slo_threshold_ms:.2f}ms"
        )
        ax_slo.legend(loc="best")
        fig_slo.tight_layout()
        slo_plot_path = self.output_dir / "slo_attainment_vs_rps.png"
        fig_slo.savefig(slo_plot_path, dpi=180)
        plt.close(fig_slo)

        # 3) P95-only curve
        fig_p95, ax_p95 = plt.subplots(figsize=(10, 6))
        ax_p95.plot(x, p95_y, marker="s", linewidth=2.2, color="#d62728", label="P95 Latency (ms)")
        ax_p95.set_xlabel("RPS (req/s)")
        ax_p95.set_ylabel("P95 Latency (ms)")
        ax_p95.grid(True, alpha=0.3)
        ax_p95.set_title(
            f"{self.benchmark_name} | Model={self.model.split('/')[-1]} | "
            "P95 Latency vs RPS"
        )
        ax_p95.legend(loc="best")
        fig_p95.tight_layout()
        p95_plot_path = self.output_dir / "p95_latency_vs_rps.png"
        fig_p95.savefig(p95_plot_path, dpi=180)
        plt.close(fig_p95)

        return {
            "combined_plot_png": combined_plot_path,
            "slo_plot_png": slo_plot_path,
            "p95_plot_png": p95_plot_path,
        }

    def run(self) -> dict[str, str]:
        rows: list[SweepRow] = []
        for rps in self.rps_values:
            rows.append(self._run_one(rps))

        csv_path, json_path = self._write_summary(rows)
        plot_paths = self._plot(rows)

        return {
            "output_dir": str(self.output_dir),
            "summary_csv": str(csv_path),
            "summary_json": str(json_path),
            "combined_plot_png": str(plot_paths["combined_plot_png"]),
            "slo_plot_png": str(plot_paths["slo_plot_png"]),
            "p95_plot_png": str(plot_paths["p95_plot_png"]),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RPS sweep benchmark and draw SLO/P95 curves")
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON")
    parser.add_argument(
        "--log-dir",
        default="./benchmark_outputs/logs",
        help="Directory to write runner logs",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logger = _setup_logging(Path(args.log_dir))

    cfg = _load_config(Path(args.config))
    runner = BenchmarkRunner(cfg, logger)

    deployment = cfg.get("deployment", {})
    raw_manage_services = deployment.get("manage_services", True)
    if isinstance(raw_manage_services, bool):
        manage_services = raw_manage_services
    elif isinstance(raw_manage_services, str):
        normalized = raw_manage_services.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            manage_services = True
        elif normalized in {"0", "false", "no", "off"}:
            manage_services = False
        else:
            raise ValueError("deployment.manage_services must be a bool or a bool-like string")
    else:
        raise ValueError("deployment.manage_services must be a bool or a bool-like string")

    orchestrator: ServiceOrchestrator | None = None
    try:
        if manage_services:
            orchestrator = ServiceOrchestrator(cfg, logger, runner.output_dir)
            target_url = orchestrator.start()
            logger.info("Resolved benchmark target URL from orchestrator: %s", target_url)
            runner.set_target_base_url(target_url)
        elif runner.host is None or runner.port is None:
            raise ValueError("deployment.manage_services=false requires benchmark.target_base_url in config")

        outputs = runner.run()
    finally:
        if orchestrator is not None:
            orchestrator.stop()

    logger.info("Benchmark finished successfully")
    logger.info("Output directory: %s", outputs["output_dir"])
    logger.info("Summary CSV: %s", outputs["summary_csv"])
    logger.info("Summary JSON: %s", outputs["summary_json"])
    logger.info("Combined Plot PNG: %s", outputs["combined_plot_png"])
    logger.info("SLO Plot PNG: %s", outputs["slo_plot_png"])
    logger.info("P95 Plot PNG: %s", outputs["p95_plot_png"])


if __name__ == "__main__":
    main()
