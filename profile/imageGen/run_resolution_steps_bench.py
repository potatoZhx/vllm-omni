#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GlobalConfig:
    model: str
    host: str
    port: int
    ready_path: str
    request_path: str
    ready_timeout_sec: int
    poll_interval_sec: float
    stop_timeout_sec: int
    max_gpu_per_case: int
    log_root: str
    prompt: str
    negative_prompt: str
    num_outputs: int
    warmup_runs_per_combo: int
    measured_runs_per_combo: int
    request_timeout_sec: int
    inter_run_sleep_sec: float
    seed_base: int
    use_fixed_seed: bool
    resolutions: list[str]
    step_values: list[int]
    extra_cli_args: list[str]


@dataclass
class CaseConfig:
    case_id: str
    case_name: str
    enabled: bool
    sp: int
    cfg: int
    tp: int
    vae_use_slicing: bool
    vae_use_tiling: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image generation resolution/steps benchmarks.")
    parser.add_argument("--config", required=True, help="Path to YAML benchmark config")
    parser.add_argument("--model", default="", help="Optional model override")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_global_config(cfg: dict[str, Any], model_override: str) -> GlobalConfig:
    g = cfg.get("global", {})
    model = model_override if model_override else str(g.get("model", "Qwen/Qwen-Image"))
    return GlobalConfig(
        model=model,
        host=str(g.get("host", "127.0.0.1")),
        port=int(g.get("port", 8091)),
        ready_path=str(g.get("ready_path", "/v1/models")),
        request_path=str(g.get("request_path", "/v1/images/generations")),
        ready_timeout_sec=int(g.get("ready_timeout_sec", 600)),
        poll_interval_sec=float(g.get("poll_interval_sec", 2)),
        stop_timeout_sec=int(g.get("stop_timeout_sec", 60)),
        max_gpu_per_case=int(g.get("max_gpu_per_case", 8)),
        log_root=str(g.get("log_root", "./image_steps_results")),
        prompt=str(g.get("prompt", "a photo of a cute cat in a garden, ultra detailed")),
        negative_prompt=str(g.get("negative_prompt", "")),
        num_outputs=int(g.get("num_outputs", 1)),
        warmup_runs_per_combo=int(g.get("warmup_runs_per_combo", 1)),
        measured_runs_per_combo=int(g.get("measured_runs_per_combo", 5)),
        request_timeout_sec=int(g.get("request_timeout_sec", 1800)),
        inter_run_sleep_sec=float(g.get("inter_run_sleep_sec", 0)),
        seed_base=int(g.get("seed_base", 1234)),
        use_fixed_seed=bool(g.get("use_fixed_seed", False)),
        resolutions=[str(x) for x in g.get("resolutions", ["512x512", "768x768", "1024x1024"])],
        step_values=[int(x) for x in g.get("step_values", [5, 10, 50])],
        extra_cli_args=[str(x) for x in g.get("extra_cli_args", [])],
    )


def parse_cases(cfg: dict[str, Any]) -> list[CaseConfig]:
    cases = []
    for case in cfg.get("cases", []):
        cases.append(
            CaseConfig(
                case_id=str(case.get("id", "")),
                case_name=str(case.get("name", "")),
                enabled=bool(case.get("enabled", True)),
                sp=int(case.get("sp", 1)),
                cfg=int(case.get("cfg", 1)),
                tp=int(case.get("tp", 1)),
                vae_use_slicing=bool(case.get("vae_use_slicing", False)),
                vae_use_tiling=bool(case.get("vae_use_tiling", False)),
            )
        )
    return cases


def ensure_summary_csv(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "case_name",
                "gpu",
                "sp",
                "cfg",
                "tp",
                "vae_use_slicing",
                "vae_use_tiling",
                "resolution",
                "steps",
                "first_startup_s",
                "warmup_runs",
                "measured_runs",
                "latency_mean_s",
                "latency_std_s",
                "status",
                "note",
            ]
        )


def write_summary_row(summary_csv: Path, row: list[Any]) -> None:
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def append_run_row(run_csv: Path, row: list[Any]) -> None:
    with open(run_csv, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def build_server_cmd(global_cfg: GlobalConfig, case: CaseConfig, run_log: Path) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        "--model",
        global_cfg.model,
        "--omni",
        "--port",
        str(global_cfg.port),
        "--host",
        global_cfg.host,
        "--log-file",
        str(run_log),
        "--ulysses-degree",
        str(case.sp),
        "--cfg-parallel-size",
        str(case.cfg),
        "--tensor-parallel-size",
        str(case.tp),
    ]
    if case.vae_use_slicing:
        cmd.append("--vae-use-slicing")
    if case.vae_use_tiling:
        cmd.append("--vae-use-tiling")
    cmd.extend(global_cfg.extra_cli_args)
    return cmd


def start_server(cmd: list[str], run_log: Path) -> subprocess.Popen[Any]:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(run_log, "w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_file.close()
    return proc


def stop_server(proc: subprocess.Popen[Any] | None, timeout_sec: int) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_ready(url: str, timeout_sec: int, interval_sec: float) -> tuple[bool, float]:
    start = time.monotonic()
    while True:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, time.monotonic() - start
                code = resp.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        except Exception:
            code = "NA"

        waited = int(time.monotonic() - start)
        print(f"  waiting ready: {url} (http={code}, waited={waited}s/{timeout_sec}s)")
        if time.monotonic() - start >= timeout_sec:
            return False, 0.0
        time.sleep(interval_sec)


def build_payload(global_cfg: GlobalConfig, size: str, steps: int, seed: int) -> bytes:
    payload: dict[str, Any] = {
        "prompt": global_cfg.prompt,
        "size": size,
        "n": global_cfg.num_outputs,
        "num_inference_steps": steps,
        "seed": seed,
    }
    if global_cfg.negative_prompt:
        payload["negative_prompt"] = global_cfg.negative_prompt
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def request_once(endpoint: str, payload: bytes, timeout_sec: int) -> tuple[str, float, str]:
    # Use curl subprocess instead of urllib to avoid long, opaque hangs on some NPU environments.
    start = time.monotonic()
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        endpoint,
        "-H",
        "Content-Type: application/json",
        "--max-time",
        str(timeout_sec),
        "--data-binary",
        payload.decode("utf-8"),
        "-o",
        "/dev/null",
        "-w",
        "%{http_code},%{time_total}",
    ]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=max(timeout_sec + 5, 10),
            check=False,
        )
        latency = time.monotonic() - start

        output = (result.stdout or "").strip()
        if output:
            parts = output.split(",", 1)
            if len(parts) == 2:
                http_code = parts[0].strip() or "000"
                return http_code, latency, "" if http_code == "200" else f"http-{http_code}"

        if result.returncode == 28:
            return "000", latency, "request-timeout"
        return "000", latency, "request-error"
    except subprocess.TimeoutExpired:
        latency = time.monotonic() - start
        return "000", latency, "request-timeout"
    except Exception:
        latency = time.monotonic() - start
        return "000", latency, "request-error"


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def write_failed_combo_summaries(
    summary_csv: Path,
    combos: list[tuple[str, int]],
    start_idx: int,
    case: CaseConfig,
    gpu_count: int,
    first_startup_s: str,
    global_cfg: GlobalConfig,
    note: str,
    status: str = "FAIL",
) -> None:
    for idx in range(start_idx, len(combos)):
        size, steps = combos[idx]
        write_summary_row(
            summary_csv,
            [
                case.case_id,
                case.case_name,
                gpu_count,
                case.sp,
                case.cfg,
                case.tp,
                str(case.vae_use_slicing).lower(),
                str(case.vae_use_tiling).lower(),
                size,
                steps,
                first_startup_s,
                global_cfg.warmup_runs_per_combo,
                global_cfg.measured_runs_per_combo,
                "",
                "",
                status,
                note,
            ],
        )


def should_abort_case(note: str) -> bool:
    return note in {"request-timeout", "request-error", "service-exited"}


def run_case(summary_csv: Path, run_root: Path, global_cfg: GlobalConfig, case: CaseConfig) -> None:
    if not case.enabled:
        return

    combos = [(size, steps) for size in global_cfg.resolutions for steps in global_cfg.step_values]
    gpu_count = case.sp * case.cfg * case.tp

    if gpu_count > global_cfg.max_gpu_per_case:
        print(f"[SKIP] case {case.case_id} ({case.case_name}) requires {gpu_count} GPUs (> {global_cfg.max_gpu_per_case})")
        write_failed_combo_summaries(
            summary_csv,
            combos,
            0,
            case,
            gpu_count,
            "",
            global_cfg,
            f"requires>{global_cfg.max_gpu_per_case}gpus",
            status="SKIP",
        )
        return

    case_dir = run_root / f"case_{case.case_id}_{sanitize_name(case.case_name)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    run_csv = case_dir / "runs.csv"
    run_log = case_dir / "service.log"

    with open(run_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "case_name",
                "gpu",
                "sp",
                "cfg",
                "tp",
                "vae_use_slicing",
                "vae_use_tiling",
                "resolution",
                "steps",
                "run_type",
                "run_idx",
                "seed",
                "http_code",
                "latency_s",
                "status",
                "error",
            ]
        )

    cmd = build_server_cmd(global_cfg, case, run_log)
    ready_url = f"http://{global_cfg.host}:{global_cfg.port}{global_cfg.ready_path}"
    infer_url = f"http://{global_cfg.host}:{global_cfg.port}{global_cfg.request_path}"

    print(f"[CASE {case.case_id}] {case.case_name}")
    print("  command:", " ".join(cmd))
    print(f"  ready check: {ready_url}")
    print(f"  infer endpoint: {infer_url}")
    print(f"  service log: {run_log}")

    proc: subprocess.Popen[Any] | None = None
    first_startup_s = ""

    try:
        proc = start_server(cmd, run_log)

        ready_ok, startup_s = wait_ready(ready_url, global_cfg.ready_timeout_sec, global_cfg.poll_interval_sec)
        if not ready_ok:
            print(f"[FAIL] case {case.case_id} startup did not become ready in {global_cfg.ready_timeout_sec}s")
            write_failed_combo_summaries(
                summary_csv,
                combos,
                0,
                case,
                gpu_count,
                "",
                global_cfg,
                "startup-timeout",
            )
            return

        first_startup_s = f"{startup_s:.6f}"
        print(f"  first startup ready in {first_startup_s}s")

        for combo_idx, (size, steps) in enumerate(combos):
            print(f"  [COMBO] size={size} steps={steps}")
            combo_csv = case_dir / f"combo_{size}_{steps}.csv"
            measured_latencies: list[float] = []
            combo_note = ""

            with open(combo_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["run_idx", "seed", "http_code", "latency_s", "status", "error"])

            for w in range(1, global_cfg.warmup_runs_per_combo + 1):
                if proc.poll() is not None:
                    combo_note = "service-exited"
                    break

                seed = global_cfg.seed_base if global_cfg.use_fixed_seed else (global_cfg.seed_base + w)
                payload = build_payload(global_cfg, size, steps, seed)
                print(
                    f"    sending warmup request {w}/{global_cfg.warmup_runs_per_combo} "
                    f"(seed={seed}, timeout={global_cfg.request_timeout_sec}s)"
                )
                http_code, latency_s, err = request_once(infer_url, payload, global_cfg.request_timeout_sec)

                status = "PASS" if http_code == "200" else "FAIL"
                if status == "FAIL":
                    combo_note = err or f"warmup-http-{http_code}"

                append_run_row(
                    run_csv,
                    [
                        case.case_id,
                        case.case_name,
                        gpu_count,
                        case.sp,
                        case.cfg,
                        case.tp,
                        str(case.vae_use_slicing).lower(),
                        str(case.vae_use_tiling).lower(),
                        size,
                        steps,
                        "warmup",
                        w,
                        seed,
                        http_code,
                        f"{latency_s:.6f}",
                        status,
                        combo_note if status == "FAIL" else "",
                    ],
                )
                print(f"    warmup {w}/{global_cfg.warmup_runs_per_combo}: http={http_code} latency={latency_s:.6f}s")

                if status == "FAIL":
                    break
                if global_cfg.inter_run_sleep_sec > 0:
                    time.sleep(global_cfg.inter_run_sleep_sec)

            if combo_note:
                if should_abort_case(combo_note):
                    write_failed_combo_summaries(
                        summary_csv,
                        combos,
                        combo_idx,
                        case,
                        gpu_count,
                        first_startup_s,
                        global_cfg,
                        f"case-aborted:{combo_note}",
                    )
                    print(f"  [ABORT] case {case.case_id} due to {combo_note}, moving to next case")
                    return

                write_summary_row(
                    summary_csv,
                    [
                        case.case_id,
                        case.case_name,
                        gpu_count,
                        case.sp,
                        case.cfg,
                        case.tp,
                        str(case.vae_use_slicing).lower(),
                        str(case.vae_use_tiling).lower(),
                        size,
                        steps,
                        first_startup_s,
                        global_cfg.warmup_runs_per_combo,
                        global_cfg.measured_runs_per_combo,
                        "",
                        "",
                        "FAIL",
                        combo_note,
                    ],
                )
                continue

            for run_idx in range(1, global_cfg.measured_runs_per_combo + 1):
                if proc.poll() is not None:
                    combo_note = "service-exited"
                    break

                seed = (
                    global_cfg.seed_base
                    if global_cfg.use_fixed_seed
                    else (global_cfg.seed_base + global_cfg.warmup_runs_per_combo + run_idx)
                )
                payload = build_payload(global_cfg, size, steps, seed)
                print(
                    f"    sending measure request {run_idx}/{global_cfg.measured_runs_per_combo} "
                    f"(seed={seed}, timeout={global_cfg.request_timeout_sec}s)"
                )
                http_code, latency_s, err = request_once(infer_url, payload, global_cfg.request_timeout_sec)
                status = "PASS" if http_code == "200" else "FAIL"
                if status == "FAIL":
                    combo_note = err or f"http-{http_code}"

                with open(combo_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            run_idx,
                            seed,
                            http_code,
                            f"{latency_s:.6f}",
                            status,
                            combo_note if status == "FAIL" else "",
                        ]
                    )

                append_run_row(
                    run_csv,
                    [
                        case.case_id,
                        case.case_name,
                        gpu_count,
                        case.sp,
                        case.cfg,
                        case.tp,
                        str(case.vae_use_slicing).lower(),
                        str(case.vae_use_tiling).lower(),
                        size,
                        steps,
                        "measure",
                        run_idx,
                        seed,
                        http_code,
                        f"{latency_s:.6f}",
                        status,
                        combo_note if status == "FAIL" else "",
                    ],
                )
                print(
                    f"    measure {run_idx}/{global_cfg.measured_runs_per_combo}: "
                    f"http={http_code} latency={latency_s:.6f}s"
                )

                if status == "FAIL":
                    break
                measured_latencies.append(latency_s)
                if global_cfg.inter_run_sleep_sec > 0:
                    time.sleep(global_cfg.inter_run_sleep_sec)

            if combo_note:
                if should_abort_case(combo_note):
                    write_failed_combo_summaries(
                        summary_csv,
                        combos,
                        combo_idx,
                        case,
                        gpu_count,
                        first_startup_s,
                        global_cfg,
                        f"case-aborted:{combo_note}",
                    )
                    print(f"  [ABORT] case {case.case_id} due to {combo_note}, moving to next case")
                    return

                write_summary_row(
                    summary_csv,
                    [
                        case.case_id,
                        case.case_name,
                        gpu_count,
                        case.sp,
                        case.cfg,
                        case.tp,
                        str(case.vae_use_slicing).lower(),
                        str(case.vae_use_tiling).lower(),
                        size,
                        steps,
                        first_startup_s,
                        global_cfg.warmup_runs_per_combo,
                        global_cfg.measured_runs_per_combo,
                        "",
                        "",
                        "FAIL",
                        combo_note,
                    ],
                )
                continue

            if not measured_latencies:
                write_summary_row(
                    summary_csv,
                    [
                        case.case_id,
                        case.case_name,
                        gpu_count,
                        case.sp,
                        case.cfg,
                        case.tp,
                        str(case.vae_use_slicing).lower(),
                        str(case.vae_use_tiling).lower(),
                        size,
                        steps,
                        first_startup_s,
                        global_cfg.warmup_runs_per_combo,
                        global_cfg.measured_runs_per_combo,
                        "",
                        "",
                        "FAIL",
                        "no-measured-results",
                    ],
                )
                continue

            mean_s = statistics.mean(measured_latencies)
            std_s = statistics.stdev(measured_latencies) if len(measured_latencies) > 1 else 0.0
            write_summary_row(
                summary_csv,
                [
                    case.case_id,
                    case.case_name,
                    gpu_count,
                    case.sp,
                    case.cfg,
                    case.tp,
                    str(case.vae_use_slicing).lower(),
                    str(case.vae_use_tiling).lower(),
                    size,
                    steps,
                    first_startup_s,
                    global_cfg.warmup_runs_per_combo,
                    global_cfg.measured_runs_per_combo,
                    f"{mean_s:.6f}",
                    f"{std_s:.6f}",
                    "PASS",
                    "",
                ],
            )
            print(f"  [PASS] size={size} steps={steps} mean={mean_s:.6f}s std={std_s:.6f}s")

        print(f"[CASE {case.case_id}] done")
    finally:
        # Forcefully recycle process group to avoid a hung case blocking later cases.
        stop_server(proc, global_cfg.stop_timeout_sec)


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(str(cfg_path))
    global_cfg = build_global_config(cfg, args.model)

    if args.model:
        print(f"Using model override: {global_cfg.model}")

    script_dir = Path(__file__).resolve().parent
    run_root = (script_dir / global_cfg.log_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    summary_csv = run_root / "summary.csv"
    ensure_summary_csv(summary_csv)

    for case in parse_cases(cfg):
        run_case(summary_csv, run_root, global_cfg, case)

    print(f"Done. summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
