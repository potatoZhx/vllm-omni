# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeProfileRecord:
    task_type: str
    width: int
    height: int
    num_frames: int
    steps: int
    latency_s: float
    instance_type: str | None = None


class RuntimeProfileEstimator:
    """Estimate diffusion runtime from profiled JSON records with fallback scaling."""

    def __init__(self, records: list[RuntimeProfileRecord] | None = None):
        self.records = records or []
        self._grouped: dict[tuple[str, int, int, int], list[tuple[int, float]]] = defaultdict(list)
        for record in self.records:
            self._grouped[(record.task_type, record.width, record.height, record.num_frames)].append(
                (record.steps, record.latency_s)
            )
        for key in self._grouped:
            self._grouped[key].sort()

    @classmethod
    def from_path(cls, profile_path: str | None, instance_type: str | None = None) -> "RuntimeProfileEstimator":
        if not profile_path:
            return cls()

        path = Path(profile_path)
        if not path.exists():
            logger.warning("Runtime profile path %s does not exist; using heuristic fallback.", path)
            return cls()

        files = [path] if path.is_file() else sorted(path.glob("*.json"))
        records: list[RuntimeProfileRecord] = []
        for json_file in files:
            try:
                payload = json.loads(json_file.read_text())
            except Exception as exc:
                logger.warning("Failed to parse runtime profile %s: %s", json_file, exc)
                continue
            records.extend(cls._parse_payload(payload, instance_type=instance_type))

        logger.info("Loaded %d runtime profile records from %s", len(records), path)
        return cls(records)

    @staticmethod
    def _parse_payload(payload: object, instance_type: str | None = None) -> list[RuntimeProfileRecord]:
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = payload.get("profiles") or payload.get("entries") or []
        else:
            entries = []

        records: list[RuntimeProfileRecord] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_instance = entry.get("instance_type")
            if instance_type and entry_instance not in (None, instance_type):
                continue

            task_type = str(entry.get("task_type", "image")).lower()
            width = entry.get("width")
            height = entry.get("height")
            resolution = entry.get("resolution")
            if (width is None or height is None) and resolution is not None:
                width = resolution
                height = resolution
            if width is None or height is None:
                continue

            num_frames = int(entry.get("num_frames", 1) or 1)
            steps = entry.get("steps", entry.get("num_inference_steps"))
            if steps is None:
                continue

            latency_ms = entry.get("latency_ms")
            latency_s = entry.get("latency_s")
            if latency_s is None and latency_ms is not None:
                latency_s = float(latency_ms) / 1000.0
            if latency_s is None:
                continue

            records.append(
                RuntimeProfileRecord(
                    task_type=task_type,
                    width=int(width),
                    height=int(height),
                    num_frames=num_frames,
                    steps=int(steps),
                    latency_s=max(float(latency_s), 0.001),
                    instance_type=str(entry_instance) if entry_instance is not None else None,
                )
            )
        return records

    def estimate_runtime_s(
        self,
        *,
        task_type: str,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        fallback_s: float,
    ) -> float:
        if not self._grouped:
            return fallback_s

        task_type = task_type.lower()
        key = (task_type, int(width), int(height), int(num_frames))
        points = self._grouped.get(key)
        if points is not None:
            return self._interpolate_steps(points, int(steps))

        nearest = self._find_nearest_group(task_type=task_type, width=width, height=height, num_frames=num_frames)
        if nearest is None:
            return fallback_s

        nearest_key, nearest_points = nearest
        base_estimate = self._interpolate_steps(nearest_points, int(steps))
        _, base_width, base_height, base_frames = nearest_key
        area_scale = (float(width) * float(height)) / float(base_width * base_height)
        frame_scale = float(num_frames) / float(base_frames)
        return max(base_estimate * area_scale * frame_scale, 0.001)

    @staticmethod
    def _interpolate_steps(points: list[tuple[int, float]], steps: int) -> float:
        if len(points) == 1:
            base_steps, base_latency = points[0]
            return max(base_latency * (float(steps) / float(base_steps)), 0.001)

        for profiled_steps, profiled_latency in points:
            if profiled_steps == steps:
                return profiled_latency

        if steps <= points[0][0]:
            base_steps, base_latency = points[0]
            return max(base_latency * (float(steps) / float(base_steps)), 0.001)

        if steps >= points[-1][0]:
            base_steps, base_latency = points[-1]
            return max(base_latency * (float(steps) / float(base_steps)), 0.001)

        for (left_steps, left_latency), (right_steps, right_latency) in zip(points, points[1:]):
            if left_steps <= steps <= right_steps:
                ratio = float(steps - left_steps) / float(right_steps - left_steps)
                return left_latency + ratio * (right_latency - left_latency)

        return points[-1][1]

    def _find_nearest_group(
        self, *, task_type: str, width: int, height: int, num_frames: int
    ) -> tuple[tuple[str, int, int, int], list[tuple[int, float]]] | None:
        candidates = [(key, points) for key, points in self._grouped.items() if key[0] == task_type]
        if not candidates:
            return None

        def _score(item: tuple[tuple[str, int, int, int], list[tuple[int, float]]]) -> float:
            (_, base_width, base_height, base_frames), _ = item
            area_ratio = (float(width) * float(height)) / float(base_width * base_height)
            frame_ratio = float(num_frames) / float(base_frames)
            return abs(math.log(area_ratio)) + abs(math.log(frame_ratio))

        return min(candidates, key=_score)
