from __future__ import annotations

import json
from pathlib import Path


RuntimeProfileKey = tuple[str | None, int | None, int | None, int | None, int | None]


def load_runtime_profile(profile_path: str | Path) -> dict[RuntimeProfileKey, float]:
    """Load runtime profile JSON into estimator lookup table.

    Expected JSON shape:
    {
      "profiles": [
        {
          "instance_type": "wan-video-tp2",
          "width": 1280,
          "height": 720,
          "num_frames": 16,
          "steps": 30,
          "latency_ms": 4960
        }
      ]
    }
    """
    path = Path(profile_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Runtime profile file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in runtime profile file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Runtime profile root must be an object in {path}")

    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError(f"Runtime profile file {path} must contain a 'profiles' array")

    profile_map: dict[RuntimeProfileKey, float] = {}
    for index, item in enumerate(profiles):
        if not isinstance(item, dict):
            raise ValueError(f"Runtime profile entry #{index} in {path} must be an object")

        latency_ms = item.get("latency_ms")
        if not isinstance(latency_ms, (int, float)) or latency_ms <= 0:
            raise ValueError(f"Runtime profile entry #{index} in {path} has invalid latency_ms")

        key: RuntimeProfileKey = (
            _optional_str(item.get("instance_type"), "instance_type", index, path),
            _optional_int(item.get("width"), "width", index, path),
            _optional_int(item.get("height"), "height", index, path),
            _optional_int(item.get("num_frames"), "num_frames", index, path),
            _optional_int(item.get("steps"), "steps", index, path),
        )
        profile_map[key] = float(latency_ms) / 1000.0

    return profile_map


def _optional_int(value: object, field_name: str, index: int, path: Path) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"Runtime profile entry #{index} in {path} has invalid {field_name}")
    return value


def _optional_str(value: object, field_name: str, index: int, path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Runtime profile entry #{index} in {path} has invalid {field_name}")
    return value
