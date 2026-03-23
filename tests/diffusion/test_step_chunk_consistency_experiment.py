# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "diffusion" / "step_chunk_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("step_chunk_consistency", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _encode_png_b64(color: tuple[int, int, int], compress_level: int) -> str:
    image = Image.new("RGB", (8, 8), color=color)
    buf = BytesIO()
    image.save(buf, format="PNG", compress_level=compress_level)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_image_fingerprint_uses_pixel_content_not_png_bytes():
    module = _load_module()

    fingerprints = module.fingerprint_image_payloads(
        [
            _encode_png_b64((255, 0, 0), compress_level=0),
            _encode_png_b64((255, 0, 0), compress_level=9),
        ]
    )

    assert fingerprints[0].raw_sha256 != fingerprints[1].raw_sha256
    assert fingerprints[0].semantic_sha256 == fingerprints[1].semantic_sha256
    assert fingerprints[0].size == [8, 8]
    assert fingerprints[0].mode == "RGB"


def test_compare_fingerprints_detects_semantic_mismatch():
    module = _load_module()

    baseline = module.fingerprint_image_payloads([_encode_png_b64((255, 0, 0), compress_level=6)])
    candidate = module.fingerprint_image_payloads([_encode_png_b64((0, 255, 0), compress_level=6)])

    exact_match, semantic_match = module.compare_fingerprints(baseline, candidate)

    assert exact_match is False
    assert semantic_match is False


def test_extract_b64_items_requires_non_empty_data():
    module = _load_module()

    with pytest.raises(ValueError, match="non-empty 'data' list"):
        module.extract_b64_items({"data": []})
