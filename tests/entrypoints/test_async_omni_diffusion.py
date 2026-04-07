from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import vllm_omni.entrypoints.async_omni_diffusion as async_od_module
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeODConfig:
    def __init__(self, model: str) -> None:
        self.model = model
        self.model_class_name = None
        self.omni_kv_config: dict[str, object] = {}
        self.tf_model_config = None
        self.cfg_kv_collect_func = None
        self.uses_step_level_scheduler = False

    def update_multimodal_support(self) -> None:
        return None


class _FakeDiffusionEngine:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    def step(self, request):
        self.requests.append(request)
        return [SimpleNamespace(request_id=None, images=["img"])]

    def close(self) -> None:
        self.closed = True


def _patch_async_diffusion_deps(
    monkeypatch: pytest.MonkeyPatch,
    fake_engine: _FakeDiffusionEngine,
) -> None:
    monkeypatch.setattr(
        async_od_module,
        "get_hf_file_to_dict",
        lambda file_name, model: {"_class_name": "FakePipeline"} if file_name == "model_index.json" else {},
    )
    monkeypatch.setattr(
        async_od_module.TransformerConfig,
        "from_dict",
        staticmethod(lambda cfg: SimpleNamespace()),
    )
    monkeypatch.setattr(
        async_od_module.DiffusionEngine,
        "make_engine",
        staticmethod(lambda od_config: fake_engine),
    )


def test_resolve_executor_max_workers_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(async_od_module._EXECUTOR_MAX_WORKERS_ENV, "7")

    assert async_od_module._resolve_executor_max_workers() == 7


def test_async_omni_diffusion_generate_logs_executor_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_engine = _FakeDiffusionEngine()
    _patch_async_diffusion_deps(monkeypatch, fake_engine)
    monkeypatch.setenv(async_od_module._EXECUTOR_MAX_WORKERS_ENV, "3")
    info_messages: list[str] = []

    original_logger_info = async_od_module.logger.info

    def _capture_info(message: str, *args, **kwargs) -> None:
        rendered = message % args if args else message
        info_messages.append(rendered)
        original_logger_info(message, *args, **kwargs)

    monkeypatch.setattr(async_od_module.logger, "info", _capture_info)

    app = async_od_module.AsyncOmniDiffusion(
        model="dummy-model",
        od_config=_FakeODConfig(model="dummy-model"),
    )
    try:
        result = asyncio.run(
            app.generate(
                prompt="hello",
                sampling_params=OmniDiffusionSamplingParams(),
                request_id="req-1",
            )
        )
    finally:
        app.shutdown()

    assert app._executor._max_workers == 3
    assert fake_engine.requests
    assert fake_engine.requests[0].request_ids == ["req-1"]
    assert result.request_id == "req-1"
    assert fake_engine.closed is True

    messages = "\n".join(info_messages)
    assert "[AsyncDiffusionExecutorEnqueue] req=req-1 kind=single" in messages
    assert "[AsyncDiffusionExecutorStart] req=req-1 kind=single" in messages
    assert "[AsyncDiffusionExecutorDone] req=req-1 kind=single" in messages
