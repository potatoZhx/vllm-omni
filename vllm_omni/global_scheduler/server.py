from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vllm_omni.version import __version__

from .config import GlobalSchedulerConfig, load_config


def create_app(config: GlobalSchedulerConfig) -> FastAPI:
    app = FastAPI(title="vLLM-Omni Global Scheduler", version=__version__)
    app.state.global_scheduler_config = config

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            content={
                "status": "ok",
                "scheduler": config.scheduler.type,
                "version": __version__,
            }
        )

    return app


def run_server(config_path: str) -> None:
    config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vLLM-Omni global scheduler server")
    parser.add_argument("--config", required=True, help="Path to global scheduler YAML config")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_server(args.config)


if __name__ == "__main__":
    main()
