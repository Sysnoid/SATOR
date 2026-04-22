from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import Response

from ..security.ip_filters import IPFilterMiddleware
from ..settings import Settings, get_settings
from .deps import run_store_sweeps, shutdown_executor
from .errors import register_error_handlers
from .routes.jobs import router as jobs_router
from .routes.optimize import router as optimize_router
from .routes.reconstruct import router as reconstruct_router

log = logging.getLogger("sator")


async def _periodic_store_sweep() -> None:
    while True:
        try:
            await asyncio.sleep(float(get_settings().store_sweep_interval_sec))
            run_store_sweeps()
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001
            log.exception("store_sweep_failed")


@asynccontextmanager
async def _lifespan(_: FastAPI):
    run_store_sweeps()
    sweep_task = asyncio.create_task(_periodic_store_sweep())
    try:
        yield
    finally:
        sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweep_task
        shutdown_executor()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="SATOR OS Engine", version="0.1.0", lifespan=_lifespan)

    # Middlewares
    app.add_middleware(IPFilterMiddleware, settings=settings)

    # Routes
    app.include_router(optimize_router, prefix="/v1", tags=["optimize"])
    app.include_router(reconstruct_router, prefix="/v1", tags=["reconstruct"])
    app.include_router(jobs_router, prefix="/v1", tags=["jobs"])
    register_error_handlers(app)

    if settings.enable_metrics:

        @app.get("/metrics", include_in_schema=True)
        def prometheus_metrics() -> Response:
            from ..observability import metrics

            if not metrics.prom_available():
                return Response(
                    "prometheus_client is not installed; pip install prometheus-client",
                    status_code=503,
                    media_type="text/plain",
                )
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/livez")
    def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        return {"status": "ready"}

    return app
