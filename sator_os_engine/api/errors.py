from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..settings import get_settings

log = logging.getLogger("sator")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN001
        log.exception("unhandled_error path=%s", request.url.path, exc_info=exc)
        settings = get_settings()
        if settings.expose_error_details:
            return JSONResponse({"code": "INTERNAL_ERROR", "message": str(exc)}, status_code=500)
        return JSONResponse(
            {"code": "INTERNAL_ERROR", "message": "Internal server error"},
            status_code=500,
        )
