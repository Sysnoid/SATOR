from __future__ import annotations

from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

from ..settings import Settings
from .client_ip import effective_client_ip, parse_trusted_proxy_cidrs_csv


class IPFilterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self._trusted = parse_trusted_proxy_cidrs_csv(settings.trusted_proxy_cidrs)

    async def dispatch(self, request: Request, call_next: Callable):
        ip = effective_client_ip(request, self._trusted)
        if self.settings.ip_blacklist and ip in self.settings.ip_blacklist:
            return JSONResponse({"detail": "IP blocked"}, status_code=HTTP_403_FORBIDDEN)
        if self.settings.ip_whitelist and ip not in self.settings.ip_whitelist:
            return JSONResponse({"detail": "IP not allowed"}, status_code=HTTP_403_FORBIDDEN)
        return await call_next(request)
