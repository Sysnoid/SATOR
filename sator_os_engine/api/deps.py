from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from ..runtime.executor import Executor
from ..runtime.jobs import JobStore
from ..security.api_keys import get_api_key
from ..security.client_ip import effective_client_ip, parse_trusted_proxy_cidrs_csv
from ..security.idempotency import IdempotencyStore
from ..security.rate_limit import SimpleRateLimiter
from ..settings import Settings, get_settings

_job_store: JobStore | None = None
_idem_store: IdempotencyStore | None = None
_limiter: SimpleRateLimiter | None = None
_executor: Executor | None = None


def get_job_store(settings: Settings = Depends(get_settings)) -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore(ttl_sec=settings.job_ttl_sec, timeout_sec=settings.job_timeout_sec)
    return _job_store


def get_executor(
    job_store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
) -> Executor:
    global _executor
    if _executor is None:
        _executor = Executor(job_store, max_workers=settings.concurrency, timeout_sec=settings.job_timeout_sec)
    return _executor


def shutdown_executor() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


def run_store_sweeps() -> None:
    """Proactively expire idempotency keys, trim rate-limiter growth, and drop old terminal jobs."""
    from ..observability import metrics

    global _job_store, _idem_store, _limiter
    s = get_settings()
    if _idem_store is not None:
        _idem_store.sweep_expired()
        metrics.idempotency_store_entries.set(len(_idem_store))
    if _limiter is not None:
        _limiter.sweep(max_keys=s.rate_limit_max_keys)
        metrics.rate_limiter_keys.set(len(_limiter))
    if _job_store is not None:
        _job_store.sweep_terminated_past_ttl()
        metrics.job_store_entries.set(len(_job_store))


def get_idempotency_store(settings: Settings = Depends(get_settings)) -> IdempotencyStore:
    global _idem_store
    if _idem_store is None:
        _idem_store = IdempotencyStore(ttl_sec=settings.job_ttl_sec)
    return _idem_store


def rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
) -> str:
    global _limiter
    if _limiter is None:
        _limiter = SimpleRateLimiter(per_minute=settings.rate_limit_per_min)
    ip = effective_client_ip(request, parse_trusted_proxy_cidrs_csv(settings.trusted_proxy_cidrs))
    if not _limiter.allow(api_key, ip):
        from ..observability import metrics

        metrics.rate_limit_rejected_total.inc()
        raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    return api_key


def idempotency(
    api_key: str = Depends(get_api_key),
    idem_store: IdempotencyStore = Depends(get_idempotency_store),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str | None:
    if not idempotency_key:
        return None
    existing = idem_store.get(api_key, idempotency_key)
    return existing
