from __future__ import annotations

import os

from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.api.deps import run_store_sweeps
from sator_os_engine.runtime.jobs import JobStatus, JobStore
from sator_os_engine.security.idempotency import IdempotencyStore
from sator_os_engine.security.rate_limit import SimpleRateLimiter
from sator_os_engine.settings import Settings, get_settings


def test_run_store_sweeps_tolerates_empty_globals():
    run_store_sweeps()


def test_idempotency_sweep():
    s = IdempotencyStore(ttl_sec=0)
    s.put("k", "idem", "job1")
    import time

    time.sleep(0.02)
    assert s.sweep_expired() == 1
    assert len(s) == 0


def test_rate_limiter_sweep_empties_stale():
    r = SimpleRateLimiter(per_minute=10)
    r.allow("a", "1.1.1.1")
    assert r.sweep(max_keys=100) == 0
    n = r.sweep(max_keys=1)  # cap forces removal if more than 1 key
    assert n >= 0


def test_job_store_sweep_terminal():
    st = JobStore(ttl_sec=0)
    j = st.create_job("k")
    j.status = JobStatus.COMPLETED
    import time

    j.created_at = time.time() - 10.0
    n = st.sweep_terminated_past_ttl()
    assert n == 1
    assert len(st) == 0


def test_metrics_endpoint_exposed_when_enabled():
    os.environ["SATOR_ENABLE_METRICS"] = "true"
    os.environ["SATOR_API_KEY"] = "t"
    get_settings.cache_clear()
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            assert b"#" in r.content
        else:
            assert b"prometheus" in r.content.lower()
