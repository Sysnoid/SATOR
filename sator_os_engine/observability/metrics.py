from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    _MISSING = False
except ImportError:  # pragma: no cover - install prometheus-client for real metrics
    _MISSING = True

    class _Noop:
        def labels(self, **_kwargs: object) -> _Noop:
            return self

        def inc(self, _amount: float = 1) -> None:
            pass

        def dec(self, _amount: float = 1) -> None:
            pass

        def observe(self, _value: float) -> None:
            pass

        def set(self, _value: float) -> None:
            pass

    def _factory(*_a: object, **_k: object) -> _Noop:
        return _Noop()

    Counter = Gauge = Histogram = _factory  # type: ignore[misc, assignment]

# One registry (default) — suitable for a single process.

jobs_created = Counter("sator_jobs_created_total", "Number of jobs created")

jobs_finished = Counter(
    "sator_jobs_finished_total",
    "Jobs that reached a terminal state",
    ["outcome"],
)

job_duration_seconds = Histogram(
    "sator_job_duration_seconds",
    "Wall time to complete a job (worker thread + executor)",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

jobs_in_flight = Gauge("sator_jobs_in_flight", "Jobs currently in RUNNING state in the pool")

rate_limit_rejected_total = Counter("sator_rate_limit_rejected_total", "HTTP requests rejected by rate limiter")

idempotency_store_entries = Gauge(
    "sator_idempotency_store_entries", "Rows in the in-memory idempotency map (after last sweep)"
)

rate_limiter_keys = Gauge("sator_rate_limiter_keys", "Active (api_key, ip) keys in the rate limiter (after last sweep)")

job_store_entries = Gauge("sator_job_store_entries", "Jobs retained in the in-memory job store (after last sweep)")


def prom_available() -> bool:
    return not _MISSING
