from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..observability import metrics
from .jobs import JobStatus, JobStore


class Executor:
    def __init__(self, store: JobStore, max_workers: int = 4, timeout_sec: int = 300) -> None:
        self.store = store
        self.timeout_sec = timeout_sec
        self._tp = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sator-worker")

    async def submit(self, job_id: str, func: Callable[[], dict[str, Any]]) -> None:
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        await self.store.set_status(job_id, JobStatus.RUNNING)
        metrics.jobs_in_flight.inc()
        try:
            result = await asyncio.wait_for(loop.run_in_executor(self._tp, func), timeout=self.timeout_sec)
            await self.store.complete(job_id, result)
            metrics.jobs_finished.labels(outcome="completed").inc()
        except asyncio.TimeoutError:
            await self.store.fail(job_id, "Job timed out")
            metrics.jobs_finished.labels(outcome="timeout").inc()
        except Exception as e:  # noqa: BLE001
            await self.store.fail(job_id, f"Job failed: {e}")
            metrics.jobs_finished.labels(outcome="failed").inc()
        finally:
            metrics.job_duration_seconds.observe(time.monotonic() - t0)
            metrics.jobs_in_flight.dec()

    def shutdown(self, wait: bool = True) -> None:
        self._tp.shutdown(wait=wait)
