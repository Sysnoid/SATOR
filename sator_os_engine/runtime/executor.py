from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue as queue_mod
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..observability import metrics
from .jobs import JobStatus, JobStore


def _process_entry(kind: str, args: tuple[Any, ...], queue: Any) -> None:
    """Run one job in a child process. The parent terminates this process on timeout."""
    try:
        from .worker import dispatch

        queue.put(("ok", dispatch(kind, args)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


class Executor:
    def __init__(self, store: JobStore, max_workers: int = 4, timeout_sec: int = 300) -> None:
        self.store = store
        self.timeout_sec = timeout_sec
        self.max_workers = max_workers
        self._waiters = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sator-wait")
        self._sem: asyncio.Semaphore | None = None
        self._procs: set[mp.Process] = set()
        self._proc_lock = threading.Lock()

    async def submit(self, job_id: str, kind: str, args: tuple[Any, ...] = ()) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_workers)
        async with self._sem:
            await self._run(job_id, kind, args)

    async def _run(self, job_id: str, kind: str, args: tuple[Any, ...]) -> None:
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        await self.store.set_status(job_id, JobStatus.RUNNING)
        metrics.jobs_in_flight.inc()
        ctx = mp.get_context("spawn")
        queue: Any = ctx.Queue()
        proc = ctx.Process(target=_process_entry, args=(kind, args, queue), daemon=False)
        with self._proc_lock:
            self._procs.add(proc)
        proc.start()
        try:
            status, payload = await loop.run_in_executor(self._waiters, self._wait, proc, queue)
            if status == "timeout":
                await self.store.fail(job_id, "Job timed out")
                metrics.jobs_finished.labels(outcome="timeout").inc()
            elif status == "ok":
                from .json_safe import sanitize_for_json

                try:
                    payload = sanitize_for_json(payload)
                except ValueError as exc:
                    await self.store.fail(job_id, f"Job failed: {exc}")
                    metrics.jobs_finished.labels(outcome="failed").inc()
                else:
                    await self.store.complete(job_id, payload)
                    metrics.jobs_finished.labels(outcome="completed").inc()
            else:
                await self.store.fail(job_id, f"Job failed: {payload}")
                metrics.jobs_finished.labels(outcome="failed").inc()
        except Exception as e:  # noqa: BLE001
            await self.store.fail(job_id, f"Job failed: {e}")
            metrics.jobs_finished.labels(outcome="failed").inc()
        finally:
            if proc.is_alive():
                proc.terminate()
                proc.join(2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(1)
            with self._proc_lock:
                self._procs.discard(proc)
            metrics.job_duration_seconds.observe(time.monotonic() - t0)
            metrics.jobs_in_flight.dec()

    def _wait(self, proc: mp.Process, queue: Any) -> tuple[str, Any]:
        # Read the queue while the child is alive. Joining first deadlocks when
        # the result is larger than the OS pipe buffer: the child blocks in
        # ``queue.put`` and never exits.
        deadline = time.monotonic() + float(self.timeout_sec)
        while time.monotonic() < deadline:
            try:
                status, payload = queue.get(timeout=0.2)
            except queue_mod.Empty:
                if not proc.is_alive():
                    return "err", f"worker exited with code {proc.exitcode} and no result"
                continue
            proc.join(5)
            return status, payload
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join(1)
        return "timeout", None

    def shutdown(self, wait: bool = True) -> None:
        with self._proc_lock:
            procs = list(self._procs)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        if wait:
            for proc in procs:
                proc.join(2)
        self._waiters.shutdown(wait=wait)
