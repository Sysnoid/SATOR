from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    id: str
    owner_key: str
    status: JobStatus = JobStatus.QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


class JobStore:
    def __init__(self, ttl_sec: int = 600, timeout_sec: int = 300) -> None:
        self.ttl_sec = ttl_sec
        self.timeout_sec = timeout_sec
        self._jobs: dict[str, Job] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def create_job(self, owner_key: str) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, owner_key=owner_key)
        self._jobs[job_id] = job
        self._locks[job_id] = asyncio.Lock()
        from ..observability import metrics

        metrics.jobs_created.inc()
        return job

    def get_job(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if time.time() - job.created_at > self.ttl_sec and job.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            # expired
            del self._jobs[job_id]
            return None
        return job

    async def set_status(self, job_id: str, status: JobStatus) -> None:
        if job_id not in self._jobs:
            return
        async with self._locks[job_id]:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = time.time()

    async def complete(self, job_id: str, result: dict[str, Any]) -> None:
        if job_id not in self._jobs:
            return
        async with self._locks[job_id]:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETED
            job.result = result
            job.updated_at = time.time()

    async def fail(self, job_id: str, error: str) -> None:
        if job_id not in self._jobs:
            return
        async with self._locks[job_id]:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = time.time()

    def sweep_terminated_past_ttl(self) -> int:
        """Delete terminal jobs that are past ``ttl_sec`` since creation (even if never fetched)."""
        now = time.time()
        to_del: list[str] = []
        for jid, job in list(self._jobs.items()):
            if (
                job.status
                in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                )
                and (now - job.created_at) > self.ttl_sec
            ):
                to_del.append(jid)
        for jid in to_del:
            self._jobs.pop(jid, None)
            self._locks.pop(jid, None)
        return len(to_del)

    def __len__(self) -> int:
        return len(self._jobs)
