from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from ...core.models.optimize import OptimizeRequest
from ...core.optimizer.mobo_engine import run_optimization
from ...runtime.async_tasks import spawn_task_logged
from ...runtime.executor import Executor
from ...runtime.jobs import JobStatus
from ..deps import get_executor, get_idempotency_store, get_job_store, get_settings, idempotency, rate_limit

router = APIRouter()


@router.post("/optimize", status_code=202)
async def submit_optimize(
    payload: OptimizeRequest,
    api_key: str = Depends(rate_limit),
    idem_existing: str | None = Depends(idempotency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    job_store=Depends(get_job_store),
    settings=Depends(get_settings),
    idem_store=Depends(get_idempotency_store),
    executor: Executor = Depends(get_executor),
):
    # Idempotency shortcut
    if idem_existing:
        return {"job_id": idem_existing}

    job = job_store.create_job(owner_key=api_key)
    if idempotency_key:
        idem_store.put(api_key, idempotency_key, job.id)

    def work():
        return run_optimization(payload, device=settings.device, cuda_device=settings.cuda_device)

    spawn_task_logged(executor.submit(job.id, work), label=f"http-optimize job_id={job.id}")

    return {"job_id": job.id, "status": JobStatus.QUEUED}
