from __future__ import annotations

from fastapi import APIRouter, Depends, Header

from ...core.models.explain import ExplainRequest
from ...runtime.async_tasks import spawn_task_logged
from ...runtime.executor import Executor
from ...runtime.jobs import JobStatus
from ..deps import get_executor, get_idempotency_store, get_job_store, get_settings, idempotency, rate_limit

router = APIRouter()


@router.post("/explain", status_code=202)
async def explain_candidates(
    payload: ExplainRequest,
    api_key: str = Depends(rate_limit),
    idem_existing: str | None = Depends(idempotency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    job_store=Depends(get_job_store),
    settings=Depends(get_settings),
    idem_store=Depends(get_idempotency_store),
    executor: Executor = Depends(get_executor),
):
    """Score recipes the caller supplies, in a worker process. No new batch is proposed."""
    if idem_existing:
        return {"job_id": idem_existing}

    job = job_store.create_job(owner_key=api_key)
    if idempotency_key:
        idem_store.put(api_key, idempotency_key, job.id)
    spawn_task_logged(
        executor.submit(job.id, "explain", (payload.model_dump(mode="json"), settings.device, settings.cuda_device)),
        label=f"http-explain job_id={job.id}",
    )
    return {"job_id": job.id, "status": JobStatus.QUEUED}
