from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse

from ...core.models.reconstruct import ReconstructionRequest
from ...runtime.async_tasks import spawn_task_logged
from ...runtime.executor import Executor
from ...runtime.jobs import JobStatus
from ...runtime.worker import run_reconstruct_payload
from ..deps import get_executor, get_idempotency_store, get_job_store, idempotency, rate_limit

router = APIRouter()


@router.post("/reconstruct")
async def submit_reconstruct(
    payload: ReconstructionRequest,
    sync: bool = Query(default=False),
    api_key: str = Depends(rate_limit),
    idem_existing: str | None = Depends(idempotency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    job_store=Depends(get_job_store),
    idem_store=Depends(get_idempotency_store),
    executor: Executor = Depends(get_executor),
):
    if idem_existing:
        return JSONResponse({"job_id": idem_existing}, status_code=202)

    body = payload.model_dump(mode="json")
    if sync:
        return JSONResponse(run_reconstruct_payload(body), status_code=200)

    job = job_store.create_job(owner_key=api_key)
    if idempotency_key:
        idem_store.put(api_key, idempotency_key, job.id)

    spawn_task_logged(
        executor.submit(job.id, "reconstruct", (body,)),
        label=f"http-reconstruct job_id={job.id}",
    )
    return JSONResponse({"job_id": job.id, "status": JobStatus.QUEUED}, status_code=202)
