from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.models.design import DesignRequest
from ...core.optimizer.design import initial_design
from ..deps import rate_limit

router = APIRouter()


@router.post("/design")
async def submit_design(payload: DesignRequest, api_key: str = Depends(rate_limit)):
    del api_key
    return initial_design(
        list(payload.search_space.get("parameters") or []),
        int(payload.n),
        seed=payload.seed,
        sum_constraints=payload.sum_constraints,
        ratio_constraints=payload.ratio_constraints,
        max_nonzero=payload.max_nonzero,
        min_dose=payload.min_dose,
    )
