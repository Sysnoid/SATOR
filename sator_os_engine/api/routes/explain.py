from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends

from ...core.models.explain import ExplainRequest
from ...core.models.optimize import OptimizeRequest
from ...core.optimizer.mobo_engine import run_optimization
from ...core.optimizer.space import encode_value
from ..deps import rate_limit

router = APIRouter()


@router.post("/explain")
async def explain_candidates(payload: ExplainRequest, api_key: str = Depends(rate_limit)):
    """Score recipes the caller supplies. No new batch is proposed."""
    del api_key
    params = list(payload.search_space.get("parameters") or [])
    rows = [
        [encode_value(candidate[str(param["name"])], param) for param in params] for candidate in payload.candidates
    ]
    request = OptimizeRequest(
        dataset=payload.dataset,
        search_space=payload.search_space,
        objectives=payload.objectives,
        constraints=payload.constraints,
        optimization_config=payload.optimization_config,
    )
    return run_optimization(request, explain_rows=np.asarray(rows, dtype=float))
