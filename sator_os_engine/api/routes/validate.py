from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from ...core.optimizer.dataset_check import validate_dataset
from ...core.optimizer.utils import sum_groups_from_config
from ..deps import rate_limit

router = APIRouter()


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: dict
    search_space: dict
    sum_constraints: list[dict] | None = None


@router.post("/validate")
async def validate(payload: ValidateRequest, api_key: str = Depends(rate_limit)):
    """Report duplicates, bounds, missing values, sums, and collinear columns."""
    del api_key
    raw_x = payload.dataset.get("X") if isinstance(payload.dataset, dict) else None
    raw_y = payload.dataset.get("Y") if isinstance(payload.dataset, dict) else None
    if raw_x is None or raw_y is None:
        return {"ok": False, "issues": [{"code": "shape", "message": "dataset.X and dataset.Y are required"}]}
    try:
        x = np.asarray(raw_x, dtype=float)
        y = np.asarray(raw_y, dtype=float)
    except (TypeError, ValueError):
        return {"ok": False, "issues": [{"code": "shape", "message": "dataset.X and dataset.Y must be numeric"}]}
    params = list(payload.search_space.get("parameters") or []) if isinstance(payload.search_space, dict) else []
    dim = int(x.shape[1]) if x.ndim == 2 else 0
    request = SimpleNamespace(optimization_config=SimpleNamespace(sum_constraints=payload.sum_constraints))
    return validate_dataset(x, y, params, sum_groups_from_config(request, dim))
