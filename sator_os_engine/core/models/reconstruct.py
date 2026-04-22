from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PCAInfo(BaseModel):
    pc_mins: list[float]
    pc_maxs: list[float]
    components: list[list[float]] | None = None
    mean: list[float] | None = None


class ReconstructionRequest(BaseModel):
    coordinates: list[float]  # normalized [0,1]^D or natural depending on config
    pca_info: PCAInfo | None = None
    bounds: dict[str, Any]
    n_ingredients: int = Field(ge=0, default=0)
    target_precision: float = Field(1e-7, gt=0)
    sum_target: float = Field(1.0, gt=0)
    ratio_constraints: list[dict[str, float]] | None = None  # list of {i,j,min_ratio,max_ratio}
    ingredient_names: list[str] | None = None  # in order; length n_ingredients
    parameter_names: list[str] | None = None  # in order; length of bounds.parameters


class ReconstructionResponse(BaseModel):
    job_id: str | None = None
    success: bool | None = None
    reconstructed_formulation: dict[str, list[float]] | None = None
    reconstruction_metrics: dict[str, Any] | None = None
    error: str | None = None
