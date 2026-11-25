from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PCAInfo(BaseModel):
    pc_mins: list[float]
    pc_maxs: list[float]
    components: list[list[float]] | None = None
    mean: list[float] | None = None
    scaler_mean: list[float] | None = None
    scaler_scale: list[float] | None = None


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

    @model_validator(mode="after")
    def _shapes(self) -> ReconstructionRequest:
        info = self.pca_info
        if info is None:
            return self
        n_coords = len(self.coordinates)
        if len(info.pc_mins) != n_coords or len(info.pc_maxs) != n_coords:
            raise ValueError(
                f"len(coordinates)={n_coords} must equal len(pc_mins)={len(info.pc_mins)} "
                f"and len(pc_maxs)={len(info.pc_maxs)}"
            )
        n_bounds = 0
        if isinstance(self.bounds, dict):
            n_bounds = len(self.bounds.get("ingredients") or []) + len(self.bounds.get("parameters") or [])
        if self.n_ingredients > n_bounds:
            raise ValueError(f"n_ingredients ({self.n_ingredients}) cannot exceed the number of bounds ({n_bounds})")
        if info.components is not None:
            if len(info.components) != n_coords:
                raise ValueError(f"len(components)={len(info.components)} must equal len(coordinates)={n_coords}")
            width = len(info.components[0]) if info.components else 0
            if any(len(row) != width for row in info.components):
                raise ValueError("every components row must have the same width")
            if info.mean is not None and len(info.mean) != width:
                raise ValueError(f"len(mean)={len(info.mean)} must equal components width {width}")
            if n_bounds and width != n_bounds:
                raise ValueError(f"components width {width} must equal the number of bounds {n_bounds}")
            if info.scaler_mean is not None and len(info.scaler_mean) != width:
                raise ValueError("len(scaler_mean) must equal components width")
            if info.scaler_scale is not None and len(info.scaler_scale) != width:
                raise ValueError("len(scaler_scale) must equal components width")
        if self.ingredient_names is not None and len(self.ingredient_names) != self.n_ingredients:
            raise ValueError("ingredient_names length must equal n_ingredients")
        n_parameters = len(self.bounds.get("parameters") or []) if isinstance(self.bounds, dict) else 0
        if self.parameter_names is not None and len(self.parameter_names) != n_parameters:
            raise ValueError("parameter_names length must equal the number of parameter bounds")
        return self


class ReconstructionResponse(BaseModel):
    job_id: str | None = None
    success: bool | None = None
    reconstructed_formulation: dict[str, list[float]] | None = None
    reconstruction_metrics: dict[str, Any] | None = None
    error: str | None = None
