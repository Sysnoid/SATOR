from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    algorithm: str = Field("qnehvi")
    # Multi-objective (min/max): qnehvi=qLogEHVI (default), qehvi, qnoisyehvi, parego|qei, qpi, qucb|ucb.
    # Single-objective (min/max): same names map to qLogEI, qEI, qPI, qUCB, etc. Use acquisition="sobol" to force the legacy Sobol+scoring grid.
    acquisition: str = Field("qnehvi")
    batch_size: int = Field(4, ge=1)
    max_evaluations: int = Field(100, ge=1)
    seed: int | None = None
    use_pca: bool = False
    pca_dimension: int | None = None
    parameter_scaling: str | None = None  # none|standardize|minmax
    value_normalization: str | None = None  # none|standardize|minmax
    target_tolerance: float | None = None
    target_variance_penalty: float | None = None
    sum_constraints: list[dict[str, Any]] | None = None  # [{"indices":[0,1,2],"target_sum":1.0}]
    ratio_constraints: list[dict[str, Any]] | None = None  # [{"i":0,"j":1,"min_ratio":0.5,"max_ratio":2.0}]
    # GP surface/volume maps for visualization (2D/3D only)
    return_maps: bool = False
    map_space: str = Field("input")  # input|pca
    map_resolution: list[int] | None = None  # [nx,ny,(nz)]
    # Advanced numerical settings (optional)
    advanced: dict[str, Any] | None = None
    # Optional GP configuration and acquisition parameters
    # lengthscale|outputscale|noise: applied before MLL fit as hints. If fix_lengthscale|fix_outputscale|fix_noise
    # is true, that value is re-applied after fit and the parameter is frozen (not overwritten by optimization).
    gp_config: dict[str, Any] | None = None
    acquisition_params: dict[str, Any] | None = None  # e.g., {"ucb_beta": 0.2}
    outputs: dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    dataset: dict[str, Any]
    search_space: dict[str, Any]
    objectives: dict[str, Any]
    constraints: dict[str, Any] | None = None
    optimization_config: OptimizationConfig


class OptimizeResponse(BaseModel):
    job_id: str | None = None
    predictions: list[dict[str, Any]] | None = None
    pareto: dict[str, Any] | None = None
    encoding_info: dict[str, Any] | None = None  # e.g., pc_mins/maxs
    diagnostics: dict[str, Any] | None = None
