"""Request for scoring recipes the caller already has."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .optimize import OptimizationConfig, OptimizeRequest


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: dict[str, Any]
    search_space: dict[str, Any]
    objectives: dict[str, Any]
    constraints: dict[str, Any] | None = None
    optimization_config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    candidates: list[dict[str, Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def _candidates_match_the_search_space(self) -> ExplainRequest:
        OptimizeRequest(
            dataset=self.dataset,
            search_space=self.search_space,
            objectives=self.objectives,
            constraints=self.constraints,
            optimization_config=self.optimization_config,
        )
        params = list(self.search_space.get("parameters") or [])
        names = [str(param["name"]) for param in params]
        for index, candidate in enumerate(self.candidates):
            missing = [name for name in names if name not in candidate]
            if missing:
                raise ValueError(f"candidates[{index}] is missing {', '.join(missing)}")
        return self
