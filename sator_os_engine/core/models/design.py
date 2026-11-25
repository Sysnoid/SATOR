from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DesignRequest(BaseModel):
    """Space-filling design over a search space. Runs synchronously."""

    model_config = ConfigDict(extra="forbid")

    search_space: dict[str, Any]
    n: int = Field(8, ge=1, le=64)
    seed: int | None = None
    sum_constraints: list[dict[str, Any]] | None = None
    ratio_constraints: list[dict[str, Any]] | None = None
    max_nonzero: int | None = Field(default=None, ge=1)
    min_dose: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _parameters(self) -> DesignRequest:
        params = self.search_space.get("parameters") if isinstance(self.search_space, dict) else None
        if not isinstance(params, list) or not params:
            raise ValueError("search_space.parameters must be a non-empty list")
        return self
