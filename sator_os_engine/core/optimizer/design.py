"""Space-filling initial designs that respect bounds, sums, ratios, and cardinality."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .cardinality import apply_cardinality
from .space import candidate_by_name, snap_numeric
from .utils import resolve_seed


def _bounds(params: list[dict[str, Any]]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for param in params:
        if param.get("type") == "categorical":
            n_choices = max(len(param.get("choices") or []), 1)
            bounds.append((0.0, float(n_choices - 1)))
        else:
            bounds.append((float(param["min"]), float(param["max"])))
    return bounds


def _groups(sum_constraints: list[dict[str, Any]] | None, dim: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for item in sum_constraints or []:
        idxs = [int(i) for i in item.get("indices", []) if 0 <= int(i) < dim]
        if not idxs:
            continue
        target = item.get("target", item.get("target_sum", 1.0))
        groups.append({"indices": idxs, "target": float(1.0 if target is None else target)})
    return groups


def initial_design(
    params: list[dict[str, Any]],
    n: int,
    *,
    seed: int | None = None,
    sum_constraints: list[dict[str, Any]] | None = None,
    ratio_constraints: list[dict[str, Any]] | None = None,
    max_nonzero: int | None = None,
    min_dose: float | None = None,
) -> dict[str, Any]:
    """Draw a Sobol design and project every row onto the feasible set."""
    if not params:
        raise ValueError("search_space.parameters must be a non-empty list")
    bounds = _bounds(params)
    groups = _groups(sum_constraints, len(params))
    resolved = resolve_seed(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    engine = SobolEngine(dimension=len(params), scramble=True, seed=resolved % (2**31 - 1) or 1)
    draw = engine.draw(max(n * 8, 32), dtype=torch.double).cpu().numpy()
    kept: list[np.ndarray] = []
    seen: list[np.ndarray] = []
    span = np.maximum(hi - lo, 1e-12)
    for unit in draw:
        row = lo + (hi - lo) * np.asarray(unit, dtype=float)
        projected, ok = apply_cardinality(
            row, bounds, groups, ratio_constraints, max_nonzero=max_nonzero, min_dose=min_dose
        )
        if not ok:
            continue
        projected = snap_numeric(projected, params)
        projected, ok = apply_cardinality(
            projected, bounds, groups, ratio_constraints, max_nonzero=max_nonzero, min_dose=min_dose
        )
        if not ok:
            continue
        norm = (projected - lo) / span
        if any(float(np.linalg.norm(norm - prev)) < 1e-6 for prev in seen):
            continue
        kept.append(projected)
        seen.append(norm)
        if len(kept) >= n:
            break
    if len(kept) < n:
        raise ValueError(f"could only place {len(kept)} of {n} feasible design points")
    rows = np.stack(kept[:n], axis=0)
    return {
        "candidates": [candidate_by_name(row, params) for row in rows],
        "X": rows.tolist(),
        "diagnostics": {"seed": resolved, "n": n},
    }
