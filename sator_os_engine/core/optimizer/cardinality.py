"""At most K non-zero ingredients, and a minimum dose when an ingredient is used."""

from __future__ import annotations

from typing import Any

import numpy as np

from .project import project_to_feasible


def _supports(groups: list[dict[str, Any]], dim: int) -> list[list[int]]:
    if groups:
        return [list(group["indices"]) for group in groups]
    return [list(range(dim))]


def sparsify(
    x: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    max_nonzero: int | None,
    min_dose: float | None,
) -> np.ndarray:
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    for idxs in _supports(groups, y.size):
        if min_dose is not None:
            for i in idxs:
                if 0.0 < y[i] < float(min_dose):
                    y[i] = 0.0
        if max_nonzero is None:
            continue
        positive = [i for i in idxs if y[i] > 1e-12]
        while len(positive) > int(max_nonzero):
            smallest = min(positive, key=lambda i: y[i])
            y[smallest] = 0.0
            positive.remove(smallest)
    return y


def lock_zeros(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
    groups: list[dict[str, Any]],
    max_nonzero: int | None,
    min_dose: float | None,
) -> list[tuple[float, float]]:
    if max_nonzero is None and min_dose is None:
        return bounds
    locked = list(bounds)
    for idxs in _supports(groups, len(x)):
        for i in idxs:
            if float(x[i]) <= 1e-12:
                locked[i] = (0.0, 0.0)
    return locked


def apply_cardinality(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
    groups: list[dict[str, Any]] | None,
    ratios: list[dict[str, Any]] | None,
    *,
    max_nonzero: int | None = None,
    min_dose: float | None = None,
    supports: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, bool]:
    """Project onto the feasible set, then enforce cardinality on ``supports``.

    ``supports`` defaults to the sum groups. Pass a different list when the
    cardinality limit covers ingredients that are not tied to a sum constraint.
    """
    groups = list(groups or [])
    limited = groups if supports is None else list(supports)
    y = sparsify(x, limited, max_nonzero=max_nonzero, min_dose=min_dose)
    locked = lock_zeros(y, bounds, limited, max_nonzero, min_dose)
    return project_to_feasible(y, locked, groups, ratios)
