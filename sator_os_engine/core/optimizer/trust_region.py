"""Replay a TuRBO-1 trust region from an ordered dataset.

The box is a fraction of each search side, centred on the best row so far.
Successes expand it and failures shrink it. Nothing is stored between calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _nondominated(points: np.ndarray) -> np.ndarray:
    """Rows that are not weakly dominated. Larger is better in every column."""
    keep: list[np.ndarray] = []
    for row in points:
        dominated = False
        for other in points:
            if np.array_equal(row, other):
                continue
            if bool(np.all(other >= row - 1e-12) and np.any(other > row + 1e-12)):
                dominated = True
                break
        if not dominated:
            keep.append(row)
    return np.vstack(keep)


def _hypervolume(points: np.ndarray, ref: np.ndarray) -> float:
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    front = _nondominated(points)
    above = np.all(front > ref.reshape(1, -1), axis=1)
    if not np.any(above):
        return 0.0
    value = Hypervolume(ref_point=torch.tensor(ref, dtype=torch.double)).compute(
        torch.tensor(front[above], dtype=torch.double)
    )
    return float(value)


def replay_trust_region(
    x: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]],
    signs: np.ndarray,
    *,
    length_init: float = 0.8,
    length_min: float = 0.5**7,
    length_max: float = 1.6,
    success_tolerance: int = 3,
    failure_tolerance: int | None = None,
) -> dict[str, Any]:
    """Trust-region box after walking ``x`` and ``y`` from first row to last.

    Single-objective success is a new best in the signed frame. Multi-objective
    success is an increase in hypervolume. ``signs`` maps user units onto that
    frame (+1 maximise, -1 minimise).
    """
    rows = np.atleast_2d(np.asarray(x, dtype=float))
    outcomes = np.atleast_2d(np.asarray(y, dtype=float))
    if outcomes.shape[0] == 1 and outcomes.shape[1] == rows.shape[0] and rows.shape[0] != 1:
        outcomes = outcomes.T
    direction = np.asarray(signs, dtype=float).reshape(1, -1)
    signed = outcomes * direction
    span = np.array([max(float(hi) - float(lo), 1e-9) for lo, hi in bounds], dtype=float)
    lower = np.array([float(lo) for lo, _ in bounds], dtype=float)
    upper = np.array([float(hi) for _, hi in bounds], dtype=float)
    length = float(length_init)
    successes = 0
    failures = 0
    restarted = False
    fail_limit = int(rows.shape[1] if failure_tolerance is None else failure_tolerance)
    best = 0
    ref = None
    if signed.shape[1] == 1:
        best_value = float(signed[0, 0])
    else:
        finite = signed[np.isfinite(signed).all(axis=1)]
        ref = finite.min(axis=0) - 0.1 * np.maximum(finite.max(axis=0) - finite.min(axis=0), 1e-6)
        best_value = _hypervolume(signed[:1], ref)

    for index in range(1, rows.shape[0]):
        if signed.shape[1] == 1:
            improved = float(signed[index, 0]) > best_value + 1e-12
            candidate_value = float(signed[index, 0])
        else:
            candidate_value = _hypervolume(signed[: index + 1], ref)
            improved = candidate_value > best_value + 1e-12
        if improved:
            best_value = candidate_value
            best = index
            successes += 1
            failures = 0
            if successes >= int(success_tolerance):
                length = min(length * 2.0, float(length_max))
                successes = 0
        else:
            failures += 1
            successes = 0
            if failures >= fail_limit:
                length = length / 2.0
                failures = 0
                if length < float(length_min):
                    length = float(length_init)
                    restarted = True
    center = rows[best]
    half = 0.5 * float(length) * span
    box_lo = np.maximum(lower, center - half)
    box_hi = np.minimum(upper, center + half)
    return {
        "center": center.tolist(),
        "length": float(length),
        "success_count": int(successes),
        "failure_count": int(failures),
        "restarted": bool(restarted),
        "bounds": [[float(a), float(b)] for a, b in zip(box_lo, box_hi, strict=True)],
    }


def thompson_candidates(model: Any, bounds: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Pick ``n`` points inside ``bounds`` by maximum posterior sample."""
    from botorch.generation.sampling import MaxPosteriorSampling
    from torch.quasirandom import SobolEngine

    dimension = int(bounds.shape[-1])
    engine = SobolEngine(dimension, scramble=True, seed=int(seed) % (2**31 - 1) or 1)
    unit = engine.draw(512).to(dtype=bounds.dtype, device=bounds.device)
    grid = bounds[0] + (bounds[1] - bounds[0]) * unit
    picked = MaxPosteriorSampling(model=model, replacement=False)(grid, num_samples=int(n))
    return picked.reshape(int(n), dimension)
