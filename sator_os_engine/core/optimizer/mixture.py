"""Isometric log-ratio coordinates for sum-constrained ingredient blocks."""

from __future__ import annotations

from typing import Any

import numpy as np


def _contrast(d: int) -> np.ndarray:
    """Orthonormal Helmert contrasts, shape ``(d - 1, d)``."""
    v = np.zeros((d - 1, d), dtype=float)
    for i in range(d - 1):
        scale = np.sqrt(1.0 / ((i + 1) * (i + 2)))
        v[i, : i + 1] = scale
        v[i, i + 1] = -np.sqrt((i + 1) / (i + 2.0))
    return v


def ilr_forward(composition: np.ndarray) -> np.ndarray:
    """Map positive rows that sum to one onto ``d - 1`` ILR coordinates."""
    x = np.asarray(composition, dtype=float)
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)
    x = np.clip(x, 1e-12, None)
    x = x / x.sum(axis=1, keepdims=True)
    clr = np.log(x) - np.log(x).mean(axis=1, keepdims=True)
    z = clr @ _contrast(x.shape[1]).T
    return z.reshape(-1) if single else z


def ilr_inverse(coordinates: np.ndarray) -> np.ndarray:
    """Inverse ILR. Rows sum to one."""
    z = np.asarray(coordinates, dtype=float)
    single = z.ndim == 1
    if single:
        z = z.reshape(1, -1)
    clr = z @ _contrast(z.shape[1] + 1)
    comp = np.exp(clr - clr.max(axis=1, keepdims=True))
    comp = comp / comp.sum(axis=1, keepdims=True)
    return comp.reshape(-1) if single else comp


class MixtureEncoder:
    """Replace each sum group of size ≥ 2 with its ILR coordinates."""

    def __init__(self, groups: list[dict[str, Any]], dim: int) -> None:
        self.dim = int(dim)
        self.groups = [list(group["indices"]) for group in groups if len(group.get("indices", [])) >= 2]
        if not self.groups:
            raise ValueError("mixture_transform 'ilr' needs a sum group with at least 2 ingredients")
        self._covered = {i for group in self.groups for i in group}

    def forward(self, x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        parts: list[np.ndarray] = []
        emitted: set[int] = set()
        for j in range(self.dim):
            if j in emitted:
                continue
            group = next((g for g in self.groups if j in g), None)
            if group is None:
                parts.append(rows[:, j : j + 1])
                emitted.add(j)
            else:
                parts.append(ilr_forward(rows[:, group]))
                emitted.update(group)
        return np.hstack(parts)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(z, dtype=float))
        out = np.zeros((rows.shape[0], self.dim), dtype=float)
        cursor = 0
        emitted: set[int] = set()
        for j in range(self.dim):
            if j in emitted:
                continue
            group = next((g for g in self.groups if j in g), None)
            if group is None:
                out[:, j] = rows[:, cursor]
                cursor += 1
                emitted.add(j)
            else:
                width = len(group) - 1
                restored = ilr_inverse(rows[:, cursor : cursor + width])
                for local, index in enumerate(group):
                    out[:, index] = restored[:, local]
                cursor += width
                emitted.update(group)
        return out

    def bounds(self, z_train: np.ndarray, margin: float = 0.2) -> np.ndarray:
        """Search box around the training coordinates, shape ``(2, k)``."""
        z = np.atleast_2d(np.asarray(z_train, dtype=float))
        lo = z.min(axis=0)
        hi = z.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        return np.vstack([lo - margin * span, hi + margin * span])
