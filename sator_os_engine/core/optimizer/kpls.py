"""Partial-least-squares kernel on the original inputs.

The GP distance is a Matern kernel on a few PLS scores. The optimizer still
proposes a full recipe, so sums and ratios stay ordinary constraints.
"""

from __future__ import annotations

import numpy as np
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from sklearn.cross_decomposition import PLSRegression
from torch import Tensor


def kpls_rank(n_rows: int, n_features: int, requested: int | None) -> int:
    """Rank capped by the data and by a default of four directions."""
    limit = max(1, min(n_rows - 1, n_features))
    if requested is None:
        return min(4, limit)
    return max(1, min(int(requested), limit))


def fit_kpls_weights(x_norm: np.ndarray, y: np.ndarray, requested: int | None) -> tuple[np.ndarray, int]:
    """PLS weights of standardized outcomes on already-normalized inputs."""
    rows = np.asarray(x_norm, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    rank = kpls_rank(rows.shape[0], rows.shape[1], requested)
    scale = max(float(np.std(target)), 1e-6)
    standardized = (target - float(np.mean(target))) / scale
    pls = PLSRegression(n_components=rank, scale=False)
    pls.fit(rows, standardized)
    return np.asarray(pls.x_weights_, dtype=float), rank


class ProjectedMaternKernel(Kernel):
    """Matern-5/2 ARD kernel after a fixed linear projection."""

    def __init__(self, weights: Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.register_buffer("weights", weights)
        self.latent = MaternKernel(nu=2.5, ard_num_dims=int(weights.shape[-1]))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        return self.latent.forward(x1.matmul(self.weights), x2.matmul(self.weights), diag=diag, **params)


def kpls_covar(weights: Tensor) -> ScaleKernel:
    return ScaleKernel(ProjectedMaternKernel(weights))
