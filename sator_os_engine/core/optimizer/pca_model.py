"""Evaluate a latent GP on original-space inputs."""

from __future__ import annotations

import numpy as np
import torch
from botorch.models.model import Model
from torch import Tensor


class PcaObservationModel(Model):
    """GP trained on normalized PCA scores, queried with original x.

    ``posterior(X)`` applies the linear encoder ``z = (x - mean) @ C.T`` and
    the training normalization, then delegates to the latent model. Acquisition
    can therefore search the feasible set in x while the GP stays in z.
    """

    def __init__(
        self,
        inner: Model,
        components: Tensor,
        mean: Tensor,
        pc_mins: Tensor,
        pc_range: Tensor,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("components", components)
        self.register_buffer("mean", mean)
        self.register_buffer("pc_mins", pc_mins)
        self.register_buffer("pc_range", pc_range)
        self._num_outputs = int(inner.num_outputs)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def likelihood(self):  # noqa: ANN201
        return self.inner.likelihood

    def encode(self, X: Tensor) -> Tensor:
        z_raw = (X - self.mean) @ self.components.T
        return (z_raw - self.pc_mins) / self.pc_range

    def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):  # noqa: ANN001
        # ModelList.posterior forwards output_indices. The latent GP accepts it
        # through **kwargs and ignores it for a single output.
        return self.inner.posterior(
            self.encode(X),
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def subset_output(self, idcs: list[int]) -> Model:
        return PcaObservationModel(
            self.inner.subset_output(idcs),
            self.components,
            self.mean,
            self.pc_mins,
            self.pc_range,
        )


def _as_rows(x: np.ndarray) -> np.ndarray:
    rows = np.asarray(x, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows


def reconstruct_from_scores(
    scores: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Vector form of ``scaled_pca_inverse``."""
    sigma = np.maximum(np.asarray(scaler_scale, dtype=float).reshape(-1), 1e-12)
    mu = np.asarray(scaler_mean, dtype=float).reshape(-1)
    mean_eff = np.asarray(mean, dtype=float).reshape(-1)
    c_pca = np.asarray(components, dtype=float) * sigma
    mu_pca = (mean_eff - mu) / sigma
    return (_as_rows(scores) @ c_pca + mu_pca) * sigma + mu


def off_subspace_residual(
    x: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """Distance, in scaled units, from each row to its PCA reconstruction."""
    rows = _as_rows(x)
    mean_eff = np.asarray(mean, dtype=float).reshape(-1)
    scores = (rows - mean_eff) @ np.asarray(components, dtype=float).T
    restored = reconstruct_from_scores(scores, components, mean, scaler_mean, scaler_scale)
    sigma = np.maximum(np.asarray(scaler_scale, dtype=float).reshape(-1), 1e-12)
    return np.linalg.norm((rows - restored) / sigma, axis=1)


def pull_toward_subspace(
    x: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    cap: float,
) -> np.ndarray:
    """Move each row toward its PCA reconstruction until the residual is within ``cap``.

    The scores stay the same, so this only removes null-space drift.
    """
    rows = _as_rows(x)
    mean_eff = np.asarray(mean, dtype=float).reshape(-1)
    scores = (rows - mean_eff) @ np.asarray(components, dtype=float).T
    restored = reconstruct_from_scores(scores, components, mean, scaler_mean, scaler_scale)
    sigma = np.maximum(np.asarray(scaler_scale, dtype=float).reshape(-1), 1e-12)
    residual = np.linalg.norm((rows - restored) / sigma, axis=1)
    alpha = np.minimum(1.0, float(cap) / np.maximum(residual, 1e-12))
    return restored + alpha[:, None] * (rows - restored)


def latent_box_inequalities(
    components: np.ndarray,
    mean: np.ndarray,
    pc_mins: np.ndarray,
    pc_range: np.ndarray,
    margin: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Keep normalized PCA scores inside ``[-m, 1+m]``. Each row is linear in x."""
    loadings = np.asarray(components, dtype=float)
    center = np.asarray(mean, dtype=float).reshape(-1)
    span = np.asarray(pc_range, dtype=float)
    lower_raw = np.asarray(pc_mins, dtype=float) + (-float(margin)) * span
    upper_raw = np.asarray(pc_mins, dtype=float) + (1.0 + float(margin)) * span
    index = torch.arange(center.shape[0], dtype=torch.long, device=device)
    inequalities: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for row, low, high in zip(loadings, lower_raw, upper_raw, strict=True):
        shift = float(row @ center)
        coefficients = torch.tensor(row, dtype=dtype, device=device)
        inequalities.append((index, coefficients, float(low) + shift))
        inequalities.append((index, -coefficients, -(float(high) + shift)))
    return inequalities
