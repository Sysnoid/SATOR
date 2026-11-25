"""Evaluate a latent GP on original-space inputs."""

from __future__ import annotations

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

    def encode(self, X: Tensor) -> Tensor:
        z_raw = (X - self.mean) @ self.components.T
        return (z_raw - self.pc_mins) / self.pc_range

    def posterior(self, X, observation_noise=False, posterior_transform=None):  # noqa: ANN001
        return self.inner.posterior(
            self.encode(X),
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )

    def subset_output(self, idcs: list[int]) -> Model:
        return PcaObservationModel(
            self.inner.subset_output(idcs),
            self.components,
            self.mean,
            self.pc_mins,
            self.pc_range,
        )
