from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize, TransformedPosterior
from gpytorch.mlls import ExactMarginalLogLikelihood


def _apply_gp_hint_priors(m: SingleTaskGP, cfg_gp: dict[str, Any], tX: torch.Tensor) -> None:
    """Set initial lengthscale / outputscale / noise before MLL optimization (may be overridden by fit)."""
    dtype, device = tX.dtype, tX.device
    ls = cfg_gp.get("lengthscale")
    if ls is not None:
        if isinstance(ls, (list, tuple, np.ndarray)):
            ls_t = torch.tensor(ls, dtype=dtype, device=device)
        else:
            ls_t = torch.tensor([float(ls)] * tX.shape[-1], dtype=dtype, device=device)
        with suppress(Exception):
            m.covar_module.base_kernel.lengthscale = ls_t
    oscale = cfg_gp.get("outputscale")
    if oscale is not None:
        with suppress(Exception):
            m.covar_module.outputscale = float(oscale)
    noise = cfg_gp.get("noise")
    if noise is not None:
        with suppress(Exception):
            m.likelihood.noise_covar.initialize(noise=float(noise))


def _freeze_gp_hypers_post_fit(m: SingleTaskGP, cfg_gp: dict[str, Any], tX: torch.Tensor) -> None:
    """After MLL fit, optionally fix user-specified hyperparameters (not re-optimized)."""
    dtype, device = tX.dtype, tX.device
    m.eval()
    fix_ls = bool(cfg_gp.get("fix_lengthscale"))
    fix_os = bool(cfg_gp.get("fix_outputscale"))
    fix_noise = bool(cfg_gp.get("fix_noise"))
    if not (fix_ls or fix_os or fix_noise):
        return
    with torch.no_grad():
        if fix_ls and cfg_gp.get("lengthscale") is not None:
            ls = cfg_gp["lengthscale"]
            if isinstance(ls, (list, tuple, np.ndarray)):
                ls_t = torch.tensor(ls, dtype=dtype, device=device).view(1, -1)
            else:
                ls_t = torch.full((1, tX.shape[-1]), float(ls), dtype=dtype, device=device)
            try:
                m.covar_module.base_kernel.lengthscale = ls_t
                raw = getattr(m.covar_module.base_kernel, "raw_lengthscale", None)
                if raw is not None:
                    raw.requires_grad_(False)
            except Exception:
                pass
        if fix_os and cfg_gp.get("outputscale") is not None:
            try:
                m.covar_module.outputscale = torch.tensor(float(cfg_gp["outputscale"]), dtype=dtype, device=device)
                raw = getattr(m.covar_module, "raw_outputscale", None)
                if raw is not None:
                    raw.requires_grad_(False)
            except Exception:
                pass
        if fix_noise and cfg_gp.get("noise") is not None:
            try:
                n = float(cfg_gp["noise"])
                m.likelihood.noise_covar.initialize(noise=n)
                raw = getattr(m.likelihood.noise_covar, "raw_noise", None)
                if raw is not None:
                    raw.requires_grad_(False)
            except Exception:
                pass


class _FixedStandardize(InputTransform, nn.Module):
    """Z-score inputs with frozen training mean and standard deviation."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std.clamp_min(1e-12)

    def untransform(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean

    def equals(self, other: InputTransform) -> bool:
        if not isinstance(other, _FixedStandardize):
            return False
        return bool(torch.allclose(self.mean, other.mean) and torch.allclose(self.std, other.std))


class _MinMaxOutcome(OutcomeTransform, nn.Module):
    """Affine map of each objective onto the unit interval, inverted on the posterior."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, Y: torch.Tensor, Yvar: torch.Tensor | None = None, X: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scale = self.scale.clamp_min(1e-12)
        y_var = None if Yvar is None else Yvar / scale**2
        return (Y - self.loc) / scale, y_var

    def untransform(
        self, Y: torch.Tensor, Yvar: torch.Tensor | None = None, X: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y_var = None if Yvar is None else Yvar * self.scale**2
        return self.loc + self.scale * Y, y_var

    def subset_output(self, idcs: list[int]) -> _MinMaxOutcome:
        return _MinMaxOutcome(self.loc[..., idcs], self.scale[..., idcs])

    def untransform_posterior(self, posterior, X: torch.Tensor | None = None):  # noqa: ANN001
        loc = self.loc
        scale = self.scale
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=lambda s: loc + scale * s,
            mean_transform=lambda m, v: loc + scale * m,
            variance_transform=lambda m, v: (scale**2) * v,
        )


def _input_transform(tX: torch.Tensor, bounds: torch.Tensor | None, scaling: str | None):
    mode = (scaling or "minmax").strip().lower()
    d = int(tX.shape[-1])
    if mode == "none":
        return None
    if mode == "standardize":
        mean = tX.mean(dim=0)
        std = tX.std(dim=0, unbiased=False)
        return _FixedStandardize(mean, std.clamp_min(1e-6))
    if bounds is not None:
        return Normalize(d=d, bounds=bounds)
    return Normalize(d=d)


def _outcome_transform(yi: torch.Tensor, mode: str | None):
    name = (mode or "standardize").strip().lower()
    if name == "none":
        return None
    if name == "minmax":
        lo = yi.min(dim=0).values
        hi = yi.max(dim=0).values
        return _MinMaxOutcome(lo, (hi - lo).clamp_min(1e-6))
    return Standardize(m=1)


def build_models(
    tX: torch.Tensor,
    tY: torch.Tensor,
    cfg,
    bounds: torch.Tensor | None = None,
    row_masks: list[torch.Tensor] | None = None,
) -> ModelListGP:
    # Without an input transform the GP's RBF kernel learns one lengthscale
    # in the *raw* feature space. When features have wildly different scales
    # (e.g. mass fractions in [0, 1] and homogenization speed in [2000, 12000])
    # the kernel collapses onto the dominant dimension and the posterior is
    # almost flat across the dominated ones. ``Normalize`` applies min/max
    # normalization from the training bounds inside the model so the kernel
    # sees unit-scale inputs.
    scaling = getattr(cfg, "parameter_scaling", None) if cfg is not None else None
    y_norm = getattr(cfg, "value_normalization", None) if cfg is not None else None
    models = []
    for i in range(tY.shape[-1]):
        yi_all = tY[..., i : i + 1]
        if row_masks is not None:
            mask = row_masks[i].to(device=tX.device)
        else:
            mask = torch.isfinite(yi_all).all(dim=-1) & torch.isfinite(tX).all(dim=-1)
        if int(mask.sum()) < 1:
            raise ValueError(f"objective column {i} has no finite observations")
        xi = tX[mask]
        yi = yi_all[mask]
        # A fresh module per objective: one InputTransform cannot be registered twice.
        train_x = xi if scaling and str(scaling).strip().lower() == "standardize" else tX
        m = SingleTaskGP(
            xi,
            yi,
            input_transform=_input_transform(train_x, bounds, scaling),
            outcome_transform=_outcome_transform(yi, y_norm),
        )
        try:
            cfg_gp = {}
            if hasattr(cfg, "gp_config") and cfg.gp_config:
                cfg_gp = dict(cfg.gp_config)
            elif hasattr(cfg, "advanced") and cfg.advanced and isinstance(cfg.advanced, dict):
                cfg_gp = dict(cfg.advanced.get("gp", {}))
            if cfg_gp:
                _apply_gp_hint_priors(m, cfg_gp, tX)
        except Exception:
            pass
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        fit_gpytorch_mll(mll)
        try:
            cfg_gp = {}
            if hasattr(cfg, "gp_config") and cfg.gp_config:
                cfg_gp = dict(cfg.gp_config)
            elif hasattr(cfg, "advanced") and cfg.advanced and isinstance(cfg.advanced, dict):
                cfg_gp = dict(cfg.advanced.get("gp", {}))
            if cfg_gp:
                _freeze_gp_hypers_post_fit(m, cfg_gp, tX)
        except Exception:
            pass
        models.append(m)
    return ModelListGP(*models)


def bounds_input(params: list[dict[str, Any]], tdtype: torch.dtype, tdevice: torch.device) -> torch.Tensor:
    lbs = []
    ubs = []
    for p in params:
        ptype = p.get("type", "float")
        if ptype in ("float", "int"):
            lbs.append(float(p["min"]))
            ubs.append(float(p["max"]))
        elif ptype == "categorical":
            n_choices = len(p.get("choices") or [])
            if n_choices < 1:
                raise RuntimeError(f"categorical parameter {p.get('name')!r} has no choices")
            lbs.append(0.0)
            ubs.append(float(n_choices - 1))
        else:
            raise RuntimeError(f"Unsupported parameter type {ptype!r}")
    return torch.stack(
        [
            torch.tensor(lbs, dtype=tdtype, device=tdevice),
            torch.tensor(ubs, dtype=tdtype, device=tdevice),
        ]
    )


def bounds_model_pca(
    k: int,
    tdtype: torch.dtype,
    tdevice: torch.device,
    margin: float = 0.0,
) -> torch.Tensor:
    """Latent box. ``margin`` expands the training envelope ``[0, 1]`` to ``[-m, 1+m]``."""
    m = float(margin)
    zmin = torch.full((k,), -m, dtype=tdtype, device=tdevice)
    zmax = torch.full((k,), 1.0 + m, dtype=tdtype, device=tdevice)
    return torch.stack([zmin, zmax])
