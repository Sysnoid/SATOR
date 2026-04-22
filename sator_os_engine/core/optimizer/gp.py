from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
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


def build_models(tX: torch.Tensor, tY: torch.Tensor, cfg) -> ModelListGP:
    # Without an input transform the GP's RBF kernel learns one lengthscale
    # in the *raw* feature space. When features have wildly different scales
    # (e.g. mass fractions in [0, 1] and homogenization speed in [2000, 12000])
    # the kernel collapses onto the dominant dimension and the posterior is
    # almost flat across the dominated ones. ``Normalize`` applies min/max
    # normalization from the training bounds inside the model so the kernel
    # sees unit-scale inputs.
    d = int(tX.shape[-1])
    models = []
    for i in range(tY.shape[-1]):
        yi = tY[..., i : i + 1]
        m = SingleTaskGP(
            tX,
            yi,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
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
        if p.get("type", "float") not in ("float", "int"):
            raise RuntimeError("Only float/int parameters supported in v0.1")
        lbs.append(float(p["min"]))
        ubs.append(float(p["max"]))
    return torch.stack(
        [
            torch.tensor(lbs, dtype=tdtype, device=tdevice),
            torch.tensor(ubs, dtype=tdtype, device=tdevice),
        ]
    )


def bounds_model_pca(k: int, tdtype: torch.dtype, tdevice: torch.device) -> torch.Tensor:
    zmin = torch.zeros(k, dtype=tdtype, device=tdevice)
    zmax = torch.ones(k, dtype=tdtype, device=tdevice)
    return torch.stack([zmin, zmax])
