from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from torch.quasirandom import SobolEngine

from .utils import build_linear_constraints as _build_linear_constraints
from .utils import enforce_sum_constraints_np as _enforce_sum_constraints_np
from .utils import feasible_mask as _feasible_mask

_log = logging.getLogger("sator")


def _input_space_to_z_norm(
    grid_np: np.ndarray,
    pca,
    pc_mins: np.ndarray,
    pc_range: np.ndarray,
) -> np.ndarray:
    """Project input points to the same normalized PCA space used to fit the GPs (see ``fit_pca_normalize``)."""
    z_raw = pca.transform(grid_np)
    return (z_raw - pc_mins) / pc_range


def _normalize_acquisition_name(req: Any) -> str:
    cfg = getattr(req, "optimization_config", None)
    if cfg is None:
        return "qnehvi"
    a = getattr(cfg, "acquisition", None) or getattr(cfg, "algorithm", None) or "qnehvi"
    s = str(a).strip().lower().replace("-", "").replace("_", "")
    return s


def _acquisition_params(req: Any) -> dict[str, Any]:
    cfg = getattr(req, "optimization_config", None)
    if cfg is None:
        return {}
    ap = getattr(cfg, "acquisition_params", None)
    return dict(ap) if isinstance(ap, dict) else {}


def _sobol_qmc_sampler(
    sample_size: int, seed: int | None, dtype: torch.dtype, device: torch.device
) -> SobolQMCNormalSampler:
    return SobolQMCNormalSampler(
        sample_shape=torch.Size([int(sample_size)]), seed=int(seed) if seed is not None else None
    )


def _linear_inequality_for_botorch(
    req: Any,
    params: list[dict[str, Any]],
    use_pca_model: bool,
    d_model: int,
    tdevice: torch.device,
    tdtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor, float]] | None:
    if use_pca_model or d_model != len(params):
        return None
    ineq, _ = _build_linear_constraints(req, params)
    if not ineq:
        return None
    return [
        (
            torch.tensor(idxs, dtype=torch.long, device=tdevice),
            torch.tensor(coeffs, dtype=tdtype, device=tdevice),
            float(rhs),
        )
        for idxs, coeffs, rhs in ineq
    ]


def _parego_scalarized_objective(
    tY: torch.Tensor, rng_seed: int | None, tdevice: torch.device, tdtype: torch.dtype
) -> tuple[GenericMCObjective, torch.Tensor]:
    """Random Chebyshev weights (ParEGO-style) and best scalarized value on ``tY``."""
    m = int(tY.shape[-1])
    w = sample_simplex(m, n=1, seed=int(rng_seed) if rng_seed is not None else None, dtype=tdtype, device=tdevice).view(
        -1
    )
    scalarize = get_chebyshev_scalarization(w, tY)
    best_f = scalarize(tY).max()
    obj = GenericMCObjective(lambda samples, X=None: scalarize(samples))
    return obj, best_f


def _single_task_model(model: Any) -> Any:
    return model.models[0] if hasattr(model, "models") else model


def select_candidates_single_objective(
    *,
    model,
    params: list[dict[str, Any]],
    bounds_input: torch.Tensor,
    bounds_model: torch.Tensor,
    use_pca_model: bool,
    pca,
    pc_mins,
    pc_range,
    n: int,
    rng_seed: int | None,
    tdtype,
    tdevice,
    req,
    Y_np: np.ndarray,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    if not hasattr(bounds_model, "shape"):
        bounds_model = torch.tensor(bounds_model, dtype=tdtype, device=tdevice)

    name = _normalize_acquisition_name(req)
    ap = _acquisition_params(req)
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    cfg0 = obj_cfgs[0] if obj_cfgs else {}
    goal = str(cfg0.get("goal", "min")).lower()
    min_like = ("min", "minimize", "minimize_below", "minimize_above")
    max_like = ("max", "maximize", "maximize_below", "maximize_above")
    simple_goal = goal in min_like or goal in max_like
    ucb_beta = float(ap.get("ucb_beta", ap.get("beta", 0.1)) or 0.1)
    pi_tau = float(ap.get("pi_tau", ap.get("tau", 1e-3)) or 1e-3)
    qmc_n = int(ap.get("qmc_samples", 256) or 256)
    # BoTorch path: only for plain min / max goals (same signed training frame as the GP in mobo_engine)
    if simple_goal and name in {
        "qnehvi",
        "qlogehvi",
        "qlogei",
        "qei",
        "qlehvi",
        "qehvi",
        "qucb",
        "ucb",
        "qpi",
        "pi",
        "parego",
        "qparego",
    }:
        st = _single_task_model(model)
        sgn = -1.0 if goal in min_like else 1.0
        y_s = (np.asarray(Y_np, dtype=float).reshape(-1) * sgn).astype(float)
        best_f = float(np.max(y_s))
        torch.manual_seed(int(rng_seed or 0))
        sampler = _sobol_qmc_sampler(qmc_n, rng_seed, tdtype, tdevice)
        if name in ("qucb", "ucb"):
            acqf: Any = qUpperConfidenceBound(st, beta=ucb_beta, sampler=sampler)
        elif name in ("qpi", "pi"):
            acqf = qProbabilityOfImprovement(st, best_f=best_f, sampler=sampler, tau=pi_tau)
        elif name == "qehvi":
            acqf = qExpectedImprovement(st, best_f=best_f, sampler=sampler)
        else:
            # default qnehvi, parego, qlogei, qei, …: log EI (stable)
            acqf = qLogExpectedImprovement(st, best_f=best_f, sampler=sampler)
        d_m = int(bounds_model.shape[1])
        ineq = _linear_inequality_for_botorch(req, params, use_pca_model, d_m, tdevice, tdtype)
        cand, _ = optimize_acqf(
            acqf,
            bounds=bounds_model,
            q=n,
            num_restarts=8,
            raw_samples=256,
            inequality_constraints=ineq,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand_np = cand.detach().cpu().numpy()
        if use_pca_model and pca is not None and pc_mins is not None and pc_range is not None:
            z_norm = cand_np
            z_raw = z_norm * np.asarray(pc_range) + np.asarray(pc_mins)
            x_in = pca.inverse_transform(z_raw)
            x_in = _enforce_sum_constraints_np(x_in, params, req)
            z_norm = (pca.transform(x_in) - np.asarray(pc_mins)) / np.asarray(pc_range)
            return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        x_in = _enforce_sum_constraints_np(cand_np, params, req)
        return torch.tensor(x_in, dtype=tdtype, device=tdevice)

    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=rng_seed or 0)
    raw_n = 512
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    if use_pca_model and pca is not None:
        grid_np = grid.detach().cpu().numpy()
        z_norm = _input_space_to_z_norm(grid_np, pca, pc_mins, pc_range)
        Zgrid = torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        post = model.models[0].posterior(Zgrid)
    else:
        post = model.models[0].posterior(grid)
    mu = post.mean.detach().cpu().numpy().ravel()
    var = post.variance.detach().cpu().numpy().ravel()

    target_val = cfg0.get("target_value")
    # Base score by direction
    if goal in ("min", "minimize", "minimize_below", "minimize_above"):
        score = -mu
    elif goal in ("max", "maximize", "maximize_below", "maximize_above"):
        score = mu
    elif goal in ("within_range",):
        score = np.zeros_like(mu)
    elif goal == "target" and target_val is not None:
        tol = float(getattr(req.optimization_config, "target_tolerance", 0.0) or 0.0)
        var_w = float(getattr(req.optimization_config, "target_variance_penalty", 0.05) or 0.05)
        if tol > 0:
            score = -(np.abs(mu - float(target_val)) / tol) + var_w * np.sqrt(var)
        else:
            score = -np.abs(mu - float(target_val)) + var_w * np.sqrt(var)
    else:
        score = -mu
    # Threshold / range shaping (demo-compatible, value scale)
    thr = cfg0.get("threshold") or cfg0.get("thresholds")
    rng = cfg0.get("range") or cfg0.get("ranges")
    if isinstance(thr, dict) and (thr.get("value") is not None):
        t_val = float(thr.get("value"))
        ttype_raw = str(thr.get("type", "")).lower()
        ttype = ttype_raw or ("<=" if ("below" in goal) else ">=" if ("above" in goal) else None)
        wthr = float(thr.get("weight", 0.25))
        if ttype in (">=", ">", "ge", "above"):
            score = score + wthr * np.maximum(mu - t_val, 0.0)
        elif ttype in ("<=", "<", "le", "below"):
            score = score + wthr * np.maximum(t_val - mu, 0.0)
    if isinstance(rng, dict) and (rng.get("min") is not None) and (rng.get("max") is not None):
        a = float(rng.get("min"))
        b = float(rng.get("max"))
        if a > b:
            a, b = b, a
        wr = float(rng.get("weight", 0.25))
        below = np.maximum(a - mu, 0.0)
        above = np.maximum(mu - b, 0.0)
        penalty = below + above
        score = score - wr * (penalty**2)
        if rng.get("ideal") is not None:
            ideal = float(rng.get("ideal"))
            wi = float(rng.get("ideal_weight", rng.get("weight", 0.25)))
            score = score - wi * ((mu - ideal) ** 2)
    # slight variance regularization
    score = score - 0.05 * np.sqrt(var)

    feas = _feasible_mask(grid.detach().cpu().numpy().tolist(), req, params)
    score = np.where(np.array(feas, dtype=bool), score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_input_np = grid.detach().cpu().numpy().copy()[top_idx]
    cand_input_np = _enforce_sum_constraints_np(cand_input_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_input_np)
        z_norm = (z_raw - pc_mins) / pc_range
        return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
    return torch.tensor(cand_input_np, dtype=tdtype, device=tdevice)


def select_candidates_multiobjective(
    *,
    model,
    params: list[dict[str, Any]],
    bounds_input: torch.Tensor,
    bounds_model: torch.Tensor,
    use_pca_model: bool,
    pca,
    pc_mins,
    pc_range,
    n: int,
    rng_seed: int | None,
    tdtype,
    tdevice,
    req,
    goals: list[str],
    Y_np: np.ndarray,
    train_X: torch.Tensor,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    if not hasattr(bounds_model, "shape"):
        bounds_model = torch.tensor(bounds_model, dtype=tdtype, device=tdevice)
    has_advanced = any(g not in ("min", "max") for g in goals)
    tY = torch.tensor(
        Y_np * np.array([1.0 if g == "max" else -1.0 for g in goals], dtype=float), dtype=tdtype, device=tdevice
    )
    if not has_advanced:
        name = _normalize_acquisition_name(req)
        ap = _acquisition_params(req)
        qmc_n = int(ap.get("qmc_samples", 256) or 256)
        ucb_beta = float(ap.get("ucb_beta", ap.get("beta", 0.1)) or 0.1)
        pi_tau = float(ap.get("pi_tau", ap.get("tau", 1e-3)) or 1e-3)
        torch.manual_seed(int(rng_seed or 0))
        rp = tY.min(dim=0).values - 0.1 * tY.abs().mean(dim=0).clamp_min(1.0)
        part = NondominatedPartitioning(ref_point=rp.detach().cpu(), Y=tY.detach().cpu())
        d_model = int(bounds_model.shape[1])
        botorch_ineq = _linear_inequality_for_botorch(req, params, use_pca_model, d_model, tdevice, tdtype)
        acqf: Any
        if name in ("qehvi", "ehvi"):
            acqf = qExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), partitioning=part)
        elif name in ("qnoisyehvi", "qlognoisyehvi"):
            acqf = qLogNoisyExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), X_baseline=train_X)
        elif name in ("qnehvi", "qlogehvi", "qlehvi", "logehvi"):
            acqf = qLogExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), partitioning=part)
        elif name in ("parego", "qparego", "qei"):
            sampler = _sobol_qmc_sampler(qmc_n, rng_seed, tdtype, tdevice)
            mo_obj, best_f = _parego_scalarized_objective(tY, rng_seed, tdevice, tdtype)
            acqf = qLogExpectedImprovement(model, best_f=best_f, sampler=sampler, objective=mo_obj)
        elif name in ("qpi", "pi"):
            sampler = _sobol_qmc_sampler(qmc_n, rng_seed, tdtype, tdevice)
            mo_obj, best_f = _parego_scalarized_objective(tY, rng_seed, tdevice, tdtype)
            acqf = qProbabilityOfImprovement(model, best_f=best_f, sampler=sampler, objective=mo_obj, tau=pi_tau)
        elif name in ("qucb", "ucb"):
            sampler = _sobol_qmc_sampler(qmc_n, rng_seed, tdtype, tdevice)
            mo_obj, _ = _parego_scalarized_objective(tY, rng_seed, tdevice, tdtype)
            acqf = qUpperConfidenceBound(model, beta=ucb_beta, sampler=sampler, objective=mo_obj)
        else:
            _log.debug("Unknown multi-objective acquisition name %r; using qLogExpectedHypervolumeImprovement", name)
            acqf = qLogExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), partitioning=part)
        cand, _ = optimize_acqf(
            acqf,
            bounds=bounds_model,
            q=n,
            num_restarts=8,
            raw_samples=256,
            inequality_constraints=botorch_ineq,
            options={"batch_limit": 5, "maxiter": 200},
        )
        cand_np = cand.detach().cpu().numpy()
        if use_pca_model and pca is not None:
            z_norm = cand_np
            z_raw = z_norm * pc_range + pc_mins
            x_in = pca.inverse_transform(z_raw)
            x_in = _enforce_sum_constraints_np(x_in, params, req)
            z_norm = (pca.transform(x_in) - pc_mins) / pc_range
            return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        x_in = _enforce_sum_constraints_np(cand_np, params, req)
        return torch.tensor(x_in, dtype=tdtype, device=tdevice)

    # Advanced: sampling + scoring
    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=rng_seed or 0)
    raw_n = 1024
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    if use_pca_model and pca is not None:
        grid_np = grid.detach().cpu().numpy()
        z_norm = _input_space_to_z_norm(grid_np, pca, pc_mins, pc_range)
        Zgrid = torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        posts = [m.posterior(Zgrid) for m in model.models]
    else:
        posts = [m.posterior(grid) for m in model.models]
    mu_list = [p.mean.detach().cpu().numpy().ravel() for p in posts]
    var_list = [p.variance.detach().cpu().numpy().ravel() for p in posts]
    score = np.zeros(raw_n)
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    for k, cfg_o in enumerate(obj_cfgs):
        goal = str(cfg_o.get("goal", "min")).lower()
        target_val = cfg_o.get("target_value")
        mu = mu_list[k]
        var = var_list[k]
        # base direction
        if goal in ("min", "minimize", "minimize_below", "minimize_above"):
            score_k = -mu
        elif goal in ("max", "maximize", "maximize_below", "maximize_above"):
            score_k = mu
        elif goal == "within_range":
            score_k = np.zeros_like(mu)
        elif goal == "target" and target_val is not None:
            tol = float(getattr(req.optimization_config, "target_tolerance", 0.0) or 0.0)
            var_w = float(getattr(req.optimization_config, "target_variance_penalty", 0.05) or 0.05)
            if tol > 0:
                score_k = -(np.abs(mu - float(target_val)) / tol) + var_w * np.sqrt(var)
            else:
                score_k = -np.abs(mu - float(target_val)) + var_w * np.sqrt(var)
        elif goal in ("explore", "probe"):
            score_k = np.sqrt(var)
        elif goal == "improve":
            gk = str(goals[k] if k < len(goals) else "min").lower()
            sgn = 1.0 if gk == "max" else -1.0
            y_col = Y_np[:, k] if Y_np.ndim > 1 else np.asarray(Y_np, dtype=float).ravel()
            t_y_ref = y_col * sgn
            best = float(np.max(t_y_ref))
            score_k = np.maximum(0.0, best - mu)
        else:
            score_k = -mu
        # threshold/range shaping
        thr = cfg_o.get("threshold") or cfg_o.get("thresholds")
        rng = cfg_o.get("range") or cfg_o.get("ranges")
        if isinstance(thr, dict) and (thr.get("value") is not None):
            t_val = float(thr.get("value"))
            ttype_raw = str(thr.get("type", "")).lower()
            ttype = ttype_raw or ("<=" if ("below" in goal) else ">=" if ("above" in goal) else None)
            wthr = float(thr.get("weight", 0.25))
            if ttype in (">=", ">", "ge", "above"):
                score_k = score_k + wthr * np.maximum(mu - t_val, 0.0)
            elif ttype in ("<=", "<", "le", "below"):
                score_k = score_k + wthr * np.maximum(t_val - mu, 0.0)
        if isinstance(rng, dict) and (rng.get("min") is not None) and (rng.get("max") is not None):
            a = float(rng.get("min"))
            b = float(rng.get("max"))
            if a > b:
                a, b = b, a
            wr = float(rng.get("weight", 0.25))
            below = np.maximum(a - mu, 0.0)
            above = np.maximum(mu - b, 0.0)
            penalty = below + above
            score_k = score_k - wr * (penalty**2)
            if rng.get("ideal") is not None:
                ideal = float(rng.get("ideal"))
                wi = float(rng.get("ideal_weight", rng.get("weight", 0.25)))
                score_k = score_k - wi * ((mu - ideal) ** 2)
        score_k = score_k - 0.05 * np.sqrt(var)
        score += score_k
    feas = _feasible_mask(grid.detach().cpu().numpy().tolist(), req, params)
    score = np.where(np.array(feas, dtype=bool), score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_np = grid.detach().cpu().numpy().copy()[top_idx]
    cand_np = _enforce_sum_constraints_np(cand_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_np)
        z_norm = (z_raw - pc_mins) / pc_range
        return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
    return torch.tensor(cand_np, dtype=tdtype, device=tdevice)
