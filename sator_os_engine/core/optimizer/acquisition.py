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

from .utils import (
    KNOWN_ACQUISITIONS,
    MAX_GOALS,
    MIN_GOALS,
    THRESHOLD_GOALS,
    normalize_acquisition_name,
    normalize_goal,
    optimization_lock,
    resolve_seed,
    threshold_spec,
    training_sign,
)
from .utils import build_linear_constraints as _build_linear_constraints
from .utils import enforce_ratio_constraints_np as _enforce_ratio_constraints_np
from .utils import enforce_sum_constraints_np as _enforce_sum_constraints_np
from .utils import evaluate_enforced_goals as _evaluate_enforced_goals
from .utils import extract_enforced_goal_specs as _extract_enforced_goal_specs
from .utils import feasible_mask as _feasible_mask

_log = logging.getLogger("sator")


# =============================================================================
# Soft vs hard threshold goals -- READ BEFORE DEBUGGING "MY THRESHOLD WAS
# IGNORED"
# =============================================================================
# The following goal names are **SOFT**: they only add a shaping term to the
# acquisition score. They do NOT filter candidates and they do NOT guarantee
# that returned predictions satisfy the threshold on the GP posterior.
#
#   - "within_range"                         (soft range penalty + ideal pull)
#   - "target"                               (distance-to-target penalty)
#
# Threshold goals ("minimize_below", "maximize_above", "maximize_below",
# "minimize_above") are outcome constraints on qLogEI / qLogEHVI. They are
# not the Sobol penalty path. Use them when the feasible side of T matters
# and a weight must not flip that side.
#
# For hard thresholds that MUST hold on the GP posterior, use the
# "enforce_*" goal family (parsed by utils.extract_enforced_goal_specs and
# evaluated by utils.evaluate_enforced_goals):
#
#   - "enforce_above"         {"goal": "enforce_above", "threshold_value": T}
#   - "enforce_below"         {"goal": "enforce_below", "threshold_value": T}
#   - "enforce_within_range"  {"goal": "enforce_within_range",
#                              "range": {"min": lo, "max": hi}}
#
# enforce_* goals apply a hard feasibility mask on the Sobol scoring grid
# and tag every returned prediction with ``enforced_goals_satisfied`` +
# ``enforced_violations``. Set ``optimization_config.enforcement_uncertainty_margin``
# > 0 to require a confidence-bound (LCB/UCB) to satisfy the threshold.
# See docs/06-objectives-and-constraints.md for the full contract.
# =============================================================================


def _input_space_to_z_norm(
    grid_np: np.ndarray,
    pca,
    pc_mins: np.ndarray,
    pc_range: np.ndarray,
) -> np.ndarray:
    """Project input points to the same normalized PCA space used to fit the GPs (see ``fit_pca_normalize``)."""
    z_raw = pca.transform(grid_np)
    return (z_raw - pc_mins) / pc_range


_EHVI_NAMES = frozenset({"qehvi", "ehvi", "qnehvi", "qlogehvi", "qlehvi", "logehvi", "qnoisyehvi", "qlognoisyehvi"})
_BOTORCH_NAMES = KNOWN_ACQUISITIONS - {"sobol"}


def _normalize_acquisition_name(req: Any) -> str:
    cfg = getattr(req, "optimization_config", None)
    if cfg is None:
        return "qnehvi"
    raw = getattr(cfg, "acquisition", None) or getattr(cfg, "algorithm", None) or "qnehvi"
    name = normalize_acquisition_name(raw)
    if name not in KNOWN_ACQUISITIONS:
        raise ValueError(f"unknown acquisition {raw!r}")
    return name


def _objective_sign(goal: str) -> float:
    """Sign the engine used when it trained on ``Y * sign``. See ``training_sign``."""
    return training_sign(goal)


def _posterior_mean_to_user_units(mu: np.ndarray, goal: str) -> np.ndarray:
    """Undo the engine's sign flip so ``mu`` is in original objective units."""
    return np.asarray(mu, dtype=float) * _objective_sign(goal)


def _target_soft_score(
    req: Any,
    cfg_o: dict[str, Any],
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    """Soft ``goal=target`` score in user units.

    Per-objective knobs (preferred) fall back to ``optimization_config``:

    - ``target_tolerance`` — scale for ``|μ − T|`` (global config if omitted)
    - ``target_variance_penalty`` — exploration bonus on ``√var``
    - ``weight`` — multiplies this objective's contribution when scores are
      summed across targets (default ``1.0``). Use this to emphasize a gap
      channel (e.g. weight a* higher than L*).
    """
    target_val = cfg_o.get("target_value")
    if target_val is None:
        return np.zeros_like(mu, dtype=float)

    tol = cfg_o.get("target_tolerance")
    if tol is None:
        tol = getattr(req.optimization_config, "target_tolerance", None)
    tol = float(tol or 0.0)

    var_w = cfg_o.get("target_variance_penalty")
    if var_w is None:
        var_w = getattr(req.optimization_config, "target_variance_penalty", None)
    if var_w is None:
        var_w = 0.05
    var_w = float(var_w)

    weight = float(cfg_o.get("weight", 1.0) or 1.0)
    err = np.abs(mu - float(target_val))
    base = -(err / tol) + var_w * np.sqrt(var) if tol > 0 else -err + var_w * np.sqrt(var)
    return weight * base


def _acquisition_params(req: Any) -> dict[str, Any]:
    cfg = getattr(req, "optimization_config", None)
    if cfg is None:
        return {}
    ap = getattr(cfg, "acquisition_params", None)
    return dict(ap) if isinstance(ap, dict) else {}


def _advanced(req: Any) -> dict[str, Any]:
    cfg = getattr(req, "optimization_config", None)
    if cfg is None:
        return {}
    adv = getattr(cfg, "advanced", None)
    return dict(adv) if isinstance(adv, dict) else {}


def _objective_cfgs(req: Any, goals: list[str] | None = None) -> list[dict[str, Any]]:
    raw = list(req.objectives.values()) if isinstance(getattr(req, "objectives", None), dict) else []
    cfgs = [c if isinstance(c, dict) else {} for c in raw]
    if goals is not None and len(cfgs) < len(goals):
        cfgs = cfgs + [{"goal": g} for g in goals[len(cfgs) :]]
    return cfgs


def _has_threshold(goal: str, cfg: dict[str, Any]) -> bool:
    return threshold_spec({**cfg, "goal": normalize_goal(goal)}) is not None


def _is_directional(goal: str, cfg: dict[str, Any]) -> bool:
    g = normalize_goal(goal)
    if g in ("min", "minimize", "max", "maximize"):
        return True
    return g in THRESHOLD_GOALS and not _has_threshold(g, cfg)


def _optimizer_controls(req: Any, *, q: int, n_obj: int, name: str) -> dict[str, Any]:
    ap = _acquisition_params(req)
    adv = _advanced(req)
    num_restarts = int(ap.get("num_restarts", adv.get("num_restarts", 8)))
    raw_samples = int(ap.get("raw_samples", adv.get("raw_samples", 256)))
    maxiter = int(ap.get("acq_maxiter", adv.get("acq_maxiter", adv.get("maxiter", 200))))
    batch_limit = int(ap.get("batch_limit", adv.get("batch_limit", 5)))
    sequential = bool(ap.get("sequential", adv.get("sequential", False)))
    if name in _EHVI_NAMES and (q > 1 or n_obj > 2):
        sequential = True
    return {
        "num_restarts": max(1, num_restarts),
        "raw_samples": max(1, raw_samples),
        "options": {"batch_limit": max(1, batch_limit), "maxiter": max(1, maxiter)},
        "sequential": sequential,
    }


def _reference_point(tY: torch.Tensor, req: Any, signs: np.ndarray) -> torch.Tensor:
    """Hypervolume reference in the signed training frame.

    Default is ``min(Y) - 0.1 * (max(Y) - min(Y))`` per objective, which stays
    in the same place when an objective is rescaled. A user ``ref_point`` is
    read in original Y units and multiplied by the training signs.
    """
    ap = _acquisition_params(req)
    user = ap.get("ref_point", _advanced(req).get("ref_point"))
    if user is not None:
        rp = torch.tensor(list(user), dtype=tY.dtype, device=tY.device)
        sgn = torch.as_tensor(signs, dtype=tY.dtype, device=tY.device)
        return rp * sgn
    y_min = tY.min(dim=0).values
    y_max = tY.max(dim=0).values
    span = (y_max - y_min).clamp_min(1e-6)
    return y_min - 0.1 * span


def _threshold_constraints(cfgs: list[dict[str, Any]]) -> list[Any] | None:
    """Outcome constraints in user units. Positive means violated."""
    cons: list[Any] = []
    for index, cfg in enumerate(cfgs):
        goal = normalize_goal(cfg.get("goal", "min"))
        spec = threshold_spec({**cfg, "goal": goal})
        if spec is None:
            continue
        side, level = spec
        sign = training_sign(goal)

        def _constraint(
            z: torch.Tensor, *, index: int = index, side: str = side, level: float = level, sign: float = sign
        ) -> torch.Tensor:
            f = z[..., index] * sign
            if side == "below":
                return f - level
            return level - f

        cons.append(_constraint)
    return cons or None


def _best_signed(y_user: np.ndarray, goal: str, cfg: dict[str, Any]) -> float:
    y = np.asarray(y_user, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("objective has no finite observations")
    sign = training_sign(goal)
    signed = y * sign
    spec = threshold_spec({**cfg, "goal": normalize_goal(goal)})
    if spec is not None:
        side, level = spec
        feasible = y <= level if side == "below" else y >= level
        if np.any(feasible):
            return float(np.max(signed[feasible]))
    return float(np.max(signed))


def _bound_arrays(bounds: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = bounds.detach().cpu().numpy() if hasattr(bounds, "detach") else np.asarray(bounds, dtype=float)
    return np.asarray(arr[0], dtype=float), np.asarray(arr[1], dtype=float)


def _project_rows(rows: np.ndarray, params: list[dict[str, Any]], req: Any) -> np.ndarray:
    x = _enforce_sum_constraints_np(rows, params, req)
    x = _enforce_ratio_constraints_np(x, params, req)
    return _enforce_sum_constraints_np(x, params, req)


def _dedup_rows(
    rows: np.ndarray,
    observed: np.ndarray | None,
    bounds: Any,
    n: int,
    seed: int,
) -> np.ndarray:
    """Drop points within 1e-6 (normalised) of data or each other, then refill."""
    lo, hi = _bound_arrays(bounds)
    span = np.maximum(hi - lo, 1e-12)
    kept: list[np.ndarray] = []
    pool: list[np.ndarray] = []

    def _norm(row: np.ndarray) -> np.ndarray:
        return (np.asarray(row, dtype=float) - lo) / span

    if observed is not None and np.size(observed):
        for row in np.atleast_2d(np.asarray(observed, dtype=float)):
            if row.shape[0] == lo.shape[0]:
                pool.append(_norm(row))

    def _novel(row: np.ndarray) -> bool:
        z = _norm(row)
        return all(float(np.linalg.norm(z - prev)) >= 1e-6 for prev in pool)

    for row in np.atleast_2d(np.asarray(rows, dtype=float)):
        if row.shape[0] != lo.shape[0] or not _novel(row):
            continue
        kept.append(np.asarray(row, dtype=float))
        pool.append(_norm(row))
        if len(kept) >= n:
            return np.stack(kept[:n], axis=0)
    sob = SobolEngine(dimension=int(lo.shape[0]), scramble=True, seed=int(seed) % (2**31 - 1) or 1)
    draw = sob.draw(max(64, n * 16)).cpu().numpy()
    for unit in draw:
        row = lo + (hi - lo) * np.asarray(unit, dtype=float)
        if not _novel(row):
            continue
        kept.append(row)
        pool.append(_norm(row))
        if len(kept) >= n:
            break
    while len(kept) < n:
        step = (len(kept) + 1) * 1e-3
        base = kept[0] if kept else (lo + hi) / 2.0
        row = np.clip(base + step, lo, hi)
        kept.append(row)
        pool.append(_norm(row))
    return np.stack(kept[:n], axis=0)


def _standardize_score(score: np.ndarray, y_col: np.ndarray, var: np.ndarray, *, explore: bool) -> np.ndarray:
    """Divide by the training standard deviation and add a UCB bonus."""
    scale = float(np.std(np.asarray(y_col, dtype=float)))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    out = np.asarray(score, dtype=float) / scale
    if explore:
        out = out + 0.1 * np.sqrt(np.maximum(var, 0.0)) / scale
    return out


def _as_pending(x_pending: torch.Tensor | None, tdtype, tdevice) -> torch.Tensor | None:
    if x_pending is None:
        return None
    return x_pending.to(dtype=tdtype, device=tdevice)


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


def _enforcement_mask_from_posteriors(
    req: Any,
    mu_list: list[np.ndarray],
    var_list: list[np.ndarray],
) -> np.ndarray | None:
    """Build a boolean feasibility mask from enforce_* goal specs.

    Returns ``None`` when no enforce_* goal is configured (callers should
    skip the whole check). Otherwise returns shape ``(n_grid,)`` where
    ``True`` means every enforce_* threshold is satisfied on the GP
    posterior for that Sobol row. ``mu_list`` / ``var_list`` are ordered
    to match ``req.objectives``.

    The GPs in ``mobo_engine.run_optimization`` are trained on a
    sign-flipped ``Y`` (``+1`` for ``max`` goals, ``-1`` for everything
    else) so that all objectives live in a minimization frame. User
    thresholds, on the other hand, are always stated in the original
    ``Y`` units. We therefore flip ``mu`` back to user units before
    comparing, using the same sign convention as the orchestrator
    (see ``_posterior_mean_to_user_units``).
    """
    specs = _extract_enforced_goal_specs(req)
    if not specs:
        return None
    if not mu_list:
        return None
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    mu_cols = []
    for k, m in enumerate(mu_list):
        goal_k = str(obj_cfgs[k].get("goal", "min")).lower() if k < len(obj_cfgs) else "min"
        mu_cols.append(_posterior_mean_to_user_units(m, goal_k))
    mu_mat = np.column_stack(mu_cols)
    if var_list and len(var_list) == len(mu_list):
        # Variance is invariant under sign flip (Var[-X] = Var[X]).
        var_mat = np.column_stack([np.asarray(v, dtype=float).ravel() for v in var_list])
    else:
        var_mat = None
    margin = 0.0
    cfg = getattr(req, "optimization_config", None)
    if cfg is not None:
        margin = float(getattr(cfg, "enforcement_uncertainty_margin", 0.0) or 0.0)
    mask, _ = _evaluate_enforced_goals(specs, mu_mat, var_mat, margin)
    return mask


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
    train_X: torch.Tensor | None = None,
    X_pending: torch.Tensor | None = None,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    if not hasattr(bounds_model, "shape"):
        bounds_model = torch.tensor(bounds_model, dtype=tdtype, device=tdevice)

    seed = resolve_seed(rng_seed)
    name = _normalize_acquisition_name(req)
    ap = _acquisition_params(req)
    obj_cfgs = _objective_cfgs(req)
    cfg0 = obj_cfgs[0] if obj_cfgs else {"goal": "min"}
    goal = normalize_goal(cfg0.get("goal", "min"))
    constrained = _has_threshold(goal, cfg0)
    directional = _is_directional(goal, cfg0)
    ucb_beta = float(ap.get("ucb_beta", ap.get("beta", 0.1)) or 0.1)
    pi_tau = float(ap.get("pi_tau", ap.get("tau", 1e-3)) or 1e-3)
    qmc_n = int(ap.get("qmc_samples", 256) or 256)
    # Threshold goals always use a constrained qLogEI. Plain min/max use BoTorch
    # unless the caller forced the Sobol grid. Both share the engine's signed Y.
    if constrained or (directional and name in _BOTORCH_NAMES):
        st = _single_task_model(model)
        best_f = _best_signed(np.asarray(Y_np, dtype=float), goal, cfg0)
        pending = _as_pending(X_pending, tdtype, tdevice)
        cons = _threshold_constraints([cfg0]) if constrained else None
        sampler = _sobol_qmc_sampler(qmc_n, seed, tdtype, tdevice)
        if constrained or name not in ("qucb", "ucb", "qpi", "pi", "qehvi"):
            acqf: Any = qLogExpectedImprovement(st, best_f=best_f, sampler=sampler, constraints=cons, X_pending=pending)
        elif name in ("qucb", "ucb"):
            acqf = qUpperConfidenceBound(st, beta=ucb_beta, sampler=sampler)
        elif name in ("qpi", "pi"):
            acqf = qProbabilityOfImprovement(st, best_f=best_f, sampler=sampler, tau=pi_tau)
        else:
            acqf = qExpectedImprovement(st, best_f=best_f, sampler=sampler)
        d_m = int(bounds_model.shape[1])
        ineq = _linear_inequality_for_botorch(req, params, use_pca_model, d_m, tdevice, tdtype)
        controls = _optimizer_controls(req, q=n, n_obj=1, name=name if not constrained else "qlogei")
        with optimization_lock, torch.random.fork_rng():
            torch.manual_seed(seed)
            cand, _ = optimize_acqf(
                acqf,
                bounds=bounds_model,
                q=n,
                num_restarts=controls["num_restarts"],
                raw_samples=controls["raw_samples"],
                inequality_constraints=ineq,
                options=controls["options"],
                sequential=controls["sequential"],
            )
        cand_np = cand.detach().cpu().numpy()
        observed = train_X.detach().cpu().numpy() if train_X is not None else None
        if pending is not None:
            pend_np = pending.detach().cpu().numpy()
            observed = pend_np if observed is None else np.vstack([observed, pend_np])
        if use_pca_model and pca is not None and pc_mins is not None and pc_range is not None:
            z_raw = cand_np * np.asarray(pc_range) + np.asarray(pc_mins)
            x_in = _project_rows(pca.inverse_transform(z_raw), params, req)
            z_norm = (pca.transform(x_in) - np.asarray(pc_mins)) / np.asarray(pc_range)
            lo, hi = _bound_arrays(bounds_model)
            z_norm = np.clip(z_norm, lo, hi)
            z_norm = _dedup_rows(z_norm, observed, bounds_model, n, seed)
            return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        x_in = _project_rows(cand_np, params, req)
        x_in = _dedup_rows(x_in, observed, bounds_model, n, seed)
        x_in = _project_rows(x_in, params, req)
        return torch.tensor(x_in, dtype=tdtype, device=tdevice)

    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=seed)
    raw_n = 512
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    # Sobol samples uniformly over bounds; they will essentially never satisfy
    # a sum-to-target constraint exactly. Project the grid onto the sum target
    # first so downstream feasibility checks see realistic candidates rather
    # than dismissing every sample because of a sum mismatch.
    grid_input_np = _enforce_sum_constraints_np(grid.detach().cpu().numpy(), params, req)
    grid = torch.tensor(grid_input_np, dtype=tdtype, device=tdevice)
    in_envelope = None
    if use_pca_model and pca is not None:
        z_norm = _input_space_to_z_norm(grid_input_np, pca, pc_mins, pc_range)
        # The GP was fit on z_norm in [0,1]^k (the training PC envelope).
        # Samples that project outside that envelope are extrapolations and
        # would be plotted outside the rendered GP surface. Track which
        # samples stay inside so we can prefer them in the feasibility mask
        # below; this mirrors bounds_model_pca=[0,1]^k used by the BoTorch
        # path, which keeps its optimiser strictly in-envelope by construction.
        lo_m, hi_m = _bound_arrays(bounds_model)
        in_envelope = np.all((z_norm >= lo_m) & (z_norm <= hi_m), axis=1)
        Zgrid = torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        post = model.models[0].posterior(Zgrid)
    else:
        post = model.models[0].posterior(grid)
    mu_signed = post.mean.detach().cpu().numpy().ravel()
    var = post.variance.detach().cpu().numpy().ravel()
    # Soft scores / thresholds are in original Y units; undo engine sign flip.
    mu = _posterior_mean_to_user_units(mu_signed, goal)

    # Base score by direction (``mu`` is now in user / original Y units)
    if goal in MIN_GOALS:
        score = -mu
    elif goal in MAX_GOALS:
        score = mu
    elif goal in ("within_range",):
        score = np.zeros_like(mu)
    elif goal == "target" and cfg0.get("target_value") is not None:
        score = _target_soft_score(req, cfg0, mu, var)
    elif goal in ("explore", "probe"):
        score = np.sqrt(var)
    elif goal == "improve":
        # Prefer predicted improvement over the best observed value (user units).
        y_col = Y_np[:, 0] if getattr(Y_np, "ndim", 1) > 1 else np.asarray(Y_np, dtype=float).ravel()
        if training_sign(goal) > 0:
            best = float(np.max(y_col))
            score = np.maximum(0.0, mu - best)
        else:
            best = float(np.min(y_col))
            score = np.maximum(0.0, best - mu)
    else:
        score = -mu
    # Threshold / range shaping. Threshold *goals* are handled on the BoTorch
    # path above; this block remains for an explicit threshold dict on a soft goal.
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
    y_col = Y_np[:, 0] if getattr(Y_np, "ndim", 1) > 1 else np.asarray(Y_np, dtype=float).ravel()
    score = _standardize_score(score, y_col, var, explore=goal != "target")

    grid_for_feas = grid.detach().cpu().numpy()
    feas_arr = np.array(_feasible_mask(grid_for_feas.tolist(), req, params), dtype=bool)
    # Combine input-space feasibility (sum / bounds / ratio) with PC-envelope
    # feasibility (PCA GP trained on z_norm in [0,1]^k). The envelope filter
    # is applied softly: if no sample is both input-feasible and in-envelope,
    # we back off to input-feasibility alone, then to score-only ranking.
    if use_pca_model and in_envelope is not None:
        combined = feas_arr & in_envelope
        if combined.any():
            feas_arr = combined
    # Hard-enforce goal feasibility on the GP posterior (enforce_above /
    # enforce_below / enforce_within_range). Single-objective path only
    # reaches here for non-simple goals, so cfg0 is the relevant objective;
    # still feed it through the generic helper in case more objectives are
    # declared (the helper indexes by req.objectives order).
    # ``mu_signed`` is still in the training frame. The helper unsigns once.
    enf_mask = _enforcement_mask_from_posteriors(req, [mu_signed], [var])
    if enf_mask is not None:
        combined = feas_arr & enf_mask
        if combined.any():
            feas_arr = combined
    if feas_arr.any():
        score = np.where(feas_arr, score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_input_np = grid_for_feas.copy()[top_idx]
    cand_input_np = _enforce_sum_constraints_np(cand_input_np, params, req)
    cand_input_np = _enforce_ratio_constraints_np(cand_input_np, params, req)
    cand_input_np = _enforce_sum_constraints_np(cand_input_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_input_np)
        z_norm = (z_raw - pc_mins) / pc_range
        lo, hi = _bound_arrays(bounds_model)
        z_norm = np.clip(z_norm, lo, hi)
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
    X_pending: torch.Tensor | None = None,
) -> torch.Tensor:
    # Coerce bounds to tensors if lists were passed
    if not hasattr(bounds_input, "shape"):
        bounds_input = torch.tensor(bounds_input, dtype=tdtype, device=tdevice)
    if not hasattr(bounds_model, "shape"):
        bounds_model = torch.tensor(bounds_model, dtype=tdtype, device=tdevice)
    seed = resolve_seed(rng_seed)
    name = _normalize_acquisition_name(req)
    goal_names = [normalize_goal(g) for g in goals]
    cfgs = _objective_cfgs(req, goal_names)
    for i, g in enumerate(goal_names):
        cfgs[i] = {**cfgs[i], "goal": g}
    constrained = any(_has_threshold(g, cfgs[i]) for i, g in enumerate(goal_names))
    directional = all(_is_directional(g, cfgs[i]) or _has_threshold(g, cfgs[i]) for i, g in enumerate(goal_names))
    signs = np.array([training_sign(g) for g in goal_names], dtype=float)
    y_raw = np.asarray(Y_np, dtype=float)
    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)
    finite_rows = np.isfinite(y_raw).all(axis=1)
    if not np.any(finite_rows):
        raise ValueError("no row is finite across every objective")
    tY = torch.tensor(y_raw[finite_rows] * signs, dtype=tdtype, device=tdevice)
    baseline_X = train_X[torch.as_tensor(finite_rows, device=train_X.device)]
    if directional and (constrained or name in _BOTORCH_NAMES):
        ap = _acquisition_params(req)
        qmc_n = int(ap.get("qmc_samples", 256) or 256)
        ucb_beta = float(ap.get("ucb_beta", ap.get("beta", 0.1)) or 0.1)
        pi_tau = float(ap.get("pi_tau", ap.get("tau", 1e-3)) or 1e-3)
        pending = _as_pending(X_pending, tdtype, tdevice)
        cons = _threshold_constraints(cfgs)
        rp = _reference_point(tY, req, signs)
        part = NondominatedPartitioning(ref_point=rp.detach().cpu(), Y=tY.detach().cpu())
        d_model = int(bounds_model.shape[1])
        botorch_ineq = _linear_inequality_for_botorch(req, params, use_pca_model, d_model, tdevice, tdtype)
        acqf: Any
        if name in ("qehvi", "ehvi") and not constrained:
            acqf = qExpectedHypervolumeImprovement(model=model, ref_point=rp.tolist(), partitioning=part)
        elif name in ("qnoisyehvi", "qlognoisyehvi") or constrained:
            acqf = (
                qLogNoisyExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=rp.tolist(),
                    X_baseline=baseline_X,
                    prune_baseline=True,
                    constraints=cons,
                    X_pending=pending,
                )
                if name in ("qnoisyehvi", "qlognoisyehvi")
                else qLogExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=rp.tolist(),
                    partitioning=part,
                    constraints=cons,
                    X_pending=pending,
                )
            )
        elif name in ("qnehvi", "qlogehvi", "qlehvi", "logehvi"):
            acqf = qLogExpectedHypervolumeImprovement(
                model=model, ref_point=rp.tolist(), partitioning=part, constraints=cons, X_pending=pending
            )
        elif name in ("parego", "qparego", "qei"):
            sampler = _sobol_qmc_sampler(qmc_n, seed, tdtype, tdevice)
            mo_obj, best_f = _parego_scalarized_objective(tY, seed, tdevice, tdtype)
            acqf = qLogExpectedImprovement(
                model, best_f=best_f, sampler=sampler, objective=mo_obj, constraints=cons, X_pending=pending
            )
        elif name in ("qpi", "pi"):
            sampler = _sobol_qmc_sampler(qmc_n, seed, tdtype, tdevice)
            mo_obj, best_f = _parego_scalarized_objective(tY, seed, tdevice, tdtype)
            acqf = qProbabilityOfImprovement(model, best_f=best_f, sampler=sampler, objective=mo_obj, tau=pi_tau)
        elif name in ("qucb", "ucb"):
            sampler = _sobol_qmc_sampler(qmc_n, seed, tdtype, tdevice)
            mo_obj, _ = _parego_scalarized_objective(tY, seed, tdevice, tdtype)
            acqf = qUpperConfidenceBound(model, beta=ucb_beta, sampler=sampler, objective=mo_obj)
        else:
            acqf = qLogExpectedHypervolumeImprovement(
                model=model, ref_point=rp.tolist(), partitioning=part, constraints=cons, X_pending=pending
            )
        controls = _optimizer_controls(req, q=n, n_obj=len(goal_names), name=name)
        with optimization_lock, torch.random.fork_rng():
            torch.manual_seed(seed)
            cand, _ = optimize_acqf(
                acqf,
                bounds=bounds_model,
                q=n,
                num_restarts=controls["num_restarts"],
                raw_samples=controls["raw_samples"],
                inequality_constraints=botorch_ineq,
                options=controls["options"],
                sequential=controls["sequential"],
            )
        cand_np = cand.detach().cpu().numpy()
        observed = train_X.detach().cpu().numpy()
        if pending is not None:
            observed = np.vstack([observed, pending.detach().cpu().numpy()])
        lo, hi = _bound_arrays(bounds_model)
        if use_pca_model and pca is not None:
            z_raw = cand_np * np.asarray(pc_range) + np.asarray(pc_mins)
            x_in = _project_rows(pca.inverse_transform(z_raw), params, req)
            z_norm = np.clip((pca.transform(x_in) - np.asarray(pc_mins)) / np.asarray(pc_range), lo, hi)
            z_norm = _dedup_rows(z_norm, observed, bounds_model, n, seed)
            return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
        x_in = _dedup_rows(_project_rows(cand_np, params, req), observed, bounds_model, n, seed)
        return torch.tensor(_project_rows(x_in, params, req), dtype=tdtype, device=tdevice)

    # Advanced: sampling + scoring
    sob = SobolEngine(dimension=bounds_input.shape[1], scramble=True, seed=seed)
    raw_n = 1024
    grid01 = sob.draw(raw_n, dtype=tdtype)
    grid = bounds_input[0] + (bounds_input[1] - bounds_input[0]) * grid01
    # Sobol samples uniformly over bounds and essentially never sum to the
    # declared mixture target. Project the grid onto the sum target first so
    # that the feasibility check further down is not a no-op.
    grid_input_np = _enforce_sum_constraints_np(grid.detach().cpu().numpy(), params, req)
    grid = torch.tensor(grid_input_np, dtype=tdtype, device=tdevice)
    in_envelope = None
    if use_pca_model and pca is not None:
        z_norm = _input_space_to_z_norm(grid_input_np, pca, pc_mins, pc_range)
        # See single-objective counterpart: GP is trained on z_norm in
        # [0,1]^k; track which Sobol samples stay in-envelope so the
        # feasibility mask can prefer them over PC extrapolations.
        lo_m, hi_m = _bound_arrays(bounds_model)
        in_envelope = np.all((z_norm >= lo_m) & (z_norm <= hi_m), axis=1)
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
        # Posterior is in signed training space; soft scores use original Y units.
        mu = _posterior_mean_to_user_units(mu_list[k], goal)
        var = var_list[k]
        # base direction
        if goal in MIN_GOALS:
            score_k = -mu
        elif goal in MAX_GOALS:
            score_k = mu
        elif goal == "within_range":
            score_k = np.zeros_like(mu)
        elif goal == "target" and cfg_o.get("target_value") is not None:
            score_k = _target_soft_score(req, cfg_o, mu, var)
        elif goal in ("explore", "probe"):
            score_k = np.sqrt(var)
        elif goal == "improve":
            y_col = Y_np[:, k] if Y_np.ndim > 1 else np.asarray(Y_np, dtype=float).ravel()
            # Direction from the declared goal list when present; default minimize.
            gk = normalize_goal(goals[k] if k < len(goals) else "min")
            if gk in MAX_GOALS:
                best = float(np.max(y_col))
                score_k = np.maximum(0.0, mu - best)
            else:
                best = float(np.min(y_col))
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
        y_col = Y_np[:, k] if getattr(Y_np, "ndim", 1) > 1 else np.asarray(Y_np, dtype=float).ravel()
        score_k = _standardize_score(score_k, y_col, var, explore=goal != "target")
        score += score_k
    grid_for_feas = grid.detach().cpu().numpy()
    feas_arr = np.array(_feasible_mask(grid_for_feas.tolist(), req, params), dtype=bool)
    if use_pca_model and in_envelope is not None:
        combined = feas_arr & in_envelope
        if combined.any():
            feas_arr = combined
    # Hard-enforce goal feasibility on the GP posterior across all objectives.
    # Helper unsigns mu itself (signed training frame → user units).
    enf_mask = _enforcement_mask_from_posteriors(req, mu_list, var_list)
    if enf_mask is not None:
        combined = feas_arr & enf_mask
        if combined.any():
            feas_arr = combined
    if feas_arr.any():
        score = np.where(feas_arr, score, -np.inf)
    top_idx = np.argsort(score)[-n:][::-1]
    cand_np = grid_for_feas.copy()[top_idx]
    cand_np = _enforce_sum_constraints_np(cand_np, params, req)
    cand_np = _enforce_ratio_constraints_np(cand_np, params, req)
    cand_np = _enforce_sum_constraints_np(cand_np, params, req)
    if use_pca_model and pca is not None:
        z_raw = pca.transform(cand_np)
        z_norm = (z_raw - pc_mins) / pc_range
        lo, hi = _bound_arrays(bounds_model)
        z_norm = np.clip(z_norm, lo, hi)
        return torch.tensor(z_norm, dtype=tdtype, device=tdevice)
    return torch.tensor(cand_np, dtype=tdtype, device=tdevice)
