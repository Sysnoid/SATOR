from __future__ import annotations

from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import torch

from ..models.optimize import OptimizeRequest
from .device import resolve_torch_device
from .maps import compute_gp_maps
from .project import project_to_feasible
from .space import candidate_by_name, clip_to_bounds, encode_dataset, snap_numeric
from .utils import (
    evaluate_enforced_goals as _evaluate_enforced_goals,
)
from .utils import (
    extract_enforced_goal_specs as _extract_enforced_goal_specs,
)
from .utils import (
    optimization_lock,
    ratio_constraints_from_config,
    resolve_seed,
    sum_groups_from_config,
    training_sign,
)
from .utils import (
    pareto_front as _pareto_front,
)

## moved to utils.sample_candidates


## moved to utils.dummy_objective


## moved to utils.pareto_front


## moved to utils.build_linear_constraints


## moved to utils.feasible_mask


## moved to utils.enforce_sum_constraints_np


def _component_versions() -> tuple[str, str]:
    import botorch

    try:
        sator_version = version("sator-os-engine")
    except PackageNotFoundError:
        sator_version = "0.2.1"
    return sator_version, botorch.__version__


def _parameter_bounds(params: list[dict[str, Any]]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for param in params:
        if param.get("type") == "categorical":
            n_choices = max(len(param.get("choices") or []), 1)
            bounds.append((0.0, float(n_choices - 1)))
        else:
            bounds.append((float(param["min"]), float(param["max"])))
    return bounds


def _decode_latent(
    z_norm_row: np.ndarray,
    pca,
    pc_mins: np.ndarray,
    pc_range: np.ndarray,
    params: list[dict[str, Any]],
    req: OptimizeRequest,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Project a latent point to a feasible recipe and report the latent error."""
    from ...reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct
    from .preprocess import input_to_z_norm

    z_norm = np.asarray(z_norm_row, dtype=float).reshape(1, -1)
    z_raw = (z_norm * np.asarray(pc_range) + np.asarray(pc_mins)).reshape(-1)
    x_hat = np.asarray(pca.inverse_transform(z_raw.reshape(1, -1)), dtype=float).reshape(-1)
    bounds = _parameter_bounds(params)
    groups = sum_groups_from_config(req, len(params))
    ratios = ratio_constraints_from_config(req)
    rec = slsqp_reconstruct(
        target_encoded=z_raw,
        encoder_components=np.asarray(pca.components_),
        encoder_mean=np.asarray(pca.mean_),
        ingredient_bounds=[],
        parameter_bounds=[],
        n_ingredients=0,
        target_precision=1e-3,
        sum_groups=groups,
        variable_bounds=bounds,
        column_names=[str(p["name"]) for p in params],
        x_reference=x_hat,
        scaler_scale=getattr(pca, "scaler_scale_", None),
        ratio_constraints=ratios or None,
    )
    x = snap_numeric(np.asarray(rec["solution"], dtype=float), params)
    x, feasible = project_to_feasible(x, bounds, groups, ratios)
    x = snap_numeric(x, params)
    enc = (x - np.asarray(pca.mean_, dtype=float)) @ np.asarray(pca.components_, dtype=float).T
    err = float(np.linalg.norm(enc - z_raw))
    z_scale = float(np.linalg.norm(z_raw))
    on_target = err <= 1e-3 * (1.0 + z_scale)
    block = {
        "ingredients": rec.get("ingredients", []),
        "parameters": rec.get("parameters", []),
        "combined": x.tolist(),
        "success": bool(feasible and on_target),
        "feasible": bool(feasible),
        "on_target": bool(on_target),
        "final_error": err,
        "latent_error_rel": err / (1.0 + z_scale),
        "by_name": candidate_by_name(x, params),
    }
    z_actual = input_to_z_norm(pca, pc_mins, pc_range, x.reshape(1, -1)).reshape(-1)
    return x, block, z_actual


def run_optimization(req: OptimizeRequest, device: str = "cpu", cuda_device: int = 0) -> dict[str, Any]:
    rng_seed = resolve_seed(req.optimization_config.seed)

    # Try BoTorch path if dataset provided; otherwise fallback
    dataset = req.dataset or {}
    X = dataset.get("X")
    Y = dataset.get("Y")

    if X is not None and Y is not None:
        try:
            with optimization_lock, torch.random.fork_rng():
                torch.manual_seed(rng_seed)
                return _run_optimization_locked(req, device, cuda_device, rng_seed)
        except Exception as e:
            # Avoid embedding tracebacks in the error string (leaks paths; job result may be logged).
            raise RuntimeError(f"Optimization failed: {e}") from e

    # If no dataset was provided, fail explicitly
    raise ValueError("OptimizeRequest must include dataset.X and dataset.Y")


def _run_optimization_locked(req: OptimizeRequest, device: str, cuda_device: int, rng_seed: int) -> dict[str, Any]:
    cfg = req.optimization_config
    num = int(cfg.batch_size)
    dataset = req.dataset or {}
    X = dataset.get("X")
    Y = dataset.get("Y")
    tdevice, cuda_idx = resolve_torch_device(device, cuda_device)
    tdtype = torch.double

    params = req.search_space.get("parameters", [])
    X_np = encode_dataset(X, params)
    X_np, n_clipped = clip_to_bounds(X_np, params)
    Y_np = np.asarray(Y, dtype=float)
    if Y_np.ndim == 1:
        Y_np = Y_np.reshape(-1, 1)
    sums_cfg = (
        (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    )
    from .preprocess import enforce_sum_to_target_training, fit_pca_normalize, input_to_z_norm

    X_np = enforce_sum_to_target_training(X_np, sums_cfg)
    y_finite = np.isfinite(Y_np)
    x_finite = np.isfinite(X_np).all(axis=1)
    dropped_rows = int(np.sum(~y_finite.all(axis=1) | ~x_finite))
    k_req = req.optimization_config.pca_dimension
    use_pca_model = bool(req.optimization_config.use_pca and k_req is not None)
    pca = None
    pc_mins = pc_maxs = pc_range = None
    Z = None
    if use_pca_model:
        pca, pc_mins, pc_maxs, pc_range, Z = fit_pca_normalize(X_np, k_req)
        X_model = Z
    else:
        X_model = X_np

    tX = torch.tensor(X_model, dtype=tdtype, device=tdevice)
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    obj_names = list(req.objectives.keys()) if isinstance(req.objectives, dict) else []
    goals = [str(item.get("goal", "min")).lower() for item in obj_cfgs] if obj_cfgs else ["min"] * Y_np.shape[1]
    signs = np.array([training_sign(g) for g in goals], dtype=float)
    tY = torch.tensor(np.where(np.isfinite(Y_np), Y_np, 0.0) * signs, dtype=tdtype, device=tdevice)
    n_obj = tY.shape[-1]
    row_masks = [torch.tensor(y_finite[:, i] & x_finite, dtype=torch.bool) for i in range(n_obj)]

    from .gp import bounds_input as gp_bounds_input
    from .gp import bounds_model_pca, build_models

    bounds_input = gp_bounds_input(params, tdtype, tdevice)
    margin = float(getattr(cfg, "pca_margin", 0.0) or 0.0)
    if use_pca_model and pca is not None:
        bounds_model = bounds_model_pca(X_model.shape[1], tdtype, tdevice, margin=margin)
        model_bounds = bounds_model
    else:
        bounds_model = bounds_input
        model_bounds = bounds_input
    model = build_models(tX, tY, cfg, bounds=model_bounds, row_masks=row_masks)

    pending_raw = dataset.get("pending_X", dataset.get("X_pending")) if isinstance(dataset, dict) else None
    x_pending = None
    if pending_raw:
        pend = encode_dataset(pending_raw, params)
        pend, _ = clip_to_bounds(pend, params)
        if use_pca_model and pca is not None:
            pend = input_to_z_norm(pca, pc_mins, pc_range, pend)
        x_pending = torch.tensor(pend, dtype=tdtype, device=tdevice)

    from .acquisition import select_candidates_multiobjective, select_candidates_single_objective

    if n_obj == 1:
        cand = select_candidates_single_objective(
            model=model,
            params=params,
            bounds_input=bounds_input,
            bounds_model=bounds_model,
            use_pca_model=use_pca_model,
            pca=pca,
            pc_mins=pc_mins if use_pca_model and pca is not None else None,
            pc_range=pc_range if use_pca_model and pca is not None else None,
            n=num,
            rng_seed=rng_seed,
            tdtype=tdtype,
            tdevice=tdevice,
            req=req,
            Y_np=Y_np,
            train_X=tX,
            X_pending=x_pending,
        )
    else:
        cand = select_candidates_multiobjective(
            model=model,
            params=params,
            bounds_input=bounds_input,
            bounds_model=bounds_model,
            use_pca_model=use_pca_model,
            pca=pca,
            pc_mins=pc_mins if use_pca_model and pca is not None else None,
            pc_range=pc_range if use_pca_model and pca is not None else None,
            n=num,
            rng_seed=rng_seed,
            tdtype=tdtype,
            tdevice=tdevice,
            req=req,
            goals=goals,
            Y_np=Y_np,
            train_X=tX,
            X_pending=x_pending,
        )

    preds: list[dict[str, Any]] = []
    search_bounds = _parameter_bounds(params)
    groups = sum_groups_from_config(req, len(params))
    ratios = ratio_constraints_from_config(req)
    for i in range(cand.shape[0]):
        requested = cand[i : i + 1]
        candidate_feasible = True
        reconstructed = None
        encoded = None
        encoded_actual = None
        if use_pca_model and pca is not None:
            z_requested = requested.detach().cpu().numpy().reshape(-1)
            x_vec, reconstructed, z_actual = _decode_latent(z_requested, pca, pc_mins, pc_range, params, req)
            candidate_feasible = bool(reconstructed.get("feasible"))
            encoded = z_requested.tolist()
            encoded_actual = np.asarray(z_actual, dtype=float).reshape(-1).tolist()
            x_model = torch.tensor(np.asarray(z_actual, dtype=float).reshape(1, -1), dtype=tdtype, device=tdevice)
            cand_dict = dict(reconstructed["by_name"])
        else:
            x_vec = snap_numeric(requested.detach().cpu().numpy().reshape(-1), params)
            x_vec, candidate_feasible = project_to_feasible(x_vec, search_bounds, groups, ratios)
            x_vec = snap_numeric(x_vec, params)
            x_model = torch.tensor(x_vec.reshape(1, -1), dtype=tdtype, device=tdevice)
            cand_dict = candidate_by_name(x_vec, params)
        means = []
        variances = []
        for k, m in enumerate(model.models):
            post = m.posterior(x_model)
            mu_t = float(post.mean.detach().cpu().numpy().ravel()[0])
            vr = float(post.variance.detach().cpu().numpy().ravel()[0])
            means.append(float(mu_t * signs[k]))
            variances.append(vr)
        pred_item: dict[str, Any] = {
            "candidate": cand_dict,
            "objectives": means,
            "variances": variances,
            "candidate_feasible": bool(candidate_feasible),
        }
        if encoded is not None:
            pred_item["encoded"] = encoded
            pred_item["encoded_actual"] = encoded_actual
        if reconstructed is not None:
            pred_item["reconstructed"] = reconstructed
        preds.append(pred_item)

    # Hard-constraint goal validation (enforce_above / enforce_below /
    # enforce_within_range). Runs after candidate selection so that
    # every returned prediction carries an honest feasibility flag
    # even when the selector was forced to relax the enforcement mask.
    enf_specs = _extract_enforced_goal_specs(req)
    enforcement_diag: dict[str, Any] | None = None
    if enf_specs and preds:
        mu_mat = np.asarray([p.get("objectives", []) for p in preds], dtype=float)
        var_mat = np.asarray([p.get("variances", []) for p in preds], dtype=float)
        enf_margin = float(getattr(cfg, "enforcement_uncertainty_margin", 0.0) or 0.0)
        mask, viol_rows = _evaluate_enforced_goals(enf_specs, mu_mat, var_mat, enf_margin)
        per_obj_violations: dict[str, int] = {s["name"]: 0 for s in enf_specs}
        for i, p in enumerate(preds):
            row_ok = bool(mask[i])
            p["enforced_goals_satisfied"] = row_ok
            p["enforced_violations"] = list(viol_rows[i])
            for lbl in viol_rows[i]:
                obj_name = lbl.split("<")[0].split(">")[0]
                if obj_name in per_obj_violations:
                    per_obj_violations[obj_name] += 1
        enforcement_diag = {
            "enabled": True,
            "uncertainty_margin": enf_margin,
            "n_total": len(preds),
            "n_satisfied": int(np.sum(mask)),
            "all_infeasible": bool(not mask.any()),
            "per_objective_violations": per_obj_violations,
            "goals": [{"objective": s["name"], "kind": s["kind"], "lo": s["lo"], "hi": s["hi"]} for s in enf_specs],
        }

    pareto_idx = _pareto_front([p["objectives"] for p in preds], minimize_frame=-signs)
    pareto = {"indices": pareto_idx, "points": [preds[i]["objectives"] for i in pareto_idx]}

    encoding_info = None
    if use_pca_model and pca is not None:
        with suppress(Exception):
            info: dict[str, Any] = {
                "pc_mins": pc_mins.tolist(),
                "pc_maxs": pc_maxs.tolist(),
                "components": pca.components_.tolist(),
                "mean": pca.mean_.tolist(),
                "k_requested": getattr(pca, "k_requested_", k_req),
                "k_effective": getattr(pca, "k_effective_", int(X_model.shape[1])),
                "k_dropped": int(getattr(pca, "k_dropped_", 0)),
                "pca_margin": margin,
            }
            ev_ratio = getattr(pca, "explained_variance_ratio_", None)
            if ev_ratio is not None:
                info["explained_variance_ratio"] = np.asarray(ev_ratio).tolist()
            scaler_mean = getattr(pca, "scaler_mean_", None)
            scaler_scale = getattr(pca, "scaler_scale_", None)
            if scaler_mean is not None and scaler_scale is not None:
                info["scaler_mean"] = np.asarray(scaler_mean).tolist()
                info["scaler_scale"] = np.asarray(scaler_scale).tolist()
            encoding_info = info

    sator_version, botorch_version = _component_versions()
    diagnostics: dict[str, Any] = {
        "device": str(tdevice),
        "cuda_device_index": cuda_idx,
        "seed": int(rng_seed),
        "batch_size": num,
        "dropped_rows": dropped_rows,
        "bounds_clipped": int(n_clipped),
        "objective_names": obj_names,
        "sator_version": sator_version,
        "botorch_version": botorch_version,
    }
    if enforcement_diag is not None:
        diagnostics["enforcement"] = enforcement_diag
    response: dict[str, Any] = {
        "predictions": preds,
        "pareto": pareto,
        "encoding_info": encoding_info,
        "diagnostics": diagnostics,
        "objectives": req.objectives if isinstance(req.objectives, dict) else {},
        "encoded_dataset": Z.tolist() if (use_pca_model and pca is not None) else None,
    }

    if cfg.return_maps:
        try:
            gp_maps = compute_gp_maps(
                model=model,
                cfg=cfg,
                req=req,
                params=params,
                use_pca_model=use_pca_model,
                pca=pca,
                Z=Z if use_pca_model and pca is not None else None,
                X=X_np,
                tdtype=tdtype,
                tdevice=tdevice,
                signs=signs,
                pc_mins=pc_mins if (use_pca_model and pca is not None) else None,
                pc_range=pc_range if (use_pca_model and pca is not None) else None,
            )
            if gp_maps:
                response["gp_maps"] = gp_maps
        except Exception as e:
            raise RuntimeError(f"failed to compute gp maps: {e}") from e

    return response
