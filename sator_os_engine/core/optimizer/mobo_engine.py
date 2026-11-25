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


def run_optimization(
    req: OptimizeRequest,
    device: str = "cpu",
    cuda_device: int = 0,
    *,
    explain_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    rng_seed = resolve_seed(req.optimization_config.seed)

    # Try BoTorch path if dataset provided; otherwise fallback
    dataset = req.dataset or {}
    X = dataset.get("X")
    Y = dataset.get("Y")

    if X is not None and Y is not None:
        try:
            with optimization_lock, torch.random.fork_rng():
                torch.manual_seed(rng_seed)
                return _run_optimization_locked(req, device, cuda_device, rng_seed, explain_rows=explain_rows)
        except Exception as e:
            # Avoid embedding tracebacks in the error string (leaks paths; job result may be logged).
            raise RuntimeError(f"Optimization failed: {e}") from e

    # If no dataset was provided, fail explicitly
    raise ValueError("OptimizeRequest must include dataset.X and dataset.Y")


def _run_optimization_locked(
    req: OptimizeRequest,
    device: str,
    cuda_device: int,
    rng_seed: int,
    *,
    explain_rows: np.ndarray | None = None,
) -> dict[str, Any]:
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
    from .observations import prepare_observations

    yvar_raw = dataset.get("Yvar") if isinstance(dataset, dict) else None
    failed_raw = dataset.get("failed") if isinstance(dataset, dict) else None
    prepared = prepare_observations(X_np, Y_np, yvar_raw, failed_raw)
    X_np = prepared["X"]
    Y_np = prepared["Y"]
    y_finite = np.isfinite(Y_np)
    x_finite = np.isfinite(X_np).all(axis=1)
    dropped_rows = int(np.sum(~y_finite.all(axis=1) | ~x_finite))
    k_req = req.optimization_config.pca_dimension
    use_pca_model = bool(req.optimization_config.use_pca and k_req is not None)
    mixture = str(getattr(cfg, "mixture_transform", None) or "none").strip().lower()
    use_ilr = mixture == "ilr"
    obj_cfgs = list(req.objectives.values()) if isinstance(req.objectives, dict) else []
    obj_names = list(req.objectives.keys()) if isinstance(req.objectives, dict) else []
    goals = [str(item.get("goal", "min")).lower() for item in obj_cfgs] if obj_cfgs else ["min"] * Y_np.shape[1]
    encoder = None
    pca = None
    pc_mins = pc_maxs = pc_range = None
    Z = None
    if use_ilr:
        from .mixture import MixtureEncoder
        from .utils import PLAIN_DIRECTIONAL_GOALS, THRESHOLD_GOALS

        allowed = PLAIN_DIRECTIONAL_GOALS | THRESHOLD_GOALS
        if any(goal not in allowed for goal in goals):
            raise ValueError("mixture_transform 'ilr' supports min/max and threshold goals")
        encoder = MixtureEncoder(sum_groups_from_config(req, X_np.shape[1]), X_np.shape[1])
        X_model = encoder.forward(X_np)
    elif use_pca_model:
        pca, pc_mins, pc_maxs, pc_range, Z = fit_pca_normalize(X_np, k_req)
        X_model = Z
    else:
        X_model = X_np

    tX = torch.tensor(X_model, dtype=tdtype, device=tdevice)
    signs = np.array([training_sign(g) for g in goals], dtype=float)
    tY = torch.tensor(np.where(np.isfinite(Y_np), Y_np, 0.0) * signs, dtype=tdtype, device=tdevice)
    n_obj = tY.shape[-1]
    row_masks = [torch.tensor(y_finite[:, i] & x_finite, dtype=torch.bool) for i in range(n_obj)]

    from .gp import bounds_input as gp_bounds_input
    from .gp import bounds_model_pca, build_models

    bounds_input = gp_bounds_input(params, tdtype, tdevice)
    margin = float(getattr(cfg, "pca_margin", 0.0) or 0.0)
    if use_ilr and encoder is not None:
        box = encoder.bounds(X_model)
        bounds_model = torch.tensor(box, dtype=tdtype, device=tdevice)
        model_bounds = bounds_model
    elif use_pca_model and pca is not None:
        bounds_model = bounds_model_pca(X_model.shape[1], tdtype, tdevice, margin=margin)
        model_bounds = bounds_model
    else:
        bounds_model = bounds_input
        model_bounds = bounds_input
    from .acquisition import _categorical_fixed_features

    cat_dims = [i for i, param in enumerate(params) if param.get("type") == "categorical"]
    use_mixed = bool(cat_dims) and not use_pca_model and not use_ilr
    if use_mixed and _categorical_fixed_features(params, len(params)) is None:
        use_mixed = False
    yvar_t = None
    if prepared["Yvar"] is not None:
        yvar_t = torch.tensor(prepared["Yvar"], dtype=tdtype, device=tdevice)
    feas_t = None
    if prepared["feasibility"] is not None:
        feas_t = torch.tensor(prepared["feasibility"], dtype=tdtype, device=tdevice)
    model = build_models(
        tX,
        tY,
        cfg,
        bounds=model_bounds,
        row_masks=row_masks,
        train_Yvar=yvar_t,
        feasibility=feas_t,
        cat_dims=cat_dims if use_mixed else None,
    )
    pca_in_x = bool(use_pca_model and pca is not None)
    acq_model = model
    acq_bounds = bounds_model
    acq_train = tX
    acq_use_pca = use_pca_model
    acq_pca = pca
    if pca_in_x:
        from botorch.models import ModelListGP

        from .pca_model import (
            PcaObservationModel,
            latent_box_inequalities,
            off_subspace_residual,
            pull_toward_subspace,
        )

        components_t = torch.tensor(np.asarray(pca.components_), dtype=tdtype, device=tdevice)
        mean_t = torch.tensor(np.asarray(pca.mean_), dtype=tdtype, device=tdevice)
        pc_mins_t = torch.tensor(np.asarray(pc_mins), dtype=tdtype, device=tdevice)
        pc_range_t = torch.tensor(np.asarray(pc_range), dtype=tdtype, device=tdevice)
        observed = [PcaObservationModel(sub, components_t, mean_t, pc_mins_t, pc_range_t) for sub in model.models]
        # A single output stays a plain model so qLogEI can call posterior
        # directly. Several outputs need a list so each posterior is evaluated
        # at the same x and then encoded.
        acq_model = observed[0] if len(observed) == 1 else ModelListGP(*observed)
        acq_bounds = bounds_input
        acq_train = torch.tensor(X_np, dtype=tdtype, device=tdevice)
        acq_use_pca = False
        acq_pca = None
        score_box = latent_box_inequalities(
            np.asarray(pca.components_),
            np.asarray(pca.mean_),
            np.asarray(pc_mins),
            np.asarray(pc_range),
            margin,
            dtype=tdtype,
            device=tdevice,
        )
        trust = getattr(cfg, "pca_trust", "training")
        scale = getattr(pca, "scaler_scale_", None)
        scaler_center = getattr(pca, "scaler_mean_", None)
        if isinstance(trust, str) and trust.strip().lower() == "none":
            subspace_cap = None
        elif not isinstance(trust, str):
            subspace_cap = max(float(trust), 0.0)
        elif scale is None or scaler_center is None:
            subspace_cap = None
        else:
            trained_residual = off_subspace_residual(
                X_np,
                np.asarray(pca.components_),
                np.asarray(pca.mean_),
                np.asarray(scaler_center),
                np.asarray(scale),
            )
            subspace_cap = max(float(np.max(trained_residual)), 1e-8)
    else:
        score_box = None
        subspace_cap = None

    pending_raw = dataset.get("pending_X", dataset.get("X_pending")) if isinstance(dataset, dict) else None
    x_pending = None
    if pending_raw:
        pend = encode_dataset(pending_raw, params)
        pend, _ = clip_to_bounds(pend, params)
        if use_ilr and encoder is not None:
            pend = encoder.forward(pend)
        elif use_pca_model and pca is not None and not pca_in_x:
            pend = input_to_z_norm(pca, pc_mins, pc_range, pend)
        x_pending = torch.tensor(pend, dtype=tdtype, device=tdevice)

    from .acquisition import select_candidates_multiobjective, select_candidates_single_objective

    route_name = str(getattr(cfg, "route", "direct") or "direct").strip().lower()
    trust_region_state = None
    if route_name == "trust_region":
        from .trust_region import replay_trust_region

        gp_cfg = dict(getattr(cfg, "gp_config", None) or {})
        tr_kwargs = {
            arg: gp_cfg[key]
            for key, arg in (
                ("tr_length_init", "length_init"),
                ("tr_length_min", "length_min"),
                ("tr_length_max", "length_max"),
                ("tr_success_tolerance", "success_tolerance"),
                ("tr_failure_tolerance", "failure_tolerance"),
            )
            if gp_cfg.get(key) is not None
        }
        trust_region_state = replay_trust_region(X_np, Y_np, _parameter_bounds(params), signs, **tr_kwargs)
        box = torch.tensor(trust_region_state["bounds"], dtype=tdtype, device=tdevice).T.contiguous()
        acq_bounds = box

    scoring_supplied = explain_rows is not None
    if scoring_supplied:
        rows = np.asarray(explain_rows, dtype=float)
        if use_ilr and encoder is not None:
            rows = encoder.forward(rows)
        elif use_pca_model and pca is not None and not pca_in_x:
            rows = input_to_z_norm(pca, pc_mins, pc_range, rows)
        cand = torch.tensor(np.atleast_2d(rows), dtype=tdtype, device=tdevice)
        num = int(cand.shape[0])
    elif route_name == "trust_region" and n_obj == 1 and num > 8:
        from .trust_region import thompson_candidates

        cand = thompson_candidates(acq_model, acq_bounds, num, int(rng_seed))
    elif n_obj == 1:
        cand = select_candidates_single_objective(
            model=acq_model,
            params=params,
            bounds_input=bounds_input,
            bounds_model=acq_bounds,
            use_pca_model=acq_use_pca,
            pca=acq_pca,
            pc_mins=pc_mins if acq_use_pca and acq_pca is not None else None,
            pc_range=pc_range if acq_use_pca and acq_pca is not None else None,
            n=num,
            rng_seed=rng_seed,
            tdtype=tdtype,
            tdevice=tdevice,
            req=req,
            Y_np=Y_np,
            train_X=acq_train,
            X_pending=x_pending,
            return_model_space=use_ilr,
            optional_ineq=score_box,
        )
    else:
        cand = select_candidates_multiobjective(
            model=acq_model,
            params=params,
            bounds_input=bounds_input,
            bounds_model=acq_bounds,
            use_pca_model=acq_use_pca,
            pca=acq_pca,
            pc_mins=pc_mins if acq_use_pca and acq_pca is not None else None,
            pc_range=pc_range if acq_use_pca and acq_pca is not None else None,
            n=num,
            rng_seed=rng_seed,
            tdtype=tdtype,
            tdevice=tdevice,
            req=req,
            goals=goals,
            Y_np=Y_np,
            train_X=acq_train,
            X_pending=x_pending,
            return_model_space=use_ilr,
            optional_ineq=score_box,
        )

    preds: list[dict[str, Any]] = []
    explained_x: list[np.ndarray] = []
    from .cardinality import apply_cardinality

    search_bounds = _parameter_bounds(params)
    groups = sum_groups_from_config(req, len(params))
    ratios = ratio_constraints_from_config(req)
    rules = getattr(cfg, "ingredient_rules", None) or {}
    max_nonzero = None if rules.get("max_nonzero") is None else int(rules["max_nonzero"])
    min_dose = None if rules.get("min_dose") is None else float(rules["min_dose"])
    supports = groups
    if (max_nonzero is not None or min_dose is not None) and not groups:
        ingredient_idx = [
            i
            for i, param in enumerate(params)
            if param.get("type", "float") == "float" and float(param.get("min", 0.0)) <= 0.0
        ]
        supports = [{"indices": ingredient_idx}] if ingredient_idx else []

    def _original_recipe(raw: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        recipe, ok = apply_cardinality(
            raw,
            search_bounds,
            groups,
            ratios,
            max_nonzero=max_nonzero,
            min_dose=min_dose,
            supports=None if supports is groups else supports,
        )
        recipe = snap_numeric(recipe, params)
        return recipe, ok, candidate_by_name(recipe, params)

    for i in range(cand.shape[0]):
        requested = cand[i : i + 1]
        reconstructed = None
        encoded = None
        encoded_actual = None
        residual = None
        violations: list[str] | None = None
        if scoring_supplied:
            from .project import feasibility_violations

            x_vec = np.asarray(explain_rows[i], dtype=float).reshape(-1)
            violations = feasibility_violations(x_vec, search_bounds, groups, ratios)
            candidate_feasible = not violations
            cand_dict = {}
            for j, param in enumerate(params):
                name = str(param["name"])
                kind = param.get("type", "float")
                if kind == "categorical":
                    choices = list(param.get("choices") or [])
                    level = int(round(float(x_vec[j])))
                    cand_dict[name] = choices[level] if 0 <= level < len(choices) else float(x_vec[j])
                elif kind == "int":
                    cand_dict[name] = int(round(float(x_vec[j])))
                else:
                    cand_dict[name] = float(x_vec[j])
            if pca_in_x:
                latent_model = acq_model.models[0] if hasattr(acq_model, "models") else acq_model
                z_actual = (
                    latent_model.encode(torch.tensor(x_vec.reshape(1, -1), dtype=tdtype, device=tdevice))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                encoded = z_actual.tolist()
                encoded_actual = z_actual.tolist()
                x_model = torch.tensor(z_actual.reshape(1, -1), dtype=tdtype, device=tdevice)
                residual = float(
                    off_subspace_residual(
                        x_vec,
                        np.asarray(pca.components_),
                        np.asarray(pca.mean_),
                        np.asarray(pca.scaler_mean_),
                        np.asarray(pca.scaler_scale_),
                    )[0]
                )
            elif use_ilr and encoder is not None:
                z_actual = encoder.forward(x_vec).reshape(-1)
                encoded = z_actual.tolist()
                encoded_actual = z_actual.tolist()
                x_model = torch.tensor(z_actual.reshape(1, -1), dtype=tdtype, device=tdevice)
            elif use_pca_model and pca is not None:
                from .preprocess import input_to_z_norm as _to_z

                z_actual = _to_z(pca, pc_mins, pc_range, x_vec.reshape(1, -1)).reshape(-1)
                encoded = z_actual.tolist()
                encoded_actual = z_actual.tolist()
                x_model = torch.tensor(np.asarray(z_actual, dtype=float).reshape(1, -1), dtype=tdtype, device=tdevice)
            else:
                x_model = torch.tensor(x_vec.reshape(1, -1), dtype=tdtype, device=tdevice)
        elif pca_in_x:
            raw = requested.detach().cpu().numpy().reshape(-1)
            x_vec, candidate_feasible, cand_dict = _original_recipe(raw)
            if subspace_cap is not None:
                residual = float(
                    off_subspace_residual(
                        x_vec,
                        np.asarray(pca.components_),
                        np.asarray(pca.mean_),
                        np.asarray(pca.scaler_mean_),
                        np.asarray(pca.scaler_scale_),
                    )[0]
                )
                if residual > subspace_cap * (1.0 + 1e-4) + 1e-8:
                    pulled = pull_toward_subspace(
                        x_vec,
                        np.asarray(pca.components_),
                        np.asarray(pca.mean_),
                        np.asarray(pca.scaler_mean_),
                        np.asarray(pca.scaler_scale_),
                        subspace_cap,
                    ).reshape(-1)
                    x_vec, candidate_feasible, cand_dict = _original_recipe(pulled)
                    residual = float(
                        off_subspace_residual(
                            x_vec,
                            np.asarray(pca.components_),
                            np.asarray(pca.mean_),
                            np.asarray(pca.scaler_mean_),
                            np.asarray(pca.scaler_scale_),
                        )[0]
                    )
            latent_model = acq_model.models[0] if hasattr(acq_model, "models") else acq_model
            z_actual = (
                latent_model.encode(torch.tensor(x_vec.reshape(1, -1), dtype=tdtype, device=tdevice))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            encoded = z_actual.tolist()
            encoded_actual = z_actual.tolist()
            x_model = torch.tensor(z_actual.reshape(1, -1), dtype=tdtype, device=tdevice)
            reconstructed = {
                "combined": x_vec.tolist(),
                "by_name": cand_dict,
                "success": bool(candidate_feasible),
                "feasible": bool(candidate_feasible),
                "on_target": True,
                "final_error": 0.0,
                "latent_error_rel": 0.0,
            }
        elif use_pca_model and pca is not None:
            z_requested = requested.detach().cpu().numpy().reshape(-1)
            x_vec, reconstructed, _z_actual = _decode_latent(z_requested, pca, pc_mins, pc_range, params, req)
            x_vec, candidate_feasible, cand_dict = _original_recipe(x_vec)
            from .preprocess import input_to_z_norm as _to_z

            z_actual = _to_z(pca, pc_mins, pc_range, x_vec.reshape(1, -1)).reshape(-1)
            encoded = z_requested.tolist()
            encoded_actual = np.asarray(z_actual, dtype=float).reshape(-1).tolist()
            x_model = torch.tensor(np.asarray(z_actual, dtype=float).reshape(1, -1), dtype=tdtype, device=tdevice)
            if reconstructed is not None:
                reconstructed = dict(reconstructed)
                reconstructed["combined"] = x_vec.tolist()
                reconstructed["by_name"] = cand_dict
                reconstructed["feasible"] = bool(candidate_feasible)
        elif use_ilr and encoder is not None:
            z_requested = requested.detach().cpu().numpy().reshape(-1)
            x_vec, candidate_feasible, cand_dict = _original_recipe(encoder.inverse(z_requested).reshape(-1))
            z_actual = encoder.forward(x_vec).reshape(-1)
            encoded = z_requested.tolist()
            encoded_actual = z_actual.tolist()
            x_model = torch.tensor(z_actual.reshape(1, -1), dtype=tdtype, device=tdevice)
        else:
            x_vec, candidate_feasible, cand_dict = _original_recipe(requested.detach().cpu().numpy().reshape(-1))
            x_model = torch.tensor(x_vec.reshape(1, -1), dtype=tdtype, device=tdevice)
        means = []
        variances = []
        for k, m in enumerate(model.models):
            if k >= n_obj:
                break
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
        if residual is not None:
            pred_item["off_subspace_residual"] = residual
        if violations is not None:
            pred_item["supplied_feasible"] = bool(candidate_feasible)
            pred_item["violations"] = violations
        if encoded is not None:
            pred_item["encoded"] = encoded
            pred_item["encoded_actual"] = encoded_actual
        if reconstructed is not None:
            pred_item["reconstructed"] = reconstructed
        preds.append(pred_item)
        explained_x.append(np.asarray(x_vec, dtype=float).reshape(-1))

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

    from .explain import explain_prediction, parameter_sensitivity

    feature_names = [str(param["name"]) for param in params]
    if use_pca_model and pca is not None:
        sensitivity_space = "pca"
    elif use_ilr:
        sensitivity_space = "ilr"
    else:
        sensitivity_space = "input"
    sensitivities = parameter_sensitivity(
        model,
        feature_names,
        n_obj=n_obj,
        space=sensitivity_space,
        components=None if pca is None else np.asarray(pca.components_, dtype=float),
        encoder=encoder,
    )
    sensitive_names = [item["parameter"] for item in sensitivities.get("parameters", [])]
    finite_train = np.isfinite(X_np).all(axis=1) & np.isfinite(Y_np).all(axis=1)
    train_index = np.flatnonzero(finite_train)
    for pred, recipe in zip(preds, explained_x, strict=True):
        pred["explanation"] = explain_prediction(
            means=pred["objectives"],
            variances=pred["variances"],
            recipe=recipe,
            trained_x=X_np[train_index],
            trained_y=Y_np[train_index],
            bounds=search_bounds,
            objective_names=obj_names,
            goals=goals,
            sensitive=sensitive_names,
            row_index=train_index,
        )

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

    from .acquisition import resolve_acquisition_name

    sator_version, botorch_version = _component_versions()
    diagnostics: dict[str, Any] = {
        "device": str(tdevice),
        "cuda_device_index": cuda_idx,
        "seed": int(rng_seed),
        "batch_size": num,
        "mode": str(getattr(cfg, "mode", "lab")),
        "acquisition": "explain" if scoring_supplied else resolve_acquisition_name(req, n_obj, tX),
        "sensitivities": sensitivities,
        "dropped_rows": dropped_rows,
        "rows_collapsed": int(prepared["collapsed"]),
        "fixed_noise": bool(prepared["fixed_noise"]),
        "feasibility_model": prepared["feasibility"] is not None,
        "mixed_model": bool(use_mixed),
        "mixture_transform": "ilr" if use_ilr else "none",
        "model": str(getattr(cfg, "model", "gp") or "gp").strip().lower(),
        "route": str(getattr(cfg, "route", "direct") or "direct").strip().lower(),
        "pca_search": "x" if pca_in_x else "none",
        "bounds_clipped": int(n_clipped),
        "objective_names": obj_names,
        "sator_version": sator_version,
        "botorch_version": botorch_version,
    }
    if diagnostics["model"] == "kpls":
        kpls_report: dict[str, Any] = {}
        for name, submodel in zip(obj_names, model.models, strict=False):
            weights = getattr(submodel, "kpls_weights", None)
            if weights is None:
                continue
            magnitude = np.abs(weights.detach().cpu().numpy()).sum(axis=1)
            order = np.argsort(-magnitude)
            kpls_report[name] = {
                "k": int(weights.shape[-1]),
                "parameters": [feature_names[int(i)] for i in order],
            }
        diagnostics["kpls"] = kpls_report
    if trust_region_state is not None:
        diagnostics["trust_region"] = trust_region_state
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
