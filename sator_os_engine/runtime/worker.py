"""Picklable job bodies. Imports stay inside each function so a sleep job does not load Torch."""

from __future__ import annotations

from typing import Any


def run_optimize_payload(payload: dict[str, Any], device: str, cuda_device: int) -> dict[str, Any]:
    from ..core.models.optimize import OptimizeRequest
    from ..core.optimizer.mobo_engine import run_optimization

    req = OptimizeRequest.model_validate(payload)
    return run_optimization(req, device=device, cuda_device=cuda_device)


def run_reconstruct_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from ..core.models.reconstruct import ReconstructionRequest
    from ..reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct

    request = ReconstructionRequest.model_validate(payload)
    coords = request.coordinates
    if request.pca_info and request.pca_info.pc_mins and request.pca_info.pc_maxs:
        pc_mins = np.array(request.pca_info.pc_mins, dtype=float)
        pc_maxs = np.array(request.pca_info.pc_maxs, dtype=float)
        coords_nat = np.array(coords, dtype=float) * (pc_maxs - pc_mins) + pc_mins
    else:
        coords_nat = np.array(coords, dtype=float)

    components = None
    mean = None
    if request.pca_info and request.pca_info.components:
        components = np.array(request.pca_info.components, dtype=float)
        mean = np.array(request.pca_info.mean, dtype=float) if request.pca_info.mean is not None else None

    ingredient_bounds = request.bounds.get("ingredients", []) if isinstance(request.bounds, dict) else []
    parameter_bounds = request.bounds.get("parameters", []) if isinstance(request.bounds, dict) else []
    if components is None:
        return {"success": False, "feasible": False, "on_target": False, "error": "Missing PCA components in pca_info"}

    scale = None
    x_reference = None
    if (
        request.pca_info
        and request.pca_info.scaler_scale
        and request.pca_info.scaler_mean
        and components is not None
        and mean is not None
    ):
        from ..core.optimizer.preprocess import scaled_pca_inverse

        scale = np.array(request.pca_info.scaler_scale, dtype=float)
        x_reference = scaled_pca_inverse(
            coords_nat,
            components,
            mean,
            np.array(request.pca_info.scaler_mean, dtype=float),
            scale,
        )
    variable_bounds = None
    if request.variable_bounds:
        variable_bounds = [(float(a), float(b)) for a, b in request.variable_bounds]
    res = slsqp_reconstruct(
        target_encoded=coords_nat,
        encoder_components=components,
        encoder_mean=mean,
        ingredient_bounds=ingredient_bounds,
        parameter_bounds=parameter_bounds,
        n_ingredients=request.n_ingredients,
        target_precision=request.target_precision,
        sum_target=request.sum_target,
        ratio_constraints=request.ratio_constraints,
        ingredient_names=request.ingredient_names,
        parameter_names=request.parameter_names,
        scaler_scale=scale,
        x_reference=x_reference,
        sum_groups=request.sum_groups,
        variable_bounds=variable_bounds,
        column_names=request.column_names,
    )
    out_form = {
        "ingredients": res["ingredients"],
        "parameters": res["parameters"],
        "combined": res["solution"],
    }
    if "solution_by_name" in res:
        out_form["by_name"] = res["solution_by_name"]
    return {
        "success": res["success"],
        "feasible": res.get("feasible"),
        "on_target": res.get("on_target"),
        "reconstructed_formulation": out_form,
        "reconstruction_metrics": {
            "final_error": res["final_error"],
            "latent_error_rel": res.get("latent_error_rel"),
            "iterations": res["iterations"],
            "method": res["method"],
        },
    }


def run_sleep_payload(seconds: float) -> dict[str, Any]:
    import time

    time.sleep(float(seconds))
    return {"slept": float(seconds)}


def dispatch(kind: str, args: tuple[Any, ...]) -> dict[str, Any]:
    if kind == "optimize":
        return run_optimize_payload(*args)
    if kind == "reconstruct":
        return run_reconstruct_payload(*args)
    if kind == "sleep":
        return run_sleep_payload(*args)
    raise ValueError(f"unknown job kind {kind!r}")
