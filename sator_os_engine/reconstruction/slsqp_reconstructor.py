from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from ..core.optimizer.project import is_feasible, project_to_feasible


def _sum_constraint_factory(indices: list[int], target_sum: float):
    def sum_constraint(x: np.ndarray) -> float:
        if not indices:
            return 0.0
        return float(np.sum(x[indices]) - target_sum)

    return sum_constraint


def _combine_bounds(
    ingredient_bounds: list[list[float]], parameter_bounds: list[list[float]]
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    bounds.extend([(float(a), float(b)) for a, b in ingredient_bounds])
    bounds.extend([(float(a), float(b)) for a, b in parameter_bounds])
    return bounds


def _legacy_groups(n_ingredients: int, sum_target: float, dim: int) -> list[dict[str, Any]]:
    if n_ingredients <= 0:
        return []
    idxs = list(range(min(n_ingredients, dim)))
    return [{"indices": idxs, "target": float(sum_target)}]


def _split_solution(
    solution: np.ndarray,
    sum_groups: list[dict[str, Any]],
    n_ingredients: int,
) -> tuple[list[float], list[float]]:
    dim = int(solution.shape[0])
    if sum_groups:
        ing: list[int] = []
        for group in sum_groups:
            for i in group.get("indices", []):
                ii = int(i)
                if 0 <= ii < dim and ii not in ing:
                    ing.append(ii)
    else:
        ing = list(range(min(n_ingredients, dim)))
    par = [i for i in range(dim) if i not in set(ing)]
    return solution[ing].tolist(), solution[par].tolist()


def reconstruct(
    target_encoded: np.ndarray,
    encoder_components: np.ndarray,
    encoder_mean: np.ndarray | None,
    ingredient_bounds: list[list[float]],
    parameter_bounds: list[list[float]],
    n_ingredients: int,
    target_precision: float = 1e-3,
    sum_target: float = 1.0,
    ratio_constraints: list[dict[str, float]] | None = None,
    ingredient_names: list[str] | None = None,
    parameter_names: list[str] | None = None,
    sum_groups: list[dict[str, Any]] | None = None,
    variable_bounds: list[tuple[float, float]] | list[list[float]] | None = None,
    column_names: list[str] | None = None,
    x_reference: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
    latent_penalty: float = 1e3,
) -> dict[str, Any]:
    """Closest feasible recipe to the PCA inverse that also hits ``target_encoded``.

    ``success`` is true only when the recipe is feasible and the latent error is
    within ``target_precision * (1 + ||z||)``. A feasible point that cannot hit
    ``z`` (the coordinate is outside the image of the constraints) returns
    ``feasible=True`` and ``success=False``.
    """
    components = np.array(encoder_components, dtype=float)
    if variable_bounds is not None:
        bounds = [(float(a), float(b)) for a, b in variable_bounds]
    else:
        bounds = _combine_bounds(ingredient_bounds, parameter_bounds)
    dim_x = len(bounds)
    mean = np.array(encoder_mean, dtype=float) if encoder_mean is not None else np.zeros(dim_x)
    if mean.shape[0] != dim_x:
        mean = np.resize(mean, dim_x)
    groups = list(sum_groups) if sum_groups is not None else _legacy_groups(n_ingredients, sum_target, dim_x)
    ratios = list(ratio_constraints or [])

    def encoder_func(x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(x) - mean) @ components.T
        return np.asarray(z, dtype=float).reshape(-1)

    z = np.asarray(target_encoded, dtype=float).reshape(-1)
    if x_reference is not None:
        x_hat = np.asarray(x_reference, dtype=float).reshape(-1)
    else:
        # Minimum-norm preimage of the linear encoder, before the feasible projection.
        x_hat = mean + np.linalg.pinv(components) @ z
    if x_hat.shape[0] != dim_x:
        x_hat = np.resize(x_hat, dim_x)
    scale = np.ones(dim_x, dtype=float)
    if scaler_scale is not None:
        s = np.asarray(scaler_scale, dtype=float).reshape(-1)
        if s.shape[0] == dim_x:
            scale = np.maximum(s, 1e-12)

    x0, _ = project_to_feasible(x_hat, bounds, groups, ratios)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    rho = float(latent_penalty)

    def objective(x: np.ndarray) -> float:
        dx = (np.asarray(x, dtype=float) - x_hat) / scale
        lat = encoder_func(x) - z
        return 0.5 * float(np.dot(dx, dx)) + 0.5 * rho * float(np.dot(lat, lat))

    constraints: list[Any] = []
    for group in groups:
        idxs = [int(i) for i in group.get("indices", []) if 0 <= int(i) < dim_x]
        if not idxs:
            continue
        target = float(group.get("target", group.get("target_sum", 1.0)))
        constraints.append(NonlinearConstraint(_sum_constraint_factory(idxs, target), 0.0, 0.0))
    for rc in ratios:
        try:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
        except (TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i == j or i >= dim_x or j >= dim_x:
            continue
        if rc.get("max_ratio") is not None:
            a = np.zeros((1, dim_x))
            a[0, i] = 1.0
            a[0, j] = -float(rc["max_ratio"])
            constraints.append(LinearConstraint(a, -np.inf, 0.0))
        if rc.get("min_ratio") is not None:
            a = np.zeros((1, dim_x))
            a[0, i] = -1.0
            a[0, j] = float(rc["min_ratio"])
            constraints.append(LinearConstraint(a, -np.inf, 0.0))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": min(float(target_precision), 1e-9), "maxiter": 300, "disp": False},
    )
    repaired, repaired_ok = project_to_feasible(np.asarray(result.x, dtype=float), bounds, groups, ratios)
    if repaired_ok:
        final_solution = repaired
    else:
        final_solution, repaired_ok = project_to_feasible(x_hat, bounds, groups, ratios)
    final_solution = np.clip(final_solution, lb, ub)
    feasible = bool(repaired_ok) and is_feasible(final_solution, bounds, groups, ratios)

    latent = encoder_func(final_solution) - z
    final_error = float(np.linalg.norm(latent))
    z_norm = float(np.linalg.norm(z))
    tol = float(target_precision)
    on_target = final_error <= tol * (1.0 + z_norm)
    latent_rel = final_error / (1.0 + z_norm)

    ingredients, parameters = _split_solution(final_solution, groups if sum_groups is not None else [], n_ingredients)
    # Legacy calls (no explicit sum_groups) keep the leading-ingredient split.
    if sum_groups is None:
        ingredients = final_solution[:n_ingredients].tolist() if n_ingredients > 0 else []
        parameters = final_solution[n_ingredients:].tolist() if n_ingredients < dim_x else []

    out: dict[str, Any] = {
        "success": bool(feasible and on_target),
        "feasible": bool(feasible),
        "on_target": bool(on_target),
        "solution": final_solution.tolist(),
        "ingredients": ingredients,
        "parameters": parameters,
        "final_error": final_error,
        "latent_error_rel": float(latent_rel),
        "iterations": int(getattr(result, "nit", 0)),
        "method": "SLSQP_Constrained",
    }
    names: list[str] | None = None
    if column_names is not None and len(column_names) == dim_x:
        names = [str(n) for n in column_names]
    elif (
        ingredient_names is not None
        and parameter_names is not None
        and len(ingredient_names) == n_ingredients
        and len(parameter_names) == max(dim_x - n_ingredients, 0)
        and sum_groups is None
    ):
        names = [str(n) for n in ingredient_names] + [str(n) for n in parameter_names]
    if names is not None and len(names) == dim_x:
        out["solution_by_name"] = {name: float(val) for name, val in zip(names, final_solution, strict=True)}
    return out
